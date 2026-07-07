"""Enrich canonical image descriptions with an OpenRouter vision model."""

from __future__ import annotations

from pathlib import Path
import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import CANONICAL_DIR, OPENROUTER_IMAGE_ENRICHMENT_DIR, RAW_DATA_LAKE_DIR  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_file(PROJECT_ROOT)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    known, _ = bootstrap.parse_known_args(argv)
    pipeline_config = _load_pipeline_config(_resolve(known.config))
    defaults = pipeline_config.get("image_enrichment", {})

    parser = argparse.ArgumentParser(description="Use a VLM to enrich canonical image descriptions.", parents=[bootstrap])
    parser.add_argument("--canonical-dir", type=Path, default=Path(defaults.get("canonical_dir", CANONICAL_DIR)))
    parser.add_argument("--data-root", type=Path, default=Path(defaults.get("data_root", RAW_DATA_LAKE_DIR)))
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.get("output_dir", OPENROUTER_IMAGE_ENRICHMENT_DIR)))
    parser.add_argument("--model", default=defaults.get("model") or os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash-lite"))
    parser.add_argument("--api-key-env", default=defaults.get("api_key_env", "OPENROUTER_API_KEY"))
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--only-source-prefix", action="append", default=list(defaults.get("only_source_prefix") or []))
    parser.add_argument("--min-description-chars", type=int, default=int(defaults.get("min_description_chars", 0)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    canonical_dir = _resolve(args.canonical_dir)
    images_path = canonical_dir / "images.jsonl"
    images = _read_jsonl(images_path)
    selected = _select_rows(images, args.only_source_prefix, args.min_description_chars, args.limit)

    if args.dry_run:
        print(json.dumps({"selected": len(selected), "total": len(images)}, ensure_ascii=False, indent=2))
        return

    enricher = OpenRouterVisionEnricher(
        model=args.model,
        output_dir=_resolve(args.output_dir),
        api_key_env=args.api_key_env,
    )
    enriched_count = 0
    skipped_count = 0
    errors: list[dict] = []
    by_id = {row.get("image_id"): row for row in images}

    for row in selected:
        image_path = _resolve_image_path(row, _resolve(args.data_root), canonical_dir)
        if image_path is None:
            skipped_count += 1
            errors.append({"image_id": row.get("image_id"), "source_path": row.get("source_path"), "error": "image_file_not_found"})
            continue
        try:
            result = enricher.enrich(row, image_path)
        except Exception as exc:  # noqa: BLE001
            skipped_count += 1
            errors.append(
                {
                    "image_id": row.get("image_id"),
                    "source_path": row.get("source_path"),
                    "image_path": image_path.as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        description = str(result.get("description") or "").strip()
        if not description:
            skipped_count += 1
            continue
        target = by_id.get(row.get("image_id"))
        if target is None:
            continue
        previous = str(target.get("description") or "")
        target["description"] = description
        metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
        metadata["previous_description"] = previous
        metadata["description_source"] = "openrouter-vlm-enrichment"
        metadata["vision_model"] = args.model
        metadata["vision_enriched_at_unix"] = int(time.time())
        metadata["vision_cache_path"] = result.get("cache_path")
        target["metadata"] = metadata
        enriched_count += 1
        _write_jsonl(images_path, images)

    _write_jsonl(images_path, images)
    report = {
        "model": args.model,
        "total_images": len(images),
        "selected": len(selected),
        "enriched_count": enriched_count,
        "skipped_count": skipped_count,
        "errors": errors,
    }
    report_path = _resolve(args.output_dir) / "enrichment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


class OpenRouterVisionEnricher:
    def __init__(self, model: str, output_dir: Path, api_key_env: str) -> None:
        self.model = model
        self.output_dir = output_dir
        self.api_key_env = api_key_env

    def enrich(self, row: dict, image_path: Path) -> dict:
        cache_path = self._cache_path(row, image_path)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cache_path"] = cache_path.as_posix()
            return payload
        payload = self._call_model(row, image_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["cache_path"] = cache_path.as_posix()
        return payload

    def _cache_path(self, row: dict, image_path: Path) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "image_id": row.get("image_id"),
                    "source_path": row.get("source_path"),
                    "image_path": image_path.as_posix(),
                    "image_hash": _file_hash(image_path),
                    "description": row.get("description") or "",
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.output_dir / self.model.replace("/", "__") / f"{digest}.json"

    def _call_model(self, row: dict, image_path: Path) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing openai package. Install it with: pip install openai") from exc
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set.")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60,
            max_retries=1,
            default_headers={
                "HTTP-Referer": "https://local.data-lake-retrieval",
                "X-Title": "Data-Lake Image Description Enrichment",
            },
        )
        source_path = str(row.get("source_path") or "")
        current_description = str(row.get("description") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        prompt = (
            "Enrich this image description for retrieval in a multimodal QA system.\n"
            "Use the actual visual content, OCR-visible text, and the provided metadata.\n"
            "Return only valid JSON with keys: description, visible_text, objects, colors, retrieval_tags.\n"
            "Requirements:\n"
            "- Be concrete and complete.\n"
            "- If the image contains digits, state which digits appear, whether it is a single digit or multiple digits, and colors when visible.\n"
            "- If it is a document/report page, summarize the page topic and important named entities.\n"
            "- Include source path clues only as context, not as invented visual facts.\n\n"
            f"Source path: {source_path}\n"
            f"Canonical role: {row.get('role')}\n"
            f"Current Lift/OCR description:\n{current_description}\n\n"
            f"Metadata:\n{json.dumps(metadata, ensure_ascii=False)[:2000]}"
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json(content)
        description = _compose_description(parsed, fallback=content)
        return {"description": description, "raw_response": content, "parsed": parsed}


def _select_rows(rows: list[dict], prefixes: list[str], min_description_chars: int, limit: int | None) -> list[dict]:
    selected = []
    normalized_prefixes = [prefix.replace("\\", "/") for prefix in prefixes]
    for row in rows:
        source_path = str(row.get("source_path") or "").replace("\\", "/")
        if normalized_prefixes and not any(source_path.startswith(prefix) for prefix in normalized_prefixes):
            continue
        if min_description_chars and len(str(row.get("description") or "").strip()) >= min_description_chars:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _resolve_image_path(row: dict, data_root: Path, canonical_dir: Path) -> Path | None:
    candidates = []
    source_path = str(row.get("source_path") or "")
    image_path = str(row.get("image_path") or "")
    for value in (image_path, source_path):
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        candidates.append(data_root / value)
        candidates.append(canonical_dir / value)
        candidates.append(PROJECT_ROOT / value)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _parse_json(content: str) -> dict:
    value = content.strip()
    if value.startswith("```"):
        value = value.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compose_description(parsed: dict, fallback: str) -> str:
    if not parsed:
        return fallback.strip()
    parts = []
    for key in ("description", "visible_text"):
        value = parsed.get(key)
        if value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}")
    for key in ("objects", "colors", "retrieval_tags"):
        value = parsed.get(key)
        if isinstance(value, list) and value:
            parts.append(f"{key.replace('_', ' ').title()}: " + ", ".join(str(item) for item in value))
        elif value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(parts).strip()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_pipeline_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
