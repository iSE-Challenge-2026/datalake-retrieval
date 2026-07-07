"""Enrich canonical table descriptions with an OpenRouter chat model."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import CANONICAL_DIR, OPENROUTER_TABLE_ENRICHMENT_DIR  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_file(PROJECT_ROOT)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    known, _ = bootstrap.parse_known_args(argv)
    pipeline_config = _load_pipeline_config(_resolve(known.config))
    defaults = pipeline_config.get("table_enrichment", {})

    parser = argparse.ArgumentParser(description="Use an LLM to enrich canonical table descriptions.", parents=[bootstrap])
    parser.add_argument("--canonical-dir", type=Path, default=Path(defaults.get("canonical_dir", CANONICAL_DIR)))
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.get("output_dir", OPENROUTER_TABLE_ENRICHMENT_DIR)))
    parser.add_argument("--model", default=defaults.get("model") or os.getenv("OPENROUTER_TABLE_DESCRIPTION_MODEL") or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"))
    parser.add_argument("--api-key-env", default=defaults.get("api_key_env", "OPENROUTER_API_KEY"))
    parser.add_argument("--limit", type=int, default=defaults.get("limit"))
    parser.add_argument("--only-source-prefix", action="append", default=list(defaults.get("only_source_prefix") or []))
    parser.add_argument("--force", action="store_true", default=bool(defaults.get("force", False)))
    parser.add_argument("--no-force", dest="force", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    canonical_dir = _resolve(args.canonical_dir)
    tables_path = canonical_dir / "tables.jsonl"
    tables = _read_jsonl(tables_path)
    selected = _select_rows(tables, args.only_source_prefix, args.limit, args.force)

    if args.dry_run:
        print(json.dumps({"selected": len(selected), "total": len(tables)}, ensure_ascii=False, indent=2))
        return

    enricher = OpenRouterTableEnricher(
        model=args.model,
        output_dir=_resolve(args.output_dir),
        api_key_env=args.api_key_env,
    )
    by_id = {row.get("table_id"): row for row in tables}
    enriched_count = 0
    skipped_count = 0
    errors: list[dict] = []

    for row in selected:
        try:
            result = enricher.enrich(row)
        except Exception as exc:  # noqa: BLE001
            skipped_count += 1
            errors.append(
                {
                    "table_id": row.get("table_id"),
                    "source_path": row.get("source_path"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        llm_description = str(result.get("llm_description") or "").strip()
        if not llm_description:
            skipped_count += 1
            continue
        target = by_id.get(row.get("table_id"))
        if target is None:
            continue
        metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
        metadata["table_description_source"] = "openrouter-llm"
        metadata["table_description_model"] = args.model
        metadata["table_description_enriched_at_unix"] = int(time.time())
        metadata["table_description_cache_path"] = result.get("cache_path")
        target["llm_description"] = llm_description
        target["metadata"] = metadata
        enriched_count += 1
        _write_jsonl(tables_path, tables)

    _write_jsonl(tables_path, tables)
    report = {
        "model": args.model,
        "total_tables": len(tables),
        "selected": len(selected),
        "enriched_count": enriched_count,
        "skipped_count": skipped_count,
        "errors": errors,
    }
    report_path = _resolve(args.output_dir) / "enrichment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


class OpenRouterTableEnricher:
    def __init__(self, model: str, output_dir: Path, api_key_env: str) -> None:
        self.model = model
        self.output_dir = output_dir
        self.api_key_env = api_key_env

    def enrich(self, row: dict) -> dict:
        cache_path = self._cache_path(row)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cache_path"] = cache_path.as_posix()
            return payload
        payload = self._call_model(row)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["cache_path"] = cache_path.as_posix()
        return payload

    def _cache_path(self, row: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "table_id": row.get("table_id"),
                    "source_path": row.get("source_path"),
                    "table_path": row.get("table_path"),
                    "description": row.get("description") or "",
                    "columns": row.get("columns") or [],
                    "sample_rows": row.get("sample_rows") or [],
                    "tail_sample_rows": row.get("tail_sample_rows") or [],
                    "metadata_text": row.get("metadata_text") or "",
                    "table_shape": row.get("table_shape") or {},
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.output_dir / self.model.replace("/", "__") / f"{digest}.json"

    def _call_model(self, row: dict) -> dict:
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
                "X-Title": "Data-Lake Table Description Enrichment",
            },
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write factual table descriptions for retrieval in a data lake QA system. "
                        "Use only the provided metadata and table samples. Do not invent facts or computed statistics."
                    ),
                },
                {"role": "user", "content": _prompt(row)},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json(content)
        description = _compose_description(parsed, fallback=content)
        return {"llm_description": description, "raw_response": content, "parsed": parsed}


def _prompt(row: dict) -> str:
    source_path = str(row.get("source_path") or "")
    table_path = str(row.get("table_path") or source_path)
    file_name = Path(source_path).name
    payload = {
        "source_path": source_path,
        "file_name": file_name,
        "table_path": table_path,
        "source_extension": row.get("source_extension"),
        "parser": row.get("parser"),
        "role": row.get("role"),
        "locator": row.get("locator"),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "metadata_or_explanation_text": row.get("metadata_text") or "",
        "columns": row.get("columns") if isinstance(row.get("columns"), list) else [],
        "first_5_rows": row.get("sample_rows") if isinstance(row.get("sample_rows"), list) else [],
        "last_5_rows": row.get("tail_sample_rows") if isinstance(row.get("tail_sample_rows"), list) else [],
        "table_shape": row.get("table_shape") if isinstance(row.get("table_shape"), dict) else {},
        "structured_description": row.get("description") or "",
    }
    return (
        "Create a retrieval-focused description of this table.\n"
        "Return only valid JSON with keys: summary, likely_questions, important_columns, entities_or_values, retrieval_tags.\n"
        "Requirements:\n"
        "- Mention the source path, file name, and sheet/table identity when useful.\n"
        "- Explain what each table appears to contain from metadata, column names, first rows, and last rows.\n"
        "- Include likely user query terms and domain terms.\n"
        "- Preserve important column names and sample entities/values.\n"
        "- Do not calculate new statistics and do not claim coverage beyond the samples.\n\n"
        f"Table context JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)[:18000]}"
    )


def _select_rows(rows: list[dict], prefixes: list[str], limit: int | None, force: bool) -> list[dict]:
    selected = []
    normalized_prefixes = [prefix.replace("\\", "/") for prefix in prefixes]
    for row in rows:
        source_path = str(row.get("source_path") or "").replace("\\", "/")
        if normalized_prefixes and not any(source_path.startswith(prefix) for prefix in normalized_prefixes):
            continue
        if not force and str(row.get("llm_description") or "").strip():
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


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
    labels = {
        "summary": "Summary",
        "likely_questions": "Likely Questions",
        "important_columns": "Important Columns",
        "entities_or_values": "Entities Or Values",
        "retrieval_tags": "Retrieval Tags",
    }
    for key, label in labels.items():
        value = parsed.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if str(item).strip())
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts).strip()


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
