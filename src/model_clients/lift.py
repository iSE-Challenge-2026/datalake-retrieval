"""Datalab Lift API client used by canonical image parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import base64
import binascii
import json
import os
import time

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/document_components.json"


@dataclass
class LiftDataObject:
    object_id: str
    uri: str
    content_type: str = "unknown"
    metadata: dict[str, Any] | None = None


@dataclass
class LiftParsedResult:
    object_id: str
    source_uri: str
    source_format: str
    rows: list[dict[str, Any]]
    text: str | None
    metadata: dict[str, Any]


@dataclass
class LiftAPIConfig:
    api_key_env: str = "DATALAB_API_KEY"
    operation: str = "extract"
    mode: str = "balanced"
    schema_path: str | None = None
    output_dir: str | None = "data/processed/Data-Lake/model_raw/datalab_parsing"
    fallback_to_local: bool = True
    extract_images: bool = True
    save_raw_outputs: bool = True
    timeout: int = 300
    max_polls: int = 300
    poll_interval: int = 1

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class LiftAPIParserClient:
    """Small wrapper around the Datalab hosted extraction API."""

    def __init__(self, config: LiftAPIConfig) -> None:
        self.config = config

    def parse_file(self, path: str | Path, data_object: LiftDataObject) -> LiftParsedResult:
        file_path = Path(path)
        metadata = data_object.metadata or {}
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise RuntimeError(f"Lift API does not support file type: {file_path.suffix}")
        if not self.config.api_key:
            raise RuntimeError(f"{self.config.api_key_env} is not set.")

        try:
            from datalab_sdk import ConvertOptions, DatalabClient, ExtractOptions
        except ImportError as exc:
            raise RuntimeError("Missing datalab-python-sdk. Install it with: pip install datalab-python-sdk") from exc

        schema = _load_schema(self.config.schema_path)
        client = DatalabClient(timeout=self.config.timeout)

        started = time.monotonic()
        conversion = None
        if self.config.operation == "convert":
            result = client.convert(
                str(file_path),
                options=ConvertOptions(mode=self.config.mode, disable_image_extraction=False, output_format="markdown"),
                max_polls=self.config.max_polls,
                poll_interval=self.config.poll_interval,
            )
            image_source = "convert"
            extraction = _conversion_extraction(result)
            images = _normalize_images(_get_attr(result, "images", {}))
        else:
            options = ExtractOptions(page_schema=json.dumps(schema), mode=self.config.mode)
            result = client.extract(
                str(file_path),
                options=options,
                max_polls=self.config.max_polls,
                poll_interval=self.config.poll_interval,
            )
            extraction = _parse_extraction(_get_attr(result, "extraction_schema_json"))
            images = _normalize_images(_get_attr(result, "images", {}))
            image_source = "extract"
            if self.config.extract_images and not images:
                conversion = client.convert(
                    str(file_path),
                    options=ConvertOptions(mode=self.config.mode, disable_image_extraction=False, output_format="markdown"),
                    max_polls=self.config.max_polls,
                    poll_interval=self.config.poll_interval,
                )
                images = _normalize_images(_get_attr(conversion, "images", {}))
                image_source = "convert"

        image_files = _write_images(self.config.output_dir, file_path, images)
        raw_output_paths = _write_lift_raw_outputs(
            self.config.output_dir,
            file_path,
            extract_result=result,
            convert_result=conversion,
            enabled=self.config.save_raw_outputs,
        )
        payload = {
            "input": {
                "object_id": data_object.object_id,
                "uri": data_object.uri,
                "file_name": file_path.name,
                "content_type": data_object.content_type,
                "metadata": metadata,
            },
            "operation": self.config.operation,
            "mode": self.config.mode,
            "status": _get_attr(result, "status"),
            "page_count": _get_attr(result, "page_count"),
            "latency_seconds": round(time.monotonic() - started, 3),
            "extraction": extraction,
            "images": images,
            "image_files": image_files,
            "image_source": image_source,
            "raw_lift_outputs": raw_output_paths,
        }
        output_path = _write_output(self.config.output_dir, file_path, payload)
        text = _best_result_text(result, extraction, conversion)
        return LiftParsedResult(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(metadata.get("format", file_path.suffix.lstrip("."))),
            rows=[{"extraction": extraction, "text": text}],
            text=text,
            metadata={
                "parser": "lift-api",
                "mode": self.config.mode,
                "status": payload["status"],
                "page_count": payload["page_count"],
                "latency_seconds": payload["latency_seconds"],
                "raw_output_path": str(output_path) if output_path else None,
                "image_count": len(image_files),
                "image_files": image_files,
                "image_source": image_source,
                "raw_lift_outputs": raw_output_paths,
            },
        )


def _load_schema(schema_path: str | None) -> dict[str, Any]:
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise RuntimeError(f"Lift schema must be a JSON object: {path}")
    return schema


def _write_output(output_dir: str | None, file_path: Path, payload: dict[str, Any]) -> Path | None:
    if not output_dir:
        return None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{file_path.stem}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _write_images(output_dir: str | None, file_path: Path, images: dict[str, str]) -> list[dict[str, str]]:
    if not output_dir or not images:
        return []
    image_dir = Path(output_dir) / f"{file_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_files: list[dict[str, str]] = []
    for index, (name, encoded) in enumerate(images.items(), start=1):
        safe_name = _safe_image_name(name, index, encoded)
        output_path = image_dir / safe_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.write_bytes(_decode_base64_image(encoded))
        except (binascii.Error, ValueError) as exc:
            image_files.append({"name": name, "path": str(output_path), "status": "decode_failed", "error": str(exc)})
            continue
        image_files.append({"name": name, "path": str(output_path), "status": "saved"})
    return image_files


def _write_lift_raw_outputs(
    output_dir: str | None,
    file_path: Path,
    *,
    extract_result: Any,
    convert_result: Any = None,
    enabled: bool = True,
) -> dict[str, str]:
    if not output_dir or not enabled:
        return {}
    raw_dir = Path(output_dir) / f"{file_path.stem}_raw_lift"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    paths.update(_write_result_artifacts(raw_dir, "extract", extract_result))
    if convert_result is not None:
        paths.update(_write_result_artifacts(raw_dir, "convert", convert_result))
    return paths


def _write_result_artifacts(raw_dir: Path, prefix: str, result: Any) -> dict[str, str]:
    paths: dict[str, str] = {}
    result_data = _result_to_dict(result)
    raw_json_path = raw_dir / f"{prefix}.raw.json"
    raw_json_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    paths[f"{prefix}_raw_json"] = str(raw_json_path)
    for field, suffix in (("markdown", ".md"), ("html", ".html"), ("extraction_schema_json", ".extraction_schema.json")):
        value = _get_attr(result, field)
        if isinstance(value, str) and value:
            path = raw_dir / f"{prefix}{suffix}"
            path.write_text(value, encoding="utf-8")
            paths[f"{prefix}_{field}"] = str(path)
    for field, suffix in (("json", ".document.json"), ("chunks", ".chunks.json")):
        value = _get_attr(result, field)
        if value:
            path = raw_dir / f"{prefix}{suffix}"
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            paths[f"{prefix}_{field}"] = str(path)
    return paths


def _result_to_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": result}


def _normalize_images(raw_images: Any) -> dict[str, str]:
    if not isinstance(raw_images, dict):
        return {}
    return {str(name): str(encoded) for name, encoded in raw_images.items() if encoded}


def _decode_base64_image(encoded: str) -> bytes:
    if "," in encoded and encoded.strip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    return base64.b64decode("".join(encoded.split()), validate=True)


def _safe_image_name(name: str, index: int, encoded: str) -> str:
    path_name = Path(name.replace("\\", "/")).name.strip()
    if path_name and Path(path_name).suffix:
        return path_name
    suffix = _image_suffix_from_data_uri(encoded) or ".png"
    stem = Path(path_name).stem if path_name else f"image_{index:03d}"
    return f"{stem}{suffix}"


def _image_suffix_from_data_uri(encoded: str) -> str | None:
    if not encoded.startswith("data:image/"):
        return None
    mime = encoded.split(";", 1)[0].removeprefix("data:image/")
    if mime == "jpeg":
        return ".jpg"
    return f".{mime}" if mime else None


def _parse_extraction(raw_json: Any) -> Any:
    if raw_json is None or isinstance(raw_json, (dict, list)):
        return raw_json
    try:
        return json.loads(str(raw_json))
    except json.JSONDecodeError:
        return raw_json


def _conversion_extraction(result: Any) -> dict[str, Any]:
    markdown = _get_attr(result, "markdown") or ""
    html = _get_attr(result, "html") or ""
    chunks = _get_attr(result, "chunks") or []
    text_parts = [str(markdown)] if markdown else []
    if not text_parts and html:
        text_parts.append(str(html))
    if not text_parts and chunks:
        text_parts.append(json.dumps(chunks, ensure_ascii=False))
    return {
        "document_type": "image",
        "language": None,
        "title": None,
        "markdown": "\n\n".join(text_parts),
        "text": "\n\n".join(text_parts),
        "tables": [],
        "figures": [],
        "formulas": [],
    }


def _best_result_text(result: Any, extraction: Any, conversion: Any = None) -> str | None:
    candidates: list[str] = []
    for value in (
        _get_attr(result, "markdown"),
        _get_attr(conversion, "markdown") if conversion is not None else None,
        _extraction_text(extraction),
    ):
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    return _best_text(candidates)


def _extraction_text(extraction: Any) -> str | None:
    if not isinstance(extraction, dict):
        return None
    candidates: list[str] = []
    for item in extraction.get("tables") or []:
        if isinstance(item, dict):
            for field in ("content", "text", "markdown", "caption", "description"):
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
    for item in extraction.get("figures") or []:
        if isinstance(item, dict):
            for field in ("caption", "description", "text", "content"):
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
    for field in ("main_text", "markdown", "text", "content"):
        value = extraction.get(field)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    return _best_text(candidates)


def _best_text(candidates: list[str]) -> str | None:
    cleaned = [value.strip() for value in candidates if value and value.strip()]
    if not cleaned:
        return None
    return max(cleaned, key=len)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
