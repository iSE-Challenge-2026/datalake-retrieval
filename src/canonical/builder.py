"""Build canonical text, image, and table artifacts from a raw data lake."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree as ET
import base64
import csv
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".docx",
}
DOC_EXTENSIONS = {".pdf", ".pptx", ".ppt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".sql", ".db", ".sqlite", ".sqlite3", ".duckdb", ".parquet"}

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
SHEET_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True)
class CanonicalBuildConfig:
    data_root: Path
    output_dir: Path
    asr_provider: str = "none"
    asr_model: str = "openai/whisper-large-v3"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    audio_language: str | None = None
    max_text_chars: int = 2_000_000
    max_table_preview_rows: int = 80
    copy_extracted_images: bool = True
    compute_file_hash: bool = False
    extract_pdf_images: bool = False
    extract_pptx_images: bool = True
    ppt_conversion_cache_dir: Path | None = None
    asr_cache_dir: Path | None = None
    raw_image_parser: str = "lift-api"
    datalab_parsing_dir: Path | None = None
    raw_image_lift_output_dir: Path | None = None
    raw_image_lift_schema_path: Path | None = None
    raw_image_lift_mode: str = "balanced"
    raw_image_lift_api_key_env: str = "DATALAB_API_KEY"


@dataclass
class CanonicalText:
    text_id: str
    source_path: str
    source_extension: str
    text: str
    role: str
    parser: str
    locator: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalImage:
    image_id: str
    source_path: str
    source_extension: str
    image_path: str
    role: str
    description: str = ""
    locator: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTable:
    table_id: str
    source_path: str
    source_extension: str
    table_path: str
    role: str
    parser: str
    preview_text: str = ""
    description: str = ""
    columns: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, str]] = field(default_factory=list)
    tail_sample_rows: list[dict[str, str]] = field(default_factory=list)
    llm_description: str = ""
    metadata_text: str = ""
    table_shape: dict[str, Any] = field(default_factory=dict)
    locator: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)


def build_canonical_artifacts(config: CanonicalBuildConfig) -> dict[str, Any]:
    """Build canonical artifacts and write JSONL outputs."""
    data_root = config.data_root.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_image_dir = output_dir / "extracted_images"
    if config.copy_extracted_images:
        extracted_image_dir.mkdir(parents=True, exist_ok=True)

    texts: list[CanonicalText] = []
    images: list[CanonicalImage] = []
    tables: list[CanonicalTable] = []
    errors: list[dict[str, Any]] = []

    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel_path = path.relative_to(data_root).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_EXTENSIONS:
                texts.extend(_text_file_records(path, rel_path, suffix, config))
            elif suffix in DOC_EXTENSIONS:
                doc_texts, doc_images = _doc_file_records(path, rel_path, suffix, config, extracted_image_dir)
                texts.extend(doc_texts)
                images.extend(doc_images)
            elif suffix in IMAGE_EXTENSIONS:
                image_texts, image_records = _raw_image_records(path, rel_path, suffix, config)
                texts.extend(image_texts)
                images.extend(image_records)
            elif suffix in AUDIO_EXTENSIONS:
                text = _audio_text(path, suffix, config)
                texts.append(
                    CanonicalText(
                        text_id=_stable_id(rel_path, "audio-transcript"),
                        source_path=rel_path,
                        source_extension=suffix,
                        text=text,
                        role="audio_transcript",
                        parser=_audio_parser_name(config),
                        metadata=_file_metadata(path),
                    )
                )
            elif suffix in TABLE_EXTENSIONS:
                tables.extend(_table_records(path, rel_path, suffix, config))
            else:
                texts.append(
                    CanonicalText(
                        text_id=_stable_id(rel_path, "unknown"),
                        source_path=rel_path,
                        source_extension=suffix,
                        text=f"Unhandled file available at {rel_path}",
                        role="unhandled_placeholder",
                        parser="placeholder",
                        metadata=_file_metadata(path),
                    )
                )
        except Exception as exc:
            errors.append(
                {
                    "source_path": rel_path,
                    "extension": suffix,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    images = _dedupe_images(images)

    _write_jsonl(output_dir / "texts.jsonl", (asdict(item) for item in texts))
    _write_jsonl(output_dir / "images.jsonl", (asdict(item) for item in images))
    _write_jsonl(output_dir / "tables.jsonl", (asdict(item) for item in tables))
    _write_jsonl(output_dir / "errors.jsonl", errors)

    summary = {
        "contract_version": "canonical-artifacts-v1",
        "data_root": data_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "counts": {
            "texts": len(texts),
            "images": len(images),
            "tables": len(tables),
            "errors": len(errors),
        },
        "extraction": _extraction_summary(texts, images, tables),
        "asr": {
            "provider": config.asr_provider,
            "model": config.asr_model if config.asr_provider != "none" else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _extraction_summary(
    texts: list[CanonicalText],
    images: list[CanonicalImage],
    tables: list[CanonicalTable],
) -> dict[str, Any]:
    by_extension: dict[str, dict[str, int]] = {}
    for item in texts:
        bucket = by_extension.setdefault(item.source_extension or "<none>", {"text_records": 0, "real_text": 0, "placeholder_text": 0})
        bucket["text_records"] += 1
        if _is_placeholder_text(item.text, item.parser):
            bucket["placeholder_text"] += 1
        elif _is_real_text(item.text):
            bucket["real_text"] += 1
    return {
        "by_text_extension": by_extension,
        "image_records": len(images),
        "image_records_with_description": sum(1 for item in images if item.description.strip()),
        "table_records": len(tables),
        "note": "tables are profiled with metadata/explanation text, columns, sample rows, and retrieval descriptions; executable table normalization is still deferred.",
    }


def _dedupe_images(images: list[CanonicalImage]) -> list[CanonicalImage]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[CanonicalImage] = []
    for image in images:
        key = (image.source_path, image.image_path, image.role, image.description[:200])
        if key in seen:
            continue
        seen.add(key)
        output.append(image)
    return output


def _clean_markdown_for_description(text: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text or "")
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    return value


def _compact_description(text: str, limit: int = 1600) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _is_real_text(text: str) -> bool:
    value = (text or "").strip()
    if len(value) < 20:
        return bool(value) and not _looks_like_placeholder(value)
    return not _looks_like_placeholder(value)


def _is_placeholder_text(text: str, parser: str) -> bool:
    return "placeholder" in (parser or "") or _looks_like_placeholder(text)


def _looks_like_placeholder(text: str) -> bool:
    value = (text or "").strip().lower()
    return (
        "available at" in value[:300]
        or "is disabled" in value[:300]
        or "not implemented" in value[:300]
        or "did not produce text" in value[:300]
    )


def _text_file_records(path: Path, rel_path: str, suffix: str, config: CanonicalBuildConfig) -> list[CanonicalText]:
    if suffix == ".docx":
        text, parser = _docx_text(path)
    elif suffix in {".html", ".htm"}:
        text = _strip_html(_read_text(path, config.max_text_chars))
        parser = "html-text"
    else:
        text = _read_text(path, config.max_text_chars)
        parser = "plain-text"
    return [
        CanonicalText(
            text_id=_stable_id(rel_path, "text"),
            source_path=rel_path,
            source_extension=suffix,
            text=text,
            role="text",
            parser=parser,
            metadata=_file_metadata(path),
        )
    ]


def _doc_file_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
    extracted_image_dir: Path,
) -> tuple[list[CanonicalText], list[CanonicalImage]]:
    datalab_records = _datalab_doc_file_records(path, rel_path, suffix, config)
    if datalab_records is not None:
        return datalab_records

    if suffix == ".pdf":
        text, parser = _pdf_text(path)
        images = _pdf_image_records(path, rel_path, suffix, config, extracted_image_dir) if config.extract_pdf_images else []
    elif suffix == ".pptx":
        text, parser = _pptx_text(path)
        images = _pptx_image_records(path, rel_path, suffix, config, extracted_image_dir) if config.extract_pptx_images else []
    else:
        text, parser, images = _legacy_ppt_records(path, rel_path, config, extracted_image_dir)

    texts = [
        CanonicalText(
            text_id=_stable_id(rel_path, "doc-text"),
            source_path=rel_path,
            source_extension=suffix,
            text=text,
            role="document_text",
            parser=parser,
            metadata=_file_metadata(path),
        )
    ]
    return texts, images


def _datalab_doc_file_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
) -> tuple[list[CanonicalText], list[CanonicalImage]] | None:
    cache_path = _datalab_cache_path(path, rel_path, config)
    if cache_path is None:
        return None

    payload = _payload_from_lift_output(cache_path, rel_path, suffix)
    text = str(payload.get("text") or "").strip()
    if not text:
        return None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    extraction = payload.get("extraction") if isinstance(payload.get("extraction"), dict) else {}
    text_record = CanonicalText(
        text_id=_stable_id(rel_path, "datalab-doc-text"),
        source_path=rel_path,
        source_extension=suffix,
        text=text,
        role="document_text",
        parser="datalab-parsing-cache",
        metadata={
            **_file_metadata(path),
            "raw_lift_output_path": metadata.get("raw_output_path"),
            "page_count": metadata.get("page_count"),
            "image_count": metadata.get("image_count"),
            "image_source": metadata.get("image_source"),
        },
    )

    descriptions = _image_descriptions_from_lift_text(text)
    descriptions.extend(_image_descriptions_from_extraction(extraction))
    image_records: list[CanonicalImage] = []
    image_files = payload.get("image_files") if isinstance(payload.get("image_files"), list) else []
    for index, image_file in enumerate(image_files):
        image_path_value = _image_file_path_value(image_file)
        if not image_path_value:
            continue
        image_path = _canonical_output_relative_path(image_path_value, config.output_dir)
        description = descriptions[index] if index < len(descriptions) else ""
        image_records.append(
            CanonicalImage(
                image_id=_stable_id(rel_path, "datalab-doc-image", index, image_path),
                source_path=rel_path,
                source_extension=suffix,
                image_path=image_path,
                role="document_embedded_image",
                description=_compact_description(description, limit=1600),
                locator=f"datalab_image={index}",
                metadata={
                    **_file_metadata(path),
                    "parser": "datalab-parsing-cache",
                    "description_source": "datalab-caption" if description else "datalab-image-file",
                    "raw_lift_output_path": metadata.get("raw_output_path"),
                },
            )
        )
    return [text_record], image_records


def _raw_image_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
) -> tuple[list[CanonicalText], list[CanonicalImage]]:
    raw_image = _raw_image_record(path, rel_path, suffix)
    if config.raw_image_parser == "none":
        return [], [raw_image]
    if config.raw_image_parser == "lift-api":
        return _lift_raw_image_records(path, rel_path, suffix, config, raw_image)
    raise ValueError(f"Unsupported raw image parser: {config.raw_image_parser}")


def _raw_image_record(path: Path, rel_path: str, suffix: str) -> CanonicalImage:
    return CanonicalImage(
        image_id=_stable_id(rel_path, "raw-image"),
        source_path=rel_path,
        source_extension=suffix,
        image_path=rel_path,
        role="raw_image",
        metadata=_file_metadata(path),
    )


def _lift_raw_image_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
    raw_image: CanonicalImage,
) -> tuple[list[CanonicalText], list[CanonicalImage]]:
    parsed = _parse_raw_image_with_lift(path, rel_path, suffix, config)
    text = str(parsed.get("text") or "").strip()
    metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    extraction = parsed.get("extraction") if isinstance(parsed.get("extraction"), dict) else {}
    image_files = parsed.get("image_files") if isinstance(parsed.get("image_files"), list) else []

    texts: list[CanonicalText] = []
    images: list[CanonicalImage] = []
    if text:
        texts.append(
            CanonicalText(
                text_id=_stable_id(rel_path, "raw-image-lift-text"),
                source_path=rel_path,
                source_extension=suffix,
                text=text,
                role="raw_image_ocr_text",
                parser="lift-api",
                metadata={
                    **_file_metadata(path),
                    "raw_lift_output_path": metadata.get("raw_output_path"),
                    "image_count": metadata.get("image_count"),
                    "image_source": metadata.get("image_source"),
                },
            )
        )
    elif not image_files:
        raw_image.metadata["parser"] = "lift-api"
        raw_image.metadata["description_source"] = "raw-image-unparsed"
        images.append(raw_image)

    descriptions = _image_descriptions_from_lift_text(text)
    descriptions.extend(_image_descriptions_from_extraction(extraction))
    for index, image_file in enumerate(image_files):
        image_path_value = _image_file_path_value(image_file)
        if not image_path_value:
            continue
        image_path = _canonical_output_relative_path(image_path_value, config.output_dir)
        description = descriptions[index] if index < len(descriptions) else ""
        images.append(
            CanonicalImage(
                image_id=_stable_id(rel_path, "raw-image-lift-region", index, image_path),
                source_path=rel_path,
                source_extension=suffix,
                image_path=image_path,
                role="raw_image_extracted_region",
                description=_compact_description(description, limit=1600),
                locator=f"lift_image={index}",
                metadata={
                    **_file_metadata(path),
                    "parser": "lift-api",
                    "parent_image_id": raw_image.image_id,
                    "description_source": "lift-api-region-caption" if description else "lift-api-image-file",
                    "raw_lift_output_path": metadata.get("raw_output_path"),
                },
            )
        )
    return texts, images


def _parse_raw_image_with_lift(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
) -> dict[str, Any]:
    cache_dir = _datalab_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_stable_id(rel_path, _file_hash(path), 'lift-api')}.json"
    stem_cache_paths = (cache_dir / f"{path.stem}.json", cache_dir / "raw_outputs" / f"{path.stem}.json")
    for stem_cache_path in stem_cache_paths:
        if stem_cache_path.exists():
            payload = _payload_from_lift_output(stem_cache_path, rel_path, suffix)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        from src.model_clients.lift import LiftAPIConfig, LiftAPIParserClient, LiftDataObject
    except ImportError as exc:
        raise RuntimeError("Lift API parser dependencies are not available.") from exc

    data_object = LiftDataObject(
        object_id=_stable_id(rel_path, "raw-image"),
        uri=f"data/raw/Data-Lake/{rel_path}",
        content_type=mimetypes.guess_type(path.name)[0] or "image",
        metadata={"format": suffix.lstrip("."), "source_path": rel_path},
    )
    lift_config = LiftAPIConfig(
        api_key_env=config.raw_image_lift_api_key_env,
        operation="extract",
        mode=config.raw_image_lift_mode,
        schema_path=str(config.raw_image_lift_schema_path) if config.raw_image_lift_schema_path else None,
        output_dir=str(cache_dir / "raw_outputs"),
        fallback_to_local=False,
        extract_images=True,
        save_raw_outputs=True,
    )
    parsed = LiftAPIParserClient(lift_config).parse_file(path, data_object)
    extraction = parsed.rows[0].get("extraction") if parsed.rows else None
    payload = {
        "source_path": rel_path,
        "source_extension": suffix,
        "text": parsed.text or "",
        "extraction": extraction,
        "metadata": parsed.metadata,
        "image_files": list(parsed.metadata.get("image_files") or []),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _datalab_cache_dir(config: CanonicalBuildConfig) -> Path:
    return config.datalab_parsing_dir or config.raw_image_lift_output_dir or (config.output_dir.parent / "model_raw" / "datalab_parsing")


def _datalab_cache_path(path: Path, rel_path: str, config: CanonicalBuildConfig) -> Path | None:
    cache_dir = _datalab_cache_dir(config)
    candidates = [
        cache_dir / f"{path.stem}.json",
        cache_dir / "raw_outputs" / f"{path.stem}.json",
        cache_dir / f"{_stable_id(rel_path, _file_hash(path), 'lift-api')}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _payload_from_lift_output(output_path: Path, rel_path: str, suffix: str) -> dict[str, Any]:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    extraction = data.get("extraction") if isinstance(data.get("extraction"), dict) else {}
    text, text_source = _best_lift_output_text(data, extraction, output_path)
    image_files = _local_lift_image_files(output_path.parent, output_path.stem)
    if not image_files and isinstance(data.get("image_files"), list):
        image_files = data["image_files"]
    return {
        "source_path": rel_path,
        "source_extension": suffix,
        "text": text,
        "extraction": extraction,
        "metadata": {
            "parser": "lift-api",
            "mode": data.get("mode"),
            "status": data.get("status"),
            "page_count": data.get("page_count"),
            "latency_seconds": data.get("latency_seconds"),
            "raw_output_path": str(output_path),
            "text_source": text_source,
            "image_count": len(image_files),
            "image_files": image_files,
            "image_source": data.get("image_source"),
            "raw_lift_outputs": data.get("raw_lift_outputs") or {},
        },
        "image_files": image_files,
    }


def _best_lift_output_text(data: dict[str, Any], extraction: dict[str, Any], output_path: Path) -> tuple[str, str | None]:
    candidates: list[tuple[str, str]] = []

    raw_outputs = data.get("raw_lift_outputs") if isinstance(data.get("raw_lift_outputs"), dict) else {}
    for key in (
        "extract_markdown",
        "convert_markdown",
        "extract_raw_json",
        "convert_raw_json",
        "extract_html",
        "convert_html",
    ):
        value = raw_outputs.get(key)
        if isinstance(value, str):
            candidates.extend(_lift_text_candidates_from_path(Path(value), f"raw_lift_outputs.{key}"))

    raw_dir = output_path.parent / f"{output_path.stem}_raw_lift"
    for name in (
        "extract.md",
        "convert.md",
        "extract.raw.json",
        "convert.raw.json",
        "extract.html",
        "convert.html",
    ):
        candidates.extend(_lift_text_candidates_from_path(raw_dir / name, f"{raw_dir.name}/{name}"))

    for key in ("markdown", "text", "content"):
        value = extraction.get(key)
        if isinstance(value, str):
            candidates.append((f"extraction.{key}", value))

    for key in ("markdown", "text", "content"):
        value = data.get(key)
        if isinstance(value, str):
            candidates.append((key, value))

    for item_key in ("tables", "figures"):
        items = extraction.get(item_key)
        if isinstance(items, list):
            parts: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                for field in ("content", "markdown", "text", "caption", "description"):
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
            if parts:
                candidates.append((f"extraction.{item_key}", "\n\n".join(parts)))

    for key in ("main_text", "summary"):
        value = extraction.get(key)
        if isinstance(value, str):
            candidates.append((f"extraction.{key}", value))

    cleaned = [(source, text.strip()) for source, text in candidates if text and text.strip()]
    if not cleaned:
        return "", None
    source, text = max(cleaned, key=lambda item: len(item[1]))
    return text, source


def _lift_text_candidates_from_path(path: Path, source: str) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return _lift_text_candidates_from_json(data, source)
        return [(source, path.read_text(encoding="utf-8"))]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []


def _lift_text_candidates_from_json(data: Any, source: str) -> list[tuple[str, str]]:
    if not isinstance(data, dict):
        return []
    candidates: list[tuple[str, str]] = []
    for key in ("markdown", "text", "content", "html"):
        value = data.get(key)
        if isinstance(value, str):
            candidates.append((f"{source}.{key}", value))
    extraction = data.get("extraction") if isinstance(data.get("extraction"), dict) else {}
    for key in ("markdown", "text", "content"):
        value = extraction.get(key)
        if isinstance(value, str):
            candidates.append((f"{source}.extraction.{key}", value))
    return candidates


def _local_lift_image_files(root: Path, stem: str) -> list[dict[str, str]]:
    image_dir = root / f"{stem}_images"
    if not image_dir.exists():
        return []
    return [
        {"name": path.name, "path": str(path), "status": "saved"}
        for path in sorted(image_dir.rglob("*"))
        if path.is_file()
    ]


def _image_file_path_value(image_file: Any) -> str:
    if isinstance(image_file, dict):
        return str(image_file.get("path") or image_file.get("image_path") or image_file.get("name") or "").strip()
    return str(image_file or "").strip()


def _image_descriptions_from_lift_text(text: str) -> list[str]:
    descriptions: list[str] = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text or ""):
        description = match.group(1).strip()
        if description:
            descriptions.append(description)
    return descriptions


def _image_descriptions_from_extraction(extraction: dict[str, Any]) -> list[str]:
    descriptions: list[str] = []
    for key in ("figures", "images"):
        value = extraction.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("caption") or item.get("description") or item.get("text") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                descriptions.append(text)
    return descriptions


def _canonical_output_relative_path(path_value: str, output_dir: Path) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path_value.replace("\\", "/")


def _table_records(path: Path, rel_path: str, suffix: str, config: CanonicalBuildConfig) -> list[CanonicalTable]:
    if suffix == ".xlsx":
        sheets = _xlsx_sheet_rows(path, config.max_table_preview_rows)
        records: list[CanonicalTable] = []
        for sheet in sheets:
            sheet_name = str(sheet.get("sheet_name") or "Sheet")
            rows = sheet.get("rows") if isinstance(sheet.get("rows"), list) else []
            profile = _profile_rows(rel_path, suffix, rows, parser="xlsx-sheet-profile")
            profile["preview_text"] = f"Sheet {sheet_name}\n{profile['preview_text']}"
            profile["metadata_text"] = f"Sheet {sheet_name}: {profile['metadata_text']}".strip()
            profile["table_shape"] = {
                **profile["table_shape"],
                "kind": "xlsx_sheet",
                "sheet_name": sheet_name,
                "sheet_count": len(sheets),
            }
            profile["description"] = _table_description(
                rel_path=f"{rel_path}#sheet={sheet_name}",
                suffix=suffix,
                parser="xlsx-sheet-profile",
                columns=profile["columns"],
                sample_rows=profile["sample_rows"],
                tail_sample_rows=profile["tail_sample_rows"],
                metadata_text=profile["metadata_text"],
                table_shape=profile["table_shape"],
            )
            records.append(_table_record_from_profile(path, rel_path, suffix, profile, sheet_name=sheet_name))
        return records or [_table_record_from_profile(path, rel_path, suffix, _table_profile(path, rel_path, suffix, config))]
    return [_table_record_from_profile(path, rel_path, suffix, _table_profile(path, rel_path, suffix, config))]


def _table_record_from_profile(
    path: Path,
    rel_path: str,
    suffix: str,
    profile: dict[str, Any],
    sheet_name: str | None = None,
) -> CanonicalTable:
    table_path = f"{rel_path}#sheet={sheet_name}" if sheet_name else rel_path
    locator = f"sheet={sheet_name}" if sheet_name else "file"
    return CanonicalTable(
        table_id=_stable_id(table_path, "table"),
        source_path=rel_path,
        source_extension=suffix,
        table_path=table_path,
        role="raw_table",
        parser=profile["parser"],
        preview_text=profile["preview_text"],
        description=profile["description"],
        columns=profile["columns"],
        sample_rows=profile["sample_rows"],
        tail_sample_rows=profile["tail_sample_rows"],
        llm_description=profile.get("llm_description", ""),
        metadata_text=profile["metadata_text"],
        table_shape=profile["table_shape"],
        locator=locator,
        metadata=_file_metadata(path),
    )


def _audio_text(path: Path, suffix: str, config: CanonicalBuildConfig) -> str:
    sidecar = _read_audio_sidecar(path)
    if sidecar:
        return sidecar
    if config.asr_provider == "none":
        return f"Audio file available at {path.name}. ASR is disabled."
    if config.asr_provider == "openrouter":
        return _openrouter_transcribe(path, suffix, config)
    raise ValueError(f"Unsupported ASR provider: {config.asr_provider}")


def _read_audio_sidecar(path: Path) -> str:
    for suffix in (".txt", ".vtt", ".srt"):
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            return _read_text(sidecar, 2_000_000)
    return ""


def _openrouter_transcribe(path: Path, suffix: str, config: CanonicalBuildConfig) -> str:
    cache_dir = config.asr_cache_dir or (config.output_dir / "asr_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = _file_hash(path)
    cache_path = cache_dir / f"{file_hash}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return str(payload.get("text") or "")

    api_key = os.getenv(config.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.openrouter_api_key_env} is not set.")

    request_payload: dict[str, Any] = {
        "model": config.asr_model,
        "input_audio": {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "format": suffix.lstrip("."),
        },
        "temperature": 0,
    }
    if config.audio_language:
        request_payload["language"] = config.audio_language

    request = urllib.request.Request(
        f"{config.openrouter_base_url.rstrip('/')}/audio/transcriptions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter ASR failed with HTTP {exc.code}: {detail}") from exc

    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result.get("text") or "")


def _audio_parser_name(config: CanonicalBuildConfig) -> str:
    if config.asr_provider == "none":
        return "audio-placeholder"
    return f"{config.asr_provider}:{config.asr_model}"


def _docx_text(path: Path) -> tuple[str, str]:
    try:
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
            text = " ".join(item.text or "" for item in root.findall(".//w:t", WORD_NS)).strip()
            return text, "docx-ooxml"
    except (BadZipFile, KeyError, ET.ParseError):
        return f"DOCX file available at {path.name}, but local OOXML parsing failed.", "docx-placeholder"


def _pptx_text(path: Path) -> tuple[str, str]:
    try:
        with ZipFile(path) as archive:
            slide_names = sorted(
                (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
            )
            parts: list[str] = []
            for index, name in enumerate(slide_names, start=1):
                root = ET.fromstring(archive.read(name))
                text = " ".join(item.text or "" for item in root.findall(".//a:t", PPT_NS)).strip()
                if text:
                    parts.append(f"Slide {index}: {text}")
            return "\n\n".join(parts), "pptx-ooxml"
    except (BadZipFile, KeyError, ET.ParseError, AttributeError):
        return f"PPTX file available at {path.name}, but local OOXML parsing failed.", "pptx-placeholder"


def _legacy_ppt_records(
    path: Path,
    rel_path: str,
    config: CanonicalBuildConfig,
    extracted_image_dir: Path,
) -> tuple[str, str, list[CanonicalImage]]:
    converted_path, converter, error = _convert_ppt_to_pptx(path, rel_path, config)
    if converted_path is None:
        return (
            f"Legacy PPT file available at {rel_path}. Local text extraction is not implemented. Conversion failed: {error}",
            "ppt-placeholder",
            [],
        )

    text, parser = _pptx_text(converted_path)
    if not text.strip() or parser.endswith("placeholder"):
        text = f"Legacy PPT file available at {rel_path}. Converted with {converter}, but local text extraction did not produce text."
        parser = "ppt-converted-placeholder"
    else:
        parser = f"ppt-converted-{converter}+{parser}"

    images = (
        _pptx_image_records(
            converted_path,
            rel_path,
            ".ppt",
            config,
            extracted_image_dir,
            metadata_path=path,
            metadata_extra={
                "converted_from": "ppt",
                "converted_pptx_path": converted_path.as_posix(),
                "ppt_converter": converter,
            },
        )
        if config.extract_pptx_images
        else []
    )
    return text, parser, images


def _convert_ppt_to_pptx(path: Path, rel_path: str, config: CanonicalBuildConfig) -> tuple[Path | None, str, str]:
    cache_dir = config.ppt_conversion_cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"{_stable_id(rel_path, _file_hash(path))}.pptx"
    else:
        output_path = Path(tempfile.gettempdir()) / "ise_ppt_conversion_cache" / f"{_stable_id(rel_path, _file_hash(path))}.pptx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, "cache", ""

    converter, error = _convert_ppt_with_powerpoint(path, output_path)
    if converter:
        return output_path, converter, ""

    converter, lo_error = _convert_ppt_with_libreoffice(path, output_path)
    if converter:
        return output_path, converter, ""
    return None, "", "; ".join(part for part in (error, lo_error) if part)


def _convert_ppt_with_powerpoint(path: Path, output_path: Path) -> tuple[str | None, str]:
    if os.name != "nt":
        return None, "PowerPoint COM conversion is only available on Windows"
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
    except ImportError as exc:
        return None, f"PowerPoint COM unavailable: {type(exc).__name__}: {exc}"

    app = None
    presentation = None
    try:
        pythoncom.CoInitialize()
        app = win32com.client.DispatchEx("PowerPoint.Application")
        app.Visible = 1
        presentation = app.Presentations.Open(str(path.resolve()), False, False, False)
        presentation.SaveAs(str(output_path.resolve()), 24)
        return "powerpoint-com", ""
    except Exception as exc:  # noqa: BLE001
        if output_path.exists() and output_path.stat().st_size > 0:
            return "powerpoint-com", ""
        return None, f"PowerPoint COM conversion failed: {type(exc).__name__}: {exc}"
    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


def _convert_ppt_with_libreoffice(path: Path, output_path: Path) -> tuple[str | None, str]:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        return None, "LibreOffice executable not found"
    with tempfile.TemporaryDirectory(prefix="ise_ppt_lo_") as tmp:
        tmp_dir = Path(tmp)
        command = [
            executable,
            "--headless",
            "--convert-to",
            "pptx",
            "--outdir",
            str(tmp_dir),
            str(path.resolve()),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except Exception as exc:  # noqa: BLE001
            return None, f"LibreOffice conversion failed: {type(exc).__name__}: {exc}"
        candidates = sorted(tmp_dir.glob("*.pptx"))
        if result.returncode != 0 or not candidates:
            detail = (result.stderr or result.stdout or "").strip()
            return None, f"LibreOffice conversion failed: {detail or 'no pptx output'}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidates[0], output_path)
        return "libreoffice", ""


def _pptx_image_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
    extracted_image_dir: Path,
    metadata_path: Path | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> list[CanonicalImage]:
    try:
        with ZipFile(path) as archive:
            media_names = sorted(name for name in archive.namelist() if name.startswith("ppt/media/"))
            records: list[CanonicalImage] = []
            for index, name in enumerate(media_names):
                image_bytes = archive.read(name)
                image_suffix = Path(name).suffix.lower() or ".bin"
                image_id = _stable_id(rel_path, "pptx-media", index, hashlib.sha1(image_bytes).hexdigest())
                image_path = rel_path
                metadata = {"embedded_name": name, "embedded_size": len(image_bytes), **_file_metadata(metadata_path or path)}
                if metadata_extra:
                    metadata.update(metadata_extra)
                if config.copy_extracted_images and image_suffix in IMAGE_EXTENSIONS:
                    target = extracted_image_dir / f"{image_id}{image_suffix}"
                    target.write_bytes(image_bytes)
                    image_path = target.relative_to(config.output_dir).as_posix()
                    metadata["materialized"] = True
                records.append(
                    CanonicalImage(
                        image_id=image_id,
                        source_path=rel_path,
                        source_extension=suffix,
                        image_path=image_path,
                        role="document_embedded_image",
                        locator=name,
                        metadata=metadata,
                    )
                )
            return records
    except (BadZipFile, KeyError):
        return []


def _pdf_text(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        parts = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"Page {index}:\n{text}")
        if parts:
            return "\n\n".join(parts), "pdf-pypdf"
    except Exception:
        pass
    return f"PDF file available at {path.name}. Local text extraction did not produce text.", "pdf-placeholder"


def _pdf_image_records(
    path: Path,
    rel_path: str,
    suffix: str,
    config: CanonicalBuildConfig,
    extracted_image_dir: Path,
) -> list[CanonicalImage]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return []

    records: list[CanonicalImage] = []
    doc = fitz.open(str(path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_index, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                image = doc.extract_image(xref)
                image_bytes = image.get("image") or b""
                image_suffix = "." + str(image.get("ext") or "bin").lower()
                image_id = _stable_id(rel_path, "pdf-image", page_index + 1, image_index, hashlib.sha1(image_bytes).hexdigest())
                image_path = rel_path
                metadata = {
                    "page": page_index + 1,
                    "xref": xref,
                    "embedded_size": len(image_bytes),
                    **_file_metadata(path),
                }
                if config.copy_extracted_images and image_suffix in IMAGE_EXTENSIONS:
                    target = extracted_image_dir / f"{image_id}{image_suffix}"
                    target.write_bytes(image_bytes)
                    image_path = target.relative_to(config.output_dir).as_posix()
                    metadata["materialized"] = True
                records.append(
                    CanonicalImage(
                        image_id=image_id,
                        source_path=rel_path,
                        source_extension=suffix,
                        image_path=image_path,
                        role="document_embedded_image",
                        locator=f"page={page_index + 1};image={image_index}",
                        metadata=metadata,
                    )
                )
        return records
    finally:
        doc.close()


def _table_profile(path: Path, rel_path: str, suffix: str, config: CanonicalBuildConfig) -> dict[str, Any]:
    if suffix in {".csv", ".tsv"}:
        rows = _csv_rows(path, config.max_table_preview_rows, delimiter="\t" if suffix == ".tsv" else None)
        return _profile_rows(rel_path, suffix, rows, parser="delimited-profile")
    if suffix == ".xlsx":
        sheets = _xlsx_sheet_rows(path, config.max_table_preview_rows)
        return _profile_sheets(rel_path, suffix, sheets, parser="xlsx-profile")
    if suffix == ".sql":
        text = _read_text(path, config.max_text_chars)
        description = _sql_description(rel_path, text)
        return {
            "parser": "sql-raw",
            "preview_text": text,
            "description": description,
            "columns": [],
            "sample_rows": [],
            "tail_sample_rows": [],
            "llm_description": "",
            "metadata_text": "",
            "table_shape": {"kind": "sql", "statement_count": len([part for part in text.split(";") if part.strip()])},
        }
    preview_text = f"Raw table file available at {path.name}. Detailed parsing is deferred."
    return {
        "parser": "table-raw",
        "preview_text": preview_text,
        "description": preview_text,
        "columns": [],
        "sample_rows": [],
        "tail_sample_rows": [],
        "llm_description": "",
        "metadata_text": "",
        "table_shape": {"kind": "raw"},
    }


def _table_preview(path: Path, suffix: str, config: CanonicalBuildConfig) -> tuple[str, str]:
    if suffix in {".csv", ".tsv"}:
        return _csv_preview(path, config.max_table_preview_rows, delimiter="\t" if suffix == ".tsv" else None), "delimited-preview"
    if suffix == ".xlsx":
        return _xlsx_preview(path, config.max_table_preview_rows), "xlsx-preview"
    if suffix == ".sql":
        return _read_text(path, config.max_text_chars), "sql-raw"
    return f"Raw table file available at {path.name}. Detailed parsing is deferred.", "table-raw"


def _csv_rows(path: Path, max_rows: int, delimiter: str | None = None) -> list[list[str]]:
    text = _read_text(path, 500_000)
    delimiter = delimiter or ","
    rows: list[list[str]] = []
    for index, row in enumerate(csv.reader(text.splitlines(), delimiter=delimiter)):
        if index >= max_rows:
            break
        rows.append([str(cell).strip() for cell in row])
    return rows


def _csv_preview(path: Path, max_rows: int, delimiter: str | None = None) -> str:
    return "\n".join("\t".join(row) for row in _csv_rows(path, max_rows, delimiter))


def _xlsx_sheet_rows(path: Path, max_rows_per_sheet: int) -> list[dict[str, Any]]:
    if path.stat().st_size > 30_000_000:
        return [
            {
                "sheet_name": "workbook",
                "rows": [[f"Large workbook {path.name}; detailed table preview is deferred."]],
                "deferred": True,
            }
        ]
    try:
        with ZipFile(path) as archive:
            shared = _xlsx_shared_strings(archive)
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pkgrel:Relationship", SHEET_NS)}
            sheets: list[dict[str, Any]] = []
            for sheet in workbook.findall("main:sheets/main:sheet", SHEET_NS):
                sheet_name = sheet.attrib.get("name", "Sheet")
                rel_id = sheet.attrib.get(f"{{{SHEET_NS['rel']}}}id", "")
                target = relmap.get(rel_id, "")
                if not target.startswith("/"):
                    target = "xl/" + target.lstrip("/")
                target = target.replace("xl/xl/", "xl/")
                rows = _xlsx_rows(archive, target, shared, max_rows_per_sheet)
                if rows:
                    sheets.append({"sheet_name": sheet_name, "rows": rows})
            return sheets
    except (BadZipFile, KeyError, ET.ParseError):
        return [{"sheet_name": "workbook", "rows": [[f"XLSX file available at {path.name}, but local preview failed."]]}]


def _xlsx_preview(path: Path, max_rows_per_sheet: int) -> str:
    return "\n\n".join(
        f"Sheet {sheet['sheet_name']}\n" + "\n".join("\t".join(row) for row in sheet.get("rows", []))
        for sheet in _xlsx_sheet_rows(path, max_rows_per_sheet)
    )


def _profile_sheets(rel_path: str, suffix: str, sheets: list[dict[str, Any]], parser: str) -> dict[str, Any]:
    parts: list[str] = []
    all_columns: list[str] = []
    all_samples: list[dict[str, str]] = []
    all_tail_samples: list[dict[str, str]] = []
    metadata_parts: list[str] = []
    sheet_summaries: list[dict[str, Any]] = []
    for sheet in sheets:
        sheet_name = str(sheet.get("sheet_name") or "Sheet")
        rows = sheet.get("rows") if isinstance(sheet.get("rows"), list) else []
        profile = _analyze_table_rows(rows)
        metadata = profile["metadata_text"]
        if metadata:
            metadata_parts.append(f"Sheet {sheet_name}: {metadata}")
        all_columns.extend(column for column in profile["columns"] if column not in all_columns)
        for row in profile["sample_rows"]:
            with_sheet = {"sheet": sheet_name, **row}
            all_samples.append(with_sheet)
        for row in profile["tail_sample_rows"]:
            with_sheet = {"sheet": sheet_name, **row}
            all_tail_samples.append(with_sheet)
        sheet_summaries.append(
            {
                "sheet_name": sheet_name,
                "columns": profile["columns"],
                "sample_row_count": len(profile["sample_rows"]),
                "metadata_row_count": profile["metadata_row_count"],
                "observed_row_count": len(rows),
            }
        )
        parts.append(f"Sheet {sheet_name}\n" + "\n".join("\t".join(str(cell) for cell in row) for row in rows))
    preview_text = "\n\n".join(parts)
    description = _table_description(
        rel_path=rel_path,
        suffix=suffix,
        parser=parser,
        columns=all_columns,
        sample_rows=all_samples[:5],
        tail_sample_rows=all_tail_samples[-5:],
        metadata_text="\n".join(metadata_parts),
        table_shape={"sheets": sheet_summaries, "sheet_count": len(sheets)},
    )
    return {
        "parser": parser,
        "preview_text": preview_text,
        "description": description,
        "columns": all_columns,
        "sample_rows": all_samples[:5],
        "tail_sample_rows": all_tail_samples[-5:],
        "llm_description": "",
        "metadata_text": "\n".join(metadata_parts),
        "table_shape": {"sheets": sheet_summaries, "sheet_count": len(sheets)},
    }


def _profile_rows(rel_path: str, suffix: str, rows: list[list[str]], parser: str) -> dict[str, Any]:
    profile = _analyze_table_rows(rows)
    preview_text = "\n".join("\t".join(row) for row in rows)
    table_shape = {
        "kind": "delimited",
        "observed_row_count": len(rows),
        "metadata_row_count": profile["metadata_row_count"],
        "header_row_index": profile["header_row_index"],
    }
    description = _table_description(
        rel_path=rel_path,
        suffix=suffix,
        parser=parser,
        columns=profile["columns"],
        sample_rows=profile["sample_rows"],
        tail_sample_rows=profile["tail_sample_rows"],
        metadata_text=profile["metadata_text"],
        table_shape=table_shape,
    )
    return {
        "parser": parser,
        "preview_text": preview_text,
        "description": description,
        "columns": profile["columns"],
        "sample_rows": profile["sample_rows"],
        "tail_sample_rows": profile["tail_sample_rows"],
        "llm_description": "",
        "metadata_text": profile["metadata_text"],
        "table_shape": table_shape,
    }


def _analyze_table_rows(rows: list[list[str]]) -> dict[str, Any]:
    cleaned_rows = [[str(cell).strip() for cell in row] for row in rows if any(str(cell).strip() for cell in row)]
    if not cleaned_rows:
        return {
            "columns": [],
            "sample_rows": [],
            "tail_sample_rows": [],
            "metadata_text": "",
            "metadata_row_count": 0,
            "header_row_index": None,
        }
    header_index = _guess_header_index(cleaned_rows)
    metadata_rows = cleaned_rows[:header_index] if header_index is not None else []
    header = cleaned_rows[header_index] if header_index is not None else _default_columns(cleaned_rows[0])
    data_rows = cleaned_rows[(header_index + 1) if header_index is not None else 0 :]
    columns = _normalize_columns(header)
    sample_rows = [_row_to_dict(columns, row) for row in data_rows[:5]]
    tail_sample_rows = [_row_to_dict(columns, row) for row in data_rows[-5:]]
    metadata_text = "\n".join(" | ".join(cell for cell in row if cell) for row in metadata_rows)
    return {
        "columns": columns,
        "sample_rows": sample_rows,
        "tail_sample_rows": tail_sample_rows,
        "metadata_text": metadata_text,
        "metadata_row_count": len(metadata_rows),
        "header_row_index": header_index,
    }


def _guess_header_index(rows: list[list[str]]) -> int | None:
    best_index: int | None = None
    best_score = -1
    for index, row in enumerate(rows[:20]):
        non_empty = [cell for cell in row if cell]
        if len(non_empty) < 2:
            continue
        unique_ratio = len(set(non_empty)) / len(non_empty)
        text_cells = sum(1 for cell in non_empty if re.search(r"[A-Za-z_\u00C0-\u1EF9]", cell))
        next_row = rows[index + 1] if index + 1 < len(rows) else []
        next_non_empty = sum(1 for cell in next_row if cell)
        score = len(non_empty) + text_cells + int(unique_ratio == 1.0) + int(next_non_empty >= max(2, len(non_empty) // 2))
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _normalize_columns(header: list[str]) -> list[str]:
    columns: list[str] = []
    for index, value in enumerate(header):
        column = re.sub(r"\s+", " ", str(value).strip()) or f"column_{index + 1}"
        if column in columns:
            column = f"{column}_{index + 1}"
        columns.append(column)
    return columns


def _default_columns(row: list[str]) -> list[str]:
    return [f"column_{index + 1}" for index in range(max(1, len(row)))]


def _row_to_dict(columns: list[str], row: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for index, column in enumerate(columns):
        output[column] = row[index] if index < len(row) else ""
    return output


def _table_description(
    rel_path: str,
    suffix: str,
    parser: str,
    columns: list[str],
    sample_rows: list[dict[str, str]],
    tail_sample_rows: list[dict[str, str]],
    metadata_text: str,
    table_shape: dict[str, Any],
) -> str:
    parts = [
        f"Source table: {rel_path}",
        f"File type: {suffix.lstrip('.') or 'table'}",
        f"Parser: {parser}",
    ]
    if metadata_text.strip():
        parts.append("Metadata or explanation text before/around the table:\n" + metadata_text.strip()[:2500])
    if columns:
        parts.append("Column names: " + ", ".join(columns[:80]))
    if sample_rows:
        parts.append("First sample rows:\n" + json.dumps(sample_rows, ensure_ascii=False, indent=2)[:5000])
    if tail_sample_rows and tail_sample_rows != sample_rows:
        parts.append("Last sample rows:\n" + json.dumps(tail_sample_rows, ensure_ascii=False, indent=2)[:5000])
    if table_shape:
        parts.append("Observed table shape/profile: " + json.dumps(table_shape, ensure_ascii=False, sort_keys=True)[:2500])
    return "\n\n".join(parts)


def _sql_description(rel_path: str, text: str) -> str:
    tables = sorted(set(re.findall(r"\b(?:CREATE\s+TABLE|FROM|JOIN|INTO|UPDATE)\s+[`\"[]?([A-Za-z_][\w.]*)", text, flags=re.IGNORECASE)))
    columns = sorted(set(re.findall(r"\b([A-Za-z_][\w]*)\s+(?:INTEGER|INT|REAL|TEXT|VARCHAR|DATE|FLOAT|DOUBLE|BOOLEAN|NUMERIC)\b", text, flags=re.IGNORECASE)))
    parts = [f"Source SQL table/script: {rel_path}", "File type: sql", "Parser: sql-raw"]
    if tables:
        parts.append("Referenced or created tables: " + ", ".join(tables[:50]))
    if columns:
        parts.append("Column-like definitions: " + ", ".join(columns[:80]))
    parts.append("SQL preview:\n" + text[:5000])
    return "\n\n".join(parts)


def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in item.findall(".//main:t", SHEET_NS)) for item in root.findall("main:si", SHEET_NS)]


def _xlsx_rows(archive: ZipFile, target: str, shared: list[str], max_rows: int) -> list[list[str]]:
    root = ET.fromstring(archive.read(target))
    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", SHEET_NS):
        values: list[str] = []
        for cell in row.findall("main:c", SHEET_NS):
            index = _col_to_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            values[index] = _xlsx_cell_value(cell, shared)
        if any(values):
            rows.append(values)
        if len(rows) >= max_rows:
            break
    return rows


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find("main:v", SHEET_NS)
    if cell.attrib.get("t") == "s" and value is not None:
        index = int(value.text or 0)
        return shared[index] if index < len(shared) else ""
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", SHEET_NS))
    return value.text if value is not None else ""


def _col_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "A1")
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _read_text(path: Path, max_chars: int) -> str:
    data = path.read_bytes()[: max_chars * 4]
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return data.decode(encoding)[:max_chars]
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")[:max_chars]


def _strip_html(text: str) -> str:
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip_depth = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() in {"script", "style", "noscript"}:
                self._skip_depth += 1

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
                self._skip_depth -= 1

        def handle_data(self, data: str) -> None:
            if self._skip_depth:
                return
            value = data.strip()
            if value:
                self.parts.append(value)

    parser = Parser()
    parser.feed(text)
    return html.unescape(" ".join(parser.parts))


def _file_metadata(path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(path.name)
    stat = path.stat()
    return {
        "size": stat.st_size,
        "sha256": None,
        "mime_type": mime_type,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
