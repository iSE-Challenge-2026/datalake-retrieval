"""Audit canonical artifacts before retrieval experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import CANONICAL_AUDIT_REPORTS_DIR, CANONICAL_DIR, OPENROUTER_VECTOR_RECORDS_PATH  # noqa: E402


FOCUS_GROUPS = {
    "topic_16": lambda path: "topic_16_page" in path,
    "number_image": lambda path: path.startswith("number_image/"),
    "KTCT": lambda path: path.startswith("KTCT/"),
    "tables": lambda path: Path(path).suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".sql", ".db", ".sqlite", ".sqlite3", ".duckdb", ".parquet"},
    "audio": lambda path: Path(path).suffix.lower() in {".m4a", ".mp3", ".wav", ".flac", ".ogg"},
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Audit canonical texts/images/tables/chunks/vector coverage.")
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DIR)
    parser.add_argument("--vector-records", type=Path, default=OPENROUTER_VECTOR_RECORDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_AUDIT_REPORTS_DIR)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    canonical_dir = _resolve(args.canonical_dir)
    vector_records_path = _resolve(args.vector_records)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = _resolve(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_audit_summary(canonical_dir, vector_records_path)
    _write_outputs(summary, output_dir)
    print(json.dumps({"output_dir": output_dir.as_posix(), "summary": summary["overview"]}, ensure_ascii=False, indent=2))


def build_audit_summary(canonical_dir: Path, vector_records_path: Path | None = None) -> dict[str, Any]:
    texts = _read_jsonl(canonical_dir / "texts.jsonl")
    images = _read_jsonl(canonical_dir / "images.jsonl")
    tables = _read_jsonl(canonical_dir / "tables.jsonl")
    chunks = _read_jsonl(canonical_dir / "chunks.jsonl")
    vectors = _read_jsonl(vector_records_path) if vector_records_path else []

    text_audit = _audit_texts(texts)
    image_audit = _audit_images(images, texts)
    table_audit = _audit_tables(tables)
    chunk_audit = _audit_chunks(chunks)
    embedding_audit = _audit_embeddings(chunks, vectors)
    focus_audit = _audit_focus_groups(texts, images, tables, chunks)

    return {
        "contract_version": "canonical-quality-audit-v1",
        "canonical_dir": canonical_dir.as_posix(),
        "vector_records_path": vector_records_path.as_posix() if vector_records_path else None,
        "overview": {
            "text_records": len(texts),
            "image_records": len(images),
            "table_records": len(tables),
            "chunk_records": len(chunks),
            "vector_records": len(vectors),
        },
        "texts": text_audit,
        "images": image_audit,
        "tables": table_audit,
        "chunks": chunk_audit,
        "embeddings": embedding_audit,
        "focus_groups": focus_audit,
    }


def _audit_texts(texts: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(str(row.get("text") or "").strip()) for row in texts]
    placeholder_count = sum(1 for row in texts if _looks_like_placeholder(str(row.get("text") or ""), str(row.get("parser") or "")))
    return {
        "count": len(texts),
        "by_extension": _counter(row.get("source_extension") for row in texts),
        "by_parser": _counter(row.get("parser") for row in texts),
        "by_role": _counter(row.get("role") for row in texts),
        "empty_count": sum(1 for length in lengths if length == 0),
        "too_short_count": sum(1 for length in lengths if 0 < length < 30),
        "placeholder_count": placeholder_count,
        "length_stats": _stats(lengths),
    }


def _audit_images(images: list[dict[str, Any]], texts: list[dict[str, Any]]) -> dict[str, Any]:
    ocr_sources = {str(row.get("source_path") or "") for row in texts if _is_raw_image_ocr_text(row)}
    raw_images = [row for row in images if str(row.get("role") or "") == "raw_image"]
    raw_with_description = [row for row in raw_images if str(row.get("description") or "").strip()]
    raw_with_ocr = [row for row in raw_images if str(row.get("source_path") or "") in ocr_sources]
    extracted_roles = {"raw_image_extracted_region", "api_image_region", "api_extracted_image", "document_embedded_image"}
    description_lengths = [len(str(row.get("description") or "").strip()) for row in images]
    return {
        "count": len(images),
        "by_extension": _counter(row.get("source_extension") for row in images),
        "by_role": _counter(row.get("role") for row in images),
        "raw_image_count": len(raw_images),
        "raw_image_with_ocr_text": len(raw_with_ocr),
        "raw_image_with_description": len(raw_with_description),
        "raw_image_missing_ocr_text": sorted(
            str(row.get("source_path") or "") for row in raw_images if str(row.get("source_path") or "") not in ocr_sources
        )[:100],
        "extracted_image_count": sum(1 for row in images if str(row.get("role") or "") in extracted_roles),
        "description_length_stats": _stats(description_lengths),
    }


def _audit_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(tables),
        "by_extension": _counter(row.get("source_extension") for row in tables),
        "by_parser": _counter(row.get("parser") for row in tables),
        "by_role": _counter(row.get("role") for row in tables),
        "with_description": sum(1 for row in tables if str(row.get("description") or "").strip()),
        "with_llm_description": sum(1 for row in tables if str(row.get("llm_description") or "").strip()),
        "with_columns": sum(1 for row in tables if isinstance(row.get("columns"), list) and row.get("columns")),
        "with_sample_rows": sum(1 for row in tables if isinstance(row.get("sample_rows"), list) and row.get("sample_rows")),
        "xlsx_sheet_records": sum(1 for row in tables if "#sheet=" in str(row.get("table_path") or "")),
    }


def _is_raw_image_ocr_text(row: dict[str, Any]) -> bool:
    role = str(row.get("role") or "")
    extension = str(row.get("source_extension") or "").lower()
    text = str(row.get("text") or "").strip()
    return bool(text) and extension in IMAGE_EXTENSIONS and role in {"raw_image_ocr_text", "api_text"}


def _audit_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(str(row.get("text") or "")) for row in chunks]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        by_source[str(row.get("source_path") or "")].append(row)
    top_sources = sorted(
        (
            {
                "source_path": source,
                "chunk_count": len(rows),
                "source_kinds": sorted({str(row.get("source_kind") or "") for row in rows}),
                "max_chunk_length": max(len(str(row.get("text") or "")) for row in rows),
            }
            for source, rows in by_source.items()
            if source
        ),
        key=lambda item: item["chunk_count"],
        reverse=True,
    )[:30]
    return {
        "count": len(chunks),
        "by_source_kind": _counter(row.get("source_kind") for row in chunks),
        "by_source_role": _counter(row.get("source_role") for row in chunks),
        "source_count": len(by_source),
        "length_stats": _stats(lengths),
        "top_sources_by_chunk_count": top_sources,
    }


def _audit_embeddings(chunks: list[dict[str, Any]], vectors: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ids = {str(row.get("chunk_id") or "") for row in chunks if row.get("chunk_id")}
    vector_ids = {str(row.get("chunk_id") or row.get("vector_id") or "") for row in vectors if row.get("chunk_id") or row.get("vector_id")}
    dimensions = [int(row.get("embedding_dimension") or len(row.get("embedding") or [])) for row in vectors if row.get("embedding") or row.get("embedding_dimension")]
    models = _counter(row.get("embedding_model") for row in vectors)
    comparable_hashes = 0
    stale_hashes = 0
    vector_by_id = {str(row.get("chunk_id") or row.get("vector_id") or ""): row for row in vectors}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        vector = vector_by_id.get(chunk_id)
        if not vector:
            continue
        expected_hash = _sha256(str(chunk.get("embedding_text") or ""))
        actual_hash = str((vector.get("metadata") or {}).get("embedding_text_hash") or "")
        if actual_hash:
            comparable_hashes += 1
            if actual_hash != expected_hash:
                stale_hashes += 1
    return {
        "chunk_count": len(chunk_ids),
        "vector_count": len(vector_ids),
        "missing_vector_count": len(chunk_ids - vector_ids),
        "extra_vector_count": len(vector_ids - chunk_ids),
        "missing_vector_sample": sorted(chunk_ids - vector_ids)[:50],
        "extra_vector_sample": sorted(vector_ids - chunk_ids)[:50],
        "embedding_models": models,
        "embedding_dimensions": _counter(dimensions),
        "comparable_hash_count": comparable_hashes,
        "stale_hash_count": stale_hashes,
    }


def _audit_focus_groups(
    texts: list[dict[str, Any]],
    images: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_kind = {"texts": texts, "images": images, "tables": tables, "chunks": chunks}
    output: dict[str, Any] = {}
    for group_name, matcher in FOCUS_GROUPS.items():
        group: dict[str, Any] = {}
        for kind, rows in rows_by_kind.items():
            matched = [row for row in rows if matcher(str(row.get("source_path") or ""))]
            group[kind] = {
                "count": len(matched),
                "roles": _counter(row.get("role") or row.get("source_role") for row in matched),
                "parsers": _counter(row.get("parser") for row in matched),
                "sample_sources": sorted({str(row.get("source_path") or "") for row in matched})[:20],
            }
        output[group_name] = group
    return output


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    _write_top_sources_csv(summary["chunks"]["top_sources_by_chunk_count"], output_dir / "top_sources_by_chunk_count.csv")


def _markdown_report(summary: dict[str, Any]) -> str:
    overview = summary["overview"]
    lines = [
        "# Canonical Quality Audit",
        "",
        "## Overview",
        "",
        f"- Text records: {overview['text_records']}",
        f"- Image records: {overview['image_records']}",
        f"- Table records: {overview['table_records']}",
        f"- Chunk records: {overview['chunk_records']}",
        f"- Vector records: {overview['vector_records']}",
        "",
        "## Key Checks",
        "",
        f"- Placeholder text records: {summary['texts']['placeholder_count']}",
        f"- Raw images with OCR text: {summary['images']['raw_image_with_ocr_text']}/{summary['images']['raw_image_count']}",
        f"- Extracted image records: {summary['images']['extracted_image_count']}",
        f"- Tables with LLM description: {summary['tables']['with_llm_description']}/{summary['tables']['count']}",
        f"- Missing vectors: {summary['embeddings']['missing_vector_count']}",
        f"- Stale comparable embedding hashes: {summary['embeddings']['stale_hash_count']}",
        "",
        "## Top Sources By Chunk Count",
        "",
        "| Source | Chunks | Kinds | Max Chunk Length |",
        "|---|---:|---|---:|",
    ]
    for row in summary["chunks"]["top_sources_by_chunk_count"][:15]:
        lines.append(
            f"| {row['source_path']} | {row['chunk_count']} | {', '.join(row['source_kinds'])} | {row['max_chunk_length']} |"
        )
    lines.extend(["", "## Focus Groups", ""])
    for name, group in summary["focus_groups"].items():
        lines.append(f"### {name}")
        for kind, stats in group.items():
            lines.append(f"- {kind}: {stats['count']}")
        lines.append("")
    return "\n".join(lines)


def _write_top_sources_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "chunk_count", "source_kinds", "max_chunk_length"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "source_kinds": ", ".join(row.get("source_kinds") or [])})


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) if value not in (None, "") else "<missing>" for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": round(float(statistics.median(values)), 2),
        "mean": round(float(statistics.mean(values)), 2),
        "max": max(values),
    }


def _looks_like_placeholder(text: str, parser: str) -> bool:
    value = (text or "").strip().lower()
    return (
        "placeholder" in (parser or "").lower()
        or "available at" in value[:300]
        or "is disabled" in value[:300]
        or "not implemented" in value[:300]
        or "did not produce text" in value[:300]
    )


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
