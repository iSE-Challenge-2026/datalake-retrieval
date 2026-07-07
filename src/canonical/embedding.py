"""Chunk and embed canonical text/image/table artifacts for retrieval experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol
import hashlib
import json
import math
import re


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class CanonicalChunk:
    chunk_id: str
    source_path: str
    source_kind: str
    source_record_id: str
    source_role: str
    locator: str
    text: str
    embedding_text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkBuildConfig:
    canonical_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


@dataclass(frozen=True)
class EmbeddingBuildConfig:
    chunks_path: Path
    output_dir: Path
    model: str
    dimension: int = 1536
    batch_size: int = 32


def build_canonical_chunks(config: ChunkBuildConfig) -> tuple[list[CanonicalChunk], dict[str, Any]]:
    """Create chunks from canonical text records, image descriptions, and table descriptions."""
    chunks: list[CanonicalChunk] = []
    canonical_dir = config.canonical_dir
    for row in _read_jsonl(canonical_dir / "texts.jsonl"):
        text = str(row.get("text") or "").strip()
        if not text or _looks_like_placeholder(text, str(row.get("parser") or "")):
            continue
        chunks.extend(
            _record_chunks(
                source_path=str(row.get("source_path") or ""),
                source_kind="text",
                source_record_id=str(row.get("text_id") or ""),
                source_role=str(row.get("role") or ""),
                locator=str(row.get("locator") or "file"),
                text=text,
                metadata={
                    "source_extension": row.get("source_extension"),
                    "parser": row.get("parser"),
                    "canonical_record_type": "text",
                },
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        )

    for row in _read_jsonl(canonical_dir / "images.jsonl"):
        description = str(row.get("description") or "").strip()
        if not description or _looks_like_placeholder(description, ""):
            continue
        chunks.extend(
            _record_chunks(
                source_path=str(row.get("source_path") or ""),
                source_kind="image",
                source_record_id=str(row.get("image_id") or ""),
                source_role=str(row.get("role") or ""),
                locator=str(row.get("locator") or "image"),
                text=description,
                metadata={
                    "source_extension": row.get("source_extension"),
                    "image_path": row.get("image_path"),
                    "canonical_record_type": "image",
                },
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        )

    for row in _read_jsonl(canonical_dir / "tables.jsonl"):
        structured_description = str(row.get("description") or "").strip()
        llm_description = str(row.get("llm_description") or "").strip()
        description = _compose_table_embedding_text(llm_description, structured_description)
        if not description:
            description = _fallback_table_description(row)
        if not description or _looks_like_placeholder(description, str(row.get("parser") or "")):
            continue
        chunks.extend(
            _record_chunks(
                source_path=str(row.get("source_path") or ""),
                source_kind="table",
                source_record_id=str(row.get("table_id") or ""),
                source_role=str(row.get("role") or ""),
                locator=str(row.get("locator") or "table"),
                text=description,
                metadata={
                    "source_extension": row.get("source_extension"),
                    "parser": row.get("parser"),
                    "canonical_record_type": "table",
                    "table_path": row.get("table_path"),
                    "columns": row.get("columns") if isinstance(row.get("columns"), list) else [],
                    "has_llm_description": bool(llm_description),
                    "table_shape": row.get("table_shape") if isinstance(row.get("table_shape"), dict) else {},
                },
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        )

    report = {
        "contract_version": "canonical-chunks-report-v1",
        "chunk_count": len(chunks),
        "source_kind_counts": _counts(chunk.source_kind for chunk in chunks),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    return chunks, report


def write_chunks(chunks: list[CanonicalChunk], path: Path, report: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, (asdict(chunk) for chunk in chunks))
    if report is not None:
        (path.parent / "chunks_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_vector_records(
    chunks: list[CanonicalChunk],
    provider: EmbeddingProvider,
    output_dir: Path,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed chunks with cache reuse and write vector records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "embedding_cache.jsonl"
    existing_cache = _load_embedding_cache(cache_path, provider.model)

    vector_records: list[dict[str, Any]] = []
    new_cache_rows: list[dict[str, Any]] = []
    to_embed: list[CanonicalChunk] = []
    reused_count = 0

    for chunk in chunks:
        text_hash = _text_hash(chunk.embedding_text)
        cache_key = _cache_key(chunk.chunk_id, provider.model, text_hash)
        cached = existing_cache.get(cache_key)
        if cached is not None:
            vector_records.append(_vector_record(chunk, cached, provider.model))
            reused_count += 1
            continue
        chunk.metadata["embedding_text_hash"] = text_hash
        to_embed.append(chunk)

    embedded_count = 0
    for batch_start in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_start : batch_start + batch_size]
        embeddings = provider.embed_batch([chunk.embedding_text for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            vector_records.append(_vector_record(chunk, embedding, provider.model))
            text_hash = chunk.metadata["embedding_text_hash"]
            new_cache_rows.append(
                {
                    "cache_key": _cache_key(chunk.chunk_id, provider.model, text_hash),
                    "chunk_id": chunk.chunk_id,
                    "embedding_model": provider.model,
                    "embedding_text_hash": text_hash,
                    "embedding": embedding,
                }
            )
            embedded_count += 1

    _append_jsonl(cache_path, new_cache_rows)
    vector_records.sort(key=lambda item: item["chunk_id"])
    report = {
        "contract_version": "canonical-embedding-report-v1",
        "status": "passed",
        "model": provider.model,
        "dimension": provider.dimension,
        "chunk_count": len(chunks),
        "generated_count": len(vector_records),
        "embedded_count": embedded_count,
        "reused_count": reused_count,
        "batch_size": batch_size,
    }
    return vector_records, report


def write_vector_outputs(vector_records: list[dict[str, Any]], report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "vector_records.jsonl", vector_records)
    (output_dir / "embedding_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[CanonicalChunk]:
    return [CanonicalChunk(**row) for row in _read_jsonl(path)]


def load_vector_records(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _record_chunks(
    source_path: str,
    source_kind: str,
    source_record_id: str,
    source_role: str,
    locator: str,
    text: str,
    metadata: dict[str, Any],
    chunk_size: int,
    overlap: int,
) -> list[CanonicalChunk]:
    if not source_path or not source_record_id:
        return []
    pieces = _split_text(text, chunk_size, overlap)
    chunks: list[CanonicalChunk] = []
    for index, (piece, start, end) in enumerate(pieces):
        chunk_locator = f"{locator};chunk={index}"
        embedding_text = _embedding_text(source_path, source_kind, source_role, chunk_locator, piece)
        chunk_id = _stable_id(source_record_id, source_kind, index, start, end, _text_hash(piece))
        chunks.append(
            CanonicalChunk(
                chunk_id=chunk_id,
                source_path=source_path,
                source_kind=source_kind,
                source_record_id=source_record_id,
                source_role=source_role,
                locator=chunk_locator,
                text=piece,
                embedding_text=embedding_text,
                start_char=start,
                end_char=end,
                metadata=dict(metadata),
            )
        )
    return chunks


def _fallback_table_description(row: dict[str, Any]) -> str:
    source_path = str(row.get("source_path") or "")
    preview = str(row.get("preview_text") or "").strip()
    columns = row.get("columns") if isinstance(row.get("columns"), list) else []
    parts = [f"Source table: {source_path}"]
    if columns:
        parts.append("Column names: " + ", ".join(str(column) for column in columns))
    if preview:
        parts.append("Preview:\n" + preview[:6000])
    return "\n\n".join(parts)


def _compose_table_embedding_text(llm_description: str, structured_description: str) -> str:
    parts = []
    if llm_description:
        parts.append("LLM table description:\n" + llm_description)
    if structured_description:
        parts.append("Structured table profile:\n" + structured_description)
    return "\n\n".join(parts)


def _embedding_text(source_path: str, source_kind: str, source_role: str, locator: str, text: str) -> str:
    return f"Source: {source_path}\nType: {source_kind}\nRole: {source_role}\nLocator: {locator}\n\n{text}"


def _split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    value = re.sub(r"\s+", " ", text or "").strip()
    if not value:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    chunks: list[tuple[str, int, int]] = []
    start = 0
    step = chunk_size - overlap
    while start < len(value):
        end = min(start + chunk_size, len(value))
        chunks.append((value[start:end], start, end))
        if end == len(value):
            break
        start += step
    return chunks


def _vector_record(chunk: CanonicalChunk, embedding: list[float], model: str) -> dict[str, Any]:
    return {
        "vector_id": chunk.chunk_id,
        "chunk_id": chunk.chunk_id,
        "source_path": chunk.source_path,
        "source_kind": chunk.source_kind,
        "source_role": chunk.source_role,
        "locator": chunk.locator,
        "text": chunk.text,
        "embedding": embedding,
        "embedding_model": model,
        "embedding_dimension": len(embedding),
        "metadata": chunk.metadata,
    }


def _load_embedding_cache(cache_path: Path, model: str) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for row in _read_jsonl(cache_path):
        if row.get("embedding_model") != model:
            continue
        embedding = row.get("embedding")
        if isinstance(embedding, list):
            cache[str(row.get("cache_key"))] = [float(value) for value in embedding]
    return cache


def _cache_key(chunk_id: str, model: str, text_hash: str) -> str:
    return _stable_id(chunk_id, model, text_hash)


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def _looks_like_placeholder(text: str, parser: str) -> bool:
    value = (text or "").strip().lower()
    return (
        "placeholder" in (parser or "").lower()
        or "available at" in value[:300]
        or "is disabled" in value[:300]
        or "not implemented" in value[:300]
        or "did not produce text" in value[:300]
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
