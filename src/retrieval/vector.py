"""Brute-force vector retrieval over canonical embedding JSONL records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
import json

from src.canonical.embedding import cosine_similarity


DEFAULT_VECTOR_CHUNK_K = 200
DEFAULT_VECTOR_AGGREGATION = "max_mean_top3"
MATCHED_CHUNK_BONUS_CAP = 3


@dataclass
class VectorChunkHit:
    chunk_id: str
    source_path: str
    source_kind: str
    source_role: str
    locator: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRetrievalResult:
    source_path: str
    score: float
    source_kind: str
    reasons: list[str] = field(default_factory=lambda: ["vector"])
    chunks: list[VectorChunkHit] = field(default_factory=list)
    score_components: dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    def __init__(self, vector_records: list[dict[str, Any]], aggregation: str = DEFAULT_VECTOR_AGGREGATION) -> None:
        self.vector_records = vector_records
        self.dimension = _infer_dimension(vector_records)
        self.aggregation = aggregation
        self.source_chunk_counts = Counter(str(record.get("source_path") or "") for record in vector_records)

    @classmethod
    def from_jsonl(cls, path: str | Path, aggregation: str = DEFAULT_VECTOR_AGGREGATION) -> "VectorRetriever":
        return cls(_read_jsonl(Path(path)), aggregation=aggregation)

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        chunk_k: int = DEFAULT_VECTOR_CHUNK_K,
    ) -> list[VectorRetrievalResult]:
        hits: list[VectorChunkHit] = []
        for record in self.vector_records:
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                continue
            score = cosine_similarity(query_embedding, [float(value) for value in embedding])
            hits.append(
                VectorChunkHit(
                    chunk_id=str(record.get("chunk_id") or record.get("vector_id") or ""),
                    source_path=str(record.get("source_path") or ""),
                    source_kind=str(record.get("source_kind") or "text"),
                    source_role=str(record.get("source_role") or ""),
                    locator=str(record.get("locator") or ""),
                    score=score,
                    text=str(record.get("text") or ""),
                    metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return _group_hits(hits[:chunk_k], self.source_chunk_counts, self.aggregation)[:top_k]


def _group_hits(
    hits: list[VectorChunkHit],
    source_chunk_counts: Counter[str] | None = None,
    aggregation: str = DEFAULT_VECTOR_AGGREGATION,
) -> list[VectorRetrievalResult]:
    grouped: dict[str, list[VectorChunkHit]] = defaultdict(list)
    for hit in hits:
        if hit.source_path:
            grouped[hit.source_path].append(hit)

    results: list[VectorRetrievalResult] = []
    for source_path, source_hits in grouped.items():
        source_hits.sort(key=lambda item: item.score, reverse=True)
        score, components = _source_score(source_path, source_hits, source_chunk_counts or Counter(), aggregation)
        results.append(
            VectorRetrievalResult(
                source_path=source_path,
                score=score,
                source_kind=source_hits[0].source_kind,
                chunks=source_hits[:3],
                score_components=components,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _source_score(
    source_path: str,
    source_hits: list[VectorChunkHit],
    source_chunk_counts: Counter[str],
    aggregation: str,
) -> tuple[float, dict[str, Any]]:
    top3 = source_hits[:3]
    max_chunk_score = top3[0].score
    mean_top3 = sum(hit.score for hit in top3) / len(top3)
    matched_chunk_count = len(source_hits)
    source_chunk_count = source_chunk_counts.get(source_path, matched_chunk_count)

    if aggregation == "legacy":
        score = source_hits[0].score + 0.05 * sum(hit.score for hit in source_hits[1:5])
    elif aggregation == "max_mean_top3":
        score = max_chunk_score + 0.10 * mean_top3 + 0.02 * min(matched_chunk_count, MATCHED_CHUNK_BONUS_CAP)
    else:
        raise ValueError(f"Unsupported vector aggregation: {aggregation}")

    return score, {
        "aggregation": aggregation,
        "max_chunk_score": max_chunk_score,
        "mean_top3_chunk_score": mean_top3,
        "matched_chunk_count": matched_chunk_count,
        "matched_chunk_bonus_count": min(matched_chunk_count, MATCHED_CHUNK_BONUS_CAP),
        "source_chunk_count": source_chunk_count,
    }


def _infer_dimension(vector_records: list[dict[str, Any]]) -> int:
    for record in vector_records:
        embedding = record.get("embedding")
        if isinstance(embedding, list):
            return len(embedding)
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
