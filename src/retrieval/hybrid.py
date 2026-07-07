"""Hybrid retrieval fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RetrievalLike(Protocol):
    source_path: str
    score: float
    source_kind: str
    reasons: list[str]
    chunks: list[Any]


@dataclass
class HybridRetrievalResult:
    source_path: str
    score: float
    source_kind: str
    reasons: list[str] = field(default_factory=list)
    chunks: list[Any] = field(default_factory=list)


def reciprocal_rank_fusion(
    lexical_results: list[RetrievalLike],
    vector_results: list[RetrievalLike],
    top_k: int,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[HybridRetrievalResult]:
    """Fuse lexical and vector source rankings with weighted RRF."""
    by_source: dict[str, HybridRetrievalResult] = {}
    raw_scores: dict[str, float] = {}

    _accumulate(by_source, raw_scores, lexical_results, "lexical", rrf_k, lexical_weight)
    _accumulate(by_source, raw_scores, vector_results, "vector", rrf_k, vector_weight)

    for source_path, score in raw_scores.items():
        by_source[source_path].score = score

    results = list(by_source.values())
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def _accumulate(
    by_source: dict[str, HybridRetrievalResult],
    raw_scores: dict[str, float],
    results: list[RetrievalLike],
    reason: str,
    rrf_k: int,
    weight: float,
) -> None:
    for rank, result in enumerate(results, start=1):
        source_path = result.source_path
        if source_path not in by_source:
            by_source[source_path] = HybridRetrievalResult(
                source_path=source_path,
                score=0.0,
                source_kind=result.source_kind,
                reasons=[],
                chunks=[],
            )
            raw_scores[source_path] = 0.0
        raw_scores[source_path] += weight / (rrf_k + rank)
        fused = by_source[source_path]
        fused.reasons = sorted(set(fused.reasons + [reason] + list(result.reasons or [])))
        fused.chunks.extend(result.chunks[:3])
