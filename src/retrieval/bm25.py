"""BM25 retrieval over canonical chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
import json
import math

from .text import compact, tokenize


@dataclass
class BM25ChunkHit:
    chunk_id: str
    source_path: str
    source_kind: str
    source_role: str
    locator: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25RetrievalResult:
    source_path: str
    score: float
    source_kind: str
    reasons: list[str] = field(default_factory=lambda: ["bm25"])
    chunks: list[BM25ChunkHit] = field(default_factory=list)


class BM25Retriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.2,
        b: float = 0.75,
        path_weight: float = 1.6,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.path_weight = path_weight
        self.documents = [_document_for_chunk(chunk) for chunk in chunks]
        self.path_documents = [str(chunk.get("source_path") or "").replace("/", " ") for chunk in chunks]
        self.doc_tokens = [tokenize(document) for document in self.documents]
        self.path_tokens = [tokenize(document) for document in self.path_documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.idf = _idf(self.doc_tokens)
        self.path_idf = _idf(self.path_tokens)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Retriever":
        return cls(_read_jsonl(Path(path)))

    def retrieve(self, question: str, top_k: int = 20, chunk_k: int = 120) -> list[BM25RetrievalResult]:
        query_tokens = tokenize(question)
        hits: list[BM25ChunkHit] = []
        for index, chunk in enumerate(self.chunks):
            score = _bm25_score(
                query_tokens,
                self.doc_tokens[index],
                self.idf,
                self.doc_lengths[index],
                self.avg_doc_length,
                self.k1,
                self.b,
            )
            path_score = _bm25_score(
                query_tokens,
                self.path_tokens[index],
                self.path_idf,
                len(self.path_tokens[index]),
                max(1.0, sum(len(tokens) for tokens in self.path_tokens) / len(self.path_tokens)) if self.path_tokens else 1.0,
                self.k1,
                self.b,
            )
            score += self.path_weight * path_score
            if score <= 0:
                continue
            hits.append(
                BM25ChunkHit(
                    chunk_id=str(chunk.get("chunk_id") or ""),
                    source_path=str(chunk.get("source_path") or ""),
                    source_kind=str(chunk.get("source_kind") or "text"),
                    source_role=str(chunk.get("source_role") or ""),
                    locator=str(chunk.get("locator") or ""),
                    score=score,
                    text=compact(str(chunk.get("text") or ""), 900),
                    metadata=chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {},
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return _group_hits(hits[:chunk_k])[:top_k]


def _document_for_chunk(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(part)
        for part in (
            chunk.get("source_path"),
            chunk.get("source_kind"),
            chunk.get("source_role"),
            chunk.get("locator"),
            chunk.get("text"),
        )
        if part
    )


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    doc_length: int,
    avg_doc_length: float,
    k1: float,
    b: float,
) -> float:
    if not query_tokens or not doc_tokens or avg_doc_length <= 0:
        return 0.0
    counts = Counter(doc_tokens)
    score = 0.0
    for token in query_tokens:
        tf = counts.get(token, 0)
        if not tf:
            continue
        denom = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
        score += idf.get(token, 0.0) * (tf * (k1 + 1)) / denom
    return score


def _idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    document_count = len(tokenized_docs)
    df: Counter[str] = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))
    return {
        token: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for token, count in df.items()
    }


def _group_hits(hits: list[BM25ChunkHit]) -> list[BM25RetrievalResult]:
    grouped: dict[str, list[BM25ChunkHit]] = defaultdict(list)
    for hit in hits:
        if hit.source_path:
            grouped[hit.source_path].append(hit)
    results: list[BM25RetrievalResult] = []
    for source_path, source_hits in grouped.items():
        source_hits.sort(key=lambda item: item.score, reverse=True)
        score = source_hits[0].score + 0.05 * sum(hit.score for hit in source_hits[1:5])
        results.append(
            BM25RetrievalResult(
                source_path=source_path,
                score=score,
                source_kind=source_hits[0].source_kind,
                chunks=source_hits[:5],
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
