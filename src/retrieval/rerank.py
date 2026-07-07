"""Lightweight reranking for retrieved source candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import os
import re

from src.project_paths import OPENROUTER_RERANK_DIR

from .text import token_counts, weighted_overlap


class ChunkLike(Protocol):
    score: float
    text: str


class ResultLike(Protocol):
    source_path: str
    score: float
    source_kind: str
    reasons: list[str]
    chunks: list[ChunkLike]


@dataclass
class RerankedRetrievalResult:
    source_path: str
    score: float
    source_kind: str
    reasons: list[str] = field(default_factory=list)
    chunks: list[Any] = field(default_factory=list)
    rerank_score: float = 0.0
    rerank_features: dict[str, float] = field(default_factory=dict)


def heuristic_rerank(question: str, results: list[ResultLike], top_k: int) -> list[RerankedRetrievalResult]:
    """Rerank candidates using query/source/chunk evidence signals.

    This reranker intentionally avoids external API calls. It works best after a
    broad lexical/vector/hybrid retrieval pass, where candidate recall is high.
    """
    if not results:
        return []

    query_counts = token_counts(question)
    candidate_features = [_features(query_counts, result) for result in results]
    normalized = _normalize_features(candidate_features)

    reranked: list[RerankedRetrievalResult] = []
    for result, features in zip(results, normalized, strict=True):
        both_channels = 1.0 if {"lexical", "vector"}.issubset(set(result.reasons or [])) else 0.0
        score = (
            0.15 * features["base_score"]
            + 0.40 * features["max_chunk_score"]
            + 0.15 * features["mean_top_chunk_score"]
            + 0.15 * features["text_overlap"]
            + 0.10 * features["path_overlap"]
            + 0.05 * both_channels
        )
        reasons = sorted(set(list(result.reasons or []) + ["heuristic_rerank"]))
        reranked.append(
            RerankedRetrievalResult(
                source_path=result.source_path,
                score=score,
                source_kind=result.source_kind,
                reasons=reasons,
                chunks=list(result.chunks or []),
                rerank_score=score,
                rerank_features={**features, "both_channels": both_channels},
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k]


class OpenRouterLLMReranker:
    """OpenRouter chat-model reranker with JSON cache."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path = OPENROUTER_RERANK_DIR,
        api_key_env: str = "OPENROUTER_API_KEY",
        canonical_dir: str | Path | None = None,
        strategy: str = "current",
        filter_keep_false_zero: bool = False,
    ) -> None:
        self.model = model or os.getenv("OPENROUTER_RERANK_MODEL") or os.getenv("OPENROUTER_MODEL") or "google/gemini-2.5-flash-lite"
        self.cache_dir = Path(cache_dir)
        self.api_key_env = api_key_env
        self.source_summaries = _load_source_summaries(Path(canonical_dir)) if canonical_dir else {}
        if strategy not in {"current", "strict_source", "chunk_judge"}:
            raise ValueError(f"Unsupported rerank strategy: {strategy}")
        self.strategy = strategy
        self.filter_keep_false_zero = filter_keep_false_zero

    def rerank(self, question: str, results: list[ResultLike], top_k: int) -> list[RerankedRetrievalResult]:
        if not results:
            return []
        cache_path = self._cache_path(question, results)
        if cache_path.exists():
            scores = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            scores = self._call_model(question, results)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._apply_scores(results, scores, top_k)

    def _cache_path(self, question: str, results: list[ResultLike]) -> Path:
        payload = {
            "model": self.model,
            "strategy": self.strategy,
            "question": question,
            "candidates": [
                {
                    "source_path": result.source_path,
                    "score": round(float(result.score or 0.0), 6),
                    "score_components": getattr(result, "score_components", None),
                    "source_summary": self.source_summaries.get(result.source_path, {}),
                    "chunks": [str(getattr(chunk, "text", "") or "")[:800] for chunk in (result.chunks or [])[:3]],
                }
                for result in results
            ],
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / self.model.replace("/", "__") / f"{digest}.json"

    def _call_model(self, question: str, results: list[ResultLike]) -> dict[str, dict[str, Any]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing openai package. Install it with: pip install openai") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env} for OpenRouter rerank.")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://local.data-lake-retrieval",
                "X-Title": "Data-Lake Retrieval Rerank",
            },
        )
        candidates = [
            _candidate_payload(index, result, self.source_summaries.get(result.source_path))
            for index, result in enumerate(results, start=1)
        ]
        system_prompt, user_instructions = _rerank_prompts(self.strategy)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Question:\n"
                        f"{question}\n\n"
                        "Candidates:\n"
                        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
                        f"{user_instructions}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json_object(content)
        items = parsed.get("items") if isinstance(parsed, dict) else None
        scores: dict[str, dict[str, Any]] = {}
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                source_path = str(item.get("source_path") or "")
                if not source_path:
                    continue
                scores[source_path] = {
                    "relevance": _clip(float(item.get("relevance") or 0.0), 0.0, 1.0),
                    "reason": str(item.get("reason") or ""),
                    "evidence_role": str(item.get("evidence_role") or ""),
                    "keep": bool(item.get("keep", False)),
                    "required_entity_match": bool(item.get("required_entity_match", False)),
                    "required_data_match": bool(item.get("required_data_match", False)),
                    "chunk_judgements": item.get("chunk_judgements") if isinstance(item.get("chunk_judgements"), list) else [],
                }
        return scores

    def _apply_scores(
        self,
        results: list[ResultLike],
        scores: dict[str, dict[str, Any]],
        top_k: int,
    ) -> list[RerankedRetrievalResult]:
        base_scores = [float(result.score or 0.0) for result in results]
        base_min = min(base_scores)
        base_max = max(base_scores)
        reranked: list[RerankedRetrievalResult] = []
        for result in results:
            score_info = scores.get(result.source_path, {})
            relevance = _clip(float(score_info.get("relevance") or 0.0), 0.0, 1.0)
            base = 0.0 if base_max <= base_min else (float(result.score or 0.0) - base_min) / (base_max - base_min)
            final_score = 0.85 * relevance + 0.15 * base
            evidence_role = str(score_info.get("evidence_role", ""))
            keep = bool(score_info.get("keep", False))
            reason = str(score_info.get("reason", ""))
            filtered_as_irrelevant = _is_irrelevant_judgement(
                relevance,
                evidence_role,
                keep,
                reason,
                strategy=self.strategy,
                filter_keep_false_zero=self.filter_keep_false_zero,
            )
            reranked.append(
                RerankedRetrievalResult(
                    source_path=result.source_path,
                    score=final_score if not filtered_as_irrelevant else -1.0,
                    source_kind=result.source_kind,
                    reasons=sorted(set(list(result.reasons or []) + ["llm_rerank"])),
                    chunks=list(result.chunks or []),
                    rerank_score=final_score if not filtered_as_irrelevant else -1.0,
                    rerank_features={
                        "llm_relevance": relevance,
                        "base_score_normalized": base,
                        "llm_reason": reason,
                        "evidence_role": evidence_role,
                        "keep": keep,
                        "filtered_as_irrelevant": filtered_as_irrelevant,
                        "required_entity_match": bool(score_info.get("required_entity_match", False)),
                        "required_data_match": bool(score_info.get("required_data_match", False)),
                        "chunk_judgements": score_info.get("chunk_judgements") or [],
                    },
                )
            )
        kept = [item for item in reranked if not item.rerank_features.get("filtered_as_irrelevant")]
        kept.sort(key=lambda item: item.score, reverse=True)
        return kept[:top_k]


def _features(query_counts: dict[str, int], result: ResultLike) -> dict[str, float]:
    chunks = list(result.chunks or [])
    chunk_scores = [float(getattr(chunk, "score", 0.0) or 0.0) for chunk in chunks]
    chunk_scores.sort(reverse=True)
    top_chunk_text = " ".join(str(getattr(chunk, "text", "") or "") for chunk in chunks[:3])
    return {
        "base_score": float(result.score or 0.0),
        "max_chunk_score": chunk_scores[0] if chunk_scores else 0.0,
        "mean_top_chunk_score": sum(chunk_scores[:3]) / len(chunk_scores[:3]) if chunk_scores else 0.0,
        "path_overlap": weighted_overlap(query_counts, token_counts(result.source_path), {}),
        "text_overlap": weighted_overlap(query_counts, token_counts(top_chunk_text), {}),
    }


def _candidate_payload(index: int, result: ResultLike, source_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "rank": index,
        "source_path": result.source_path,
        "source_kind": result.source_kind,
        "source_extension": Path(result.source_path).suffix.lower(),
        "initial_score": round(float(result.score or 0.0), 6),
        "score_components": getattr(result, "score_components", None),
        "reasons": list(result.reasons or []),
        "source_summary": source_summary or {},
        "snippets": [
            {
                "chunk_index": chunk_index,
                "score": round(float(getattr(chunk, "score", 0.0) or 0.0), 6),
                "locator": str(getattr(chunk, "locator", "") or ""),
                "metadata": _compact_value(getattr(chunk, "metadata", {}), 900),
                "text": str(getattr(chunk, "text", "") or "")[:900],
            }
            for chunk_index, chunk in enumerate((result.chunks or [])[:3], start=1)
        ],
    }


def _rerank_prompts(strategy: str) -> tuple[str, str]:
    base_schema = (
        "Return only valid JSON with this schema:\n"
        "{\"items\":[{\"source_path\":\"...\",\"relevance\":0.0,"
        "\"evidence_role\":\"direct_answer|data_item|aggregate_member|context|topical_only|task_shape_only|irrelevant\","
        "\"required_entity_match\":true,\"required_data_match\":true,"
        "\"keep\":true,\"reason\":\"short\",\"chunk_judgements\":[]}]}\n"
        "Relevance must be 0 to 1.\n"
    )
    current_system = (
        "You are a strict retrieval evidence judge for a multimodal QA system. Score whether each "
        "candidate source is useful evidence, input data, or an item that must be inspected to answer "
        "the question. Use source path, file type, metadata, canonical summaries, and snippets. "
        "Return only valid JSON."
    )
    current_user = (
        f"{base_schema}"
        "A candidate is relevant if it directly contains the answer, contains data needed for "
        "calculation/counting/comparison, or is one item in a set that must be inspected or aggregated. "
        "For table, CSV, XLSX, SQL, database, or sheet candidates, mark the source as data_item and keep=true "
        "when the file name, sheet name, columns, sample rows, or SQL schema match the requested calculation, "
        "filter, aggregation, correlation, count, lookup, or multiple-choice computation. "
        "A table does not need to contain the final computed answer to be relevant; it is relevant if it "
        "contains the raw data needed to execute the answer. "
        "Do not mark a source irrelevant just because it does not contain the final answer by itself. "
        "Do not reward generic topical similarity when the source is not useful evidence."
    )
    if strategy == "current":
        return current_system, current_user

    strict_system = (
        "You are a source-level relevance filter for retrieval. First infer the question-specific retrieval intent: "
        "the required entities, files/folders, data fields, and operation needed to answer. Then judge each candidate "
        "source. Keep a source only if it contains direct evidence, raw data for the operation, or an explicit member "
        "of a requested aggregate set. Reject sources that merely share broad topic words or similar task format. "
        "Return only valid JSON."
    )
    strict_user = (
        f"{base_schema}"
        "Use the question to infer required entities and required data. "
        "Set evidence_role=topical_only when a candidate only shares broad concepts with the question, such as generic "
        "sustainability/economy/social-development text without the requested projects or entities. "
        "Set evidence_role=task_shape_only when a candidate only looks like the same task type, such as generic formulas, "
        "multiple-choice text, or calculations, but lacks the requested data/entities. "
        "Set required_entity_match=false if the source does not mention or represent the requested entity, dataset, folder, "
        "project, class, file family, or table. Set required_data_match=false if it lacks the raw data or evidence needed. "
        "For tables/SQL/databases, keep only if columns/schema/sample rows/source path match the requested data. "
        "For multi-file image/folder questions, use aggregate_member when the source is one file that must be inspected. "
        "Drop topical_only, task_shape_only, and irrelevant candidates by setting keep=false."
    )
    if strategy == "strict_source":
        return strict_system, strict_user

    chunk_system = (
        "You are a chunk-level evidence judge followed by a source-level retrieval filter. Judge each snippet first, "
        "then decide whether the whole source should be kept. A source is useful only if at least one snippet or source "
        "summary contains direct answer evidence, raw data for computation, or a requested aggregate member. "
        "Return only valid JSON."
    )
    chunk_user = (
        f"{base_schema}"
        "For each candidate, fill chunk_judgements with objects like "
        "{\"chunk_index\":1,\"label\":\"relevant_evidence|topical_only|task_shape_only|irrelevant\",\"reason\":\"short\"}. "
        "Label topical_only when a chunk only shares broad subject words. Label task_shape_only when it only resembles "
        "the task format, such as formulas or multiple-choice structure, without the requested entity/data. "
        "Keep the source only when at least one chunk or source summary is relevant_evidence, direct answer evidence, "
        "raw data needed for the operation, or an aggregate member explicitly requested by the question. "
        "Drop sources whose best chunks are only topical_only, task_shape_only, or irrelevant."
    )
    return chunk_system, chunk_user


def _load_source_summaries(canonical_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not canonical_dir.exists():
        return summaries
    _merge_image_summaries(summaries, canonical_dir / "images.jsonl")
    _merge_table_summaries(summaries, canonical_dir / "tables.jsonl")
    _merge_text_summaries(summaries, canonical_dir / "texts.jsonl")
    return summaries


def _merge_image_summaries(summaries: dict[str, dict[str, Any]], path: Path) -> None:
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        if not source_path:
            continue
        summary = summaries.setdefault(source_path, {"modalities": []})
        _append_unique(summary["modalities"], "image")
        image_rows = summary.setdefault("images", [])
        image_rows.append(
            {
                "image_id": row.get("image_id"),
                "image_path": row.get("image_path"),
                "role": row.get("role"),
                "locator": row.get("locator"),
                "description": _limit(str(row.get("description") or ""), 2200),
                "metadata": _compact_value(row.get("metadata"), 1200),
            }
        )
        summary["images"] = image_rows[:4]


def _merge_table_summaries(summaries: dict[str, dict[str, Any]], path: Path) -> None:
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        if not source_path:
            continue
        summary = summaries.setdefault(source_path, {"modalities": []})
        _append_unique(summary["modalities"], "table")
        table_rows = summary.setdefault("tables", [])
        table_rows.append(
            {
                "table_id": row.get("table_id"),
                "table_path": row.get("table_path"),
                "role": row.get("role"),
                "parser": row.get("parser"),
                "locator": row.get("locator"),
                "columns": row.get("columns") if isinstance(row.get("columns"), list) else [],
                "sample_rows": _compact_value(row.get("sample_rows"), 1800),
                "tail_sample_rows": _compact_value(row.get("tail_sample_rows"), 1800),
                "metadata_text": _limit(str(row.get("metadata_text") or ""), 1600),
                "llm_description": _limit(str(row.get("llm_description") or ""), 3200),
                "description": _limit(str(row.get("description") or ""), 2600),
                "table_shape": _compact_value(row.get("table_shape"), 1200),
                "metadata": _compact_value(row.get("metadata"), 1000),
            }
        )
        summary["tables"] = table_rows[:6]


def _merge_text_summaries(summaries: dict[str, dict[str, Any]], path: Path) -> None:
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        if not source_path:
            continue
        summary = summaries.setdefault(source_path, {"modalities": []})
        _append_unique(summary["modalities"], "text")
        text_rows = summary.setdefault("texts", [])
        text_rows.append(
            {
                "text_id": row.get("text_id"),
                "role": row.get("role"),
                "parser": row.get("parser"),
                "locator": row.get("locator"),
                "text_sample": _limit(str(row.get("text") or ""), 1400),
                "metadata": _compact_value(row.get("metadata"), 800),
            }
        )
        summary["texts"] = text_rows[:3]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _compact_value(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value or "")
    return _limit(text, limit)


def _limit(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_irrelevant_judgement(
    relevance: float,
    evidence_role: str,
    keep: bool,
    reason: str,
    strategy: str = "current",
    filter_keep_false_zero: bool = False,
) -> bool:
    role = evidence_role.strip().lower()
    if strategy == "current":
        return role == "irrelevant"
    if role in {"irrelevant", "topical_only", "task_shape_only"}:
        return True
    return filter_keep_false_zero and not keep and relevance <= 0.05


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_features(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    keys = list(rows[0])
    bounds = {
        key: (min(row[key] for row in rows), max(row[key] for row in rows))
        for key in keys
    }
    output: list[dict[str, float]] = []
    for row in rows:
        normalized = {}
        for key in keys:
            low, high = bounds[key]
            normalized[key] = 0.0 if high <= low else (row[key] - low) / (high - low)
        output.append(normalized)
    return output
