"""Batch retrieval evaluation against sample question source annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .benchmarks import write_bench_run
from .engine import evaluate_results
from .pipeline import RetrievalPipeline, RetrievalPipelineConfig
from .questions import read_questions


@dataclass
class RetrievalEvalConfig:
    questions: Path
    pipeline: RetrievalPipelineConfig
    output: Path | None = None
    bench_dir: Path | None = None
    run_name: str | None = None
    source_universe: Path | None = None
    dump_score_details: bool = False
    preset: str | None = None


def run_retrieval_eval(config: RetrievalEvalConfig, *, project_root: Path) -> dict[str, Any]:
    resolved = _resolve_eval_config(config, project_root)
    questions = read_questions(resolved.questions)
    universe = source_universe(resolved.source_universe or resolved.pipeline.data_root)
    pipeline = RetrievalPipeline(resolved.pipeline, project_root=project_root, universe=universe)
    rows = []
    for question in questions:
        results = pipeline.retrieve(question.question)
        predicted = [result.source_path for result in results]
        metrics = evaluate_results(predicted, question.data_sources, universe=universe)
        diagnostics = diagnostics_for(predicted, question.data_sources, metrics, universe) if resolved.dump_score_details else {}
        rows.append(
            {
                "id": question.question_id,
                "question": question.question,
                "expected": question.data_sources,
                "predicted": predicted,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "matched_sources": metrics["matched_sources"],
                "expanded_expected": metrics.get("expanded_expected_sources"),
                "pattern_recall": metrics.get("pattern_recall"),
                **diagnostics,
                "top_results": [_result_payload(result) for result in results[:10]],
            }
        )

    summary = summarize_rows(rows)
    payload = {"summary": summary, "results": rows}
    if resolved.output:
        resolved.output.parent.mkdir(parents=True, exist_ok=True)
        resolved.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir = None
    if resolved.bench_dir:
        run_dir = write_bench_run(
            bench_dir=resolved.bench_dir,
            run_name=resolved.run_name,
            payload=payload,
            summary=summary,
            config=resolved.pipeline,
            questions_path=resolved.questions,
            data_root=resolved.pipeline.data_root,
            dump_score_details=resolved.dump_score_details,
            preset=resolved.preset,
        )
    return {"summary": summary, "results": rows, "run_dir": run_dir}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_sources = [row for row in rows if row["expected"]]
    no_sources = [row for row in rows if not row["expected"]]
    perfect = [row for row in with_sources if row["recall"] >= 1.0]
    return {
        "question_count": len(rows),
        "questions_with_sources": len(with_sources),
        "recall_at_k_macro": round(sum(row["recall"] for row in with_sources) / len(with_sources), 4) if with_sources else 0.0,
        "precision_at_k_macro": round(sum(row["precision"] for row in with_sources) / len(with_sources), 4) if with_sources else 0.0,
        "perfect_recall_count": len(perfect),
        "perfect_recall_rate": round(len(perfect) / len(with_sources), 4) if with_sources else 0.0,
        "no_source_questions": len(no_sources),
        "pattern_recall_at_k_macro": round(
            sum(row.get("pattern_recall", row["recall"]) for row in with_sources) / len(with_sources),
            4,
        ) if with_sources else 0.0,
    }


def source_universe(root: Path) -> list[str] | None:
    if not root.exists() or not root.is_dir():
        return None
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]


def diagnostics_for(predicted: list[str], expected: list[str], metrics: dict[str, Any], universe: list[str] | None) -> dict[str, Any]:
    expanded_expected = metrics.get("expanded_expected_sources") or expected
    matched_sources = set(metrics.get("matched_sources") or [])
    missed = [source for source in expanded_expected if source not in matched_sources]
    false_positives = [
        path
        for path in predicted
        if expected and evaluate_results([path], expected, universe=universe).get("precision", 0.0) <= 0.0
    ]
    return {
        "missed_expected": missed,
        "false_positives": false_positives,
    }


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "source_path": result.source_path,
        "score": round(result.score, 4),
        "kind": result.source_kind,
        "reasons": result.reasons,
        "rerank_features": getattr(result, "rerank_features", None),
        "score_components": getattr(result, "score_components", None),
        "chunks": [
            {
                "score": round(chunk.score, 4),
                "locator": chunk.locator,
                "text": chunk.text[:300],
            }
            for chunk in result.chunks[:2]
        ],
    }


def _resolve_eval_config(config: RetrievalEvalConfig, project_root: Path) -> RetrievalEvalConfig:
    pipeline = config.pipeline
    return RetrievalEvalConfig(
        questions=_resolve(config.questions, project_root),
        pipeline=RetrievalPipelineConfig(
            data_root=_resolve(pipeline.data_root, project_root),
            canonical_dir=_resolve(pipeline.canonical_dir, project_root) if pipeline.canonical_dir else None,
            chunks_path=_resolve(pipeline.chunks_path, project_root),
            vector_records=_resolve(pipeline.vector_records, project_root) if pipeline.vector_records else None,
            vector_chunk_k=pipeline.vector_chunk_k,
            vector_aggregation=pipeline.vector_aggregation,
            embedding_model=pipeline.embedding_model,
            embedding_dimension=pipeline.embedding_dimension,
            top_k=pipeline.top_k,
            no_special_cases=pipeline.no_special_cases,
            retrieval_mode=pipeline.retrieval_mode,
            hybrid_lexical_mode=pipeline.hybrid_lexical_mode,
            hybrid_candidate_k=pipeline.hybrid_candidate_k,
            rrf_k=pipeline.rrf_k,
            lexical_weight=pipeline.lexical_weight,
            vector_weight=pipeline.vector_weight,
            rerank_mode=pipeline.rerank_mode,
            rerank_candidate_k=pipeline.rerank_candidate_k,
            rerank_model=pipeline.rerank_model,
            rerank_cache_dir=_resolve(pipeline.rerank_cache_dir, project_root),
            llm_rerank_strategy=pipeline.llm_rerank_strategy,
            llm_filter_keep_false_zero=pipeline.llm_filter_keep_false_zero,
            expand_mentioned_folders=pipeline.expand_mentioned_folders,
            folder_expand_max_files=pipeline.folder_expand_max_files,
            app_title=pipeline.app_title,
        ),
        output=_resolve(config.output, project_root) if config.output else None,
        bench_dir=_resolve(config.bench_dir, project_root) if config.bench_dir else None,
        run_name=config.run_name,
        source_universe=_resolve(config.source_universe, project_root) if config.source_universe else None,
        dump_score_details=config.dump_score_details,
        preset=config.preset,
    )


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path
