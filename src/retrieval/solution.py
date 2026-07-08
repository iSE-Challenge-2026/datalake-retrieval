"""End-to-end solution runner that exports retrieved source file paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import csv
import json

from .evaluation import source_universe
from .pipeline import RetrievalPipeline, RetrievalPipelineConfig
from .questions import read_questions


PipelineFactory = Callable[..., RetrievalPipeline]


@dataclass(frozen=True)
class RetrievalSolutionConfig:
    questions: Path
    output: Path
    pipeline: RetrievalPipelineConfig
    source_universe: Path | None = None
    output_format: str = "json"
    include_details: bool = False


def run_retrieval_solution(
    config: RetrievalSolutionConfig,
    *,
    project_root: Path,
    pipeline_factory: PipelineFactory = RetrievalPipeline,
) -> dict[str, Any]:
    """Run retrieval for every question and write local absolute source paths.

    The retrieval index stores source paths relative to the data lake. This
    runner resolves them against the configured local data root so each machine
    exports paths that are valid on that machine.
    """
    resolved = _resolve_config(config, project_root)
    questions = read_questions(resolved.questions)
    source_root = (resolved.source_universe or resolved.pipeline.data_root).resolve()
    universe = source_universe(source_root) or []
    pipeline = pipeline_factory(resolved.pipeline, project_root=project_root, universe=universe)

    rows = []
    for question in questions:
        results = pipeline.retrieve(question.question)
        source_files = [
            _absolute_source_path(result.source_path, source_root)
            for result in results
        ]
        row: dict[str, Any] = {
            "id": question.question_id,
            "question": question.question,
            "source_files": source_files,
        }
        if resolved.include_details:
            row["top_results"] = [
                _result_payload(result, source_root)
                for result in results
            ]
        rows.append(row)

    payload = {
        "summary": {
            "question_count": len(rows),
            "top_k": resolved.pipeline.top_k,
            "source_root": source_root.as_posix(),
        },
        "results": rows,
    }
    write_solution_output(payload, resolved.output, resolved.output_format)
    return payload


def write_solution_output(payload: dict[str, Any], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if output_format == "csv":
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "question", "source_files"])
            writer.writeheader()
            for row in payload["results"]:
                writer.writerow(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "source_files": json.dumps(row["source_files"], ensure_ascii=False),
                    }
                )
        return
    raise ValueError(f"Unsupported solution output format: {output_format}")


def _resolve_config(config: RetrievalSolutionConfig, project_root: Path) -> RetrievalSolutionConfig:
    return RetrievalSolutionConfig(
        questions=_resolve(config.questions, project_root),
        output=_resolve(config.output, project_root),
        pipeline=RetrievalPipelineConfig(
            data_root=_resolve(config.pipeline.data_root, project_root),
            canonical_dir=_resolve(config.pipeline.canonical_dir, project_root) if config.pipeline.canonical_dir else None,
            chunks_path=_resolve(config.pipeline.chunks_path, project_root),
            vector_records=_resolve(config.pipeline.vector_records, project_root) if config.pipeline.vector_records else None,
            vector_chunk_k=config.pipeline.vector_chunk_k,
            vector_aggregation=config.pipeline.vector_aggregation,
            embedding_model=config.pipeline.embedding_model,
            embedding_dimension=config.pipeline.embedding_dimension,
            top_k=config.pipeline.top_k,
            no_special_cases=config.pipeline.no_special_cases,
            retrieval_mode=config.pipeline.retrieval_mode,
            hybrid_lexical_mode=config.pipeline.hybrid_lexical_mode,
            hybrid_candidate_k=config.pipeline.hybrid_candidate_k,
            rrf_k=config.pipeline.rrf_k,
            lexical_weight=config.pipeline.lexical_weight,
            vector_weight=config.pipeline.vector_weight,
            rerank_mode=config.pipeline.rerank_mode,
            rerank_candidate_k=config.pipeline.rerank_candidate_k,
            rerank_model=config.pipeline.rerank_model,
            rerank_cache_dir=_resolve(config.pipeline.rerank_cache_dir, project_root),
            llm_rerank_strategy=config.pipeline.llm_rerank_strategy,
            llm_filter_keep_false_zero=config.pipeline.llm_filter_keep_false_zero,
            expand_mentioned_folders=config.pipeline.expand_mentioned_folders,
            folder_expand_max_files=config.pipeline.folder_expand_max_files,
            app_title=config.pipeline.app_title,
        ),
        source_universe=_resolve(config.source_universe, project_root) if config.source_universe else None,
        output_format=config.output_format,
        include_details=config.include_details,
    )


def _absolute_source_path(source_path: str, source_root: Path) -> str:
    path = Path(source_path)
    if path.is_absolute():
        return str(path.resolve())
    return str((source_root / source_path).resolve())


def _result_payload(result: Any, source_root: Path) -> dict[str, Any]:
    return {
        "source_path": result.source_path,
        "absolute_source_path": _absolute_source_path(result.source_path, source_root),
        "score": round(float(result.score or 0.0), 4),
        "kind": result.source_kind,
        "reasons": list(result.reasons or []),
        "chunks": [
            {
                "score": round(float(getattr(chunk, "score", 0.0) or 0.0), 4),
                "locator": str(getattr(chunk, "locator", "") or ""),
                "text": str(getattr(chunk, "text", "") or "")[:300],
            }
            for chunk in (result.chunks or [])[:3]
        ],
    }


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path
