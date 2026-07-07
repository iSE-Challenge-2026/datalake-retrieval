"""Benchmark artifact writing for retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from .pipeline import RetrievalPipelineConfig


def write_bench_run(
    *,
    bench_dir: Path,
    run_name: str | None,
    payload: dict[str, Any],
    summary: dict[str, Any],
    config: RetrievalPipelineConfig,
    questions_path: Path,
    data_root: Path,
    dump_score_details: bool,
    preset: str | None = None,
) -> Path:
    run_name = run_name or default_run_name(config)
    run_dir = bench_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = benchmark_manifest(
        run_name=run_name,
        config=config,
        questions_path=questions_path,
        data_root=data_root,
        dump_score_details=dump_score_details,
        preset=preset,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (bench_dir / "latest.json").write_text(
        json.dumps({"run_dir": run_dir.as_posix(), **manifest, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def read_bench_run(run_dir: Path) -> dict[str, Any]:
    return {
        "manifest": _read_json(run_dir / "manifest.json"),
        "summary": _read_json(run_dir / "summary.json"),
        "results": _read_json(run_dir / "results.json"),
    }


def benchmark_manifest(
    *,
    run_name: str,
    config: RetrievalPipelineConfig,
    questions_path: Path,
    data_root: Path,
    dump_score_details: bool,
    preset: str | None = None,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "created_at_unix": int(time.time()),
        "preset": preset,
        "mode": config.retrieval_mode or ("vector" if config.vector_records else "lexical"),
        "hybrid_lexical_mode": config.hybrid_lexical_mode,
        "chunks_path": config.chunks_path.as_posix() if config.chunks_path else None,
        "questions": questions_path.as_posix(),
        "top_k": config.top_k,
        "no_special_cases": bool(config.no_special_cases),
        "data_root": data_root.as_posix() if data_root else None,
        "canonical_dir": config.canonical_dir.as_posix() if config.canonical_dir else None,
        "vector_records": config.vector_records.as_posix() if config.vector_records else None,
        "vector_chunk_k": config.vector_chunk_k,
        "vector_aggregation": config.vector_aggregation,
        "dump_score_details": bool(dump_score_details),
        "embedding_model": config.embedding_model,
        "embedding_dimension": config.embedding_dimension,
        "hybrid_candidate_k": config.hybrid_candidate_k,
        "rrf_k": config.rrf_k,
        "lexical_weight": config.lexical_weight,
        "vector_weight": config.vector_weight,
        "rerank_mode": config.rerank_mode,
        "rerank_candidate_k": config.rerank_candidate_k,
        "rerank_model": config.rerank_model,
        "rerank_cache_dir": config.rerank_cache_dir.as_posix() if config.rerank_cache_dir else None,
        "llm_rerank_strategy": config.llm_rerank_strategy,
        "llm_filter_keep_false_zero": bool(config.llm_filter_keep_false_zero),
        "expand_mentioned_folders": bool(config.expand_mentioned_folders),
        "folder_expand_max_files": config.folder_expand_max_files,
    }


def default_run_name(config: RetrievalPipelineConfig) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mode = config.retrieval_mode or ("vector" if config.vector_records else "lexical")
    rules = "no_rules" if config.no_special_cases else "rules"
    return f"{timestamp}_{mode}_top{config.top_k}_{rules}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
