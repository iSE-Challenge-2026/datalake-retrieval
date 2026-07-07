"""Evaluate retrieval recall/precision against sample Data Sources."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evaluation import RetrievalEvalConfig, run_retrieval_eval  # noqa: E402
from src.retrieval.pipeline import RetrievalPipelineConfig  # noqa: E402
from src.retrieval.presets import PRESETS, apply_preset  # noqa: E402
from src.retrieval.vector import DEFAULT_VECTOR_AGGREGATION  # noqa: E402
from src.project_paths import OUTPUT_BENCHMARKS_DIR  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parser().parse_args(argv)
    load_dotenv_file(PROJECT_ROOT)

    pipeline_config = _pipeline_config_from_args(args)
    result = run_retrieval_eval(
        RetrievalEvalConfig(
            questions=args.questions,
            pipeline=pipeline_config,
            output=args.output,
            bench_dir=args.bench_dir,
            run_name=args.run_name,
            source_universe=args.source_universe,
            dump_score_details=bool(args.dump_score_details),
            preset=args.preset,
        ),
        project_root=PROJECT_ROOT,
    )
    summary = result["summary"]
    rows = result["results"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    misses = [row for row in rows if row["expected"] and row["recall"] < 1.0]
    if misses:
        print("\nMisses:")
        for row in misses:
            print(f"- {row['id']}: expected={row['expected']} predicted_top5={row['predicted'][:5]}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate source retrieval against sample questions.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--vector-records", type=Path)
    parser.add_argument("--vector-chunk-k", type=int)
    parser.add_argument("--vector-aggregation", choices=["max_mean_top3", "legacy"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-special-cases", dest="no_special_cases", action="store_true", default=None)
    parser.add_argument("--special-cases", dest="no_special_cases", action="store_false")
    parser.add_argument("--chunks-path", type=Path)
    parser.add_argument("--retrieval-mode", choices=["lexical", "bm25", "vector", "hybrid"])
    parser.add_argument("--hybrid-lexical-mode", choices=["lexical", "bm25"])
    parser.add_argument("--hybrid-candidate-k", type=int)
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--lexical-weight", type=float)
    parser.add_argument("--vector-weight", type=float)
    parser.add_argument("--rerank-mode", choices=["none", "heuristic", "llm"])
    parser.add_argument("--rerank-candidate-k", type=int)
    parser.add_argument("--rerank-model")
    parser.add_argument("--rerank-cache-dir", type=Path)
    parser.add_argument("--llm-rerank-strategy", choices=["current", "strict_source", "chunk_judge"])
    parser.add_argument("--llm-filter-keep-false-zero", dest="llm_filter_keep_false_zero", action="store_true", default=None)
    parser.add_argument("--no-llm-filter-keep-false-zero", dest="llm_filter_keep_false_zero", action="store_false")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bench-dir", type=Path, default=OUTPUT_BENCHMARKS_DIR)
    parser.add_argument("--run-name")
    parser.add_argument("--source-universe", type=Path)
    parser.add_argument("--dump-score-details", action="store_true")
    parser.add_argument("--expand-mentioned-folders", dest="expand_mentioned_folders", action="store_true", default=None)
    parser.add_argument("--no-expand-mentioned-folders", dest="expand_mentioned_folders", action="store_false")
    parser.add_argument("--folder-expand-max-files", type=int)
    return parser


def _pipeline_config_from_args(args: argparse.Namespace) -> RetrievalPipelineConfig:
    overrides = {
        "preset": args.preset,
        "data_root": args.data_root,
        "canonical_dir": args.canonical_dir,
        "vector_records": args.vector_records,
        "vector_chunk_k": args.vector_chunk_k,
        "vector_aggregation": args.vector_aggregation,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "top_k": args.top_k,
        "no_special_cases": args.no_special_cases,
        "chunks_path": args.chunks_path,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_lexical_mode": args.hybrid_lexical_mode,
        "hybrid_candidate_k": args.hybrid_candidate_k,
        "rrf_k": args.rrf_k,
        "lexical_weight": args.lexical_weight,
        "vector_weight": args.vector_weight,
        "rerank_mode": args.rerank_mode,
        "rerank_candidate_k": args.rerank_candidate_k,
        "rerank_model": args.rerank_model,
        "rerank_cache_dir": args.rerank_cache_dir,
        "llm_rerank_strategy": args.llm_rerank_strategy,
        "llm_filter_keep_false_zero": args.llm_filter_keep_false_zero,
        "expand_mentioned_folders": args.expand_mentioned_folders,
        "folder_expand_max_files": args.folder_expand_max_files,
    }
    values = apply_preset(overrides, asdict(RetrievalPipelineConfig()))
    values.pop("preset", None)
    if values.get("vector_aggregation") is None:
        values["vector_aggregation"] = DEFAULT_VECTOR_AGGREGATION
    return RetrievalPipelineConfig(**values)


if __name__ == "__main__":
    main()
