"""Run source/chunk retrieval for one question."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evaluation import source_universe  # noqa: E402
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig  # noqa: E402
from src.retrieval.presets import PRESETS, apply_preset  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parser().parse_args(argv)
    load_dotenv_file(PROJECT_ROOT)
    config = _pipeline_config_from_args(args)
    resolved_data_root = _resolve(args.source_universe or config.data_root)
    universe = source_universe(resolved_data_root) or []
    pipeline = RetrievalPipeline(config, project_root=PROJECT_ROOT, universe=universe)
    results = pipeline.retrieve(args.question)
    print(json.dumps([_result_payload(result) for result in results], ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve source files and chunks for a question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--chunks-path", type=Path)
    parser.add_argument("--vector-records", type=Path)
    parser.add_argument("--vector-chunk-k", type=int)
    parser.add_argument("--vector-aggregation", choices=["max_mean_top3", "legacy"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--top-k", type=int)
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
    parser.add_argument("--no-special-cases", dest="no_special_cases", action="store_true", default=None)
    parser.add_argument("--special-cases", dest="no_special_cases", action="store_false")
    parser.add_argument("--expand-mentioned-folders", dest="expand_mentioned_folders", action="store_true", default=None)
    parser.add_argument("--no-expand-mentioned-folders", dest="expand_mentioned_folders", action="store_false")
    parser.add_argument("--folder-expand-max-files", type=int)
    parser.add_argument("--source-universe", type=Path)
    return parser


def _pipeline_config_from_args(args: argparse.Namespace) -> RetrievalPipelineConfig:
    overrides = {
        "preset": args.preset,
        "data_root": args.data_root,
        "canonical_dir": args.canonical_dir,
        "chunks_path": args.chunks_path,
        "vector_records": args.vector_records,
        "vector_chunk_k": args.vector_chunk_k,
        "vector_aggregation": args.vector_aggregation,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "top_k": args.top_k,
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
        "no_special_cases": args.no_special_cases,
        "expand_mentioned_folders": args.expand_mentioned_folders,
        "folder_expand_max_files": args.folder_expand_max_files,
    }
    values = apply_preset(overrides, asdict(RetrievalPipelineConfig()))
    values.pop("preset", None)
    return RetrievalPipelineConfig(**values)


def _result_payload(result) -> dict:
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
                "text": chunk.text[:600],
            }
            for chunk in result.chunks[:3]
        ],
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
