"""Shared retrieval pipeline used by CLI wrappers and evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from src.model_clients.embeddings import OpenRouterEmbeddingProvider
from src.project_paths import CANONICAL_CHUNKS_PATH, OPENROUTER_RERANK_DIR, RAW_DATA_LAKE_DIR

from .bm25 import BM25Retriever
from .engine import DocumentRetriever, RetrievalConfig
from .expansion import ExpandedChunkHit, expand_mentioned_folders, load_chunk_index
from .hybrid import reciprocal_rank_fusion
from .rerank import OpenRouterLLMReranker, heuristic_rerank
from .vector import DEFAULT_VECTOR_AGGREGATION, DEFAULT_VECTOR_CHUNK_K, VectorRetriever


@dataclass
class RetrievalPipelineConfig:
    data_root: Path = RAW_DATA_LAKE_DIR
    canonical_dir: Path | None = None
    chunks_path: Path = CANONICAL_CHUNKS_PATH
    vector_records: Path | None = None
    vector_chunk_k: int = DEFAULT_VECTOR_CHUNK_K
    vector_aggregation: str = DEFAULT_VECTOR_AGGREGATION
    embedding_model: str | None = None
    embedding_dimension: int = 1536
    top_k: int = 20
    no_special_cases: bool = False
    retrieval_mode: str | None = None
    hybrid_lexical_mode: str = "bm25"
    hybrid_candidate_k: int = 50
    rrf_k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    rerank_mode: str = "none"
    rerank_candidate_k: int = 50
    rerank_model: str | None = None
    rerank_cache_dir: Path = OPENROUTER_RERANK_DIR
    llm_rerank_strategy: str = "current"
    llm_filter_keep_false_zero: bool = False
    expand_mentioned_folders: bool = False
    folder_expand_max_files: int = 80
    app_title: str = "Data-Lake Retrieval Eval"


class RetrievalPipeline:
    def __init__(
        self,
        config: RetrievalPipelineConfig,
        *,
        project_root: Path,
        universe: list[str] | None = None,
    ) -> None:
        self.config = _resolve_config(config, project_root)
        self.project_root = project_root
        self.mode = self.config.retrieval_mode or ("vector" if self.config.vector_records else "lexical")
        self.universe = universe or []
        self.provider = None
        self.vector_retriever = None
        self.lexical_retriever = None
        self.bm25_retriever = None
        self.llm_reranker = None
        self.chunk_index: dict[str, list[ExpandedChunkHit]] = {}
        self._build()

    def retrieve(self, question: str) -> list[Any]:
        retrieval_top_k = self.config.rerank_candidate_k if self.config.rerank_mode != "none" else self.config.top_k
        if self.mode == "vector":
            query_embedding = self.provider.embed_batch([question])[0]
            results = self.vector_retriever.retrieve(
                query_embedding,
                top_k=retrieval_top_k,
                chunk_k=self.config.vector_chunk_k,
            )
        elif self.mode == "hybrid":
            query_embedding = self.provider.embed_batch([question])[0]
            if self.config.hybrid_lexical_mode == "bm25":
                lexical_results = self.bm25_retriever.retrieve(question, top_k=self.config.hybrid_candidate_k)
            else:
                lexical_results = self.lexical_retriever.retrieve(question, top_k=self.config.hybrid_candidate_k)
            vector_results = self.vector_retriever.retrieve(
                query_embedding,
                top_k=self.config.hybrid_candidate_k,
                chunk_k=self.config.vector_chunk_k,
            )
            results = reciprocal_rank_fusion(
                lexical_results,
                vector_results,
                top_k=retrieval_top_k,
                rrf_k=self.config.rrf_k,
                lexical_weight=self.config.lexical_weight,
                vector_weight=self.config.vector_weight,
            )
        elif self.mode == "bm25":
            results = self.bm25_retriever.retrieve(question, top_k=retrieval_top_k)
        else:
            results = self.lexical_retriever.retrieve(question, top_k=retrieval_top_k)

        if self.config.expand_mentioned_folders:
            results = expand_mentioned_folders(
                question,
                results,
                self.universe,
                self.chunk_index,
                max_files=self.config.folder_expand_max_files,
            )
        if self.config.rerank_mode == "heuristic":
            return heuristic_rerank(question, results, top_k=self.config.top_k)
        if self.config.rerank_mode == "llm":
            return self.llm_reranker.rerank(question, results, top_k=self.config.top_k)
        return results

    def _build(self) -> None:
        if self.mode in {"vector", "hybrid"} and not self.config.vector_records:
            raise ValueError("--vector-records is required for vector or hybrid retrieval.")
        if self.mode == "hybrid" and self.config.hybrid_lexical_mode == "lexical" and not self.config.canonical_dir:
            raise ValueError("--canonical-dir is required for hybrid lexical retrieval.")

        if self.mode in {"vector", "hybrid"}:
            model = self.config.embedding_model or os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
            self.provider = OpenRouterEmbeddingProvider(
                model=model,
                dimension=self.config.embedding_dimension,
                api_key_env="OPENROUTER_API_KEY",
                app_title=self.config.app_title,
            )
            self.vector_retriever = VectorRetriever.from_jsonl(
                self.config.vector_records,
                aggregation=self.config.vector_aggregation,
            )
        if self.mode == "bm25" or (self.mode == "hybrid" and self.config.hybrid_lexical_mode == "bm25"):
            self.bm25_retriever = BM25Retriever.from_jsonl(self.config.chunks_path)
        if self.mode == "lexical" or (self.mode == "hybrid" and self.config.hybrid_lexical_mode == "lexical"):
            config = RetrievalConfig(
                top_k=self.config.hybrid_candidate_k if self.mode == "hybrid" else self.config.top_k,
                enable_special_cases=not self.config.no_special_cases,
            )
            if self.config.canonical_dir:
                self.lexical_retriever = DocumentRetriever.from_canonical(self.config.canonical_dir, config=config)
            else:
                self.lexical_retriever = DocumentRetriever.from_data_lake(
                    self.config.data_root,
                    config=config,
                )
        if self.config.rerank_mode == "llm":
            self.llm_reranker = OpenRouterLLMReranker(
                model=self.config.rerank_model,
                cache_dir=self.config.rerank_cache_dir,
                canonical_dir=self.config.canonical_dir,
                strategy=self.config.llm_rerank_strategy,
                filter_keep_false_zero=self.config.llm_filter_keep_false_zero,
            )
        if self.config.expand_mentioned_folders:
            self.chunk_index = load_chunk_index(self.config.chunks_path)


def _resolve_config(config: RetrievalPipelineConfig, project_root: Path) -> RetrievalPipelineConfig:
    return RetrievalPipelineConfig(
        data_root=_resolve(config.data_root, project_root),
        canonical_dir=_resolve(config.canonical_dir, project_root) if config.canonical_dir else None,
        chunks_path=_resolve(config.chunks_path, project_root),
        vector_records=_resolve(config.vector_records, project_root) if config.vector_records else None,
        vector_chunk_k=config.vector_chunk_k,
        vector_aggregation=config.vector_aggregation,
        embedding_model=config.embedding_model,
        embedding_dimension=config.embedding_dimension,
        top_k=config.top_k,
        no_special_cases=config.no_special_cases,
        retrieval_mode=config.retrieval_mode,
        hybrid_lexical_mode=config.hybrid_lexical_mode,
        hybrid_candidate_k=config.hybrid_candidate_k,
        rrf_k=config.rrf_k,
        lexical_weight=config.lexical_weight,
        vector_weight=config.vector_weight,
        rerank_mode=config.rerank_mode,
        rerank_candidate_k=config.rerank_candidate_k,
        rerank_model=config.rerank_model,
        rerank_cache_dir=_resolve(config.rerank_cache_dir, project_root),
        llm_rerank_strategy=config.llm_rerank_strategy,
        llm_filter_keep_false_zero=config.llm_filter_keep_false_zero,
        expand_mentioned_folders=config.expand_mentioned_folders,
        folder_expand_max_files=config.folder_expand_max_files,
        app_title=config.app_title,
    )


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path
