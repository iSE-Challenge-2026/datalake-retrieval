"""Shared artifact locations for the Data-Lake retrieval pipeline."""

from __future__ import annotations

from pathlib import Path


RAW_DATA_LAKE_DIR = Path("data/raw/Data-Lake")
PROCESSED_DATA_LAKE_DIR = Path("data/processed/Data-Lake")

CANONICAL_DIR = PROCESSED_DATA_LAKE_DIR / "canonical"
CANONICAL_CHUNKS_PATH = CANONICAL_DIR / "chunks.jsonl"

MODEL_RAW_DIR = PROCESSED_DATA_LAKE_DIR / "model_raw"
OPENROUTER_EMBEDDINGS_DIR = MODEL_RAW_DIR / "openrouter_embeddings"
OPENROUTER_VECTOR_RECORDS_PATH = OPENROUTER_EMBEDDINGS_DIR / "vector_records.jsonl"
OPENROUTER_ASR_CACHE_DIR = MODEL_RAW_DIR / "openrouter_asr" / "asr_cache"
OPENROUTER_IMAGE_ENRICHMENT_DIR = MODEL_RAW_DIR / "openrouter_image_enrichment"
OPENROUTER_TABLE_ENRICHMENT_DIR = MODEL_RAW_DIR / "openrouter_table_enrichment"
DATALAB_PARSING_DIR = MODEL_RAW_DIR / "datalab_parsing"
RAW_IMAGE_LIFT_OUTPUT_DIR = DATALAB_PARSING_DIR

OUTPUT_DIR = Path("data/output/Data-Lake")
OUTPUT_BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"
OUTPUT_RERANK_CACHE_DIR = OUTPUT_DIR / "rerank_cache"
OPENROUTER_RERANK_DIR = OUTPUT_RERANK_CACHE_DIR / "openrouter_rerank"
OPENROUTER_RERANK_FLASH_DIR = OUTPUT_RERANK_CACHE_DIR / "openrouter_rerank_gemini_2_5_flash"
OPENROUTER_RERANK_FLASH_FOLDER_EXPAND_DIR = OUTPUT_RERANK_CACHE_DIR / "openrouter_rerank_gemini_2_5_flash_folder_expand"
