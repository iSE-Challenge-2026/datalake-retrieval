"""Named retrieval benchmark presets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.project_paths import (
    CANONICAL_CHUNKS_PATH,
    CANONICAL_DIR,
    OPENROUTER_RERANK_FLASH_DIR,
    OPENROUTER_RERANK_FLASH_FOLDER_EXPAND_DIR,
    OPENROUTER_VECTOR_RECORDS_PATH,
)


PRESETS: dict[str, dict[str, Any]] = {
    "vector_flash": {
        "canonical_dir": CANONICAL_DIR,
        "chunks_path": CANONICAL_CHUNKS_PATH,
        "vector_records": OPENROUTER_VECTOR_RECORDS_PATH,
        "retrieval_mode": "vector",
        "vector_chunk_k": 80,
        "vector_aggregation": "legacy",
        "rerank_mode": "llm",
        "rerank_candidate_k": 20,
        "top_k": 20,
        "llm_rerank_strategy": "current",
        "rerank_model": "google/gemini-2.5-flash",
        "rerank_cache_dir": OPENROUTER_RERANK_FLASH_DIR,
        "no_special_cases": True,
        "expand_mentioned_folders": False,
    },
    "folder_expand_flash": {
        "canonical_dir": CANONICAL_DIR,
        "chunks_path": CANONICAL_CHUNKS_PATH,
        "vector_records": OPENROUTER_VECTOR_RECORDS_PATH,
        "retrieval_mode": "vector",
        "vector_chunk_k": 80,
        "vector_aggregation": "legacy",
        "rerank_mode": "llm",
        "rerank_candidate_k": 20,
        "top_k": 20,
        "llm_rerank_strategy": "current",
        "rerank_model": "google/gemini-2.5-flash",
        "rerank_cache_dir": OPENROUTER_RERANK_FLASH_FOLDER_EXPAND_DIR,
        "no_special_cases": True,
        "expand_mentioned_folders": True,
        "folder_expand_max_files": 80,
    },
}


def apply_preset(overrides: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Merge defaults, optional named preset, and explicit non-None overrides."""
    preset_name = overrides.get("preset")
    values = dict(defaults)
    if preset_name:
        if preset_name not in PRESETS:
            raise ValueError(f"Unsupported retrieval preset: {preset_name}")
        values.update(PRESETS[preset_name])
    for key, value in overrides.items():
        if key == "preset":
            continue
        if value is not None:
            values[key] = value
    values["preset"] = preset_name
    return values
