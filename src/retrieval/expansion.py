"""Candidate expansion helpers for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re


@dataclass
class ExpandedChunkHit:
    chunk_id: str
    source_path: str
    source_kind: str
    source_role: str
    locator: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpandedRetrievalResult:
    source_path: str
    score: float
    source_kind: str
    reasons: list[str] = field(default_factory=lambda: ["folder_expand"])
    chunks: list[ExpandedChunkHit] = field(default_factory=list)


def load_chunk_index(chunks_path: Path) -> dict[str, list[ExpandedChunkHit]]:
    if not chunks_path.exists():
        return {}
    by_source: dict[str, list[ExpandedChunkHit]] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_path = str(row.get("source_path") or "")
            if not source_path:
                continue
            hit = ExpandedChunkHit(
                chunk_id=str(row.get("chunk_id") or ""),
                source_path=source_path,
                source_kind=str(row.get("source_kind") or "file"),
                source_role=str(row.get("source_role") or ""),
                locator=str(row.get("locator") or ""),
                score=0.0,
                text=str(row.get("text") or "")[:900],
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            )
            by_source.setdefault(source_path, []).append(hit)
    return by_source


def expand_mentioned_folders(
    question: str,
    results: list[Any],
    universe: list[str],
    chunk_index: dict[str, list[ExpandedChunkHit]],
    max_files: int,
) -> list[Any]:
    folders = mentioned_folders(question, universe)
    if not folders:
        return results
    by_source = {result.source_path: result for result in results}
    base_score = min((float(result.score or 0.0) for result in results), default=0.0) - 1e-6
    for folder in folders:
        matches = [path for path in universe if path == folder or path.startswith(f"{folder}/")]
        if len(matches) > max_files:
            continue
        for path in matches:
            if path in by_source:
                by_source[path].reasons = sorted(set(list(by_source[path].reasons or []) + ["folder_expand_match"]))
                continue
            chunks = [chunk for chunk in chunk_index.get(path, [])[:3]]
            source_kind = chunks[0].source_kind if chunks else kind_from_path(path)
            by_source[path] = ExpandedRetrievalResult(
                source_path=path,
                score=base_score,
                source_kind=source_kind,
                reasons=["folder_expand"],
                chunks=chunks,
            )
    return list(by_source.values())


def mentioned_folders(question: str, universe: list[str]) -> list[str]:
    spans = re.findall(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', question)
    candidates = [next(part for part in match if part).strip().strip("/\\") for match in spans if any(match)]
    folders: list[str] = []
    universe_folders = {path.rsplit("/", 1)[0] for path in universe if "/" in path}
    for candidate in candidates:
        normalized = candidate.replace("\\", "/").strip("/")
        if not normalized:
            continue
        if normalized in universe_folders:
            folders.append(normalized)
            continue
        for folder in sorted(universe_folders):
            if folder.lower().endswith(f"/{normalized.lower()}") or folder.lower() == normalized.lower():
                folders.append(folder)
    output: list[str] = []
    seen = set()
    for folder in folders:
        if folder not in seen:
            seen.add(folder)
            output.append(folder)
    return output


def kind_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix in {".csv", ".xlsx", ".xls", ".sql", ".db", ".sqlite", ".json"}:
        return "table"
    if suffix in {".m4a", ".mp3", ".wav", ".flac"}:
        return "audio"
    return "text"
