"""Text normalization helpers for retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable


STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "a",
    "an",
    "is",
    "are",
    "be",
    "what",
    "which",
    "how",
    "many",
    "with",
    "by",
    "from",
    "where",
    "did",
    "does",
    "do",
    "là",
    "của",
    "cho",
    "tôi",
    "hãy",
    "trong",
    "và",
    "các",
    "có",
    "nào",
    "gì",
    "bao",
    "nhiêu",
    "được",
    "với",
    "theo",
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokenize(text: object, *, expand: bool = True) -> list[str]:
    raw = unicodedata.normalize("NFKC", str(text or "")).lower()
    folded = strip_accents(raw)
    tokens: list[str] = []
    for source in (raw, folded):
        tokens.extend(re.findall(r"[a-z0-9]+", source, flags=re.IGNORECASE))
        tokens.extend(char for char in source if "\u4e00" <= char <= "\u9fff")

    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS and (len(token) >= 2 or "\u4e00" <= token <= "\u9fff")
    ]
    if expand:
        return filtered + sorted(expand_synonyms(filtered))
    return filtered


def expand_synonyms(tokens: Iterable[str]) -> set[str]:
    token_set = set(tokens)
    expanded: set[str] = set()
    if {"hoc", "bong"} <= token_set:
        expanded.update({"scholarship"})
    if {"hang", "khong"} <= token_set:
        expanded.update({"airline", "airlines", "aviation", "air"})
    if token_set & {"anh", "hinh"}:
        expanded.update({"image", "photo", "picture", "jpg", "png"})
    if token_set & {"diem", "lop", "toan"}:
        expanded.update({"grade", "grades", "class", "math"})
    if {"kinh", "te", "chinh", "tri"} <= token_set or "xhcn" in token_set:
        expanded.update({"ktct", "political", "economy", "ktttrxhcn"})
    if {"thanh", "vien"} <= token_set or "nhom" in token_set:
        expanded.update({"member", "members", "team", "ise"})
    if {"am", "thanh"} <= token_set or token_set & {"audio", "m4a", "mp3", "workshop"}:
        expanded.update({"audio", "m4a", "meeting", "workshop"})
    if {"du", "an"} <= token_set:
        expanded.update({"project"})
    if {"thu", "vien"} <= token_set:
        expanded.update({"library", "smart"})
    if {"lam", "sach"} <= token_set or {"song", "minh", "hoa"} <= token_set:
        expanded.update({"river", "cleanup", "community"})
    if "novacare" in token_set:
        expanded.update({"customer", "support", "startup", "ai"})
    return expanded


def quoted_spans(text: str) -> list[str]:
    spans: list[str] = []
    for groups in re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', text or ""):
        spans.extend(group.strip() for group in groups if group.strip())
    return spans


def token_counts(text: object) -> Counter[str]:
    return Counter(tokenize(text))


def weighted_overlap(query_counts: Counter[str], doc_counts: Counter[str], idf: dict[str, float]) -> float:
    if not query_counts or not doc_counts:
        return 0.0
    score = 0.0
    for token, q_count in query_counts.items():
        d_count = doc_counts.get(token, 0)
        if d_count:
            score += min(q_count, d_count) * idf.get(token, 1.0)
    query_norm = math.sqrt(sum((count * idf.get(token, 1.0)) ** 2 for token, count in query_counts.items()))
    doc_norm = math.sqrt(sum((count * idf.get(token, 1.0)) ** 2 for token, count in doc_counts.items()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return score / (query_norm * doc_norm)


def compact(text: str, limit: int = 1000) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
