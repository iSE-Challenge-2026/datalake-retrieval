from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import tempfile
import unittest

from src.retrieval.rerank import OpenRouterLLMReranker, _load_source_summaries


@dataclass
class FakeResult:
    source_path: str
    score: float
    source_kind: str = "table"
    reasons: list[str] = field(default_factory=list)
    chunks: list = field(default_factory=list)


class RerankTableTests(unittest.TestCase):
    def test_table_summary_includes_llm_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "images.jsonl", [])
            _write_jsonl(root / "texts.jsonl", [])
            _write_jsonl(
                root / "tables.jsonl",
                [
                    {
                        "table_id": "t1",
                        "source_path": "Credit.csv",
                        "table_path": "Credit.csv",
                        "columns": ["Limit", "Balance"],
                        "sample_rows": [{"Limit": "3606", "Balance": "333"}],
                        "tail_sample_rows": [{"Limit": "4897", "Balance": "331"}],
                        "llm_description": "Credit table with Limit and Balance columns for correlation.",
                    }
                ],
            )

            summaries = _load_source_summaries(root)

        table = summaries["Credit.csv"]["tables"][0]
        self.assertIn("Limit", table["columns"])
        self.assertIn("Credit table", table["llm_description"])
        self.assertIn("4897", table["tail_sample_rows"])

    def test_apply_scores_filters_irrelevant_candidates(self) -> None:
        reranker = OpenRouterLLMReranker(cache_dir="unused")
        results = [
            FakeResult("keep.csv", 1.0, reasons=["vector"]),
            FakeResult("drop.csv", 0.9, reasons=["vector"]),
            FakeResult("unknown.csv", 0.8, reasons=["vector"]),
        ]
        scores = {
            "keep.csv": {"relevance": 1.0, "evidence_role": "data_item", "keep": True, "reason": "needed"},
            "drop.csv": {"relevance": 0.0, "evidence_role": "irrelevant", "keep": False, "reason": "wrong table"},
            "unknown.csv": {"relevance": 0.0, "evidence_role": "", "keep": False, "reason": "not enough context"},
        }

        output = reranker._apply_scores(results, scores, top_k=3)

        self.assertEqual([item.source_path for item in output], ["keep.csv", "unknown.csv"])

    def test_current_strategy_keeps_topical_or_task_shape_labels(self) -> None:
        reranker = OpenRouterLLMReranker(cache_dir="unused", strategy="current")
        results = [
            FakeResult("topical.pdf", 1.0, source_kind="text", reasons=["vector"]),
            FakeResult("shape.pdf", 0.9, source_kind="text", reasons=["vector"]),
        ]
        scores = {
            "topical.pdf": {
                "relevance": 0.2,
                "evidence_role": "topical_only",
                "keep": False,
                "reason": "same broad topic",
            },
            "shape.pdf": {
                "relevance": 0.2,
                "evidence_role": "task_shape_only",
                "keep": False,
                "reason": "same task format",
            },
        }

        output = reranker._apply_scores(results, scores, top_k=2)

        self.assertEqual([item.source_path for item in output], ["topical.pdf", "shape.pdf"])

    def test_strict_strategy_filters_topical_or_task_shape_labels(self) -> None:
        reranker = OpenRouterLLMReranker(cache_dir="unused", strategy="strict_source")
        results = [
            FakeResult("needed.csv", 1.0, reasons=["vector"]),
            FakeResult("topical.pdf", 0.9, source_kind="text", reasons=["vector"]),
            FakeResult("shape.pdf", 0.8, source_kind="text", reasons=["vector"]),
        ]
        scores = {
            "needed.csv": {
                "relevance": 1.0,
                "evidence_role": "data_item",
                "keep": True,
                "reason": "contains requested raw data",
            },
            "topical.pdf": {
                "relevance": 0.2,
                "evidence_role": "topical_only",
                "keep": False,
                "reason": "same broad topic",
            },
            "shape.pdf": {
                "relevance": 0.2,
                "evidence_role": "task_shape_only",
                "keep": False,
                "reason": "same task format",
            },
        }

        output = reranker._apply_scores(results, scores, top_k=3)

        self.assertEqual([item.source_path for item in output], ["needed.csv"])

    def test_strict_strategy_keeps_explicit_keep_false_with_zero_relevance_by_default(self) -> None:
        reranker = OpenRouterLLMReranker(cache_dir="unused", strategy="chunk_judge")
        results = [
            FakeResult("needed.csv", 1.0, reasons=["vector"]),
            FakeResult("drop.pdf", 0.9, source_kind="text", reasons=["vector"]),
            FakeResult("uncertain.pdf", 0.8, source_kind="text", reasons=["vector"]),
        ]
        scores = {
            "needed.csv": {
                "relevance": 1.0,
                "evidence_role": "data_item",
                "keep": True,
                "reason": "contains requested raw data",
            },
            "drop.pdf": {
                "relevance": 0.0,
                "evidence_role": "",
                "keep": False,
                "reason": "",
            },
            "uncertain.pdf": {
                "relevance": 0.1,
                "evidence_role": "",
                "keep": False,
                "reason": "unclear",
            },
        }

        output = reranker._apply_scores(results, scores, top_k=3)

        self.assertEqual([item.source_path for item in output], ["needed.csv", "uncertain.pdf", "drop.pdf"])

    def test_strict_strategy_can_filter_explicit_keep_false_with_zero_relevance(self) -> None:
        reranker = OpenRouterLLMReranker(
            cache_dir="unused",
            strategy="chunk_judge",
            filter_keep_false_zero=True,
        )
        results = [
            FakeResult("needed.csv", 1.0, reasons=["vector"]),
            FakeResult("drop.pdf", 0.9, source_kind="text", reasons=["vector"]),
            FakeResult("uncertain.pdf", 0.8, source_kind="text", reasons=["vector"]),
        ]
        scores = {
            "needed.csv": {
                "relevance": 1.0,
                "evidence_role": "data_item",
                "keep": True,
                "reason": "contains requested raw data",
            },
            "drop.pdf": {
                "relevance": 0.0,
                "evidence_role": "",
                "keep": False,
                "reason": "",
            },
            "uncertain.pdf": {
                "relevance": 0.1,
                "evidence_role": "",
                "keep": False,
                "reason": "unclear",
            },
        }

        output = reranker._apply_scores(results, scores, top_k=3)

        self.assertEqual([item.source_path for item in output], ["needed.csv", "uncertain.pdf"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
