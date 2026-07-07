from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.engine import evaluate_results


class RetrievalMetricsBM25Tests(unittest.TestCase):
    def test_expanded_wildcard_recall_is_flat_file_recall(self) -> None:
        metrics = evaluate_results(
            predicted=["number_image/2.png", "number_image/3.png"],
            expected=["number_image/*"],
            universe=["number_image/1.png", "number_image/2.png", "number_image/3.png", "other/file.txt"],
        )

        self.assertEqual(metrics["expected_count"], 3)
        self.assertEqual(metrics["matched_count"], 2)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertEqual(metrics["pattern_recall"], 1.0)

    def test_bm25_retrieves_exact_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "a",
                    "source_path": "a.txt",
                    "source_kind": "text",
                    "source_role": "text",
                    "locator": "file;chunk=0",
                    "text": "alpha beta gamma",
                    "metadata": {},
                },
                {
                    "chunk_id": "b",
                    "source_path": "b.txt",
                    "source_kind": "text",
                    "source_role": "text",
                    "locator": "file;chunk=0",
                    "text": "credit score loan default",
                    "metadata": {},
                },
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            results = BM25Retriever.from_jsonl(path).retrieve("credit loan", top_k=1)

        self.assertEqual(results[0].source_path, "b.txt")


if __name__ == "__main__":
    unittest.main()
