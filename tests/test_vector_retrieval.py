from __future__ import annotations

from collections import Counter
import unittest

from src.retrieval.vector import VectorChunkHit, _group_hits


class VectorRetrievalTests(unittest.TestCase):
    def test_max_mean_top3_aggregation_limits_long_file_bias(self) -> None:
        hits = [
            VectorChunkHit("a1", "long.pdf", "text", "text", "1", 0.86, "near match"),
            VectorChunkHit("a2", "long.pdf", "text", "text", "2", 0.85, "near match"),
            VectorChunkHit("a3", "long.pdf", "text", "text", "3", 0.85, "near match"),
            VectorChunkHit("a4", "long.pdf", "text", "text", "4", 0.85, "near match"),
            VectorChunkHit("b1", "answer.txt", "text", "text", "1", 0.90, "strong match"),
        ]

        results = _group_hits(
            hits,
            Counter({"long.pdf": 100, "answer.txt": 1}),
            aggregation="max_mean_top3",
        )

        self.assertEqual(results[0].source_path, "answer.txt")
        self.assertEqual(len(results[1].chunks), 3)
        self.assertEqual(results[1].score_components["matched_chunk_bonus_count"], 3)
        self.assertEqual(results[1].score_components["source_chunk_count"], 100)

    def test_legacy_aggregation_is_still_available(self) -> None:
        hits = [
            VectorChunkHit("a1", "long.pdf", "text", "text", "1", 0.86, "near match"),
            VectorChunkHit("a2", "long.pdf", "text", "text", "2", 0.85, "near match"),
            VectorChunkHit("a3", "long.pdf", "text", "text", "3", 0.85, "near match"),
            VectorChunkHit("a4", "long.pdf", "text", "text", "4", 0.85, "near match"),
            VectorChunkHit("b1", "answer.txt", "text", "text", "1", 0.90, "strong match"),
        ]

        results = _group_hits(hits, Counter(), aggregation="legacy")

        self.assertEqual(results[0].source_path, "long.pdf")
        self.assertEqual(results[0].score_components["aggregation"], "legacy")


if __name__ == "__main__":
    unittest.main()
