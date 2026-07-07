from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import json
import subprocess
import sys
import tempfile
import unittest

from src.retrieval.benchmarks import benchmark_manifest
from src.retrieval.expansion import ExpandedRetrievalResult, expand_mentioned_folders, load_chunk_index
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from src.retrieval.presets import apply_preset


class RetrievalRefactorTests(unittest.TestCase):
    def test_folder_expansion_uses_quoted_folder_only(self) -> None:
        base = [ExpandedRetrievalResult("number_image/2.png", 1.0, "image")]
        expanded = expand_mentioned_folders(
            'How many images in "number_image" contain a blue digit?',
            base,
            ["number_image/2.png", "number_image/3.png", "other/3.png"],
            {},
            max_files=10,
        )

        self.assertEqual([item.source_path for item in expanded], ["number_image/2.png", "number_image/3.png"])

    def test_folder_expansion_skips_large_folders(self) -> None:
        base = [ExpandedRetrievalResult("big/a.txt", 1.0, "text")]
        expanded = expand_mentioned_folders(
            'Inspect "big"',
            base,
            ["big/a.txt", "big/b.txt", "big/c.txt"],
            {},
            max_files=2,
        )

        self.assertEqual([item.source_path for item in expanded], ["big/a.txt"])

    def test_load_chunk_index_provides_expansion_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "chunk_id": "c1",
                        "source_path": "number_image/2.png",
                        "source_kind": "image",
                        "source_role": "raw_image",
                        "locator": "chunk=0",
                        "text": "blue number two",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            index = load_chunk_index(path)

        self.assertEqual(index["number_image/2.png"][0].text, "blue number two")

    def test_manifest_records_preset_and_existing_fields(self) -> None:
        config = RetrievalPipelineConfig(
            retrieval_mode="vector",
            vector_records=Path("vectors.jsonl"),
            rerank_model="google/gemini-2.5-flash",
            expand_mentioned_folders=True,
        )

        manifest = benchmark_manifest(
            run_name="run",
            config=config,
            questions_path=Path("questions.xlsx"),
            data_root=Path("data/raw/Data-Lake"),
            dump_score_details=True,
            preset="folder_expand_flash",
        )

        self.assertEqual(manifest["preset"], "folder_expand_flash")
        self.assertEqual(manifest["mode"], "vector")
        self.assertTrue(manifest["expand_mentioned_folders"])
        self.assertEqual(manifest["rerank_model"], "google/gemini-2.5-flash")

    def test_preset_can_be_overridden(self) -> None:
        values = apply_preset(
            {
                "preset": "folder_expand_flash",
                "expand_mentioned_folders": False,
                "top_k": 10,
            },
            {
                "retrieval_mode": None,
                "expand_mentioned_folders": False,
                "top_k": 20,
            },
        )

        self.assertEqual(values["retrieval_mode"], "vector")
        self.assertFalse(values["expand_mentioned_folders"])
        self.assertEqual(values["top_k"], 10)

    def test_pipeline_vector_only_order_with_fake_embedding_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vector_path = root / "vectors.jsonl"
            _write_jsonl(
                vector_path,
                [
                    _vector("a", "a.txt", [0.1, 0.9], "weak"),
                    _vector("b", "b.txt", [1.0, 0.0], "strong"),
                ],
            )
            config = RetrievalPipelineConfig(
                retrieval_mode="vector",
                vector_records=vector_path,
                top_k=2,
                vector_chunk_k=2,
                rerank_mode="none",
            )

            with patch("src.retrieval.pipeline.OpenRouterEmbeddingProvider", FakeEmbeddingProvider):
                pipeline = RetrievalPipeline(config, project_root=root)
                results = pipeline.retrieve("query")

        self.assertEqual([item.source_path for item in results], ["b.txt", "a.txt"])

    def test_retrieve_eval_cli_help_smoke(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/retrieve_eval.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--preset", result.stdout)


class FakeEmbeddingProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _vector(chunk_id: str, source_path: str, embedding: list[float], text: str) -> dict:
    return {
        "vector_id": chunk_id,
        "chunk_id": chunk_id,
        "source_path": source_path,
        "source_kind": "text",
        "source_role": "text",
        "locator": "chunk=0",
        "text": text,
        "embedding": embedding,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
