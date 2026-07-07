from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from src.canonical.embedding import (
    ChunkBuildConfig,
    build_canonical_chunks,
    build_vector_records,
)


class FakeProvider:
    model = "fake-embedding"
    dimension = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class CanonicalEmbeddingTests(unittest.TestCase):
    def test_chunking_text_and_image_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "texts.jsonl",
                [
                    {
                        "text_id": "t1",
                        "source_path": "doc.txt",
                        "source_extension": ".txt",
                        "text": "abcdef" * 300,
                        "role": "text",
                        "parser": "plain-text",
                        "locator": "file",
                    }
                ],
            )
            _write_jsonl(
                root / "images.jsonl",
                [
                    {
                        "image_id": "i1",
                        "source_path": "image.jpg",
                        "source_extension": ".jpg",
                        "image_path": "image.jpg",
                        "role": "raw_image",
                        "description": "blue digit five",
                        "locator": "file",
                    }
                ],
            )
            _write_jsonl(root / "tables.jsonl", [])

            chunks, report = build_canonical_chunks(ChunkBuildConfig(root, chunk_size=100, chunk_overlap=10))

        self.assertGreater(len([chunk for chunk in chunks if chunk.source_kind == "text"]), 1)
        self.assertEqual(len([chunk for chunk in chunks if chunk.source_kind == "image"]), 1)
        self.assertEqual(report["source_kind_counts"]["image"], 1)
        self.assertIn("Source: image.jpg", [chunk for chunk in chunks if chunk.source_kind == "image"][0].embedding_text)

    def test_chunking_skips_placeholders_and_empty_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "texts.jsonl",
                [
                    {
                        "text_id": "ppt",
                        "source_path": "deck.ppt",
                        "source_extension": ".ppt",
                        "text": "Legacy PPT file available at deck.ppt. Local text extraction is not implemented.",
                        "role": "document_text",
                        "parser": "ppt-placeholder",
                        "locator": "file",
                    }
                ],
            )
            _write_jsonl(
                root / "images.jsonl",
                [
                    {
                        "image_id": "empty",
                        "source_path": "empty.jpg",
                        "source_extension": ".jpg",
                        "image_path": "empty.jpg",
                        "role": "raw_image",
                        "description": "",
                        "locator": "file",
                    }
                ],
            )
            _write_jsonl(root / "tables.jsonl", [])

            chunks, _ = build_canonical_chunks(ChunkBuildConfig(root))

        self.assertEqual(chunks, [])

    def test_chunking_table_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "texts.jsonl", [])
            _write_jsonl(root / "images.jsonl", [])
            _write_jsonl(
                root / "tables.jsonl",
                [
                    {
                        "table_id": "tbl1",
                        "source_path": "data.csv",
                        "source_extension": ".csv",
                        "table_path": "data.csv",
                        "role": "raw_table",
                        "parser": "delimited-profile",
                        "description": "Source table: data.csv\n\nColumn names: customer_id, score",
                        "llm_description": "Customer score table for segments and credit analysis.",
                        "locator": "file",
                    }
                ],
            )

            chunks, report = build_canonical_chunks(ChunkBuildConfig(root))

        table_chunks = [chunk for chunk in chunks if chunk.source_kind == "table"]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(report["source_kind_counts"]["table"], 1)
        self.assertIn("Customer score table", table_chunks[0].text)
        self.assertIn("Structured table profile", table_chunks[0].text)
        self.assertTrue(table_chunks[0].metadata["has_llm_description"])
        self.assertIn("Type: table", table_chunks[0].embedding_text)

    def test_embedding_cache_reuses_existing_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "texts.jsonl",
                [
                    {
                        "text_id": "t1",
                        "source_path": "doc.txt",
                        "source_extension": ".txt",
                        "text": "hello retrieval",
                        "role": "text",
                        "parser": "plain-text",
                        "locator": "file",
                    }
                ],
            )
            _write_jsonl(root / "images.jsonl", [])
            _write_jsonl(root / "tables.jsonl", [])
            chunks, _ = build_canonical_chunks(ChunkBuildConfig(root))
            output_dir = root / "embeddings"

            first_provider = FakeProvider()
            first_vectors, first_report = build_vector_records(chunks, first_provider, output_dir, batch_size=8)
            second_provider = FakeProvider()
            second_vectors, second_report = build_vector_records(chunks, second_provider, output_dir, batch_size=8)

        self.assertEqual(first_provider.calls, 1)
        self.assertEqual(second_provider.calls, 0)
        self.assertEqual(first_vectors, second_vectors)
        self.assertEqual(first_report["embedded_count"], 1)
        self.assertEqual(second_report["reused_count"], 1)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
