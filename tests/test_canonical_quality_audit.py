from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_canonical_quality.py"
SPEC = importlib.util.spec_from_file_location("audit_canonical_quality", SCRIPT_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class CanonicalQualityAuditTests(unittest.TestCase):
    def test_audit_handles_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            canonical.mkdir()
            _write_jsonl(canonical / "texts.jsonl", [{"source_path": "a.txt", "text": "hello world"}])
            _write_jsonl(canonical / "images.jsonl", [{"source_path": "image.jpg", "role": "raw_image"}])
            _write_jsonl(canonical / "tables.jsonl", [{"source_path": "table.csv", "columns": ["a"]}])
            _write_jsonl(
                canonical / "chunks.jsonl",
                [
                    {
                        "chunk_id": "c1",
                        "source_path": "a.txt",
                        "source_kind": "text",
                        "source_role": "text",
                        "text": "hello world",
                        "embedding_text": "Source: a.txt\n\nhello world",
                    }
                ],
            )

            summary = audit.build_audit_summary(canonical, root / "missing_vectors.jsonl")

        self.assertEqual(summary["overview"]["text_records"], 1)
        self.assertEqual(summary["images"]["raw_image_count"], 1)
        self.assertEqual(summary["embeddings"]["missing_vector_count"], 1)
        self.assertIn("top_sources_by_chunk_count", summary["chunks"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
