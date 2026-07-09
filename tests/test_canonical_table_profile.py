from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from src.canonical.builder import CanonicalBuildConfig, build_canonical_artifacts
from src.canonical import builder


class CanonicalTableProfileTests(unittest.TestCase):
    def test_csv_profile_separates_metadata_header_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "raw"
            output_dir = root / "canonical"
            data_root.mkdir()
            (data_root / "scores.csv").write_text(
                "Exported from survey system\n"
                "Contains anonymized credit scores\n"
                "customer_id,score,segment\n"
                "c1,720,A\n"
                "c2,650,B\n",
                encoding="utf-8",
            )

            build_canonical_artifacts(CanonicalBuildConfig(data_root=data_root, output_dir=output_dir))
            tables = _read_jsonl(output_dir / "tables.jsonl")

        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table["parser"], "delimited-profile")
        self.assertEqual(table["columns"], ["customer_id", "score", "segment"])
        self.assertIn("Exported from survey system", table["metadata_text"])
        self.assertEqual(table["sample_rows"][0]["customer_id"], "c1")
        self.assertEqual(table["tail_sample_rows"][-1]["customer_id"], "c2")
        self.assertIn("Column names: customer_id, score, segment", table["description"])

    def test_xlsx_profile_creates_one_table_record_per_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.xlsx"
            path.write_bytes(b"placeholder")
            config = CanonicalBuildConfig(data_root=Path(tmp), output_dir=Path(tmp) / "out")
            fake_sheets = [
                {"sheet_name": "Customers", "rows": [["note"], ["customer_id", "score"], ["c1", "720"]]},
                {"sheet_name": "Orders", "rows": [["order_id", "amount"], ["o1", "10"]]},
            ]

            with patch.object(builder, "_xlsx_sheet_rows", return_value=fake_sheets):
                records = builder._table_records(path, "book.xlsx", ".xlsx", config)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].table_path, "book.xlsx#sheet=Customers")
        self.assertEqual(records[0].locator, "sheet=Customers")
        self.assertEqual(records[0].source_path, "book.xlsx")
        self.assertEqual(records[1].columns, ["order_id", "amount"])
        self.assertEqual(records[1].tail_sample_rows[-1]["order_id"], "o1")
        self.assertIn("book.xlsx#sheet=Orders", records[1].description)

    def test_sql_profile_executes_script_and_extracts_table_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "grades.sql"
            path.write_text(
                "CREATE TABLE class_scores (student_id INTEGER, class_name TEXT, math_score REAL);\n"
                "INSERT INTO class_scores VALUES (1, '10A1', 8.0);\n"
                "INSERT INTO class_scores VALUES (2, '10A1', 6.5);\n",
                encoding="utf-8",
            )
            config = CanonicalBuildConfig(data_root=root, output_dir=root / "out")

            records = builder._table_records(path, "grades.sql", ".sql", config)

        self.assertEqual(len(records), 1)
        table = records[0]
        self.assertEqual(table.parser, "sql-executed-table")
        self.assertEqual(table.table_path, "grades.sql#table=class_scores")
        self.assertEqual(table.locator, "table=class_scores")
        self.assertEqual(table.columns, ["student_id", "class_name", "math_score"])
        self.assertEqual(table.sample_rows[0]["math_score"], "8.0")
        self.assertEqual(table.table_shape["row_count"], 2)
        self.assertIn("First sample rows", table.description)

    def test_legacy_ppt_uses_converted_pptx_parser_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppt_path = root / "deck.ppt"
            converted_path = root / "cache" / "deck.pptx"
            ppt_path.write_bytes(b"legacy ppt")
            converted_path.parent.mkdir()
            converted_path.write_bytes(b"converted")
            config = CanonicalBuildConfig(
                data_root=root,
                output_dir=root / "out",
                ppt_conversion_cache_dir=root / "cache",
            )

            with (
                patch.object(builder, "_convert_ppt_to_pptx", return_value=(converted_path, "test-converter", "")),
                patch.object(builder, "_pptx_text", return_value=("Slide 1: Real text", "pptx-ooxml")),
                patch.object(builder, "_pptx_image_records", return_value=[]),
            ):
                texts, images = builder._doc_file_records(ppt_path, "deck.ppt", ".ppt", config, root / "out" / "extracted_images")

        self.assertEqual(images, [])
        self.assertEqual(texts[0].text, "Slide 1: Real text")
        self.assertEqual(texts[0].parser, "ppt-converted-test-converter+pptx-ooxml")

    def test_pdf_prefers_datalab_cache_before_native_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "native.pdf"
            pdf_path.write_bytes(b"%PDF placeholder")
            config = CanonicalBuildConfig(
                data_root=root,
                output_dir=root / "out",
                extract_pdf_images=True,
            )
            datalab_text = builder.CanonicalText(
                text_id="cached",
                source_path="native.pdf",
                source_extension=".pdf",
                text="Datalab markdown text with figure descriptions",
                role="document_text",
                parser="datalab-parsing-cache",
            )

            with (
                patch.object(builder, "_datalab_doc_file_records", return_value=([datalab_text], [])) as datalab_records,
                patch.object(builder, "_pdf_text") as pdf_text,
                patch.object(builder, "_pdf_image_records") as pdf_images,
            ):
                texts, images = builder._doc_file_records(pdf_path, "native.pdf", ".pdf", config, root / "out" / "extracted_images")

        self.assertEqual(images, [])
        self.assertEqual(texts[0].text, "Datalab markdown text with figure descriptions")
        self.assertEqual(texts[0].parser, "datalab-parsing-cache")
        datalab_records.assert_called_once()
        pdf_text.assert_not_called()
        pdf_images.assert_not_called()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
