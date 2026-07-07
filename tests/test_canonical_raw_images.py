from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.canonical import builder
from src.canonical.builder import CanonicalBuildConfig


class CanonicalRawImageTests(unittest.TestCase):
    def test_raw_image_parser_none_keeps_only_raw_image_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "poster.jpg"
            image_path.write_bytes(b"fake image")
            config = CanonicalBuildConfig(data_root=root, output_dir=root / "out", raw_image_parser="none")

            texts, images = builder._raw_image_records(image_path, "poster.jpg", ".jpg", config)

        self.assertEqual(texts, [])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].role, "raw_image")
        self.assertEqual(images[0].image_path, "poster.jpg")

    def test_lift_raw_image_parser_creates_ocr_text_and_extracted_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "poster.jpg"
            image_path.write_bytes(b"fake image")
            output_dir = root / "out"
            output_dir.mkdir()
            extracted = output_dir / "lift_raw_images" / "raw_outputs" / "poster_raw_lift" / "figure.jpg"
            extracted.parent.mkdir(parents=True)
            extracted.write_bytes(b"fake extracted image")
            config = CanonicalBuildConfig(
                data_root=root,
                output_dir=output_dir,
                raw_image_parser="lift-api",
                raw_image_lift_output_dir=output_dir / "lift_raw_images",
            )
            payload = {
                "text": "OCR heading\n\n![simple chart showing revenue](figure.jpg)",
                "extraction": {"figures": [{"caption": "fallback figure caption"}]},
                "metadata": {"raw_output_path": "raw.json", "image_count": 1, "image_source": "extract"},
                "image_files": [str(extracted)],
            }

            with patch.object(builder, "_parse_raw_image_with_lift", return_value=payload):
                texts, images = builder._raw_image_records(image_path, "poster.jpg", ".jpg", config)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].role, "raw_image_ocr_text")
        self.assertEqual(texts[0].parser, "lift-api")
        self.assertEqual(images[0].role, "raw_image")
        self.assertIn("OCR heading", images[0].description)
        self.assertEqual(images[1].role, "raw_image_extracted_region")
        self.assertEqual(images[1].description, "simple chart showing revenue")
        self.assertEqual(images[1].image_path, "lift_raw_images/raw_outputs/poster_raw_lift/figure.jpg")

    def test_raw_image_parser_defaults_to_lift_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = CanonicalBuildConfig(data_root=root, output_dir=root / "out")

        self.assertEqual(config.raw_image_parser, "lift-api")


if __name__ == "__main__":
    unittest.main()
