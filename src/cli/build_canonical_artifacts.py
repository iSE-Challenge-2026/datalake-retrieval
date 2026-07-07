"""Build canonical text/image/table artifacts from the raw data lake."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical import CanonicalBuildConfig, build_canonical_artifacts  # noqa: E402
from src.project_paths import (  # noqa: E402
    CANONICAL_DIR,
    DATALAB_PARSING_DIR,
    OPENROUTER_ASR_CACHE_DIR,
    RAW_DATA_LAKE_DIR,
    RAW_IMAGE_LIFT_OUTPUT_DIR,
)
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_file(PROJECT_ROOT)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    known, _ = bootstrap.parse_known_args(argv)
    pipeline_config = _load_pipeline_config(_resolve(known.config))
    defaults = pipeline_config.get("canonical", {})

    parser = argparse.ArgumentParser(description="Build canonical retrieval artifacts.", parents=[bootstrap])
    parser.add_argument("--data-root", type=Path, default=Path(defaults.get("data_root", RAW_DATA_LAKE_DIR)))
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.get("output_dir", CANONICAL_DIR)))
    parser.add_argument("--asr-provider", choices=["none", "openrouter"], default=defaults.get("asr_provider", "none"))
    parser.add_argument("--asr-model", default=defaults.get("asr_model") or os.getenv("OPENROUTER_ASR_MODEL", "openai/whisper-large-v3"))
    parser.add_argument("--audio-language", default=defaults.get("audio_language") or os.getenv("OPENROUTER_ASR_LANGUAGE") or None)
    parser.add_argument("--asr-cache-dir", type=Path, default=Path(defaults.get("asr_cache_dir", OPENROUTER_ASR_CACHE_DIR)))
    parser.add_argument("--max-text-chars", type=int, default=int(defaults.get("max_text_chars", 2_000_000)))
    parser.add_argument("--max-table-preview-rows", type=int, default=int(defaults.get("max_table_preview_rows", 80)))
    parser.add_argument("--no-copy-extracted-images", action="store_true")
    parser.add_argument("--extract-pdf-images", action="store_true", default=bool(defaults.get("extract_pdf_images", False)))
    parser.add_argument("--no-extract-pdf-images", dest="extract_pdf_images", action="store_false")
    parser.add_argument("--extract-pptx-images", dest="extract_pptx_images", action="store_true", default=bool(defaults.get("extract_pptx_images", True)))
    parser.add_argument("--no-extract-pptx-images", dest="extract_pptx_images", action="store_false")
    parser.add_argument("--ppt-conversion-cache-dir", type=Path, default=_optional_path(defaults.get("ppt_conversion_cache_dir")))
    parser.add_argument("--datalab-parsing-dir", type=Path, default=Path(defaults.get("datalab_parsing_dir", DATALAB_PARSING_DIR)))
    parser.add_argument("--raw-image-parser", choices=["none", "lift-api"], default=defaults.get("raw_image_parser", "lift-api"))
    parser.add_argument("--raw-image-lift-output-dir", type=Path, default=Path(defaults.get("raw_image_lift_output_dir", RAW_IMAGE_LIFT_OUTPUT_DIR)))
    parser.add_argument("--raw-image-lift-schema-path", type=Path, default=_optional_path(defaults.get("raw_image_lift_schema_path")))
    parser.add_argument("--raw-image-lift-mode", default=defaults.get("raw_image_lift_mode", "balanced"))
    parser.add_argument("--raw-image-lift-api-key-env", default=defaults.get("raw_image_lift_api_key_env", "DATALAB_API_KEY"))
    args = parser.parse_args(argv)

    config = CanonicalBuildConfig(
        data_root=_resolve(args.data_root),
        output_dir=_resolve(args.output_dir),
        asr_provider=args.asr_provider,
        asr_model=args.asr_model,
        audio_language=args.audio_language,
        asr_cache_dir=_resolve(args.asr_cache_dir) if args.asr_cache_dir else None,
        max_text_chars=args.max_text_chars,
        max_table_preview_rows=args.max_table_preview_rows,
        copy_extracted_images=not args.no_copy_extracted_images,
        extract_pdf_images=args.extract_pdf_images,
        extract_pptx_images=args.extract_pptx_images,
        ppt_conversion_cache_dir=_resolve(args.ppt_conversion_cache_dir) if args.ppt_conversion_cache_dir else None,
        datalab_parsing_dir=_resolve(args.datalab_parsing_dir) if args.datalab_parsing_dir else None,
        raw_image_parser=args.raw_image_parser,
        raw_image_lift_output_dir=_resolve(args.raw_image_lift_output_dir) if args.raw_image_lift_output_dir else None,
        raw_image_lift_schema_path=_resolve(args.raw_image_lift_schema_path) if args.raw_image_lift_schema_path else None,
        raw_image_lift_mode=args.raw_image_lift_mode,
        raw_image_lift_api_key_env=args.raw_image_lift_api_key_env,
    )
    summary = build_canonical_artifacts(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_pipeline_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


if __name__ == "__main__":
    main()
