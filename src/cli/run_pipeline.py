"""Run the full Data-Lake ingestion and indexing pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical.image_enrichment import main as image_enrichment_main  # noqa: E402
from src.canonical.table_enrichment import main as table_enrichment_main  # noqa: E402
from src.cli.build_canonical_artifacts import main as canonical_main  # noqa: E402
from src.cli.build_canonical_embeddings import main as embeddings_main  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


DEFAULT_STAGES = ("canonical", "image_enrichment", "table_enrichment", "embeddings")
STAGE_ALIASES = {
    "ingestion": "canonical",
    "normalize": "canonical",
    "normalization": "canonical",
    "image": "image_enrichment",
    "images": "image_enrichment",
    "table": "table_enrichment",
    "tables": "table_enrichment",
    "chunking": "embeddings",
    "indexing": "embeddings",
    "embedding": "embeddings",
}


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_file(PROJECT_ROOT)
    args = _parser().parse_args(argv)
    config_path = _resolve(args.config)
    stages = _parse_stages(args.stages)
    if args.skip_enrichment:
        stages = [stage for stage in stages if stage not in {"image_enrichment", "table_enrichment"}]

    commands = _stage_commands(stages, config_path)
    if args.dry_run:
        print(json.dumps({"config": config_path.as_posix(), "stages": stages, "commands": commands}, ensure_ascii=False, indent=2))
        return

    started = time.time()
    completed: list[dict] = []
    for stage in stages:
        stage_started = time.time()
        print(json.dumps({"stage": stage, "status": "started"}, ensure_ascii=False))
        _run_stage(stage, config_path)
        completed.append({"stage": stage, "seconds": round(time.time() - stage_started, 3)})
        print(json.dumps({"stage": stage, "status": "completed", "seconds": completed[-1]["seconds"]}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "status": "completed",
                "config": config_path.as_posix(),
                "stages": completed,
                "total_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Data-Lake ingestion, enrichment, chunking, and indexing pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help="Comma-separated stages: canonical,image_enrichment,table_enrichment,embeddings.",
    )
    parser.add_argument("--skip-enrichment", action="store_true", help="Run canonical and embeddings without image/table LLM enrichment.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_stages(value: str) -> list[str]:
    stages: list[str] = []
    for item in value.split(","):
        stage = STAGE_ALIASES.get(item.strip(), item.strip())
        if not stage:
            continue
        if stage not in DEFAULT_STAGES:
            raise ValueError(f"Unsupported pipeline stage: {stage}")
        stages.append(stage)
    return stages


def _stage_commands(stages: list[str], config_path: Path) -> list[dict]:
    commands = []
    for stage in stages:
        commands.append({"stage": stage, "argv": _stage_argv(stage, config_path)})
    return commands


def _run_stage(stage: str, config_path: Path) -> None:
    argv = _stage_argv(stage, config_path)
    if stage == "canonical":
        canonical_main(argv)
    elif stage == "image_enrichment":
        image_enrichment_main(argv)
    elif stage == "table_enrichment":
        table_enrichment_main(argv)
    elif stage == "embeddings":
        embeddings_main(argv)
    else:
        raise ValueError(f"Unsupported pipeline stage: {stage}")


def _stage_argv(stage: str, config_path: Path) -> list[str]:
    if stage in {"canonical", "image_enrichment", "table_enrichment", "embeddings"}:
        return ["--config", config_path.as_posix()]
    raise ValueError(f"Unsupported pipeline stage: {stage}")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
