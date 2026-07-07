"""Export retrieval handoff artifacts to a standalone folder."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import (  # noqa: E402
    CANONICAL_DIR,
    OPENROUTER_EMBEDDINGS_DIR,
    OUTPUT_DIR,
    RAW_DATA_LAKE_DIR,
)


DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "handoff" / "data_lake_retrieval_bundle"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Copy Data-Lake retrieval handoff artifacts into one folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-raw", action="store_true", help="Also copy data/raw/Data-Lake for downstream reasoning.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output dir if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without writing files.")
    args = parser.parse_args(argv)

    output_dir = _resolve(args.output_dir)
    items = _handoff_items(include_raw=args.include_raw)
    missing = _missing_sources(items)
    required_missing = [item for item in missing if item.get("required", True)]
    if required_missing:
        raise FileNotFoundError("Missing handoff sources:\n" + "\n".join(item["source"].as_posix() for item in required_missing))
    existing_items = [item for item in items if item["source"].exists()]

    manifest = _build_manifest(output_dir, existing_items, missing, include_raw=args.include_raw)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        _safe_remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in existing_items:
        target = output_dir / item["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source = item["source"]
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README_handoff.md").write_text(_readme(manifest), encoding="utf-8")
    print(json.dumps({"output_dir": output_dir.as_posix(), "items": len(existing_items), "skipped": len(missing)}, ensure_ascii=False, indent=2))


def _handoff_items(*, include_raw: bool) -> list[dict]:
    items = [
        {"name": "canonical", "source": _resolve(CANONICAL_DIR), "target": Path("canonical")},
        {
            "name": "openrouter_embeddings",
            "source": _resolve(OPENROUTER_EMBEDDINGS_DIR),
            "target": Path("model_raw/openrouter_embeddings"),
        },
        {"name": "pipeline_config", "source": _resolve(Path("configs/pipeline.yaml")), "target": Path("configs/pipeline.yaml")},
        {"name": "env_example", "source": _resolve(Path(".env.example")), "target": Path(".env.example")},
    ]
    if include_raw:
        items.append({"name": "raw_data_lake", "source": _resolve(RAW_DATA_LAKE_DIR), "target": Path("raw/Data-Lake")})
    return items


def _build_manifest(output_dir: Path, items: list[dict], missing: list[dict], *, include_raw: bool) -> dict:
    return {
        "contract_version": "data-lake-retrieval-handoff-v1",
        "created_at_unix": int(time.time()),
        "output_dir": output_dir.as_posix(),
        "include_raw": include_raw,
        "items": [
            {
                "name": item["name"],
                "source": _portable(item["source"]),
                "target": item["target"].as_posix(),
                "kind": "directory" if item["source"].is_dir() else "file",
                "file_count": _file_count(item["source"]),
                "size_bytes": _size_bytes(item["source"]),
                "jsonl_rows": _jsonl_counts(item["source"]),
            }
            for item in items
        ],
        "skipped_missing_optional_items": [
            {
                "name": item["name"],
                "source": _portable(item["source"]),
                "target": item["target"].as_posix(),
            }
            for item in missing
            if not item.get("required", True)
        ],
        "notes": [
            "canonical plus openrouter_embeddings is sufficient for retrieval.",
            "include raw/Data-Lake when the receiving QA system must inspect original files for reasoning or execution.",
            "openrouter enrichment outputs are not required because their descriptions are merged into canonical JSONL files.",
        ],
    }


def _readme(manifest: dict) -> str:
    lines = [
        "# Data-Lake Retrieval Handoff",
        "",
        "This bundle contains reusable artifacts for document retrieval.",
        "",
        "## Contents",
        "",
    ]
    for item in manifest["items"]:
        lines.append(f"- `{item['target']}` from `{item['source']}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use `canonical/chunks.jsonl` with `model_raw/openrouter_embeddings/vector_records.jsonl` for vector retrieval.",
            "- Use `canonical/texts.jsonl`, `images.jsonl`, and `tables.jsonl` to inspect retrieved records.",
            "- Raw files are included only when this bundle was created with `--include-raw`.",
            "",
        ]
    )
    return "\n".join(lines)


def _missing_sources(items: list[dict]) -> list[dict]:
    return [item for item in items if not item["source"].exists()]


def _safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to overwrite outside project root: {resolved}")
    shutil.rmtree(resolved)


def _file_count(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file())


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _jsonl_counts(path: Path) -> dict[str, int]:
    candidates = [path] if path.is_file() and path.suffix == ".jsonl" else sorted(path.rglob("*.jsonl")) if path.is_dir() else []
    return {_portable(candidate): sum(1 for line in candidate.open(encoding="utf-8") if line.strip()) for candidate in candidates}


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
