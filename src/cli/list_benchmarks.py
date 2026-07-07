"""List saved retrieval benchmark runs."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import OUTPUT_BENCHMARKS_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="List saved retrieval benchmark summaries.")
    parser.add_argument("--bench-dir", type=Path, default=OUTPUT_BENCHMARKS_DIR)
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args(argv)

    bench_dir = _resolve(args.bench_dir)
    rows = []
    manifests = sorted(bench_dir.glob("**/manifest.json"))
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        relative = run_dir.relative_to(bench_dir).as_posix()
        if not args.include_archive and relative.startswith("archive/"):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        rows.append(
            {
                "run_name": run_dir.name,
                "group": relative.rsplit("/", 1)[0] if "/" in relative else ".",
                "mode": manifest.get("mode"),
                "top_k": manifest.get("top_k"),
                "recall": summary.get("recall_at_k_macro"),
                "precision": summary.get("precision_at_k_macro"),
                "pattern_recall": summary.get("pattern_recall_at_k_macro"),
                "perfect": summary.get("perfect_recall_rate"),
                "lexical_weight": manifest.get("lexical_weight"),
                "vector_weight": manifest.get("vector_weight"),
                "results": (run_dir / "results.json").as_posix(),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
