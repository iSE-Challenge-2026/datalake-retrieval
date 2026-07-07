from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path(
    "data/processed/final_test/lift_outputs/"
    "newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_018_raw_lift/"
    "extract.document.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the structure of a Lift JSON output file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="Path to a Lift JSON file. Defaults to the final_test extract.document.json.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum nesting depth to print.",
    )
    parser.add_argument(
        "--sample-lists",
        action="store_true",
        help="Print the structure of the first item in each list.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    print(f"File: {path}")
    print("Structure:")
    print_structure(data, max_depth=args.max_depth, sample_lists=args.sample_lists)


def print_structure(
    value: Any,
    *,
    max_depth: int,
    sample_lists: bool,
    indent: int = 0,
    name: str = "root",
) -> None:
    prefix = "  " * indent
    type_name = type(value).__name__

    if indent >= max_depth:
        print(f"{prefix}{name}: {type_name} ...")
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: object ({len(value)} keys)")
        for key, child in value.items():
            print_structure(
                child,
                max_depth=max_depth,
                sample_lists=sample_lists,
                indent=indent + 1,
                name=str(key),
            )
        return

    if isinstance(value, list):
        print(f"{prefix}{name}: array ({len(value)} items)")
        if sample_lists and value:
            print_structure(
                value[0],
                max_depth=max_depth,
                sample_lists=sample_lists,
                indent=indent + 1,
                name="[0]",
            )
        return

    print(f"{prefix}{name}: {type_name}")


if __name__ == "__main__":
    main()
