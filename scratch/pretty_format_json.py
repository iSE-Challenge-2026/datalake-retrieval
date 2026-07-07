from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PATH = Path(
    "data/processed/final_test/lift_outputs/"
    "newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_018_raw_lift/"
    "extract.document.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-format a JSON file with indentation.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="JSON file to format. Defaults to the final_test Lift extract.document.json.",
    )
    parser.add_argument(
        "--output",
        help="Write formatted JSON to this path instead of overwriting the input file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces per indent level.",
    )
    args = parser.parse_args()

    input_path = Path(args.path)
    output_path = Path(args.output) if args.output else input_path

    with input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    output_path.write_text(
        json.dumps(data, indent=args.indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    action = "Wrote" if args.output else "Formatted"
    print(f"{action}: {output_path}")


if __name__ == "__main__":
    main()
