import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


SUBMISSION_COLUMNS = ["id", "answer", "evidences"]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_runs_folder(folder_name: str, runs_dir: str = "runs") -> Path:
    folder = Path(folder_name)
    if not folder.is_absolute() and len(folder.parts) == 1:
        folder_in_runs = Path(runs_dir) / folder_name
        if folder_in_runs.exists():
            return folder_in_runs.resolve()

    if folder.exists():
        return folder.resolve()

    raise FileNotFoundError(f"Runs folder not found: {folder_name}")


def _question_id_from_run_dir(run_dir: Path) -> Optional[int]:
    match = re.search(r"question[_-]?(\d+)$", run_dir.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)$", run_dir.name)
    return int(match.group(1)) if match else None


def _iter_run_dirs(runs_folder: Path) -> List[Path]:
    if (runs_folder / "final_output").exists() or (runs_folder / "pipeline_state.json").exists():
        return [runs_folder]

    run_dirs = [
        path
        for path in runs_folder.iterdir()
        if path.is_dir() and _question_id_from_run_dir(path) is not None
    ]
    return sorted(run_dirs, key=lambda path: (_question_id_from_run_dir(path) or 0, path.name))


def _normalize_evidence_path(path_value: str, data_dir: str = "competition/data_lake") -> str:
    path_text = str(path_value).strip()
    if not path_text:
        return path_text

    path = Path(path_text)
    data_root = Path(data_dir).resolve()

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    try:
        return resolved.relative_to(data_root).as_posix()
    except ValueError:
        pass

    normalized = path_text.replace("\\", "/")
    marker = data_root.as_posix().rstrip("/") + "/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]

    return normalized


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _answer_from_run(run_dir: Path) -> str:
    result = _load_json(run_dir / "final_output" / "result.json")
    answer = result.get("final_answer", result.get("answer", ""))
    if answer is None:
        return ""
    return str(answer).strip()


def _evidences_from_state(run_dir: Path, data_dir: str) -> List[str]:
    state = _load_json(run_dir / "pipeline_state.json")
    evidences = []

    data_descriptions = state.get("data_descriptions", {})
    if isinstance(data_descriptions, dict):
        evidences.extend(data_descriptions.keys())

    return [_normalize_evidence_path(path, data_dir) for path in evidences]


def _evidences_from_step_metadata(run_dir: Path, data_dir: str) -> List[str]:
    evidences = []
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return evidences

    for metadata_path in sorted(steps_dir.glob("*/metadata.json")):
        metadata = _load_json(metadata_path)
        attached_files = metadata.get("attached_files")
        if isinstance(attached_files, list):
            evidences.extend(str(path) for path in attached_files)

        filename = metadata.get("filename")
        if filename:
            evidences.append(str(filename))

        prompt = metadata.get("prompt", "")
        if isinstance(prompt, str):
            evidences.extend(
                match.strip()
                for match in re.findall(r"(?m)^Path:\s*(.+)$", prompt)
            )

    return [_normalize_evidence_path(path, data_dir) for path in evidences]


def _load_question_evidence_fallback(questions_file: str) -> Dict[int, List[str]]:
    path = Path(questions_file)
    if not path.exists():
        return {}

    fallback: Dict[int, List[str]] = {}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
        rows = frame.to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    for row in rows:
        try:
            question_id = int(row.get("STT", ""))
        except ValueError:
            continue

        raw_sources = (row.get("Data Sources") or "").strip()
        if not raw_sources:
            fallback[question_id] = []
            continue

        try:
            sources = json.loads(raw_sources)
        except json.JSONDecodeError:
            sources = []

        fallback[question_id] = [str(source).replace("\\", "/") for source in sources]

    return fallback


def _evidences_for_run(
    run_dir: Path,
    question_id: int,
    data_dir: str,
    fallback_by_id: Dict[int, List[str]],
) -> List[str]:
    evidences = []
    evidences.extend(_evidences_from_state(run_dir, data_dir))
    evidences.extend(_evidences_from_step_metadata(run_dir, data_dir))

    evidences = _dedupe_preserve_order(evidences)
    if evidences:
        return evidences

    return fallback_by_id.get(question_id, [])


def build_submission(
    runs_folder_name: str,
    output_csv: Optional[str] = None,
    runs_dir: str = "runs",
    data_dir: str = "competition/data_lake",
    questions_file: str = "data/0.Sample_Data.xlsx",
) -> Path:
    """Build a submission CSV from run artifacts.

    Args:
        runs_folder_name: Folder name inside runs, e.g. "competition", or a direct path.
        output_csv: Output CSV path. Defaults to <runs_folder>/submission.csv.
        runs_dir: Base runs directory used when runs_folder_name is not a path.
        data_dir: Data root used to convert absolute evidence paths to relative paths.
        questions_file: Optional fallback source for evidence names.

    Returns:
        Path to the written submission CSV.
    """
    runs_folder = _resolve_runs_folder(runs_folder_name, runs_dir)
    run_dirs = _iter_run_dirs(runs_folder)
    fallback_by_id = _load_question_evidence_fallback(questions_file)

    if output_csv is None:
        output_path = runs_folder / "submission.csv"
    else:
        output_path = Path(output_csv)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in run_dirs:
        question_id = _question_id_from_run_dir(run_dir)
        if question_id is None:
            continue

        rows.append({
            "id": question_id,
            "answer": _answer_from_run(run_dir),
            "evidences": json.dumps(
                _evidences_for_run(run_dir, question_id, data_dir, fallback_by_id),
                ensure_ascii=False,
            ),
        })

    rows.sort(key=lambda row: row["id"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission.csv from run artifacts.")
    parser.add_argument("runs_folder_name", help='Folder inside runs, e.g. "competition".')
    parser.add_argument("--output", help="Output CSV path. Defaults to <runs_folder>/submission.csv.")
    parser.add_argument("--runs-dir", default="agent/runs", help='Base runs directory. Defaults to "runs".')
    parser.add_argument(
        "--data-dir",
        default="data/raw/Data-Lake",
        help='Data root for relative evidence paths. Defaults to "competition/data_lake".',
    )
    parser.add_argument(
        "--questions-file",
        default="data/0.Sample_Data.xlsx",
        help="Questions XLSX/CSV used as fallback for evidences.",
    )
    args = parser.parse_args()

    output_path = build_submission(
        runs_folder_name=args.runs_folder_name,
        output_csv=args.output,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        questions_file=args.questions_file,
    )
    print(f"Wrote submission to {output_path}")


if __name__ == "__main__":
    main()
