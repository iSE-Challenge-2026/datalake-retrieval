import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "0.Sample_Data.xlsx"
DEFAULT_RETRIEVAL_JSON = PROJECT_ROOT / "data" / "output" / "Data-Lake" / "retrieval_source_paths.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "agent" / "config" / "config.yaml"

def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_questions(questions_path: Path) -> list[dict]:
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    if questions_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(questions_path)
    else:
        frame = pd.read_csv(questions_path)

    id_column = next((column for column in ("STT", "id", "ID") if column in frame.columns), None)
    question_column = next((column for column in ("Question", "question") if column in frame.columns), None)
    if id_column is None or question_column is None:
        raise ValueError("Questions file must contain an id column (STT/id/ID) and a question column (Question/question).")

    rows: list[dict] = []
    for _, row in frame.iterrows():
        question_id = row[id_column]
        question_text = row[question_column]
        if pd.isna(question_id) or pd.isna(question_text):
            continue

        if isinstance(question_id, float) and question_id.is_integer():
            normalized_id = str(int(question_id))
        else:
            normalized_id = str(question_id).strip()

        rows.append({
            "id": normalized_id,
            "question": str(question_text).strip(),
        })

    return rows


def _load_retrieval_source_paths(retrieval_json_path: Path) -> dict[str, list[str]]:
    if not retrieval_json_path.exists():
        raise FileNotFoundError(f"Retrieval JSON not found: {retrieval_json_path}")

    payload = json.loads(retrieval_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    elif isinstance(payload, dict):
        results = [{"id": key, "source_files": value} for key, value in payload.items()]
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError(f"Unsupported retrieval JSON structure in {retrieval_json_path}")

    source_paths_by_id: dict[str, list[str]] = {}
    for row in results:
        question_id = str(row.get("id", "")).strip()
        if not question_id:
            continue

        source_files = row.get("source_files", []) or []
        normalized_files = []
        for source_file in source_files:
            source_path = Path(str(source_file))
            normalized_files.append(str(source_path.resolve()) if source_path.exists() else str(source_path))

        source_paths_by_id[question_id] = normalized_files

    return source_paths_by_id


def main(args):
    """CLI interface with resume and edit capabilities."""
    from agent import Agent, DSConfig

    questions_path = _resolve_path(args.questions)
    retrieval_json_path = _resolve_path(args.retrieval_json)
    questions = _load_questions(questions_path)
    retrieval_source_paths = _load_retrieval_source_paths(retrieval_json_path)

    try:
        config_defaults = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        config_defaults = {}

    for work in questions[3:5]:
        run_id = f"competition/question_{work['id']}"
        data_files = retrieval_source_paths.get(work["id"], [])
        print(f"Question {work['id']}: {len(data_files)} files found")
        if not data_files:
            print(f"[WARN] No retrieved files found for question id {work['id']}")

        config = DSConfig(
            **config_defaults,
            run_id=run_id,
            query=work["question"],
            data_files=data_files,
            data_dir=str((PROJECT_ROOT / "data" / "raw" / "Data-Lake").resolve()),
        )
        if not config.model_name:
            parser.error("Model name must be specified via config file.")

        agent = Agent(config)

        if config.edit_last and config.run_id:
            agent.controller.edit_last_step_code()
            continue

        print("Files found for analysis:", [Path(file).name for file in data_files])

        result = agent.run_pipeline(config.query, data_files)

        print(f"\n{'=' * 60}")
        print(f"RUN COMPLETED: {result['run_id']}")
        print(f"OUTPUT: {result['output_file']}")
        print(f"FINAL RESULT:{result['final_result']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Science Agent")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--interactive", action="store_true", help="Pause between steps")
    parser.add_argument("--edit-last", action="store_true", help="Edit last generated code")
    parser.add_argument("--max-rounds", type=int, help="Max refinement rounds")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Questions file with ids and questions.")
    parser.add_argument(
        "--retrieval-json",
        type=Path,
        default=DEFAULT_RETRIEVAL_JSON,
        help="Retrieved source paths JSON keyed by question id.",
    )
    args = parser.parse_args()

    main(args)
