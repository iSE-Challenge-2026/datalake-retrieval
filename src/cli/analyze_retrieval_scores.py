"""Analyze score separation between expected and non-expected retrieval hits."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import statistics
import sys


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze retrieval score separation from benchmark results.")
    parser.add_argument("results", nargs="+", type=Path, help="Benchmark results.json files.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    reports = [analyze_results(path) for path in args.results]
    payload = {"runs": reports}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


def analyze_results(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    file_correct: list[float] = []
    file_wrong: list[float] = []
    chunk_correct: list[float] = []
    chunk_wrong: list[float] = []
    gaps: list[float] = []
    per_question: list[dict] = []

    for row in payload.get("results", []):
        expected = set(row.get("expanded_expected") or row.get("expected") or [])
        if not expected:
            continue

        correct_scores: list[float] = []
        wrong_scores: list[float] = []
        top_results = row.get("top_results") or []
        for result in top_results:
            source_path = result.get("source_path")
            is_correct = source_path in expected
            score = float(result.get("score") or 0.0)
            if is_correct:
                correct_scores.append(score)
                file_correct.append(score)
            else:
                wrong_scores.append(score)
                file_wrong.append(score)

            for chunk in result.get("chunks") or []:
                chunk_score = float(chunk.get("score") or 0.0)
                if is_correct:
                    chunk_correct.append(chunk_score)
                else:
                    chunk_wrong.append(chunk_score)

        max_correct = max(correct_scores) if correct_scores else None
        max_wrong = max(wrong_scores) if wrong_scores else None
        gap = None
        if max_correct is not None and max_wrong is not None:
            gap = max_correct - max_wrong
            gaps.append(gap)

        per_question.append(
            {
                "id": row.get("id"),
                "recall": round(float(row.get("recall") or 0.0), 4),
                "precision": round(float(row.get("precision") or 0.0), 4),
                "correct_file_hits_in_saved_top_results": len(correct_scores),
                "wrong_file_hits_in_saved_top_results": len(wrong_scores),
                "max_correct_file_score": _round_or_none(max_correct),
                "max_wrong_file_score": _round_or_none(max_wrong),
                "score_gap": _round_or_none(gap),
                "top_is_correct": bool(top_results and top_results[0].get("source_path") in expected),
                "top_source": top_results[0].get("source_path") if top_results else None,
            }
        )

    return {
        "path": path.as_posix(),
        "summary": payload.get("summary", {}),
        "note": "Analysis uses saved top_results only; retrieve_eval currently stores top 10 sources and top 2 chunks per source.",
        "file_correct": _describe(file_correct),
        "file_wrong": _describe(file_wrong),
        "chunk_correct": _describe(chunk_correct),
        "chunk_wrong": _describe(chunk_wrong),
        "gap_max_correct_minus_max_wrong": _describe(gaps),
        "gap_positive_count": sum(1 for value in gaps if value > 0),
        "gap_count": len(gaps),
        "per_question": per_question,
    }


def _describe(values: list[float]) -> dict | None:
    if not values:
        return None
    sorted_values = sorted(values)

    def percentile(p: float) -> float:
        index = round((len(sorted_values) - 1) * p)
        return sorted_values[index]

    return {
        "n": len(sorted_values),
        "min": round(sorted_values[0], 6),
        "p10": round(percentile(0.10), 6),
        "p25": round(percentile(0.25), 6),
        "median": round(statistics.median(sorted_values), 6),
        "mean": round(statistics.mean(sorted_values), 6),
        "p75": round(percentile(0.75), 6),
        "p90": round(percentile(0.90), 6),
        "max": round(sorted_values[-1], 6),
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


if __name__ == "__main__":
    main()
