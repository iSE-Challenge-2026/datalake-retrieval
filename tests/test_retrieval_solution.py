from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import tempfile
import unittest

from src.retrieval.pipeline import RetrievalPipelineConfig
from src.retrieval.solution import RetrievalSolutionConfig, run_retrieval_solution


class RetrievalSolutionTests(unittest.TestCase):
    def test_solution_exports_absolute_paths_from_local_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "raw" / "Data-Lake"
            data_root.mkdir(parents=True)
            (data_root / "docs").mkdir()
            (data_root / "docs" / "answer.txt").write_text("answer", encoding="utf-8")

            questions = root / "questions.csv"
            with questions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question"])
                writer.writeheader()
                writer.writerow({"id": "q1", "question": "Where is the answer?"})

            output = root / "output" / "sources.json"
            payload = run_retrieval_solution(
                RetrievalSolutionConfig(
                    questions=questions,
                    output=output,
                    pipeline=RetrievalPipelineConfig(
                        data_root=data_root,
                        retrieval_mode="lexical",
                        top_k=1,
                    ),
                ),
                project_root=root,
                pipeline_factory=FakePipeline,
            )

            expected = str((data_root / "docs" / "answer.txt").resolve())
            self.assertEqual(payload["results"][0]["source_files"], [expected])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["results"][0]["source_files"], [expected])


class FakePipeline:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def retrieve(self, question: str) -> list[FakeResult]:
        return [FakeResult(source_path="docs/answer.txt")]


@dataclass
class FakeResult:
    source_path: str
    score: float = 1.0
    source_kind: str = "text"
    reasons: list[str] = field(default_factory=lambda: ["fake"])
    chunks: list[object] = field(default_factory=list)


if __name__ == "__main__":
    unittest.main()
