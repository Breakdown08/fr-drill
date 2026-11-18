"""Builders for the answer-eval tests: grades, rows, runs."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from fr_drill import answers, cli
from fr_drill.answers import AnswerRow, Grade, Run, Summary


def grade(
    score: float = 1.0,
    passed: bool = True,
    forbidden: list[str] | None = None,
    refused: bool = False,
    cited: list[str] | None = None,
    cite_in_context: float | None = None,
    cite_expected: bool = False,
) -> Grade:
    return Grade(
        score=score,
        passed=passed,
        facts_hit=0,
        facts_total=0,
        missed=[],
        forbidden=forbidden or [],
        refused=refused,
        cited=cited or [],
        cite_in_context=cite_in_context,
        cite_expected=cite_expected,
    )


def row(
    id: str,
    g: Grade,
    expect: str = "answer",
    answer: str = "ответ",
    retrieved: list[str] | None = None,
    retrieve_s: float = 0.5,
    generate_s: float = 1.0,
    stats: dict[str, float] | None = None,
) -> AnswerRow:
    return AnswerRow(
        id=id,
        q="вопрос " + id,
        expect=expect,
        retrieved=retrieved if retrieved is not None else ["1"],
        answer=answer,
        retrieve_s=retrieve_s,
        generate_s=generate_s,
        stats=stats or {},
        grade=g,
    )


def scored(id: str, score: float, expect: str = "answer", retrieved: list[str] | None = None) -> AnswerRow:
    return row(id, grade(score=score, passed=score == 1.0), expect=expect, retrieved=retrieved)


def run_of(rows: list[AnswerRow], model: str = "m1", backend: str = "ollama") -> Run:
    return Run(
        schema=answers.SCHEMA,
        created="2026-11-18T12:30:05+00:00",
        backend=backend,
        model=model,
        options={"num_ctx": 8192.0, "temperature": 0.0, "seed": 0.0},
        k=5,
        index_points=7,
        items_file="data/rag/answers.json",
        items_sha256="0" * 64,
        platform="Linux-test",
        cpus=8,
        summary=summary_of(),
        rows=rows,
    )


def summary_of(
    n_answer: int = 3,
    n_no_answer: int = 1,
    answer_score: float | None = 0.5,
    pass_rate: float | None = 1 / 3,
    wrong_rate: float | None = 1 / 3,
    false_refusal_rate: float | None = 1 / 3,
    no_answer_accuracy: float | None = 1.0,
    cited_rate: float | None = 2 / 3,
    cite_precision: float | None = 0.75,
    cite_expected_rate: float | None = 1 / 3,
) -> Summary:
    return Summary(
        n_answer=n_answer,
        n_no_answer=n_no_answer,
        answer_score=answer_score,
        pass_rate=pass_rate,
        wrong_rate=wrong_rate,
        false_refusal_rate=false_refusal_rate,
        no_answer_accuracy=no_answer_accuracy,
        cited_rate=cited_rate,
        cite_precision=cite_precision,
        cite_expected_rate=cite_expected_rate,
        generate_median_s=2.5,
        generate_mean_s=4.0,
        generate_max_s=10.0,
        generate_total_s=16.0,
        retrieve_total_s=2.0,
    )


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class Result:
    def __init__(self, code: object, out: str, err: str) -> None:
        self.code: object = code
        self.out: str = out
        self.err: str = err


def main_in_process(capsys: pytest.CaptureFixture[str], *argv: str) -> Result:
    """cli.main in this process: the exit code, stdout and stderr."""
    code: object = None
    try:
        cli.main(list(argv))
    except SystemExit as e:
        code = e.code
    captured: Any = capsys.readouterr()
    return Result(code, captured.out, captured.err)


def save_run(run: Run, path: Path) -> Path:
    """A run file written by hand, independent of answers.write_run."""
    path.write_text(json.dumps(dataclasses.asdict(run), ensure_ascii=False, indent=1), encoding="utf-8")
    return path
