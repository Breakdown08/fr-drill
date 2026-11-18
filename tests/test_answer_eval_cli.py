"""Answer eval: the answer-eval and answer-report commands through cli.main (spec checks 22-27)."""

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
from answers_builders import Result, grade, main_in_process, row, run_of, save_run, write_text
from rag_fakes import FakeEmbedder, FakeGenerator, pdf3
from util import run_cli, run_py

from fr_drill import answers, rag
from fr_drill.answers import AnswerRow, Run

ITEMS: dict[str, list[dict[str, object]]] = {
    "items": [
        {"id": "one", "chapter": "1", "q": "Intro bravo", "pages": ["1"], "must": [["ответ"]]},
        {"id": "two", "chapter": "1", "q": "Intro delta echo", "pages": ["2"], "must": [["нету"]]},
        {"id": "out", "expect": "no_answer", "q": "Intro charlie"},
    ]
}


class Factory:
    """A make_generator stand-in that remembers how it was called."""

    def __init__(self, gen: FakeGenerator) -> None:
        self.gen: FakeGenerator = gen
        self.calls: list[tuple[str, str | None, bool]] = []

    def __call__(self, backend: str, model: str | None = None, deterministic: bool = False) -> FakeGenerator:
        self.calls.append((backend, model, deterministic))
        return self.gen


class Failing(FakeGenerator):
    """Answers the first prompt, raises RagError on the second."""

    def generate(self, prompt: str) -> str:
        if self.prompts:
            raise rag.RagError("backend down")
        return super().generate(prompt)


@pytest.fixture
def items_file(tmp_path: Path) -> Path:
    return write_text(tmp_path / "items.json", json.dumps(ITEMS, ensure_ascii=False))


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> Factory:
    f: Factory = Factory(FakeGenerator("ОТВЕТ (с. 1)"))
    monkeypatch.setattr(rag, "make_embedder", lambda cache_dir: FakeEmbedder(64))
    monkeypatch.setattr(rag, "make_generator", f)
    return f


@pytest.fixture
def indexed(tmp_path: Path, factory: Factory, capsys: pytest.CaptureFixture[str]) -> Path:
    index: Path = tmp_path / "idx"
    book: Path = pdf3(tmp_path / "book.pdf", nested=False)
    res: Result = main_in_process(capsys, "index", str(book), "--chapter", "1", "--index", str(index))
    assert res.code == 0, res.err
    return index


def test_answer_eval_writes_a_run_and_prints_the_summary(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "runs" / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 0, res.err
    run: Run = answers.read_run(out)
    assert run.backend == "ollama"
    assert run.model == rag.OLLAMA_MODEL
    assert factory.calls == [("ollama", None, True)]
    assert [r.id for r in run.rows] == ["one", "two", "out"]
    assert run.index_points == 2
    assert run.k == 5
    assert run.items_sha256 == answers.items_sha256(items_file)
    assert run.items_file == str(items_file)
    assert run.schema == answers.SCHEMA
    assert run.rows[0].answer == "ОТВЕТ (с. 1)"
    assert run.rows[0].grade.score == 1.0
    assert run.rows[2].grade.score == 0.0
    assert res.out.splitlines() == answers.format_summary(run.summary)
    assert len(res.out.splitlines()) == 4
    assert f"wrote {out}" in res.err
    lines: list[str] = [ln for ln in res.err.splitlines() if re.fullmatch(r"\d\.\d\d  \d+\.\d s  \w+", ln)]
    assert [ln.split()[-1] for ln in lines] == ["one", "two", "out"]
    assert lines[0].startswith("1.00  ")


def test_answer_eval_json_matches_the_file(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--json", "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 0, res.err
    printed: dict[str, Any] = json.loads(res.out)
    assert set(printed) == {f.name for f in dataclasses.fields(Run)}
    assert printed == json.loads(out.read_text(encoding="utf-8"))


def test_answer_eval_limit(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--limit", "1", "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 0, res.err
    assert [r.id for r in answers.read_run(out).rows] == ["one"]


def test_answer_eval_limit_zero_is_an_error(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--limit", "0", "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 2
    assert "fr-drill: " in res.err
    assert "usage:" not in res.err
    assert not out.exists()


def test_answer_eval_claude_with_a_model(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys,
        "answer-eval",
        str(items_file),
        "--backend",
        "claude",
        "--model",
        "sonnet",
        "--out",
        str(out),
        "--index",
        str(indexed),
    )
    assert res.code == 0, res.err
    assert factory.calls[0][:2] == ("claude", "sonnet")
    run: Run = answers.read_run(out)
    assert run.backend == "claude"
    assert run.model == "sonnet"
    assert run.options == {}


def test_answer_eval_claude_default_model(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys,
        "answer-eval",
        str(items_file),
        "--backend",
        "claude",
        "--out",
        str(out),
        "--index",
        str(indexed),
    )
    assert res.code == 0, res.err
    assert answers.read_run(out).model == rag.CLAUDE_MODEL


def test_answer_eval_k_is_recorded(
    indexed: Path, items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--k", "1", "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 0, res.err
    run: Run = answers.read_run(out)
    assert run.k == 1
    assert all(len(r.retrieved) == 1 for r in run.rows)


def test_answer_eval_without_an_index(
    items_file: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--out", str(out), "--index", str(tmp_path / "empty")
    )
    assert res.code == 2
    assert "fr-drill index" in res.err
    assert not out.exists()


def test_answer_eval_backend_error_leaves_no_file(
    indexed: Path,
    items_file: Path,
    factory: Factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(rag, "make_generator", Factory(Failing("ОТВЕТ")))
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(items_file), "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 2
    assert "fr-drill: " in res.err
    assert "backend down" in res.err
    assert "Traceback" not in res.err
    assert not out.exists()


def test_answer_eval_bad_items_file(
    indexed: Path, factory: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out: Path = tmp_path / "run.json"
    res: Result = main_in_process(
        capsys, "answer-eval", str(tmp_path / "nope.json"), "--out", str(out), "--index", str(indexed)
    )
    assert res.code == 2
    assert "fr-drill: " in res.err
    assert "nope.json" in res.err
    assert "usage:" not in res.err
    assert not out.exists()


def test_answer_eval_unknown_backend_is_an_argparse_error() -> None:
    done: Any = run_cli("answer-eval", "--backend", "gpt")
    assert done.returncode == 2
    assert "argument --backend" in done.stderr
    assert "Traceback" not in done.stderr


# answer-report


@pytest.fixture
def run_a(tmp_path: Path) -> Path:
    rows: list[AnswerRow] = [
        row("one", grade(0.0, False), answer="ОТВЕТ (с. 1)", retrieved=["1"]),
        row("two", grade(0.5, False), answer="нету", retrieved=["2"]),
        row("out", grade(1.0, True), expect="no_answer", answer=answers.NO_ANSWER, retrieved=["1"]),
    ]
    path: Path = tmp_path / "a.json"
    save_run(run_of(rows, model="alpha"), path)
    return path


@pytest.fixture
def run_b(tmp_path: Path) -> Path:
    rows: list[AnswerRow] = [
        row("one", grade(1.0, True), answer="ОТВЕТ (с. 1)", retrieved=["1"]),
        row("two", grade(0.0, False), answer="пусто", retrieved=["2"]),
        row("out", grade(0.0, False), expect="no_answer", answer="что-то", retrieved=["1"]),
    ]
    path: Path = tmp_path / "b.json"
    save_run(run_of(rows, model="beta"), path)
    return path


def test_report_one_run(run_a: Path, items_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res: Result = main_in_process(capsys, "answer-report", str(run_a), "--items", str(items_file))
    assert res.code == 0, res.err
    lines: list[str] = res.out.splitlines()
    assert len(lines) == 5
    assert lines[0] == f"ollama alpha  2026-11-18T12:30:05+00:00  k=5  index=7  {run_a}"
    assert lines[1].startswith("answers   n=2")
    assert not any(ln.startswith("B - A") for ln in lines)


def test_report_rescores_by_the_current_items(
    run_a: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(capsys, "answer-report", str(run_a), "--items", str(items_file), "--json")
    assert res.code == 0, res.err
    got: dict[str, Any] = json.loads(res.out)
    summary: dict[str, Any] = got["runs"][0]["summary"]
    # the stored grades said 0.0 and 0.5; the answers cover both facts, so the current items give 1.0
    assert summary["answer_score"] == 1.0
    assert summary["no_answer_accuracy"] == 1.0


def test_report_two_runs_ends_with_the_comparison(
    run_a: Path, run_b: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(capsys, "answer-report", str(run_a), str(run_b), "--items", str(items_file))
    assert res.code == 0, res.err
    lines: list[str] = res.out.splitlines()
    assert len(lines) == 13
    assert lines[5] == "" and lines[11] == ""
    assert lines[0].startswith("ollama alpha  ") and lines[6].startswith("ollama beta  ")
    assert lines[-1].startswith("B - A  score ")
    assert re.fullmatch(
        r"B - A  score [+-]\d\.\d{3}  95% CI \[[+-]\d\.\d{3}, [+-]\d\.\d{3}\]  sd \d\.\d{3}"
        r"  better \d+  worse \d+  same \d+  \(n=2\)",
        lines[-1],
    )


def test_report_comparison_numbers(
    run_a: Path, run_b: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(
        capsys, "answer-report", str(run_a), str(run_b), "--items", str(items_file), "--json"
    )
    assert res.code == 0, res.err
    got: dict[str, Any] = json.loads(res.out)
    assert [r["model"] for r in got["runs"]] == ["alpha", "beta"]
    assert set(got["runs"][0]) == {
        "file",
        "backend",
        "model",
        "created",
        "k",
        "index_points",
        "skipped",
        "summary",
    }
    assert got["runs"][0]["file"] == str(run_a)
    assert got["runs"][0]["skipped"] == 0
    cmp: dict[str, Any] = got["comparison"]
    # rescored: A has one=1.0 (ответ found), two=1.0? no: "нету" in "нету" -> 1.0; B has one=1.0, two=0.0
    assert cmp["n"] == 2
    assert cmp["mean_a"] == 1.0
    assert cmp["mean_b"] == 0.5
    assert cmp["diff"] == -0.5
    assert (cmp["better"], cmp["worse"], cmp["same"]) == (0, 1, 1)


def test_report_json_has_null_comparison_for_one_run(
    run_a: Path, items_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(capsys, "answer-report", str(run_a), "--items", str(items_file), "--json")
    got: dict[str, Any] = json.loads(res.out)
    assert set(got) == {"runs", "comparison"}
    assert got["comparison"] is None
    assert len(got["runs"]) == 1


def test_report_warns_about_rows_without_an_item(
    run_a: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short: dict[str, list[dict[str, object]]] = {"items": [ITEMS["items"][0], ITEMS["items"][2]]}
    items: Path = write_text(tmp_path / "short.json", json.dumps(short))
    res: Result = main_in_process(capsys, "answer-report", str(run_a), "--items", str(items))
    assert res.code == 0, res.err
    assert f"fr-drill: 1 rows of {run_a} have no item in {items}" in res.err


def test_report_warns_when_retrieval_differs(
    run_a: Path, items_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows: list[AnswerRow] = [
        row("one", grade(1.0), answer="ОТВЕТ", retrieved=["1"]),
        row("two", grade(1.0), answer="нету", retrieved=["9"]),
    ]
    other: Path = tmp_path / "other.json"
    save_run(run_of(rows, model="gamma"), other)
    res: Result = main_in_process(capsys, "answer-report", str(run_a), str(other), "--items", str(items_file))
    assert res.code == 0, res.err
    assert "fr-drill: retrieval differs on 1 of 2 paired items" in res.err


def test_report_without_the_run_file(
    items_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(
        capsys, "answer-report", str(tmp_path / "nope.json"), "--items", str(items_file)
    )
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")
    assert "nope.json" in res.err


def test_report_without_the_items_file(
    run_a: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = main_in_process(capsys, "answer-report", str(run_a), "--items", str(tmp_path / "no.json"))
    assert res.code == 2
    assert "fr-drill: " in res.err
    assert "no.json" in res.err
    assert "usage:" not in res.err


def test_report_with_no_common_answer_rows_is_an_error(
    run_a: Path, items_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other: Path = tmp_path / "other.json"
    save_run(run_of([row("out", grade(1.0), expect="no_answer")], model="d"), other)
    res: Result = main_in_process(capsys, "answer-report", str(run_a), str(other), "--items", str(items_file))
    assert res.code == 2
    assert "fr-drill: " in res.err
    assert "usage:" not in res.err


# lazy imports

HELP_CODE: str = (
    "import sys\n"
    "import fr_drill.cli as cli\n"
    "for argv in (['answer-eval', '--help'], ['answer-report', '--help']):\n"
    "    try:\n"
    "        cli.main(argv)\n"
    "    except SystemExit as e:\n"
    "        assert e.code == 0, argv\n"
    "    else:\n"
    "        raise AssertionError(argv)\n"
    "    bad = [m for m in ('qdrant_client', 'fastembed', 'fr_drill.rag') if m in sys.modules]\n"
    "    assert bad == [], (argv, bad)\n"
    "print('ok')\n"
)

REPORT_CODE: str = (
    "import sys\n"
    "import fr_drill.cli as cli\n"
    "try:\n"
    "    cli.main(['answer-report', sys.argv[1], '--items', sys.argv[2]])\n"
    "except SystemExit as e:\n"
    "    assert e.code == 0, e.code\n"
    "else:\n"
    "    raise AssertionError('no exit')\n"
    "bad = [m for m in ('qdrant_client', 'fastembed', 'fr_drill.rag') if m in sys.modules]\n"
    "assert bad == [], bad\n"
    "print('ok')\n"
)


def test_help_of_the_new_commands_does_not_load_the_helper() -> None:
    assert run_py(HELP_CODE).splitlines()[-1] == "ok"


def test_answer_report_works_without_the_helper(run_a: Path, items_file: Path) -> None:
    assert run_py(REPORT_CODE, str(run_a), str(items_file)).splitlines()[-1] == "ok"


def test_answer_eval_help_lists_its_options() -> None:
    done: Any = run_cli("answer-eval", "--help")
    assert done.returncode == 0
    flag: str
    for flag in ("--backend", "--model", "--k", "--limit", "--out", "--json", "--index", "--qdrant-url"):
        assert flag in done.stdout, flag


def test_answer_report_help_lists_its_options() -> None:
    done: Any = run_cli("answer-report", "--help")
    assert done.returncode == 0
    assert "--items" in done.stdout and "--json" in done.stdout
