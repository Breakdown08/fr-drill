"""Textbook helper: the commands through cli.main, with fake backends (spec checks 37-42)."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from rag_fakes import FakeEmbedder, FakeGenerator, make_pdf, pdf3

import fr_drill
from fr_drill import cli, rag

IDX_TEXT: str = "Intro bravo charlie"
CITE1: str = "book, p. 1, 1 Intro"


class Result:
    def __init__(self, code: object, out: str, err: str) -> None:
        self.code: object = code
        self.out: str = out
        self.err: str = err


@pytest.fixture
def gen(monkeypatch: pytest.MonkeyPatch) -> FakeGenerator:
    g: FakeGenerator = FakeGenerator("ОТВЕТ ФЕЙКА (с. 1)")
    monkeypatch.setattr(rag, "make_embedder", lambda cache_dir: FakeEmbedder(64))
    monkeypatch.setattr(rag, "make_generator", lambda backend, model=None: g)
    return g


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> Result:
    code: object = None
    try:
        cli.main(list(argv))
    except SystemExit as e:
        code = e.code
    captured: Any = capsys.readouterr()
    return Result(code, captured.out, captured.err)


@pytest.fixture
def book(tmp_path: Path) -> Path:
    return pdf3(tmp_path / "book.pdf", nested=False)


@pytest.fixture
def indexed(book: Path, tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]) -> Path:
    index: Path = tmp_path / "idx"
    res: Result = run(capsys, "index", str(book), "--chapter", "1", "--index", str(index))
    assert res.code == 0, res.err
    return index


def test_index_then_search_json(
    book: Path, tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    index: Path = tmp_path / "idx"
    res: Result = run(capsys, "index", str(book), "--chapter", "1", "--index", str(index))
    assert res.code == 0
    assert res.out.strip() == "indexed 2 chunks from 2 pages of book"
    assert any((index / "qdrant").rglob("*"))
    res = run(capsys, "search", "Intro delta echo", "--json", "--index", str(index))
    assert res.code == 0
    rows: list[dict[str, Any]] = json.loads(res.out)
    assert {"score", "book", "page", "pdf_page", "chapter", "chunk", "text"} == set(rows[0])
    assert rows[0]["book"] == "book"
    assert rows[0]["page"] == "2"
    assert rows[0]["pdf_page"] == 3
    assert rows[0]["chapter"] == "1 Intro"
    assert rows[0]["chunk"] == 0
    assert "delta" in rows[0]["text"]
    assert isinstance(rows[0]["score"], float)
    assert len(rows) <= 5


def test_search_k_limits_the_result(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res: Result = run(capsys, "search", IDX_TEXT, "--k", "1", "--json", "--index", str(indexed))
    assert len(json.loads(res.out)) == 1


def test_search_text_format(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res: Result = run(capsys, "search", IDX_TEXT, "--index", str(indexed))
    assert res.code == 0
    lines: list[str] = res.out.splitlines()
    assert re.fullmatch(r"\[1\] -?\d+\.\d{3}  book, p\. 1, 1 Intro", lines[0])
    assert lines[1].startswith("  ") and "bravo" in lines[1]
    assert re.match(r"\[2\] ", lines[2])


def test_search_text_is_cut_at_300_chars(
    tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    long: str = " ".join(f"word{i}" for i in range(60))
    pdf: Path = make_pdf(tmp_path / "long.pdf", [long])
    index: Path = tmp_path / "idx"
    assert run(capsys, "index", str(pdf), "--index", str(index)).code == 0
    res: Result = run(capsys, "search", "word1", "--index", str(index))
    assert res.out.splitlines()[1] == "  " + long[:300]


def test_ask_prints_the_answer_and_sources(
    indexed: Path, gen: FakeGenerator, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    got: list[tuple[str, str | None]] = []

    def make(backend: str, model: str | None = None) -> FakeGenerator:
        got.append((backend, model))
        return gen

    monkeypatch.setattr(rag, "make_generator", make)
    res: Result = run(capsys, "ask", "Intro bravo", "--backend", "claude", "--index", str(indexed))
    assert res.code == 0
    assert got == [("claude", None)]
    lines: list[str] = res.out.splitlines()
    assert lines[0] == "ОТВЕТ ФЕЙКА (с. 1)"
    assert lines[1] == ""
    assert lines[2] == "Sources:"
    assert lines[3] == f"[1] {CITE1}"
    assert len(gen.prompts) == 1 and "Intro bravo" in gen.prompts[0]


def test_ask_defaults_to_ollama_and_passes_the_model(
    indexed: Path, gen: FakeGenerator, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    got: list[tuple[str, str | None]] = []

    def make(backend: str, model: str | None = None) -> FakeGenerator:
        got.append((backend, model))
        return gen

    monkeypatch.setattr(rag, "make_generator", make)
    assert run(capsys, "ask", "q", "--index", str(indexed)).code == 0
    assert run(capsys, "ask", "q", "--model", "qwen", "--index", str(indexed)).code == 0
    assert got == [("ollama", None), ("ollama", "qwen")]


def test_ask_on_an_empty_index_prints_only_no_answer(
    tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    from qdrant_client.models import Distance, VectorParams

    index: Path = tmp_path / "idx"
    client: Any = rag.open_client(index)
    client.create_collection(rag.COLLECTION, VectorParams(size=64, distance=Distance.COSINE))
    client.close()
    res: Result = run(capsys, "ask", "q", "--index", str(index))
    assert res.code == 0
    assert res.out.strip() == rag.NO_ANSWER
    assert gen.prompts == []


def test_search_without_an_index_is_a_clean_error(
    tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = run(capsys, "search", "q", "--index", str(tmp_path / "empty"))
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")
    assert "fr-drill index" in res.err
    assert "Traceback" not in res.err


def test_index_of_a_pdf_without_text_fails(
    tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf: Path = make_pdf(tmp_path / "blank.pdf", ["", ""])
    res: Result = run(capsys, "index", str(pdf), "--index", str(tmp_path / "idx"))
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")
    assert "Traceback" not in res.err


def test_index_of_unknown_pages_fails(
    book: Path, tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = run(capsys, "index", str(book), "--pages", "99-100", "--index", str(tmp_path / "idx"))
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")


def test_index_of_a_missing_file_fails(
    tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = run(capsys, "index", str(tmp_path / "no.pdf"), "--index", str(tmp_path / "idx"))
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")
    assert str(tmp_path / "no.pdf") in res.err
    assert "Traceback" not in res.err


def test_index_by_printed_pages(
    book: Path, tmp_path: Path, gen: FakeGenerator, capsys: pytest.CaptureFixture[str]
) -> None:
    res: Result = run(capsys, "index", str(book), "--pages", "2", "--index", str(tmp_path / "idx"))
    assert res.code == 0
    assert res.out.strip() == "indexed 1 chunks from 1 pages of book"


def test_eval_json(indexed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file: Path = tmp_path / "eval.json"
    file.write_text(json.dumps({"items": [{"q": "Intro bravo", "pages": ["1"]}]}), encoding="utf-8")
    res: Result = run(capsys, "eval", str(file), "--json", "--index", str(indexed))
    assert res.code == 0
    report: dict[str, Any] = json.loads(res.out)
    assert set(report) == {"k", "recall", "hit_rate", "rows"}
    assert report["k"] == 5
    assert report["recall"] == 1.0 and report["hit_rate"] == 1.0
    assert set(report["rows"][0]) == {"q", "expected", "retrieved", "recall", "hit"}


def test_eval_text(indexed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file: Path = tmp_path / "eval.json"
    file.write_text(json.dumps({"items": [{"q": "Intro bravo", "pages": ["1"]}]}), encoding="utf-8")
    res: Result = run(capsys, "eval", str(file), "--index", str(indexed))
    assert res.code == 0
    lines: list[str] = res.out.splitlines()
    assert lines[0].startswith("1.00  Intro bravo  expected ['1'] got [")
    assert lines[-1] == "recall@5 = 1.000  hit@5 = 1.000  (n=1)"


def test_eval_without_the_file(indexed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res: Result = run(capsys, "eval", str(tmp_path / "nope.json"), "--index", str(indexed))
    assert res.code == 2
    assert res.err.startswith("fr-drill: ")


def test_missing_helper_dependencies_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delattr(fr_drill, "rag", raising=False)
    monkeypatch.setitem(sys.modules, "fr_drill.rag", None)
    res: Result = run(capsys, "search", "q")
    assert res.code == 2
    assert "fr-drill: install the helper with: uv sync --extra rag" in res.err
