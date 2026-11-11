"""Package paths and the jsonl helpers (spec checks 8-11)."""

import json
from pathlib import Path
from typing import Any

import pytest
from util import ROOT, clean_env, run_py

from fr_drill import append_jsonl, read_jsonl

SHOW: str = (
    "import json, fr_drill; "
    "from fr_drill import drill; "
    "print(json.dumps({k: str(getattr(fr_drill, k)) for k in ('ROOT', 'DATA', 'PROGRESS', 'VOCAB')} "
    "| {'drill.ATTEMPTS': str(drill.ATTEMPTS)}))"
)

TABLE: dict[str, str] = {
    "drill.VERBS": "data/drills/verbs-er.json",
    "drill.ATTEMPTS": "progress/drill.jsonl",
    "drill.PAGE": "src/fr_drill/drill.html",
    "exam.PAGE": "src/fr_drill/exam.html",
    "vocab.WORDS": "data/drills/vocab.json",
    "vocab.ATTEMPTS": "progress/vocab.jsonl",
    "numbers.WORDS": "data/drills/numbers.json",
    "numbers.ATTEMPTS": "progress/numbers.jsonl",
    "negation.CONTEXTS": "data/drills/negation-er.json",
    "negation.ATTEMPTS": "progress/negation.jsonl",
    "etre_avoir.CONTEXTS": "data/drills/etre-avoir.json",
    "etre_avoir.ATTEMPTS": "progress/etre-avoir.jsonl",
    "sentences.BANK": "data/drills/sentences.json",
    "sentences.ATTEMPTS": "progress/sentences.jsonl",
    "prepositions.BANK": "data/drills/a-de.json",
    "prepositions.VERBS": "data/drills/a-de-verbs.json",
    "prepositions.ATTEMPTS": "progress/prepositions.jsonl",
    "pronouns.BANK": "data/drills/pronouns.json",
    "pronouns.ATTEMPTS": "progress/pronouns.jsonl",
    "gender.BANK": "data/drills/gender.json",
    "gender.ATTEMPTS": "progress/gender.jsonl",
    "articles.BANK": "data/drills/articles.json",
    "articles.ATTEMPTS": "progress/articles.jsonl",
    "bank.BANK": "data/bank",
    "bank.TOPICS": "data/syllabus/topics.json",
    "exam.ATTEMPTS": "progress/exam.jsonl",
    "exam.GRADES": "progress/exam-grades.jsonl",
    "syllabus.SYLLABUS": "data/syllabus",
}

DUMP: str = """
import importlib, json, sys
out = {}
for name in json.loads(sys.argv[1]):
    mod, attr = name.split(".")
    try:
        out[name] = str(getattr(importlib.import_module("fr_drill." + mod), attr))
    except (AttributeError, ImportError):
        out[name] = None
print(json.dumps(out))
"""


def shown(env: dict[str, str], cwd: Path | None = None) -> dict[str, str]:
    result: dict[str, str] = json.loads(run_py(SHOW, env=env, cwd=cwd))
    return result


def test_defaults_without_variables() -> None:
    got: dict[str, str] = shown(clean_env())
    assert Path(got["ROOT"]) == ROOT
    assert (Path(got["ROOT"]) / "pyproject.toml").is_file()
    assert Path(got["DATA"]) == ROOT / "data"
    assert Path(got["PROGRESS"]) == ROOT / "progress"
    assert Path(got["VOCAB"]) == ROOT / "data" / "drills" / "vocab.json"


def test_variables_override(tmp_path: Path) -> None:
    got: dict[str, str] = shown(
        clean_env({"FR_DRILL_PROGRESS": str(tmp_path), "FR_DRILL_VOCAB": str(tmp_path / "v.json")})
    )
    assert Path(got["PROGRESS"]) == tmp_path
    assert Path(got["VOCAB"]) == tmp_path / "v.json"
    assert Path(got["drill.ATTEMPTS"]) == tmp_path / "drill.jsonl"
    assert Path(got["ROOT"]) == ROOT


def test_empty_variables_mean_the_default() -> None:
    got: dict[str, str] = shown(clean_env({"FR_DRILL_PROGRESS": "", "FR_DRILL_VOCAB": ""}))
    assert Path(got["PROGRESS"]) == ROOT / "progress"
    assert Path(got["VOCAB"]) == ROOT / "data" / "drills" / "vocab.json"


def test_relative_variable_is_taken_from_the_current_directory(tmp_path: Path) -> None:
    got: dict[str, str] = shown(
        clean_env({"FR_DRILL_PROGRESS": "hist", "FR_DRILL_VOCAB": "v.json"}), cwd=tmp_path
    )
    assert (tmp_path / got["PROGRESS"]).resolve() == (tmp_path / "hist").resolve()
    assert (tmp_path / got["VOCAB"]).resolve() == (tmp_path / "v.json").resolve()


@pytest.fixture(scope="module")
def constants() -> dict[str, str | None]:
    out: dict[str, str | None] = json.loads(run_py(DUMP, json.dumps(list(TABLE)), env=clean_env()))
    return out


@pytest.mark.parametrize("name", list(TABLE))
def test_path_constant(name: str, constants: dict[str, str | None]) -> None:
    assert constants[name] is not None, f"{name} is not defined"
    assert Path(constants[name] or "") == ROOT / TABLE[name]


def test_read_jsonl_missing_file_and_blank_lines(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "none.jsonl") == []
    p: Path = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_append_nothing_creates_and_changes_nothing(tmp_path: Path) -> None:
    p: Path = tmp_path / "a.jsonl"
    append_jsonl(p, [])
    assert not p.exists()
    p.write_text('{"a": 1}\n', encoding="utf-8")
    append_jsonl(p, [])
    assert p.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_append_writes_one_line_per_row_and_keeps_cyrillic(tmp_path: Path) -> None:
    p: Path = tmp_path / "a.jsonl"
    rows: list[dict[str, Any]] = [{"w": "жук"}, {"w": "café"}]
    append_jsonl(p, rows[:1])
    append_jsonl(p, rows[1:])
    raw: bytes = p.read_bytes()
    assert raw.count(b"\n") == 2 and raw.endswith(b"\n")
    assert "ж".encode() in raw
    assert b"\\u0436" not in raw
    assert read_jsonl(p) == rows
