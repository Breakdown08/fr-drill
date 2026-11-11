"""The fr-drill command line, through a subprocess (spec checks 12-18)."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from util import DATA, NEEDS_VOCAB, PRIVATE, clean_env, run_cli

TASK_KEYS: set[str] = {
    "test",
    "id",
    "consigne",
    "prompt",
    "points",
    "min_words",
    "answer",
    "words",
    "criteria",
}


def test_help_lists_every_command() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("--help")
    assert done.returncode == 0
    word: str
    for word in ("drill", "exam", "bank", "syllabus", "overlap"):
        assert word in done.stdout
    assert "fr-drill" in done.stdout


def test_help_has_no_private_names() -> None:
    out: str = "".join(
        run_cli(*a, "--help").stdout.lower() for a in ([], ["drill"], ["exam"], ["bank"], ["syllabus"])
    )
    assert "fr-drill" in out
    assert not any(name in out for name in PRIVATE)


def test_no_command_is_a_usage_error() -> None:
    done: subprocess.CompletedProcess[str] = run_cli()
    assert done.returncode == 2
    assert done.stderr.strip() != ""


@pytest.mark.parametrize(
    ("command", "words"),
    [
        ("drill", ["--port", "--check", "--stats", "--no-open"]),
        ("exam", ["run", "last", "pending", "grade", "--stage", "--port", "--no-open"]),
        ("syllabus", ["--check", "--out"]),
        ("overlap", ["--run", "--sentence"]),
        ("bank", ["check"]),
    ],
)
def test_command_help(command: str, words: list[str]) -> None:
    done: subprocess.CompletedProcess[str] = run_cli(command, "--help")
    assert done.returncode == 0
    word: str
    for word in words:
        assert word in done.stdout, word


def test_exam_help_describes_pending_and_grade() -> None:
    out: str = " ".join(run_cli("exam", "--help").stdout.split())
    assert "pending: letters waiting for a grade (JSON)" in out
    assert "grade: a grade of a letter (JSON file or -)" in out


def test_defaults_and_choices() -> None:
    assert "8770" in run_cli("drill", "--help").stdout
    assert "8771" in run_cli("exam", "--help").stdout
    assert run_cli("exam", "nonsense").returncode == 2
    assert run_cli("bank", "nonsense").returncode == 2


@pytest.mark.parametrize("flag", ["--check", "--stats"])
def test_drill_without_a_vocabulary_file(flag: str, tmp_path: Path) -> None:
    missing: Path = tmp_path / "none.json"
    env: dict[str, str] = clean_env(
        {"FR_DRILL_VOCAB": str(missing), "FR_DRILL_PROGRESS": str(tmp_path / "p")}
    )
    done: subprocess.CompletedProcess[str] = run_cli("drill", flag, env=env)
    assert done.returncode == 1
    assert "missing" in done.stdout and "FR_DRILL_VOCAB" in done.stdout
    assert str(missing) in done.stdout
    assert len(done.stdout.strip().splitlines()) == 1
    assert not (tmp_path / "p").exists()


def test_drill_server_mode_without_a_vocabulary_file_also_stops(tmp_path: Path) -> None:
    env: dict[str, str] = clean_env({"FR_DRILL_VOCAB": str(tmp_path / "none.json")})
    done: subprocess.CompletedProcess[str] = run_cli("drill", "--no-open", "--port", "0", env=env)
    assert done.returncode == 1
    assert "missing" in done.stdout


def test_exam_last_with_no_history_creates_nothing(tmp_path: Path) -> None:
    progress: Path = tmp_path / "p"
    done: subprocess.CompletedProcess[str] = run_cli(
        "exam", "last", env=clean_env({"FR_DRILL_PROGRESS": str(progress)})
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "no tests yet"
    assert not progress.exists()


def letter_row(test: str = "T1") -> dict[str, Any]:
    return {
        "at": "2026-11-09T13:41:06+07:00",
        "test": test,
        "stage": 1,
        "id": "presentation-001",
        "type": "write",
        "part": "pe",
        "topics": ["act.introduce"],
        "answer": "Bonjour ! Je m'appelle Marc. " * 8,
        "ok": None,
        "kind": "pending",
        "words": 40,
    }


def waiting(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "exam.jsonl").write_text(
        json.dumps(letter_row(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "exam-grades.jsonl").write_text("", encoding="utf-8")
    return clean_env({"FR_DRILL_PROGRESS": str(tmp_path)})


def test_exam_last_names_the_new_command(tmp_path: Path) -> None:
    done: subprocess.CompletedProcess[str] = run_cli("exam", "last", env=waiting(tmp_path))
    assert done.returncode == 0
    assert "ждёт оценки — fr-drill exam pending" in done.stdout
    assert not any(name in done.stdout.lower() for name in PRIVATE)


def test_exam_pending_is_one_task_in_json(tmp_path: Path) -> None:
    done: subprocess.CompletedProcess[str] = run_cli("exam", "pending", env=waiting(tmp_path))
    assert done.returncode == 0
    tasks: list[dict[str, Any]] = json.loads(done.stdout)
    assert len(tasks) == 1
    assert set(tasks[0]) - {"doc"} == TASK_KEYS
    assert tasks[0]["id"] == "presentation-001" and tasks[0]["words"] == 40


def test_exam_grade_from_stdin_then_nothing_pending(tmp_path: Path) -> None:
    env: dict[str, str] = waiting(tmp_path)
    reply: dict[str, Any] = {
        "test": "T1",
        "id": "presentation-001",
        "comment": "Повторы.",
        "scores": {"task": 1, "coherence": 1, "sociolinguistic": 1, "lexicon": 1, "morphosyntax": 1},
        "fixes": [],
    }
    done: subprocess.CompletedProcess[str] = run_cli("exam", "grade", "-", env=env, stdin=json.dumps(reply))
    assert done.returncode == 0, done.stderr
    assert "graded presentation-001" in done.stdout
    assert json.loads(run_cli("exam", "pending", env=env).stdout) == []
    assert (tmp_path / "exam-grades.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_bank_check() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("bank", "check")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok: 53 items, 1 docs in 10 files (ce 6, pe 1, practice 46)"


def test_syllabus_check() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("syllabus", "--check")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok: 67 topics"


def test_syllabus_out_writes_there_and_leaves_the_data_alone(tmp_path: Path) -> None:
    committed: Path = DATA / "syllabus" / "syllabus.md"
    before: bytes = committed.read_bytes()
    out: Path = tmp_path / "s.md"
    done: subprocess.CompletedProcess[str] = run_cli("syllabus", "--out", str(out))
    assert done.returncode == 0, done.stderr
    text: str = out.read_text(encoding="utf-8")
    assert text.startswith("# Карта тем")
    assert "`fr-drill syllabus`" in text
    assert "[[" not in text
    assert committed.read_bytes() == before


@NEEDS_VOCAB
def test_drill_check_output_matches_the_reference() -> None:
    """Spec check 25. The vocabulary line and the numbers line differ from the reference by name only."""
    done: subprocess.CompletedProcess[str] = run_cli("drill", "--check", env=clean_env(keep=True))
    assert done.returncode == 0, done.stdout + done.stderr
    lines: list[str] = done.stdout.splitlines()
    assert len(lines) == 11
    assert lines[0] == "verbs-er.json: ok, 33 verbs"
    name: str = Path(os.environ.get("FR_DRILL_VOCAB") or DATA / "drills" / "vocab.json").name
    assert lines[1].startswith(f"{name}: ok, ") and lines[1].endswith(" cards")
    assert lines[2] == "numbers: ok, 30 base words built by the rule"
    assert lines[3:] == [
        "negation-er.json: ok, 43 phrases",
        "etre-avoir.json: ok, 103 phrases",
        "sentences.json: ok, 81 sentences",
        "a-de.json: ok, 95 sentences",
        "a-de-verbs.json: ok, 22 verbs",
        "pronouns.json: ok, 37 sentences",
        "gender.json: ok, 368 nouns",
        "articles.json: ok, 847 sentences",
    ]


def test_drill_names_a_broken_vocabulary_by_its_file_name(tmp_path: Path) -> None:
    broken: Path = tmp_path / "my-words.json"
    broken.write_text("{not json", encoding="utf-8")
    env: dict[str, str] = clean_env({"FR_DRILL_VOCAB": str(broken), "FR_DRILL_PROGRESS": str(tmp_path / "p")})
    done: subprocess.CompletedProcess[str] = run_cli("drill", "--check", env=env)
    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == "my-words.json: invalid"


@NEEDS_VOCAB
def test_drill_stats_on_an_empty_history(tmp_path: Path) -> None:
    progress: Path = tmp_path / "p"
    done: subprocess.CompletedProcess[str] = run_cli(
        "drill", "--stats", env=clean_env({"FR_DRILL_PROGRESS": str(progress)}, keep=True)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    lines: list[str] = done.stdout.splitlines()
    assert lines[0] == "no drill answers yet"
    assert "words: no answers yet" in lines and "pronouns: no answers yet" in lines
    assert not progress.exists()
