"""Exam: grading, numbers, picking, HTTP on temp files, letters waiting for a grade."""

import copy
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from util import call, serving

from fr_drill import exam

type Bank = tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]


@pytest.fixture(scope="module")
def bank_data() -> Bank:
    return exam.load_bank()


@pytest.fixture(scope="module")
def by_id(bank_data: Bank) -> dict[str, dict[str, Any]]:
    return {i["id"]: i for i in bank_data[0]}


def test_real_bank_loads(bank_data: Bank) -> None:
    assert len(bank_data[0]) >= 50


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("26", 26.0),
        ("26 ans", 26.0),
        ("12,50 €", 12.5),
        ("12.5", 12.5),
        ("3 euros", 3.0),
        ("19h", "19:00"),
        ("19 h", "19:00"),
        ("19:00", "19:00"),
        ("19h30", "19:30"),
        ("9 h 05", "09:05"),
        ("vingt-six", None),
        ("25:00", None),
        ("", None),
    ],
)
def test_parse_number(text: str, want: float | str | None) -> None:
    assert exam.parse_number(text) == want


def test_grade_by_type(by_id: dict[str, dict[str, Any]]) -> None:
    etre5: dict[str, Any] = by_id["etre-005"]  # practice short: êtes
    pisc: dict[str, Any] = by_id["profil-004"]  # ce short: Karim
    num: dict[str, Any] = by_id["profil-001"]  # 26
    time: dict[str, Any] = {"type": "number", "part": "ce", "answer": "19:00"}
    choice: dict[str, Any] = by_id["etre-001"]
    letter: dict[str, Any] = by_id["presentation-001"]
    cases: list[tuple[dict[str, Any], object, tuple[bool | None, str | None]]] = [
        (etre5, "êtes", (True, None)),
        (etre5, " Êtes. ", (True, None)),
        (etre5, "etes", (False, "accent")),
        (etre5, "sont", (False, "wrong")),
        (etre5, "", (False, "empty")),
        (etre5, None, (False, "empty")),
        (pisc, "karim", (True, None)),
        (pisc, "Kärim", (True, None)),
        (pisc, "Paul", (False, "wrong")),
        (num, "26 ans", (True, None)),
        (num, "25", (False, "wrong")),
        (num, "vingt-six", (False, "format")),
        (time, "19 h", (True, None)),
        (time, "19", (False, "wrong")),
        (choice, 0, (True, None)),
        (choice, 2, (False, "wrong")),
        (choice, None, (False, "empty")),
        (letter, "mot " * 40, (None, "pending")),
        (letter, "mot " * 39, (None, "short")),
        (letter, "  ", (None, "empty")),
        (letter, None, (None, "empty")),
    ]
    item: dict[str, Any]
    answer: object
    want: tuple[bool | None, str | None]
    for item, answer, want in cases:
        assert exam.grade(item, answer) == want, (item.get("id"), answer)


def test_expected_and_words() -> None:
    assert exam.expected({"type": "number", "answer": 12.5}) == "12,5"
    assert exam.words("J'ai un rendez-vous demain.") == 4  # j'ai and rendez-vous are one word each


def test_stage_one_test_shape(bank_data: Bank) -> None:
    items: list[dict[str, Any]] = bank_data[0]
    stages: dict[str, Any] = bank_data[2]
    t: list[dict[str, Any]] = exam.pick(items, stages, [], 1, rng=random.Random(1))
    parts: list[str] = [i["part"] for i in t]
    assert parts.count("practice") == 10 and "ce" in parts and parts[-1] == "pe"
    assert parts == sorted(parts, key=["practice", "ce", "pe"].index)
    assert len({i["topics"][0] for i in t if i["part"] == "practice"}) >= 7
    ce: list[dict[str, Any]] = [i for i in t if i["part"] == "ce"]
    assert [i["id"] for i in ce] == [i["id"] for i in items if i.get("doc") == ce[0]["doc"]]
    assert exam.pick(items, stages, [], 0, rng=random.Random(1)) == []


def test_past_mistakes_fill_half_of_practice(bank_data: Bank) -> None:
    items: list[dict[str, Any]] = bank_data[0]
    stages: dict[str, Any] = bank_data[2]
    wrong: list[str] = ["etre-002", "avoir-003", "genre-006", "cest-002", "pays-003", "articles-004"]
    hist: list[dict[str, Any]] = [{"id": w, "at": "2026-10-27T10:00:00", "ok": False} for w in wrong]
    hist += [
        {"id": i["id"], "at": "2026-10-27T09:00:00", "ok": True}
        for i in items
        if i["part"] == "practice" and i["id"] not in wrong
    ][:20]
    t: list[dict[str, Any]] = exam.pick(items, stages, hist, 1, rng=random.Random(2))
    got: set[str] = {i["id"] for i in t} & set(wrong)
    assert len(got) == 5
    seen: set[str] = {h["id"] for h in hist}
    assert all(i["id"] not in seen for i in t if i["part"] == "practice" and i["id"] not in wrong)
    fixed: list[dict[str, Any]] = [{"id": w, "at": "2026-10-27T11:00:00", "ok": True} for w in wrong]
    t = exam.pick(items, stages, hist + fixed, 1, rng=random.Random(2))
    assert not ({i["id"] for i in t} & set(wrong))


def test_http_test_and_submit(bank_data: Bank, by_id: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    items: list[dict[str, Any]] = bank_data[0]
    docs: dict[str, Any] = bank_data[1]
    stages: dict[str, Any] = bank_data[2]
    consignes: dict[str, Any] = bank_data[3]
    attempts: Path = tmp_path / "exam.jsonl"
    ex: Any = exam.Exam(items, docs, stages, consignes, 1, attempts)
    port: int
    with serving(lambda port: exam.handler(ex, port)) as port:
        assert call(port, "GET", "/")[0] == 200
        assert call(port, "GET", "/test", host="evil.com")[0] == 403
        st: int
        test: dict[str, Any]
        st, test = call(port, "GET", "/test")
        assert st == 200 and len(test["items"]) == 17
        pub: list[dict[str, Any]] = test["items"]
        assert not any(k in i for i in pub for k in ("answer", "accept", "explain", "source"))
        assert all("ru" in i["consigne"] for i in pub)
        ch: list[dict[str, Any]] = [i for i in pub if i["type"] == "choice"]
        assert all(sorted(o["i"] for o in i["options"]) == list(range(len(i["options"]))) for i in ch)
        assert all(i["doc"]["text"] for i in pub if i["part"] == "ce")

        answers: dict[str, Any] = {}
        i: dict[str, Any]
        for i in pub:
            item: dict[str, Any] = by_id[i["id"]]
            if item["type"] == "choice":
                answers[i["id"]] = item["answer"]
            elif item["type"] == "short":
                answers[i["id"]] = item["accept"][0]
            elif item["type"] == "number":
                answers[i["id"]] = str(item["answer"])
            else:
                answers[i["id"]] = "Bonjour ! Je m'appelle Marc. " * 8
        first: str = next(i["id"] for i in pub if by_id[i["id"]]["type"] == "choice")
        answers[first] = (by_id[first]["answer"] + 1) % len(by_id[first]["options"])
        assert (
            call(port, "POST", "/submit", {"test": test["test"], "answers": answers}, ctype="text/plain")[0]
            == 415
        )
        res: dict[str, Any]
        st, res = call(port, "POST", "/submit", {"test": test["test"], "answers": answers})
        assert st == 200 and (res["score"], res["of"]) == (15, 16)
        bad: dict[str, Any] = next(r for r in res["items"] if r["id"] == first)
        assert bad["expected"] == by_id[first]["options"][by_id[first]["answer"]] and bad["explain"]
        rows: list[dict[str, Any]] = [
            json.loads(x) for x in attempts.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 17 and {r["test"] for r in rows} == {test["test"]}
        assert next(r for r in rows if r["type"] == "write")["words"] == 40
        assert call(port, "POST", "/submit", {"test": test["test"], "answers": answers})[0] == 404


def play_one_test(bank_data: Bank, attempts: Path) -> dict[str, Any]:
    """Submit a test whose letter is long enough, through the Exam object; returns the test."""
    items: list[dict[str, Any]] = bank_data[0]
    docs: dict[str, Any] = bank_data[1]
    stages: dict[str, Any] = bank_data[2]
    consignes: dict[str, Any] = bank_data[3]
    ex: Any = exam.Exam(items, docs, stages, consignes, 1, attempts)
    test: dict[str, Any] = ex.test()
    answers: dict[str, Any] = {
        i["id"]: "Bonjour ! Je m'appelle Marc. " * 8 for i in test["items"] if i["type"] == "write"
    }
    ex.submit({"test": test["test"], "answers": answers})
    return test


def test_pending_letter_carries_points_and_criteria(bank_data: Bank, tmp_path: Path) -> None:
    attempts: Path = tmp_path / "exam.jsonl"
    play_one_test(bank_data, attempts)
    tasks: list[dict[str, Any]] = exam.pending(attempts, tmp_path / "grades.jsonl")
    assert len(tasks) == 1 and tasks[0]["points"] and len(tasks[0]["criteria"]) == 5
    assert set(tasks[0]) - {"doc"} == {
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


def reply_for(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "test": task["test"],
        "id": task["id"],
        "comment": "Повторы, пунктов не хватает.",
        "scores": {"task": 0.5, "coherence": 1, "sociolinguistic": 2, "lexicon": 1, "morphosyntax": 2.5},
        "fixes": [
            {
                "was": "Je m'appelle Marc. Je m'appelle",
                "fix": "Je m'appelle Marc. J'ai 28 ans",
                "why": "повтор",
            }
        ],
    }


GRADE_REFUSALS: list[tuple[str, Callable[[dict[str, Any]], object], str]] = [
    ("missing criterion", lambda r: r["scores"].pop("task"), "exactly"),
    ("score 4", lambda r: r["scores"].update(task=4), "half points"),
    ("score 1.3", lambda r: r["scores"].update(task=1.3), "half points"),
    ("no comment", lambda r: r.update(comment=" "), "comment"),
    ("unknown letter", lambda r: r.update(test="x"), "no letter"),
]


@pytest.mark.parametrize(("name", "change", "want"), GRADE_REFUSALS, ids=[g[0] for g in GRADE_REFUSALS])
def test_grade_refused(
    name: str, change: Callable[[dict[str, Any]], object], want: str, bank_data: Bank, tmp_path: Path
) -> None:
    attempts: Path = tmp_path / "exam.jsonl"
    grades: Path = tmp_path / "grades.jsonl"
    play_one_test(bank_data, attempts)
    task: dict[str, Any] = exam.pending(attempts, grades)[0]
    reply: dict[str, Any] = copy.deepcopy(reply_for(task))
    change(reply)
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        exam.accept_grade(reply, attempts, grades)
    assert want in str(err.value)
    assert not grades.exists() or grades.read_text(encoding="utf-8") == ""


def test_grade_accepted_once(bank_data: Bank, tmp_path: Path) -> None:
    attempts: Path = tmp_path / "exam.jsonl"
    grades: Path = tmp_path / "grades.jsonl"
    play_one_test(bank_data, attempts)
    task: dict[str, Any] = exam.pending(attempts, grades)[0]
    reply: dict[str, Any] = reply_for(task)
    row: dict[str, Any] = exam.accept_grade(reply, attempts, grades)
    assert (row["total"], row["of"]) == (7, 15)
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        exam.accept_grade(reply, attempts, grades)
    assert "already" in str(err.value)
    assert exam.pending(attempts, grades) == []
    assert exam.last(attempts, grades) == 0


def test_an_empty_letter_is_not_pending(bank_data: Bank, tmp_path: Path) -> None:
    items: list[dict[str, Any]] = bank_data[0]
    docs: dict[str, Any] = bank_data[1]
    stages: dict[str, Any] = bank_data[2]
    consignes: dict[str, Any] = bank_data[3]
    attempts: Path = tmp_path / "exam.jsonl"
    ex: Any = exam.Exam(items, docs, stages, consignes, 1, attempts)
    t: dict[str, Any] = ex.test()
    ex.submit({"test": t["test"], "answers": {}})
    assert exam.pending(attempts, tmp_path / "grades.jsonl") == []


def test_last_says_what_to_run_for_a_letter_without_a_grade(
    bank_data: Bank, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts: Path = tmp_path / "exam.jsonl"
    play_one_test(bank_data, attempts)
    assert exam.last(attempts, tmp_path / "grades.jsonl") == 0
    out: str = capsys.readouterr().out
    assert "ждёт оценки — fr-drill exam pending" in out
    assert "ny" + "arl" not in out.lower() and "kad" + "ath" not in out.lower()


def test_last_with_no_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert exam.last(tmp_path / "none.jsonl", tmp_path / "grades.jsonl") == 0
    assert capsys.readouterr().out.strip() == "no tests yet"


def test_exam_page_labels() -> None:
    page: str = exam.PAGE.read_text(encoding="utf-8")
    assert "ждёт проверки" in page
    assert "меньше минимума слов — ждёт проверки" in page
    assert "Письмо ждёт проверки: fr-drill exam pending, затем оценка через fr-drill exam grade." in page
    assert "ny" + "arl" not in page.lower()
