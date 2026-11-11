"""Pronouns tab: the bank, grading, a session and a check on a temp attempts file."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from fr_drill import pronouns as O


@pytest.fixture(scope="module")
def bank() -> list[dict[str, Any]]:
    return O.load()


@pytest.fixture(scope="module")
def by(bank: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["task"]: s for s in bank}


def one(fr: str, gap: str) -> list[str]:
    core: set[str] = {"je", "j", "suis", "habite", "à", "paris", "moi", "avec", "tu", "travailles"}
    problems: list[str] = O.problems({"fr": fr, "gap": gap, "ru": "x", "use": "u"}, {"u": {}}, core, set())
    return problems


def test_every_pronoun_of_both_rows_is_an_answer_somewhere(bank: list[dict[str, Any]]) -> None:
    assert set(O.PANEL) <= {s["gap"] for s in bank}, Counter(s["gap"] for s in bank)


def test_subject_and_stressed_are_both_well_represented(bank: list[dict[str, Any]]) -> None:
    subject: int = sum(s["gap"] in O.SUBJECT for s in bank)
    stressed: int = sum(s["gap"] in O.STRESSED for s in bank)
    assert min(subject, stressed) >= 12


def test_every_panel_pronoun_has_a_person() -> None:
    assert set(O.PANEL) <= set(O.PERSON)


def test_gaps_are_cut_where_the_word_starts(by: dict[str, dict[str, Any]]) -> None:
    assert "___habite à Paris." in by, [t for t in by if "habite" in t]
    assert "___, je suis russe." in by


def test_grade_right_and_role(by: dict[str, dict[str, Any]]) -> None:
    s: dict[str, Any] = by["Tu travailles avec ___ ?"]
    assert O.grade("moi", s) == (True, None, [])
    got: Any = O.grade("je", s)
    assert got[:2] == (False, "role") and O.ROLE in got[2]


def test_grade_person_gives_only_the_use(by: dict[str, dict[str, Any]]) -> None:
    s: dict[str, Any] = by["Tu travailles avec ___ ?"]
    got: Any = O.grade("toi", s)
    assert got[:2] == (False, "person") and got[2] == [s["rule"]]


def test_grade_elision(by: dict[str, dict[str, Any]]) -> None:
    assert O.grade("je", by["___habite à Paris."])[:2] == (False, "elision")


def test_elle_answers_from_either_row(by: dict[str, dict[str, Any]]) -> None:
    assert O.grade("elle", by["Le café est pour ___."])[0]


def test_problems_clean_sentence_passes() -> None:
    assert one("Je suis à Paris.", "je") == []


@pytest.mark.parametrize(
    ("fr", "gap"),
    [
        ("Je habite à Paris.", "je"),  # je before a vowel
        ("J'suis à Paris.", "j'"),  # j' before a consonant
        ("Je suis à Paris.", "à"),  # gap not in the panel
        ("Moi, avec moi.", "moi"),  # gap twice
        ("Je suis à Lyon.", "je"),  # unknown word
    ],
)
def test_problems_refuses(fr: str, gap: str) -> None:
    assert one(fr, gap) != []


def test_session_check_and_stats(bank: list[dict[str, Any]], tmp_path: Path) -> None:
    trainer: Any = O.Pronouns(bank, tmp_path / "pronouns.jsonl")
    got: dict[str, Any] = trainer.session()
    assert len(got["items"]) == O.SIZE
    assert all("gap" not in i and "fr" not in i for i in got["items"])
    assert got["subject"] == O.SUBJECT and got["stressed"] == O.STRESSED
    res: dict[str, Any] = trainer.check({"session": got["session"], "id": "0", "choice": "eux", "ms": 1200})
    assert (tmp_path / "pronouns.jsonl").read_text(encoding="utf-8").count('"drill": "pronouns"') == 1
    assert "expected" in res
    with pytest.raises(KeyError):
        trainer.check({"session": got["session"], "id": "0", "choice": "lol"})
    assert trainer.stats()["answers"] == 1
