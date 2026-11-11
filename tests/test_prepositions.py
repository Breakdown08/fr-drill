"""À and de tab: the bank, grading, picking, HTTP on a temp attempts file."""

import random
from collections import Counter
from typing import Any

import pytest
from conftest import Live
from util import NEEDS_VOCAB, call

from fr_drill import prepositions as P

type Sentence = dict[str, Any]

pytestmark: pytest.MarkDecorator = NEEDS_VOCAB


@pytest.fixture(scope="module")
def bank(cards: list[dict[str, Any]]) -> list[Sentence]:
    return P.load()


@pytest.fixture(scope="module")
def by(bank: list[Sentence]) -> dict[str, Sentence]:
    return {s["task"]: s for s in bank}


@pytest.fixture(scope="module")
def a_de_verbs(cards: list[dict[str, Any]]) -> list[Sentence]:
    return P.load_verbs()


@pytest.fixture(scope="module")
def frames(a_de_verbs: list[Sentence]) -> dict[str, Sentence]:
    return {v["frame"]["fr"]: v for v in a_de_verbs}


def one(fr: str, gap: str) -> list[str]:
    core: set[str] = {"je", "vais", "gare", "la", "le", "bureau"} | {
        "à",
        "de",
        "au",
        "suis",
        "architecte",
        "l",
    }
    problems: list[str] = P.problems(
        {"fr": fr, "gap": gap, "ru": "x", "use": "u"}, {"u": {}}, core, set(), {"la gare", "le bureau"}
    )
    return problems


def test_bank_covers_both_prepositions_and_every_form(bank: list[Sentence]) -> None:
    assert {s["gap"] for s in bank} == set(P.TO + P.FROM), Counter(s["gap"] for s in bank)
    assert abs(sum(s["gap"] in P.TO for s in bank) - sum(s["gap"] in P.FROM for s in bank)) <= 10
    assert set(P.FORMS) == set(P.TO + P.FROM)


def test_gaps_are_cut_out(by: dict[str, Sentence]) -> None:
    gap: Sentence = by.get("Je vais ___ la gare.") or by["Je vais ___ gare."]
    assert gap["gap"] == "à" or gap["after"] == "gare."
    assert {"Je travaille ___ huit heures à midi.", "Je travaille de huit heures ___ midi."} <= set(by)
    assert "C'est le message ___architecte." in by or "C'est le message ___ architecte." in by


def test_grade_forms_and_prepositions(by: dict[str, Sentence]) -> None:
    s: Sentence = by["Je suis ___ bureau."]
    assert P.grade("au", s) == (True, None, [])
    assert P.grade("à", s)[:2] == (False, "form") and "à + le" in P.grade("à", s)[2][0]
    got: Any = P.grade("du", s)
    assert got[1] == "prep" and "Где и куда" in got[2][0] and len(got[2]) == 2
    assert len(P.grade("de", by["Je vais ___ Paris."])[2]) == 1
    assert "у кого-то" in P.grade("de", by["Je demande l'adresse ___ Marie."])[2][0]


def test_problems_clean_sentence_passes() -> None:
    assert one("Je vais à la gare.", "à la") == []


@pytest.mark.parametrize(
    ("fr", "gap"),
    [
        ("Je vais au gare.", "au"),
        ("Je vais à la bureau.", "à la"),
        ("Je vais à la architecte.", "à la"),
        ("Je vais à l'gare.", "à l'"),
        ("Je vais à la gare.", "de la"),
        ("Je vais à la mer.", "à la"),
    ],
)
def test_problems_refuses(fr: str, gap: str) -> None:
    assert one(fr, gap) != []


def test_session_of_size_and_a_mistake_comes_back(bank: list[Sentence]) -> None:
    picked: list[Sentence] = P.pick(bank, [], rng=random.Random(1))
    assert len(picked) == P.SIZE and len({s["fr"] for s in picked}) == P.SIZE
    miss: Sentence = bank[0]
    drawn: int = sum(
        miss in P.pick(bank, [{"task": miss["task"], "ok": False}], rng=random.Random(i)) for i in range(200)
    )
    fresh: int = sum(miss in P.pick(bank, [], rng=random.Random(i)) for i in range(200))
    assert drawn > fresh


def test_verbs_data(a_de_verbs: list[Sentence], frames: dict[str, Sentence]) -> None:
    assert {v["frame"]["gap"] for v in a_de_verbs} == set(P.FRAME)
    assert {"téléphoner à qn", "parler de qch", "regarder qch"} <= set(frames)
    assert {"jouer à игра, спорт", "jouer de инструмент", "oublier qch", "oublier de + глагол"} <= set(frames)
    table: dict[str, list[str]] = {"à": P.TO, "de": P.FROM, P.NONE: P.ARTICLES}
    assert all(s["gap"] in table[v["frame"]["gap"]] for v in a_de_verbs for s in v["sentences"])


def test_grade_frame(frames: dict[str, Sentence]) -> None:
    frame: Sentence = frames["aider qn"]["frame"]
    assert P.grade_verb(P.NONE, frame) == (True, None, [])
    got: Any = P.grade_verb("à", frame)
    assert got[:2] == (False, "prep") and "aider" in got[2][0]


def test_grade_sentences_of_verbs(frames: dict[str, Sentence]) -> None:
    s: Sentence = next(x for x in frames["regarder qch"]["sentences"] if x["gap"] == "la")
    assert P.grade_verb("à la", s)[1] == "prep"
    assert P.grade_verb("le", s)[:2] == (False, "form") and P.grade_verb("le", s)[2] == [P.BARE]
    s = next(x for x in frames["jouer de инструмент"]["sentences"] if x["gap"] == "du")
    assert P.grade_verb("de", s)[1] == "form" and "de + le" in P.grade_verb("de", s)[2][0]
    assert P.grade_verb("le", s)[1] == "prep" and len(P.grade_verb("le", s)[2]) == 2


def test_verb_session_and_comebacks(a_de_verbs: list[Sentence], frames: dict[str, Sentence]) -> None:
    picked: list[Sentence] = P.pick_verbs(a_de_verbs, [], rng=random.Random(2))
    assert len(picked) == P.VERB_SIZE * (1 + P.PER_VERB)
    assert all(x["choices"] is P.FRAME for x in picked[: P.VERB_SIZE])
    assert all(x["choices"] is not P.FRAME for x in picked[P.VERB_SIZE :])
    miss: Sentence = frames["aider qn"]["sentences"][0]
    history: list[dict[str, Any]] = [{"task": miss["task"], "ok": False}]
    drawn: int = sum(
        any(x["task"] == miss["task"] for x in P.pick_verbs(a_de_verbs, history, rng=random.Random(i)))
        for i in range(200)
    )
    fresh: int = sum(
        any(x["task"] == miss["task"] for x in P.pick_verbs(a_de_verbs, [], rng=random.Random(i)))
        for i in range(200)
    )
    assert drawn > fresh


def test_http_sentences(live: Live, bank: list[Sentence]) -> None:
    port: int = live.port
    status: int
    data: dict[str, Any]
    status, data = call(port, "GET", "/prepositions/session")
    assert (
        status == 200
        and len(data["items"]) == 20
        and set(data["items"][0]) == {"id", "ru", "before", "after"}
    )
    assert data["to"] == P.TO and data["from"] == P.FROM
    item: dict[str, Any] = data["items"][0]
    s: Sentence = next(x for x in bank if x["before"] == item["before"] and x["after"] == item["after"])
    res: dict[str, Any]
    status, res = call(
        port,
        "POST",
        "/prepositions/check",
        {"session": data["session"], "id": item["id"], "choice": s["gap"], "ms": 800},
    )
    assert (
        status == 200
        and res["ok"]
        and res["kind"] is None
        and res["expected"] == s["fr"]
        and not res["rules"]
    )
    item = data["items"][1]
    s = next(x for x in bank if x["before"] == item["before"] and x["after"] == item["after"])
    other: str = P.FROM[0] if s["gap"] in P.TO else P.TO[0]
    status, res = call(
        port, "POST", "/prepositions/check", {"session": data["session"], "id": item["id"], "choice": other}
    )
    assert res["ok"] is False and res["kind"] == "prep" and res["rules"]
    assert (
        call(
            port,
            "POST",
            "/prepositions/check",
            {"session": data["session"], "id": item["id"], "choice": "dans"},
        )[0]
        != 200
    )
    assert call(port, "POST", "/prepositions/check", {"session": "none", "id": "0", "choice": "à"})[0] != 200
    st: dict[str, Any]
    status, st = call(port, "GET", "/prepositions/stats")
    assert st["answers"] == 2 and st["ok"] == 1 and st["kinds"] == {"prep": 1}
    assert sum(x["tries"] for x in st["uses"].values()) == 2


def test_http_verbs(live: Live) -> None:
    port: int = live.port
    status: int
    data: dict[str, Any]
    status, data = call(port, "GET", "/prepositions/verbs/session")
    assert status == 200 and len(data["items"]) == 24
    assert set(data["items"][0]) == {"id", "ru", "before", "after", "choices"}
    assert data["items"][0]["choices"] == P.FRAME and data["none"] == P.NONE
    assert set(data["items"][-1]["choices"]) == set(P.TO + P.FROM + P.ARTICLES)
    assert (
        call(port, "POST", "/prepositions/check", {"session": data["session"], "id": "0", "choice": "au"})[0]
        != 200
    )
    res: dict[str, Any]
    status, res = call(
        port, "POST", "/prepositions/check", {"session": data["session"], "id": "0", "choice": "de"}
    )
    assert status == 200 and "ok" in res and res["expected"] and res["use"]
