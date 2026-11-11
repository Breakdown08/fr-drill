"""Articles tab: the bank, validation, grading by layer, level sessions, checks and closed levels."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from util import NEEDS_VOCAB

from fr_drill import articles as A
from fr_drill import read_jsonl

type Sentence = dict[str, Any]

pytestmark: pytest.MarkDecorator = NEEDS_VOCAB


@pytest.fixture(scope="module")
def bank(cards: list[dict[str, Any]]) -> list[Sentence]:
    return A.load()


@pytest.fixture(scope="module")
def by(bank: list[Sentence]) -> dict[str, Sentence]:
    return {s["task"]: s for s in bank}


def test_every_level_has_enough_sentences(bank: list[Sentence]) -> None:
    levels: Counter[int] = Counter(s["level"] for s in bank)
    assert all(levels[k] >= A.NEW and levels[k] + len(A.earlier(bank, k)) >= A.SIZE for k in A.LEVELS), levels


def test_every_gap_is_on_its_levels_panel(bank: list[Sentence]) -> None:
    assert all(s["gap"] in A.LEVELS[s["level"]][1] for s in bank)
    assert {s["gap"] for s in bank if s["level"] == 1} == {"un", "une"}
    assert all(set(A.LEVELS[k][1]) <= set(A.LEVELS[k + 1][1]) for k in range(3, 10))


def test_panel_by_rows() -> None:
    assert A.panel(1) == [{"row": "неопределённый", "keys": ["un", "une"]}]
    assert [r["keys"] for r in A.panel(9)] == A.ROWS


def test_the_noun_after_the_gap_and_first_mention(by: dict[str, Sentence]) -> None:
    assert by["Je mange ___ pain."]["noun"] == "pain"
    assert by["J'ai un chat. ___ chat mange du poisson."]["level"] == 4


def test_sentences_made_from_the_cards(bank: list[Sentence], by: dict[str, Sentence]) -> None:
    cards: list[Sentence] = [s for s in bank if s.get("made")]
    assert {s["level"] for s in cards} == {1, 2, 3} and len(cards) > 300
    assert by["C'est ___ banc."]["gap"] == "un" and by["Où est ___ tomate ?"]["gap"] == "la"
    assert by["Où est ___étoile ?"]["gap"] == "l'" and by["___ hamburger est ici."]["gap"] == "le"
    assert by["___ chambre est ici."]["ru"] == "Спальня здесь."
    assert "C'est ___ lait." not in by and "Où est ___ étoile ?" not in by
    assert by["Où est ___ brosse à dents ?"]["noun"] == "brosse à dents"


def test_the_two_phrases_of_the_textbook_are_gone(bank: list[Sentence]) -> None:
    text: str = " ".join(s["fr"] for s in bank).lower().replace("’", "'")
    assert "ce n'est pas du fromage" not in text
    assert "les enfants jouent dans le jardin" not in text


def test_bad_bank_names_every_problem(tmp_path: Path) -> None:
    bad: dict[str, Any] = json.loads(A.BANK.read_text(encoding="utf-8"))
    bad["sentences"] += [
        {"fr": "Je mange le pizza.", "gap": "le", "ru": "x", "use": "general"},
        {"fr": "Je ferme la appartement.", "gap": "la", "ru": "x", "use": "known"},
        {"fr": "Je mange du pain.", "gap": "du", "ru": "x", "use": "known"},
        {"fr": "Je mange du dragon.", "gap": "du", "ru": "x", "use": "some"},
        {"fr": "Je ferme la porte.", "gap": "la", "ru": "Я закрываю дверь.", "use": "known"},
        {"fr": "Il a la la.", "gap": "la", "ru": "x", "use": "known"},
    ]
    (tmp_path / "bad.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        A.load(tmp_path / "bad.json")
    msg: str = str(err.value)
    part: str
    for part in (
        "'le' before 'pizza' of the other gender",
        "'la' before 'appartement': elision is wrong",
        "'du' is not on the panel of level 2",
        "'dragon' is not learned anywhere",
        "'Je ferme ___ porte.' twice",
        "must stand exactly once",
    ):
        assert part in msg, part


RULES_OF: dict[str, str] = {"plage": "-age → м. Исключение: la plage."}


def test_grade_by_layer() -> None:
    s: Sentence = {"gap": "la", "noun": "plage", "rule": "тот самый"}
    assert A.grade("la", s, RULES_OF) == (True, None, [])
    assert A.grade("le", s, RULES_OF) == (False, "gender", ["plage — женского рода.", RULES_OF["plage"]])
    assert A.grade("les", s, RULES_OF)[:2] == (False, "form")
    assert A.grade("une", s, RULES_OF) == (False, "type", ["тот самый", A.FORMS["la"]])
    assert A.grade("du", {"gap": "de", "noun": "pain", "rule": "pas de"}, RULES_OF)[1] == "de"
    assert A.grade("de", {"gap": "du", "noun": "pain", "rule": "сколько-то"}, RULES_OF)[1] == "de"
    assert A.grade("du", {"gap": "de la", "noun": "viande", "rule": ""}, {})[1] == "gender"


@pytest.mark.parametrize("level", [1, 2])
def test_first_levels_draw_twenty_of_their_own(bank: list[Sentence], level: int) -> None:
    rng: random.Random = random.Random(1)
    own: list[Sentence] = A.pick(bank, [], level, rng)
    assert len(own) == A.SIZE and {x["level"] for x in own} == {level}
    assert sum(bool(x.get("made")) for x in own) == A.SIZE // 2
    again: list[Sentence] = A.pick(bank, [], level, rng)
    assert len({x["task"] for x in own} & {x["task"] for x in again}) < A.SIZE * 3 // 4


def test_level_three_and_nine_mix(bank: list[Sentence]) -> None:
    rng: random.Random = random.Random(1)
    three: list[Sentence] = A.pick(bank, [], 3, rng)
    assert len(three) == A.SIZE and sum(x["level"] == 3 for x in three) == A.NEW
    assert sum(bool(x.get("made")) for x in three) == A.NEW // 2
    assert all(x["level"] == 3 or not x.get("made") for x in three)
    nine: list[Sentence] = A.pick(bank, [], 9, rng)
    assert sum(x["level"] == 9 for x in nine) == A.NEW and len({x["task"] for x in nine}) == A.SIZE
    assert all(x["gap"] in A.LEVELS[9][1] for x in nine)


def test_session_and_check(bank: list[Sentence], tmp_path: Path) -> None:
    arts: Any = A.Articles(bank, tmp_path / "articles.jsonl", rules_of={})
    sess: dict[str, Any] = arts.session(1)
    assert sess["panel"] == A.panel(1)
    assert all(set(i) == {"id", "ru", "before", "after"} for i in sess["items"])
    first: dict[str, Any] = sess["items"][0]
    want: Sentence = next(x for x in bank if x["before"] == first["before"] and x["after"] == first["after"])
    wrong: str = "une" if want["gap"] == "un" else "un"
    res: dict[str, Any] = arts.check(
        {"session": sess["session"], "id": first["id"], "choice": wrong, "ms": 700}
    )
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "articles.jsonl")
    assert not res["ok"] and res["expected"] == want["fr"] and rows[-1]["kind"] == res["kind"]
    assert rows[-1]["level"] == 1 and rows[-1]["bank"] == A.VERSION and rows[-1]["drill"] == "articles"
    second: dict[str, Any] = sess["items"][1]
    gap: str = next(x for x in bank if x["before"] == second["before"] and x["after"] == second["after"])[
        "gap"
    ]
    res = arts.check({"session": sess["session"], "id": second["id"], "choice": gap})
    assert res["ok"] and res["rules"]
    body: dict[str, Any]
    for body in (
        {"session": "nope", "id": "0", "choice": "le"},
        {"session": sess["session"], "id": "0", "choice": "le"},
        {"session": sess["session"], "id": "0", "choice": "aux"},
    ):
        with pytest.raises(KeyError):
            arts.check(body)
    with pytest.raises(KeyError):
        arts.session(11)
    st: dict[str, Any] = arts.stats()
    assert st["answers"] == 2 and st["ok"] == 1 and st["levels"] == {}


def full(sid: str, level: int, ok: int, bank: int | None = None) -> list[dict[str, Any]]:
    version: int = A.VERSION if bank is None else bank
    return [
        {"session": sid, "level": level, "ok": i < ok, "topic": "x", "kind": None, "bank": version}
        for i in range(A.SIZE)
    ]


def test_levels_close_on_two_passing_sessions_in_a_row() -> None:
    history: list[dict[str, Any]] = (
        full("a", 1, 18) + full("b", 1, 19) + full("c", 2, 20) + full("d", 2, 15) + full("e", 3, 20)
    )
    history += [{"session": "old", "stage": 1, "ok": True, "topic": "x", "kind": None}] * A.SIZE
    history += full("f", 3, 20)[:5] + full("g", 4, 20, bank=1) + full("h", 4, 20, bank=1)
    lv: dict[int, dict[str, Any]] = A.levels(history)
    # half sessions, stage rows and old banks are ignored; a weak last session does not close a level
    assert lv[1] == {"scores": [18, 19], "closed": True}
    assert lv[2] == {"scores": [20, 15], "closed": False}
    assert lv[3] == {"scores": [20], "closed": False}
    assert set(lv) == {1, 2, 3}
