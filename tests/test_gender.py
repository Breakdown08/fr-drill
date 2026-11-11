"""Gender tab: rules, the bank, grading, a session and a check on a temp attempts file."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from util import NEEDS_VOCAB

from fr_drill import gender as G
from fr_drill import read_jsonl, vocab

DATA: dict[str, Any] = {
    "no_rule": "запомни",
    "groups": [
        {"ru": "Страны", "g": "e", "rule": "на -e ж.", "words": ["la France", "le Mexique", "le Japon"]},
        {"ru": "Люди", "g": None, "rule": "по полу", "words": ["la mère"]},
    ],
    "endings": [
        {"end": ["t"], "g": "m", "rule": "-t м."},
        {"end": ["et"], "g": "m", "rule": "-et м."},
        {"end": ["tion"], "g": "f", "rule": "-tion ж."},
        {"end": ["eur"], "g": None, "rule": "двояко"},
    ],
}


def test_longest_ending_wins() -> None:
    got: Any = G.ending("billet", DATA["endings"])
    assert got[0] == "et"


def test_ending_of_the_first_word_only() -> None:
    assert G.ending("salle de bains", DATA["endings"]) is None


def test_explain_rule_exception_and_memorize() -> None:
    assert G.explain("la nation", "nation", "f", DATA) == ("-tion", "rule", "-tion ж.")
    assert G.explain("la nuit", "nuit", "f", DATA)[1] == "exception"
    assert G.explain("la fleur", "fleur", "f", DATA)[1] == "memorize"
    assert G.explain("la porte", "porte", "f", DATA) == (G.NO_RULE, "memorize", "запомни")


def test_explain_groups_go_before_endings() -> None:
    assert G.explain("la France", "France", "f", DATA)[:2] == ("Страны", "rule")
    assert G.explain("le Mexique", "Mexique", "m", DATA)[1] == "exception"
    assert G.explain("le Japon", "Japon", "m", DATA)[1] == "rule"
    assert G.explain("la mère", "mère", "f", DATA)[1] == "rule"


@pytest.fixture(scope="module")
def bank(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return G.load()


@pytest.fixture(scope="module")
def by(bank: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["fr"]: s for s in bank}


@NEEDS_VOCAB
def test_every_article_noun_of_the_word_cards(bank: list[dict[str, Any]]) -> None:
    assert len(bank) == sum(1 for c in vocab.load() if vocab.articles(c))


@NEEDS_VOCAB
def test_elided_nouns_and_seasons(by: dict[str, dict[str, Any]]) -> None:
    assert by["l'été"]["pair"] == ["un", "une"] and by["l'été"]["full"] == "un été"
    assert by["l'été"]["topic"] == "Времена года" and by["l'été"]["kind"] == "rule"


@NEEDS_VOCAB
def test_known_exceptions_and_rules(by: dict[str, dict[str, Any]]) -> None:
    assert all(by[w]["kind"] == "exception" for w in ["la plage", "le musée", "la main", "la photo"])
    assert all(by[w]["kind"] == "rule" for w in ["la télévision", "le fromage", "la voiture", "le bateau"])


@NEEDS_VOCAB
def test_rules_explain_most_nouns(bank: list[dict[str, Any]]) -> None:
    kinds: Counter[str] = Counter(s["kind"] for s in bank)
    assert kinds["rule"] > kinds["memorize"] > kinds["exception"] > 0


@NEEDS_VOCAB
def test_bad_bank_names_every_problem(tmp_path: Path) -> None:
    bad: dict[str, Any] = json.loads(G.BANK.read_text(encoding="utf-8"))
    bad["groups"][0]["words"].append("le dragon")
    bad["groups"][1]["words"].append("la France")
    bad["endings"].append({"end": ["tion"], "g": "x", "rule": ""})
    (tmp_path / "bad.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        G.load(tmp_path / "bad.json")
    msg: str = str(err.value)
    assert "'le dragon' is not a noun card" in msg and "'la France' is in two groups" in msg
    assert "ending -tion twice" in msg and "g: m, f or null" in msg


@NEEDS_VOCAB
def test_grade(by: dict[str, dict[str, Any]]) -> None:
    item: dict[str, Any] = by["la plage"]
    assert G.grade("la", item) == (True, None, [])
    ok: Any
    kind: Any
    rules: Any
    ok, kind, rules = G.grade("le", item)
    assert not ok and kind == "exception" and rules == [item["rule"], G.EXCEPTION]
    assert G.grade("la", by["le fromage"])[1] == "rule"


@NEEDS_VOCAB
def test_session_and_check(bank: list[dict[str, Any]], tmp_path: Path) -> None:
    g: Any = G.Gender(bank, tmp_path / "gender.jsonl")
    s: dict[str, Any] = g.session()
    assert len(s["items"]) == G.SIZE and len({i["noun"] for i in s["items"]}) == G.SIZE
    assert all(set(i) == {"id", "ru", "noun", "pair"} for i in s["items"])
    first: dict[str, Any] = s["items"][0]
    want: dict[str, Any] = next(x for x in bank if x["noun"] == first["noun"])
    wrong: str = next(p for p in first["pair"] if p != want["gap"])
    res: dict[str, Any] = g.check({"session": s["session"], "id": first["id"], "choice": wrong, "ms": 900})
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "gender.jsonl")
    assert not res["ok"] and res["expected"] == want["full"] and rows[-1]["kind"] == want["kind"]
    assert rows[-1]["drill"] == "gender" and rows[-1]["topic"] == want["topic"]
    second: dict[str, Any] = s["items"][1]
    right: str = next(x for x in bank if x["noun"] == second["noun"])["gap"]
    res = g.check({"session": s["session"], "id": second["id"], "choice": right})
    assert res["ok"] and res["rules"] and read_jsonl(tmp_path / "gender.jsonl")[-1]["kind"] is None
    body: dict[str, Any]
    for body in (
        {"session": "nope", "id": "0", "choice": "le"},
        {"session": s["session"], "id": "0", "choice": "les"},
    ):
        with pytest.raises(KeyError):
            g.check(body)
    st: dict[str, Any] = g.stats()
    assert st["answers"] == 2 and st["ok"] == 1 and st["kinds"] == {want["kind"]: 1}
