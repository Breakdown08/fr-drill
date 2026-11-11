"""Negation tab: phrases by rule, grading, picking, HTTP on a temp attempts file."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import Live
from util import call

from fr_drill import drill, negation, read_jsonl

HEADS: dict[tuple[str, str], str] = {
    ("parler", "je"): "je ne parle pas",
    ("aimer", "je"): "je n'aime pas",
    ("habiter", "tu"): "tu n'habites pas",
    ("écouter", "nous"): "nous n'écoutons pas",
    ("étudier", "elle"): "elle n'étudie pas",
    ("manger", "nous"): "nous ne mangeons pas",
    ("commencer", "nous"): "nous ne commençons pas",
    ("regarder", "ils"): "ils ne regardent pas",
    ("oublier", "vous"): "vous n'oubliez pas",
    ("inviter", "on"): "on n'invite pas",
}
GRADES: list[tuple[str, str, str, str, bool, str | None]] = [
    ("je n'aime pas", "aimer", "je", "le café", True, None),
    ("Je n’aime pas le café.", "aimer", "je", "le café", True, None),
    ("je  n' aime  pas", "aimer", "je", "le café", True, None),
    ("elle ne téléphone pas à Marie", "téléphoner", "elle", "à Marie", True, None),
    ("j'aime pas", "aimer", "je", "le café", False, "ne"),
    ("je n'aime le café", "aimer", "je", "le café", False, "pas"),
    ("je ne aime pas", "aimer", "je", "le café", False, "elision"),
    ("j'n'aime pas", "aimer", "je", "le café", False, "elision"),
    ("nous ne écoutons pas", "écouter", "nous", "la radio", False, "elision"),
    ("je ne pas parle", "parler", "je", "anglais", False, "order"),
    ("ne je parle pas", "parler", "je", "anglais", False, "pronoun"),
    ("tu ne parles pas", "parler", "vous", "vite", False, "pronoun"),
    ("il n'etudie pas", "étudier", "il", "le soir", False, "accent"),
    ("nous ne mangons pas", "manger", "nous", "le matin", False, "spelling"),
    ("ils ne regarde pas", "regarder", "ils", "la télé", False, "ending"),
    ("vous ne dites pas", "parler", "vous", "vite", False, "other"),
    ("ils ne regardent pas le télé", "regarder", "ils", "la télé", False, "tail"),
    ("ils ne regardent pas la tele", "regarder", "ils", "la télé", False, "tail"),
]


@pytest.fixture(scope="module")
def phrases(verbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return negation.load(verbs=verbs)


def test_phrases_valid_and_every_verb_has_one(
    phrases: list[dict[str, Any]], verbs: list[dict[str, Any]]
) -> None:
    assert len(phrases) >= 40
    assert {v["inf"] for v in verbs} <= {p["inf"] for p in phrases}


@pytest.mark.parametrize(("verb", "person"), list(HEADS))
def test_head(verb: str, person: str) -> None:
    assert negation.head(verb, person) == HEADS[(verb, person)]


@pytest.mark.parametrize(("answer", "verb", "person", "rest", "ok", "kind"), GRADES)
def test_grade(answer: str, verb: str, person: str, rest: str, ok: bool, kind: str | None) -> None:
    assert negation.grade(answer, verb, person, rest) == (ok, kind)


def test_bad_list_names_every_problem(tmp_path: Path, verbs: list[dict[str, Any]]) -> None:
    bad: Path = tmp_path / "n.json"
    contexts: list[dict[str, str]] = [
        {"inf": "finir", "fr": "x", "ru": "не x"},
        {"inf": "manger", "fr": "du pain", "ru": "не есть хлеб"},
        {"inf": "parler", "fr": "", "ru": "говорить"},
        {"inf": "aimer", "fr": "le café", "ru": "не любить"},
        {"inf": "aimer", "fr": "le café", "ru": "не любить"},
    ]
    bad.write_text(json.dumps({"contexts": contexts}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        negation.load(bad, verbs)
    m: str = str(err.value)
    part: str
    for part in ("'finir' is not", "turns into de", "no rest", "must start with", "twice"):
        assert part in m


def test_fresh_session(phrases: list[dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = negation.pick(phrases, [], rng=random.Random(1))
    per: Counter[str] = Counter(i["person"] for i in items)
    assert len(items) == 15 and min(per.values()) >= 2
    assert len({i["verb"] for i in items}) >= 12


def test_past_mistake_comes_back_often(phrases: list[dict[str, Any]]) -> None:
    history: list[dict[str, Any]] = [
        {
            "drill": "negation-er",
            "verb": "aimer",
            "rest": "le café",
            "person": "je",
            "ok": False,
            "session": "s",
        }
        for _ in range(6)
    ]
    history += [
        {"drill": "negation-er", "verb": c["inf"], "rest": c["fr"], "person": p, "ok": True, "session": "s"}
        for c in phrases
        for p in drill.PERSONS
    ]
    hits: int = sum(
        any(
            i["verb"] == "aimer" and i["rest"] == "le café" and i["person"] == "je"
            for i in negation.pick(phrases, history, rng=random.Random(s))
        )
        for s in range(40)
    )
    assert hits >= 20


def test_http_flow(live: Live, tmp_path: Path) -> None:
    port: int = live.port
    s: int
    page: Any
    s, page = call(port, "GET", "/")
    assert s == 200 and "tab-neg" in page and "/negation/check" in page
    sess: dict[str, Any]
    s, sess = call(port, "GET", "/negation/session")
    assert s == 200 and len(sess["items"]) == 15
    assert set(sess["items"][0]) == {"id", "verb", "rest", "ru", "pronoun"}
    it: dict[str, Any] = sess["items"][0]
    full: str = negation.sentence(it["verb"], it["pronoun"], it["rest"])
    r: dict[str, Any]
    s, r = call(
        port,
        "POST",
        "/negation/check",
        {"session": sess["session"], "id": it["id"], "answer": full, "ms": 3000},
    )
    assert s == 200 and r == {"ok": True, "kind": None, "expected": full}
    it2: dict[str, Any] = sess["items"][1]
    answer: str = drill.phrase(it2["verb"], it2["pronoun"])
    s, r = call(
        port, "POST", "/negation/check", {"session": sess["session"], "id": it2["id"], "answer": answer}
    )
    assert r["ok"] is False and r["kind"] in ("ne", "pas")
    assert call(port, "POST", "/negation/check", {"session": "nope", "id": "0", "answer": "a"})[0] == 404
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "negation.jsonl")
    assert len(rows) == 2 and rows[0]["ok"] and rows[0]["drill"] == "negation-er"
    assert rows[0]["rest"] == it["rest"] and rows[0]["ms"] == 3000
    st: dict[str, Any]
    s, st = call(port, "GET", "/negation/stats")
    assert st["answers"] == 2 and st["ok"] == 1 and st["sessions"] == 1
