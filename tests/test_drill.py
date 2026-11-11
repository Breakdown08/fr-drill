"""Irregular-free present tense of -er verbs: forms by rule, grading, picking, HTTP on temp files."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import Live
from util import call

from fr_drill import drill, read_jsonl

FORMS: dict[tuple[str, str], str] = {
    ("parler", "je"): "je parle",
    ("parler", "tu"): "tu parles",
    ("parler", "on"): "on parle",
    ("parler", "nous"): "nous parlons",
    ("parler", "vous"): "vous parlez",
    ("parler", "elles"): "elles parlent",
    ("aimer", "je"): "j'aime",
    ("habiter", "je"): "j'habite",
    ("écouter", "je"): "j'écoute",
    ("étudier", "il"): "il étudie",
    ("manger", "nous"): "nous mangeons",
    ("manger", "ils"): "ils mangent",
    ("commencer", "nous"): "nous commençons",
    ("commencer", "vous"): "vous commencez",
    ("voyager", "je"): "je voyage",
    ("jouer", "nous"): "nous jouons",
}
GRADES: list[tuple[str, str, str, bool, str | None]] = [
    ("Nous  Parlons ", "parler", "nous", True, None),
    ("j’aime", "aimer", "je", True, None),
    ("j' aime", "aimer", "je", True, None),
    ("je aime", "aimer", "je", False, "elision"),
    ("j'parle", "parler", "je", False, "elision"),
    ("il etudie", "étudier", "il", False, "accent"),
    ("nous mangons", "manger", "nous", False, "spelling"),
    ("nous commencons", "commencer", "nous", False, "spelling"),
    ("tu parle", "parler", "tu", False, "ending"),
    ("ils parlont", "parler", "ils", False, "ending"),
    ("vous parlons", "parler", "vous", False, "ending"),
    ("nous parlons", "parler", "vous", False, "pronoun"),
    ("vous dites", "parler", "vous", False, "other"),
]


def test_verb_list_is_valid(verbs: list[dict[str, Any]]) -> None:
    assert len(verbs) >= 30


@pytest.mark.parametrize(("verb", "person"), list(FORMS))
def test_phrase_by_rule(verb: str, person: str) -> None:
    assert drill.phrase(verb, person) == FORMS[(verb, person)]


@pytest.mark.parametrize(("answer", "verb", "person", "ok", "kind"), GRADES)
def test_grade(answer: str, verb: str, person: str, ok: bool, kind: str | None) -> None:
    assert drill.grade(answer, verb, person) == (ok, kind)


def test_bad_verb_list_names_every_problem(tmp_path: Path) -> None:
    bad: Path = tmp_path / "v.json"
    verbs: list[dict[str, str]] = [
        {"inf": "acheter", "ru": "x"},
        {"inf": "préférer", "ru": "x"},
        {"inf": "finir", "ru": "x"},
        {"inf": "parler"},
        {"inf": "aller", "ru": "x"},
    ]
    bad.write_text(json.dumps({"verbs": verbs}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        drill.load(bad)
    m: str = str(err.value)
    part: str
    for part in (
        "'acheter' changes",
        "'préférer' changes",
        "'finir' is not",
        "'parler' has no translation",
        "'aller' is not",
    ):
        assert part in m


def test_fresh_session(verbs: list[dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = drill.pick(verbs, [], rng=random.Random(1))
    per: Counter[str] = Counter(i["person"] for i in items)
    assert len(items) == 20 and min(per.values()) >= 3
    assert len({i["verb"] for i in items}) >= 15
    assert [i["id"] for i in items] == [str(n) for n in range(20)]


def past_mistakes(verbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {"drill": "present-er", "verb": "manger", "person": "nous", "ok": False, "session": "s"}
        for _ in range(6)
    ]
    history += [
        {"drill": "present-er", "verb": v["inf"], "person": p, "ok": True, "session": "s"}
        for v in verbs
        for p in drill.PERSONS
        if (v["inf"], p) != ("manger", "nous")
    ]
    return history


def test_past_mistake_comes_back_much_more_often(verbs: list[dict[str, Any]]) -> None:
    def has(history: list[dict[str, Any]], seed: int) -> bool:
        items: list[dict[str, Any]] = drill.pick(verbs, history, rng=random.Random(seed))
        return any(i["verb"] == "manger" and i["person"] == "nous" for i in items)

    history: list[dict[str, Any]] = past_mistakes(verbs)
    hits: int = sum(has(history, s) for s in range(40))
    base: int = sum(has([], s) for s in range(40))
    assert hits >= 30 and hits > 3 * base


def test_weakest_person_gets_the_extra_turns(verbs: list[dict[str, Any]]) -> None:
    history: list[dict[str, Any]] = past_mistakes(verbs)
    weak: Counter[str] = Counter(
        i["person"] for s in range(20) for i in drill.pick(verbs, history, rng=random.Random(s))
    )
    assert weak["nous"] == max(weak.values())


def test_drill_writes_its_own_file(tmp_path: Path, verbs: list[dict[str, Any]]) -> None:
    d: Any = drill.Drill(verbs, tmp_path / "attempts" / "drill.jsonl")
    sess: dict[str, Any] = d.session(3)
    it: dict[str, Any] = sess["items"][0]
    d.check({"session": sess["session"], "id": it["id"], "answer": drill.phrase(it["verb"], it["pronoun"])})
    assert len(read_jsonl(tmp_path / "attempts" / "drill.jsonl")) == 1


def test_http_flow(live: Live, tmp_path: Path) -> None:
    port: int = live.port
    s: int
    page: Any
    s, page = call(port, "GET", "/")
    assert s == 200 and "<title>Français A1</title>" in page and "tab-words" in page
    assert call(port, "GET", "/session", host="evil.com")[0] == 403
    sess: dict[str, Any]
    s, sess = call(port, "GET", "/session?n=6")
    assert s == 200 and len(sess["items"]) == 6
    assert set(sess["items"][0]) == {"id", "verb", "ru", "pronoun"}
    it: dict[str, Any] = sess["items"][0]
    right: str = drill.phrase(it["verb"], it["pronoun"])
    r: dict[str, Any]
    s, r = call(
        port, "POST", "/check", {"session": sess["session"], "id": it["id"], "answer": right, "ms": 2100}
    )
    assert s == 200 and r == {"ok": True, "kind": None, "expected": right}
    it2: dict[str, Any] = sess["items"][1]
    s, r = call(port, "POST", "/check", {"session": sess["session"], "id": it2["id"], "answer": "xx yy"})
    assert r["ok"] is False and r["expected"] == drill.phrase(it2["verb"], it2["pronoun"])
    assert call(port, "POST", "/check", {"session": "x"}, ctype="text/plain")[0] == 415
    assert call(port, "POST", "/check", {"session": "nope", "id": "0", "answer": "a"})[0] == 404
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "drill.jsonl")
    assert len(rows) == 2 and rows[0]["ok"] and rows[0]["ms"] == 2100
    assert rows[1]["kind"] == "pronoun" and rows[0]["drill"] == "present-er"
    st: dict[str, Any]
    s, st = call(port, "GET", "/stats")
    assert st["answers"] == 2 and st["ok"] == 1 and st["sessions"] == 1 and st["kinds"] == {"pronoun": 1}


def test_page_has_no_private_names() -> None:
    page: str = drill.PAGE.read_text(encoding="utf-8").lower()
    assert "kad" + "ath" not in page and "ny" + "arl" not in page
