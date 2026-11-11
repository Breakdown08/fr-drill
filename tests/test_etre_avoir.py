"""Irregular verbs tab: forms, grading, picking, HTTP on a temp attempts file."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import Live
from util import call

from fr_drill import drill, etre_avoir, read_jsonl

HEADS: dict[tuple[str, str], str] = {
    ("être", "je"): "je suis",
    ("être", "tu"): "tu es",
    ("être", "on"): "on est",
    ("être", "nous"): "nous sommes",
    ("être", "vous"): "vous êtes",
    ("être", "elles"): "elles sont",
    ("avoir", "je"): "j'ai",
    ("avoir", "tu"): "tu as",
    ("avoir", "elle"): "elle a",
    ("avoir", "nous"): "nous avons",
    ("avoir", "vous"): "vous avez",
    ("avoir", "ils"): "ils ont",
    ("aller", "je"): "je vais",
    ("aller", "tu"): "tu vas",
    ("aller", "on"): "on va",
    ("aller", "nous"): "nous allons",
    ("aller", "vous"): "vous allez",
    ("aller", "elles"): "elles vont",
    ("faire", "je"): "je fais",
    ("faire", "vous"): "vous faites",
    ("faire", "ils"): "ils font",
    ("prendre", "il"): "il prend",
    ("prendre", "nous"): "nous prenons",
    ("prendre", "elles"): "elles prennent",
    ("venir", "tu"): "tu viens",
    ("venir", "vous"): "vous venez",
    ("venir", "ils"): "ils viennent",
}
GRADES: list[tuple[str, str, str, str, bool, str | None]] = [
    ("j'ai", "avoir", "je", "faim", True, None),
    ("J’ai faim.", "avoir", "je", "faim", True, None),
    ("j' ai  faim", "avoir", "je", "faim", True, None),
    ("vous êtes en retard !", "être", "vous", "en retard", True, None),
    ("elle a vingt ans", "avoir", "elle", "vingt ans", True, None),
    ("je ai faim", "avoir", "je", "faim", False, "elision"),
    ("je suis faim", "avoir", "je", "faim", False, "verb"),
    ("tu es vingt ans", "avoir", "tu", "vingt ans", False, "verb"),
    ("nous avons à Paris", "être", "nous", "à Paris", False, "verb"),
    ("il as faim", "avoir", "il", "faim", False, "person"),
    ("tu est ici", "être", "tu", "ici", False, "person"),
    ("j'est ici", "être", "je", "ici", False, "person"),
    ("j'as faim", "avoir", "je", "faim", False, "person"),
    ("vous etes ici", "être", "vous", "ici", False, "accent"),
    ("il à faim", "avoir", "il", "faim", False, "accent"),
    ("tu suis ici", "être", "vous", "ici", False, "pronoun"),
    ("ils sommes ici", "être", "ils", "ici", False, "person"),
    ("ils mangent ici", "être", "ils", "ici", False, "other"),
    ("ils font ici", "être", "ils", "ici", False, "verb"),
    ("ils prennent le bus", "prendre", "ils", "le bus", True, None),
    ("ils prenent le bus", "prendre", "ils", "le bus", False, "other"),
    ("ils vont le bus", "prendre", "ils", "le bus", False, "verb"),
    ("vous faites du sport", "faire", "vous", "du sport", True, None),
    ("vous faisez du sport", "faire", "vous", "du sport", False, "other"),
    ("nous faisons", "faire", "nous", "la cuisine", True, None),
    ("elle viens demain", "venir", "elle", "demain", False, "person"),
    ("je viens ici", "venir", "je", "ici", True, None),
    ("je suis ici", "venir", "je", "ici", False, "verb"),
    ("ils vont ici", "être", "ils", "ici", False, "verb"),
    ("je vais au travail", "aller", "je", "au travail", True, None),
    ("nous allons", "aller", "nous", "à Paris", True, None),
    ("je suis au travail", "aller", "je", "au travail", False, "verb"),
    ("tu va bien", "aller", "tu", "bien", False, "person"),
    ("ils allons", "aller", "ils", "mal", False, "person"),
    ("ils", "être", "ils", "ici", False, "other"),
    ("", "être", "ils", "ici", False, "other"),
    ("ils sont ic", "être", "ils", "ici", False, "tail"),
]


@pytest.fixture(scope="module")
def phrases() -> list[dict[str, Any]]:
    return etre_avoir.load()


def test_phrases_cover_every_verb(phrases: list[dict[str, Any]]) -> None:
    assert len(phrases) >= 30
    per_verb: Counter[str] = Counter(p["verb"] for p in phrases)
    assert set(per_verb) == set(etre_avoir.FORMS) and min(per_verb.values()) >= 15


@pytest.mark.parametrize(("verb", "person"), list(HEADS))
def test_head(verb: str, person: str) -> None:
    assert etre_avoir.head(verb, person) == HEADS[(verb, person)]


@pytest.mark.parametrize(("answer", "verb", "person", "rest", "ok", "kind"), GRADES)
def test_grade(answer: str, verb: str, person: str, rest: str, ok: bool, kind: str | None) -> None:
    assert etre_avoir.grade(answer, verb, person, rest) == (ok, kind)


def test_bad_list_names_every_problem(tmp_path: Path) -> None:
    bad: Path = tmp_path / "x.json"
    contexts: list[dict[str, Any]] = [
        {"verb": "voir", "fr": "à Paris", "ru": "идти"},
        {"verb": "avoir", "fr": "chaud", "ru": "жарко", "lit": 1},
        {"verb": "avoir", "fr": "mon stylo", "ru": "иметь ручку"},
        {"verb": "avoir", "fr": "", "ru": "иметь"},
        {"verb": "avoir", "fr": "faim", "ru": ""},
        {"verb": "avoir", "fr": "soif", "ru": "пить"},
        {"verb": "avoir", "fr": "soif", "ru": "пить"},
    ]
    bad.write_text(json.dumps({"contexts": contexts}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        etre_avoir.load(bad)
    m: str = str(err.value)
    part: str
    for part in (
        "is not one of",
        "changes with the person",
        "no rest",
        "no translation",
        "twice",
        "lit is not",
        "no contexts for être",
    ):
        assert part in m


def test_fresh_session(phrases: list[dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = etre_avoir.pick(phrases, [], rng=random.Random(1))
    per: Counter[str] = Counter(i["person"] for i in items)
    assert len(items) == 18 and set(per.values()) == {3}
    assert Counter(i["verb"] for i in items) == {v: 3 for v in etre_avoir.FORMS}
    assert len({i["rest"] for i in items}) >= 15


def test_one_verb_session(phrases: list[dict[str, Any]]) -> None:
    only: list[dict[str, Any]] = etre_avoir.pick(phrases, [], rng=random.Random(1), verbs=("avoir",))
    per: Counter[str] = Counter(i["person"] for i in only)
    assert {i["verb"] for i in only} == {"avoir"} and set(per.values()) == {3}


def test_other_verbs_mistakes_do_not_tilt_the_persons(phrases: list[dict[str, Any]]) -> None:
    weak: list[dict[str, Any]] = [
        {
            "drill": "etre-avoir",
            "verb": "être",
            "rest": "en retard",
            "person": "nous",
            "ok": False,
            "session": "s",
        }
        for _ in range(30)
    ]
    s: int
    for s in range(10):
        got: list[dict[str, Any]] = etre_avoir.pick(phrases, weak, rng=random.Random(s), verbs=("avoir",))
        assert Counter(i["person"] for i in got) == {p: 3 for p in drill.PERSONS}


def test_past_mistake_comes_back(phrases: list[dict[str, Any]]) -> None:
    history: list[dict[str, Any]] = [
        {"drill": "etre-avoir", "verb": c["verb"], "rest": c["fr"], "person": p, "ok": True, "session": "s"}
        for c in phrases
        for p in drill.PERSONS
    ]
    # recent: only the last HISTORY answers count
    history += [
        {"drill": "etre-avoir", "verb": "avoir", "rest": "faim", "person": "je", "ok": False, "session": "s"}
        for _ in range(6)
    ]

    def comebacks(verbs: tuple[str, ...]) -> int:
        return sum(
            any(
                i["verb"] == "avoir" and i["rest"] == "faim" and i["person"] == "je"
                for i in etre_avoir.pick(phrases, history, rng=random.Random(s), verbs=verbs)
            )
            for s in range(40)
        )

    assert comebacks(("avoir",)) >= 32
    assert comebacks(tuple(etre_avoir.FORMS)) >= 10  # mixed sessions too: about a third


def test_http_flow(live: Live, phrases: list[dict[str, Any]], tmp_path: Path) -> None:
    port: int = live.port
    s: int
    page: Any
    s, page = call(port, "GET", "/")
    assert s == 200 and "tab-aux" in page and "/etre-avoir/check" in page
    sess: dict[str, Any]
    s, sess = call(port, "GET", "/etre-avoir/session")
    assert s == 200 and len(sess["items"]) == 18
    assert set(sess["items"][0]) == {"id", "verb", "rest", "ru", "lit", "pronoun"}
    it: dict[str, Any] = sess["items"][0]
    full: str = etre_avoir.sentence(it["verb"], it["pronoun"], it["rest"])
    r: dict[str, Any]
    s, r = call(
        port,
        "POST",
        "/etre-avoir/check",
        {"session": sess["session"], "id": it["id"], "answer": full, "ms": 2500},
    )
    assert s == 200 and r == {"ok": True, "kind": None, "expected": full, "lit": it["lit"]}
    it2: dict[str, Any] = sess["items"][1]
    other: str = "avoir" if it2["verb"] == "être" else "être"
    answer: str = etre_avoir.head(other, it2["pronoun"])
    s, r = call(
        port, "POST", "/etre-avoir/check", {"session": sess["session"], "id": it2["id"], "answer": answer}
    )
    assert r["ok"] is False and r["kind"] == "verb"
    assert call(port, "POST", "/etre-avoir/check", {"session": "nope", "id": "0", "answer": "a"})[0] == 404
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "etre-avoir.jsonl")
    assert len(rows) == 2 and rows[0]["ok"] and rows[0]["drill"] == "etre-avoir"
    assert rows[0]["rest"] == it["rest"] and rows[0]["ms"] == 2500
    st: dict[str, Any]
    s, st = call(port, "GET", "/etre-avoir/stats")
    assert st["answers"] == 2 and st["ok"] == 1 and st["sessions"] == 1
    assert sum(v["tries"] for v in st["verbs"].values()) == 2


def test_http_verb_filter_and_modes(live: Live, phrases: list[dict[str, Any]], tmp_path: Path) -> None:
    port: int = live.port
    s: int
    one: dict[str, Any]
    s, one = call(port, "GET", "/etre-avoir/session?verb=%C3%AAtre")
    assert s == 200 and {i["verb"] for i in one["items"]} == {"être"}
    both: dict[str, Any]
    s, both = call(port, "GET", "/etre-avoir/session")
    assert s == 200 and {i["verb"] for i in both["items"]} == set(etre_avoir.FORMS)
    al: dict[str, Any]
    s, al = call(port, "GET", "/etre-avoir/session?verb=aller")
    assert s == 200 and {i["verb"] for i in al["items"]} == {"aller"}
    assert call(port, "GET", "/etre-avoir/session?verb=voir")[0] == 400
    assert call(port, "GET", "/etre-avoir/session?mode=guess")[0] == 400
    ch: dict[str, Any]
    s, ch = call(port, "GET", "/etre-avoir/session?mode=choose&verb=avoir")
    assert s == 200 and ch["mode"] == "choose"
    assert all(set(i) == {"id", "rest", "ru", "pronoun"} for i in ch["items"])
    it: dict[str, Any] = ch["items"][0]
    want: dict[str, Any] = next(c for c in phrases if c["fr"] == it["rest"] and c["ru"] == it["ru"])
    r: dict[str, Any]
    answer: str = etre_avoir.head(want["verb"], it["pronoun"])
    s, r = call(
        port, "POST", "/etre-avoir/check", {"session": ch["session"], "id": it["id"], "answer": answer}
    )
    assert r["ok"] and r["lit"] == want.get("lit", "")
    rows: list[dict[str, Any]] = read_jsonl(tmp_path / "etre-avoir.jsonl")
    assert rows[-1]["mode"] == "choose"
    st: dict[str, Any]
    s, st = call(port, "GET", "/etre-avoir/stats")
    assert st["choose"] == {"tries": 1, "ok": 1}


def test_page_has_the_choose_option_and_the_renamed_tab(live: Live) -> None:
    page: Any = call(live.port, "GET", "/")[1]
    assert 'value="choose"' in page and "?mode=choose" in page
    assert "Неправильные глаголы" in page and 'id="x-verb"' in page
