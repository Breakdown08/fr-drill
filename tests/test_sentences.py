"""Sentences tab: the bank, what opens, grading, picking, HTTP on temp attempts files."""

import json
import random
from collections import Counter
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from conftest import build_trainers
from util import NEEDS_VOCAB, call, serving

from fr_drill import drill, sentences

type Sentence = dict[str, Any]

pytestmark: pytest.MarkDecorator = NEEDS_VOCAB


@pytest.fixture(scope="module")
def bank(verbs: list[dict[str, Any]], cards: list[dict[str, Any]]) -> list[Sentence]:
    return sentences.load(verbs=verbs, cards=cards)


@pytest.fixture(scope="module")
def by_fr(bank: list[Sentence]) -> dict[str, Sentence]:
    return {s["fr"]: s for s in bank}


def test_bank_is_valid(bank: list[Sentence]) -> None:
    assert len(bank) >= 80


@pytest.mark.parametrize(
    ("fr", "want"),
    [
        ("Je n'aime pas le café.", ["je", "n'aime", "pas", "le", "café"]),
        ("Tu es en vacances ?", ["tu", "es", "en", "vacances"]),
        ("Bonne nuit, Marie.", ["bonne", "nuit", "Marie"]),
        ("Il n'arrive pas à l'heure.", ["il", "n'arrive", "pas", "à", "l'", "heure"]),
    ],
)
def test_tiles(fr: str, want: list[str]) -> None:
    assert sentences.tiles(fr) == want


def test_tiles_details() -> None:
    assert sentences.tiles("Marie est architecte.")[0] == "Marie"
    assert sentences.tiles("J’aime le café.")[0] == "j'aime"
    assert sentences.tiles("Vous êtes d'accord ?")[-2:] == ["d'", "accord"]
    assert sentences.tiles("C'est tout, s'il te plaît.")[:3] == ["c'est", "tout", "s'il"]


def test_no_sentence_offers_a_link_as_a_wrong_form(bank: list[Sentence]) -> None:
    assert not sentences.LINKS & {w for s in bank for w in s["wrong"]}


def test_every_sentence_can_be_built_and_the_wrong_tiles_cannot(bank: list[Sentence]) -> None:
    problems: list[str] = []
    s: Sentence
    for s in bank:
        built: list[str] = sentences.tiles(s["fr"])
        if not sentences.grade(built, s)[0]:
            problems.append(f"built {s['fr']}")
        other: str
        for other in s.get("accept", []):
            if not sentences.grade(sentences.tiles(other), s)[0]:
                problems.append(f"accept {other}")
        wrong: str
        for wrong in s["wrong"]:
            if sentences.grade(built[:-1] + [wrong], s) != (False, "form"):
                problems.append(f"wrong tile {wrong} in {s['fr']}")
    assert problems == []


def test_grade_kinds(by_fr: dict[str, Sentence]) -> None:
    s: Sentence = by_fr["J'aime le café."]
    assert sentences.grade(["j'aime", "le", "café"], s) == (True, None)
    assert sentences.grade(["je", "aime", "le", "café"], s) == (False, "form")
    assert sentences.grade(["le", "café", "j'aime"], s) == (False, "order")
    assert sentences.grade(["j'aime", "café"], s) == (False, "link")
    assert sentences.grade(["j'aime", "la", "café"], s) == (False, "link")
    assert sentences.grade(["j'aime"], s) == (False, "missing")
    assert sentences.grade(["j'aime", "le", "beurre"], s) == (False, "word")
    assert sentences.grade(["café", "le", "j'aime"], s) == (False, "order")
    s = by_fr["Je trouve facilement la gare."]
    assert sentences.grade(["je", "trouve", "la", "gare", "facilement"], s) == (True, None)


def refusal(sentence: Sentence, verbs: list[dict[str, Any]], cards: list[dict[str, Any]], tmp: Path) -> str:
    p: Path = tmp / "s.json"
    core: Any = json.loads(sentences.BANK.read_text(encoding="utf-8"))["core"]
    p.write_text(json.dumps({"core": core, "sentences": [sentence]}), encoding="utf-8")
    try:
        sentences.load(p, verbs, cards)
    except ValueError as e:
        return str(e)
    return ""


BASE: Sentence = {"fr": "Je parle anglais.", "ru": "x", "verbs": ["parler"], "wrong": ["parles"]}


def test_load_accepts_the_base_sentence(
    verbs: list[dict[str, Any]], cards: list[dict[str, Any]], tmp_path: Path
) -> None:
    assert refusal(BASE, verbs, cards, tmp_path) == ""


@pytest.mark.parametrize(
    ("change", "want"),
    [
        ({"fr": "Je parle hongrois."}, "not learned"),
        ({"verbs": []}, "needs 'parler'"),
        ({"verbs": ["parler", "aimer"]}, "no form of 'aimer'"),
        ({"words": ["le dragon"]}, "not a card"),
        ({"words": ["le beurre"]}, "not in the sentence"),
        ({"wrong": ["parle"]}, "is in the sentence"),
        ({"wrong": ["au"]}, "is in the panel"),
        ({"wrong": []}, "no wrong tiles"),
        ({"accept": ["Je parle."]}, "needs other tiles"),
    ],
)
def test_load_refuses(
    change: Sentence, want: str, verbs: list[dict[str, Any]], cards: list[dict[str, Any]], tmp_path: Path
) -> None:
    assert want in refusal({**BASE, **change}, verbs, cards, tmp_path)


def test_noun_homograph_of_a_verb_form_is_fine(
    verbs: list[dict[str, Any]], cards: list[dict[str, Any]], tmp_path: Path
) -> None:
    s: Sentence = {"fr": "La cuisine est ici.", "ru": "x", "verbs": ["être"], "wrong": ["sont"]}
    assert refusal(s, verbs, cards, tmp_path) == ""


def test_what_opens() -> None:
    few: list[Sentence] = [
        {"fr": "a", "verbs": ["parler"]},
        {"fr": "b", "verbs": ["parler"], "words": ["le beurre"]},
        {"fr": "c", "words": ["la table"]},
    ]
    assert [s["fr"] for s in sentences.opened(few, set(), {"le beurre"})] == []
    assert [s["fr"] for s in sentences.opened(few, {"parler"}, {"le beurre"})] == ["a", "b"]
    verb_rows: list[dict[str, Any]] = [{"verb": "parler", "ok": False}, {"verb": "aimer", "ok": True}]
    words: list[dict[str, Any]] = [
        {"word": "le beurre", "at": "2026-11-05T10:00", "ok": True},
        {"word": "la table", "at": "2026-11-05T10:00", "ok": False},
    ]
    assert sentences.learned(verb_rows, words) == ({"aimer"}, {"le beurre"})


def test_decoys(by_fr: dict[str, Sentence]) -> None:
    s: Sentence = by_fr["Tu aimes le raisin ?"]
    assert sentences.person(s) == "tu"
    assert sentences.person(by_fr["Nous ne sommes pas en retard."]) == "nous"
    assert sentences.person(by_fr["Bonne nuit, Marie."]) == "il"
    learned_verbs: set[str] = {"aimer", "parler", "fermer", "chercher", "être"}
    got: list[str] = sentences.decoys(
        s, learned_verbs, {"beurre", "raisin", "gare", "maison"}, random.Random(1)
    )
    assert set(got[:2]) <= {"parles", "fermes", "cherches"} and len(got) == 4
    assert set(got[2:]) <= {"beurre", "gare", "maison"}
    assert not {"es", "as"} & set(got)
    assert sentences.decoys(s, {"aimer"}, {"raisin"}, random.Random(1)) == []


def test_noun_lists() -> None:
    nouns: set[str] = sentences.met_nouns() | sentences.card_nouns(
        {"le beurre", "l'étoile", "pardon", "s'il te plaît"}
    )
    assert {"maison", "café", "tête", "heure", "beurre", "étoile"} <= nouns
    assert not nouns & (sentences.LINKS | {"pardon", "plaît", "comme"})


def test_every_decoy_is_graded_as_a_word(bank: list[Sentence], verbs: list[dict[str, Any]]) -> None:
    nouns: set[str] = sentences.met_nouns() | sentences.card_nouns(
        {"le beurre", "l'étoile", "pardon", "s'il te plaît"}
    )
    problems: list[str] = []
    s: Sentence
    for s in bank:
        seed: int
        for seed in range(5):
            d: str
            for d in sentences.decoys(s, {v["inf"] for v in verbs}, nouns, random.Random(seed)):
                if sentences.grade(sentences.tiles(s["fr"])[:-1] + [d], s) != (False, "word"):
                    problems.append(f"decoy {d} in {s['fr']}")
    assert problems == []


@pytest.fixture(scope="module")
def forms(verbs: list[dict[str, Any]]) -> dict[str, str]:
    return {f: inf for f, inf in sentences.verb_forms(verbs).items() if f not in sentences.met_words()}


RULES: list[tuple[str, list[str], str]] = [
    ("Elle est satisfaite.", ["elle", "est", "satisfait"], "-e"),
    ("Je suis à la maison.", ["je", "suis", "la", "maison"], "à"),
    ("Il n'arrive pas à l'heure.", ["il", "n'arrive", "pas", "froid"], "вовремя"),
    ("Elle ne travaille pas le samedi.", ["elle", "ne", "travaille", "pas", "du", "samedi"], "по субботам"),
    ("Je joue au football.", ["je", "joue", "football"], "jouer à"),
    ("Elle a une voiture.", ["elle", "a", "voiture"], "артикль"),
    ("Il y a un message.", ["il", "y", "a", "message"], "После il y a"),
    ("Il est au bureau.", ["il", "est", "bureau"], "au = à + le"),
    ("C'est un message.", ["c'est", "une", "message"], "Род"),
    ("Il déteste le beurre.", ["il", "déteste", "beurre"], "détester"),
    ("Nous avons soif.", ["nous", "avons", "chien"], "avoir soif"),
    ("Elle a froid.", ["elle", "est", "froid"], "avoir"),
    ("Marie est architecte.", ["Marie", "a", "architecte"], "être"),
    ("Tu téléphones à Marie ?", ["tu", "téléphone", "à", "Marie"], "лицо другое"),
    ("J'aime le café.", ["je", "aime", "le", "café"], "j'"),
    ("Il ne chante pas bien.", ["il", "chante", "ne", "pas", "bien"], "ne + глагол + pas"),
    ("Il y a une table dans la cuisine.", ["est", "une", "table", "dans", "la", "cuisine"], "il y a"),
    ("J'aime le café.", ["j'aime", "le", "beurre"], "beurre"),
]


def rules(by_fr: dict[str, Sentence], forms: dict[str, str], fr: str, built: list[str]) -> str:
    s: Sentence = by_fr[fr]
    return " ".join(sentences.rules_for(built, s, sentences.grade(built, s)[1], forms))


@pytest.mark.parametrize(("fr", "built", "want"), RULES, ids=[f"{r[0]}|{r[2]}" for r in RULES])
def test_rules_after_a_mistake(
    by_fr: dict[str, Sentence], forms: dict[str, str], fr: str, built: list[str], want: str
) -> None:
    assert want in rules(by_fr, forms, fr, built)


def test_rules_in_detail(by_fr: dict[str, Sentence], forms: dict[str, str]) -> None:
    got: str = rules(
        by_fr, forms, "Le shampooing est dans la cuisine.", ["le", "shampooing", "la", "cuisine"]
    )
    assert "нужен глагол" in got and "dans" in got
    assert "там est" not in rules(by_fr, forms, "J'ai mal au pied.", ["pied"])
    got = rules(
        by_fr,
        forms,
        "J'ai un chien et tu as une voiture.",
        ["j'ai", "le", "chien", "et", "tu", "es", "la", "voiture"],
    )
    assert "есть у кого-то" in got.lower()
    assert "Род" not in got and "впервые" in got  # definite is no gender mistake
    got = rules(by_fr, forms, "Nous ne sommes pas en retard.", ["nous", "sommes", "retard"])
    assert "ne + глагол + pas" in got and "опаздывать" in got
    got = rules(by_fr, forms, "Tu téléphones à Marie ?", ["tu", "téléphone", "à", "Marie"])
    assert "téléphoner" not in got  # an own rule stays out of a wrong form
    assert "без артикля" not in rules(by_fr, forms, "Marie est architecte.", ["Marie", "a", "architecte"])


def test_every_mistake_has_one_or_two_rules(bank: list[Sentence], forms: dict[str, str]) -> None:
    problems: list[str] = []
    s: Sentence
    for s in bank:
        built: list[str] = sentences.tiles(s["fr"])
        tries: list[list[str]] = [built[:i] + built[i + 1 :] for i in range(len(built))]
        tries += [built[:-1] + [w] for w in s["wrong"]]
        bad_answer: list[str]
        for bad_answer in tries:
            kind: Any = sentences.grade(bad_answer, s)[1]
            got: list[str] = sentences.rules_for(bad_answer, s, kind, forms) if kind else ["ok"]
            if not 1 <= len(got) <= sentences.MAX_RULES:
                problems.append(f"rules for {bad_answer}: {got}")
    assert problems == []


def test_weights() -> None:
    rows: list[dict[str, Any]] = [
        {"fr": "x", "ok": False},
        {"fr": "y", "ok": True},
        {"fr": "y", "ok": True},
        {"fr": "z", "ok": False},
        {"fr": "z", "ok": True},
    ]
    assert sentences.weight("x", rows) == 8  # wrong weighs most
    assert sentences.weight("new", rows) == 4  # unseen
    assert sentences.weight("y", rows) == 1  # two right
    assert sentences.weight("z", rows) == 2  # a streak breaks at a miss


def test_pick() -> None:
    pool: list[Sentence] = [{"fr": str(i)} for i in range(30)]
    history: list[dict[str, Any]] = [{"fr": str(i), "ok": True} for i in range(29) for _ in range(3)]
    got: list[Sentence] = sentences.pick(pool, history, rng=random.Random(0))
    assert len(got) == 10 == len({s["fr"] for s in got})
    hits: Counter[str] = Counter(
        s["fr"] for seed in range(300) for s in sentences.pick(pool, history, rng=random.Random(seed))
    )
    assert hits["29"] > 2 * hits["0"]
    assert len(sentences.pick(pool[:3], [])) == 3


def test_holes_and_link_hints(by_fr: dict[str, Sentence], verbs: list[dict[str, Any]]) -> None:
    assert sentences.holes(["il", "est", "au", "bureau"]) == ["il", "est", None, "bureau"]
    match: Sentence = by_fr["Ils regardent le match."]
    assert sentences.grade(sentences.tiles("Ils regardent un match."), match) == (
        True,
        None,
    )  # `also`: both articles right
    errors: list[str] = sentences.problems(
        {"fr": "J'aime le café.", "also": ["J'aime le thé."], "wrong": ["aime"], "verbs": ["aimer"]},
        set(),
        {"le", "café", "thé"},
        {},
        sentences.verb_forms(verbs),
    )
    assert any("may differ only" in e for e in errors)
    forms: dict[str, str] = sentences.verb_forms(verbs)

    def hint(fr: str, wrong: str) -> str:
        return " ".join(sentences.gap_rules(sentences.tiles(wrong), by_fr[fr], forms))

    got: str = hint("Elle a une voiture.", "Elle a un voiture.")
    assert "Прилагательное" not in got and "Род другой" in got
    assert "множественное" in hint("Elle aide les voisins.", "Elle aide le voisins.")
    gare: str = (
        "Je cherche la gare." if "Je cherche la gare." in by_fr else "Excusez-moi, je cherche la gare."
    )
    assert "гласной" in hint(gare, gare.replace("la gare", "l' gare"))
    got = hint("Nous ne sommes pas en retard.", "Nous ne sommes pas à retard.")
    assert "en, а не à" in got and "готовое выражение" in got
    assert "не на своих местах" in hint("Marie pleure dans le train.", "Marie pleure le dans train")


def test_http_sessions_checks_and_gaps(
    bank: list[Sentence], verbs: list[dict[str, Any]], cards: list[dict[str, Any]], tmp_path: Path
) -> None:
    (tmp_path / "verbs.jsonl").write_text(json.dumps({"verb": "aimer", "ok": True}) + "\n", encoding="utf-8")
    (tmp_path / "words.jsonl").write_text("", encoding="utf-8")
    t: dict[str, Any] = build_trainers(tmp_path, verbs, cards)
    live: Any = sentences.Sentences(
        bank, tmp_path / "sentences.jsonl", (tmp_path / "verbs.jsonl",), tmp_path / "words.jsonl"
    )

    def make(p: int) -> type[BaseHTTPRequestHandler]:
        handler: type[BaseHTTPRequestHandler] = drill.handler(
            t["drill"], p, t["vocab"], t["numbers"], t["negation"], t["etre"], live, t["preps"]
        )
        return handler

    port: int
    with serving(make) as port:
        status: int
        data: dict[str, Any]
        status, data = call(port, "GET", "/sentences/session")
        want: dict[str, Sentence] = {
            s["ru"]: s for s in bank if s.get("verbs") == ["aimer"] and not s.get("words")
        }
        assert status == 200
        assert {i["ru"] for i in data["items"]} == set(want)  # only what is open
        assert data["open"] == len(want) and data["total"] == len(bank)
        item: dict[str, Any] = data["items"][0]
        s: Sentence = want[item["ru"]]
        assert set(item) == {"id", "ru", "tiles"}
        own: list[str] = [x for x in sentences.tiles(s["fr"]) if x not in sentences.LINKS]
        assert not Counter(own + s["wrong"]) - Counter(item["tiles"])
        assert len(item["tiles"]) == len(own + s["wrong"]) + sentences.DECOY_NOUNS
        assert not sentences.LINKS & set(item["tiles"])
        assert set(data["articles"] + data["prepositions"]) == sentences.LINKS
        res: dict[str, Any]
        status, res = call(
            port,
            "POST",
            "/sentences/check",
            {"session": data["session"], "id": item["id"], "tiles": sentences.tiles(s["fr"]), "ms": 900},
        )
        assert status == 200 and res == {"ok": True, "kind": None, "expected": s["fr"], "rules": []}
        item = data["items"][1]
        s = want[item["ru"]]
        status, res = call(
            port,
            "POST",
            "/sentences/check",
            {"session": data["session"], "id": item["id"], "tiles": sentences.tiles(s["fr"])[::-1]},
        )
        assert res["ok"] is False and res["kind"] == "order" and res["rules"]
        assert (
            call(
                port,
                "POST",
                "/sentences/check",
                {"session": data["session"], "id": item["id"], "tiles": "je"},
            )[0]
            != 200
        )
        assert call(port, "POST", "/sentences/check", {"session": "none", "id": "0", "tiles": []})[0] != 200
        rows: list[dict[str, Any]] = [
            json.loads(x) for x in (tmp_path / "sentences.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 2 and rows[0]["fr"] and rows[1]["kind"] == "order"

        gaps: dict[str, Any]
        status, gaps = call(port, "GET", "/sentences/gaps/session")
        assert status == 200 and {i["ru"] for i in gaps["items"]} == set(want)
        assert set(gaps["items"][0]) == {"id", "ru", "tiles"}
        assert not sentences.LINKS & {t for i in gaps["items"] for t in i["tiles"]}
        assert set(gaps["articles"] + gaps["prepositions"]) == sentences.LINKS
        item = gaps["items"][0]
        s = want[item["ru"]]
        links: list[str] = [x for x in sentences.tiles(s["fr"]) if x in sentences.LINKS]
        status, res = call(
            port,
            "POST",
            "/sentences/gaps/check",
            {"session": gaps["session"], "id": item["id"], "fills": links, "ms": 700},
        )
        assert status == 200 and res == {"ok": True, "expected": s["fr"], "rules": []}
        status, res = call(
            port,
            "POST",
            "/sentences/gaps/check",
            {"session": gaps["session"], "id": item["id"], "fills": ["sur"] * len(links)},
        )
        assert res["ok"] is False and res["rules"]
        status, res = call(
            port,
            "POST",
            "/sentences/gaps/check",
            {"session": gaps["session"], "id": item["id"], "fills": ["chien"] * len(links)},
        )
        assert status != 200  # only links fill a gap
        status, res = call(
            port, "POST", "/sentences/gaps/check", {"session": gaps["session"], "id": item["id"], "fills": []}
        )
        assert status != 200  # every gap must be filled
        assert (
            call(port, "POST", "/sentences/gaps/check", {"session": "none", "id": "0", "fills": []})[0] != 200
        )
        st: dict[str, Any]
        status, st = call(port, "GET", "/sentences/gaps/stats")
        assert st["answers"] == 2 and st["ok"] == 1 and st["pairs"][0]["got"] == "sur"
        assert (tmp_path / "gaps.jsonl").exists()  # its own history, full sentences untouched
        assert len((tmp_path / "sentences.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        status, st = call(port, "GET", "/sentences/stats")
        assert st["answers"] == 2 and st["ok"] == 1 and st["kinds"] == {"order": 1}
        live.sources = (tmp_path / "empty.jsonl",)
        status, data = call(port, "GET", "/sentences/session")
        assert status == 200 and data["items"] == [] and data["open"] == 0
