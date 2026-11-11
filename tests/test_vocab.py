"""Words tab: cards, distractors, boxes, daily quota, HTTP on a temp attempts file."""

import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import Live
from util import FIXTURES, NEEDS_VOCAB, call

from fr_drill import vocab

TODAY: date = date(2026, 10, 25)

type Card = dict[str, Any]


def by(cards: list[Card]) -> dict[str, Card]:
    return {c["fr"]: c for c in cards}


def rows(words: list[str], day: date, ok: bool = True) -> list[dict[str, Any]]:
    return [{"at": f"{day.isoformat()}T10:00:00+07:00", "word": w, "ok": ok} for w in words]


def days(words: list[str], n: int, ok: bool = True) -> list[dict[str, Any]]:
    """The same words answered once a day for n days before today."""
    return [r for k in range(n, 0, -1) for r in rows(words, TODAY - timedelta(days=k), ok)]


# Fixture file and general rules: no real vocabulary needed.


def test_load_skips_lessons_with_their_own_drill() -> None:
    cards: list[Card] = vocab.load(FIXTURES / "vocab.json")
    assert {c["fr"] for c in cards} == {"le chat", "le chien", "la vache", "le cheval", "l'oiseau"}
    assert {c["lesson"] for c in cards} == {1}
    assert "deux" not in {c["fr"] for c in cards}
    assert cards[0]["topic"] == "Животные"


def test_load_refuses_an_elided_card_without_a_gender(tmp_path: Path) -> None:
    p: Path = tmp_path / "cards.json"
    words: list[dict[str, str]] = [
        {"fr": "l'ami", "ru": "друг"},
        {"fr": "a", "ru": "b"},
        {"fr": "c", "ru": "d"},
        {"fr": "e", "ru": "f"},
    ]
    p.write_text(json.dumps({"lessons": [{"n": 1, "ru": "t", "words": words}]}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        vocab.load(p)
    assert "needs a gender" in str(err.value)


def test_load_names_every_problem(tmp_path: Path) -> None:
    p: Path = tmp_path / "v.json"
    words: list[dict[str, str]] = [{"fr": "a", "ru": "б"}, {"fr": "a", "ru": ""}]
    p.write_text(json.dumps({"lessons": [{"n": 1, "ru": "x", "words": words}]}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        vocab.load(p)
    msg: str = str(err.value)
    assert "twice" in msg and "empty" in msg and "distractors" in msg


def test_meanings_drop_notes() -> None:
    assert vocab.meanings("один, неопределённый артикль (м.)") == {"один", "неопределённый артикль"}


def test_box_moves_one_step_a_day() -> None:
    assert vocab.boxes(rows(["a"] * 5, TODAY))["a"] == 1
    assert vocab.boxes(days(["a"], 3) + rows(["a"] * 4, TODAY))["a"] == 4


def test_box_miss_drops_to_zero_and_a_right_answer_lifts_to_one() -> None:
    h: list[dict[str, Any]] = days(["a"], 3) + rows(["a"], TODAY, False) + rows(["a"] * 2, TODAY)
    assert vocab.boxes(h)["a"] == 1


def test_box_is_capped() -> None:
    assert vocab.boxes(days(["a"], 9))["a"] == vocab.MAX_BOX


# Real vocabulary: skipped without it.


@NEEDS_VOCAB
def test_real_vocabulary_is_loaded(cards: list[Card]) -> None:
    assert len(cards) == 725 and len({c["lesson"] for c in cards}) == 40
    assert 4 not in {c["lesson"] for c in cards} and "deux" not in {c["fr"] for c in cards}


@NEEDS_VOCAB
def test_source_fixes_are_kept(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    assert b["la fille"]["ru"] == "дочь, девочка, девушка" and "le fille" not in b
    assert "mars" in b and "le mars" not in b and "s'asseoir" in b and b["lire"]["ru"] == "читать"
    assert b["le premier étage"]["ru"] == "второй этаж"


@NEEDS_VOCAB
def test_homonyms_of_numbers_keep_their_other_sense(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    assert b["neuf"]["ru"] == "новый (совсем новый)" and b["neuf"]["lesson"] == 12
    assert b["un"]["lesson"] == 40 and "один" not in b["un"]["ru"]
    assert b["la glace"]["ru"] == "лёд, мороженое"


@NEEDS_VOCAB
def test_no_distractor_shares_a_sense(cards: list[Card]) -> None:
    clash: list[tuple[str, str]] = [
        (c["fr"], d["fr"])
        for c in cards
        for d in vocab.distractors(c, cards)
        if vocab.meanings(c["ru"]) & vocab.meanings(d["ru"])
    ]
    assert clash == []


@NEEDS_VOCAB
def test_synonyms_are_kept_apart_and_distractors_share_the_lesson(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    assert "nous" not in [d["fr"] for d in vocab.distractors(b["on"], cards)]
    assert "le prénom" not in [d["fr"] for d in vocab.distractors(b["le nom"], cards)]
    assert all(d["lesson"] == 21 for d in vocab.distractors(b["la porte"], cards))


@NEEDS_VOCAB
def test_empty_history_gives_ten_new_words_from_ten_topics(cards: list[Card]) -> None:
    first: list[Card] = vocab.pick(cards, [], TODAY, random.Random(1))
    assert len(first) == 10 and all(w["new"] for w in first)
    assert len({w["lesson"] for w in first}) == 10
    other: set[str] = {w["fr"] for w in vocab.pick(cards, [], TODAY, random.Random(2))}
    assert other != {w["fr"] for w in first}


@NEEDS_VOCAB
def test_all_topics_come_up_numbers_never(cards: list[Card]) -> None:
    spread: Counter[int] = Counter(
        w["lesson"] for s in range(100) for w in vocab.pick(cards, [], TODAY, random.Random(s))
    )
    assert set(spread) == {c["lesson"] for c in cards}


@NEEDS_VOCAB
def test_fewer_topics_than_slots_repeat_rounds(cards: list[Card]) -> None:
    few: list[str] = [c["fr"] for c in cards if c["lesson"] not in (1, 2)]
    two_left: list[Card] = vocab.pick(cards, days(few, 1), TODAY, random.Random(3))
    assert sum(w["new"] for w in two_left) == 5
    assert {w["lesson"] for w in two_left if w["new"]} == {1, 2}


@NEEDS_VOCAB
def test_daily_quota(cards: list[Card]) -> None:
    first: list[Card] = vocab.pick(cards, [], TODAY, random.Random(1))
    hist: list[dict[str, Any]] = rows([w["fr"] for w in first], TODAY)
    again: list[Card] = vocab.pick(cards, hist, TODAY, random.Random(3))
    assert len(again) == 10 and not any(w["new"] for w in again)
    tomorrow: list[Card] = vocab.pick(cards, hist, TODAY + timedelta(days=1), random.Random(4))
    assert sum(w["new"] for w in tomorrow) == 5 and len(tomorrow) == 10


@NEEDS_VOCAB
def test_weak_word_is_reviewed_most_and_strong_words_leave_room(cards: list[Card]) -> None:
    first: list[Card] = vocab.pick(cards, [], TODAY, random.Random(1))
    words10: list[str] = [w["fr"] for w in first]
    weak_hist: list[dict[str, Any]] = days(words10, 5) + rows(words10[:1], TODAY, False)
    drawn: Counter[str] = Counter(
        w["fr"]
        for s in range(200)
        for w in vocab.pick(cards, weak_hist, TODAY, random.Random(s))
        if not w["new"]
    )
    assert drawn[words10[0]] == max(drawn.values()) and drawn[words10[0]] == 200
    strong: list[dict[str, Any]] = rows(words10 * 5, TODAY - timedelta(days=3))
    new_only: int = sum(
        1 for s in range(50) for w in vocab.pick(cards, strong, TODAY, random.Random(s)) if w["new"]
    )
    assert new_only == 250


@NEEDS_VOCAB
def test_questions_pairs_options_and_prompts(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    first: list[Card] = vocab.pick(cards, [], TODAY, random.Random(1))
    qs: list[dict[str, Any]] = vocab.questions(first, cards, random.Random(6))
    pairs: list[dict[str, Any]] = [q for q in qs if q["dir"] != "article"]
    assert len(pairs) == 20 and [q["dir"] for q in pairs] == ["fr-ru"] * 10 + ["ru-fr"] * 10
    assert Counter(q["word"] for q in pairs) == Counter({w["fr"]: 2 for w in first})
    assert all(len(set(q["options"])) == 4 for q in pairs)
    assert all(
        q["options"][q["answer"]] == (b[q["word"]]["ru"] if q["dir"] == "fr-ru" else b[q["word"]]["label"])
        for q in pairs
    )
    assert all(
        q["prompt"] == (b[q["word"]]["label"] if q["dir"] == "fr-ru" else b[q["word"]]["ru"]) for q in pairs
    )


@NEEDS_VOCAB
def test_gender_shows_on_elided_and_plural_cards(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    assert b["l'étoile"]["label"] == "une étoile" and b["l'arbre"]["label"] == "un arbre"
    assert b["les cheveux"]["label"] == "les cheveux (м.)" and b["l'enfant"]["label"] == "un, une enfant"
    assert b["l'Espagne"]["label"] == "l'Espagne (ж.)" and b["le beurre"]["label"] == "le beurre"
    assert b["parler"]["label"] == "parler"


@NEEDS_VOCAB
def test_article_is_asked_for_countable_nouns_only(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    assert vocab.articles(b["l'étoile"]) == (("un", "une"), "une", "étoile")
    assert vocab.articles(b["la gare"]) == (("le", "la"), "la", "gare")
    assert not any(
        vocab.articles(b[w]) for w in ("l'enfant", "l'Espagne", "les cheveux", "parler", "un", "le")
    )


@NEEDS_VOCAB
def test_article_question_after_the_pairs(cards: list[Card]) -> None:
    b: dict[str, Card] = by(cards)
    words: list[Card] = [dict(b[w], new=False) for w in ("la gare", "l'étoile", "parler")]
    art: list[dict[str, Any]] = [
        q for q in vocab.questions(words, cards, random.Random(1)) if q["dir"] == "article"
    ]
    assert {q["word"] for q in art} == {"la gare", "l'étoile"}
    q: dict[str, Any]
    for q in art:
        assert len(q["options"]) == 2 and q["prompt"] not in q["full"].split()[:1]
        assert q["note"] == b[q["word"]]["ru"] and q["options"][q["answer"]] == q["full"].split()[0]


@NEEDS_VOCAB
def test_http_session_check_and_stats(live: Live, cards: list[Card], tmp_path: Path) -> None:
    port: int = live.port
    assert call(port, "GET", "/vocab/session", host="evil.com")[0] == 403
    s: int
    sess: dict[str, Any]
    s, sess = call(port, "GET", "/vocab/session")
    assert s == 200 and len([i for i in sess["items"] if i["dir"] != "article"]) == 20 and sess["new"] == 10
    assert set(sess["items"][0]) == {"id", "topic", "new", "dir", "prompt", "note", "options"}
    it: dict[str, Any] = sess["items"][0]
    answer: str = next(c for c in cards if c["label"] == it["prompt"])["ru"]
    r: dict[str, Any]
    s, r = call(
        port,
        "POST",
        "/vocab/check",
        {"session": sess["session"], "id": it["id"], "choice": it["options"].index(answer), "ms": 1500},
    )
    assert s == 200 and r["ok"] is True and it["options"][r["answer"]] == answer
    wrong: int = next(i for i, o in enumerate(it["options"]) if o != answer)
    s, r = call(port, "POST", "/vocab/check", {"session": sess["session"], "id": it["id"], "choice": wrong})
    assert s == 200 and r["ok"] is False
    assert call(port, "POST", "/vocab/check", {"session": sess["session"], "id": "0", "choice": 7})[0] == 404
    assert call(port, "POST", "/vocab/check", {"session": "nope", "id": "0", "choice": 0})[0] == 404
    log: list[dict[str, Any]] = [
        json.loads(x) for x in (tmp_path / "vocab.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(log) == 2 and log[0]["ok"] and not log[1]["ok"] and log[0]["word"] == it["prompt"]
    assert log[0]["ms"] == 1500 and log[0]["dir"] == "fr-ru"
    st: dict[str, Any]
    s, st = call(port, "GET", "/vocab/stats")
    assert st["seen"] == 1 and st["boxes"][0] == 1 and st["new_today"] == 1 and st["cards"] == len(cards)
