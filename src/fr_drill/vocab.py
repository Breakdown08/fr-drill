"""A1 words, the second tab of `fr-drill drill`: pick the translation out of four, then the article.

Cards come from the vocabulary file (`VOCAB`) (a personal vocabulary file, see README). A session is
SIZE words, each asked both ways — French → Russian first, then Russian → French — and then every
noun once more: which article, le or la, un or une. A noun shows its gender everywhere: l'étoile and
les cheveux hide it, so they read `une étoile`, `les cheveux (м.)`; the card's `fr` stays the key of
the history. Nouns of both genders, plurals and `definite` words (l'Espagne) get no article question.
New words come one per topic from topics picked at random, at most NEW_PER_DAY a day across all sessions;
the rest of a session is review. Lessons with `own_drill` (numbers) belong to another drill and are left out.

Every word has a box 0..MAX_BOX replayed from `progress/vocab.jsonl`: a right answer moves
it up one, at most once a day, so a box counts days the word held, not repeats in a row; a wrong
one drops it to 0 at once. Review draws words by weight 2^(MAX_BOX − box), so a word that went
wrong comes back 32 times as often as one right on five different days. The article is part of the
word: a wrong article drops the box like a wrong translation.

GET  /vocab/session   new session: questions with options, no answers
POST /vocab/check     {session, id, choice, ms} → {ok, answer}
GET  /vocab/stats     cards seen, boxes, new today
"""

import json
import random
import re
import secrets
import threading
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fr_drill import PROGRESS, VOCAB, append_jsonl, read_jsonl

WORDS: Path = VOCAB
ATTEMPTS: Path = PROGRESS / "vocab.jsonl"
SIZE: int = 10
NEW_PER_DAY: int = 10
MAX_BOX: int = 5
KNOWN: int = 4  # box from which a word counts as known
REVIEW_FIRST: int = SIZE // 2  # review slots kept even while new words are left today
OPTIONS: int = 4
ARTICLE_PAIRS: tuple[tuple[str, str], tuple[str, str]] = (("le", "la"), ("un", "une"))
MARK: dict[str, str] = {"m": "м.", "f": "ж."}


def label(fr: str, gender: str | None, definite: bool = False) -> str:
    """The card as shown, its gender visible: l'étoile → une étoile, les cheveux → les cheveux (м.)."""
    if fr.startswith("l'") and gender == "mf":
        return f"un, une {fr[2:]}"
    if fr.startswith("l'") and not definite:
        return f"{'un' if gender == 'm' else 'une'} {fr[2:]}"
    return f"{fr} ({MARK[gender]})" if gender else fr


def articles(card: dict[str, Any]) -> tuple[tuple[str, str], str, str] | None:
    """(the two articles, the right one, the noun alone) if the card can be asked for its article."""
    noun: Any
    _: Any
    first: Any
    first, _, noun = card["label"].partition(" ")
    pair: tuple[str, str] | None = next((p for p in ARTICLE_PAIRS if first in p), None)
    return (pair, first, noun) if pair and noun else None


def load(path: Path = WORDS) -> list[dict[str, Any]]:
    """Flat card list in lesson order; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    seen: set[Any]
    errors: list[Any]
    cards: list[Any]
    cards, errors, seen = [], [], set()
    les: Any
    for les in data.get("lessons", []):
        if les.get("own_drill"):
            continue
        w: Any
        for w in les.get("words", []):
            ru: Any
            fr: Any
            fr, ru = w.get("fr", ""), w.get("ru", "")
            if not fr or not ru:
                errors.append(f"lesson {les.get('n')}: empty card {w}")
            if fr in seen:
                errors.append(f"lesson {les.get('n')}: {fr!r} twice")
            seen.add(fr)
            gender: Any = w.get("g")
            if gender not in (None, "m", "f", "mf") or (gender is None and fr.startswith(("l'", "les "))):
                errors.append(f"lesson {les.get('n')}: {fr!r} needs a gender, g: m, f or mf")
                gender = None
            cards.append(
                {
                    "fr": fr,
                    "ru": ru,
                    "lesson": les["n"],
                    "topic": les["ru"],
                    "label": label(fr, gender, w.get("definite", False)),
                }
            )
    if not cards:
        errors.append("no cards")
    c: dict[str, Any]
    for c in cards:
        if len(distractors(c, cards)) < OPTIONS - 1:
            errors.append(f"{c['fr']!r}: fewer than {OPTIONS - 1} distractors")
    if errors:
        raise ValueError("\n".join(errors))
    return cards


def meanings(ru: str) -> set[str]:
    """Senses without notes: "один, неопределённый артикль (м.)" → {"один", "неопределённый артикль"}."""
    bare: str = re.sub(r"\s*\([^)]*\)", "", ru)
    return {m.strip() for m in bare.split(",") if m.strip()}


def distractors(card: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cards of the same lesson that share no sense with this one; other lessons if too few."""
    clear: list[dict[str, Any]] = [
        c for c in cards if c["fr"] != card["fr"] and not meanings(c["ru"]) & meanings(card["ru"])
    ]
    same: list[dict[str, Any]] = [c for c in clear if c["lesson"] == card["lesson"]]
    return same if len(same) >= OPTIONS - 1 else clear


def boxes(history: list[dict[str, Any]]) -> dict[str, int]:
    """Box of every word answered at least once: up one step a day at most, down to 0 on a miss."""
    box: dict[str, int] = {}
    raised: dict[str, str] = {}  # word → day it last went up
    r: dict[str, Any]
    for r in history:
        day: Any
        w: Any
        w, day = r["word"], r["at"][:10]
        box.setdefault(w, 0)
        if not r["ok"]:
            box[w] = 0
            raised.pop(w, None)
        elif raised.get(w) != day:
            box[w] = min(MAX_BOX, box[w] + 1)
            raised[w] = day
    return box


def new_today(history: list[dict[str, Any]], today: date) -> set[str]:
    """Words whose first answer was today."""
    first: dict[str, str] = {}
    r: dict[str, Any]
    for r in history:
        first.setdefault(r["word"], r["at"][:10])
    return {w for w, d in first.items() if d == today.isoformat()}


def weighted[T](pool: list[T], weights: list[float], k: int, rng: random.Random) -> list[T]:
    out: list[T]
    pool, weights, out = list(pool), list(weights), []
    while pool and len(out) < k:
        i: int = rng.choices(range(len(pool)), weights)[0]
        out.append(pool.pop(i))
        weights.pop(i)
    return out


def pick(
    cards: list[dict[str, Any]], history: list[dict[str, Any]], today: date, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """SIZE words: some review, new words from different topics within today's quota, review fills the
    rest."""
    rng = rng or random.Random()
    box: dict[str, int] = boxes(history)
    seen: list[dict[str, Any]] = [c for c in cards if c["fr"] in box]
    weight: list[float] = [2.0 ** (MAX_BOX - box[c["fr"]]) for c in seen]
    quota: int = max(0, NEW_PER_DAY - len(new_today(history, today)))
    review: list[dict[str, Any]] = weighted(seen, weight, min(REVIEW_FIRST, len(seen)), rng)
    fresh: list[Any]
    need: int
    need, fresh = min(quota, SIZE - len(review)), []
    unseen: list[dict[str, Any]] = [c for c in cards if c["fr"] not in box]
    while unseen and len(fresh) < need:  # rounds: one word from each of the topics picked at random
        topics: list[Any] = sorted({c["lesson"] for c in unseen})
        les: Any
        for les in rng.sample(topics, min(len(topics), need - len(fresh))):
            fresh.append(rng.choice([c for c in unseen if c["lesson"] == les]))
        unseen = [c for c in unseen if c not in fresh]
    rest: list[tuple[dict[str, Any], float]] = [
        (c, w) for c, w in zip(seen, weight, strict=False) if c not in review
    ]
    review += weighted([c for c, _ in rest], [w for _, w in rest], SIZE - len(review) - len(fresh), rng)
    return [dict(c, new=c["fr"] not in box) for c in fresh + review]


def questions(
    words: list[dict[str, Any]], cards: list[dict[str, Any]], rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """Each word twice: French → Russian, then Russian → French; then the article of every noun; each part
    shuffled."""
    rng = rng or random.Random()
    out: list[Any] = []
    direction: str
    for direction in ("fr-ru", "ru-fr"):
        half: list[dict[str, Any]] = list(words)
        rng.shuffle(half)
        w: dict[str, Any]
        for w in half:
            options: list[dict[str, Any]] = rng.sample(distractors(w, cards), OPTIONS - 1) + [w]
            rng.shuffle(options)
            side: str = "ru" if direction == "fr-ru" else "label"
            out.append(
                {
                    "id": str(len(out)),
                    "word": w["fr"],
                    "lesson": w["lesson"],
                    "topic": w["topic"],
                    "new": w["new"],
                    "dir": direction,
                    "prompt": w["label"] if direction == "fr-ru" else w["ru"],
                    "note": "",
                    "full": w["label"],
                    "options": [o[side] for o in options],
                    "answer": options.index(w),
                }
            )
    found: tuple[tuple[str, str], str, str] | None
    nouns: list[tuple[dict[str, Any], tuple[tuple[str, str], str, str]]] = [
        (w, found) for w in words if (found := articles(w))
    ]
    rng.shuffle(nouns)
    noun: str
    right: str
    pair: tuple[str, str]
    for w, (pair, right, noun) in nouns:
        out.append(
            {
                "id": str(len(out)),
                "word": w["fr"],
                "lesson": w["lesson"],
                "topic": w["topic"],
                "new": w["new"],
                "dir": "article",
                "prompt": noun,
                "note": w["ru"],
                "full": w["label"],
                "options": list(pair),
                "answer": pair.index(right),
            }
        )
    return out


def stats(cards: list[dict[str, Any]], history: list[dict[str, Any]], today: date) -> dict[str, Any]:
    box: dict[str, int] = boxes(history)
    by: Counter[int] = Counter(box.values())
    return {
        "cards": len(cards),
        "seen": len(box),
        "known": sum(n for b, n in by.items() if b >= KNOWN),
        "boxes": [by.get(b, 0) for b in range(MAX_BOX + 1)],
        "new_today": len(new_today(history, today)),
        "new_per_day": NEW_PER_DAY,
        "answers": len(history),
        "ok": sum(r["ok"] for r in history),
    }


class Vocab:
    def __init__(self, cards: list[dict[str, Any]], attempts: Path = ATTEMPTS) -> None:
        self.cards: list[dict[str, Any]] = cards
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        words: list[dict[str, Any]] = pick(self.cards, read_jsonl(self.attempts), date.today())
        items: list[dict[str, Any]] = questions(words, self.cards)
        with self.lock:
            self.sessions[sid] = {i["id"]: i for i in items}
        public: tuple[str, str, str, str, str, str, str] = (
            "id",
            "topic",
            "new",
            "dir",
            "prompt",
            "note",
            "options",
        )
        return {
            "session": sid,
            "words": len(words),
            "new": sum(w["new"] for w in words),
            "items": [{k: i[k] for k in public} for i in items],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        choice: Any | None = body.get("choice")
        if not isinstance(choice, int) or not 0 <= choice < len(item["options"]):
            raise KeyError("bad choice")
        ok: Any = choice == item["answer"]
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session": body["session"],
            "word": item["word"],
            "lesson": item["lesson"],
            "dir": item["dir"],
            "expected": item["options"][item["answer"]],
            "chosen": item["options"][choice],
            "ok": ok,
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
        }
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, [row])
        return {"ok": ok, "answer": item["answer"], "full": item["full"]}

    def stats(self) -> dict[str, Any]:
        return stats(self.cards, read_jsonl(self.attempts), date.today())


def show_stats(cards: list[dict[str, Any]], path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = stats(cards, read_jsonl(path), date.today())
    if not s["answers"]:
        print("words: no answers yet")
        return
    print(
        f"words: {s['seen']}/{s['cards']} met, {s['known']} known (box ≥{KNOWN}), "
        f"{100 * s['ok'] // s['answers']}% right, new today {s['new_today']}/{s['new_per_day']}"
    )
    print("  boxes 0..5: " + " ".join(str(n) for n in s["boxes"]))
