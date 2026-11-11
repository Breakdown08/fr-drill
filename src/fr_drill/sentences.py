"""Sentences, the sixth tab of `fr-drill drill`: build a French sentence from tiles.

A task gives the Russian and the tiles of the sentence shuffled with wrong forms from
`data/drills/sentences.json` and with decoys: forms of other learned verbs in the same person,
learned nouns. «Я люблю кофе» → j'aime · café among je, aime, cherche, ferme, gare, beurre.
Articles and prepositions are no tiles: all of LINKS stand in a panel, the same for every sentence,
so the set of tiles does not hint which one is needed. l' and d' stand there too; other elided
words stay one tile, so `je aime` can be built and is a mistake.

A sentence opens once its words are learned: every verb has a right answer on the conjugation,
negation or irregular tab, every card a box ≥ 1 on the words tab. `load` proves that nothing else
slips in: any other word must be in `core` or in the phrases of those two tabs. A session draws
by weight: what went wrong first, then what is unseen, then what held the least.

Mistakes by kind: `form` (a wrong form used), `word` (a decoy used), `link` (the wrong article or
preposition), `order` (right tiles, wrong order), `missing` (a tile left out).

After a mistake `rules` names at most MAX_RULES rules for what was actually built: the sentence's
own rule for a set expression, a missing verb, a confusion of être and avoir, a person, an elision,
an agreement, a missing article or preposition.

The second mode trains the links alone: the sentence stands built, every article and preposition is
a gap, the panel fills it. GAP_SIZE sentences a session, with its own history in `gaps.jsonl`.
`also` in the bank lists sentences with another link that are just as right: «ищу ключи» is
les clés and des clés. Both modes take them.

GET  /sentences/session        new session: Russian and shuffled tiles, no answers; the panel
POST /sentences/check          {session, id, tiles, ms} → {ok, kind, expected, rules}
GET  /sentences/stats          open sentences and accuracy by mistake kind
GET  /sentences/gaps/session   new session: Russian and tiles with null for each gap; the panel
POST /sentences/gaps/check     {session, id, fills, ms} → {ok, expected, rules}
GET  /sentences/gaps/stats     answers and the links taken for one another
"""

import json
import random
import re
import secrets
import threading
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import (
    DATA,
    PROGRESS,
    append_jsonl,
    drill,
    etre_avoir,
    negation,
    read_jsonl,
    vocab,
)

BANK: Path = DATA / "drills/sentences.json"
ATTEMPTS: Path = PROGRESS / "sentences.jsonl"
DRILL: str = "sentences"
GAPS_DRILL: str = "gaps"
SIZE: int = 10
GAP_SIZE: int = 20
MAX_TILES: int = 30
MAX_RULES: int = 2
DECOY_VERBS: int = 2
DECOY_NOUNS: int = 2
ARTICLES: list[str] = ["le", "la", "l'", "les", "un", "une", "des", "du"]
PREPOSITIONS: list[str] = [
    "à",
    "au",
    "aux",
    "de",
    "d'",
    "dans",
    "en",
    "avec",
    "chez",
    "sur",
    "sous",
    "pour",
    "sans",
    "par",
]
LINKS: set[str] = set(ARTICLES + PREPOSITIONS)
DEFINITE: set[str] = {"le", "la", "l'", "les"}
PLURAL: set[str] = {"les", "des"}
NOUN_AFTER_ARTICLE: re.Pattern[str] = re.compile(r"(?:^|\s)(?:les?|la|l'|une?)\s*([^\s']+)$")
NAMES: set[str] = {
    "marie",
    "sophie",
    "paris",
    "lyon",
}  # keep the capital on a tile, every other word loses it


def tiles(fr: str) -> list[str]:
    """The sentence as tiles: lowercase but for names, no punctuation, l' and d' apart from their word."""
    apart: str = re.sub(r"\b([ld]')", r"\1 ", re.sub(r"[,.?!]", " ", drill.norm(fr)))
    return [p.capitalize() if p in NAMES else p for p in apart.split()]


def holes(built: list[str]) -> list[str | None]:
    """The tiles with None for each article and preposition."""
    return [None if t in LINKS else t for t in built]


def plain(fr: str) -> list[str]:
    """Words for comparing and for the coverage check: lowercase, elided ones apart."""
    return re.sub(r"[,.?!]", " ", drill.norm(fr)).replace("'", " ").split()


def verb_forms(verbs: list[dict[str, Any]]) -> dict[str, str]:
    """Every present form → its infinitive."""
    forms: dict[str, str] = {f: inf for inf, table in etre_avoir.FORMS.items() for f in table.values()}
    v: dict[str, Any]
    for v in verbs:
        forms.update({drill.form(v["inf"], person): v["inf"] for person in drill.PERSONS})
    return forms


def met_words() -> set[str]:
    """The words of the phrases on the negation and irregular tabs: seen there, not on a card."""
    words: set[str] = set()
    contexts: list[dict[str, Any]]
    for contexts in (negation.load(), etre_avoir.load()):
        c: dict[str, Any]
        for c in contexts:
            words.update(plain(c["fr"]))
    return words


def problems(
    s: dict[str, Any], core: set[str], met: set[str], cards: dict[str, str], forms: dict[str, str]
) -> list[str]:
    """What is wrong with one sentence."""
    errors: list[Any]
    fr: Any
    fr, errors = s.get("fr", ""), []
    own: set[Any] = set(s.get("forms", []))
    carded: set[Any] = set(own)
    card: Any
    for card in s.get("words", []):
        if card not in cards:
            errors.append(f"{fr!r}: {card!r} is not a card")
            continue
        if not (set(plain(card)) | own) - core & set(plain(fr)):
            errors.append(f"{fr!r}: {card!r} is not in the sentence")
        carded.update(plain(card))
    word: str
    for word in plain(fr):
        if word in core or word in carded or word in met:  # met first: la cuisine is no form of cuisiner
            continue
        if word not in forms:
            errors.append(f"{fr!r}: {word!r} is not learned anywhere")
        elif forms[word] not in s.get("verbs", []):
            errors.append(f"{fr!r}: {word!r} needs {forms[word]!r} in verbs")
    inf: Any
    for inf in s.get("verbs", []):
        if not {f for f, v in forms.items() if v == inf} & set(plain(fr)):
            errors.append(f"{fr!r}: no form of {inf!r}")
    built: list[str] = tiles(fr)
    bad: Any
    for bad in s.get("wrong", []):
        if bad in built:
            errors.append(f"{fr!r}: wrong tile {bad!r} is in the sentence")
        if bad in LINKS:
            errors.append(f"{fr!r}: wrong tile {bad!r} is in the panel")
    if not s.get("wrong"):
        errors.append(f"{fr!r}: no wrong tiles")
    other: Any
    for other in s.get("also", []):
        if holes(tiles(other)) != holes(built):
            errors.append(f"{fr!r}: also {other!r} may differ only in its links")
    for other in s.get("accept", []):
        if Counter(tiles(other)) != Counter(built):
            errors.append(f"{fr!r}: accept {other!r} needs other tiles")
    return errors


def load(
    path: Path = BANK, verbs: list[dict[str, Any]] | None = None, cards: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The sentences; raise ValueError listing every problem."""
    verbs = verbs if verbs is not None else drill.load()
    known: dict[str, Any] = {c["fr"]: c["ru"] for c in (cards if cards is not None else vocab.load())}
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    forms: dict[str, str]
    met: set[str]
    core: set[Any]
    core, met, forms = set(data.get("core", [])), met_words(), verb_forms(verbs)
    seen: set[Any]
    errors: list[Any]
    errors, seen = [], set()
    s: Any
    for s in data.get("sentences", []):
        ru: Any
        fr: Any
        fr, ru = s.get("fr", ""), s.get("ru", "")
        if not fr.strip() or not ru.strip():
            errors.append(f"{s}: empty")
        if fr in seen:
            errors.append(f"{fr!r} twice")
        seen.add(fr)
        errors += problems(s, core, met, known, forms)
    if not seen:
        errors.append("no sentences")
    if errors:
        raise ValueError("\n".join(errors))
    sentences: list[dict[str, Any]] = data["sentences"]
    return sentences


def learned(verb_rows: list[dict[str, Any]], word_rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """(verbs with a right answer, cards with a box ≥ 1)."""
    return {r["verb"] for r in verb_rows if r["ok"]}, {
        w for w, box in vocab.boxes(word_rows).items() if box >= 1
    }


def opened(sentences: list[dict[str, Any]], verbs: set[str], words: set[str]) -> list[dict[str, Any]]:
    return [s for s in sentences if set(s.get("verbs", [])) <= verbs and set(s.get("words", [])) <= words]


def met_nouns() -> set[str]:
    """The nouns of the phrases on the negation and irregular tabs."""
    found: Iterator[re.Match[str] | None] = (
        NOUN_AFTER_ARTICLE.search(drill.norm(c["fr"])) for c in [*negation.load(), *etre_avoir.load()]
    )
    return {m.group(1) for m in found if m}


def card_nouns(words: set[str]) -> set[str]:
    """The bare nouns of learned cards: le beurre → beurre, l'étoile → étoile."""
    found: Iterator[re.Match[str] | None] = (NOUN_AFTER_ARTICLE.search(drill.norm(w)) for w in words)
    return {m.group(1) for m in found if m}


def person(s: dict[str, Any]) -> str:
    """The person of the sentence's verb, il when it has none."""
    said: set[str] = set(plain(s["fr"]))
    inf: Any
    for inf in s.get("verbs", []):
        table: dict[str, str] = etre_avoir.FORMS.get(inf) or {p: drill.form(inf, p) for p in drill.PERSONS}
        form: str
        p: str
        for p, form in table.items():
            if form in said:
                return p
    return "il"


def decoys(s: dict[str, Any], verbs: set[str], nouns: set[str], rng: random.Random) -> list[str]:
    """Forms of other learned verbs in the sentence's person and learned nouns, none of them in the task."""
    taken: set[str | Any] = set(tiles(s["fr"])) | set(plain(s["fr"])) | set(s["wrong"])
    other: Any
    for other in s.get("accept", []):
        taken.update(tiles(other))
    regular: list[str] = sorted(verbs - set(s.get("verbs", [])) - set(etre_avoir.FORMS))
    forms: list[str] = [f for f in (drill.form(v, person(s)) for v in regular) if f not in taken]
    free: list[str] = sorted(nouns - taken)
    return rng.sample(forms, min(DECOY_VERBS, len(forms))) + rng.sample(free, min(DECOY_NOUNS, len(free)))


def weight(fr: str, history: list[dict[str, Any]]) -> float:
    """Wrong last time 8, unseen 4, then halved by each right answer in a row, down to 1/2."""
    rows: list[Any] = [r["ok"] for r in history if r["fr"] == fr]
    if not rows:
        return 4.0
    if not rows[-1]:
        return 8.0
    streak: int = 0
    ok: Any
    for ok in reversed(rows):
        if not ok:
            break
        streak += 1
    weight: float = 4.0 / 2 ** min(streak, 3)
    return weight


def pick(
    pool: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = SIZE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    rng = rng or random.Random()
    return vocab.weighted(pool, [weight(s["fr"], history) for s in pool], min(size, len(pool)), rng)


def grade(built: list[str], s: dict[str, Any]) -> tuple[bool, str | None]:
    """(ok, kind of mistake): form, word, link, order, missing."""
    right: list[list[str]] = [tiles(x) for x in [s["fr"], *s.get("accept", []), *s.get("also", [])]]
    if built in right:
        return True, None
    expected: list[str] = right[0]
    if Counter(built) == Counter(expected):
        return False, "order"
    if set(built) & set(s["wrong"]):
        return False, "form"
    if any(t not in LINKS and t not in expected for t in built):
        return False, "word"
    same: bool = Counter(t for t in built if t not in LINKS) == Counter(t for t in expected if t not in LINKS)
    return (False, "link") if same else (False, "missing")


def link_rules(built: list[str], expected: list[str]) -> list[str]:
    """What the articles and prepositions of the answer got wrong."""
    gone: Counter[str] = Counter(t for t in expected if t in LINKS) - Counter(t for t in built if t in LINKS)
    extra: Counter[str] = Counter(t for t in built if t in LINKS) - Counter(t for t in expected if t in LINKS)
    if not gone and not extra:
        return []
    rules: list[Any] = []
    other: str | None = next(iter(extra), None)
    t: str
    for t in gone:
        if t == "au":
            rules.append(
                "au = à + le: je suis au bureau, je joue au football. Перед la слияния нет: à la maison."
            )
        elif t == "aux":
            rules.append("aux = à + les: je joue aux cartes.")
        elif t in ("l'", "d'"):
            rules.append("Перед гласной и немой h le, la → l', de → d': l'adresse, l'heure, d'accord.")
        elif other in ("l'", "d'"):
            rules.append(
                "l' и d' только перед гласной или немой h. "
                "Перед согласной — полная форма: la maison, le café."
            )
        elif t in ARTICLES and other in ARTICLES and "du" not in (t, other):
            rules.append(
                "les, des — множественное число, le, la, un, une — единственное."
                if (t in PLURAL) != (other in PLURAL)
                else "un, une — впервые названное или одно из многих: j'ai un chien. le, la — известное "
                "или вообще: j'aime le café."
                if (t in DEFINITE) != (other in DEFINITE)
                else "Род другой: артикль запоминай вместе со словом — "
                "le café, la gare, un message, une table."
            )
        elif other:
            rules.append(f"Здесь нужно {t}, а не {other}.")
        elif t in ARTICLES:
            rules.append(
                "Перед существительным почти всегда стоит артикль или предлог. Без них — имена и профессии."
            )
        else:
            rules.append(
                f"Здесь нужен предлог {t}: в русском его может не быть, во французском он не выпадает."
            )
    if not gone:
        rules.append(f"{', '.join(extra)} здесь лишнее: связка нужна не перед каждым словом.")
    return rules


def rules_for(built: list[str], s: dict[str, Any], kind: str | None, forms: dict[str, str]) -> list[str]:
    """The rules for this mistake, most useful first, at most MAX_RULES."""
    expected: list[str] = tiles(s["fr"])
    said: set[str]
    words: list[str]
    words, said = plain(" ".join(built)), set(plain(s["fr"]))
    found: list[Any] = []
    if said & set(forms) and not set(words) & set(forms):
        found.append(
            "Французскому предложению всегда нужен глагол: где в русском пусто или тире, там est — "
            "le plat est chaud, il est au bureau."
            if "être" in s.get("verbs", [])
            else "Французскому предложению всегда нужен глагол, даже если в русском его не видно."
        )
    if "pas" in expected and "pas" not in built:
        found.append(
            "Это отрицание: ne + глагол + pas, глагол стоит между ними — il ne parle pas, je n'aime pas."
        )
    own: Any | None = (
        s.get("rule") if kind in ("link", "word", "missing") else None
    )  # a wrong form is no matter of the expression
    if own:
        found.append(own)
    if kind == "order":
        found.append(
            "Отрицание обнимает глагол: ne + глагол + pas."
            if "pas" in expected
            else "Порядок слов: кто + глагол + остальное. Прилагательное и место идут после глагола."
        )
    used: set[str] = {forms[w] for w in words if w in forms and w not in said}
    if "y" in expected and {"est", "c'est"} & set(built):
        found.append("«Есть, находится» про то, чего ещё не называли, — il y a: il y a une table.")
    elif "avoir" in s.get("verbs", []) and "être" in used:
        found.append(
            "«Есть у кого-то», голод, жажда, холод, страх, возраст — через avoir: j'ai un chien, j'ai faim."
        )
    elif "être" in s.get("verbs", []) and "avoir" in used:
        found.append("Место, профессия, качество — через être: il est au bureau, elle est triste.")
    elif used & set(s.get("verbs", [])):
        found.append(
            "Глагол тот, лицо другое. На -er: je -e, tu -es, il -e, nous -ons, vous -ez, ils -ent."
            if used - set(etre_avoir.FORMS)
            else "Глагол тот, лицо другое: je suis, tu es, il est, nous sommes, vous êtes, ils sont; "
            "j'ai, tu as, il a, nous avons, vous avez, ils ont."
        )
    if ("je" in built and any(t.startswith("j'") for t in expected)) or (
        "ne" in built and any(t.startswith("n'") for t in expected)
    ):
        found.append("Перед гласной и немой h je → j', ne → n': j'aime, je n'aime pas.")
    if "n'" in built:
        found.append("n' только перед гласной. Перед согласной — ne: il ne chante pas.")
    if "il" in built and "il" not in expected and "c'est" in expected:
        found.append("Перед артиклем — c'est, а не il est: c'est un message.")
    t: str
    for t in (t for t in built if t not in LINKS):  # un and une are no agreement of an adjective
        if t + "e" in expected:
            found.append("Прилагательное согласуется: elle → окончание -e, satisfaite, portugaise.")
        elif t.endswith("e") and t[:-1] in expected:
            found.append("Прилагательное согласуется: il → без -e, satisfait, suédois.")
    if {"bon", "bonne"} & set(built) and set(s["wrong"]) & set(built):
        found.append("bon с мужским родом, bonne с женским: bon après-midi, bonne nuit, bonne soirée.")
    if "commencons" in built:
        found.append("Перед -ons c становится ç, чтобы звук остался мягким: commençons.")
    if "va-tu" in built:
        found.append("Форма tu от aller — vas, s остаётся: comment vas-tu ?")
    if not own or Counter(t for t in built if t in LINKS) - Counter(t for t in expected if t in LINKS):
        found += link_rules(
            built, expected
        )  # a wrong link is named even in a set expression, a missing one is not
    stray: list[str] = [t for t in built if t not in LINKS and t not in expected and t not in s["wrong"]]
    if stray:
        found.append(f"{', '.join(stray)} в этой фразе не нужно: смотри, какое слово названо по-русски.")
    if not found:
        found.append("Собери все слова фразы: каждое слово из русского должно появиться во французском.")
    return list(dict.fromkeys(found))[:MAX_RULES]


def gap_rules(built: list[str], s: dict[str, Any], forms: dict[str, str]) -> list[str]:
    """The rules for wrong fills; the same links in other gaps have no rule of their own."""
    if Counter(built) == Counter(tiles(s["fr"])):
        return [s["rule"]] if s.get("rule") else ["Связки верные, но стоят не на своих местах."]
    return rules_for(built, s, "link", forms)


def gap_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: Counter[tuple[Any, Any]] = Counter(
        (want, got)
        for r in history
        for want, got in zip(r["links"], r["fills"], strict=False)
        if want != got and not r["ok"]
    )
    return {
        "answers": len(history),
        "ok": sum(r["ok"] for r in history),
        "sessions": len({r["session"] for r in history}),
        "pairs": [{"want": want, "got": got, "n": n} for (want, got), n in pairs.most_common(5)],
    }


def stats(history: list[dict[str, Any]], open_count: int, total: int) -> dict[str, Any]:
    return {
        "answers": len(history),
        "ok": sum(r["ok"] for r in history),
        "sessions": len({r["session"] for r in history}),
        "open": open_count,
        "total": total,
        "kinds": dict(Counter(r["kind"] for r in history if not r["ok"]).most_common()),
    }


class Sentences:
    def __init__(
        self,
        bank: list[dict[str, Any]],
        attempts: Path = ATTEMPTS,
        sources: tuple[Path, ...] | None = None,
        words: Path = vocab.ATTEMPTS,
    ) -> None:
        self.bank: list[dict[str, Any]] = bank
        self.attempts: Path = attempts
        self.sources: tuple[Path, ...] = (
            sources if sources is not None else (drill.ATTEMPTS, negation.ATTEMPTS, etre_avoir.ATTEMPTS)
        )
        self.words: Path = words
        self.forms: dict[str, str] = {
            f: inf for f, inf in verb_forms(drill.load()).items() if f not in met_words()
        }
        self.gaps: Path = attempts.with_name("gaps.jsonl")
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.gap_sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def known(self) -> tuple[set[str], set[str]]:
        rows: list[dict[str, Any]] = [r for path in self.sources for r in read_jsonl(path)]
        return learned(rows, read_jsonl(self.words))

    def pool(self) -> list[dict[str, Any]]:
        return opened(self.bank, *self.known())

    def session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        words: set[str]
        verbs: set[str]
        verbs, words = self.known()
        pool: list[dict[str, Any]] = opened(self.bank, verbs, words)
        nouns: set[str] = met_nouns() | card_nouns(words)
        chosen: list[dict[str, Any]] = pick(pool, read_jsonl(self.attempts))
        with self.lock:
            self.sessions[sid] = {str(i): s for i, s in enumerate(chosen)}
        items: list[Any] = []
        s: dict[str, Any]
        i: int
        for i, s in enumerate(chosen):
            own: list[str] = [t for t in tiles(s["fr"]) if t not in LINKS]
            shuffled: Any = own + s["wrong"] + decoys(s, verbs, nouns, random.Random())
            random.shuffle(shuffled)
            items.append({"id": str(i), "ru": s["ru"], "tiles": shuffled})
        return {
            "session": sid,
            "open": len(pool),
            "total": len(self.bank),
            "items": items,
            "articles": ARTICLES,
            "prepositions": PREPOSITIONS,
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        built: Any | None = body.get("tiles")
        if (
            not isinstance(built, list)
            or len(built) > MAX_TILES
            or not all(isinstance(t, str) for t in built)
        ):
            raise KeyError("bad tiles")
        kind: str | None
        ok: bool
        ok, kind = grade(built, item)
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": DRILL,
            "session": body["session"],
            "fr": item["fr"],
            "answer": " ".join(built),
            "ok": ok,
            "kind": kind,
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
        }
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, [row])
        rules: list[str] = [] if ok else rules_for(built, item, kind, self.forms)
        return {"ok": ok, "kind": kind, "expected": item["fr"], "rules": rules}

    def stats(self) -> dict[str, Any]:
        return stats(read_jsonl(self.attempts), len(self.pool()), len(self.bank))

    def gap_session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        pool: list[dict[str, Any]] = [s for s in self.pool() if None in holes(tiles(s["fr"]))]
        chosen: list[dict[str, Any]] = pick(pool, read_jsonl(self.gaps), GAP_SIZE)
        with self.lock:
            self.gap_sessions[sid] = {str(i): s for i, s in enumerate(chosen)}
        return {
            "session": sid,
            "open": len(pool),
            "articles": ARTICLES,
            "prepositions": PREPOSITIONS,
            "items": [
                {"id": str(i), "ru": s["ru"], "tiles": holes(tiles(s["fr"]))} for i, s in enumerate(chosen)
            ],
        }

    def gap_check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.gap_sessions.get(str(body.get("session")), {}).get(
            str(body.get("id"))
        )
        if item is None:
            raise KeyError("unknown session or item")
        expected: list[str] = tiles(item["fr"])
        links: list[str] = [t for t in expected if t in LINKS]
        fills: Any | None = body.get("fills")
        if (
            not isinstance(fills, list)
            or len(fills) != len(links)
            or not all(isinstance(t, str) and t in LINKS for t in fills)
        ):
            raise KeyError("bad fills")
        given: Iterator[Any] = iter(fills)
        built: list[Any | str] = [next(given) if t is None else t for t in holes(expected)]
        ok: bool = built in [tiles(x) for x in [item["fr"], *item.get("also", [])]]
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": GAPS_DRILL,
            "session": body["session"],
            "fr": item["fr"],
            "links": links,
            "fills": fills,
            "ok": ok,
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
        }
        with self.lock:
            self.gaps.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.gaps, [row])
        return {"ok": ok, "expected": item["fr"], "rules": [] if ok else gap_rules(built, item, self.forms)}

    def gap_stats(self) -> dict[str, Any]:
        return gap_stats(read_jsonl(self.gaps))


def show_stats(bank: list[dict[str, Any]], path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = Sentences(bank, path).stats()
    print(f"sentences: {s['open']} of {s['total']} open, {s['answers']} answers in {s['sessions']} sessions")
    if s["kinds"]:
        print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    g: dict[str, Any] = gap_stats(read_jsonl(path.with_name("gaps.jsonl")))
    if g["answers"]:
        print(f"links: {g['ok']} of {g['answers']} right in {g['sessions']} sessions")
    if g["pairs"]:
        print(
            "  taken for one another: "
            + ", ".join(f"{p['got']} for {p['want']} {p['n']}" for p in g["pairs"])
        )
