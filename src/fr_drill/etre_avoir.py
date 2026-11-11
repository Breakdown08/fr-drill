"""Irregular verbs, the fifth tab of `fr-drill drill`: présent of être, avoir, aller, faire, prendre and venir
in a short phrase. The tab trains one verb at a time or all of them mixed — mixed only once each is
known by heart.

A task gives a pronoun, the verb and the rest of the phrase (`data/drills/etre-avoir.json`):
«tu · avoir · faim» → «tu as faim». The answer may stop after the verb — the rest of the phrase
is given, typing it is optional. The forms are a table, not a rule; je + ai → j'ai.

Mode `choose` hides the verb: the pronoun, the rest and the meaning in Russian are given, the verb
is chosen by sense — «tu · … faim · хотеть есть» → «tu as faim», not «tu es». The literal
translation (`lit`, «досл. иметь голод») would give the verb away, so it comes only with the check.

Mistakes by kind: `verb` — a form of another verb (tu es faim: être for avoir, the main trap
of the topic), `person` — the right verb, another person (tu est), `elision` (je ai), `accent`
(vous etes), `pronoun`, `other`, and `tail` when only the given rest of the phrase is mistyped.

GET  /etre-avoir/session   new session: items without answers; ?verb=être for one verb, none for all;
                           ?mode=choose — all verbs, hidden
POST /etre-avoir/check     {session, id, answer, ms} → {ok, kind, expected, lit}
GET  /etre-avoir/stats     accuracy by verb, by person and by mistake kind
"""

import json
import random
import re
import secrets
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import DATA, PROGRESS, append_jsonl, drill, read_jsonl

CONTEXTS: Path = DATA / "drills/etre-avoir.json"
ATTEMPTS: Path = PROGRESS / "etre-avoir.jsonl"
DRILL: str = "etre-avoir"
SIZE: int = 18
HISTORY: int = 300
FORMS: dict[str, dict[str, str]] = {
    "être": {"je": "suis", "tu": "es", "il": "est", "nous": "sommes", "vous": "êtes", "ils": "sont"},
    "avoir": {"je": "ai", "tu": "as", "il": "a", "nous": "avons", "vous": "avez", "ils": "ont"},
    "aller": {"je": "vais", "tu": "vas", "il": "va", "nous": "allons", "vous": "allez", "ils": "vont"},
    "faire": {"je": "fais", "tu": "fais", "il": "fait", "nous": "faisons", "vous": "faites", "ils": "font"},
    "prendre": {
        "je": "prends",
        "tu": "prends",
        "il": "prend",
        "nous": "prenons",
        "vous": "prenez",
        "ils": "prennent",
    },
    "venir": {
        "je": "viens",
        "tu": "viens",
        "il": "vient",
        "nous": "venons",
        "vous": "venez",
        "ils": "viennent",
    },
}
MODES: tuple[str, str] = ("verb", "choose")
# A possessive changes with the person (j'ai mon stylo, nous avons notre stylo). Adjectives that agree
# (fatigué / fatiguées) are kept out by hand: a suffix cannot tell them from café.
POSSESSIVES: set[str] = {
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "son",
    "sa",
    "ses",
    "notre",
    "nos",
    "votre",
    "vos",
    "leur",
    "leurs",
}


def load(path: Path = CONTEXTS) -> list[dict[str, Any]]:
    """The contexts; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    seen: set[Any]
    errors: list[Any]
    errors, seen = [], set()
    c: Any
    i: int
    for i, c in enumerate(data.get("contexts", [])):
        ru: Any
        fr: Any
        verb: Any
        verb, fr, ru = c.get("verb", ""), c.get("fr", ""), c.get("ru", "")
        if verb not in FORMS:
            errors.append(f"{i}: {verb!r} is not one of {', '.join(FORMS)}")
        if not fr.strip():
            errors.append(f"{i}: {verb!r} has no rest of the phrase")
        elif POSSESSIVES & set(drill.norm(fr).split()):
            errors.append(f"{i}: {verb!r} {fr!r} — changes with the person")
        if not ru.strip():
            errors.append(f"{i}: {verb!r} {fr!r} has no translation")
        if not isinstance(c.get("lit", ""), str):
            errors.append(f"{i}: {verb!r} {fr!r} — lit is not a string")
        if (verb, fr) in seen:
            errors.append(f"{i}: {verb!r} {fr!r} twice")
        seen.add((verb, fr))
    for verb in FORMS:
        if not any(v == verb for v, _ in seen):
            errors.append(f"no contexts for {verb}")
    if errors:
        raise ValueError("\n".join(errors))
    contexts: list[dict[str, Any]] = data["contexts"]
    return contexts


def head(verb: str, pronoun: str) -> str:
    """Pronoun + form: j'ai, tu es, nous avons."""
    form: str = FORMS[verb][drill.person_of(pronoun)]
    return "j'" + form if pronoun == "je" and form[0] in drill.VOWELS else f"{pronoun} {form}"


def sentence(verb: str, pronoun: str, rest: str) -> str:
    return f"{head(verb, pronoun)} {rest}"


def words(s: str) -> list[str]:
    """Normalized words, j' apart: "J'ai faim." → ["j'", "ai", "faim"]."""
    s = re.sub(r"\bj'\s*", "j' ", drill.norm(s).rstrip(".!? "))
    return s.split()


def grade(answer: str, verb: str, pronoun: str, rest: str) -> tuple[bool, str | None]:
    """(ok, kind of mistake): elision, accent, pronoun, verb, person, other, tail."""
    r: list[str]
    e: list[str]
    a: list[str]
    a, e, r = words(answer), words(head(verb, pronoun)), words(rest)
    if a[: len(e)] == e and a[len(e) :] in ([], r):
        return True, None
    if not a:
        return False, "other"
    ev: str
    ep: str
    ap: str
    ap, ep, ev = a[0], e[0], e[-1]
    av: str = a[1] if len(a) > 1 else ""
    if {ap, ep} == {"je", "j'"} and av == ev:  # je ai, j'suis
        return False, "elision"
    if ap != ep and {ap, ep} != {"je", "j'"}:
        return False, "pronoun"
    if av != ev:
        if drill.bare(av) == drill.bare(ev):
            return False, "accent"
        others: set[str] = {drill.bare(f) for v, forms in FORMS.items() if v != verb for f in forms.values()}
        if drill.bare(av) in others:
            return False, "verb"
        if drill.bare(av) in {drill.bare(f) for f in FORMS[verb].values()}:
            return False, "person"
        return False, "other"
    return False, "tail"


def pick(
    contexts: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = SIZE,
    rng: random.Random | None = None,
    verbs: tuple[str, ...] = tuple(FORMS),
) -> list[dict[str, Any]]:
    """Persons evenly, extra turns to the weakest; the verbs in equal share; contexts that went wrong first.
    Only answers on these verbs count: a weak person in avoir says nothing about être."""
    rng = rng or random.Random()
    recent: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL and r.get("verb") in verbs][
        -HISTORY:
    ]
    misses: Counter[tuple[Any, Any, Any]] = Counter(
        (r["verb"], r["rest"], r["person"]) for r in recent if not r["ok"]
    )
    tries: Counter[Any] = Counter(r["person"] for r in recent)
    wrong: Counter[Any] = Counter(r["person"] for r in recent if not r["ok"])
    seen: Counter[tuple[Any, Any]] = Counter((r["verb"], r["rest"]) for r in recent)
    persons: list[str] = list(drill.PERSONS)
    slots: list[str] = persons * (size // len(persons))
    weakest: list[str] = sorted(
        persons, key=lambda p: (-(wrong[p] / tries[p]) if tries[p] else -1, rng.random())
    )
    slots += weakest[: size - len(slots)]
    turns: list[str] = (list(verbs) * size)[:size]
    rng.shuffle(turns)
    # With six verbs each has three turns: left to chance, a missed verb rarely meets its person again.
    fixed: set[int] = set()
    _n: int
    person: Any
    _: Any
    verb: Any
    for (verb, _, person), _n in misses.most_common():
        mine: list[int] = [i for i in range(size) if i not in fixed and slots[i] == person]
        if any(turns[i] == verb for i in mine):
            fixed.add(next(i for i in mine if turns[i] == verb))
            continue
        spare: list[int] = [j for j in range(size) if j not in fixed and turns[j] == verb]
        if mine and spare:
            j: int
            i: int
            i, j = mine[0], spare[0]
            turns[i], turns[j] = turns[j], turns[i]
            fixed.add(i)
    used: set[Any]
    items: list[Any]
    items, used = [], set()
    for person, verb in zip(slots, turns, strict=False):
        pool: list[dict[str, Any]] = [c for c in contexts if c["verb"] == verb]
        weights: list[float | int] = [
            (1 + 4 * misses[(verb, c["fr"], person)] + (2 if not seen[(verb, c["fr"])] else 0))
            * (0.1 if c["fr"] in used else 1)
            for c in pool
        ]
        c: dict[str, Any] = rng.choices(pool, weights)[0]
        used.add(c["fr"])
        items.append(
            {
                "verb": verb,
                "rest": c["fr"],
                "ru": c["ru"],
                "lit": c.get("lit", ""),
                "pronoun": rng.choice(drill.PERSONS[person][0]),
                "person": person,
            }
        )
    rng.shuffle(items)
    item: dict[str, Any]
    for i, item in enumerate(items):
        item["id"] = str(i)
    return items


def stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL]

    def share(key: str, value: str) -> dict[str, Any]:
        mine: list[dict[str, Any]] = [r for r in rows if r.get(key) == value]
        return {"tries": len(mine), "ok": sum(r["ok"] for r in mine)}

    return {
        "answers": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "sessions": len({r["session"] for r in rows}),
        "verbs": {v: share("verb", v) for v in FORMS},
        "persons": {p: share("person", p) for p in drill.PERSONS},
        "kinds": dict(Counter(r["kind"] for r in rows if not r["ok"]).most_common()),
        "choose": share("mode", "choose"),
    }


class EtreAvoir:
    def __init__(self, contexts: list[dict[str, Any]], attempts: Path = ATTEMPTS) -> None:
        self.contexts: list[dict[str, Any]] = contexts
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self, verbs: list[str] | None = None, mode: str = "verb") -> dict[str, Any]:
        """Raise ValueError on a verb outside FORMS or an unknown mode; no verbs — all of them.
        Mode choose takes all the verbs and keeps them, and the literal translation, out of the items."""
        unknown: set[str]
        if unknown := set(verbs or ()) - set(FORMS):
            raise ValueError(f"unknown verb: {', '.join(sorted(unknown))}")
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        chosen: tuple[str, ...] = (
            tuple(v for v in FORMS if v in verbs) if verbs and mode == "verb" else tuple(FORMS)
        )
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        items: list[dict[str, Any]] = pick(self.contexts, read_jsonl(self.attempts), verbs=chosen)
        i: dict[str, Any]
        for i in items:
            i["mode"] = mode
        with self.lock:
            self.sessions[sid] = {i["id"]: i for i in items}
        public: tuple[str, str, str, str, str, str] | tuple[str, str, str, str] = (
            ("id", "verb", "rest", "ru", "lit", "pronoun")
            if mode == "verb"
            else ("id", "rest", "ru", "pronoun")
        )
        return {"session": sid, "mode": mode, "items": [{k: i[k] for k in public} for i in items]}

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        answer: str = str(body.get("answer", ""))[:200]
        expected: str = sentence(item["verb"], item["pronoun"], item["rest"])
        kind: str | None
        ok: bool
        ok, kind = grade(answer, item["verb"], item["pronoun"], item["rest"])
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": DRILL,
            "session": body["session"],
            "mode": item["mode"],
            "verb": item["verb"],
            "rest": item["rest"],
            "pronoun": item["pronoun"],
            "person": item["person"],
            "expected": expected,
            "answer": answer,
            "ok": ok,
            "kind": kind,
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
        }
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, [row])
        return {"ok": ok, "kind": kind, "expected": expected, "lit": item["lit"]}

    def stats(self) -> dict[str, Any]:
        return stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = stats(read_jsonl(path))
    if not s["answers"]:
        print("no irregular verb answers yet")
        return
    print(
        f"irregular verbs: {s['answers']} answers in {s['sessions']} sessions, "
        f"{100 * s['ok'] // s['answers']}% right"
    )
    print("  " + ", ".join(f"{v} {x['ok']}/{x['tries']}" for v, x in s["verbs"].items()))
    if s["choose"]["tries"]:
        print(f"  by sense: {s['choose']['ok']}/{s['choose']['tries']}")
    if s["kinds"]:
        print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
