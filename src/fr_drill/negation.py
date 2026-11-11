"""Negation, the fourth tab of `fr-drill drill`: ne … pas with présent of -er verbs, in a short phrase.

A task gives a pronoun, a verb and the rest of the phrase (`data/drills/negation-er.json`):
«ils · regarder la télé» → «ils ne regardent pas la télé». The answer may stop after `pas` —
the rest of the phrase is given, typing it is optional. The verb form comes from `drill.form`,
so the conjugation rule is one for both tabs. Before a vowel or mute h ne → n', and je stays
je: je n'aime pas.

Mistakes by kind: `ne` or `pas` missing, `elision` (ne aime, j'n'aime), `order` (je ne pas parle),
then the verb itself as in the conjugation tab — `accent`, `spelling`, `ending`, `pronoun` —
and `tail` when only the given rest of the phrase is mistyped.

GET  /negation/session   new session: items without answers
POST /negation/check     {session, id, answer, ms} → {ok, kind, expected}
GET  /negation/stats     accuracy by person and by mistake kind
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

CONTEXTS: Path = DATA / "drills/negation-er.json"
ATTEMPTS: Path = PROGRESS / "negation.jsonl"
DRILL: str = "negation-er"
SIZE: int = 15
HISTORY: int = 300
# After ne … pas these turn into de: je ne mange pas de pain — a later step of the topic.
CHANGES_AFTER_PAS: re.Pattern[str] = re.compile(r"^(un|une|des|du|de la|de l'|d')(\s|$)|^(de l'|d')")


def load(path: Path = CONTEXTS, verbs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The contexts; raise ValueError listing every problem."""
    known: dict[Any, dict[str, Any]] = {v["inf"]: v for v in (verbs if verbs is not None else drill.load())}
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    seen: set[Any]
    errors: list[Any]
    errors, seen = [], set()
    c: Any
    i: int
    for i, c in enumerate(data.get("contexts", [])):
        ru: Any
        fr: Any
        inf: Any
        inf, fr, ru = c.get("inf", ""), c.get("fr", ""), c.get("ru", "")
        if inf not in known:
            errors.append(f"{i}: {inf!r} is not in verbs-er.json")
        if not fr.strip():
            errors.append(f"{i}: {inf!r} has no rest of the phrase")
        if CHANGES_AFTER_PAS.match(drill.norm(fr)):
            errors.append(f"{i}: {inf!r} {fr!r} — the article turns into de after pas")
        if not ru.startswith("не "):
            errors.append(f"{i}: {inf!r} {fr!r} — ru must start with «не »")
        if (inf, fr) in seen:
            errors.append(f"{i}: {inf!r} {fr!r} twice")
        seen.add((inf, fr))
    if not seen:
        errors.append("no contexts")
    if errors:
        raise ValueError("\n".join(errors))
    contexts: list[dict[str, Any]] = data["contexts"]
    return contexts


def head(inf: str, pronoun: str) -> str:
    """Pronoun + ne + verb + pas: je n'aime pas, nous ne mangeons pas."""
    verb: str = drill.form(inf, drill.person_of(pronoun))
    ne: str = "n'" if verb[0] in drill.VOWELS else "ne "
    return f"{pronoun} {ne}{verb} pas"


def sentence(inf: str, pronoun: str, rest: str) -> str:
    return f"{head(inf, pronoun)} {rest}"


def words(s: str) -> list[str]:
    """Normalized words, elided ones apart: "j'n'aime pas" → ["j'", "n'", "aime", "pas"]."""
    s = re.sub(r"\b([jn])'\s*", r"\1' ", drill.norm(s).rstrip(".!? "))
    return s.split()


def grade(answer: str, inf: str, pronoun: str, rest: str) -> tuple[bool, str | None]:
    """(ok, kind of mistake): ne, pas, pronoun, accent, spelling, ending, other, order, elision, tail."""
    r: list[str]
    e: list[str]
    a: list[str]
    a, e, r = words(answer), words(head(inf, pronoun)), words(rest)
    tail: list[str] = a[len(e) :]
    if a[: len(e)] == e and tail in ([], r):
        return True, None
    if not {"ne", "n'"} & set(a):
        return False, "ne"
    if "pas" not in a:
        return False, "pas"
    stem: str
    ev: str
    ev, stem = e[-2], drill.bare(inf[:-2])
    others: set[str] = {"ne", "n'", "pas", "j'", "je", *r}
    av: str = next((w for w in a[1:] if w not in others), "")
    if a[0] not in (e[0], "j'" if e[0] == "je" else e[0]):
        return False, "pronoun"
    if av != ev:
        if drill.bare(av) == drill.bare(ev):
            return False, "accent"
        if ev.endswith(("geons", "çons")) and drill.bare(av) == stem + "ons":
            return False, "spelling"
        if drill.bare(av).startswith(stem):
            return False, "ending"
        return False, "other"
    core: list[str] = ["ne" if w == "n'" else w for w in a if w in ("ne", "n'", av, "pas")]
    if core != ["ne", av, "pas"]:
        return False, "order"
    if a[: len(e)] != e:
        return False, "elision"
    return False, "tail"


def pick(
    contexts: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = SIZE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Persons evenly, extra turns to the weakest; contexts that went wrong for that person first."""
    rng = rng or random.Random()
    recent: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL][-HISTORY:]
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
    used: set[Any]
    items: list[Any]
    items, used = [], set()
    person: str
    for person in slots:
        weights: list[float | int] = [
            (1 + 4 * misses[(c["inf"], c["fr"], person)] + (2 if not seen[(c["inf"], c["fr"])] else 0))
            * (0.1 if c["inf"] in used else 1)
            for c in contexts
        ]
        c: dict[str, Any] = rng.choices(contexts, weights)[0]
        used.add(c["inf"])
        items.append(
            {
                "verb": c["inf"],
                "rest": c["fr"],
                "ru": c["ru"],
                "pronoun": rng.choice(drill.PERSONS[person][0]),
                "person": person,
            }
        )
    rng.shuffle(items)
    item: dict[str, Any]
    i: int
    for i, item in enumerate(items):
        item["id"] = str(i)
    return items


def stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL]
    by: dict[str, dict[str, Any]] = {
        p: {
            "tries": sum(r["person"] == p for r in rows),
            "ok": sum(r["ok"] for r in rows if r["person"] == p),
        }
        for p in drill.PERSONS
    }
    return {
        "answers": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "sessions": len({r["session"] for r in rows}),
        "persons": by,
        "kinds": dict(Counter(r["kind"] for r in rows if not r["ok"]).most_common()),
        "hard_verbs": [v for v, _ in Counter(r["verb"] for r in rows if not r["ok"]).most_common(5)],
    }


class Negation:
    def __init__(self, contexts: list[dict[str, Any]], attempts: Path = ATTEMPTS) -> None:
        self.contexts: list[dict[str, Any]] = contexts
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        items: list[dict[str, Any]] = pick(self.contexts, read_jsonl(self.attempts))
        with self.lock:
            self.sessions[sid] = {i["id"]: i for i in items}
        public: tuple[str, str, str, str, str] = ("id", "verb", "rest", "ru", "pronoun")
        return {"session": sid, "items": [{k: i[k] for k in public} for i in items]}

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
        return {"ok": ok, "kind": kind, "expected": expected}

    def stats(self) -> dict[str, Any]:
        return stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = stats(read_jsonl(path))
    if not s["answers"]:
        print("no negation answers yet")
        return
    print(
        f"negation: {s['answers']} answers in {s['sessions']} sessions, "
        f"{100 * s['ok'] // s['answers']}% right"
    )
    if s["kinds"]:
        print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    if s["hard_verbs"]:
        print("  hardest verbs: " + ", ".join(s["hard_verbs"]))
