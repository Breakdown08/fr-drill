"""Gender, the ninth tab of `fr-drill drill`: a noun of the word cards, which article — le or la.

The nouns and their gender come from the Words tab (`vocab.articles`: le/la, or un/une where the
card has l'). `data/drills/gender.json` only explains, after Heminway's Complete French ch. 2–3:
a group by meaning first (people, countries, seasons), else the longest ending of the first word.
Every noun gets a kind: `rule` (the rule gives its gender), `exception` (a rule says the other),
`memorize` (no rule: mute -e, two-sided endings like -eur). A mistake is recorded with that kind,
so the stats tell missing rules from missing memory.

GET  /gender/session   new session: nouns with their article pair, no answers
POST /gender/check     {session, id, choice, ms} → {ok, kind, expected, use, rules}
GET  /gender/stats     accuracy by rule and by mistake kind
"""

import json
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import DATA, PROGRESS, append_jsonl, prepositions, read_jsonl, vocab

BANK: Path = DATA / "drills/gender.json"
ATTEMPTS: Path = PROGRESS / "gender.jsonl"
DRILL: str = "gender"
SIZE: int = 20
NO_RULE: str = "без правила"
EXCEPTION: str = "Исключение: запомни это слово с артиклем."


def ending(noun: str, endings: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    """The longest ending of the noun's first word with its rule, or None."""
    head: str = noun.split()[0]
    found: list[tuple[Any, dict[str, Any]]] = [(e, r) for r in endings for e in r["end"] if head.endswith(e)]
    return max(found, key=lambda x: len(x[0])) if found else None


def explain(card: str, noun: str, g: str, data: dict[str, Any]) -> tuple[str, str, str]:
    """(topic, kind, rule) of one noun: its group, else its ending, else no rule."""
    grp: Any
    for grp in data["groups"]:
        if card in grp["words"]:
            want: Any | str = grp["g"] if grp["g"] != "e" else ("f" if noun.endswith("e") else "m")
            if want is None:
                return grp["ru"], "rule", grp["rule"]
            return grp["ru"], "rule" if want == g else "exception", grp["rule"]
    found: tuple[str, dict[str, Any]] | None = ending(noun, data["endings"])
    if found is None:
        return NO_RULE, "memorize", data["no_rule"]
    r: dict[str, Any]
    end: str
    end, r = found
    if r["g"] is None:
        return f"-{end}", "memorize", r["rule"]
    return f"-{end}", "rule" if r["g"] == g else "exception", r["rule"]


def load(path: Path = BANK, cards: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One item per noun of the word cards; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    cards = vocab.load() if cards is None else cards
    found: tuple[tuple[str, str], str, str] | None
    nouns: dict[Any, tuple[dict[str, Any], tuple[tuple[str, str], str, str]]] = {
        c["fr"]: (c, found) for c in cards if (found := vocab.articles(c))
    }
    errors: list[str] = []
    if not data.get("no_rule", "").strip():
        errors.append("no_rule: empty")
    grouped: set[str] = set()
    grp: Any
    for grp in data.get("groups", []):
        if grp.get("g") not in ("m", "f", "e", None) or not grp.get("rule", "").strip() or not grp.get("ru"):
            errors.append(f"group {grp.get('ru')!r}: needs ru, rule and g: m, f, e or null")
        w: Any
        for w in grp.get("words", []):
            if w not in nouns:
                errors.append(f"group {grp.get('ru')!r}: {w!r} is not a noun card")
            if w in grouped:
                errors.append(f"{w!r} is in two groups")
            grouped.add(w)
    seen: set[str] = set()
    r: Any
    for r in data.get("endings", []):
        if r.get("g") not in ("m", "f", None) or not r.get("rule", "").strip() or not r.get("end"):
            errors.append(f"ending {r.get('end')}: needs end, rule and g: m, f or null")
        e: Any
        for e in r.get("end", []):
            if e in seen:
                errors.append(f"ending -{e} twice")
            seen.add(e)
    if errors:
        raise ValueError("\n".join(errors))
    out: list[Any] = []
    noun: str
    right: str
    pair: tuple[str, str]
    c: dict[str, Any]
    fr: Any
    for fr, (c, (pair, right, noun)) in nouns.items():
        g: str = "m" if right in ("le", "un") else "f"
        rule: str
        kind: str
        topic: str
        topic, kind, rule = explain(fr, noun, g, data)
        out.append(
            {
                "fr": fr,
                "task": noun,
                "noun": noun,
                "pair": list(pair),
                "gap": right,
                "g": g,
                "ru": c["ru"],
                "full": f"{right} {noun}",
                "topic": topic,
                "kind": kind,
                "rule": rule,
            }
        )
    if not out:
        raise ValueError("no nouns")
    return out


def grade(choice: str, item: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    """(ok, kind of mistake, rules)."""
    if choice == item["gap"]:
        return True, None, []
    return False, item["kind"], [item["rule"], EXCEPTION] if item["kind"] == "exception" else [item["rule"]]


class Gender:
    def __init__(self, bank: list[dict[str, Any]], attempts: Path = ATTEMPTS) -> None:
        self.bank: list[dict[str, Any]] = bank
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        chosen: list[dict[str, Any]] = prepositions.pick(self.bank, read_jsonl(self.attempts), SIZE)
        with self.lock:
            self.sessions[sid] = {str(i): s for i, s in enumerate(chosen)}
        return {
            "session": sid,
            "items": [
                {"id": str(i), "ru": s["ru"], "noun": s["noun"], "pair": s["pair"]}
                for i, s in enumerate(chosen)
            ],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        choice: str = str(body.get("choice"))
        if choice not in item["pair"]:
            raise KeyError("bad choice")
        rules: list[str]
        kind: str | None
        ok: bool
        ok, kind, rules = grade(choice, item)
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": DRILL,
            "session": body["session"],
            "task": item["task"],
            "topic": item["topic"],
            "expected": item["gap"],
            "choice": choice,
            "ok": ok,
            "kind": kind,
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
        }
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, [row])
        return {
            "ok": ok,
            "kind": kind,
            "expected": item["full"],
            "use": item["topic"],
            "rules": rules or [item["rule"]],
        }

    def stats(self) -> dict[str, Any]:
        return prepositions.stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = prepositions.stats(read_jsonl(path))
    if not s["answers"]:
        print("gender: no answers yet")
        return
    print(f"gender: {s['ok']} of {s['answers']} right in {s['sessions']} sessions")
    print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    print("  " + ", ".join(f"{t} {x['ok']}/{x['tries']}" for t, x in s["uses"].items()))
