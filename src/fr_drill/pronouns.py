"""Pronouns, the eighth tab of `fr-drill drill`: one pronoun missing from a sentence, subject or stressed.

A task gives the Russian and the sentence with a gap (`data/drills/pronouns.json`); the panel is the
same for every task, two rows: subject je, j', tu, il, elle, on, nous, vous, ils, elles and stressed
moi, toi, lui, elle, nous, vous, eux, elles. elle, nous, vous, elles stand in both rows and are one answer.

Mistakes by kind: `role` (the right person, the wrong row: je for moi), `elision` (je for j' or the
other way round), `person` (any other). After a mistake `rules` names why this pronoun is here, a
wrong row gets the rule of the rows too. A session draws like the à and de tab, by the weight of
the sentences tab, each sentence once.

GET  /pronouns/session   new session: Russian and the text around the gap, no answers; the panel
POST /pronouns/check     {session, id, choice, ms} → {ok, kind, expected, use, rules}
GET  /pronouns/stats     accuracy by use and by mistake kind
"""

import json
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import DATA, PROGRESS, append_jsonl, drill, prepositions, read_jsonl, sentences

BANK: Path = DATA / "drills/pronouns.json"
ATTEMPTS: Path = PROGRESS / "pronouns.jsonl"
DRILL: str = "pronouns"
SIZE: int = 20
SUBJECT: list[str] = ["je", "j'", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"]
STRESSED: list[str] = ["moi", "toi", "lui", "elle", "nous", "vous", "eux", "elles"]
PANEL: list[str] = list(dict.fromkeys(SUBJECT + STRESSED))
PERSON: dict[str, str] = {
    "je": "1s",
    "j'": "1s",
    "moi": "1s",
    "tu": "2s",
    "toi": "2s",
    "il": "3m",
    "lui": "3m",
    "elle": "3f",
    "on": "on",
    "nous": "1p",
    "vous": "2p",
    "ils": "3mp",
    "eux": "3mp",
    "elles": "3fp",
}
ROLE: str = (
    "Перед глаголом — подлежащее (je, tu, il, ils); "
    "отдельно, после предлога и после c'est — ударное (moi, toi, lui, eux)."
)
ELISION: str = "Перед гласной и немой h je → j': j'aime, j'habite; перед согласной je: je suis."
VOWEL: re.Pattern[str] = re.compile(r"[aàâeéèêiîoôuûhœ]", re.IGNORECASE)


def gap_at(s: dict[str, Any]) -> re.Match[str] | None:
    """Where the gap stands; None unless it stands exactly once."""
    gap: Any = re.escape(s["gap"])
    pattern: str = rf"(?<![\w']){gap}" if s["gap"].endswith("'") else rf"(?<![\w']){gap}(?![\w'])"
    found: list[re.Match[str]] = list(re.finditer(pattern, s["fr"], re.IGNORECASE))
    return found[0] if len(found) == 1 else None


def problems(s: dict[str, Any], uses: dict[str, Any], core: set[str], forms: set[str]) -> list[str]:
    """What is wrong with one sentence."""
    errors: list[Any]
    gap: Any
    fr: Any
    fr, gap, errors = s.get("fr", ""), s.get("gap", ""), []
    if not fr.strip() or not s.get("ru", "").strip():
        return [f"{s}: empty"]
    if gap not in PANEL:
        return [f"{fr!r}: gap {gap!r} is not in the panel"]
    at: re.Match[str] | None = gap_at(s)
    if at is None:
        return [f"{fr!r}: gap {gap!r} must stand exactly once"]
    if s.get("use") not in uses:
        errors.append(f"{fr!r}: unknown use {s.get('use')!r}")
    after: list[str] = sentences.plain(fr[at.end() :])
    if gap in ("je", "j'") and after and (gap == "j'") != bool(VOWEL.match(after[0])):
        errors.append(f"{fr!r}: {gap!r} before {after[0]!r}: elision is wrong")
    errors += [
        f"{fr!r}: {w!r} is not learned anywhere"
        for w in sentences.plain(fr)
        if w not in core and w not in forms
    ]
    return errors


def load(path: Path = BANK) -> list[dict[str, Any]]:
    """The sentences, each with its `task`: the text with the gap; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    # like prepositions.known(), without the word cards: this bank does not use them
    verbs: list[dict[str, Any]] = drill.load()
    core: set[str] = set(data.get("core", [])) | sentences.met_words()
    forms: set[str] = set(sentences.verb_forms(verbs)) | {v["inf"] for v in verbs}
    out: list[Any]
    seen: set[Any]
    errors: list[Any]
    uses: Any
    uses, errors, seen, out = data.get("uses", {}), [], set(), []
    s: Any
    for s in data.get("sentences", []):
        found: list[str] = problems(s, uses, core, forms)
        errors += found
        if found:
            continue
        at: re.Match[str] | None = gap_at(s)
        assert at is not None  # problems() refused the rest
        after: Any
        before: Any
        before, after = s["fr"][: at.start()], s["fr"][at.end() :]
        task: str = f"{before}___{after}"
        if task in seen:
            errors.append(f"{task!r} twice")
        seen.add(task)
        out.append(
            dict(
                s,
                task=task,
                before=before.strip(),
                after=after.strip(),
                rule=s.get("rule") or uses[s["use"]]["rule"],
                topic=uses[s["use"]]["ru"],
            )
        )
    if not seen:
        errors.append("no sentences")
    if errors:
        raise ValueError("\n".join(errors))
    return out


def grade(choice: str, s: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    """(ok, kind of mistake, rules)."""
    gap: Any = s["gap"]
    if choice == gap:
        return True, None, []
    if {choice, gap} == {"je", "j'"}:
        return False, "elision", [ELISION]
    if PERSON[choice] == PERSON[gap]:
        return False, "role", [s["rule"], ROLE]
    return False, "person", [s["rule"]]


class Pronouns:
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
            "subject": SUBJECT,
            "stressed": STRESSED,
            "items": [
                {"id": str(i), "ru": s["ru"], "before": s["before"], "after": s["after"]}
                for i, s in enumerate(chosen)
            ],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        choice: Any | None = body.get("choice")
        if choice not in PANEL:
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
        return {"ok": ok, "kind": kind, "expected": item["fr"], "use": item["topic"], "rules": rules}

    def stats(self) -> dict[str, Any]:
        return prepositions.stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = prepositions.stats(read_jsonl(path))
    if not s["answers"]:
        print("pronouns: no answers yet")
        return
    print(f"pronouns: {s['ok']} of {s['answers']} right in {s['sessions']} sessions")
    print("  " + ", ".join(f"{t} {x['ok']}/{x['tries']}" for t, x in s["uses"].items()))
