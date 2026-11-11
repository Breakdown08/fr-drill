"""À and de, the seventh tab of `fr-drill drill`: one preposition missing from a built sentence, or verb by
verb.

A task gives the Russian and the sentence with a gap (`data/drills/a-de.json`); the panel is the
same for every task: à, au, à la, à l', aux, de, du, de la, de l', d', des, so the answer takes two
choices at once: which preposition, and which form it takes before this noun.

Mistakes by kind: `prep` (à for de or the other way round), `form` (the right preposition, the wrong
form: à for au). After a mistake `rules` names why this preposition is here, a wrong form gets the
rule of the form instead. A session draws by the weight of the sentences tab: what went wrong
first, then what is unseen, then what held the least; a sentence comes once a session even when
it has two gaps, or the first would give the second away.

Verb mode (`data/drills/a-de-verbs.json`) starts from the government of a verb. First every drawn
verb alone: à, de or none. Then their sentences, mixed so the verb has to be recalled, with a panel
that adds the bare articles: a verb without a preposition is a real answer, and it needs le or la.

GET  /prepositions/session        new session: Russian and the text around the gap, no answers; the panel
GET  /prepositions/verbs/session  new verb session: the same, each item with its `choices`
POST /prepositions/check          {session, id, choice, ms} → {ok, kind, expected, use, rules}
GET  /prepositions/stats          accuracy by use and by mistake kind
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

from fr_drill import DATA, PROGRESS, append_jsonl, drill, read_jsonl, sentences, vocab

BANK: Path = DATA / "drills/a-de.json"
VERBS: Path = DATA / "drills/a-de-verbs.json"
ATTEMPTS: Path = PROGRESS / "prepositions.jsonl"
DRILL: str = "prepositions"
SIZE: int = 20
VERB_SIZE: int = 8
PER_VERB: int = 2
NONE: str = "—"
TO: list[str] = ["à", "au", "à la", "à l'", "aux"]
FROM: list[str] = ["de", "du", "de la", "de l'", "d'", "des"]
ARTICLES: list[str] = ["le", "la", "l'", "les"]
FRAME: list[str] = ["à", "de", NONE]
BARE: str = "Предлога нет, а артикль по роду и числу: le bureau, la gare, l'architecte, les enfants."
FORMS: dict[str, str] = {
    "au": "à + le = au: au bureau. Перед la слияния нет: à la gare.",
    "aux": "à + les = aux: aux enfants, aux vacances.",
    "à la": "Перед словом женского рода à и la стоят раздельно: à la gare.",
    "à l'": "Перед гласной и немой h à + l': à l'architecte, à l'heure.",
    "à": "Перед именем, городом, числом артикля нет, поэтому просто à: à Paris, à Marie, à midi.",
    "du": "de + le = du: du bureau. Перед la слияния нет: de la gare.",
    "des": "de + les = des: des voisins, des vacances.",
    "de la": "Перед словом женского рода de и la стоят раздельно: de la gare.",
    "de l'": "Перед гласной и немой h de + l': de l'architecte.",
    "d'": "Перед гласной и немой h de → d': beaucoup d'enfants.",
    "de": "Перед именем, городом и после beaucoup артикля нет, "
    "поэтому просто de: de Paris, beaucoup de travail.",
}
VOWEL: re.Pattern[str] = re.compile(r"[aàâeéèêiîoôuûhœ]", re.IGNORECASE)
NO_ELISION: set[str] = {"huit"}


def gap_at(s: dict[str, Any]) -> re.Match[str] | None:
    """Where the gap stands; None unless it stands exactly once."""
    found: list[re.Match[str]] = list(
        re.finditer(
            rf"(?<![\w']){re.escape(s['gap'])}(?![\w'])"
            if not s["gap"].endswith("'")
            else rf"(?<![\w']){re.escape(s['gap'])}",
            s["fr"],
            re.IGNORECASE,
        )
    )
    return found[0] if len(found) == 1 else None


def problems(
    s: dict[str, Any],
    uses: dict[str, Any],
    core: set[str],
    forms: set[str],
    cards: set[str],
    panel: list[str] = TO + FROM,
) -> list[str]:
    """What is wrong with one sentence."""
    errors: list[Any]
    gap: Any
    fr: Any
    fr, gap, errors = s.get("fr", ""), s.get("gap", ""), []
    if not fr.strip() or not s.get("ru", "").strip():
        return [f"{s}: empty"]
    if gap not in panel:
        return [f"{fr!r}: gap {gap!r} is not in the panel"]
    at: re.Match[str] | None = gap_at(s)
    if at is None:
        return [f"{fr!r}: gap {gap!r} must stand exactly once"]
    if s.get("use") not in uses:
        errors.append(f"{fr!r}: unknown use {s.get('use')!r}")
    after: Any = fr[at.end() :].strip()
    noun: str = sentences.plain(after)[0] if sentences.plain(after) else ""
    elided: Any = gap.endswith("'")
    if elided != bool(VOWEL.match(noun) and noun not in NO_ELISION) and gap in (
        "à l'",
        "de l'",
        "d'",
        "de la",
        "à la",
        "de",
    ):
        errors.append(f"{fr!r}: {gap!r} before {noun!r}: elision is wrong")
    if gap in ("au", "du", "le") and f"la {noun}" in cards:
        errors.append(f"{fr!r}: {gap!r} before a feminine {noun!r}")
    if gap in ("à la", "de la", "la") and f"le {noun}" in cards:
        errors.append(f"{fr!r}: {gap!r} before a masculine {noun!r}")
    errors += [
        f"{fr!r}: {w!r} is not learned anywhere"
        for w in sentences.plain(fr)
        if w not in core and w not in forms
    ]
    return errors


def known(data: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Words a bank may use: its core with what was met, verb forms, word cards."""
    verbs: list[dict[str, Any]] = drill.load()
    return (
        set(data.get("core", [])) | sentences.met_words(),
        set(sentences.verb_forms(verbs)) | {v["inf"] for v in verbs},
        {c["fr"] for c in vocab.load()},
    )


def load(path: Path = BANK) -> list[dict[str, Any]]:
    """The sentences, each with its `task`: the text with the gap; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    cards: set[str]
    forms: set[str]
    core: set[str]
    core, forms, cards = known(data)
    out: list[Any]
    seen: set[Any]
    errors: list[Any]
    uses: Any
    uses, errors, seen, out = data.get("uses", {}), [], set(), []
    s: Any
    for s in data.get("sentences", []):
        found: list[str] = problems(s, uses, core, forms, cards)
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


def load_verbs(path: Path = VERBS) -> list[dict[str, Any]]:
    """Each verb with its `frame` item and its `sentences` items; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    cards: set[str]
    forms: set[str]
    core: set[str]
    core, forms, cards = known(data)
    out: list[Any]
    seen: set[Any]
    errors: list[Any]
    errors, seen, out = [], set(), []
    v: Any
    for v in data.get("verbs", []):
        name: str = " ".join(x for x in (v["verb"], "" if v["prep"] == NONE else v["prep"], v["after"]) if x)
        task: str = f"{v['verb']} ___ {v['after']}"
        if v["prep"] not in FRAME:
            errors.append(f"{name!r}: prep {v['prep']!r}")
        if task in seen:
            errors.append(f"{task!r} twice")
        seen.add(task)
        common: dict[str, Any] = {"topic": name, "rule": v["rule"]}
        frame: dict[str, Any] = dict(
            common,
            fr=name,
            ru=v["ru"],
            gap=v["prep"],
            task=task,
            before=v["verb"],
            after=v["after"],
            choices=FRAME,
        )
        items: list[Any] = []
        allowed: list[str] = {"à": TO, "de": FROM, NONE: ARTICLES}.get(v["prep"], [])
        s: Any
        for s in v.get("sentences", []):
            found: list[str] = problems(dict(s, use=name), {name: {}}, core, forms, cards, allowed)
            errors += found
            if found:
                continue
            at: re.Match[str] | None = gap_at(s)
            assert at is not None  # problems() refused the rest
            after: Any
            before: Any
            before, after = s["fr"][: at.start()], s["fr"][at.end() :]
            sentence_task: str = f"{before}___{after}"
            if (sentence_task, s["gap"]) in seen:  # parler au voisin and du voisin differ by the Russian only
                errors.append(f"{sentence_task!r} twice")
            seen.add((sentence_task, s["gap"]))
            items.append(
                dict(
                    s,
                    **common,
                    task=sentence_task,
                    before=before.strip(),
                    after=after.strip(),
                    choices=TO + FROM + ARTICLES,
                )
            )
        if len(items) < PER_VERB and not errors:
            errors.append(f"{name!r}: fewer than {PER_VERB} sentences")
        out.append({"frame": frame, "sentences": items})
    if not out:
        errors.append("no verbs")
    if errors:
        raise ValueError("\n".join(errors))
    return out


def pick(
    pool: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = SIZE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """By weight, every sentence once."""
    rng = rng or random.Random()
    rows: list[dict[str, Any]] = [dict(r, fr=r["task"]) for r in history]
    order: list[dict[str, Any]] = vocab.weighted(
        pool, [sentences.weight(s["task"], rows) for s in pool], len(pool), rng
    )
    chosen: dict[str, dict[str, Any]] = {}
    s: dict[str, Any]
    for s in order:
        chosen.setdefault(s["fr"], s)
    return list(chosen.values())[:size]


def pick_verbs(
    pool: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = VERB_SIZE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """The frames of the verbs that need it most, then two sentences of each, mixed."""
    rng = rng or random.Random()
    rows: list[dict[str, Any]] = [dict(r, fr=r["task"]) for r in history]
    scored: list[dict[Any, float]] = [
        {s["task"]: sentences.weight(s["task"], rows) for s in [v["frame"], *v["sentences"]]} for v in pool
    ]
    chosen: list[int] = vocab.weighted(list(range(len(pool))), [max(w.values()) for w in scored], size, rng)
    mixed: list[Any]
    frames: list[Any]
    frames, mixed = [], []
    i: int
    for i in chosen:
        frames.append(pool[i]["frame"])
        order: list[Any] = sorted(pool[i]["sentences"], key=lambda s: (-scored[i][s["task"]], rng.random()))
        mixed += order[:PER_VERB]
    rng.shuffle(mixed)
    return frames + mixed


def grade(choice: str, s: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    """(ok, kind of mistake, rules)."""
    if choice == s["gap"]:
        return True, None, []
    if (choice in TO) == (s["gap"] in TO):
        return False, "form", [FORMS[s["gap"]]]
    return False, "prep", [s["rule"], FORMS[s["gap"]]] if s["gap"] not in ("à", "de") else [s["rule"]]


def side(choice: str) -> int:
    return 0 if choice in TO else 1 if choice in FROM else 2


def grade_verb(choice: str, s: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    """(ok, kind of mistake, rules) when the panel holds no preposition too."""
    gap: Any = s["gap"]
    if choice == gap:
        return True, None, []
    if s["choices"] is FRAME:
        return False, "prep", [s["rule"]]
    if side(choice) == side(gap):
        return False, "form", [FORMS.get(gap, BARE)]
    return False, "prep", [s["rule"], FORMS[gap]] if gap in FORMS and gap not in ("à", "de") else [s["rule"]]


def stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    right: Counter[Any]
    tries: Counter[Any]
    tries, right = Counter(r["topic"] for r in history), Counter(r["topic"] for r in history if r["ok"])
    return {
        "answers": len(history),
        "ok": sum(r["ok"] for r in history),
        "sessions": len({r["session"] for r in history}),
        "kinds": dict(Counter(r["kind"] for r in history if not r["ok"]).most_common()),
        "uses": {t: {"ok": right[t], "tries": n} for t, n in tries.most_common()},
    }


class Prepositions:
    def __init__(
        self, bank: list[dict[str, Any]], attempts: Path = ATTEMPTS, verbs: list[dict[str, Any]] | None = None
    ) -> None:
        self.bank: list[dict[str, Any]] = bank
        self.verbs: list[dict[str, Any]] = load_verbs() if verbs is None else verbs
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        chosen: list[dict[str, Any]] = pick(self.bank, read_jsonl(self.attempts))
        with self.lock:
            self.sessions[sid] = {str(i): s for i, s in enumerate(chosen)}
        return {
            "session": sid,
            "to": TO,
            "from": FROM,
            "items": [
                {"id": str(i), "ru": s["ru"], "before": s["before"], "after": s["after"]}
                for i, s in enumerate(chosen)
            ],
        }

    def verb_session(self) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        chosen: list[dict[str, Any]] = pick_verbs(self.verbs, read_jsonl(self.attempts))
        with self.lock:
            self.sessions[sid] = {str(i): s for i, s in enumerate(chosen)}
        return {
            "session": sid,
            "to": TO,
            "from": FROM,
            "articles": ARTICLES,
            "none": NONE,
            "items": [
                {
                    "id": str(i),
                    "ru": s["ru"],
                    "before": s["before"],
                    "after": s["after"],
                    "choices": s["choices"],
                }
                for i, s in enumerate(chosen)
            ],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        choice: Any = body.get("choice")
        if choice not in item.get("choices", TO + FROM):
            raise KeyError("bad choice")
        rules: list[str]
        kind: str | None
        ok: bool
        ok, kind, rules = (grade_verb if "choices" in item else grade)(choice, item)
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
        return stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = stats(read_jsonl(path))
    if not s["answers"]:
        print("à / de: no answers yet")
        return
    print(f"à / de: {s['ok']} of {s['answers']} right in {s['sessions']} sessions")
    print("  " + ", ".join(f"{t} {x['ok']}/{x['tries']}" for t, x in s["uses"].items()))
