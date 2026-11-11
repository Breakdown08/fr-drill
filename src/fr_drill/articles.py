"""Articles, the tenth tab of `fr-drill drill`: one article missing from a built sentence, ten levels.

A task gives the Russian and the sentence with a gap (`data/drills/articles.json`). Every level adds
one decision and shows only its articles: 1 un/une, 2 le/la, 3 plus l', les, 4 first mention → the
known one, 5 des or les, 6 in general → le, 7 plus the partitive, 8 aimer le or manger du, 9 plus de
after a negation, 10 the exceptions (ce n'est pas, amounts). A level session draws NEW sentences of
its own and fills up with earlier ones whose article is on its panel, so what was learned stays mixed
in. A level is closed after PASS of SIZE twice in a row; the next one is the learner's choice.

Levels 1–3 also make sentences from the noun cards (`made`: a frame per use, nouns that do not fit
are skipped), so a session is a new draw of hundreds, not the same twenty in another order. Half of
a level's own sentences are made, half written; earlier levels lend written ones only. Rows carry
the bank version, and only sessions of the current version close a level.

Mistakes by layer, so the stats tell which one is missing: `gender` (the right article, the other
gender: le for la), `form` (the right article, a wrong number or elision: le for l'), `type` (the
wrong article: le for du), `de` (de missed after a negation or an amount, or put where it does not
belong). A gender mistake brings the noun's rule from the gender tab.

GET  /articles/session?level=N  new session: Russian and the text around the gap, no answers; the panel
POST /articles/check            {session, id, choice, ms} → {ok, kind, expected, use, rules}
GET  /articles/stats            accuracy by use and by mistake kind, recent scores and closed levels
"""

import json
import random
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import (
    DATA,
    PROGRESS,
    append_jsonl,
    drill,
    gender,
    prepositions,
    read_jsonl,
    sentences,
    vocab,
)

BANK: Path = DATA / "drills/articles.json"
ATTEMPTS: Path = PROGRESS / "articles.jsonl"
DRILL: str = "articles"
SIZE: int = 20
NEW: int = 12  # from the chosen level when earlier levels fill up the rest
PASS: int = 18  # of SIZE, twice in a row: the level is closed
VERSION: int = 2  # sessions of an older bank do not close a level
ROWS: list[list[str]] = [
    ["le", "la", "l'", "les"],
    ["un", "une", "des"],
    ["du", "de la", "de l'"],
    ["de", "d'"],
]
ROW_NAMES: list[str] = ["определённый", "неопределённый", "частичный", "отрицание, количество"]
FAMILY: dict[str, int] = {a: i for i, row in enumerate(ROWS) for a in row}
N: list[str]
P: list[str]
IND: list[str]
D: list[str]
D, IND, P, N = ROWS
LEVELS: dict[int, tuple[str, list[str]]] = {
    1: ("un или une", ["un", "une"]),
    2: ("le или la", ["le", "la"]),
    3: ("+ l', les", D),
    4: ("впервые → тот самый", D + ["un", "une"]),
    5: ("des или les", D + IND),
    6: ("вообще → le", D + IND),
    7: ("+ du, de la", D + IND + P),
    8: ("люблю le, ем du", D + IND + P),
    9: ("+ pas de", D + IND + P + N),
    10: ("исключения", D + IND + P + N),
}
FEM: set[str]
MASC: set[str]
MASC, FEM = {"le", "un", "du"}, {"la", "une", "de la"}
NO_VOWEL: set[str]
ELIDED: set[str]
ELIDED, NO_VOWEL = {"l'", "de l'", "d'"}, {"le", "la", "du", "de la", "de"}
NO_ELISION: set[str] = {"hamburger"}  # h aspiré: le hamburger
GENDER_PAIRS: list[set[str]] = [{"le", "la"}, {"un", "une"}, {"du", "de la"}]
FORMS: dict[str, str] = {
    "le": "le — перед словом мужского рода: le café.",
    "la": "la — перед словом женского рода: la porte.",
    "l'": "Перед гласной и немой h le и la → l': l'hiver, l'appartement.",
    "les": "Множественное число, оба рода — les: les chats, les clés.",
    "un": "un — мужской род: un chien.",
    "une": "une — женский род: une voiture.",
    "des": "Множественное число, оба рода — des: des amis, des livres.",
    "du": "de + le = du: du pain.",
    "de la": "Женский род — de la, без слияния: de la viande.",
    "de l'": "Перед гласной — de l': de l'eau, de l'aspirine.",
    "de": "После отрицания и количества — голое de: pas de pain, beaucoup de pain.",
    "d'": "Перед гласной de → d': pas d'amis, beaucoup d'amis.",
}


def panel(level: int) -> list[dict[str, Any]]:
    """The level's articles by row: [{row, keys}]."""
    keys: list[str] = LEVELS[level][1]
    return [
        {"row": ROW_NAMES[i], "keys": [a for a in row if a in keys]}
        for i, row in enumerate(ROWS)
        if set(row) & set(keys)
    ]


def nouns() -> dict[str, str]:
    """The bare noun of every article card → m or f."""
    out: dict[str, Any] = {}
    c: dict[str, Any]
    found: tuple[tuple[str, str], str, str] | None
    for c in vocab.load():
        if found := vocab.articles(c):
            out[found[2]] = "m" if found[1] in ("le", "un") else "f"
    return out


def problems(s: dict[str, Any], uses: dict[str, Any], known: set[str], genders: dict[str, str]) -> list[str]:
    """What is wrong with one sentence."""
    errors: list[Any]
    gap: Any
    fr: Any
    fr, gap, errors = s.get("fr", ""), s.get("gap", ""), []
    if not fr.strip() or not s.get("ru", "").strip():
        return [f"{s}: empty"]
    if gap not in FAMILY:
        return [f"{fr!r}: gap {gap!r} is not an article"]
    at: re.Match[str] | None = prepositions.gap_at(s)
    if at is None:
        return [f"{fr!r}: gap {gap!r} must stand exactly once"]
    if s.get("use") not in uses:
        errors.append(f"{fr!r}: unknown use {s.get('use')!r}")
    elif gap not in LEVELS[uses[s["use"]]["level"]][1]:
        errors.append(f"{fr!r}: {gap!r} is not on the panel of level {uses[s['use']]['level']}")
    after: list[str] = sentences.plain(fr[at.end() :])
    noun: str = after[0] if after else ""
    vowel: bool = bool(prepositions.VOWEL.match(noun)) and noun not in NO_ELISION
    if gap in ELIDED and not vowel or gap in NO_VOWEL and vowel:
        errors.append(f"{fr!r}: {gap!r} before {noun!r}: elision is wrong")
    if gap in MASC and genders.get(noun) == "f" or gap in FEM and genders.get(noun) == "m":
        errors.append(f"{fr!r}: {gap!r} before {noun!r} of the other gender")
    errors += [f"{fr!r}: {w!r} is not learned anywhere" for w in sentences.plain(fr) if w not in known]
    return errors


def made(
    data: dict[str, Any], uses: dict[str, Any], cards: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sentences from the noun cards by the frames of `made`, and the problems of the frames."""
    m: Any = data.get("made", {})
    found: tuple[tuple[str, str], str, str] | None
    nouns: dict[Any, tuple[dict[str, Any], tuple[tuple[str, str], str, str]]] = {
        c["label"]: (c, found) for c in cards if (found := vocab.articles(c))
    }
    errors: list[str] = [f"made: skip {w!r} is not a noun card" for w in m.get("skip", []) if w not in nouns]
    out: list[Any] = []
    f: Any
    for f in m.get("frames", []):
        level: Any = uses.get(f.get("use"), {}).get("level")
        if (
            level not in (1, 2, 3)
            or f.get("fr", "").count("___") != 1
            or "{ru}" not in f.get("ru", "").lower()
        ):
            errors.append(f"made: frame {f}: needs a use of level 1, 2 or 3, one ___ in fr and {{ru}} in ru")
            continue
        noun: str
        right: str
        _: tuple[str, str]
        c: dict[str, Any]
        label: Any
        for label, (c, (_, right, noun)) in nouns.items():
            vowel: bool = bool(prepositions.VOWEL.match(noun)) and noun not in NO_ELISION
            if label in m.get("skip", []) or (level == 3) != vowel:
                continue
            g: str = "m" if right in ("le", "un") else "f"
            art: str = {1: "un" if g == "m" else "une", 2: "le" if g == "m" else "la", 3: "l'"}[level]
            fr: Any = f["fr"].replace("___", f"{art}{'' if art.endswith(chr(39)) else ' '}{noun}")
            ru: Any = c["ru"].split(",")[0].strip()
            ru = f["ru"].replace("{ru}", ru).replace("{Ru}", ru[:1].upper() + ru[1:])
            out.append(
                {
                    "fr": fr[:1].upper() + fr[1:],
                    "gap": art,
                    "ru": ru,
                    "use": f["use"],
                    "made": True,
                    "full": noun,
                }
            )
    return out, errors


def load(path: Path = BANK) -> list[dict[str, Any]]:
    """The sentences, each with its `task`, `level` and `topic`; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    uses: Any = data.get("uses", {})
    genders: dict[str, str] = nouns()
    verbs: list[dict[str, Any]] = drill.load()
    known: set[Any | str] = (
        set(data.get("core", []))
        | sentences.met_words()
        | set(sentences.verb_forms(verbs))
        | {w for n in genders for w in sentences.plain(n)}
    )
    errors: list[str] = [
        f"use {k!r}: needs ru, rule and level 1 to {len(LEVELS)}"
        for k, u in uses.items()
        if u.get("level") not in LEVELS or not u.get("rule", "").strip() or not u.get("ru")
    ]
    if errors:
        raise ValueError("\n".join(errors))
    extra: list[dict[str, Any]]
    extra, errors = made(data, uses, vocab.load())
    out: list[Any]
    seen: set[Any]
    seen, out = set(), []
    s: Any
    for s in data.get("sentences", []) + extra:
        found: list[str] = problems(s, uses, known, genders)
        errors += found
        if found:
            continue
        at: re.Match[str] | None = prepositions.gap_at(s)
        assert at is not None  # problems() refused the rest
        after: Any
        before: Any
        before, after = s["fr"][: at.start()], s["fr"][at.end() :]
        task: str = f"{before}___{after}"
        if task in seen and s.get("made"):
            continue  # a written sentence says the same
        if task in seen:
            errors.append(f"{task!r} twice")
        seen.add(task)
        noun: Any | str = s.get("full") or (sentences.plain(after) or [""])[0]
        out.append(
            dict(
                s,
                task=task,
                before=before.strip(),
                after=after.strip(),
                noun=noun,
                level=uses[s["use"]]["level"],
                rule=uses[s["use"]]["rule"],
                topic=uses[s["use"]]["ru"],
            )
        )
    level: int
    for level in LEVELS:
        own: Any = sum(s["level"] == level and not s.get("made") for s in out)
        if not errors and (own < NEW or own + len(earlier(out, level)) < SIZE):
            errors.append(f"level {level}: {own} sentences, fewer than {NEW} of its own or {SIZE} in all")
    if errors:
        raise ValueError("\n".join(errors))
    return out


def gender_rules(cards: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """The bare noun → its rule on the gender tab."""
    return {g["noun"]: g["rule"] for g in gender.load(cards=cards)}


def grade(choice: str, s: dict[str, Any], rules_of: dict[str, str]) -> tuple[bool, str | None, list[str]]:
    """(ok, kind of mistake, rules)."""
    gap: Any = s["gap"]
    if choice == gap:
        return True, None, []
    if FAMILY[choice] == FAMILY[gap]:
        if {choice, gap} in GENDER_PAIRS:
            word: str = "мужского" if gap in MASC else "женского"
            told: list[str] = (
                [f"{s['noun']} — {word} рода.", rules_of[s["noun"]]]
                if s["noun"] in rules_of
                else [f"{s['noun']} — {word} рода."]
            )
            return False, "gender", told
        return False, "form", [FORMS[gap]]
    if FAMILY[gap] == 3 or FAMILY[choice] == 3:
        return False, "de", [s["rule"], FORMS[gap]]
    return False, "type", [s["rule"], FORMS[gap]]


def earlier(pool: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
    """Written sentences of the earlier levels whose article is on this level's panel."""
    return [s for s in pool if s["level"] < level and s["gap"] in LEVELS[level][1] and not s.get("made")]


def pick(
    pool: list[dict[str, Any]], history: list[dict[str, Any]], level: int, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """By weight: NEW sentences of the level, the rest from the earlier ones, mixed; all its own if none fit.
    Of its own, half are made from the cards when the level has made ones."""
    rng = rng or random.Random()
    before: list[dict[str, Any]] = earlier(pool, level)
    size: int = NEW if before else SIZE
    written: list[dict[str, Any]] = [s for s in pool if s["level"] == level and not s.get("made")]
    cards: list[dict[str, Any]] = [s for s in pool if s["level"] == level and s.get("made")]
    chosen: list[dict[str, Any]] = prepositions.pick(cards, history, size // 2, rng) if cards else []
    chosen += prepositions.pick(written, history, size - len(chosen), rng)
    chosen += prepositions.pick(before, history, SIZE - len(chosen), rng) if before else []
    rng.shuffle(chosen)
    return chosen


def levels(history: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Level → scores of its full sessions of the current bank, oldest first, and whether the last two
    reached PASS."""
    by: dict[str, list[dict[str, Any]]] = {}
    r: dict[str, Any]
    for r in history:
        if "level" in r and r.get("bank") == VERSION:
            by.setdefault(r["session"], []).append(r)
    out: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]]
    for rows in by.values():
        if len(rows) >= SIZE:
            out.setdefault(rows[0]["level"], {"scores": []})["scores"].append(sum(r["ok"] for r in rows))
    x: dict[str, Any]
    for x in out.values():
        x["closed"] = len(x["scores"]) >= 2 and min(x["scores"][-2:]) >= PASS
    return out


class Articles:
    def __init__(
        self, bank: list[dict[str, Any]], attempts: Path = ATTEMPTS, rules_of: dict[str, str] | None = None
    ) -> None:
        self.bank: list[dict[str, Any]] = bank
        self.attempts: Path = attempts
        self.rules_of: dict[str, str] = gender_rules() if rules_of is None else rules_of
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self, level: int = 1) -> dict[str, Any]:
        if level not in LEVELS:
            raise KeyError("bad level")
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        chosen: list[dict[str, Any]] = pick(self.bank, read_jsonl(self.attempts), level)
        with self.lock:
            self.sessions[sid] = {"level": level, "items": {str(i): s for i, s in enumerate(chosen)}}
        return {
            "session": sid,
            "level": level,
            "panel": panel(level),
            "items": [
                {"id": str(i), "ru": s["ru"], "before": s["before"], "after": s["after"]}
                for i, s in enumerate(chosen)
            ],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        sess: dict[str, Any] | None = self.sessions.get(str(body.get("session")))
        item: Any | None = sess["items"].get(str(body.get("id"))) if sess else None
        if sess is None or item is None:
            raise KeyError("unknown session or item")
        choice: str = str(body.get("choice"))
        if choice not in LEVELS[sess["level"]][1]:
            raise KeyError("bad choice")
        rules: list[str]
        kind: str | None
        ok: bool
        ok, kind, rules = grade(choice, item, self.rules_of)
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": DRILL,
            "session": body["session"],
            "level": sess["level"],
            "bank": VERSION,
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
            "expected": item["fr"],
            "use": item["topic"],
            "rules": rules or [item["rule"]],
        }

    def stats(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = read_jsonl(self.attempts)
        return prepositions.stats(history) | {"levels": levels(history)}


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = prepositions.stats(read_jsonl(path))
    if not s["answers"]:
        print("articles: no answers yet")
        return
    print(f"articles: {s['ok']} of {s['answers']} right in {s['sessions']} sessions")
    if s["kinds"]:
        print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    print("  " + ", ".join(f"{t} {x['ok']}/{x['tries']}" for t, x in s["uses"].items()))
    x: dict[str, Any]
    n: int
    for n, x in sorted(levels(read_jsonl(path)).items()):
        print(
            f"  level {n} {LEVELS[n][0]}: {' '.join(map(str, x['scores'][-5:]))}"
            f"{' — closed' if x['closed'] else ''}"
        )
