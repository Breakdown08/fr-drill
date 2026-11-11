"""Numbers, the third tab of `fr-drill drill`: French number words by rule, 0..9999.

Words are built in Python (`to_fr`), traditional spelling: dix-sept, vingt et un, soixante et onze,
quatre-vingts, quatre-vingt-un, deux cents, deux cent un, mille, deux mille. The 30 base words of
lesson 4 in the vocabulary file (`VOCAB`) must come out of the rule as they are written there.

Three kinds of question: `read` — French words, type the digits; `listen` — the browser speaks the
number (only if the page found a French voice), type the digits; `pick` — digits, pick the French
words out of four look-alikes (97 → quatre-vingt-dix-sept, soixante-dix-sept, quatre-vingt-sept…).

Numbers fall into ranges (CATS); a range opens by stage. Stage 1 is 0..100; the next stage opens
when every range of the current one is held: its last HELD_OF answers have at least HELD_OK right,
over at least two days. Retries of a missed question within a session don't count. A session leans
on ranges that went wrong and brings back numbers whose last answer was wrong.

GET  /numbers/session?voice=1   new session
POST /numbers/check             {session, id, answer, ms, again} → {ok, kind, expected, fr}
GET  /numbers/stats             ranges, stage, mistake kinds
"""

import json
import random
import secrets
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import DATA, PROGRESS, append_jsonl, read_jsonl

WORDS: Path = DATA / "drills/numbers.json"
ATTEMPTS: Path = PROGRESS / "numbers.jsonl"
SIZE: int = 20
OPTIONS: int = 4
HISTORY: int = 300
HELD_OK: int
HELD_OF: int
HELD_OF, HELD_OK = 20, 18
MISSED_MAX: int = 5  # numbers missed last time, brought back per session
# (name, low, high, stage)
CATS: list[tuple[str, int, int, int]] = [
    ("0–16", 0, 16, 1),
    ("17–69", 17, 69, 1),
    ("70–79", 70, 79, 1),
    ("80–100", 80, 100, 1),
    ("101–999", 101, 999, 2),
    ("1000–9999", 1000, 9999, 3),
]
STAGES: int = max(c[3] for c in CATS)
UNITS: list[str] = [
    "zéro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
]
TENS: dict[int, str] = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}
SOUND_ALIKE: dict[int, int] = {
    2: 12,
    12: 2,
    3: 13,
    13: 3,
    4: 14,
    14: 4,
    5: 15,
    15: 5,
    6: 16,
    16: 6,
    10: 6,
    11: 1,
}
MODES: dict[bool, list[str]] = {
    True: ["listen", "listen", "read", "pick", "pick"],
    False: ["read", "read", "pick"],
}
KINDS: tuple[str, str, str, str, str] = ("tens", "units", "hundreds", "pieces", "other")


def below100(n: int) -> str:
    if n <= 16:
        return UNITS[n]
    if n < 20:
        return "dix-" + UNITS[n - 10]
    if n < 70:
        unit: int
        tens: int
        tens, unit = divmod(n, 10)
        if unit == 0:
            return TENS[tens]
        return TENS[tens] + (" et un" if unit == 1 else "-" + UNITS[unit])
    if n < 80:
        return "soixante et onze" if n == 71 else "soixante-" + below100(n - 60)
    if n == 80:
        return "quatre-vingts"
    return "quatre-vingt-" + below100(n - 80)  # no "et": quatre-vingt-un, quatre-vingt-onze


def below1000(n: int) -> str:
    rest: int
    hundreds: int
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return below100(rest)
    head: str = "cent" if hundreds == 1 else UNITS[hundreds] + " cent"
    if not rest:
        return head + ("s" if hundreds > 1 else "")  # deux cents, but deux cent un
    return f"{head} {below100(rest)}"


def to_fr(n: int) -> str:
    """French words for 0..9999, traditional spelling: hyphens below a hundred only."""
    if not 0 <= n <= 9999:
        raise ValueError(f"{n} is out of 0..9999")
    rest: int
    thousands: int
    thousands, rest = divmod(n, 1000)
    if not thousands:
        return below1000(rest)
    head: str = "mille" if thousands == 1 else UNITS[thousands] + " mille"  # mille never takes -s
    return head if not rest else f"{head} {below1000(rest)}"


def load(path: Path = WORDS) -> list[str]:
    """Base words of the numbers file; raise ValueError if the rule doesn't give one of them."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    known: set[str] = {to_fr(n) for n in range(1001)}
    words: list[str] = [w["fr"] for w in data.get("words", [])]
    off: list[str] = [w for w in words if w not in known]
    if not words or off:
        raise ValueError("not built by the rule: " + ", ".join(off) if off else "no words")
    return words


def cat_of(n: int) -> str:
    return next(name for name, lo, hi, _ in CATS if lo <= n <= hi)


def counted(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Answers that count: retries after a miss within a session are left out."""
    return [r for r in history if not r.get("again")]


def held(rows: list[dict[str, Any]]) -> bool:
    last: list[dict[str, Any]] = rows[-HELD_OF:]
    return (
        len(last) == HELD_OF
        and sum(r["ok"] for r in last) >= HELD_OK
        and len({r["at"][:10] for r in last}) >= 2
    )


def stage(history: list[dict[str, Any]]) -> int:
    """Highest open stage: the next opens when every range of this one is held."""
    rows: list[dict[str, Any]] = counted(history)
    s: int = 1
    while s < STAGES and all(held([r for r in rows if r["cat"] == name]) for name, *_, st in CATS if st == s):
        s += 1
    return s


def sample(cat: tuple[str, int, int, int], rng: random.Random) -> int:
    _: int
    hi: int
    lo: int
    name: str
    name, lo, hi, _ = cat
    if name == "101–999" and rng.random() < 0.3:  # cents rules: deux cents, deux cent un
        return rng.randrange(2, 10) * 100 + rng.choice([0, 0, 1, 80])
    if name == "1000–9999" and rng.random() < 0.5:  # years
        return rng.randint(1900, 2099)
    return rng.randint(lo, hi)


def pick(history: list[dict[str, Any]], size: int = SIZE, rng: random.Random | None = None) -> list[int]:
    """Numbers for a session: missed ones first, then ranges weighted by recent misses."""
    rng = rng or random.Random()
    rows: list[dict[str, Any]] = counted(history)[-HISTORY:]
    top: int = stage(history)
    cats: list[tuple[str, int, int, int]] = [c for c in CATS if c[3] <= top]
    last: dict[int, bool] = {}
    r: dict[str, Any]
    for r in rows:
        last[r["n"]] = r["ok"]
    hi: int = max(c[2] for c in cats)
    missed: list[int] = [n for n, ok in last.items() if not ok and n <= hi]
    out: list[int] = rng.sample(missed, min(MISSED_MAX, len(missed)))
    weights: list[Any] = []
    c: tuple[str, int, int, int]
    for c in cats:
        mine: list[dict[str, Any]] = [r for r in rows if r["cat"] == c[0]][-HELD_OF:]
        miss: float | int = sum(not r["ok"] for r in mine) / len(mine) if mine else 0
        weights.append(1 + 4 * miss + (2 if len(mine) < 10 else 0) + (2 if c[3] == top and top > 1 else 0))
    while len(out) < size:
        n: int = sample(rng.choices(cats, weights)[0], rng)
        if n not in out or len(out) >= hi:
            out.append(n)
    rng.shuffle(out)
    return out


def lookalikes(n: int, hi: int, rng: random.Random) -> list[int]:
    """OPTIONS − 1 numbers easy to mix up with n: other ten, ±1, sound-alikes, other hundred."""
    near: set[int] = {n + 1, n - 1, n + 10, n - 10, n + 20, n - 20, SOUND_ALIKE.get(n, -1), n + 100, n - 100}
    if 10 <= n < 100:
        near.add(n % 10 * 10 + n // 10)
    if n >= 1000:
        near |= {n + 1000, n - 1000, n - n % 100 + (n % 100 + 10) % 100}
    ordered: list[int] = sorted(c for c in near if 0 <= c <= hi and c != n)
    out: list[int] = rng.sample(ordered, min(OPTIONS - 1, len(ordered)))
    while len(out) < OPTIONS - 1:
        c: int = rng.randint(max(0, n - 30), min(hi, n + 30))
        if c != n and c not in out:
            out.append(c)
    return out


def parse(answer: str) -> int | None:
    digits: str = "".join(answer.split()).replace(".", "").replace(" ", "")
    return int(digits) if digits.isdigit() and len(digits) <= 9 else None


def kind(expected: int, got: int | None) -> str:
    """Mistake kind: tens (97 → 77), units (16 → 15), hundreds, pieces (97 → 42017), other."""
    if got is None:
        return "other"
    if len(str(got)) > len(str(expected)):
        return "pieces"
    if got % 10 == expected % 10 and got // 100 == expected // 100:
        return "tens"
    if got // 10 == expected // 10:
        return "units"
    if expected >= 100 and got % 100 == expected % 100:
        return "hundreds"
    return "other"


def questions(
    nums: list[int], hi: int, voice: bool, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    rng = rng or random.Random()
    out: list[Any] = []
    n: int
    for n in nums:
        mode: str = rng.choice(MODES[voice])
        item: dict[str, object] = {
            "id": str(len(out)),
            "n": n,
            "cat": cat_of(n),
            "mode": mode,
            "fr": to_fr(n),
        }
        if mode == "pick":
            opts: list[int] = lookalikes(n, hi, rng) + [n]
            rng.shuffle(opts)
            item |= {"prompt": str(n), "options": [to_fr(o) for o in opts], "values": opts}
        else:
            item["prompt"] = to_fr(n)
        out.append(item)
    return out


def stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = counted(history)
    cats: dict[str, Any] = {}
    st: int
    _: list[Any]
    name: str
    for name, *_, st in CATS:
        mine: list[dict[str, Any]] = [r for r in rows if r["cat"] == name]
        last: list[dict[str, Any]] = mine[-HELD_OF:]
        cats[name] = {
            "stage": st,
            "tries": len(mine),
            "ok": sum(r["ok"] for r in mine),
            "last": len(last),
            "last_ok": sum(r["ok"] for r in last),
            "held": held(mine),
        }
    modes: dict[str, dict[str, Any]] = {
        m: {"tries": sum(r["mode"] == m for r in rows), "ok": sum(r["ok"] for r in rows if r["mode"] == m)}
        for m in ("read", "listen", "pick")
    }
    return {
        "answers": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "stage": stage(history),
        "stages": STAGES,
        "cats": cats,
        "modes": modes,
        "kinds": dict(Counter(r["kind"] for r in rows if not r["ok"]).most_common()),
        "hard": [n for n, _ in Counter(r["n"] for r in rows if not r["ok"]).most_common(8)],
    }


class Numbers:
    def __init__(self, attempts: Path = ATTEMPTS) -> None:
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self, voice: bool = False) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        history: list[dict[str, Any]] = read_jsonl(self.attempts)
        top: int = stage(history)
        hi: int = max(c[2] for c in CATS if c[3] <= top)
        items: list[dict[str, Any]] = questions(pick(history), hi, voice)
        with self.lock:
            self.sessions[sid] = {i["id"]: i for i in items}
        public: tuple[str, str, str, str, str] = ("id", "cat", "mode", "prompt", "options")
        return {
            "session": sid,
            "stage": top,
            "range": f"0–{hi}",
            "items": [{k: i[k] for k in public if k in i} for i in items],
        }

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        if item["mode"] == "pick":
            choice: Any | None = body.get("answer")
            if not isinstance(choice, int) or not 0 <= choice < len(item["values"]):
                raise KeyError("bad choice")
            answer: Any
            got: Any
            got, answer = item["values"][choice], item["options"][choice]
        else:
            answer = str(body.get("answer", ""))[:40]
            got = parse(answer)
        ok: Any = got == item["n"]
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session": body["session"],
            "n": item["n"],
            "cat": item["cat"],
            "mode": item["mode"],
            "answer": answer,
            "got": got,
            "ok": ok,
            "kind": None if ok else kind(item["n"], got),
            "ms": body.get("ms") if isinstance(body.get("ms"), int) else None,
            "again": body.get("again") is True,
        }
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, [row])
        out: dict[str, Any] = {"ok": ok, "kind": row["kind"], "expected": item["n"], "fr": item["fr"]}
        if item["mode"] == "pick":
            out["answer"] = item["values"].index(item["n"])
        return out

    def stats(self) -> dict[str, Any]:
        return stats(read_jsonl(self.attempts))


def show_stats(path: Path = ATTEMPTS) -> None:
    s: dict[str, Any] = stats(read_jsonl(path))
    if not s["answers"]:
        print("numbers: no answers yet")
        return
    print(
        f"numbers: {s['answers']} answers, {100 * s['ok'] // s['answers']}% right, "
        f"stage {s['stage']}/{s['stages']}"
    )
    c: Any
    name: Any
    for name, c in s["cats"].items():
        if c["tries"]:
            mark: str = " held" if c["held"] else ""
            print(f"  {name:<10} {c['ok']}/{c['tries']}, last {c['last_ok']}/{c['last']}{mark}")
    print("  " + ", ".join(f"{m} {x['ok']}/{x['tries']}" for m, x in s["modes"].items() if x["tries"]))
    if s["kinds"]:
        print("  mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    if s["hard"]:
        print("  hardest: " + ", ".join(map(str, s["hard"])))
