"""Conjugation drill in the browser: présent of regular -er verbs; A1 words, numbers, negation,
être / avoir, sentences, à and de on other tabs.

    fr-drill drill [--port 8770] [--no-open]   serve the drill page and open it
    fr-drill drill --check                     validate the verb list, word cards, number rule, negation and
                                               être / avoir phrases, sentences, à and de, pronouns, gender,
                                               articles
    fr-drill drill --stats                     accuracy by person and by mistake kind; words by box; numbers
                                               by range

Forms are built by rule from `data/drills/verbs-er.json` (stem + ending; j' before a vowel
or mute h; nous mangeons / commençons), so no task is written by hand. A session picks
persons evenly and leans on the verbs and persons that went wrong before. Every answer is
checked here, in Python, and appended to `progress/drill.jsonl` at once.

GET  /           the page
GET  /session    new session: items without answers
POST /check      {session, id, answer, ms} → {ok, kind, expected}
GET  /stats      accuracy by person and kind
/vocab/*         the words tab, see `fr_drill.vocab`
/prepositions/*  the à and de tab, see `fr_drill.prepositions`
/numbers/*       the numbers tab, see `fr_drill.numbers`
/negation/*      the negation tab, see `fr_drill.negation`
/sentences/*     the sentences tab, see `fr_drill.sentences`
/etre-avoir/*    the irregular verbs tab, see `fr_drill.etre_avoir`
/pronouns/*      the pronouns tab, see `fr_drill.pronouns`
/gender/*        the gender tab, see `fr_drill.gender`
/articles/*      the articles tab, see `fr_drill.articles`

Bound to 127.0.0.1; `Host` must be 127.0.0.1|localhost:<port>, POST only application/json.
"""

import json
import random
import secrets
import threading
import unicodedata
import webbrowser
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qs, urlsplit

from fr_drill import (
    DATA,
    PROGRESS,
    append_jsonl,
    articles,
    etre_avoir,
    gender,
    negation,
    numbers,
    prepositions,
    pronouns,
    read_jsonl,
    sentences,
    vocab,
)

VERBS: Path = DATA / "drills/verbs-er.json"
ATTEMPTS: Path = PROGRESS / "drill.jsonl"
PAGE: Path = Path(__file__).with_name("drill.html")
DRILL: str = "present-er"
# Six forms; il/elle/on and ils/elles share one form, the pronoun is picked at random.
PERSONS: dict[str, tuple[list[str], str]] = {
    "je": (["je"], "e"),
    "tu": (["tu"], "es"),
    "il": (["il", "elle", "on"], "e"),
    "nous": (["nous"], "ons"),
    "vous": (["vous"], "ez"),
    "ils": (["ils", "elles"], "ent"),
}
SIZE: int = 20
HISTORY: int = 300  # answers looked at when choosing what to repeat
VOWELS: str = "aeiouyhâàéèêëîïôûù"  # every h in the list is mute
STEM_CHANGE: tuple[str, str, str, str, str, str, str, str] = (
    "acheter",
    "appeler",
    "jeter",
    "préférer",
    "espérer",
    "essayer",
    "payer",
    "envoyer",
)
APOSTROPHES: dict[int, str] = str.maketrans({"’": "'", "‘": "'", "`": "'", "ʼ": "'"})


def load(path: Path = VERBS) -> list[dict[str, Any]]:
    """The verb list; raise ValueError listing every problem."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    seen: set[Any]
    errors: list[Any]
    errors, seen = [], set()
    v: Any
    i: int
    for i, v in enumerate(data.get("verbs", [])):
        inf: Any = v.get("inf", "")
        if not inf.endswith("er") or len(inf) < 4 or inf == "aller":
            errors.append(f"{i}: {inf!r} is not a regular -er verb")
        if inf in STEM_CHANGE or inf.endswith(("eler", "eter", "yer")) or "é" in inf[-5:-2]:
            errors.append(f"{i}: {inf!r} changes its stem — another topic")
        if not v.get("ru"):
            errors.append(f"{i}: {inf!r} has no translation")
        if inf in seen:
            errors.append(f"{i}: {inf!r} twice")
        seen.add(inf)
    if not seen:
        errors.append("no verbs")
    if errors:
        raise ValueError("\n".join(errors))
    verbs: list[dict[str, Any]] = data["verbs"]
    return verbs


def form(inf: str, person: str) -> str:
    """The verb form alone: parle, mangeons, commençons."""
    ending: str
    stem: str
    stem, ending = inf[:-2], PERSONS[person][1]
    if ending.startswith("o"):
        if stem.endswith("g"):
            stem += "e"
        elif stem.endswith("c"):
            stem = stem[:-1] + "ç"
    return stem + ending


def phrase(inf: str, pronoun: str) -> str:
    """Pronoun + form: j'aime, nous parlons."""
    verb: str = form(inf, person_of(pronoun))
    if pronoun == "je" and verb[0] in VOWELS:
        return "j'" + verb
    return f"{pronoun} {verb}"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(APOSTROPHES).strip().lower()
    s = " ".join(s.split())
    return s.replace("' ", "'")


def bare(s: str) -> str:
    """Without accents: étudie → etudie, commençons → commencons."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def split(s: str) -> tuple[str, str]:
    """(pronoun, verb) of an answer: "j'aime" → ("je", "aime")."""
    if s.startswith("j'"):
        return "je", s[2:]
    tail: str
    _: str
    head: str
    head, _, tail = s.partition(" ")
    return head, tail


def grade(answer: str, inf: str, pronoun: str) -> tuple[bool, str | None]:
    """(ok, kind of mistake): elision, spelling, accent, pronoun, ending, other."""
    e: str
    a: str
    a, e = norm(answer), norm(phrase(inf, pronoun))
    if a == e:
        return True, None
    ev: str
    ep: str
    av: str
    ap: str
    (ap, av), (ep, ev) = split(a), split(e)
    if ap == ep and av == ev:  # same words, the apostrophe is off: je aime, j'parle
        return False, "elision"
    if ev.endswith(("geons", "çons")) and bare(av) == bare(inf[:-2]) + "ons":  # mangons, commencons
        return False, "spelling"
    if bare(a) == bare(e):
        return False, "accent"
    if ap != ep:
        return False, "pronoun"
    if bare(av).startswith(bare(inf[:-2])):
        return False, "ending"
    return False, "other"


def person_of(pronoun: str) -> str:
    return next(p for p, (prons, _) in PERSONS.items() if pronoun in prons)


def pick(
    verbs: list[dict[str, Any]],
    history: list[dict[str, Any]],
    size: int = SIZE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Persons evenly, extra turns to the weakest; within a person, verbs that went wrong first."""
    rng = rng or random.Random()
    recent: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL][-HISTORY:]
    misses: Counter[tuple[Any, Any]] = Counter((r["verb"], r["person"]) for r in recent if not r["ok"])
    tries: Counter[Any] = Counter(r["person"] for r in recent)
    wrong: Counter[Any] = Counter(r["person"] for r in recent if not r["ok"])
    persons: list[str] = list(PERSONS)
    slots: list[str] = persons * (size // len(persons))
    weakest: list[str] = sorted(
        persons, key=lambda p: (-(wrong[p] / tries[p]) if tries[p] else -1, rng.random())
    )
    slots += weakest[: size - len(slots)]
    seen: Counter[Any] = Counter(r["verb"] for r in recent)
    used: set[Any]
    items: list[Any]
    items, used = [], set()
    person: str
    for person in slots:
        weights: list[float | int] = [
            (1 + 4 * misses[(v["inf"], person)] + (2 if not seen[v["inf"]] else 0))
            * (0.1 if v["inf"] in used else 1)
            for v in verbs
        ]
        v: dict[str, Any] = rng.choices(verbs, weights)[0]
        used.add(v["inf"])
        pronoun: str = rng.choice(PERSONS[person][0])
        items.append(
            {"id": str(len(items)), "verb": v["inf"], "ru": v["ru"], "pronoun": pronoun, "person": person}
        )
    rng.shuffle(items)
    item: dict[str, Any]
    i: int
    for i, item in enumerate(items):
        item["id"] = str(i)
    return items


def stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [r for r in history if r.get("drill") == DRILL]
    by: dict[str, Any] = {}
    p: str
    for p in PERSONS:
        mine: list[dict[str, Any]] = [r for r in rows if r["person"] == p]
        by[p] = {"tries": len(mine), "ok": sum(r["ok"] for r in mine)}
    kinds: Counter[Any] = Counter(r["kind"] for r in rows if not r["ok"])
    verbs: Counter[Any] = Counter(r["verb"] for r in rows if not r["ok"])
    return {
        "answers": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "sessions": len({r["session"] for r in rows}),
        "persons": by,
        "kinds": dict(kinds.most_common()),
        "hard_verbs": [v for v, _ in verbs.most_common(5)],
    }


class Drill:
    def __init__(self, verbs: list[dict[str, Any]], attempts: Path = ATTEMPTS) -> None:
        self.verbs: list[dict[str, Any]] = verbs
        self.attempts: Path = attempts
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def session(self, size: int = SIZE) -> dict[str, Any]:
        sid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        items: list[dict[str, Any]] = pick(self.verbs, read_jsonl(self.attempts), size)
        with self.lock:
            self.sessions[sid] = {i["id"]: i for i in items}
        public: tuple[str, str, str, str] = ("id", "verb", "ru", "pronoun")
        return {"session": sid, "items": [{k: i[k] for k in public} for i in items]}

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] | None = self.sessions.get(str(body.get("session")), {}).get(str(body.get("id")))
        if item is None:
            raise KeyError("unknown session or item")
        answer: str = str(body.get("answer", ""))[:100]
        expected: str = phrase(item["verb"], item["pronoun"])
        kind: str | None
        ok: bool
        ok, kind = grade(answer, item["verb"], item["pronoun"])
        row: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "drill": DRILL,
            "session": body["session"],
            "verb": item["verb"],
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


def handler(
    drill: Drill,
    port: int,
    words: vocab.Vocab,
    nums: numbers.Numbers,
    neg: negation.Negation,
    aux: etre_avoir.EtreAvoir,
    phrases: sentences.Sentences,
    preps: prepositions.Prepositions,
    prons: pronouns.Pronouns | None = None,
    gen: gender.Gender | None = None,
    arts: articles.Articles | None = None,
) -> type[BaseHTTPRequestHandler]:
    # tabs added later: tests of older tabs need not pass them
    pronoun_tab: pronouns.Pronouns = prons or pronouns.Pronouns(pronouns.load())
    gender_tab: gender.Gender = gen or gender.Gender(gender.load(cards=words.cards))
    article_tab: articles.Articles = arts or articles.Articles(articles.load())
    hosts: set[str] = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: dict[str, Any] | str, ctype: str = "application/json") -> None:
            data: bytes = (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode()
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def host_ok(self) -> bool:
            if self.headers.get("Host") in hosts:
                return True
            self.send(403, {"error": "bad host"})
            return False

        def do_GET(self) -> None:
            if not self.host_ok():
                return
            url: SplitResult = urlsplit(self.path)
            if url.path == "/":
                self.send(200, PAGE.read_text(encoding="utf-8"), "text/html")
            elif url.path == "/session":
                size: str = parse_qs(url.query).get("n", [str(SIZE)])[0]
                self.send(200, drill.session(max(1, min(60, int(size))) if size.isdigit() else SIZE))
            elif url.path == "/stats":
                self.send(200, stats(read_jsonl(drill.attempts)))
            elif url.path == "/vocab/session":
                self.send(200, words.session())
            elif url.path == "/vocab/stats":
                self.send(200, words.stats())
            elif url.path == "/numbers/session":
                self.send(200, nums.session(parse_qs(url.query).get("voice") == ["1"]))
            elif url.path == "/numbers/stats":
                self.send(200, nums.stats())
            elif url.path == "/negation/session":
                self.send(200, neg.session())
            elif url.path == "/negation/stats":
                self.send(200, neg.stats())
            elif url.path == "/etre-avoir/session":
                try:
                    self.send(
                        200,
                        aux.session(
                            parse_qs(url.query).get("verb"), parse_qs(url.query).get("mode", ["verb"])[0]
                        ),
                    )
                except ValueError as e:
                    self.send(400, {"error": str(e)})
            elif url.path == "/etre-avoir/stats":
                self.send(200, aux.stats())
            elif url.path == "/sentences/session":
                self.send(200, phrases.session())
            elif url.path == "/sentences/stats":
                self.send(200, phrases.stats())
            elif url.path == "/sentences/gaps/session":
                self.send(200, phrases.gap_session())
            elif url.path == "/sentences/gaps/stats":
                self.send(200, phrases.gap_stats())
            elif url.path == "/prepositions/session":
                self.send(200, preps.session())
            elif url.path == "/prepositions/verbs/session":
                self.send(200, preps.verb_session())
            elif url.path == "/prepositions/stats":
                self.send(200, preps.stats())
            elif url.path == "/pronouns/session":
                self.send(200, pronoun_tab.session())
            elif url.path == "/pronouns/stats":
                self.send(200, pronoun_tab.stats())
            elif url.path == "/gender/session":
                self.send(200, gender_tab.session())
            elif url.path == "/gender/stats":
                self.send(200, gender_tab.stats())
            elif url.path == "/articles/session":
                level: str = parse_qs(url.query).get("level", ["1"])[0]
                if level not in {str(n) for n in articles.LEVELS}:
                    self.send(404, {"error": "bad level"})
                else:
                    self.send(200, article_tab.session(int(level)))
            elif url.path == "/articles/stats":
                self.send(200, article_tab.stats())
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self.host_ok():
                return
            path: str = urlsplit(self.path).path
            checkers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
                "/check": drill.check,
                "/vocab/check": words.check,
                "/numbers/check": nums.check,
                "/negation/check": neg.check,
                "/etre-avoir/check": aux.check,
                "/sentences/check": phrases.check,
                "/sentences/gaps/check": phrases.gap_check,
                "/prepositions/check": preps.check,
                "/pronouns/check": pronoun_tab.check,
                "/gender/check": gender_tab.check,
                "/articles/check": article_tab.check,
            }
            if path not in checkers:
                return self.send(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "application/json only"})
            size: int = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= 10_000:
                return self.send(413, {"error": "empty or too large"})
            try:
                body: Any = json.loads(self.rfile.read(size))
                self.send(200, checkers[path](body if isinstance(body, dict) else {}))
            except json.JSONDecodeError as e:
                self.send(400, {"error": f"bad json: {e}"})
            except KeyError as e:
                self.send(404, {"error": str(e)})

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def show_stats(path: Path = ATTEMPTS) -> int:
    s: dict[str, Any] = stats(read_jsonl(path))
    if not s["answers"]:
        print("no drill answers yet")
        return 0
    print(f"{s['answers']} answers in {s['sessions']} sessions, {100 * s['ok'] // s['answers']}% right")
    x: Any
    p: Any
    for p, x in s["persons"].items():
        if x["tries"]:
            print(f"  {p:<5} {x['ok']}/{x['tries']}")
    if s["kinds"]:
        print("mistakes: " + ", ".join(f"{k} {n}" for k, n in s["kinds"].items()))
    if s["hard_verbs"]:
        print("hardest verbs: " + ", ".join(s["hard_verbs"]))
    return 0


def main(port: int = 8770, check: bool = False, show: bool = False, browse: bool = True) -> int:
    if not vocab.WORDS.exists():
        print(
            f"{vocab.WORDS}: missing — put your vocabulary file there or set FR_DRILL_VOCAB (format: README)"
        )
        return 1
    try:
        verbs: list[dict[str, Any]] = load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"verbs-er.json: invalid\n{e}")
        return 1
    try:
        cards: list[dict[str, Any]] = vocab.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"{vocab.WORDS.name}: invalid\n{e}")
        return 1
    try:
        base: list[str] = numbers.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"numbers: invalid\n{e}")
        return 1
    try:
        phrases: list[dict[str, Any]] = negation.load(verbs=verbs)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"negation-er.json: invalid\n{e}")
        return 1
    try:
        aux: list[dict[str, Any]] = etre_avoir.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"etre-avoir.json: invalid\n{e}")
        return 1
    try:
        built: list[dict[str, Any]] = sentences.load(verbs=verbs, cards=cards)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"sentences.json: invalid\n{e}")
        return 1
    try:
        a_de: list[dict[str, Any]] = prepositions.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"a-de.json: invalid\n{e}")
        return 1
    try:
        a_de_verbs: list[dict[str, Any]] = prepositions.load_verbs()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"a-de-verbs.json: invalid\n{e}")
        return 1
    try:
        prons: list[dict[str, Any]] = pronouns.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"pronouns.json: invalid\n{e}")
        return 1
    try:
        gen: list[dict[str, Any]] = gender.load(cards=cards)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"gender.json: invalid\n{e}")
        return 1
    try:
        arts: list[dict[str, Any]] = articles.load()
    except (ValueError, json.JSONDecodeError) as e:
        print(f"articles.json: invalid\n{e}")
        return 1
    if check:
        print(f"verbs-er.json: ok, {len(verbs)} verbs")
        print(f"{vocab.WORDS.name}: ok, {len(cards)} cards")
        print(f"numbers: ok, {len(base)} base words built by the rule")
        print(f"negation-er.json: ok, {len(phrases)} phrases")
        print(f"etre-avoir.json: ok, {len(aux)} phrases")
        print(f"sentences.json: ok, {len(built)} sentences")
        print(f"a-de.json: ok, {len(a_de)} sentences")
        print(f"a-de-verbs.json: ok, {len(a_de_verbs)} verbs")
        print(f"pronouns.json: ok, {len(prons)} sentences")
        print(f"gender.json: ok, {len(gen)} nouns")
        print(f"articles.json: ok, {len(arts)} sentences")
        return 0
    if show:
        show_stats()
        vocab.show_stats(cards)
        numbers.show_stats()
        negation.show_stats()
        etre_avoir.show_stats()
        sentences.show_stats(built)
        prepositions.show_stats()
        pronouns.show_stats()
        gender.show_stats()
        articles.show_stats()
        return 0
    server: ThreadingHTTPServer = ThreadingHTTPServer(
        ("127.0.0.1", port),
        handler(
            Drill(verbs),
            port,
            vocab.Vocab(cards),
            numbers.Numbers(),
            negation.Negation(phrases),
            etre_avoir.EtreAvoir(aux),
            sentences.Sentences(built),
            prepositions.Prepositions(a_de, verbs=a_de_verbs),
            pronouns.Pronouns(prons),
            gender.Gender(gen),
            articles.Articles(arts, rules_of={g["noun"]: g["rule"] for g in gen}),
        ),
    )
    url: str = f"http://127.0.0.1:{port}/"
    print(f"fr-drill drill: {url} ({len(verbs)} verbs, {len(cards)} words) — Ctrl+C to stop", flush=True)
    if browse:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
