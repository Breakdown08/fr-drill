"""Test from the task bank in the browser: the whole form at once, as at the exam, then the review.

    fr-drill exam [--stage N] [--port 8771] [--no-open]   build a test, serve it, open the browser
    fr-drill exam last                                    the last test: score by part, mistakes
    fr-drill exam pending                                 letters (`write`) waiting for the grader, as JSON
    fr-drill exam grade FILE|-                            grade of a letter into exam-grades.jsonl

A test takes items of `data/bank/` up to the stage (the stage of an item is the highest stage
of its topics): up to PRACTICE practice items — half of them items answered wrong last time,
the rest unseen or least recently seen, spread over main topics — then one document with all
its questions and one letter. Options of `choice` are shuffled on every test.

Closed items are graded here, by `format.md`: `short` in practice counts a missing accent as
a mistake of kind `accent`, in `ce` it is accepted; `number` takes 19h, 19 h, 19:00, 12,50 €,
26 ans. A letter is counted in words (DELF: what spaces separate) and waits for the grader, who
grades it by the five DELF criteria, 0..3 each. Every answer is a row of `progress/exam.jsonl`.

GET  /         the page
GET  /test     new test: items without answers
POST /submit   {test, answers: {id: value}} → per item {ok, kind, expected, explain}

Bound to 127.0.0.1; `Host` must be 127.0.0.1|localhost:<port>, POST only application/json.
"""

import json
import random
import re
import secrets
import sys
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fr_drill import PROGRESS, append_jsonl, bank, read_jsonl
from fr_drill.drill import bare, norm

ATTEMPTS: Path = PROGRESS / "exam.jsonl"
GRADES: Path = PROGRESS / "exam-grades.jsonl"
PAGE: Path = Path(__file__).with_name("exam.html")
PRACTICE: int = 10
# DELF A1, production écrite, exercise 2: five criteria, 0..3 each, half points allowed.
CRITERIA: dict[str, str] = {
    "task": "Réalisation de la tâche — тип текста и тема по заданию, все пункты раскрыты",
    "coherence": "Cohérence et cohésion — начало и конец, порядок, связки et, alors",
    "sociolinguistic": "Adéquation sociolinguistique — приветствие, прощание, tu или vous",
    "lexicon": "Lexique — слова A1 по теме",
    "morphosyntax": "Morphosyntaxe — простые фразы, présent, согласование",
}
TIME: re.Pattern[str] = re.compile(r"^(\d{1,2})\s*(?:h|:)\s*(\d{2})?$")
NUMBER: re.Pattern[str] = re.compile(r"^\d+(?:[.,]\d+)?$")
UNITS: re.Pattern[str] = re.compile(r"\s*(€|euros?|ans?|heures?)$")


def stage_of(item: dict[str, Any], stages: dict[str, Any]) -> int:
    stage: int = max(stages.get(t, 0) for t in item["topics"])
    return stage


def parse_number(s: str) -> float | str | None:
    """26, '26 ans', '12,50 €' → number; '19h', '19 h 30', '19:00' → 'HH:MM'; else None."""
    s = " ".join(s.strip().lower().split())
    m: re.Match[str] | None
    if m := TIME.match(s):
        mm: int
        h: int
        h, mm = int(m[1]), int(m[2] or 0)
        return f"{h:02d}:{mm:02d}" if h < 24 and mm < 60 else None
    s = UNITS.sub("", s)
    if NUMBER.match(s):
        return float(s.replace(",", "."))
    return None


def expected(item: dict[str, Any]) -> str | None:
    """The right answer as shown in the review."""
    if item["type"] == "choice":
        option: str = item["options"][item["answer"]]
        return option
    if item["type"] == "short":
        accepted: str = item["accept"][0]
        return accepted
    if item["type"] == "number":
        a: Any = item["answer"]
        return a if isinstance(a, str) else f"{a:g}".replace(".", ",")
    return None


def words(text: str) -> int:
    """DELF rule: a word is what spaces separate; j'ai and rendez-vous are one word."""
    return len(text.split())


def grade(item: dict[str, Any], answer: object) -> tuple[bool | None, str | None]:
    """(ok, kind): ok None — a letter, graded by the grader unless empty. Kinds:
    empty, accent, wrong, format, pending, short."""
    kind: Any = item["type"]
    if kind == "write":
        n: int = words(str(answer or ""))
        if n == 0:
            return None, "empty"  # nothing to grade: not the grader's task, not a closed item either
        return None, "pending" if n >= item["min_words"] else "short"
    if answer is None or str(answer).strip() == "":
        return False, "empty"
    if kind == "choice":
        return (True, None) if answer == item["answer"] else (False, "wrong")
    if kind == "number":
        got: float | str | None = parse_number(str(answer))
        if got is None:
            return False, "format"
        want: Any = item["answer"]
        if isinstance(want, str):
            return (True, None) if got == want else (False, "wrong")
        return (True, None) if isinstance(got, float) and abs(got - want) < 1e-9 else (False, "wrong")
    # short: punctuation at the end is not part of the answer
    a: str = norm(str(answer)).rstrip(" .!?")
    accept: list[str] = [norm(x) for x in item["accept"]]
    if a in accept:
        return True, None
    if bare(a) in {bare(x) for x in accept}:
        return (True, None) if item["part"] == "ce" else (False, "accent")
    return False, "wrong"


def latest(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The last answer row of every item."""
    last: dict[str, Any] = {}
    row: dict[str, Any]
    for row in history:
        last[row["id"]] = row
    return last


def pick(
    items: list[dict[str, Any]],
    stages: dict[str, Any],
    history: list[dict[str, Any]],
    stage: int,
    size: int = PRACTICE,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Practice (wrong first, then unseen / oldest, spread over main topics), one doc group, one letter."""
    rng = rng or random.Random()
    last: dict[str, dict[str, Any]] = latest(history)
    pool: list[dict[str, Any]] = [i for i in items if stage_of(i, stages) <= stage]

    def age(i: dict[str, Any]) -> str:
        return last[i["id"]]["at"] if i["id"] in last else ""  # unseen sorts first

    practice: list[dict[str, Any]] = [i for i in pool if i["part"] == "practice"]
    wrong: list[dict[str, Any]] = [i for i in practice if last.get(i["id"], {}).get("ok") is False]
    chosen: list[dict[str, Any]] = rng.sample(wrong, min(len(wrong), size // 2))
    by_topic: defaultdict[str, Any] = defaultdict(list)
    i: dict[str, Any]
    for i in sorted((i for i in practice if i not in chosen), key=lambda i: (age(i), rng.random())):
        by_topic[i["topics"][0]].append(i)
    groups: list[list[dict[str, Any]]] = list(by_topic.values())
    rng.shuffle(groups)
    while len(chosen) < size and any(groups):
        g: list[dict[str, Any]]
        for g in groups:
            if g and len(chosen) < size:
                chosen.append(g.pop(0))

    test: list[dict[str, Any]] = chosen
    docs: defaultdict[str, Any] = defaultdict(list)
    for i in pool:
        if i["part"] == "ce":
            docs[i["doc"]].append(i)
    if docs:
        # the document read longest ago, items in the order of the text (bank order)
        doc: Any = min(docs, key=lambda d: (max(age(i) for i in docs[d]), rng.random()))
        test += docs[doc]
    letters: list[dict[str, Any]] = [i for i in pool if i["type"] == "write"]
    if letters:
        test.append(min(letters, key=lambda i: (age(i), rng.random())))
    return test


class Exam:
    def __init__(
        self,
        items: list[dict[str, Any]],
        docs: dict[str, Any],
        stages: dict[str, Any],
        consignes: dict[str, Any],
        stage: int,
        attempts: Path = ATTEMPTS,
    ) -> None:
        self.consignes: dict[str, Any]
        self.stages: dict[str, Any]
        self.docs: dict[str, Any]
        self.items: list[dict[str, Any]]
        self.items, self.docs, self.stages, self.consignes = items, docs, stages, consignes
        self.attempts: Path
        self.stage: int
        self.stage, self.attempts = stage, attempts
        self.tests: dict[str, dict[str, dict[str, Any]]] = {}
        self.lock: threading.Lock = threading.Lock()

    def public(self, item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        c: Any = item["consigne"]
        c = self.consignes[c] if isinstance(c, str) else c
        out: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "part": item["part"],
            "consigne": {"fr": c["fr"]}
            | ({"ru": c["ru"]} if stage_of(item, self.stages) <= bank.RU_UNTIL else {}),
        }
        k: str
        for k in ("prompt", "points", "min_words"):
            if k in item:
                out[k] = item[k]
        if "doc" in item:
            out["doc"] = self.docs[item["doc"]] | {}
        if item["type"] == "choice":
            order: list[int] = list(range(len(item["options"])))
            rng.shuffle(order)
            out["options"] = [{"i": i, "text": item["options"][i]} for i in order]
        return out

    def test(self) -> dict[str, Any]:
        tid: str = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-") + secrets.token_hex(2)
        rng: random.Random = random.Random()
        items: list[dict[str, Any]] = pick(
            self.items, self.stages, read_jsonl(self.attempts), self.stage, rng=rng
        )
        with self.lock:
            self.tests[tid] = {i["id"]: i for i in items}
        return {"test": tid, "stage": self.stage, "items": [self.public(i, rng) for i in items]}

    def submit(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            items: dict[str, dict[str, Any]] | None = self.tests.pop(str(body.get("test")), None)
        if items is None:
            raise KeyError("unknown or already submitted test")
        answers: dict[str, Any] = body["answers"] if isinstance(body.get("answers"), dict) else {}
        at: str = datetime.now().astimezone().isoformat(timespec="seconds")
        results: list[Any]
        rows: list[Any]
        rows, results = [], []
        item: dict[str, Any]
        iid: str
        for iid, item in items.items():
            answer: int | str | None = answers.get(iid)
            answer = (
                answer
                if isinstance(answer, int) and not isinstance(answer, bool)
                else (str(answer)[:3000] if answer is not None else None)
            )
            kind: str | None
            ok: bool | None
            ok, kind = grade(item, answer)
            row: dict[str, Any] = {
                "at": at,
                "test": body["test"],
                "stage": self.stage,
                "id": iid,
                "type": item["type"],
                "part": item["part"],
                "topics": item["topics"],
                "answer": answer,
                "ok": ok,
                "kind": kind,
            }
            if item["type"] == "write":
                row["words"] = words(str(answer or ""))
            rows.append(row)
            results.append(
                {
                    "id": iid,
                    "ok": ok,
                    "kind": kind,
                    "expected": expected(item),
                    "explain": item.get("explain"),
                    "words": row.get("words"),
                }
            )
        with self.lock:
            self.attempts.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.attempts, rows)
        closed: list[dict[str, bool | None | str | Any]] = [r for r in results if r["ok"] is not None]
        return {"score": sum(bool(r["ok"]) for r in closed), "of": len(closed), "items": results}


def handler(exam: Exam, port: int) -> type[BaseHTTPRequestHandler]:
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
            path: str = urlsplit(self.path).path
            if path == "/":
                self.send(200, PAGE.read_text(encoding="utf-8"), "text/html")
            elif path == "/test":
                self.send(200, exam.test())
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self.host_ok():
                return
            if urlsplit(self.path).path != "/submit":
                return self.send(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "application/json only"})
            size: int = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= 50_000:
                return self.send(413, {"error": "empty or too large"})
            try:
                body: Any = json.loads(self.rfile.read(size))
                self.send(200, exam.submit(body if isinstance(body, dict) else {}))
            except json.JSONDecodeError as e:
                self.send(400, {"error": f"bad json: {e}"})
            except KeyError as e:
                self.send(404, {"error": str(e)})

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def load_bank(
    root: Path = bank.BANK, topics: Path = bank.TOPICS
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(items, docs by id, stages, consignes); raise ValueError if `fr-drill bank check` fails."""
    errors: list[str]
    consignes: object
    stages: dict[str, Any]
    files: dict[str, Any]
    files, stages, consignes, errors = bank.load(root, topics)
    errors += bank.check(files, stages, consignes)
    if errors:
        raise ValueError("\n".join(errors))
    items: list[Any] = [i for c in files.values() for i in c["items"]]
    docs: dict[str, Any] = {d["id"]: d for c in files.values() for d in c.get("docs", [])}
    return items, docs, stages, consignes if isinstance(consignes, dict) else {}


def pending(attempts: Path = ATTEMPTS, grades: Path = GRADES, root: Path = bank.BANK) -> list[dict[str, Any]]:
    """Letters not graded yet, with what the grader needs to grade them."""
    done: set[tuple[Any, Any]] = {(g["test"], g["id"]) for g in read_jsonl(grades)}
    consignes: dict[str, Any]
    _: dict[str, Any]
    docs: dict[str, Any]
    items: list[dict[str, Any]]
    items, docs, _, consignes = load_bank(root)
    by_id: dict[Any, dict[str, Any]] = {i["id"]: i for i in items}
    out: list[Any] = []
    row: dict[str, Any]
    for row in read_jsonl(attempts):
        if (
            row["type"] != "write"
            or row["kind"] == "empty"
            or (row["test"], row["id"]) in done
            or row["id"] not in by_id
        ):
            continue
        item: dict[str, Any] = by_id[row["id"]]
        c: Any = item["consigne"]
        task: dict[str, Any] = {
            "test": row["test"],
            "id": row["id"],
            "consigne": consignes[c] if isinstance(c, str) else c,
            "prompt": item.get("prompt"),
            "points": item["points"],
            "min_words": item["min_words"],
            "answer": row["answer"] or "",
            "words": row["words"],
            "criteria": CRITERIA,
        }
        if "doc" in item:
            task["doc"] = docs[item["doc"]]["text"]
        out.append(task)
    return out


def accept_grade(reply: dict[str, Any], attempts: Path = ATTEMPTS, grades: Path = GRADES) -> dict[str, Any]:
    """Check grade and append it; raise ValueError naming every problem."""
    errors: list[Any] = []
    keys: set[tuple[Any, Any]] = {
        (r["test"], r["id"]) for r in read_jsonl(attempts) if r["type"] == "write" and r["kind"] != "empty"
    }
    key: tuple[Any | None, Any | None] = (reply.get("test"), reply.get("id"))
    if key not in keys:
        errors.append(f"no letter {key} in {attempts.name}")
    if key in {(g["test"], g["id"]) for g in read_jsonl(grades)}:
        errors.append(f"{key} is already graded")
    scores: Any = reply.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(CRITERIA):
        errors.append(f"scores must have exactly {sorted(CRITERIA)}")
    else:
        v: Any
        k: Any
        for k, v in scores.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 3 or (v * 2) % 1:
                errors.append(f"{k}: {v!r} — 0..3 in half points")
    if not isinstance(reply.get("comment"), str) or not reply["comment"].strip():
        errors.append("comment (in Russian) is required")
    fixes: Any = reply.get("fixes", [])
    if not isinstance(fixes, list) or not all(
        isinstance(f, dict) and {"was", "fix"} <= set(f) for f in fixes
    ):
        errors.append("fixes must be a list of {was, fix, why?}")
    if errors:
        raise ValueError("\n".join(errors))
    row: dict[str, Any] = {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test": reply["test"],
        "id": reply["id"],
        "scores": scores,
        "total": sum(scores.values()),
        "of": 3 * len(CRITERIA),
        "comment": reply["comment"].strip(),
        "fixes": fixes,
    }
    append_jsonl(grades, [row])
    return row


def last(attempts: Path = ATTEMPTS, grades: Path = GRADES) -> int:
    rows: list[dict[str, Any]] = read_jsonl(attempts)
    if not rows:
        print("no tests yet")
        return 0
    tid: Any = rows[-1]["test"]
    rows = [r for r in rows if r["test"] == tid]
    print(f"test {tid}, stage {rows[0]['stage']}")
    part: str
    for part in ("practice", "ce", "pe"):
        closed: list[dict[str, Any]] = [r for r in rows if r["part"] == part and r["ok"] is not None]
        if closed:
            print(f"  {part:<9} {sum(r['ok'] for r in closed)}/{len(closed)}")
    marks: dict[tuple[Any, Any], dict[str, Any]] = {(g["test"], g["id"]): g for g in read_jsonl(grades)}
    r: dict[str, Any]
    for r in rows:
        if r["type"] == "write":
            g: dict[str, Any] | None = marks.get((tid, r["id"]))
            state: str = (
                "пусто"
                if r["kind"] == "empty"
                else f"{g['total']:g}/{g['of']}"
                if g
                else "ждёт оценки — fr-drill exam pending"
            )
            print(f"  письмо {r['id']}: {r['words']} слов, {state}")
    wrong: list[dict[str, Any]] = [r for r in rows if r["ok"] is False]
    if wrong:
        print("mistakes:")
        for r in wrong:
            print(f"  {r['id']} ({r['topics'][0]}): {r['kind']} — {r['answer']!r}")
    return 0


def main(
    action: str = "run", arg: str | None = None, stage: int = 1, port: int = 8771, browse: bool = True
) -> int:
    if action == "last":
        return last()
    try:
        if action == "pending":
            print(json.dumps(pending(), ensure_ascii=False, indent=1))
            return 0
        if action == "grade":
            text: str | Any = (
                sys.stdin.read() if arg in (None, "-") else Path(arg).read_text(encoding="utf-8")
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            row: dict[str, Any] = accept_grade(json.loads(text))
            print(f"graded {row['id']}: {row['total']:g}/{row['of']}")
            return 0
        consignes: dict[str, Any]
        stages: dict[str, Any]
        docs: dict[str, Any]
        items: list[dict[str, Any]]
        items, docs, stages, consignes = load_bank()
    except (ValueError, json.JSONDecodeError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    exam: Exam = Exam(items, docs, stages, consignes, stage)
    server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", port), handler(exam, port))
    url: str = f"http://127.0.0.1:{port}/"
    n: int = sum(stage_of(i, stages) <= stage for i in items)
    print(f"fr-drill exam: {url} (stage ≤ {stage}, {n} items in the bank) — Ctrl+C to stop", flush=True)
    if browse:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
