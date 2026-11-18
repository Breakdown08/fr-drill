"""Answer eval: item set, grading, run files, comparison. Standard library only."""

import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fr_drill import DATA, ROOT

ANSWERS: Path = DATA / "rag" / "answers.json"
RUNS: Path = ROOT / "runs"
NO_ANSWER: str = "В найденных фрагментах учебника ответа нет."
SCHEMA: int = 1
MAX_PATTERN_WORDS: int = 4
# The core of NO_ANSWER: small models drop "найденных" and still mean a refusal.
REFUSAL_CORE: str = "фрагментах учебника ответа нет"


class AnswerError(Exception):
    """A bad item set or a bad run file."""


@dataclass(frozen=True)
class AnswerItem:
    id: str
    q: str
    expect: str
    chapter: str | None
    pages: list[str]
    must: list[list[str]]
    forbid: list[str]


@dataclass(frozen=True)
class Grade:
    score: float
    passed: bool
    facts_hit: int
    facts_total: int
    missed: list[int]
    forbidden: list[str]
    refused: bool
    cited: list[str]
    cite_in_context: float | None
    cite_expected: bool


@dataclass(frozen=True)
class AnswerRow:
    id: str
    q: str
    expect: str
    retrieved: list[str]
    answer: str
    retrieve_s: float
    generate_s: float
    stats: dict[str, float]
    grade: Grade


@dataclass(frozen=True)
class Summary:
    n_answer: int
    n_no_answer: int
    answer_score: float | None
    pass_rate: float | None
    wrong_rate: float | None
    false_refusal_rate: float | None
    no_answer_accuracy: float | None
    cited_rate: float | None
    cite_precision: float | None
    cite_expected_rate: float | None
    generate_median_s: float
    generate_mean_s: float
    generate_max_s: float
    generate_total_s: float
    retrieve_total_s: float


@dataclass(frozen=True)
class Run:
    schema: int
    created: str
    backend: str
    model: str
    options: dict[str, float]
    k: int
    index_points: int
    items_file: str
    items_sha256: str
    platform: str
    cpus: int | None
    summary: Summary
    rows: list[AnswerRow]


@dataclass(frozen=True)
class Comparison:
    n: int
    mean_a: float
    mean_b: float
    diff: float
    sd: float
    ci_low: float
    ci_high: float
    better: int
    worse: int
    same: int


_QUOTES: str = "\u2018\u2019\u02bc\u00b4`"
_PUNCT: str = '*_\u00ab\u00bb"\u201c\u201d'
_CITE: re.Pattern[str] = re.compile(r"\(\s*[\u0441c]\.\s*([^)]*)\)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    out: str = unicodedata.normalize("NFC", text)
    ch: str
    for ch in _QUOTES:
        out = out.replace(ch, "'")
    code: int
    for code in range(0x2010, 0x2014):
        out = out.replace(chr(code), "-")
    for ch in _PUNCT:
        out = out.replace(ch, " ")
    out = out.casefold().replace("\u0451", "\u0435")
    return re.sub(r"\s+", " ", out).strip()


def matches(pattern: str, text: str) -> bool:
    stem: bool = pattern.endswith("*")
    core: str = normalize_text(pattern[:-1] if stem else pattern)
    if not core:
        raise ValueError(f"empty pattern: {pattern!r}")
    left: str = r"(?<!\w)" if re.match(r"\w", core[0]) else ""
    right: str = r"(?!\w)" if not stem and re.match(r"\w", core[-1]) else ""
    return re.search(left + re.escape(core) + right, normalize_text(text)) is not None


def cited_pages(text: str) -> list[str]:
    pages: list[str] = []
    m: re.Match[str]
    for m in _CITE.finditer(text):
        n: str
        for n in re.findall(r"\d+", m.group(1)):
            if n not in pages:
                pages.append(n)
    return pages


def is_refusal(text: str) -> bool:
    return normalize_text(REFUSAL_CORE) in normalize_text(text)


def _check_pattern(p: object, where: str) -> None:
    if not isinstance(p, str) or not p.strip():
        raise AnswerError(f"{where}: a pattern must be a non-empty string")
    if p == "*" or "*" in p[:-1]:
        raise AnswerError(f"{where}: '*' is allowed only as the last character, not alone: {p!r}")
    if len(p.split()) > MAX_PATTERN_WORDS:
        raise AnswerError(f"{where}: pattern longer than {MAX_PATTERN_WORDS} words: {p!r}")


def _item(i: int, raw: object, seen: dict[str, int], path: Path) -> AnswerItem:
    if not isinstance(raw, dict):
        raise AnswerError(f"{path}: item {i}: must be an object")
    iid: Any = raw.get("id")
    where: str = f"{path}: item {i}" + (f" ({iid})" if isinstance(iid, str) and iid else "")
    if not isinstance(iid, str) or not iid:
        raise AnswerError(f"{where}: 'id' must be a non-empty string")
    if iid in seen:
        raise AnswerError(f"{where}: duplicate id, first used by item {seen[iid]}")
    seen[iid] = i
    q: Any = raw.get("q")
    if not isinstance(q, str) or not q.strip():
        raise AnswerError(f"{where}: 'q' must be a non-empty string")
    expect: Any = raw.get("expect", "answer")
    if expect not in ("answer", "no_answer"):
        raise AnswerError(f"{where}: 'expect' must be 'answer' or 'no_answer', got {expect!r}")
    chapter: Any = raw.get("chapter")
    if chapter is not None and not isinstance(chapter, str):
        raise AnswerError(f"{where}: 'chapter' must be a string")
    pages: Any = raw.get("pages", [])
    if not isinstance(pages, list) or not all(isinstance(p, str) and p for p in pages):
        raise AnswerError(f"{where}: 'pages' must be a list of non-empty strings")
    if expect == "answer" and not pages:
        raise AnswerError(f"{where}: 'pages' must not be empty")
    must: Any = raw.get("must", [])
    forbid: Any = raw.get("forbid", [])
    if not isinstance(must, list) or not isinstance(forbid, list):
        raise AnswerError(f"{where}: 'must' and 'forbid' must be lists")
    if expect == "no_answer":
        if must or forbid:
            raise AnswerError(f"{where}: a no_answer item must not have 'must' or 'forbid'")
    elif not must:
        raise AnswerError(f"{where}: 'must' must not be empty")
    fact: Any
    for fact in must:
        if not isinstance(fact, list) or not fact:
            raise AnswerError(f"{where}: a fact must be a non-empty list of alternatives")
        alt: Any
        for alt in fact:
            _check_pattern(alt, where)
    f: Any
    for f in forbid:
        _check_pattern(f, where)
    return AnswerItem(
        id=iid,
        q=q,
        expect=expect,
        chapter=chapter,
        pages=list(pages),
        must=[list(fact) for fact in must],
        forbid=list(forbid),
    )


def load_items(path: Path) -> list[AnswerItem]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise AnswerError(f"cannot read item set {path}: {e}") from e
    except ValueError as e:
        raise AnswerError(f"item set {path} is not valid JSON: {e}") from e
    raw: Any = data.get("items") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise AnswerError(f"item set {path}: 'items' must be a non-empty list")
    seen: dict[str, int] = {}
    return [_item(i, r, seen, path) for i, r in enumerate(raw)]


def items_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        raise AnswerError(f"cannot read item set {path}: {e}") from e


def score_answer(item: AnswerItem, answer: str, retrieved: Sequence[str]) -> Grade:
    cited: list[str] = cited_pages(answer)
    in_context: float | None = sum(1 for c in cited if c in retrieved) / len(cited) if cited else None
    refused: bool = is_refusal(answer)
    if item.expect == "no_answer":
        return Grade(
            score=1.0 if refused else 0.0,
            passed=refused,
            facts_hit=0,
            facts_total=0,
            missed=[],
            forbidden=[],
            refused=refused,
            cited=cited,
            cite_in_context=in_context,
            cite_expected=False,
        )
    missed: list[int] = [
        i for i, fact in enumerate(item.must) if not any(matches(alt, answer) for alt in fact)
    ]
    total: int = len(item.must)
    hit: int = total - len(missed)
    forbidden: list[str] = [p for p in item.forbid if matches(p, answer)]
    return Grade(
        score=hit / total,
        passed=not missed and not forbidden,
        facts_hit=hit,
        facts_total=total,
        missed=missed,
        forbidden=forbidden,
        refused=refused,
        cited=cited,
        cite_in_context=in_context,
        cite_expected=any(c in item.pages for c in cited),
    )


def _share(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: Sequence[AnswerRow]) -> Summary:
    if not rows:
        raise ValueError("no rows to summarize")
    ans: list[AnswerRow] = [r for r in rows if r.expect == "answer"]
    noans: list[AnswerRow] = [r for r in rows if r.expect != "answer"]
    gen: list[float] = [r.generate_s for r in rows]
    return Summary(
        n_answer=len(ans),
        n_no_answer=len(noans),
        answer_score=_share([r.grade.score for r in ans]),
        pass_rate=_share([float(r.grade.passed) for r in ans]),
        wrong_rate=_share([float(bool(r.grade.forbidden)) for r in ans]),
        false_refusal_rate=_share([float(r.grade.refused) for r in ans]),
        no_answer_accuracy=_share([r.grade.score for r in noans]),
        cited_rate=_share([float(bool(r.grade.cited)) for r in ans]),
        cite_precision=_share([r.grade.cite_in_context for r in ans if r.grade.cite_in_context is not None]),
        cite_expected_rate=_share([float(r.grade.cite_expected) for r in ans]),
        generate_median_s=statistics.median(gen),
        generate_mean_s=sum(gen) / len(gen),
        generate_max_s=max(gen),
        generate_total_s=sum(gen),
        retrieve_total_s=sum(r.retrieve_s for r in rows),
    )


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_summary(s: Summary) -> list[str]:
    return [
        f"answers   n={s.n_answer}  score {_f(s.answer_score)}  pass {_f(s.pass_rate)}  "
        f"wrong {_f(s.wrong_rate)}  false-refusal {_f(s.false_refusal_rate)}",
        f"no-answer n={s.n_no_answer}  accuracy {_f(s.no_answer_accuracy)}",
        f"citations cited {_f(s.cited_rate)}  in-context {_f(s.cite_precision)}  "
        f"expected-page {_f(s.cite_expected_rate)}",
        f"time      generate median {s.generate_median_s:.1f} s  mean {s.generate_mean_s:.1f} s  "
        f"max {s.generate_max_s:.1f} s  total {s.generate_total_s:.0f} s  "
        f"retrieve total {s.retrieve_total_s:.1f} s",
    ]


def default_run_path(backend: str, model: str, now: datetime) -> Path:
    slug: str = re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-")
    return RUNS / f"answers-{backend}-{slug}-{now:%Y%m%d-%H%M%S}.json"


def write_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(run), ensure_ascii=False, indent=1), encoding="utf-8")


def _obj(d: object, key: str) -> object:
    if not isinstance(d, dict) or key not in d:
        raise KeyError(key)
    return d[key]


def _str(d: object, key: str) -> str:
    v: object = _obj(d, key)
    if not isinstance(v, str):
        raise TypeError(key)
    return v


def _int(d: object, key: str) -> int:
    v: object = _obj(d, key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(key)
    return v


def _num(d: object, key: str) -> float:
    v: object = _obj(d, key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise TypeError(key)
    return float(v)


def _opt_num(d: object, key: str) -> float | None:
    return None if _obj(d, key) is None else _num(d, key)


def _bool(d: object, key: str) -> bool:
    v: object = _obj(d, key)
    if not isinstance(v, bool):
        raise TypeError(key)
    return v


def _strs(d: object, key: str) -> list[str]:
    v: object = _obj(d, key)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise TypeError(key)
    return list(v)


def _ints(d: object, key: str) -> list[int]:
    v: object = _obj(d, key)
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise TypeError(key)
    return list(v)


def _nums(d: object, key: str) -> dict[str, float]:
    v: object = _obj(d, key)
    if not isinstance(v, dict):
        raise TypeError(key)
    return {k: _num(v, k) for k in v}


def _grade(d: object) -> Grade:
    return Grade(
        score=_num(d, "score"),
        passed=_bool(d, "passed"),
        facts_hit=_int(d, "facts_hit"),
        facts_total=_int(d, "facts_total"),
        missed=_ints(d, "missed"),
        forbidden=_strs(d, "forbidden"),
        refused=_bool(d, "refused"),
        cited=_strs(d, "cited"),
        cite_in_context=_opt_num(d, "cite_in_context"),
        cite_expected=_bool(d, "cite_expected"),
    )


def _row(d: object) -> AnswerRow:
    return AnswerRow(
        id=_str(d, "id"),
        q=_str(d, "q"),
        expect=_str(d, "expect"),
        retrieved=_strs(d, "retrieved"),
        answer=_str(d, "answer"),
        retrieve_s=_num(d, "retrieve_s"),
        generate_s=_num(d, "generate_s"),
        stats=_nums(d, "stats"),
        grade=_grade(_obj(d, "grade")),
    )


def _summary(d: object) -> Summary:
    return Summary(
        n_answer=_int(d, "n_answer"),
        n_no_answer=_int(d, "n_no_answer"),
        answer_score=_opt_num(d, "answer_score"),
        pass_rate=_opt_num(d, "pass_rate"),
        wrong_rate=_opt_num(d, "wrong_rate"),
        false_refusal_rate=_opt_num(d, "false_refusal_rate"),
        no_answer_accuracy=_opt_num(d, "no_answer_accuracy"),
        cited_rate=_opt_num(d, "cited_rate"),
        cite_precision=_opt_num(d, "cite_precision"),
        cite_expected_rate=_opt_num(d, "cite_expected_rate"),
        generate_median_s=_num(d, "generate_median_s"),
        generate_mean_s=_num(d, "generate_mean_s"),
        generate_max_s=_num(d, "generate_max_s"),
        generate_total_s=_num(d, "generate_total_s"),
        retrieve_total_s=_num(d, "retrieve_total_s"),
    )


def read_run(path: Path) -> Run:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise AnswerError(f"cannot read run file {path}: {e}") from e
    except ValueError as e:
        raise AnswerError(f"run file {path} is not valid JSON: {e}") from e
    try:
        if _int(data, "schema") != SCHEMA:
            raise AnswerError(f"run file {path}: schema {data['schema']} is not supported, want {SCHEMA}")
        rows: Any = _obj(data, "rows")
        if not isinstance(rows, list):
            raise TypeError("rows")
        return Run(
            schema=SCHEMA,
            created=_str(data, "created"),
            backend=_str(data, "backend"),
            model=_str(data, "model"),
            options=_nums(data, "options"),
            k=_int(data, "k"),
            index_points=_int(data, "index_points"),
            items_file=_str(data, "items_file"),
            items_sha256=_str(data, "items_sha256"),
            platform=_str(data, "platform"),
            cpus=None if _obj(data, "cpus") is None else _int(data, "cpus"),
            summary=_summary(_obj(data, "summary")),
            rows=[_row(r) for r in rows],
        )
    except (KeyError, TypeError) as e:
        raise AnswerError(f"run file {path}: missing or invalid field {e}") from e


def rescore(run: Run, items: Sequence[AnswerItem]) -> tuple[list[AnswerRow], int]:
    by_id: dict[str, AnswerItem] = {it.id: it for it in items}
    rows: list[AnswerRow] = []
    skipped: int = 0
    row: AnswerRow
    for row in run.rows:
        item: AnswerItem | None = by_id.get(row.id)
        if item is None:
            skipped += 1
            continue
        rows.append(
            dataclasses.replace(row, expect=item.expect, grade=score_answer(item, row.answer, row.retrieved))
        )
    return rows, skipped


def _pairs(a: Sequence[AnswerRow], b: Sequence[AnswerRow]) -> list[tuple[AnswerRow, AnswerRow]]:
    b_by_id: dict[str, AnswerRow] = {r.id: r for r in b if r.expect == "answer"}
    return [(r, b_by_id[r.id]) for r in a if r.expect == "answer" and r.id in b_by_id]


def compare(
    a: Sequence[AnswerRow], b: Sequence[AnswerRow], resamples: int = 10000, seed: int = 0
) -> Comparison:
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    pairs: list[tuple[AnswerRow, AnswerRow]] = _pairs(a, b)
    if not pairs:
        raise ValueError("no paired answer rows to compare")
    n: int = len(pairs)
    d: list[float] = [rb.grade.score - ra.grade.score for ra, rb in pairs]
    mean_a: float = sum(ra.grade.score for ra, _ in pairs) / n
    mean_b: float = sum(rb.grade.score for _, rb in pairs) / n
    rng: random.Random = random.Random(seed)
    means: list[float] = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return Comparison(
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_b - mean_a,
        sd=statistics.stdev(d) if n > 1 else 0.0,
        ci_low=means[math.floor(0.025 * resamples)],
        ci_high=means[math.ceil(0.975 * resamples) - 1],
        better=sum(1 for x in d if x >= 1e-9),
        worse=sum(1 for x in d if x <= -1e-9),
        same=sum(1 for x in d if abs(x) < 1e-9),
    )


def _header(run: Run, path: Path) -> str:
    return f"{run.backend} {run.model}  {run.created}  k={run.k}  index={run.index_points}  {path}"


def cmd_report(run: Path, other: Path | None, items: Path, as_json: bool) -> int:
    try:
        item_set: list[AnswerItem] = load_items(items)
        paths: list[Path] = [run] if other is None else [run, other]
        runs: list[Run] = [read_run(p) for p in paths]
        scored: list[tuple[list[AnswerRow], int]] = [rescore(r, item_set) for r in runs]
        summaries: list[Summary] = [summarize(rows) for rows, _ in scored]
        cmp: Comparison | None = None
        if other is not None:
            cmp = compare(scored[0][0], scored[1][0])
    except (AnswerError, ValueError) as e:
        print(f"fr-drill: {e}", file=sys.stderr)
        return 2
    i: int
    p: Path
    for i, p in enumerate(paths):
        if scored[i][1]:
            print(f"fr-drill: {scored[i][1]} rows of {p} have no item in {items}", file=sys.stderr)
    if cmp is not None:
        pairs: list[tuple[AnswerRow, AnswerRow]] = _pairs(scored[0][0], scored[1][0])
        differ: int = sum(1 for ra, rb in pairs if ra.retrieved != rb.retrieved)
        if differ:
            print(
                f"fr-drill: retrieval differs on {differ} of {cmp.n} paired items",
                file=sys.stderr,
            )
    if as_json:
        out: dict[str, Any] = {
            "runs": [
                {
                    "file": str(p),
                    "backend": r.backend,
                    "model": r.model,
                    "created": r.created,
                    "k": r.k,
                    "index_points": r.index_points,
                    "skipped": scored[j][1],
                    "summary": dataclasses.asdict(summaries[j]),
                }
                for j, (p, r) in enumerate(zip(paths, runs, strict=True))
            ],
            "comparison": dataclasses.asdict(cmp) if cmp is not None else None,
        }
        print(json.dumps(out, ensure_ascii=False))
        return 0
    for i, p in enumerate(paths):
        if i:
            print()
        print(_header(runs[i], p))
        line: str
        for line in format_summary(summaries[i]):
            print(line)
    if cmp is not None:
        print()
        print(
            f"B - A  score {cmp.diff:+.3f}  95% CI [{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]  "
            f"sd {cmp.sd:.3f}  better {cmp.better}  worse {cmp.worse}  same {cmp.same}  (n={cmp.n})"
        )
    return 0
