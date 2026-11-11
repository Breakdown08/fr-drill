"""Task bank of A1 French: validation of `data/bank/*.json` against `data/bank/format.md`.

A file per main topic (`gram.etre.json`) holds `{"docs": [...], "items": [...]}`;
`consignes.json` holds shared instructions `{key: {"fr", "ru"}}`. Topics and their stages
come from `data/syllabus/topics.json`. `/exam` (phase 3) builds tests from what passes here.
"""

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard

from fr_drill import DATA
from fr_drill.drill import bare, norm

BANK: Path = DATA / "bank"
TOPICS: Path = DATA / "syllabus/topics.json"
CONSIGNES: str = "consignes.json"

TYPES: set[str] = {"choice", "short", "number", "write"}
PARTS: set[str] = {"practice", "ce", "pe"}
DOC_KINDS: set[str] = {"courriel", "affiche", "annonce", "carte", "article", "message", "formulaire"}
SOURCES: set[str] = {"author", "book", "delf-sample", "teacher"}
# Up to this stage the instruction is shown in Russian too; later only in French, as at the exam.
RU_UNTIL: int = 2
GAP: str = "___"

ITEM_ID: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*-\d{3}$")
DOC_ID: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
TIME: re.Pattern[str] = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

COMMON: set[str] = {"id", "type", "part", "topics", "consigne", "prompt", "doc", "explain", "source"}
FIELDS: dict[str, set[str]] = {
    "choice": COMMON | {"options", "answer"},
    "short": COMMON | {"accept"},
    "number": COMMON | {"answer"},
    "write": COMMON | {"points", "min_words"},
}
DOC_FIELDS: set[str] = {"id", "kind", "text"}


def is_text(v: object) -> bool:
    return isinstance(v, str) and v.strip() != ""


def is_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def check_consignes(consignes: object) -> list[str]:
    if not isinstance(consignes, dict):
        return [f"{CONSIGNES}: must be an object {{key: {{fr, ru}}}}"]
    errors: list[Any] = []
    c: Any
    key: Any
    for key, c in consignes.items():
        if not isinstance(c, dict) or not is_text(c.get("fr")) or set(c) - {"fr", "ru"}:
            errors.append(f"{CONSIGNES}: {key}: must be {{fr, ru?}} with non-empty fr")
        elif "ru" in c and not is_text(c["ru"]):
            errors.append(f"{CONSIGNES}: {key}: empty ru")
    return errors


def check_item(
    item: dict[str, Any], stem: str, stages: dict[str, Any], docs: set[str], consignes: dict[str, Any]
) -> list[str]:
    """Problems of one item; `stem` is its file name without .json."""
    iid: Any = item.get("id", "?")
    errors: list[Any] = []

    def err(msg: str) -> None:
        errors.append(f"{stem}: {iid}: {msg}")

    if not ITEM_ID.match(str(item.get("id", ""))):
        err("id must look like word-NNN")
    part: Any | None
    kind: Any | None
    kind, part = item.get("type"), item.get("part")
    if kind not in TYPES:
        err(f"type must be one of {sorted(TYPES)}")
        return errors
    if part not in PARTS:
        err(f"part must be one of {sorted(PARTS)}")
    extra: set[str]
    if extra := set(item) - FIELDS[kind]:
        err(f"unknown fields {sorted(extra)}")

    topics: Any | None = item.get("topics")
    if not isinstance(topics, list) or not topics:
        err("topics must be a non-empty list")
        topics = []
    t: Any
    for t in topics:
        if t not in stages:
            err(f"unknown topic {t}")
    if topics and topics[0] != stem:
        err(f"main topic {topics[0]} does not match file {stem}.json")
    stage: Any | int = max((stages[t] for t in topics if t in stages), default=0)

    consigne: Any | None = item.get("consigne")
    if isinstance(consigne, str):
        if consigne not in consignes:
            err(f"consigne key {consigne} not in {CONSIGNES}")
        consigne = consignes.get(consigne)
    elif not isinstance(consigne, dict) or not is_text(consigne.get("fr")) or set(consigne) - {"fr", "ru"}:
        err("consigne must be {fr, ru} or a key of " + CONSIGNES)
        consigne = None
    if consigne and stage <= RU_UNTIL and not is_text(consigne.get("ru")):
        err(f"stage {stage} ≤ {RU_UNTIL}: consigne needs ru")

    doc: Any | None = item.get("doc")
    if doc is not None and doc not in docs:
        err(f"unknown doc {doc}")
    if part == "ce" and doc is None:
        err("ce needs a doc")

    prompt: Any | None = item.get("prompt")
    if prompt is None:
        if not (kind == "write" and doc is not None):
            err("prompt is required (only write with a doc may omit it)")
    elif not is_text(prompt):
        err("empty prompt")
    else:
        gaps: Any = prompt.count(GAP)
        # practice choice/short are gap-fill: one ___ that the answer fills.
        if part == "practice" and kind in {"choice", "short"} and gaps != 1:
            err(f"practice {kind} needs exactly one {GAP} in prompt, found {gaps}")
        elif gaps > 1:
            err(f"more than one {GAP} in prompt")

    if part == "practice" and not is_text(item.get("explain")):
        err("practice needs explain")

    source: Any | None = item.get("source")
    if not isinstance(source, dict) or source.get("by") not in SOURCES or not is_text(source.get("ref")):
        err(f"source must be {{by: {'|'.join(sorted(SOURCES))}, ref}}")

    if kind == "choice":
        options: Any | None = item.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 4 or not all(map(is_text, options)):
            err("choice needs 2–4 non-empty options")
        else:
            if len({norm(o) for o in options}) < len(options):
                err("options repeat after normalization")
            answer: Any | None = item.get("answer")
            if not is_int(answer) or not 0 <= answer < len(options):
                err(f"answer must be an index 0..{len(options) - 1}")
    elif kind == "short":
        accept: Any | None = item.get("accept")
        if not isinstance(accept, list) or not accept or not all(map(is_text, accept)):
            err("short needs a non-empty accept list of non-empty strings")
        else:
            # In ce a missing accent is accepted, so forms differing only by accents are one form.
            key: Callable[[str], str] = (lambda s: bare(norm(s))) if part == "ce" else norm
            if len({key(a) for a in accept}) < len(accept):
                err("accept repeats after normalization")
    elif kind == "number":
        answer = item.get("answer")
        ok: Any | bool = (is_int(answer) or isinstance(answer, float)) and answer >= 0
        if not ok and not (isinstance(answer, str) and TIME.match(answer)):
            err('number answer must be a non-negative number or "HH:MM"')
    elif kind == "write":
        points: Any | None = item.get("points")
        if not isinstance(points, list) or not points or not all(map(is_text, points)):
            err("write needs a non-empty points list")
        min_words: Any | None = item.get("min_words")
        if not is_int(min_words) or min_words <= 0:
            err("write needs min_words, a positive integer")
    return errors


def check(files: dict[str, dict[str, Any]], stages: dict[str, int], consignes: object) -> list[str]:
    """Problems across the bank; `files` maps a file stem to its parsed content."""
    errors: list[str] = check_consignes(consignes)
    known: dict[str, Any] = consignes if isinstance(consignes, dict) else {}
    seen: dict[str, str] = {}  # id → file: items and docs share one namespace
    docs: set[Any] = set()
    content: dict[str, Any]
    stem: str
    for stem, content in files.items():
        if not isinstance(content, dict) or "items" not in content or set(content) - {"docs", "items"}:
            errors.append(f"{stem}: must be {{docs?, items}}")
            continue
        d: Any
        for d in content.get("docs", []):
            did: Any | str = d.get("id", "?") if isinstance(d, dict) else "?"
            if not isinstance(d, dict) or not DOC_ID.match(str(did)):
                errors.append(f"{stem}: doc {did}: id must be lowercase words joined by -")
                continue
            if did in seen:
                errors.append(f"{stem}: {did}: duplicate id (first in {seen[did]})")
            seen[did] = stem
            docs.add(did)
            if d.get("kind") not in DOC_KINDS:
                errors.append(f"{stem}: {did}: doc kind must be one of {sorted(DOC_KINDS)}")
            if not is_text(d.get("text")):
                errors.append(f"{stem}: {did}: empty doc text")
            extra: set[str]
            if extra := set(d) - DOC_FIELDS:
                errors.append(f"{stem}: {did}: unknown doc fields {sorted(extra)}")
    for stem, content in files.items():
        if not isinstance(content, dict) or not isinstance(content.get("items"), list):
            continue
        item: Any
        for item in content["items"]:
            if not isinstance(item, dict):
                errors.append(f"{stem}: item must be an object")
                continue
            iid: Any | None = item.get("id")
            if isinstance(iid, str):
                if iid in seen:
                    errors.append(f"{stem}: {iid}: duplicate id (first in {seen[iid]})")
                seen[iid] = stem
            errors += check_item(item, stem, stages, docs, known)
    return errors


def load(
    root: Path = BANK, topics: Path = TOPICS
) -> tuple[dict[str, Any], dict[str, Any], object, list[str]]:
    """(files, stages, consignes, JSON errors) — unreadable files are reported, not raised."""
    errors: list[Any] = []

    def read(path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path.name}: {e}")
            return None

    stages: dict[str, Any] = {
        t["id"]: t["stage"] for t in json.loads(topics.read_text(encoding="utf-8"))["topics"]
    }
    consignes: object = read(root / CONSIGNES) if (root / CONSIGNES).exists() else {}
    files: dict[str, Any] = {}
    path: Path
    content: object
    for path in sorted(root.glob("*.json")):
        if path.name != CONSIGNES and (content := read(path)) is not None:
            files[path.stem] = content
    return files, stages, consignes if consignes is not None else {}, errors


def summary(files: dict[str, Any]) -> str:
    items: list[Any] = [i for c in files.values() for i in c.get("items", [])]
    docs: int = sum(len(c.get("docs", [])) for c in files.values())
    n: int
    parts: str = ", ".join(f"{p} {n}" for p in sorted(PARTS) if (n := sum(i.get("part") == p for i in items)))
    return f"ok: {len(items)} items, {docs} docs in {len(files)} files" + (f" ({parts})" if parts else "")


def main(root: Path = BANK, topics: Path = TOPICS) -> int:
    errors: list[str]
    consignes: object
    stages: dict[str, Any]
    files: dict[str, Any]
    files, stages, consignes, errors = load(root, topics)
    errors += check(files, stages, consignes)
    error: str
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(summary(files))
    return 0
