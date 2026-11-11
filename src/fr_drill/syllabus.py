"""Topic map of a language level: validation and the generated markdown note.

The source of truth is `topics.json`; `syllabus.md` next to it is generated
from it and is not edited by hand.
"""

import json
import sys
from pathlib import Path
from typing import Any

from fr_drill import DATA

SYLLABUS: Path = DATA / "syllabus"
KINDS: dict[str, str] = {"gram": "Грамматика", "lex": "Лексика", "act": "Коммуникативные задачи"}


def load(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def check(doc: dict[str, Any]) -> list[str]:
    """Return problems found in the topic map; an empty list means it is valid."""
    errors: list[Any] = []
    stages: set[int] = {int(key) for key in doc.get("stages", {})}
    topics: dict[str, Any] = {}
    topic: Any
    for topic in doc.get("topics", []):
        tid: Any = topic.get("id", "?")
        if tid in topics:
            errors.append(f"{tid}: duplicate id")
        topics[tid] = topic
        kind: Any = tid.split(".")[0]
        if kind not in KINDS or topic.get("kind") != kind:
            errors.append(f"{tid}: kind must be one of {sorted(KINDS)} and match the id prefix")
        if topic.get("stage") not in stages:
            errors.append(f"{tid}: unknown stage {topic.get('stage')}")
        if not topic.get("title"):
            errors.append(f"{tid}: empty title")

    for tid, topic in topics.items():
        req: Any
        for req in topic.get("requires", []):
            if req not in topics:
                errors.append(f"{tid}: requires unknown topic {req}")
            elif topics[req].get("stage", 0) > topic.get("stage", 0):
                errors.append(f"{tid}: requires {req} from a later stage")

    path: list[Any]
    done: set[Any]
    done, path = set(), []

    def visit(tid: str) -> None:
        if tid in path:
            errors.append("cycle: " + " → ".join(path[path.index(tid) :] + [tid]))
            return
        if tid in done or tid not in topics:
            return
        path.append(tid)
        req: Any
        for req in topics[tid].get("requires", []):
            visit(req)
        path.pop()
        done.add(tid)

    for tid in topics:
        visit(tid)
    return errors


def render(doc: dict[str, Any]) -> str:
    titles: dict[str, Any] = {t["id"]: t["title"] for t in doc["topics"]}
    lines: list[str] = [
        f"# Карта тем {doc['level']}",
        "",
        "*Сгенерировано `fr-drill syllabus` из `topics.json` — не править вручную.*",
        f"Источник: {doc['source']}. Тем: {len(doc['topics'])}.",
        "",
        "Оси: `gram` — как сказать, `lex` — какими словами, `act` — что сделать.",
        "«Требует» — темы, которые стоит пройти раньше.",
    ]
    name: Any
    key: Any
    for key, name in doc["stages"].items():
        stage: int = int(key)
        lines += ["", f"## Этап {stage}. {name}"]
        kind_name: str
        kind: str
        for kind, kind_name in KINDS.items():
            group: list[Any] = [t for t in doc["topics"] if t["stage"] == stage and t["kind"] == kind]
            if not group:
                continue
            lines += ["", f"### {kind_name}", ""]
            t: Any
            for t in group:
                line: str = f"- **{t['title']}** `{t['id']}`"
                if t.get("notes"):
                    line += f" — {t['notes']}"
                if t.get("requires"):
                    line += "  \n  требует: " + ", ".join(titles[r] for r in t["requires"])
                lines.append(line)
    return "\n".join(lines) + "\n"


def main(check_only: bool = False, root: Path = SYLLABUS, out: Path | None = None) -> int:
    doc: dict[str, Any] = load(root / "topics.json")
    errors: list[str] = check(doc)
    error: str
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    if check_only:
        print(f"ok: {len(doc['topics'])} topics")
        return 0
    target: Path = out if out is not None else root / "syllabus.md"
    target.write_text(render(doc), encoding="utf-8")
    print(f"written: {target} ({len(doc['topics'])} topics)")
    return 0
