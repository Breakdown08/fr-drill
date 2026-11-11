"""French A1 drill and DELF-style exam trainer: shared paths and jsonl helpers."""

import json
import os
from pathlib import Path
from typing import Any, TextIO

ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data"
PROGRESS: Path = (
    Path(os.environ["FR_DRILL_PROGRESS"]) if os.environ.get("FR_DRILL_PROGRESS") else ROOT / "progress"
)
VOCAB: Path = (
    Path(os.environ["FR_DRILL_VOCAB"]) if os.environ.get("FR_DRILL_VOCAB") else DATA / "drills" / "vocab.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        f: TextIO
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
