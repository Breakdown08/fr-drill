"""Fixtures: loaded data, a drill server on temp files, and the guard on the real history."""

import os
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from util import ROOT, serving

import fr_drill
from fr_drill import (
    drill,
    etre_avoir,
    negation,
    numbers,
    prepositions,
    sentences,
    vocab,
)


def snapshot(dirs: set[Path]) -> dict[str, int]:
    """Size of every file under the given directories."""
    out: dict[str, int] = {}
    d: Path
    for d in dirs:
        f: Path
        for f in sorted(d.rglob("*")) if d.exists() else []:
            if f.is_file():
                out[str(f)] = f.stat().st_size
    return out


@pytest.fixture(scope="session", autouse=True)
def history_untouched() -> Iterator[None]:
    """Spec check 31: a full run writes nothing into the real progress directories."""
    dirs: set[Path] = {ROOT / "progress"}
    progress: Any = getattr(fr_drill, "PROGRESS", None)
    if progress is not None:
        dirs.add(Path(progress))
    if os.environ.get("FR_DRILL_PROGRESS"):
        dirs.add(Path(os.environ["FR_DRILL_PROGRESS"]))
    before: dict[str, int] = snapshot(dirs)
    yield
    assert snapshot(dirs) == before


@pytest.fixture(scope="session")
def verbs() -> list[dict[str, Any]]:
    return drill.load()


@pytest.fixture(scope="session")
def cards() -> list[dict[str, Any]]:
    vocab_file: Any = getattr(fr_drill, "VOCAB", None)
    if vocab_file is not None and not vocab_file.exists():
        pytest.skip("no vocabulary file")
    return vocab.load()


class Live:
    """A running drill server on temp files; `t` holds the trainers by name."""

    def __init__(self, port: int, t: dict[str, Any]) -> None:
        self.port: int = port
        self.t: dict[str, Any] = t


def build_trainers(tmp: Path, verbs: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "drill": drill.Drill(verbs, tmp / "drill.jsonl"),
        "vocab": vocab.Vocab(cards, tmp / "vocab.jsonl"),
        "numbers": numbers.Numbers(tmp / "numbers.jsonl"),
        "negation": negation.Negation(negation.load(verbs=verbs), tmp / "negation.jsonl"),
        "etre": etre_avoir.EtreAvoir(etre_avoir.load(), tmp / "etre-avoir.jsonl"),
        "sentences": sentences.Sentences(
            sentences.load(verbs=verbs, cards=cards),
            tmp / "sentences.jsonl",
            (tmp / "verbs.jsonl",),
            tmp / "words.jsonl",
        ),
        "preps": prepositions.Prepositions(prepositions.load(), tmp / "prepositions.jsonl"),
    }


@pytest.fixture
def live(tmp_path: Path, verbs: list[dict[str, Any]], cards: list[dict[str, Any]]) -> Iterator[Live]:
    t: dict[str, Any] = build_trainers(tmp_path, verbs, cards)

    def make(p: int) -> type[BaseHTTPRequestHandler]:
        handler: type[BaseHTTPRequestHandler] = drill.handler(
            t["drill"], p, t["vocab"], t["numbers"], t["negation"], t["etre"], t["sentences"], t["preps"]
        )
        return handler

    port: int
    with serving(make) as port:
        yield Live(port, t)
