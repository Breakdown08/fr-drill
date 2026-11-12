"""Fakes and builders for the textbook-helper tests: no network, no real models."""

import io
import math
import re
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from pypdf import PdfReader, PdfWriter
from pypdf.constants import PageLabelStyle
from util import minimal_pdf

from fr_drill.rag import Chunk

# The real factories, saved by the autouse guard in conftest.py before it replaces them.
REAL: dict[str, Callable[..., Any]] = {}


class FakeEmbedder:
    """Bag of words: every token goes to one of `dim` buckets, the vector is L2-normalised."""

    def __init__(self, dim: int = 64) -> None:
        self.dim: int = dim
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def vector(self, text: str) -> list[float]:
        v: list[float] = [0.0] * self.dim
        tok: str
        for tok in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm: float = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_calls.append(list(texts))
        return [self.vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)


class FakeGenerator:
    """Remembers the prompts, answers with `reply`."""

    def __init__(self, reply: str = "  ответ (с. 1)\n") -> None:
        self.reply: str = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def mk(
    book: str = "heminway",
    pdf_page: int = 1,
    chunk: int = 0,
    text: str = "texte",
    chapter: str | None = None,
    page: str | None = None,
) -> Chunk:
    return Chunk(
        book=book,
        pdf_page=pdf_page,
        page=page if page is not None else str(pdf_page),
        chapter=chapter,
        chunk=chunk,
        text=text,
    )


def make_pdf(
    path: Path,
    texts: list[str],
    labels: list[tuple[int, int, PageLabelStyle, int]] | None = None,
    outline: list[tuple[str, int]] | None = None,
    children: list[tuple[str, int, int]] | None = None,
) -> Path:
    """A PDF with one text line per page. 0-based indexes: labels are (from, to, style, start),
    outline is (title, page), children are (title, page, index of the parent in `outline`)."""
    writer: PdfWriter = PdfWriter(clone_from=PdfReader(io.BytesIO(minimal_pdf(texts))))
    span: tuple[int, int, PageLabelStyle, int]
    for span in labels or []:
        writer.set_page_label(span[0], span[1], span[2], start=span[3])
    tops: list[Any] = []
    top: tuple[str, int]
    for top in outline or []:
        tops.append(writer.add_outline_item(top[0], top[1]))
    child: tuple[str, int, int]
    for child in children or []:
        writer.add_outline_item(child[0], child[1], parent=tops[child[2]])
    f: BinaryIO
    with path.open("wb") as f:
        writer.write(f)
    return path


PDF3_TEXTS: list[str] = ["Preface alpha", "Intro bravo charlie", "Intro delta echo"]


def pdf3(path: Path, nested: bool = True) -> Path:
    """Labels i, 1, 2; top-level chapter "1 Intro" on PDF page 2; a child of it on page 3."""
    return make_pdf(
        path,
        PDF3_TEXTS,
        labels=[(0, 0, PageLabelStyle.LOWERCASE_ROMAN, 1), (1, 2, PageLabelStyle.DECIMAL, 1)],
        outline=[("1 Intro", 1)],
        children=[("1.1 Sub", 2, 0)] if nested else [],
    )
