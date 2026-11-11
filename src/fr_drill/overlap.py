"""Verbatim overlaps between JSON data and a textbook: sentences and long runs of words found as they are.

    fr-drill overlap BOOK FILE [FILE ...] [--run 8] [--sentence 5]

The textbook (text or PDF) stays outside the repository; its path is an argument.
"""

import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RUN: int = 8  # the shortest common chain of words that counts as a match
SENTENCE: int = 5  # the shortest French sentence found whole in the book

APOSTROPHES: str = "’‘ʼ`"
CYRILLIC: re.Pattern[str] = re.compile(r"[а-яё]", re.IGNORECASE)


@dataclass(frozen=True)
class Hit:
    text: str  # the string from the data, as it is
    match: str  # matched words joined by one space, in the normal form of tokens()
    kind: Literal["sentence", "run"]


def tokens(text: str) -> list[str]:
    """Words of a text: NFC, lower case, one kind of apostrophe; an apostrophe ends its word."""
    text = unicodedata.normalize("NFC", text).lower()
    ch: str
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    found: list[str] = re.findall(r"[^\W\d_]+'?", text)
    return found


def strings(data: object) -> list[str]:
    """Every string leaf of a JSON value in order; keys and anything under `source` are left out."""
    if isinstance(data, str):
        return [data]
    out: list[str] = []
    if isinstance(data, dict):
        key: object
        value: object
        for key, value in data.items():
            if key != "source":
                out += strings(value)
    elif isinstance(data, list):
        item: object
        for item in data:
            out += strings(item)
    return out


def find(texts: list[str], book: str, run: int = RUN, sentence: int = SENTENCE) -> list[Hit]:
    """One hit per distinct string: a whole French sentence in the book, else its longest run of words."""
    words: list[str] = tokens(book)
    starts: dict[str, list[int]] = {}
    i: int
    j: int
    word: str
    for i, word in enumerate(words):
        starts.setdefault(word, []).append(i)
    hits: list[Hit] = []
    seen: set[str] = set()
    text: str
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        t: list[str] = tokens(text)
        n: int = len(t)
        if not CYRILLIC.search(text) and n >= max(sentence, 1):
            if any(words[j : j + n] == t for j in starts.get(t[0], [])):
                hits.append(Hit(text, " ".join(t), "sentence"))
                continue
        best: int = 0
        best_at: int = 0
        for i in range(n):
            if n - i <= best:
                break
            for j in starts.get(t[i], []):
                length: int = 0
                while i + length < n and j + length < len(words) and t[i + length] == words[j + length]:
                    length += 1
                if length > best:
                    best, best_at = length, i
        if best >= max(run, 1):
            hits.append(Hit(text, " ".join(t[best_at : best_at + best]), "run"))
    return hits


def book_text(path: Path) -> str:
    """Text of a book: pages of a PDF joined by a newline, anything else read as UTF-8."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages: list[str] = [page.extract_text() or "" for page in PdfReader(path).pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8")


def main(book: Path, files: list[Path], run: int = RUN, sentence: int = SENTENCE) -> int:
    try:
        text: str = book_text(book)
    except Exception as e:  # unreadable or damaged book: whatever the reader raised, the input is bad
        print(f"{book}: cannot read the book: {e}", file=sys.stderr)
        return 2
    found: int = 0
    file: Path
    for file in files:
        try:
            data: object = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"{file}: cannot read: {e}", file=sys.stderr)
            return 2
        hit: Hit
        for hit in find(strings(data), text, run, sentence):
            print(f"{file}: {hit.kind}: {hit.text!r} matches {hit.match!r}")
            found += 1
    if not found:
        print("no overlaps")
        return 0
    return 1
