"""Textbook helper: reading, selecting pages, normalising, chunking (spec checks 6-14)."""

import signal
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from rag_fakes import PDF3_TEXTS, make_pdf, pdf3

from fr_drill import rag
from fr_drill.rag import Chunk, Page, RagError

PAGES: list[Page] = [
    Page(1, "i", None, "preface"),
    Page(2, "1", "1 Intro", "one"),
    Page(3, "2", "1 Intro", "two"),
]


@contextmanager
def within(seconds: int) -> Generator[None]:
    """Fail instead of hanging when the code under test loops."""

    def on_alarm(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("the code did not finish")

    previous: Any = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# read_pages


def test_read_pages_labels_and_chapters(tmp_path: Path) -> None:
    pages: list[Page] = rag.read_pages(pdf3(tmp_path / "b.pdf", nested=False))
    assert [p.pdf_page for p in pages] == [1, 2, 3]
    assert [p.label for p in pages] == ["i", "1", "2"]
    assert [p.chapter for p in pages] == [None, "1 Intro", "1 Intro"]
    text: str
    page: Page
    for text, page in zip(PDF3_TEXTS, pages, strict=True):
        assert text in page.text


def test_nested_outline_entries_are_ignored(tmp_path: Path) -> None:
    pages: list[Page] = rag.read_pages(pdf3(tmp_path / "b.pdf", nested=True))
    assert [p.chapter for p in pages] == [None, "1 Intro", "1 Intro"]


def test_later_top_level_entry_starts_the_next_chapter(tmp_path: Path) -> None:
    path: Path = make_pdf(tmp_path / "b.pdf", ["a", "b", "c", "d"], outline=[("1 One", 0), ("2 Two", 2)])
    assert [p.chapter for p in rag.read_pages(path)] == ["1 One", "1 One", "2 Two", "2 Two"]


def test_no_outline_and_no_labels(tmp_path: Path) -> None:
    pages: list[Page] = rag.read_pages(make_pdf(tmp_path / "b.pdf", ["un", "deux"]))
    assert [p.chapter for p in pages] == [None, None]
    assert [p.label for p in pages] == ["1", "2"]
    assert [p.pdf_page for p in pages] == [1, 2]


def test_empty_page_has_empty_text(tmp_path: Path) -> None:
    pages: list[Page] = rag.read_pages(make_pdf(tmp_path / "b.pdf", ["", "x"]))
    assert pages[0].text.strip() == ""


def test_missing_file_is_a_rag_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "nope.pdf"
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.read_pages(path)
    assert str(path) in str(err.value)


def test_garbage_file_is_a_rag_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "junk.pdf"
    path.write_bytes(b"this is not a pdf at all")
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.read_pages(path)
    assert str(path) in str(err.value)


# select_pages


def test_select_by_printed_labels() -> None:
    assert rag.select_pages(PAGES, pages_arg="1-2") == PAGES[1:]
    assert rag.select_pages(PAGES, pages_arg="1") == [PAGES[1]]
    assert rag.select_pages(PAGES, pages_arg="i-1") == PAGES[:2]


def test_select_by_chapter() -> None:
    assert rag.select_pages(PAGES, chapter="1") == PAGES[1:]


def test_select_everything_without_arguments() -> None:
    assert rag.select_pages(PAGES) == PAGES


def test_select_both_arguments_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        rag.select_pages(PAGES, pages_arg="1-2", chapter="1")


def test_select_unknown_label_is_a_rag_error() -> None:
    with pytest.raises(RagError):
        rag.select_pages(PAGES, pages_arg="99")
    with pytest.raises(RagError):
        rag.select_pages(PAGES, pages_arg="1-99")


def test_select_end_label_must_follow_the_start() -> None:
    with pytest.raises(RagError):
        rag.select_pages(PAGES, pages_arg="2-1")


def test_select_end_is_the_first_label_not_before_the_start() -> None:
    pages: list[Page] = [Page(1, "1", None, "a"), Page(2, "2", None, "b"), Page(3, "1", None, "c")]
    assert rag.select_pages(pages, pages_arg="2-1") == pages[1:]


def test_select_labels_are_not_pdf_numbers() -> None:
    pages: list[Page] = [Page(n, str(n - 11), None, "t") for n in range(12, 24)]
    assert [p.pdf_page for p in rag.select_pages(pages, pages_arg="1-12")] == list(range(12, 24))


def test_select_chapter_matches_the_first_token_exactly() -> None:
    pages: list[Page] = [Page(1, "1", "1 Intro", "a"), Page(2, "2", "10 More", "b"), Page(3, "3", None, "c")]
    assert rag.select_pages(pages, chapter="1") == [pages[0]]
    assert rag.select_pages(pages, chapter="10") == [pages[1]]


def test_select_unknown_chapter_lists_the_chapters() -> None:
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.select_pages(PAGES, chapter="7")
    assert "1 Intro" in str(err.value)
    pages: list[Page] = [Page(1, "1", "1 Intro", "a"), Page(2, "2", "2 Verbs", "b")]
    err2: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err2:
        rag.select_pages(pages, chapter="7")
    assert "1 Intro; 2 Verbs" in str(err2.value)


# normalize


def test_normalize_joins_hyphenated_words_and_collapses_spaces() -> None:
    assert rag.normalize("mascu-\nline  le\n\tvillage ") == "masculine le village"
    assert rag.normalize("inté-\nressant") == "intéressant"


def test_normalize_keeps_the_hyphen_before_a_capital() -> None:
    assert rag.normalize("Saint-\nÉtienne") == "Saint- Étienne"


def test_normalize_keeps_an_ordinary_hyphen() -> None:
    assert rag.normalize("peut-être  oui") == "peut-être oui"


# chunk_pages


def words_page(prefix: str, n: int, number: int = 1, chapter: str | None = None) -> Page:
    return Page(number, str(number), chapter, " ".join(f"{prefix}{i:03d}" for i in range(n)))


def test_short_page_is_one_chunk() -> None:
    page: Page = Page(5, "ix", "1 Intro", "Le  chat\nnoir")
    chunks: list[Chunk] = rag.chunk_pages([page], "heminway")
    assert chunks == [Chunk("heminway", 5, "ix", "1 Intro", 0, "Le chat noir")]


def test_text_of_exactly_size_is_one_chunk() -> None:
    page: Page = Page(1, "1", None, "a" * 50 + " " + "b" * 49)
    chunks: list[Chunk] = rag.chunk_pages([page], "x", size=100, overlap=20)
    assert [c.text for c in chunks] == [page.text]


def test_empty_page_gives_no_chunks() -> None:
    assert rag.chunk_pages([Page(1, "1", None, "  \n \t ")], "x") == []
    assert rag.chunk_pages([Page(1, "1", None, "")], "x") == []


def test_long_page_is_cut_into_clean_overlapping_windows() -> None:
    page: Page = words_page("mot", 500, chapter="3 Verbs")
    chunks: list[Chunk] = rag.chunk_pages([page], "b", size=200, overlap=50)
    page_words: set[str] = set(page.text.split())
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text == c.text.strip() for c in chunks)
    seen: set[str] = set()
    c: Chunk
    for c in chunks:
        seen |= set(c.text.split())
        assert set(c.text.split()) <= page_words
        assert (c.book, c.pdf_page, c.page, c.chapter) == ("b", 1, "1", "3 Verbs")
    assert seen == page_words
    left: Chunk
    right: Chunk
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert set(left.text.split()) & set(right.text.split())
    assert [c.chunk for c in chunks] == list(range(len(chunks)))
    assert rag.chunk_pages([page], "b", size=200, overlap=50) == chunks


def test_chunks_never_cross_a_page() -> None:
    first: Page = words_page("aaa", 120, number=1)
    second: Page = words_page("bbb", 120, number=2)
    chunks: list[Chunk] = rag.chunk_pages([first, second], "b", size=200, overlap=50)
    mine: list[Chunk] = [c for c in chunks if c.pdf_page == 1]
    other: list[Chunk] = [c for c in chunks if c.pdf_page == 2]
    assert mine and other and len(mine) + len(other) == len(chunks)
    assert all(w.startswith("aaa") for c in mine for w in c.text.split())
    assert all(w.startswith("bbb") for c in other for w in c.text.split())
    assert [c.chunk for c in mine] == list(range(len(mine)))
    assert [c.chunk for c in other] == list(range(len(other)))


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_bad_sizes_are_value_errors(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        rag.chunk_pages([Page(1, "1", None, "texte " * 50)], "x", size=size, overlap=overlap)


def test_a_word_longer_than_size_is_cut_at_size() -> None:
    page: Page = Page(1, "1", None, "x" * 450)
    with within(10):
        chunks: list[Chunk] = rag.chunk_pages([page], "x", size=100, overlap=20)
    assert len(chunks) >= 5
    assert all(0 < len(c.text) <= 100 and set(c.text) == {"x"} for c in chunks)
    assert len(chunks[0].text) == 100


def test_long_text_without_a_space_in_the_second_half_is_cut_at_size() -> None:
    page: Page = Page(1, "1", None, "ab " + "y" * 300)
    with within(10):
        chunks: list[Chunk] = rag.chunk_pages([page], "x", size=100, overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].text.startswith("ab ")
