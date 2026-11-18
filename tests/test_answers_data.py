"""The shipped question set data/rag/answers.json (spec checks 28-33)."""

import re
from pathlib import Path

import pytest
from util import ROOT

from fr_drill import answers, rag
from fr_drill.answers import NO_ANSWER, AnswerItem, Grade

BOOK: Path = ROOT / "books" / "heminway.pdf"


def loaded() -> list[AnswerItem]:
    return answers.load_items(answers.ANSWERS)


def answer_items() -> list[AnswerItem]:
    return [i for i in loaded() if i.expect == "answer"]


def test_the_set_loads_and_is_big_enough() -> None:
    items: list[AnswerItem] = loaded()
    assert sum(1 for i in items if i.expect == "answer") >= 70
    assert sum(1 for i in items if i.expect == "no_answer") >= 5


def test_every_answer_item_is_chapter_21_pages_285_to_314() -> None:
    item: AnswerItem
    for item in answer_items():
        assert item.chapter == "21", item.id
        assert item.pages, item.id
        page: str
        for page in item.pages:
            assert re.fullmatch(r"\d+", page), (item.id, page)
            assert 285 <= int(page) <= 314, (item.id, page)


def test_no_alternative_is_found_in_its_own_question() -> None:
    item: AnswerItem
    for item in answer_items():
        fact: list[str]
        for fact in item.must:
            alt: str
            for alt in fact:
                assert not answers.matches(alt, item.q), (item.id, alt)


def ideal(item: AnswerItem) -> str:
    return " ; ".join(fact[0].removesuffix("*") for fact in item.must)


def test_the_ideal_answer_scores_one() -> None:
    item: AnswerItem
    for item in answer_items():
        g: Grade = answers.score_answer(item, ideal(item), item.pages)
        assert g.score == 1.0, (item.id, g.missed)
        assert g.passed is True, (item.id, g.forbidden)


def test_a_refusal_scores_zero_on_answer_items_and_passes_no_answer_items() -> None:
    item: AnswerItem
    for item in loaded():
        g: Grade = answers.score_answer(item, NO_ANSWER, [])
        if item.expect == "answer":
            assert g.score == 0.0, item.id
            assert g.passed is False, item.id
        else:
            assert g.passed is True, item.id
            assert g.score == 1.0, item.id


def test_ids_are_unique() -> None:
    ids: list[str] = [i.id for i in loaded()]
    assert len(ids) == len(set(ids)) > 0


LATIN: re.Pattern[str] = re.compile(r"[A-Za-zÀ-ÿŒœ]")
CYRILLIC: re.Pattern[str] = re.compile(r"[Ѐ-ӿ]")


def test_french_facts_are_on_the_book_pages() -> None:
    """Against the real textbook; skipped when the PDF is not here (check 32). Run it with -k book."""
    if not BOOK.exists():
        pytest.skip("no books/heminway.pdf")
    pages: list[rag.Page] = rag.select_pages(rag.read_pages(BOOK), chapter="21")
    by_label: dict[str, str] = {p.label: rag.normalize(p.text) for p in pages}
    item: AnswerItem
    for item in answer_items():
        text: str = " ".join(by_label.get(p, "") for p in item.pages)
        assert text.strip(), (item.id, "no text on its pages")
        fact: list[str]
        for fact in item.must:
            french: list[str] = [a for a in fact if LATIN.search(a) and not CYRILLIC.search(a)]
            if french:
                assert any(answers.matches(a, text) for a in french), (item.id, french)
        f: str
        for f in item.forbid:
            assert not answers.matches(f, text), (item.id, f)
