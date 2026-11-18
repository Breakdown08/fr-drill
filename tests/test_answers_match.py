"""Answer eval: matching, citations, refusals (spec checks 1-4)."""

import pytest

from fr_drill import answers
from fr_drill.answers import NO_ANSWER


@pytest.mark.parametrize(
    ("pattern", "text", "want"),
    [
        ("au", "Говорят **au** Japon", True),
        ("au", "aux enfants", False),
        ("au", "autre", False),
        ("en été", "EN ÉTÉ", True),
        ("merci d'", "Merci d’avoir appelé", True),
        ("женск*", "слова женского рода", True),
        ("женск*", "неженский", False),
        ("определённ*", "в определенном случае", True),
        ("à pied", "a pied", False),
    ],
)
def test_matches_table(pattern: str, text: str, want: bool) -> None:
    assert answers.matches(pattern, text) is want


def test_matches_star_alone_is_an_error() -> None:
    with pytest.raises(ValueError):
        answers.matches("*", "x")


def test_normalize_nbsp_markdown_apostrophe() -> None:
    assert answers.normalize_text("  **Au** JAPON\n l’été ") == "au japon l'été"
    assert answers.normalize_text("**Au** Japon") == "au japon"


def test_normalize_yo_and_long_dash() -> None:
    assert answers.normalize_text("ёлка — «да»") == "елка — да"


def test_normalize_short_dashes_become_hyphens() -> None:
    assert answers.normalize_text("États‐Unis États‑Unis États‒Unis États–Unis") == (
        "états-unis états-unis états-unis états-unis"
    )


def test_normalize_quotes_and_underscores_are_spaces() -> None:
    assert answers.normalize_text('a_b “c” "d"') == "a b c d"


def test_cited_pages_example() -> None:
    text: str = "Так (с. 289). И так (с. 290, 291), снова (с. 289), (стр. 5)"
    assert answers.cited_pages(text) == ["289", "290", "291"]


def test_cited_pages_none_and_latin_c() -> None:
    assert answers.cited_pages("без ссылок") == []
    assert answers.cited_pages("(c. 12)") == ["12"]
    assert answers.cited_pages("(С. 7)") == ["7"]


def test_is_refusal_true_cases() -> None:
    assert answers.is_refusal(NO_ANSWER) is True
    assert answers.is_refusal("Увы. " + NO_ANSWER.rstrip(".")) is True
    assert answers.is_refusal(NO_ANSWER.upper()) is True
    assert answers.is_refusal("Во фрагментах учебника ответа нет.") is True


def test_is_refusal_false_case() -> None:
    assert answers.is_refusal("Ответ есть (с. 290)") is False
