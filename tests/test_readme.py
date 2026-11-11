"""README (spec check 32)."""

import pytest
from util import INVENTORY, ROOT

MUST_HAVE: list[str] = [
    "FR_DRILL_PROGRESS",
    "FR_DRILL_VOCAB",
    "progress/",
    "books/",
    "index/",
    "Apache",
    "Heminway",
    "Eaquals",
    "fr-drill overlap",
    "uv run fr-drill drill",
]


@pytest.mark.parametrize("phrase", MUST_HAVE)
def test_readme_mentions(phrase: str) -> None:
    assert phrase in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_does_not_name_the_unpublished_file() -> None:
    assert INVENTORY not in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_says_there_is_no_open_vocabulary_yet_and_how_to_run() -> None:
    text: str = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv sync" in text and "uv run fr-drill exam --stage 1" in text
    assert "vocab.json" in text
