"""The data files that ship with the code: what is there, what was cleaned (spec checks 22, 24)."""

import json
from pathlib import Path
from typing import Any

import pytest
from util import DATA, INVENTORY

DRILLS: list[str] = [
    "a-de",
    "a-de-verbs",
    "articles",
    "etre-avoir",
    "gender",
    "negation-er",
    "numbers",
    "pronouns",
    "sentences",
    "verbs-er",
]
NUMBER_WORDS: list[tuple[str, str]] = [
    ("zéro", "ноль"),
    ("un", "один"),
    ("deux", "два"),
    ("trois", "три"),
    ("quatre", "четыре"),
    ("cinq", "пять"),
    ("six", "шесть"),
    ("sept", "семь"),
    ("huit", "восемь"),
    ("neuf", "девять"),
    ("dix", "десять"),
    ("onze", "одиннадцать"),
    ("douze", "двенадцать"),
    ("treize", "тринадцать"),
    ("quatorze", "четырнадцать"),
    ("quinze", "пятнадцать"),
    ("seize", "шестнадцать"),
    ("dix-sept", "семнадцать"),
    ("dix-huit", "восемнадцать"),
    ("dix-neuf", "девятнадцать"),
    ("vingt", "двадцать"),
    ("trente", "тридцать"),
    ("quarante", "сорок"),
    ("cinquante", "пятьдесят"),
    ("soixante", "шестьдесят"),
    ("soixante-dix", "семьдесят"),
    ("quatre-vingts", "восемьдесят"),
    ("quatre-vingt-dix", "девяносто"),
    ("cent", "сто"),
    ("mille", "тысяча"),
]


def load(name: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((DATA / "drills" / f"{name}.json").read_text(encoding="utf-8"))
    return doc


@pytest.mark.parametrize("name", DRILLS)
def test_drill_file_is_there_and_is_json(name: str) -> None:
    assert isinstance(load(name), dict)


def test_bank_files() -> None:
    names: set[str] = {p.name for p in (DATA / "bank").iterdir()}
    assert "format.md" in names and "consignes.json" in names
    assert len([n for n in names if n.endswith(".json") and n != "consignes.json"]) == 10


def test_numbers_file_holds_the_thirty_base_words() -> None:
    doc: dict[str, Any] = load("numbers")
    assert doc["title"] == "Числа: опорные слова"
    assert doc["note"] == "Слова, которые правило to_fr обязано дать дословно."
    assert [(w["fr"], w["ru"]) for w in doc["words"]] == NUMBER_WORDS


def test_the_inventory_is_not_published() -> None:
    assert not (DATA / "reference" / f"{INVENTORY}.md").exists()
    assert (DATA / "reference" / "delf-a1.md").is_file()


def test_no_data_file_names_the_inventory_file() -> None:
    hits: list[str] = []
    p: Path
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.name != "vocab.json" and INVENTORY in p.read_text(encoding="utf-8"):
            hits.append(str(p.relative_to(DATA)))
    assert hits == []
    assert (DATA / "bank" / "format.md").is_file()


def test_delf_reference_is_cleaned() -> None:
    text: str = (DATA / "reference" / "delf-a1.md").read_text(encoding="utf-8")
    assert "[[" not in text
    assert "Eaquals" in text


def test_gender_note_points_at_the_new_vocabulary_name() -> None:
    note: str = load("gender")["note"]
    assert "vocab.json" in note and "vocab-a1.json" not in note


def test_prepositions_verbs_note() -> None:
    note: str = load("a-de-verbs")["note"]
    assert "из verbs-er.json и вкладки неправильных глаголов" in note
    assert "Кир" + "илл" not in note
    assert "irgol.ru" in note


def test_the_two_phrases_found_in_the_textbook_are_rewritten() -> None:
    name: str
    for name in ("articles", "gender"):
        text: str = json.dumps(load(name), ensure_ascii=False).lower().replace("’", "'")
        assert "ce n'est pas du fromage" not in text, name
        assert "les enfants jouent dans le jardin" not in text, name


def test_the_textbook_is_cited_by_chapter_only() -> None:
    assert "Heminway" in load("articles")["source"] and "гл." in load("articles")["source"]
    assert "Heminway" in load("gender")["source"] and "гл." in load("gender")["source"]
