"""Answer eval: grading one answer and loading the item set (spec checks 5-9)."""

import json
from pathlib import Path

import pytest

from fr_drill import answers
from fr_drill.answers import NO_ANSWER, AnswerError, AnswerItem, Grade

ITEM: AnswerItem = AnswerItem(
    id="x",
    q="Как выбрать предлог перед названием страны?",
    expect="answer",
    chapter="21",
    pages=["297"],
    must=[["au Japon"], ["женск*"], ["aux États-Unis"]],
    forbid=["en Japon"],
)
NO_ITEM: AnswerItem = AnswerItem(
    id="o",
    q="Как написать цикл for на Python?",
    expect="no_answer",
    chapter=None,
    pages=[],
    must=[],
    forbid=[],
)
FULL: str = "Говорят au Japon, женского рода, aux États-Unis."


def test_score_partial_answer() -> None:
    g: Grade = answers.score_answer(
        ITEM, "Говорят au Japon (с. 297), страны женского рода (с. 299)", ["297", "298"]
    )
    assert g.facts_hit == 2
    assert g.facts_total == 3
    assert g.missed == [2]
    assert g.score == pytest.approx(2 / 3)
    assert g.passed is False
    assert g.forbidden == []
    assert g.refused is False
    assert g.cited == ["297", "299"]
    assert g.cite_in_context == 0.5
    assert g.cite_expected is True


def test_score_forbidden_form_fails_but_keeps_the_score() -> None:
    g: Grade = answers.score_answer(ITEM, FULL + " Не en Japon.", ["297"])
    assert g.score == 1.0
    assert g.forbidden == ["en Japon"]
    assert g.passed is False


def test_score_clean_full_answer_passes() -> None:
    g: Grade = answers.score_answer(ITEM, FULL, ["297"])
    assert g.score == 1.0
    assert g.passed is True
    assert g.missed == []
    assert g.forbidden == []


def test_score_without_citations() -> None:
    g: Grade = answers.score_answer(ITEM, FULL, ["297"])
    assert g.cited == []
    assert g.cite_in_context is None
    assert g.cite_expected is False


def test_score_forbidden_in_pattern_order() -> None:
    item: AnswerItem = AnswerItem("y", "q", "answer", None, ["1"], [["a"]], ["bb", "aa", "cc"])
    g: Grade = answers.score_answer(item, "a cc aa", [])
    assert g.forbidden == ["aa", "cc"]


def test_score_no_answer_refusal() -> None:
    g: Grade = answers.score_answer(NO_ITEM, NO_ANSWER, ["1"])
    assert g.score == 1.0
    assert g.passed is True
    assert g.refused is True
    assert g.facts_total == 0
    assert g.facts_hit == 0
    assert g.missed == []
    assert g.forbidden == []
    assert g.cite_expected is False


def test_score_no_answer_anything_else() -> None:
    g: Grade = answers.score_answer(NO_ITEM, "for i in range(3): pass (с. 5)", ["1"])
    assert g.score == 0.0
    assert g.passed is False
    assert g.refused is False
    assert g.cited == ["5"]
    assert g.cite_in_context == 0.0


def test_refusal_on_an_answer_item_does_not_change_the_score() -> None:
    g: Grade = answers.score_answer(ITEM, NO_ANSWER + " au Japon", [])
    assert g.refused is True
    assert g.facts_hit == 1
    assert g.score == pytest.approx(1 / 3)


# load_items

EXAMPLE: dict[str, list[dict[str, object]]] = {
    "items": [
        {
            "id": "ch21-seasons",
            "chapter": "21",
            "pages": ["290"],
            "q": "Как сказать «зимой», «летом» и «весной»?",
            "must": [["en hiver"], ["en été"], ["au printemps"]],
        },
        {
            "id": "ch21-countries",
            "chapter": "21",
            "pages": ["297"],
            "q": "Как выбрать предлог перед названием страны?",
            "must": [["женск*"], ["гласн*"], ["согласн*"], ["множеств*"], ["au Japon", "au Brésil"]],
            "forbid": ["en Japon", "au France"],
        },
        {"id": "out-python-loop", "expect": "no_answer", "q": "Как написать цикл for на Python?"},
    ]
}
GOOD_AN: dict[str, object] = {"id": "good", "q": "вопрос", "pages": ["1"], "must": [["a"]]}


def write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_items_example_with_defaults(tmp_path: Path) -> None:
    items: list[AnswerItem] = answers.load_items(write(tmp_path / "a.json", EXAMPLE))
    assert [i.id for i in items] == ["ch21-seasons", "ch21-countries", "out-python-loop"]
    assert items[0] == AnswerItem(
        "ch21-seasons",
        "Как сказать «зимой», «летом» и «весной»?",
        "answer",
        "21",
        ["290"],
        [["en hiver"], ["en été"], ["au printemps"]],
        [],
    )
    assert items[1].forbid == ["en Japon", "au France"]
    assert items[2] == AnswerItem(
        "out-python-loop", "Как написать цикл for на Python?", "no_answer", None, [], [], []
    )


def test_load_items_missing_file(tmp_path: Path) -> None:
    err: pytest.ExceptionInfo[AnswerError]
    with pytest.raises(AnswerError) as err:
        answers.load_items(tmp_path / "nope.json")
    assert "nope.json" in str(err.value)


def test_load_items_broken_json(tmp_path: Path) -> None:
    path: Path = tmp_path / "a.json"
    path.write_text("{not json", encoding="utf-8")
    err: pytest.ExceptionInfo[AnswerError]
    with pytest.raises(AnswerError) as err:
        answers.load_items(path)
    assert "a.json" in str(err.value)


@pytest.mark.parametrize("items", [[], {"a": 1}, "text", None])
def test_load_items_bad_items(tmp_path: Path, items: object) -> None:
    with pytest.raises(AnswerError):
        answers.load_items(write(tmp_path / "a.json", {"items": items}))


def test_load_items_without_items_key(tmp_path: Path) -> None:
    with pytest.raises(AnswerError):
        answers.load_items(write(tmp_path / "a.json", {}))


def test_load_items_duplicate_id_names_the_element(tmp_path: Path) -> None:
    err: pytest.ExceptionInfo[AnswerError]
    with pytest.raises(AnswerError) as err:
        answers.load_items(write(tmp_path / "a.json", {"items": [GOOD_AN, dict(GOOD_AN)]}))
    assert "1" in str(err.value) or "good" in str(err.value)


BAD_SECOND: list[dict[str, object]] = [
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["a"]], "expect": "maybe"},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": []},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [[]]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [[""]]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["ab*c"]]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["*"]]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["one two three four five"]]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["a"]], "forbid": ["x y z w v"]},
    {"id": "b", "q": "вопрос", "pages": ["1"], "must": [["a"]], "forbid": ["*"]},
    {"id": "b", "expect": "no_answer", "q": "вопрос", "must": [["a"]]},
    {"id": "b", "expect": "no_answer", "q": "вопрос", "forbid": ["a"]},
    {"id": "b", "q": "вопрос", "pages": [], "must": [["a"]]},
    {"id": "b", "q": "вопрос", "must": [["a"]]},
    {"id": "b", "q": "", "pages": ["1"], "must": [["a"]]},
    {"id": "", "q": "вопрос", "pages": ["1"], "must": [["a"]]},
    {"q": "вопрос", "pages": ["1"], "must": [["a"]]},
]


@pytest.mark.parametrize("bad", BAD_SECOND)
def test_load_items_names_the_second_element(tmp_path: Path, bad: dict[str, object]) -> None:
    err: pytest.ExceptionInfo[AnswerError]
    with pytest.raises(AnswerError) as err:
        answers.load_items(write(tmp_path / "a.json", {"items": [GOOD_AN, bad]}))
    assert "1" in str(err.value)


def test_load_items_accepts_four_words_and_a_star_stem(tmp_path: Path) -> None:
    item: dict[str, object] = {
        "id": "ok",
        "q": "вопрос",
        "pages": ["1"],
        "must": [["one two three four"], ["stem*"]],
    }
    got: list[AnswerItem] = answers.load_items(write(tmp_path / "a.json", {"items": [item]}))
    assert got[0].must == [["one two three four"], ["stem*"]]


def test_items_sha256_is_of_the_file_bytes(tmp_path: Path) -> None:
    import hashlib

    path: Path = write(tmp_path / "a.json", EXAMPLE)
    assert answers.items_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
