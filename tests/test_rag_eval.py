"""Textbook helper: the eval set and recall@k (spec checks 34-36)."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from rag_fakes import FakeEmbedder, mk

from fr_drill import rag
from fr_drill.rag import EvalItem, EvalReport, EvalRow, RagError

# recall_at_k


def test_recall_counts_expected_pages_in_the_top_k() -> None:
    assert rag.recall_at_k(["1", "2"], ["2", "5", "1"], 1) == 0.5
    assert rag.recall_at_k(["1", "2"], ["2", "5", "1"], 3) == 1.0
    assert rag.recall_at_k(["1"], ["2"], 5) == 0.0


def test_recall_ignores_duplicates() -> None:
    assert rag.recall_at_k(["1", "1", "2"], ["1", "1", "1"], 3) == 0.5


def test_recall_bad_arguments() -> None:
    with pytest.raises(ValueError):
        rag.recall_at_k([], ["1"], 3)
    with pytest.raises(ValueError):
        rag.recall_at_k(["1"], ["1"], 0)


# load_eval

EXAMPLE: dict[str, list[dict[str, object]]] = {
    "items": [
        {"q": "Почему c'est, а не il est?", "pages": ["308"]},
        {"q": "Когда à, а когда en перед названием страны?", "pages": ["290", "291"]},
    ]
}


def write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_eval_reads_the_example(tmp_path: Path) -> None:
    items: list[EvalItem] = rag.load_eval(write(tmp_path / "e.json", EXAMPLE))
    assert items == [
        EvalItem("Почему c'est, а не il est?", ["308"]),
        EvalItem("Когда à, а когда en перед названием страны?", ["290", "291"]),
    ]


def test_load_eval_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RagError):
        rag.load_eval(tmp_path / "nope.json")


def test_load_eval_broken_json(tmp_path: Path) -> None:
    path: Path = tmp_path / "e.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RagError):
        rag.load_eval(path)


@pytest.mark.parametrize("items", [[], {"a": 1}, "text", None])
def test_load_eval_bad_items(tmp_path: Path, items: object) -> None:
    with pytest.raises(RagError):
        rag.load_eval(write(tmp_path / "e.json", {"items": items}))


def test_load_eval_without_items_key(tmp_path: Path) -> None:
    with pytest.raises(RagError):
        rag.load_eval(write(tmp_path / "e.json", {}))


GOOD: dict[str, object] = {"q": "вопрос", "pages": ["1"]}


@pytest.mark.parametrize(
    "bad",
    [
        {"pages": ["1"]},
        {"q": "", "pages": ["1"]},
        {"q": 5, "pages": ["1"]},
        {"q": "вопрос"},
        {"q": "вопрос", "pages": []},
        {"q": "вопрос", "pages": [308]},
        {"q": "вопрос", "pages": "308"},
        {"q": "вопрос", "pages": ["1", 2]},
    ],
)
def test_load_eval_names_the_bad_element(tmp_path: Path, bad: dict[str, object]) -> None:
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.load_eval(write(tmp_path / "e.json", {"items": [GOOD, bad]}))
    assert "1" in str(err.value)


def test_load_eval_names_the_first_element_too(tmp_path: Path) -> None:
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.load_eval(write(tmp_path / "e.json", {"items": [{"q": "вопрос", "pages": []}, GOOD]}))
    assert "0" in str(err.value)


# evaluate


@pytest.fixture
def indexed() -> Iterator[tuple[QdrantClient, FakeEmbedder]]:
    client: QdrantClient = QdrantClient(":memory:")
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(
        [
            mk(pdf_page=10, page="10", text="alpha bravo charlie delta"),
            mk(pdf_page=11, page="11", text="echo foxtrot golf hotel"),
            mk(pdf_page=12, page="12", text="india juliet kilo lima"),
            mk(pdf_page=13, page="13", text="mike november oscar papa"),
            mk(pdf_page=14, page="14", text="quebec romeo sierra tango"),
        ],
        client,
        emb,
    )
    yield client, emb
    client.close()


def test_evaluate_matches_the_hand_calculation(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed
    items: list[EvalItem] = [
        EvalItem("india juliet kilo lima", ["12", "99"]),
        EvalItem("quebec romeo sierra tango", ["14"]),
        EvalItem("echo foxtrot golf hotel", ["zz"]),
    ]
    report: EvalReport = rag.evaluate(items, client, emb, k=2)
    assert report.k == 2
    assert [r.q for r in report.rows] == [i.q for i in items]
    assert [r.expected for r in report.rows] == [["12", "99"], ["14"], ["zz"]]
    row: EvalRow
    item: EvalItem
    for row, item in zip(report.rows, items, strict=True):
        assert row.retrieved == [h.chunk.page for h in rag.search(item.q, client, emb, k=2)]
        assert len(row.retrieved) == 2
    assert report.rows[0].retrieved[0] == "12"
    assert [r.recall for r in report.rows] == [0.5, 1.0, 0.0]
    assert [r.hit for r in report.rows] == [True, True, False]
    assert report.recall == pytest.approx(0.5)
    assert report.hit_rate == pytest.approx(2 / 3)


def test_evaluate_empty_items(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    with pytest.raises(ValueError):
        rag.evaluate([], indexed[0], indexed[1])


def test_evaluate_without_an_index() -> None:
    client: QdrantClient = QdrantClient(":memory:")
    with pytest.raises(RagError):
        rag.evaluate([EvalItem("q", ["1"])], client, FakeEmbedder())
    client.close()
