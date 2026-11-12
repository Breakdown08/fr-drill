"""Textbook helper: the vector store, indexing and search (spec checks 17-24)."""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag_fakes import FakeEmbedder, mk

from fr_drill import rag
from fr_drill.rag import COLLECTION, Chunk, Hit, RagError

WORDS: list[str] = [
    "alpha bravo charlie delta",
    "echo foxtrot golf hotel",
    "india juliet kilo lima",
    "mike november oscar papa",
    "quebec romeo sierra tango",
    "uniform victor whiskey xray",
]


def six() -> list[Chunk]:
    return [mk(pdf_page=1 + i // 2, chunk=i % 2, text=WORDS[i], chapter="1 Intro") for i in range(6)]


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    c: QdrantClient = QdrantClient(":memory:")
    yield c
    c.close()


def payloads(client: QdrantClient, collection: str = COLLECTION) -> list[dict[str, Any]]:
    points: Any
    cursor: Any
    points, cursor = client.scroll(collection, limit=1000, with_payload=True)
    return [dict(p.payload or {}) for p in points]


def test_index_writes_every_chunk_with_the_payload(client: QdrantClient) -> None:
    assert rag.index_chunks(six(), client, FakeEmbedder()) == 6
    assert client.count(COLLECTION).count == 6
    rows: list[dict[str, Any]] = payloads(client)
    assert all(set(r) == {"book", "pdf_page", "page", "chapter", "chunk", "text"} for r in rows)
    assert {r["text"] for r in rows} == set(WORDS)


def test_collection_is_created_with_cosine_and_the_embedder_size(client: QdrantClient) -> None:
    rag.index_chunks(six(), client, FakeEmbedder(dim=64))
    params: Any = client.get_collection(COLLECTION).config.params.vectors
    assert params.size == 64
    assert params.distance == Distance.COSINE


def test_a_custom_collection_name(client: QdrantClient) -> None:
    rag.index_chunks(six(), client, FakeEmbedder(), collection="other")
    assert client.count("other").count == 6
    assert not client.collection_exists(COLLECTION)


def test_indexing_twice_keeps_six_points(client: QdrantClient) -> None:
    rag.index_chunks(six(), client, FakeEmbedder())
    assert rag.index_chunks(six(), client, FakeEmbedder()) == 6
    assert client.count(COLLECTION).count == 6


def test_empty_input_touches_nothing(client: QdrantClient) -> None:
    assert rag.index_chunks([], client, FakeEmbedder()) == 0
    assert not client.collection_exists(COLLECTION)


def test_passages_are_embedded_in_batches(client: QdrantClient) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    chunks: list[Chunk] = [mk(pdf_page=1, chunk=i, text=f"mot{i}") for i in range(7)]
    rag.index_chunks(chunks, client, emb, batch=3)
    assert [len(c) for c in emb.passage_calls] == [3, 3, 1]
    assert [t for call in emb.passage_calls for t in call] == [c.text for c in chunks]


def test_reindexing_pages_replaces_their_old_chunks(client: QdrantClient) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    old: list[Chunk] = [
        mk("a", 1, 0, "a1c0"),
        mk("a", 1, 1, "a1c1"),
        mk("a", 2, 0, "a2c0"),
        mk("a", 2, 1, "a2c1"),
        mk("a", 2, 2, "a2c2"),
        mk("a", 3, 0, "a3c0"),
        mk("a", 3, 1, "a3c1"),
        mk("a", 3, 2, "a3c2"),
        mk("b", 2, 0, "b2c0"),
        mk("b", 3, 0, "b3c0"),
    ]
    rag.index_chunks(old, client, emb)
    assert rag.index_chunks([mk("a", 2, 0, "a2new"), mk("a", 3, 0, "a3new")], client, emb) == 2
    assert {r["text"] for r in payloads(client)} == {"a1c0", "a1c1", "a2new", "a3new", "b2c0", "b3c0"}


def test_point_id_is_deterministic_and_distinguishes_the_fields() -> None:
    base: Chunk = mk("a", 2, 1, "x")
    assert rag.point_id(base) == rag.point_id(mk("a", 2, 1, "other text"))
    ids: set[str] = {
        rag.point_id(base),
        rag.point_id(mk("b", 2, 1, "x")),
        rag.point_id(mk("a", 3, 1, "x")),
        rag.point_id(mk("a", 2, 2, "x")),
    }
    assert len(ids) == 4


def test_point_id_follows_the_uuid5_rule() -> None:
    base: Chunk = mk("heminway", 308, 4, "x")
    assert rag.point_id(base) == str(uuid.uuid5(uuid.NAMESPACE_URL, "heminway/308/4"))


def test_another_vector_size_is_an_error(client: QdrantClient) -> None:
    rag.index_chunks(six(), client, FakeEmbedder(dim=64))
    with pytest.raises(RagError):
        rag.index_chunks(six(), client, FakeEmbedder(dim=32))
    with pytest.raises(RagError):
        rag.search("alpha", client, FakeEmbedder(dim=32))
    assert client.count(COLLECTION).count == 6


def test_search_puts_the_matching_chunk_first(client: QdrantClient) -> None:
    chunks: list[Chunk] = six()
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(chunks, client, emb)
    i: int
    for i in range(6):
        hits: list[Hit] = rag.search(WORDS[i], client, emb, k=3)
        assert hits[0].chunk == chunks[i]


def test_search_respects_k_and_orders_by_score(client: QdrantClient) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(six(), client, emb)
    hits: list[Hit] = rag.search(WORDS[2], client, emb, k=3)
    assert len(hits) == 3
    assert all(isinstance(h, Hit) and isinstance(h.chunk, Chunk) for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(rag.search(WORDS[2], client, emb, k=50)) == 6
    assert len(rag.search(WORDS[2], client, emb)) == rag.K == 5


def test_search_embeds_the_query_once(client: QdrantClient) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(six(), client, emb)
    rag.search("alpha bravo", client, emb)
    assert emb.query_calls == ["alpha bravo"]


def test_search_without_a_collection_points_to_index(client: QdrantClient) -> None:
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.search("alpha", client, FakeEmbedder())
    assert "fr-drill index" in str(err.value)


def test_search_needs_a_positive_k(client: QdrantClient) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(six(), client, emb)
    with pytest.raises(ValueError):
        rag.search("alpha", client, emb, k=0)


def test_search_on_an_empty_collection_is_empty(client: QdrantClient) -> None:
    client.create_collection(COLLECTION, VectorParams(size=64, distance=Distance.COSINE))
    assert rag.search("alpha", client, FakeEmbedder()) == []


def test_open_client_creates_the_folder_and_keeps_data(tmp_path: Path) -> None:
    emb: FakeEmbedder = FakeEmbedder()
    first: QdrantClient = rag.open_client(tmp_path)
    assert (tmp_path / "qdrant").is_dir()
    rag.index_chunks(six(), first, emb)
    first.close()
    second: QdrantClient = rag.open_client(tmp_path)
    try:
        assert second.count(COLLECTION).count == 6
        assert rag.search(WORDS[1], second, emb, k=1)[0].chunk == six()[1]
    finally:
        second.close()


def test_open_client_makes_missing_parents(tmp_path: Path) -> None:
    c: QdrantClient = rag.open_client(tmp_path / "not" / "yet")
    c.close()
    assert (tmp_path / "not" / "yet" / "qdrant").is_dir()
