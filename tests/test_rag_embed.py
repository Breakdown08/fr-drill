"""Textbook helper: the fastembed wrapper (spec checks 15, 16)."""

import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from rag_fakes import REAL

from fr_drill import rag
from fr_drill.rag import FastEmbedder, RagError


class FakeModel:
    """What fastembed's TextEmbedding looks like from outside: embed(list[str]) gives vectors."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def embed(self, texts: Sequence[str]) -> Iterator[tuple[int, ...]]:
        self.calls.append(texts)
        return (tuple(range(i, i + 3)) for i, _ in enumerate(texts))


def test_passages_get_the_passage_prefix() -> None:
    model: FakeModel = FakeModel()
    out: list[list[float]] = FastEmbedder(model=model).embed_passages(["a", "b"])
    assert model.calls == [["passage: a", "passage: b"]]
    assert isinstance(model.calls[0], list)
    assert out == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]


def test_query_gets_the_query_prefix() -> None:
    model: FakeModel = FakeModel()
    out: list[float] = FastEmbedder(model=model).embed_query("q")
    assert model.calls == [["query: q"]]
    assert out == [0.0, 1.0, 2.0]


def test_vectors_are_lists_of_floats() -> None:
    emb: FastEmbedder = FastEmbedder(model=FakeModel())
    vecs: list[list[float]] = emb.embed_passages(["a"])
    assert type(vecs) is list and type(vecs[0]) is list
    assert all(type(x) is float for x in vecs[0])
    q: list[float] = emb.embed_query("q")
    assert type(q) is list and all(type(x) is float for x in q)


def test_dim_is_1024() -> None:
    assert FastEmbedder(model=FakeModel()).dim == 1024
    assert rag.EMBED_DIM == 1024


def test_load_fastembed_without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        rag.load_fastembed()
    assert "uv sync --extra rag" in str(err.value)


def test_load_fastembed_returns_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: types.ModuleType = types.ModuleType("fastembed")
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    assert rag.load_fastembed() is fake


def test_the_model_is_built_from_the_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    made: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class TextEmbedding(FakeModel):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            made.append((args, kwargs))

    fake: types.ModuleType = types.ModuleType("fastembed")
    fake.__dict__["TextEmbedding"] = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    emb: FastEmbedder = FastEmbedder(cache_dir=tmp_path / "models", threads=3)
    assert len(made) == 1
    args: tuple[Any, ...] = made[0][0]
    kwargs: dict[str, Any] = made[0][1]
    assert args == ("intfloat/multilingual-e5-large",)
    assert kwargs == {"cache_dir": str(tmp_path / "models"), "threads": 3}
    assert emb.embed_query("q") == [0.0, 1.0, 2.0]


def test_make_embedder_gives_a_fastembed_embedder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class TextEmbedding(FakeModel):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    fake: types.ModuleType = types.ModuleType("fastembed")
    fake.__dict__["TextEmbedding"] = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    assert isinstance(REAL["make_embedder"](tmp_path), FastEmbedder)
