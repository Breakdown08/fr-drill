"""Answer eval in rag: timed generation, generators, answer_eval, load_eval (spec checks 16-21, 33)."""

import json
from collections.abc import Iterator
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag_fakes import REAL, FakeEmbedder, FakeGenerator, mk
from util import serving

from fr_drill import answers, cli, rag
from fr_drill.answers import NO_ANSWER, AnswerItem, AnswerRow
from fr_drill.rag import ClaudeGenerator, Generation, Generator, Hit, OllamaGenerator, RagError

# timed_generate


class Clock:
    def __init__(self, values: list[float]) -> None:
        self.values: Iterator[float] = iter(values)
        self.calls: int = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.values)


class Metered:
    def __init__(self) -> None:
        self.plain_calls: int = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.plain_calls += 1
        return "plain"

    def generate_metered(self, prompt: str) -> tuple[str, dict[str, float]]:
        self.prompts.append(prompt)
        return " t ", {"completion_tokens": 12.0}


def test_timed_generate_plain_generator() -> None:
    clock: Clock = Clock([10.0, 12.5])
    g: Generation = rag.timed_generate(FakeGenerator("  ответ "), "p", clock)
    assert g == Generation("ответ", 2.5, {})
    assert clock.calls == 2


def test_timed_generate_metered_generator() -> None:
    m: Metered = Metered()
    g: Generation = rag.timed_generate(m, "prompt", Clock([0.0, 1.0]))
    assert g.text == "t"
    assert g.stats == {"completion_tokens": 12.0}
    assert g.seconds == 1.0
    assert m.plain_calls == 0
    assert m.prompts == ["prompt"]


# generators


class Seen:
    def __init__(self) -> None:
        self.body: bytes = b""


def stub(seen: Seen) -> AbstractContextManager[int]:
    def make(port: int) -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                seen.body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload: bytes = b'{"response": "ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler

    return serving(make)


def test_ollama_sends_temperature_and_seed() -> None:
    seen: Seen = Seen()
    port: int
    with stub(seen) as port:
        gen: OllamaGenerator = OllamaGenerator(url=f"http://127.0.0.1:{port}", temperature=0.0, seed=0)
        assert gen.generate("p") == "ok"
    assert json.loads(seen.body)["options"] == {"num_ctx": 8192, "temperature": 0.0, "seed": 0}
    assert gen.options == {"num_ctx": 8192, "temperature": 0.0, "seed": 0}


def test_ollama_without_the_new_arguments_keeps_the_old_body() -> None:
    seen: Seen = Seen()
    port: int
    with stub(seen) as port:
        gen: OllamaGenerator = OllamaGenerator(url=f"http://127.0.0.1:{port}")
        gen.generate("p")
    assert json.loads(seen.body)["options"] == {"num_ctx": 8192}
    assert gen.options == {"num_ctx": 8192}
    assert ClaudeGenerator().options == {}


def test_ollama_zero_seed_and_temperature_are_not_dropped_as_falsy() -> None:
    gen: OllamaGenerator = OllamaGenerator(temperature=0.0)
    assert gen.options == {"num_ctx": 8192, "temperature": 0.0}
    gen2: OllamaGenerator = OllamaGenerator(seed=0)
    assert gen2.options == {"num_ctx": 8192, "seed": 0}


def test_make_generator_deterministic() -> None:
    make: Any = REAL["make_generator"]
    g: Generator = make("ollama", deterministic=True)
    assert isinstance(g, OllamaGenerator)
    assert g.options["temperature"] == 0.0
    assert g.options["seed"] == 0
    assert g.model == rag.OLLAMA_MODEL
    named: Generator = make("ollama", "qwen", deterministic=True)
    assert isinstance(named, OllamaGenerator)
    assert named.model == "qwen"
    assert named.options["seed"] == rag.EVAL_SEED
    c: Generator = make("claude", deterministic=True)
    assert isinstance(c, ClaudeGenerator)
    assert c.options == {}


def test_make_generator_is_not_deterministic_by_default() -> None:
    g: Generator = REAL["make_generator"]("ollama")
    assert isinstance(g, OllamaGenerator)
    assert g.options == {"num_ctx": 8192}


def test_backend_lists_agree() -> None:
    assert cli.BACKENDS == list(rag.DEFAULT_MODEL)
    assert rag.DEFAULT_MODEL == {"ollama": rag.OLLAMA_MODEL, "claude": rag.CLAUDE_MODEL}


# answer_eval

ITEM_1: AnswerItem = AnswerItem("one", "alpha bravo charlie delta", "answer", "21", ["10"], [["ответ"]], [])
ITEM_2: AnswerItem = AnswerItem("two", "india juliet kilo lima", "answer", "21", ["12"], [["нету"]], [])
ITEM_N: AnswerItem = AnswerItem("out", "quebec romeo sierra tango", "no_answer", None, [], [], [])


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
        ],
        client,
        emb,
    )
    yield client, emb
    client.close()


def test_answer_eval_prompts_retrieval_clock_grade_progress(
    indexed: tuple[QdrantClient, FakeEmbedder],
) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed
    gen: FakeGenerator = FakeGenerator("  ответ (с. 10) \n")
    seen: list[AnswerRow] = []
    clock: Clock = Clock([0, 1, 3, 6, 10, 15, 21, 28])
    rows: list[AnswerRow] = rag.answer_eval(
        [ITEM_1, ITEM_2], client, emb, gen, k=2, clock=clock, progress=seen.append
    )
    item: AnswerItem
    want_prompts: list[str] = []
    want_pages: list[list[str]] = []
    for item in (ITEM_1, ITEM_2):
        hits: list[Hit] = rag.search(item.q, client, emb, k=2)
        want_prompts.append(rag.build_prompt(item.q, hits))
        want_pages.append([h.chunk.page for h in hits])
    assert gen.prompts == want_prompts
    assert [r.retrieved for r in rows] == want_pages
    assert [r.retrieve_s for r in rows] == [1, 5]
    assert [r.generate_s for r in rows] == [3, 7]
    assert clock.calls == 8
    assert [r.id for r in rows] == ["one", "two"]
    assert [r.answer for r in rows] == ["ответ (с. 10)"] * 2
    assert rows[0].q == ITEM_1.q and rows[0].expect == "answer"
    assert rows[0].stats == {}
    r: AnswerRow
    i: int
    for i, r in enumerate(rows):
        assert r.grade == answers.score_answer((ITEM_1, ITEM_2)[i], r.answer, r.retrieved)
    assert seen == rows


def test_answer_eval_passes_metered_stats(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed
    rows: list[AnswerRow] = rag.answer_eval([ITEM_1], client, emb, Metered())
    assert rows[0].stats == {"completion_tokens": 12.0}
    assert rows[0].answer == "t"


def test_answer_eval_empty_collection_refuses_without_the_generator() -> None:
    client: QdrantClient = QdrantClient(":memory:")
    client.create_collection(rag.COLLECTION, VectorParams(size=64, distance=Distance.COSINE))
    gen: FakeGenerator = FakeGenerator()
    clock: Clock = Clock([0, 2, 5, 9])
    rows: list[AnswerRow] = rag.answer_eval([ITEM_N, ITEM_1], client, FakeEmbedder(64), gen, clock=clock)
    assert [r.answer for r in rows] == [NO_ANSWER, NO_ANSWER]
    assert [r.generate_s for r in rows] == [0.0, 0.0]
    assert [r.retrieve_s for r in rows] == [2, 4]
    assert [r.retrieved for r in rows] == [[], []]
    assert rows[0].stats == {}
    assert gen.prompts == []
    assert rows[0].grade.passed is True
    assert rows[1].grade.passed is False
    assert clock.calls == 4
    client.close()


def test_answer_eval_without_items_is_an_error(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed
    with pytest.raises(ValueError):
        rag.answer_eval([], client, emb, FakeGenerator())


def test_answer_eval_does_not_swallow_generator_errors(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed

    class Boom:
        def generate(self, prompt: str) -> str:
            raise RagError("boom")

    with pytest.raises(RagError):
        rag.answer_eval([ITEM_1], client, emb, Boom())


def test_answer_eval_on_a_missing_collection_raises(indexed: tuple[QdrantClient, FakeEmbedder]) -> None:
    client: QdrantClient
    emb: FakeEmbedder
    client, emb = indexed
    with pytest.raises(RagError):
        rag.answer_eval([ITEM_1], client, emb, FakeGenerator(), collection="nope")


# load_eval


def write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_eval_skips_no_answer_items(tmp_path: Path) -> None:
    data: dict[str, list[dict[str, object]]] = {
        "items": [
            {"id": "a", "q": "один", "pages": ["1"], "must": [["x"]]},
            {"id": "o", "expect": "no_answer", "q": "вне"},
            {"id": "b", "q": "два", "pages": ["2"], "expect": "answer"},
        ]
    }
    got: list[rag.EvalItem] = rag.load_eval(write(tmp_path / "a.json", data))
    assert got == [rag.EvalItem("один", ["1"]), rag.EvalItem("два", ["2"])]


def test_load_eval_with_only_no_answer_items_is_an_error(tmp_path: Path) -> None:
    data: dict[str, list[dict[str, object]]] = {"items": [{"id": "o", "expect": "no_answer", "q": "вне"}]}
    with pytest.raises(RagError):
        rag.load_eval(write(tmp_path / "a.json", data))


def test_load_eval_skips_no_answer_even_when_fields_are_present(tmp_path: Path) -> None:
    data: dict[str, list[dict[str, object]]] = {
        "items": [{"id": "o", "expect": "no_answer", "q": "вне", "pages": ["1"]}]
    }
    with pytest.raises(RagError):
        rag.load_eval(write(tmp_path / "a.json", data))


def test_no_answer_is_the_same_object() -> None:
    assert rag.NO_ANSWER is answers.NO_ANSWER


def test_load_eval_reads_the_answer_set_as_an_eval_set() -> None:
    n: int = sum(1 for i in answers.load_items(answers.ANSWERS) if i.expect == "answer")
    assert len(rag.load_eval(answers.ANSWERS)) == n
