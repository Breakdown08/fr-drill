"""Textbook helper: citations, prompt, ask, generators (spec checks 25-33)."""

import json
import socket
import stat
import sys
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

from fr_drill import rag
from fr_drill.rag import (
    COLLECTION,
    NO_ANSWER,
    Answer,
    Chunk,
    ClaudeGenerator,
    Hit,
    OllamaGenerator,
    RagError,
)

# cite and build_prompt


def test_cite_with_and_without_a_chapter() -> None:
    assert rag.cite(mk("heminway", 400, 0, "x", "21 Prepositions", page="308")) == (
        "heminway, p. 308, 21 Prepositions"
    )
    assert rag.cite(mk("heminway", 400, 0, "x", None, page="308")) == "heminway, p. 308"


def test_prompt_carries_question_fragments_and_rules() -> None:
    first: Chunk = mk("heminway", 400, 0, "C'est or il est? Premier fragment.", "21 Prepositions", page="308")
    second: Chunk = mk("heminway", 401, 1, "Deuxieme fragment.", None, page="309")
    question: str = "Почему c'est, а не il est?"
    prompt: str = rag.build_prompt(question, [Hit(first, 0.9), Hit(second, 0.5)])
    assert question in prompt
    block1: str = f"[1] {rag.cite(first)}\n{first.text}"
    block2: str = f"[2] {rag.cite(second)}\n{second.text}"
    assert block1 in prompt and block2 in prompt
    assert prompt.index(block1) < prompt.index(block2)
    assert NO_ANSWER in prompt
    assert "(с. " in prompt


def test_prompt_is_in_russian() -> None:
    prompt: str = rag.build_prompt("q", [Hit(mk(), 0.5)])
    cyrillic: int = sum(1 for c in prompt if "а" <= c.lower() <= "я")
    assert cyrillic > 100


# ask


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    c: QdrantClient = QdrantClient(":memory:")
    yield c
    c.close()


def test_ask_calls_the_generator_once_with_the_prompt(client: QdrantClient) -> None:
    chunks: list[Chunk] = [
        mk(pdf_page=1, text="alpha bravo charlie"),
        mk(pdf_page=2, text="delta echo foxtrot"),
    ]
    emb: FakeEmbedder = FakeEmbedder()
    rag.index_chunks(chunks, client, emb)
    gen: FakeGenerator = FakeGenerator("  Ответ (с. 1).\n\n")
    question: str = "alpha bravo"
    answer: Answer = rag.ask(question, client, emb, gen, k=2)
    hits: list[Hit] = rag.search(question, client, emb, k=2)
    assert gen.prompts == [rag.build_prompt(question, answer.hits)]
    assert answer.text == "Ответ (с. 1)."
    assert [h.chunk for h in answer.hits] == [h.chunk for h in hits]
    assert answer.hits[0].chunk == chunks[0]


def test_ask_on_an_empty_collection_does_not_call_the_generator(client: QdrantClient) -> None:
    client.create_collection(COLLECTION, VectorParams(size=64, distance=Distance.COSINE))
    gen: FakeGenerator = FakeGenerator()
    assert rag.ask("alpha", client, FakeEmbedder(), gen) == Answer(NO_ANSWER, [])
    assert gen.prompts == []


def test_ask_without_an_index_is_a_rag_error(client: QdrantClient) -> None:
    gen: FakeGenerator = FakeGenerator()
    with pytest.raises(RagError):
        rag.ask("alpha", client, FakeEmbedder(), gen)
    assert gen.prompts == []


# OllamaGenerator


class Seen:
    """What the stub server received."""

    def __init__(self) -> None:
        self.method: str = ""
        self.path: str = ""
        self.body: bytes = b""


def stub(seen: Seen, status: int, payload: bytes) -> AbstractContextManager[int]:
    def make(port: int) -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                seen.method = "POST"
                seen.path = self.path
                seen.body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler

    return serving(make)


def test_ollama_posts_to_api_generate() -> None:
    seen: Seen = Seen()
    reply: bytes = json.dumps({"response": "  Bonjour, мир \n"}).encode()
    port: int
    with stub(seen, 200, reply) as port:
        out: str = OllamaGenerator(model="m1", url=f"http://127.0.0.1:{port}").generate("мой промпт")
    assert out == "Bonjour, мир"
    assert seen.method == "POST"
    assert seen.path == "/api/generate"
    body: dict[str, Any] = json.loads(seen.body)
    assert body["model"] == "m1"
    assert body["prompt"] == "мой промпт"
    assert body["stream"] is False
    assert body["options"]["num_ctx"] == 8192


def test_ollama_default_model_is_sent() -> None:
    seen: Seen = Seen()
    port: int
    with stub(seen, 200, b'{"response": "ok"}') as port:
        assert OllamaGenerator(url=f"http://127.0.0.1:{port}").generate("p") == "ok"
    assert json.loads(seen.body)["model"] == rag.OLLAMA_MODEL


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (500, b'{"error": "boom"}'),
        (404, b'{"error": "model not found"}'),
        (200, b'{"done": true}'),
        (200, b'{"response": 42}'),
        (200, b"this is not json"),
    ],
)
def test_ollama_bad_answers_are_rag_errors(status: int, payload: bytes) -> None:
    port: int
    with stub(Seen(), status, payload) as port:
        url: str = f"http://127.0.0.1:{port}"
        err: pytest.ExceptionInfo[RagError]
        with pytest.raises(RagError) as err:
            OllamaGenerator(url=url).generate("p")
    assert url in str(err.value)


def test_ollama_closed_port_is_a_rag_error() -> None:
    s: socket.socket = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    url: str = f"http://127.0.0.1:{port}"
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        OllamaGenerator(url=url, timeout=5).generate("p")
    assert url in str(err.value)


# ClaudeGenerator


def script(tmp_path: Path, body: str) -> Path:
    path: Path = tmp_path / "fake-claude"
    path.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_claude_gets_the_prompt_on_stdin(tmp_path: Path) -> None:
    record: Path = tmp_path / "record.json"
    cmd: Path = script(
        tmp_path,
        f"json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}, open({str(record)!r}, 'w'))\n"
        "print('  réponse \\n')",
    )
    out: str = ClaudeGenerator(command=str(cmd)).generate("мой промпт\nвторая строка")
    assert out == "réponse"
    seen: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
    assert seen["argv"] == ["-p", "--model", "haiku"]
    assert seen["stdin"] == "мой промпт\nвторая строка"


def test_claude_model_is_passed(tmp_path: Path) -> None:
    record: Path = tmp_path / "argv.json"
    cmd: Path = script(tmp_path, f"json.dump(sys.argv[1:], open({str(record)!r}, 'w'))\nprint('x')")
    ClaudeGenerator(model="sonnet", command=str(cmd)).generate("p")
    assert json.loads(record.read_text(encoding="utf-8")) == ["-p", "--model", "sonnet"]


def test_claude_failure_carries_stderr(tmp_path: Path) -> None:
    cmd: Path = script(tmp_path, "sys.stderr.write('boom: no auth')\nsys.exit(1)")
    err: pytest.ExceptionInfo[RagError]
    with pytest.raises(RagError) as err:
        ClaudeGenerator(command=str(cmd)).generate("p")
    assert "boom: no auth" in str(err.value)


def test_claude_missing_command(tmp_path: Path) -> None:
    with pytest.raises(RagError):
        ClaudeGenerator(command=str(tmp_path / "no-such-claude")).generate("p")


def test_claude_timeout(tmp_path: Path) -> None:
    cmd: Path = script(tmp_path, "import time\ntime.sleep(10)")
    with pytest.raises(RagError):
        ClaudeGenerator(command=str(cmd), timeout=0.5).generate("p")


# make_generator


def test_make_generator_builds_the_right_class(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class FakeOllama:
        def __init__(self, *args: object, **kwargs: object) -> None:
            made.append(("ollama", args, kwargs))

    class FakeClaude:
        def __init__(self, *args: object, **kwargs: object) -> None:
            made.append(("claude", args, kwargs))

    monkeypatch.setattr(rag, "OllamaGenerator", FakeOllama)
    monkeypatch.setattr(rag, "ClaudeGenerator", FakeClaude)
    REAL["make_generator"]("ollama")
    REAL["make_generator"]("ollama", "qwen")
    REAL["make_generator"]("claude", "sonnet")
    REAL["make_generator"]("claude")
    flat: list[tuple[str, list[Any]]] = [(n, [*a, *k.values()]) for n, a, k in made]
    assert flat == [
        ("ollama", [rag.OLLAMA_MODEL]),
        ("ollama", ["qwen"]),
        ("claude", ["sonnet"]),
        ("claude", [rag.CLAUDE_MODEL]),
    ]


def test_make_generator_real_classes() -> None:
    assert isinstance(REAL["make_generator"]("ollama"), OllamaGenerator)
    assert isinstance(REAL["make_generator"]("claude", "sonnet"), ClaudeGenerator)


def test_make_generator_unknown_backend() -> None:
    with pytest.raises(ValueError):
        REAL["make_generator"]("x")
