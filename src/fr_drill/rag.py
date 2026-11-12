"""Textbook helper: index, search, ask, eval."""

import dataclasses
import importlib
import json
import logging
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from fr_drill import DATA, ROOT

INDEX: Path = ROOT / "index"
EVAL: Path = DATA / "rag" / "eval.json"
COLLECTION: str = "textbooks"
EMBED_MODEL: str = "intfloat/multilingual-e5-large"
EMBED_DIM: int = 1024
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
K: int = 5
OLLAMA_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "gemma3:4b"
OLLAMA_NUM_CTX: int = 8192
CLAUDE_MODEL: str = "haiku"
NO_ANSWER: str = "В найденных фрагментах учебника ответа нет."


class RagError(Exception):
    """Any expected error: bad input, no index, backend unavailable."""


@dataclass(frozen=True)
class Page:
    pdf_page: int
    label: str
    chapter: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    book: str
    pdf_page: int
    page: str
    chapter: str | None
    chunk: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    hits: list[Hit]


@dataclass(frozen=True)
class EvalItem:
    q: str
    pages: list[str]


@dataclass(frozen=True)
class EvalRow:
    q: str
    expected: list[str]
    retrieved: list[str]
    recall: float
    hit: bool


@dataclass(frozen=True)
class EvalReport:
    k: int
    recall: float
    hit_rate: float
    rows: list[EvalRow]


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def read_pages(pdf: Path) -> list[Page]:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader: PdfReader = PdfReader(str(pdf))
        labels: list[str] = list(reader.page_labels)
        texts: list[str] = [(p.extract_text() or "") for p in reader.pages]
        starts: list[tuple[int, str]] = []
        entry: Any
        for entry in reader.outline:
            if isinstance(entry, list):
                continue
            try:
                num: int | None = reader.get_destination_page_number(entry)
            except Exception:
                num = None
            if num is None or num < 0:
                continue
            starts.append((num + 1, str(entry.title)))
    except Exception as e:
        raise RagError(f"cannot read PDF {pdf}: {e}") from e
    pages: list[Page] = []
    i: int
    text: str
    for i, text in enumerate(texts, start=1):
        chapter: str | None = None
        start: tuple[int, str]
        for start in starts:
            if start[0] <= i:
                chapter = start[1]
        pages.append(Page(pdf_page=i, label=labels[i - 1], chapter=chapter, text=text))
    return pages


def select_pages(
    pages: Sequence[Page], pages_arg: str | None = None, chapter: str | None = None
) -> list[Page]:
    if pages_arg is not None and chapter is not None:
        raise ValueError("pages and chapter are mutually exclusive")
    if pages_arg is not None:
        first: str
        sep: str
        last: str
        first, sep, last = pages_arg.partition("-")
        first = first.strip()
        last = last.strip() if sep else first
        labels: list[str] = [p.label for p in pages]
        if first not in labels:
            raise RagError(f"no page labelled {first!r}")
        a: int = labels.index(first)
        if last not in labels[a:]:
            raise RagError(f"no page labelled {last!r} after page {first!r}")
        b: int = labels.index(last, a)
        return list(pages[a : b + 1])
    if chapter is not None:
        chosen: list[Page] = [p for p in pages if p.chapter and p.chapter.split(" ")[0] == chapter]
        if not chosen:
            titles: list[str] = []
            p: Page
            for p in pages:
                if p.chapter and p.chapter not in titles:
                    titles.append(p.chapter)
            raise RagError(f"chapter {chapter!r} not found; available: {'; '.join(titles)}")
        return chosen
    return list(pages)


def normalize(text: str) -> str:
    joined: str = re.sub(r"-\n(?=[a-zà-öø-ÿ])", "", text)
    return re.sub(r"\s+", " ", joined).strip()


def chunk_pages(
    pages: Sequence[Page], book: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    page: Page
    for page in pages:
        text: str = normalize(page.text)
        n: int = 0
        start: int = 0
        while text and start < len(text):
            end: int
            if len(text) - start <= size:
                end = len(text)
            else:
                cut: int = text.rfind(" ", start + size // 2, start + size + 1)
                end = cut if cut > start else start + size
            piece: str = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        book=book,
                        pdf_page=page.pdf_page,
                        page=page.label,
                        chapter=page.chapter,
                        chunk=n,
                        text=piece,
                    )
                )
                n += 1
            if end >= len(text):
                break
            nxt: int = max(end - overlap, start + 1)
            if text[nxt - 1] != " ":
                sp: int = text.find(" ", nxt, end)
                if sp != -1:
                    nxt = sp + 1
                elif text[end] == " ":
                    nxt = end + 1
            start = nxt
    return chunks


def load_fastembed() -> Any:  # noqa: ANN401
    try:
        return importlib.import_module("fastembed")
    except ImportError as e:
        raise RagError("fastembed is not installed; install it with: uv sync --extra rag") from e


class FastEmbedder:
    dim: int = EMBED_DIM

    def __init__(
        self,
        cache_dir: Path = INDEX / "models",
        model: Any | None = None,  # noqa: ANN401
        threads: int | None = None,
    ) -> None:
        self.model: Any = (
            model
            if model is not None
            else load_fastembed().TextEmbedding(EMBED_MODEL, cache_dir=str(cache_dir), threads=threads)
        )

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self.model.embed([f"passage: {t}" for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return [[float(x) for x in v] for v in self.model.embed([f"query: {text}"])][0]


def make_embedder(cache_dir: Path) -> Embedder:
    return FastEmbedder(cache_dir)


def open_client(index: Path = INDEX, url: str | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    path: Path = index / "qdrant"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.book}/{chunk.pdf_page}/{chunk.chunk}"))


def _check_dim(client: QdrantClient, embedder: Embedder, collection: str) -> None:
    vectors: Any = client.get_collection(collection).config.params.vectors
    size: int | None = vectors.size if isinstance(vectors, VectorParams) else None
    if size != embedder.dim:
        raise RagError(
            f"collection {collection!r} has vectors of size {size}, the embedder gives {embedder.dim}: "
            "the index was built by another model, rebuild it"
        )


def index_chunks(
    chunks: Sequence[Chunk],
    client: QdrantClient,
    embedder: Embedder,
    collection: str = COLLECTION,
    batch: int = 32,
) -> int:
    if not chunks:
        return 0
    if client.collection_exists(collection):
        _check_dim(client, embedder, collection)
    else:
        client.create_collection(
            collection, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE)
        )
    books: dict[str, list[int]] = {}
    c: Chunk
    for c in chunks:
        books.setdefault(c.book, []).append(c.pdf_page)
    book: str
    pages: list[int]
    for book, pages in books.items():
        client.delete(
            collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="book", match=MatchValue(value=book)),
                        FieldCondition(key="pdf_page", range=Range(gte=min(pages), lte=max(pages))),
                    ]
                )
            ),
        )
    step: int = max(batch, 1)
    i: int
    for i in range(0, len(chunks), step):
        part: Sequence[Chunk] = chunks[i : i + step]
        vectors: list[list[float]] = embedder.embed_passages([x.text for x in part])
        client.upsert(
            collection,
            points=[
                PointStruct(
                    id=point_id(x),
                    vector=v,
                    payload={
                        "book": x.book,
                        "pdf_page": x.pdf_page,
                        "page": x.page,
                        "chapter": x.chapter,
                        "chunk": x.chunk,
                        "text": x.text,
                    },
                )
                for x, v in zip(part, vectors, strict=True)
            ],
        )
    return len(chunks)


def search(
    query: str, client: QdrantClient, embedder: Embedder, k: int = K, collection: str = COLLECTION
) -> list[Hit]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not client.collection_exists(collection):
        raise RagError("no index yet: run fr-drill index first")
    _check_dim(client, embedder, collection)
    result: Any = client.query_points(
        collection, query=embedder.embed_query(query), limit=k, with_payload=True
    )
    hits: list[Hit] = []
    pt: Any
    for pt in result.points:
        pl: dict[str, Any] = pt.payload or {}
        hits.append(
            Hit(
                chunk=Chunk(
                    book=pl["book"],
                    pdf_page=pl["pdf_page"],
                    page=pl["page"],
                    chapter=pl["chapter"],
                    chunk=pl["chunk"],
                    text=pl["text"],
                ),
                score=float(pt.score),
            )
        )
    return hits[:k]


def cite(chunk: Chunk) -> str:
    base: str = f"{chunk.book}, p. {chunk.page}"
    return f"{base}, {chunk.chapter}" if chunk.chapter else base


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    blocks: list[str] = [f"[{i}] {cite(h.chunk)}\n{h.chunk.text}" for i, h in enumerate(hits, start=1)]
    fragments: str = "\n\n".join(blocks)
    return (
        "Ты помощник по учебнику французского языка. Ответь на вопрос ученика по-русски "
        "и только по приведённым ниже фрагментам учебника.\n"
        "Французские примеры бери из фрагментов, не придумывай свои.\n"
        "После каждого утверждения ставь ссылку вида (с. N), где N: печатная страница "
        "фрагмента, на котором оно основано.\n"
        f"Если во фрагментах ответа нет, ответь ровно строкой: {NO_ANSWER}\n\n"
        f"Фрагменты:\n\n{fragments}\n\n"
        f"Вопрос: {question}\n"
    )


def ask(
    question: str,
    client: QdrantClient,
    embedder: Embedder,
    generator: Generator,
    k: int = K,
    collection: str = COLLECTION,
) -> Answer:
    hits: list[Hit] = search(question, client, embedder, k=k, collection=collection)
    if not hits:
        return Answer(NO_ANSWER, [])
    return Answer(generator.generate(build_prompt(question, hits)).strip(), hits)


class OllamaGenerator:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL, timeout: float = 600.0) -> None:
        self.model: str = model
        self.url: str = url
        self.timeout: float = timeout

    def generate(self, prompt: str) -> str:
        endpoint: str = f"{self.url.rstrip('/')}/api/generate"
        body: bytes = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            }
        ).encode()
        req: urllib.request.Request = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            resp: Any
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw: bytes = resp.read()
        except urllib.error.HTTPError as e:
            raise RagError(f"Ollama at {self.url} answered HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RagError(f"cannot reach Ollama at {self.url}: {e}") from e
        try:
            data: Any = json.loads(raw)
        except ValueError as e:
            raise RagError(f"Ollama at {self.url} did not return JSON") from e
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise RagError(f"Ollama at {self.url} returned no 'response' string")
        return str(data["response"]).strip()


class ClaudeGenerator:
    def __init__(self, model: str = CLAUDE_MODEL, command: str = "claude", timeout: float = 600.0) -> None:
        self.model: str = model
        self.command: str = command
        self.timeout: float = timeout

    def generate(self, prompt: str) -> str:
        try:
            proc: subprocess.CompletedProcess[str] = subprocess.run(
                [self.command, "-p", "--model", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as e:
            raise RagError(f"command not found: {self.command}") from e
        except subprocess.TimeoutExpired as e:
            raise RagError(f"{self.command} timed out after {self.timeout} s") from e
        except OSError as e:
            raise RagError(f"cannot run {self.command}: {e}") from e
        if proc.returncode != 0:
            raise RagError(f"{self.command} failed with code {proc.returncode}: {proc.stderr.strip()}")
        return proc.stdout.strip()


def make_generator(backend: str, model: str | None = None) -> Generator:
    if backend == "ollama":
        return OllamaGenerator(model or OLLAMA_MODEL)
    if backend == "claude":
        return ClaudeGenerator(model or CLAUDE_MODEL)
    raise ValueError(f"unknown backend: {backend!r}")


def load_eval(path: Path) -> list[EvalItem]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise RagError(f"cannot read eval file {path}: {e}") from e
    except ValueError as e:
        raise RagError(f"eval file {path} is not valid JSON: {e}") from e
    raw: Any = data.get("items") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise RagError(f"eval file {path}: 'items' must be a non-empty list")
    items: list[EvalItem] = []
    i: int
    it: Any
    for i, it in enumerate(raw):
        if not isinstance(it, dict):
            raise RagError(f"eval item {i}: must be an object")
        q: Any = it.get("q")
        pages: Any = it.get("pages")
        if not isinstance(q, str) or not q.strip():
            raise RagError(f"eval item {i}: 'q' must be a non-empty string")
        if not isinstance(pages, list) or not pages or not all(isinstance(p, str) and p for p in pages):
            raise RagError(f"eval item {i}: 'pages' must be a non-empty list of strings")
        items.append(EvalItem(q=q, pages=list(pages)))
    return items


def recall_at_k(expected: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not expected:
        raise ValueError("expected must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    return len(set(expected) & set(retrieved[:k])) / len(set(expected))


def evaluate(
    items: Sequence[EvalItem],
    client: QdrantClient,
    embedder: Embedder,
    k: int = K,
    collection: str = COLLECTION,
) -> EvalReport:
    if not items:
        raise ValueError("no eval items")
    rows: list[EvalRow] = []
    item: EvalItem
    for item in items:
        hits: list[Hit] = search(item.q, client, embedder, k=k, collection=collection)
        retrieved: list[str] = [h.chunk.page for h in hits]
        rec: float = recall_at_k(item.pages, retrieved, k)
        rows.append(
            EvalRow(q=item.q, expected=list(item.pages), retrieved=retrieved, recall=rec, hit=rec > 0)
        )
    return EvalReport(
        k=k,
        recall=sum(r.recall for r in rows) / len(rows),
        hit_rate=sum(1 for r in rows if r.hit) / len(rows),
        rows=rows,
    )


def _fail(e: Exception) -> int:
    print(f"fr-drill: {e}", file=sys.stderr)
    return 2


def cmd_index(pdf: Path, pages: str | None, chapter: str | None, index: Path, url: str | None) -> int:
    try:
        selected: list[Page] = select_pages(read_pages(pdf), pages, chapter)
        chunks: list[Chunk] = chunk_pages(selected, pdf.stem)
        if not chunks:
            raise RagError("no text on the selected pages")
        client: QdrantClient = open_client(index, url)
        try:
            n: int = index_chunks(chunks, client, make_embedder(index / "models"))
        finally:
            client.close()
    except (RagError, ValueError) as e:
        return _fail(e)
    print(f"indexed {n} chunks from {len(selected)} pages of {pdf.stem}")
    return 0


def cmd_search(query: str, k: int, as_json: bool, index: Path, url: str | None) -> int:
    try:
        client: QdrantClient = open_client(index, url)
        try:
            hits: list[Hit] = search(query, client, make_embedder(index / "models"), k=k)
        finally:
            client.close()
    except (RagError, ValueError) as e:
        return _fail(e)
    if as_json:
        rows: list[dict[str, Any]] = [
            {
                "score": h.score,
                "book": h.chunk.book,
                "page": h.chunk.page,
                "pdf_page": h.chunk.pdf_page,
                "chapter": h.chunk.chapter,
                "chunk": h.chunk.chunk,
                "text": h.chunk.text,
            }
            for h in hits
        ]
        print(json.dumps(rows, ensure_ascii=False))
        return 0
    i: int
    h: Hit
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] {h.score:.3f}  {cite(h.chunk)}")
        print("  " + h.chunk.text[:300])
    return 0


def cmd_ask(query: str, backend: str, model: str | None, k: int, index: Path, url: str | None) -> int:
    try:
        generator: Generator = make_generator(backend, model)
        client: QdrantClient = open_client(index, url)
        try:
            answer: Answer = ask(query, client, make_embedder(index / "models"), generator, k=k)
        finally:
            client.close()
    except (RagError, ValueError) as e:
        return _fail(e)
    print(answer.text)
    if answer.hits:
        print()
        print("Sources:")
        i: int
        h: Hit
        for i, h in enumerate(answer.hits, start=1):
            print(f"[{i}] {cite(h.chunk)}")
    return 0


def cmd_eval(file: Path, k: int, as_json: bool, index: Path, url: str | None) -> int:
    try:
        items: list[EvalItem] = load_eval(file)
        client: QdrantClient = open_client(index, url)
        try:
            report: EvalReport = evaluate(items, client, make_embedder(index / "models"), k=k)
        finally:
            client.close()
    except (RagError, ValueError) as e:
        return _fail(e)
    if as_json:
        print(json.dumps(dataclasses.asdict(report), ensure_ascii=False))
        return 0
    r: EvalRow
    for r in report.rows:
        print(f"{r.recall:.2f}  {r.q}  expected {r.expected} got {r.retrieved}")
    print(f"recall@{k} = {report.recall:.3f}  hit@{k} = {report.hit_rate:.3f}  (n={len(report.rows)})")
    return 0
