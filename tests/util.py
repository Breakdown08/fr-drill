"""Helpers shared by the tests: paths, subprocess runs, a live HTTP server."""

import http.client
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import fr_drill

ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data"
FIXTURES: Path = ROOT / "tests" / "fixtures"
# Names that must not leak into the public project, spelled in pieces so that this file stays clean too.
PRIVATE: tuple[str, str] = ("kad" + "ath", "ny" + "arl")
# The unpublished inventory file, by name; spelled in parts like PRIVATE.
INVENTORY: str = "a1-" + "inventory"

# Tests that need the real vocabulary are skipped when there is none (CI). Until the package
# defines VOCAB at all they run, and fail.
_VOCAB: Any = getattr(fr_drill, "VOCAB", None)
NEEDS_VOCAB: pytest.MarkDecorator = pytest.mark.skipif(
    _VOCAB is not None and not _VOCAB.exists(), reason="no vocabulary file"
)


def clean_env(extra: dict[str, str] | None = None, keep: bool = False) -> dict[str, str]:
    """The environment of a subprocess: without FR_DRILL_* unless `keep`, plus `extra`."""
    env: dict[str, str] = {k: v for k, v in os.environ.items() if keep or not k.startswith("FR_DRILL_")}
    env.update(extra or {})
    return env


def run_py(code: str, *args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    """Run a snippet in a fresh interpreter and return its stdout; fail with stderr if it crashes."""
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env=env if env is not None else clean_env(),
        cwd=cwd,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def run_cli(
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """`fr-drill <args>` in a fresh interpreter. `env` is the whole environment (see clean_env)."""
    return subprocess.run(
        [sys.executable, "-c", "from fr_drill.cli import main; main()", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=env if env is not None else clean_env(),
        cwd=cwd,
        timeout=120,
    )


@contextmanager
def serving(make: Callable[[int], type[BaseHTTPRequestHandler]]) -> Generator[int]:
    """Serve `make(port)` on a free local port; yield the port."""
    srv: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port: int = srv.server_address[1]
    srv.RequestHandlerClass = make(port)
    thread: threading.Thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def call(
    port: int,
    method: str,
    path: str,
    body: object = None,
    host: str | None = None,
    ctype: str = "application/json",
) -> tuple[int, Any]:
    """One HTTP request; JSON bodies are decoded, anything else comes back as text."""
    conn: http.client.HTTPConnection = http.client.HTTPConnection("127.0.0.1", port)
    data: bytes | None = json.dumps(body).encode() if body is not None else None
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    if data is not None:
        conn.putheader("Content-Type", ctype)
        conn.putheader("Content-Length", str(len(data)))
    conn.endheaders(data)
    resp: http.client.HTTPResponse = conn.getresponse()
    raw: bytes = resp.read()
    conn.close()
    if resp.getheader("Content-Type", "").startswith("application/json"):
        return resp.status, json.loads(raw)
    text: str = raw.decode()
    return resp.status, (json.loads(text) if text.startswith(("{", "[")) else text)
