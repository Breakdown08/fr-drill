"""Textbook helper: project files, lazy imports, README (spec checks 1, 2, 4, 5, 44)."""

import re
import subprocess
import tomllib
from typing import Any

import pytest
from util import ROOT, run_cli, run_py

COMMANDS: list[str] = ["index", "search", "ask", "eval", "answer-eval"]


def pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def names(specs: list[str]) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", s, maxsplit=1)[0].strip() for s in specs}


def test_rag_extra_is_fastembed_and_qdrant() -> None:
    extra: list[str] = pyproject()["project"]["optional-dependencies"]["rag"]
    assert names(extra) == {"fastembed", "qdrant-client"}
    assert len(extra) == 2


def test_base_dependencies_stay_pypdf_only() -> None:
    assert names(pyproject()["project"]["dependencies"]) == {"pypdf"}
    assert len(pyproject()["project"]["dependencies"]) == 1


def test_dev_group_has_qdrant_but_not_fastembed() -> None:
    assert names(pyproject()["dependency-groups"]["dev"]) == {"mypy", "pytest", "ruff", "ty", "qdrant-client"}


def test_lock_has_the_extra_and_no_torch() -> None:
    lock: str = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "fastembed"' in lock
    assert 'name = "qdrant-client"' in lock
    assert 'name = "torch"' not in lock
    assert 'name = "sentence-transformers"' not in lock


def test_import_and_help_do_not_load_the_helper() -> None:
    code: str = (
        "import sys\n"
        "import fr_drill.cli as cli\n"
        "def bad():\n"
        "    return [m for m in ('qdrant_client', 'fastembed', 'fr_drill.rag') if m in sys.modules]\n"
        "assert bad() == [], bad()\n"
        "for argv in (['--help'], ['index', '--help'], ['search', '--help'],\n"
        "             ['ask', '--help'], ['eval', '--help']):\n"
        "    try:\n"
        "        cli.main(argv)\n"
        "    except SystemExit as e:\n"
        "        assert e.code == 0, argv\n"
        "    assert bad() == [], (argv, bad())\n"
        "print('ok')\n"
    )
    assert run_py(code).splitlines()[-1] == "ok"


def test_help_lists_the_new_commands() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("--help")
    assert done.returncode == 0
    word: str
    for word in COMMANDS:
        assert word in done.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help_works(command: str) -> None:
    done: subprocess.CompletedProcess[str] = run_cli(command, "--help")
    assert done.returncode == 0
    assert "--index" in done.stdout
    assert "--qdrant-url" in done.stdout


def test_pages_and_chapter_exclude_each_other() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("index", "--pages", "1-2", "--chapter", "1", "X.pdf")
    assert done.returncode == 2
    assert "not allowed with" in done.stderr
    assert "Traceback" not in done.stderr


def test_unknown_backend_is_an_argparse_error() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("ask", "question", "--backend", "gpt")
    assert done.returncode == 2
    assert "argument --backend" in done.stderr
    assert "Traceback" not in done.stderr


README_MUST: list[str] = [
    "## Textbook helper",
    "uv sync --extra rag",
    "fr-drill index",
    "fr-drill search",
    "fr-drill ask",
    "fr-drill eval",
    "fr-drill answer-eval",
    "fr-drill answer-report",
    "multilingual-e5-large",
    "Ollama",
    "index/models/",
    "index/qdrant/",
    "--qdrant-url",
    "docker run -d -p 11434:11434 -v ollama:/root/.ollama --name ollama ollama/ollama",
    "docker exec ollama ollama pull",
    "Claude Code",
]


@pytest.mark.parametrize("phrase", README_MUST)
def test_readme_mentions(phrase: str) -> None:
    assert phrase in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_drops_the_old_claims() -> None:
    text: str = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "There is no assistant over textbooks" not in text
    assert "reserved for a helper that does not exist yet" not in text
