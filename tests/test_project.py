"""Project files: pyproject.toml, the CI workflow, .gitignore, no CLAUDE.md (spec checks 1, 3, 6, 7)."""

import subprocess
import tomllib
from pathlib import Path
from typing import Any

from util import ROOT

CI_STEPS: list[str] = [
    "uv sync --locked",
    "uv run ruff check",
    "uv run ruff format --check",
    "uv run ty check",
    "uv run mypy",
    "uv run pytest",
]


def pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata() -> None:
    project: dict[str, Any] = pyproject()["project"]
    assert project["name"] == "fr-drill"
    assert project["license"] == "Apache-2.0"
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.14"
    assert project["scripts"] == {"fr-drill": "fr_drill.cli:main"}
    assert any(d.split(">")[0].split("=")[0].strip().lower() == "pypdf" for d in project["dependencies"])


def test_dev_group() -> None:
    dev: list[str] = pyproject()["dependency-groups"]["dev"]
    names: set[str] = {d.split(">")[0].split("=")[0].split("<")[0].strip() for d in dev}
    assert names == {"mypy", "pytest", "ruff", "ty", "qdrant-client"}


def test_tool_sections_as_in_the_requirements() -> None:
    tool: dict[str, Any] = pyproject()["tool"]
    assert tool["pytest"]["ini_options"] == {"testpaths": ["tests"]}
    assert tool["ruff"]["line-length"] == 110
    assert tool["ruff"]["lint"]["select"] == ["E", "F", "I", "B", "UP", "ANN"]
    assert tool["mypy"] == {"strict": True, "files": ["src", "tests"]}


def test_build_backend_is_uv_build() -> None:
    assert pyproject()["build-system"]["build-backend"] == "uv_build"


def test_lock_file_is_in_place_and_current() -> None:
    assert (ROOT / "uv.lock").is_file()
    done: subprocess.CompletedProcess[str] = subprocess.run(
        ["uv", "lock", "--check", "--offline"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr


def run_lines(text: str) -> list[str]:
    return [
        line.split("run:", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith(("run:", "- run:"))
    ]


def test_ci_steps_in_order() -> None:
    ci: Path = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.is_file()
    assert run_lines(ci.read_text(encoding="utf-8")) == CI_STEPS


def test_ci_triggers_and_actions() -> None:
    text: str = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "push" in text and "pull_request" in text and "main" in text
    assert "ubuntu-24.04" in text
    assert "actions/checkout@v4" in text
    assert "astral-sh/setup-uv@v6" in text


def test_gitignore_lines() -> None:
    lines: list[str] = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    want: str
    for want in ("/progress/", "/books/", "/index/", "/runs/", "/data/drills/vocab.json"):
        assert want in lines, want


def test_no_claude_md_and_a_readme_exists() -> None:
    assert (ROOT / "README.md").is_file()
    assert not (ROOT / "CLAUDE.md").exists()
    assert not any(
        p.name.lower() == "claude.md"
        for p in ROOT.rglob("*")
        if ".venv" not in p.parts and ".git" not in p.parts
    )
