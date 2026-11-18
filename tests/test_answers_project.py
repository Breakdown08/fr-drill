"""Answer eval: the repository files around it (spec check 34)."""

import subprocess

from util import ROOT, run_cli

README: str = (ROOT / "README.md").read_text(encoding="utf-8")


def test_gitignore_has_runs() -> None:
    assert "/runs/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_readme_documents_the_answer_eval() -> None:
    phrase: str
    for phrase in ("fr-drill answer-eval", "fr-drill answer-report", "answers.json", "### Measured", "runs/"):
        assert phrase in README, phrase


def test_readme_measured_section_is_after_the_helper_heading() -> None:
    assert README.index("## Textbook helper") < README.index("### Measured")


def test_top_level_help_lists_both_commands() -> None:
    done: subprocess.CompletedProcess[str] = run_cli("--help")
    assert done.returncode == 0
    assert "answer-eval" in done.stdout
    assert "answer-report" in done.stdout
