"""Nothing private or borrowed in the published files (spec check 5)."""

from pathlib import Path

from util import PRIVATE, ROOT

SUFFIXES: set[str] = {".py", ".md", ".json", ".html", ".toml", ".yml", ".yaml", ".txt"}
SKIP_DIRS: set[str] = {
    ".git",
    ".venv",
    "progress",
    "books",
    "index",
    "runs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
# Spelled in parts: the scan must not publish the words it looks for.
FORBIDDEN: list[str] = [
    *PRIVATE,
    "нья" + "рл",
    "va" + "ult",
    "obsi" + "dian",
    "кир" + "илл",
    "kir" + "ill",
    "lan" + "geek",
]
PERSONAL_VOCAB: Path = ROOT / "data" / "drills" / "vocab.json"


def scanned() -> list[Path]:
    out: list[Path] = []
    p: Path
    for p in sorted(ROOT.rglob("*")):
        rel: tuple[str, ...] = p.relative_to(ROOT).parts
        if p.is_file() and p.suffix in SUFFIXES and not SKIP_DIRS & set(rel[:-1]):
            if p != PERSONAL_VOCAB and p != Path(__file__).resolve():
                out.append(p)
    return out


def test_the_published_files_are_there_to_be_scanned() -> None:
    names: set[str] = {str(p.relative_to(ROOT)) for p in scanned()}
    assert {
        "README.md",
        "pyproject.toml",
        "src/fr_drill/cli.py",
        "src/fr_drill/overlap.py",
        "src/fr_drill/drill.html",
        "src/fr_drill/exam.html",
        "data/syllabus/topics.json",
        "data/bank/format.md",
        "data/reference/delf-a1.md",
        ".github/workflows/ci.yml",
    } <= names


def test_no_private_or_borrowed_names() -> None:
    found: list[str] = []
    p: Path
    for p in scanned():
        text: str = p.read_text(encoding="utf-8").lower()
        found += [f"{p.relative_to(ROOT)}: {w}" for w in FORBIDDEN if w in text]
    assert found == []
