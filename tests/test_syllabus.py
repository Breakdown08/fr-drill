"""Syllabus: topics.json, the generated map, the --out option (spec checks 18, 22)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from util import DATA, PRIVATE

from fr_drill import syllabus

TOPICS: Path = DATA / "syllabus" / "topics.json"
SOURCE: str = (
    "Inventaire linguistique des contenus clés des niveaux du CECRL "
    "(Eaquals, CIEP, Eurocentres, ACPLS, 2015), приложение E"
)


def test_topics_name_the_original_source() -> None:
    doc: dict[str, Any] = json.loads(TOPICS.read_text(encoding="utf-8"))
    assert doc["source"] == SOURCE


def test_topics_are_valid() -> None:
    assert syllabus.check(syllabus.load(TOPICS)) == []


def test_render_marks_itself_as_generated_by_the_new_command() -> None:
    text: str = syllabus.render(syllabus.load(TOPICS))
    assert text.startswith("# Карта тем A1\n")
    assert "*Сгенерировано `fr-drill syllabus` из `topics.json` — не править вручную.*" in text
    assert f"Источник: {SOURCE}. Тем: 67." in text
    assert "[[" not in text and PRIVATE[0] not in text.lower()
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_the_committed_map_is_the_rendered_one() -> None:
    want: str = syllabus.render(syllabus.load(TOPICS))
    assert (DATA / "syllabus" / "syllabus.md").read_text(encoding="utf-8") == want


def test_main_check_only_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shutil.copy(TOPICS, tmp_path / "topics.json")
    assert syllabus.main(True, tmp_path) == 0
    assert capsys.readouterr().out.strip() == "ok: 67 topics"
    assert list(tmp_path.iterdir()) == [tmp_path / "topics.json"]


def test_main_without_out_writes_into_root(tmp_path: Path) -> None:
    shutil.copy(TOPICS, tmp_path / "topics.json")
    assert syllabus.main(False, tmp_path) == 0
    assert (tmp_path / "syllabus.md").read_text(encoding="utf-8") == syllabus.render(syllabus.load(TOPICS))


def test_main_with_out_writes_there_only(tmp_path: Path) -> None:
    root: Path = tmp_path / "root"
    root.mkdir()
    shutil.copy(TOPICS, root / "topics.json")
    out: Path = tmp_path / "elsewhere.md"
    assert syllabus.main(False, root, out) == 0
    assert out.read_text(encoding="utf-8") == syllabus.render(syllabus.load(TOPICS))
    assert not (root / "syllabus.md").exists()
    assert syllabus.main(root=root, out=tmp_path / "kw.md") == 0
    assert (tmp_path / "kw.md").exists()


def test_main_on_broken_topics_is_one(tmp_path: Path) -> None:
    doc: dict[str, Any] = json.loads(TOPICS.read_text(encoding="utf-8"))
    doc["topics"][0]["requires"] = ["no.such.topic"]
    (tmp_path / "topics.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert syllabus.main(False, tmp_path) == 1
    assert not (tmp_path / "syllabus.md").exists()
