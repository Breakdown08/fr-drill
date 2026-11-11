"""The tests never touch the real history (spec check 31): the guard itself must notice a write."""

from pathlib import Path

from conftest import snapshot


def test_snapshot_sees_new_files_and_growth(tmp_path: Path) -> None:
    before: dict[str, int] = snapshot({tmp_path})
    assert before == {}
    (tmp_path / "a.jsonl").write_text("1\n", encoding="utf-8")
    one: dict[str, int] = snapshot({tmp_path})
    assert one != before
    (tmp_path / "a.jsonl").write_text("1\n2\n", encoding="utf-8")
    assert snapshot({tmp_path}) != one


def test_snapshot_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert snapshot({tmp_path / "none"}) == {}
