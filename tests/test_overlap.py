"""Verbatim-overlap check: tokens, strings, find, book_text, main (spec checks 26-30)."""

import json
import subprocess
from pathlib import Path

import pytest
from util import clean_env, run_cli

from fr_drill import overlap
from fr_drill.overlap import Hit

BOOK: str = "Regardez : les enfants jouent dans le jardin, puis ils rentrent."
ENGLISH: str = "one two three four five six seven eight nine"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Ce n’est pas du Fromage.", ["ce", "n'", "est", "pas", "du", "fromage"]),
        ("aujourd'hui", ["aujourd'", "hui"]),
        ("le rez-de-chaussée", ["le", "rez", "de", "chaussée"]),
        ("гл. 1, 2018", ["гл"]),
        ("", []),
    ],
)
def test_tokens_examples(text: str, want: list[str]) -> None:
    assert overlap.tokens(text) == want


def test_tokens_normalises_apostrophes_case_and_unicode() -> None:
    assert overlap.tokens("L‘ami ʼx `y") == ["l'", "ami", "x", "y"]
    assert overlap.tokens("J’aime") == overlap.tokens("J'aime") == ["j'", "aime"]
    assert overlap.tokens("J`aime") == ["j'", "aime"]
    assert overlap.tokens("CAFÉ") == ["café"]
    # NFC: a decomposed é is the same word as a composed one
    assert overlap.tokens("cafe\u0301") == overlap.tokens("caf\u00e9") == ["café"]
    assert overlap.tokens("Привет, Мир") == ["привет", "мир"]
    assert overlap.tokens("a_b 12c") == ["a", "b", "c"]


def test_strings_skips_source_at_any_level_and_keys() -> None:
    data: object = {"source": "X", "a": ["b", {"c": "d", "source": {"e": "f"}}], "n": 1}
    assert overlap.strings(data) == ["b", "d"]


def test_strings_of_a_bare_string_and_odd_leaves() -> None:
    assert overlap.strings("x") == ["x"]
    assert overlap.strings({"k": None, "t": True, "z": 1.5, "l": ["a", 2, None]}) == ["a"]
    assert overlap.strings({"source": "only"}) == []
    assert overlap.strings({"keyname": "value"}) == ["value"]


def test_strings_walk_order_follows_the_keys_as_given() -> None:
    assert overlap.strings({"b": "1", "a": ["2", "3"], "c": {"x": "4"}}) == ["1", "2", "3", "4"]


def test_find_sentence_once_for_a_repeated_line() -> None:
    line: str = "Les enfants jouent dans le jardin."
    got: list[Hit] = overlap.find([line, "Il y a un chat.", line], BOOK)
    assert got == [Hit(line, "les enfants jouent dans le jardin", "sentence")]


def test_find_short_sentence_is_not_a_hit() -> None:
    assert overlap.find(["Les enfants jouent."], BOOK) == []


def test_find_run_of_eight_inside_a_russian_line() -> None:
    line: str = "Правило: one two three four five six seven eight."
    assert overlap.find([line], ENGLISH) == [Hit(line, "one two three four five six seven eight", "run")]


def test_find_run_of_seven_is_too_short() -> None:
    assert overlap.find(["Правило: one two three four five six seven."], ENGLISH) == []


def test_find_cyrillic_line_never_counts_as_a_sentence() -> None:
    assert overlap.find(["Пример: les enfants jouent dans le jardin."], BOOK) == []


def test_find_run_threshold_is_a_parameter() -> None:
    line: str = "Пример: les enfants jouent."
    assert overlap.find([line], BOOK, run=3) == [Hit(line, "les enfants jouent", "run")]


def test_find_sentence_threshold_is_a_parameter() -> None:
    line: str = "Les enfants jouent."
    assert overlap.find([line], BOOK, sentence=3) == [Hit(line, "les enfants jouent", "sentence")]


def test_find_takes_the_longest_run() -> None:
    book: str = "a b c d e f g h y p q r s t u v w x"
    line: str = "Ж a b c d e f g h Z p q r s t u v w x"
    assert [h.match for h in overlap.find([line], book, run=3)] == ["p q r s t u v w x"]


def test_find_on_equal_runs_takes_the_leftmost() -> None:
    book: str = "a b c d y p q r s"
    line: str = "Ж a b c d Z p q r s"
    assert [h.match for h in overlap.find([line], book, run=3)] == ["a b c d"]


def test_find_hits_follow_the_order_of_the_lines() -> None:
    a: str = "Les enfants jouent dans le jardin."
    b: str = "Puis ils rentrent."
    got: list[Hit] = overlap.find([b, "Rien du tout ici bien sûr.", a], BOOK, sentence=3)
    assert [h.text for h in got] == [b, a]


def test_find_matches_through_case_and_punctuation() -> None:
    assert overlap.find(["LES ENFANTS jouent... dans le jardin !"], BOOK)[0].kind == "sentence"


def test_find_nothing_for_nothing() -> None:
    assert overlap.find([], BOOK) == []
    assert overlap.find(["Il y a un chat."], "") == []


def test_book_text_reads_utf8_text(tmp_path: Path) -> None:
    p: Path = tmp_path / "b.txt"
    p.write_text("Привет, café\nligne 2", encoding="utf-8")
    assert overlap.book_text(p) == "Привет, café\nligne 2"


def minimal_pdf(pages: list[str]) -> bytes:
    """A small valid PDF, one line of Helvetica text per page."""
    n: int = len(pages)
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids: str = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    i: int
    text: str
    for i, text in enumerate(pages):
        content: bytes = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out: bytes = b"%PDF-1.4\n"
    offsets: list[int] = []
    body: bytes
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref: int = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


@pytest.mark.parametrize("suffix", [".pdf", ".PDF"])
def test_book_text_reads_pdf_pages_joined_by_newline(tmp_path: Path, suffix: str) -> None:
    p: Path = tmp_path / f"book{suffix}"
    p.write_bytes(minimal_pdf(["Les enfants jouent", "dans le jardin"]))
    text: str = overlap.book_text(p)
    assert overlap.tokens(text) == ["les", "enfants", "jouent", "dans", "le", "jardin"]
    assert text.count("\n") >= 1
    assert "jouent" in text.splitlines()[0] and "dans" in text.splitlines()[-1]


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_main_without_overlaps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    a: Path = write_json(tmp_path / "a.json", {"x": ["Il y a un chat."]})
    assert overlap.main(book, [a]) == 0
    assert capsys.readouterr().out.strip() == "no overlaps"


def test_main_reports_each_hit_and_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    a: Path = write_json(
        tmp_path / "a.json",
        {"source": "Les enfants jouent dans le jardin.", "x": ["Les enfants jouent dans le jardin."]},
    )
    b: Path = write_json(tmp_path / "b.json", ["Il y a un chat."])
    assert overlap.main(book, [a, b]) == 1
    lines: list[str] = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        f"{a}: sentence: 'Les enfants jouent dans le jardin.' matches 'les enfants jouent dans le jardin'"
    ]


def test_main_passes_the_thresholds_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    a: Path = write_json(tmp_path / "a.json", ["Les enfants jouent."])
    assert overlap.main(book, [a]) == 0
    assert overlap.main(book, [a], sentence=3) == 1
    assert "sentence" in capsys.readouterr().out


def test_main_unreadable_book_or_broken_json_is_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    good: Path = write_json(tmp_path / "a.json", ["x"])
    broken: Path = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert overlap.main(tmp_path / "no-book.txt", [good]) == 2
    assert capsys.readouterr().err.strip() != ""
    assert overlap.main(book, [broken]) == 2
    assert capsys.readouterr().err.strip() != ""
    assert overlap.main(book, [tmp_path / "none.json"]) == 2
    assert capsys.readouterr().err.strip() != ""


def test_cli_overlap_exit_codes(tmp_path: Path) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    clean: Path = write_json(tmp_path / "a.json", ["Il y a un chat."])
    done: subprocess.CompletedProcess[str] = run_cli("overlap", str(book), str(clean))
    assert done.returncode == 0 and done.stdout.strip() == "no overlaps"
    hit: Path = write_json(tmp_path / "h.json", ["Les enfants jouent dans le jardin."])
    done = run_cli("overlap", str(book), str(hit))
    assert done.returncode == 1
    assert done.stdout.startswith(f"{hit}: sentence: ")
    done = run_cli("overlap", str(book), str(tmp_path / "missing.json"))
    assert done.returncode == 2


def test_cli_overlap_options(tmp_path: Path) -> None:
    book: Path = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    short: Path = write_json(tmp_path / "s.json", ["Les enfants jouent."])
    assert run_cli("overlap", str(book), str(short), env=clean_env()).returncode == 0
    assert run_cli("overlap", str(book), str(short), "--sentence", "3").returncode == 1
