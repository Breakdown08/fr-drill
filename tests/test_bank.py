"""Bank checker: the format.md example passes, each broken rule is caught, the real bank is clean."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from util import DATA, INVENTORY, PRIVATE

from fr_drill import bank

type Files = dict[str, dict[str, Any]]
type Change = Callable[[Files], object]

OLD_SOURCE: str = "ny" + "arl"  # the old source name, spelled so that the published files stay clean
REF_PREFIX: str = "Inventaire CECRL (Eaquals, CIEP 2015), приложение E: "
CONSIGNES: dict[str, Any] = {
    "choose-form": {"fr": "Choisissez la bonne forme.", "ru": "Выберите верную форму."}
}

# The example of format.md, split into its two files.
ETRE: dict[str, Any] = {
    "items": [
        {
            "id": "etre-001",
            "type": "choice",
            "part": "practice",
            "topics": ["gram.etre", "gram.subject-pronouns"],
            "consigne": "choose-form",
            "prompt": "Nous ___ étudiants.",
            "options": ["sommes", "êtes", "sont"],
            "answer": 0,
            "explain": "nous → sommes.",
            "source": {"by": "author", "ref": REF_PREFIX + "être"},
        }
    ]
}
TIME: dict[str, Any] = {
    "docs": [
        {
            "id": "affiche-piscine",
            "kind": "affiche",
            "text": "Piscine municipale\nOuverte du lundi au samedi, de 9 h à 19 h.\nEntrée : 3,50 €.",
        }
    ],
    "items": [
        {
            "id": "piscine-001",
            "type": "number",
            "part": "ce",
            "doc": "affiche-piscine",
            "topics": ["act.time-price", "lex.time-date"],
            "consigne": {"fr": "Lisez le document et répondez.", "ru": "Прочитайте документ и ответьте."},
            "prompt": "À quelle heure ferme la piscine ?",
            "answer": "19:00",
            "source": {"by": "author", "ref": "delf-a1: CE ex. 3"},
        },
        {
            "id": "piscine-002",
            "type": "short",
            "part": "ce",
            "doc": "affiche-piscine",
            "topics": ["act.time-price"],
            "consigne": {"fr": "Lisez le document et répondez.", "ru": "Прочитайте документ и ответьте."},
            "prompt": "Quel jour la piscine est-elle fermée ?",
            "accept": ["le dimanche", "dimanche"],
            "source": {"by": "author", "ref": "delf-a1: CE ex. 3"},
        },
        {
            "id": "carte-001",
            "type": "write",
            "part": "pe",
            "doc": "affiche-piscine",
            "topics": ["act.time-price"],
            "consigne": {"fr": "Écrivez à un ami.", "ru": "Напишите другу."},
            "points": ["le jour", "l'heure", "le prix"],
            "min_words": 40,
            "source": {"by": "author", "ref": "delf-a1: PE ex. 2"},
        },
    ],
}
GOOD: Files = {"gram.etre": ETRE, "act.time-price": TIME}


def topics_doc() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((DATA / "syllabus" / "topics.json").read_text(encoding="utf-8"))
    return doc


def stages() -> dict[str, int]:
    return {t["id"]: t["stage"] for t in topics_doc()["topics"]}


def run_check(change: Change | None = None) -> list[str]:
    files: Files = copy.deepcopy(GOOD)
    if change is not None:
        change(files)
    errors: list[str] = bank.check(files, stages(), CONSIGNES)
    return errors


def etre(f: Files) -> dict[str, Any]:
    item: dict[str, Any] = f["gram.etre"]["items"][0]
    return item


def pisc(f: Files) -> dict[str, Any]:
    item: dict[str, Any] = f["act.time-price"]["items"][0]
    return item


def short(f: Files) -> dict[str, Any]:
    item: dict[str, Any] = f["act.time-price"]["items"][1]
    return item


def write(f: Files) -> dict[str, Any]:
    item: dict[str, Any] = f["act.time-price"]["items"][2]
    return item


def rename_doc(f: Files) -> None:
    f["act.time-price"]["docs"][0].update(id="piscine-002")
    i: dict[str, Any]
    for i in f["act.time-price"]["items"]:
        i.update(doc="piscine-002")


def reverse_topics(f: Files) -> None:
    etre(f)["topics"].reverse()


def add_topic(f: Files) -> None:
    etre(f)["topics"].append("gram.nope")


def drop_options_and_answer(f: Files) -> None:
    etre(f).pop("options")
    etre(f).pop("answer")


CASES: dict[str, tuple[Change, str]] = {
    "bad id format": (lambda f: etre(f).update(id="etre-1"), "word-NNN"),
    "duplicate item id across files": (lambda f: pisc(f).update(id="etre-001"), "duplicate id"),
    "doc id equals item id": (rename_doc, "duplicate id"),
    "unknown type": (lambda f: etre(f).update(type="match"), "type must be"),
    "unknown part": (lambda f: etre(f).update(part="co"), "part must be"),
    "typo in a field": (lambda f: etre(f).update(explian="x"), "unknown fields"),
    "unknown topic": (add_topic, "unknown topic"),
    "file not named by main topic": (reverse_topics, "does not match file"),
    "unknown consigne key": (lambda f: etre(f).update(consigne="nope"), "not in consignes"),
    "stage <= 2 without ru": (lambda f: etre(f).update(consigne={"fr": "Choisissez."}), "needs ru"),
    "unknown doc": (lambda f: pisc(f).update(doc="nope"), "unknown doc"),
    "ce without doc": (lambda f: pisc(f).pop("doc"), "ce needs a doc"),
    "prompt missing": (lambda f: pisc(f).pop("prompt"), "prompt is required"),
    "write without doc and prompt": (lambda f: write(f).pop("doc"), "prompt is required"),
    "practice without gap": (lambda f: etre(f).update(prompt="Nous sommes étudiants."), "exactly one ___"),
    "practice without explain": (lambda f: etre(f).pop("explain"), "needs explain"),
    "no source": (lambda f: etre(f).pop("source"), "source must be"),
    "unknown source": (lambda f: etre(f).update(source={"by": "internet", "ref": "x"}), "source must be"),
    "the old source name": (
        lambda f: etre(f).update(source={"by": OLD_SOURCE, "ref": "x"}),
        "source must be",
    ),
    "choice: 1 option": (lambda f: etre(f).update(options=["sommes"]), "2–4"),
    "choice: 5 options": (lambda f: etre(f).update(options=list("abcde")), "2–4"),
    "choice: options repeat after norm": (
        lambda f: etre(f).update(options=["sommes", "Sommes ", "sont"]),
        "repeat",
    ),
    "choice: answer out of range": (lambda f: etre(f).update(answer=3), "index"),
    "choice: answer is bool": (lambda f: etre(f).update(answer=True), "index"),
    "short: empty accept": (lambda f: short(f).update(accept=[]), "non-empty accept"),
    "short ce: repeat up to accents": (lambda f: short(f).update(accept=["dimanche", "dîmanche"]), "repeat"),
    "number: bad time": (lambda f: pisc(f).update(answer="25:00"), "HH:MM"),
    "number: text": (lambda f: pisc(f).update(answer="19h"), "HH:MM"),
    "write: no points": (lambda f: write(f).pop("points"), "points"),
    "write: min_words 0": (lambda f: write(f).update(min_words=0), "min_words"),
    "file without items": (lambda f: f["gram.etre"].pop("items"), "must be"),
    "bad doc kind": (lambda f: f["act.time-price"]["docs"][0].update(kind="poster"), "doc kind"),
}


def test_sources_are_the_public_ones() -> None:
    assert {"author", "book", "delf-sample", "teacher"} == bank.SOURCES
    assert OLD_SOURCE not in bank.SOURCES


def test_format_md_example_is_valid() -> None:
    assert run_check() == []


@pytest.mark.parametrize("name", list(CASES))
def test_each_broken_rule_is_caught(name: str) -> None:
    change: Change
    want: str
    change, want = CASES[name]
    errors: list[str] = run_check(change)
    assert any(want in e for e in errors), errors


def test_stage_three_and_later_need_no_ru() -> None:
    late: str = next(t for t, s in stages().items() if s > bank.RU_UNTIL)

    def late_stage(f: Files) -> None:
        etre(f)["topics"].append(late)
        etre(f).update(consigne={"fr": "Choisissez."})

    assert run_check(late_stage) == []


def test_short_practice_keeps_accent_variants_apart() -> None:
    def accents(f: Files) -> None:
        etre(f).update(type="short", accept=["étudiez", "etudiez"])
        drop_options_and_answer(f)

    assert run_check(accents) == []


def test_number_answer_may_be_a_float() -> None:
    assert run_check(lambda f: pisc(f).update(answer=12.5)) == []


def write_bank(root: Path) -> None:
    (root / "consignes.json").write_text(json.dumps(CONSIGNES), encoding="utf-8")
    stem: str
    content: dict[str, Any]
    for stem, content in GOOD.items():
        (root / f"{stem}.json").write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")


def test_main_and_summary_on_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_bank(tmp_path)
    topics: Path = DATA / "syllabus" / "topics.json"
    assert bank.main(tmp_path, topics) == 0
    assert "ok: 4 items, 1 docs in 2 files (ce 2, pe 1, practice 1)" in capsys.readouterr().out
    files: Any = bank.load(tmp_path, topics)[0]
    assert bank.summary(files) == "ok: 4 items, 1 docs in 2 files (ce 2, pe 1, practice 1)"


def test_broken_json_is_reported_with_exit_one(tmp_path: Path) -> None:
    write_bank(tmp_path)
    (tmp_path / "gram.avoir.json").write_text("{not json", encoding="utf-8")
    assert bank.main(tmp_path, DATA / "syllabus" / "topics.json") == 1


def test_the_real_bank_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert bank.main() == 0
    assert capsys.readouterr().out.strip() == "ok: 53 items, 1 docs in 10 files (ce 6, pe 1, practice 46)"


def bank_items() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    f: Path
    for f in sorted((DATA / "bank").glob("*.json")):
        if f.name != "consignes.json":
            out += json.loads(f.read_text(encoding="utf-8"))["items"]
    return out


def test_the_real_bank_names_no_old_source() -> None:
    f: Path
    files: list[Path] = sorted((DATA / "bank").glob("*.json"))
    assert len(files) == 11
    for f in files:
        assert OLD_SOURCE not in f.read_text(encoding="utf-8").lower(), f.name
    assert {i["source"]["by"] for i in bank_items()} == {"author"}


def test_the_real_bank_cites_the_original_inventory_by_title() -> None:
    refs: list[str] = [i["source"]["ref"] for i in bank_items()]
    assert len(refs) == 53
    assert sum(r.startswith(REF_PREFIX) for r in refs) == 46
    assert sum(r.startswith("delf-a1: ") for r in refs) == 7
    assert not any(INVENTORY in r for r in refs)


def test_format_md_is_rewritten_for_the_public_project() -> None:
    text: str = (DATA / "bank" / "format.md").read_text(encoding="utf-8")
    assert "fr-drill bank check" in text
    assert "src/fr_drill/bank.py" in text and "tests/test_bank.py" in text
    assert "проверяющий" in text
    assert '"by": "author"' in text
    assert REF_PREFIX in text
    assert "[[" not in text
    lowered: str = text.lower()
    assert OLD_SOURCE not in lowered and INVENTORY not in lowered and "check_bank" not in lowered
    assert PRIVATE[0] not in lowered
