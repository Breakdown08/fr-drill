"""Numbers tab: the rule, stages, picking, look-alikes, mistake kinds, HTTP on a temp file."""

import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import Live
from util import call

from fr_drill import numbers

KNOWN: dict[int, str] = {
    0: "zéro",
    1: "un",
    16: "seize",
    17: "dix-sept",
    21: "vingt et un",
    22: "vingt-deux",
    31: "trente et un",
    61: "soixante et un",
    70: "soixante-dix",
    71: "soixante et onze",
    72: "soixante-douze",
    77: "soixante-dix-sept",
    79: "soixante-dix-neuf",
    80: "quatre-vingts",
    81: "quatre-vingt-un",
    89: "quatre-vingt-neuf",
    90: "quatre-vingt-dix",
    91: "quatre-vingt-onze",
    97: "quatre-vingt-dix-sept",
    99: "quatre-vingt-dix-neuf",
    100: "cent",
    101: "cent un",
    180: "cent quatre-vingts",
    200: "deux cents",
    201: "deux cent un",
    280: "deux cent quatre-vingts",
    999: "neuf cent quatre-vingt-dix-neuf",
    1000: "mille",
    1001: "mille un",
    1990: "mille neuf cent quatre-vingt-dix",
    2000: "deux mille",
    2026: "deux mille vingt-six",
    2200: "deux mille deux cents",
    9999: "neuf mille neuf cent quatre-vingt-dix-neuf",
}
BASE_NUMBERS: list[int] = [*range(21), 30, 40, 50, 60, 70, 80, 90, 100, 1000]
TODAY: date = date(2026, 10, 25)


def rows(nums: list[int], day: date, ok: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "at": f"{day.isoformat()}T10:00:00+07:00",
            "n": n,
            "cat": numbers.cat_of(n),
            "mode": "read",
            "ok": ok,
            "kind": None if ok else "other",
        }
        for n in nums
    ]


def stage1_days(k: int, ok: bool = True) -> list[dict[str, Any]]:
    """Each stage-1 range answered 10 times a day for k days."""
    out: list[dict[str, Any]] = []
    d: int
    cat: tuple[str, int, int, int]
    for d in range(k, 0, -1):
        for cat in numbers.CATS:
            if cat[3] == 1:
                picked: list[int] = [random.Random(d).randint(cat[1], cat[2]) for _ in range(10)]
                out += rows(picked, TODAY - timedelta(days=d), ok)
    return out


def test_rule_gives_the_textbook_forms() -> None:
    wrong: dict[int, str] = {n: numbers.to_fr(n) for n in KNOWN if numbers.to_fr(n) != KNOWN[n]}
    assert wrong == {}


def test_all_numbers_below_ten_thousand_are_distinct() -> None:
    assert len({numbers.to_fr(n) for n in range(10000)}) == 10000


def test_ten_thousand_is_rejected() -> None:
    with pytest.raises(ValueError):
        numbers.to_fr(10000)


def test_ranges_cover_every_number_once() -> None:
    assert all(sum(lo <= n <= hi for _, lo, hi, _ in numbers.CATS) == 1 for n in range(10000))


def test_base_words_come_from_their_own_file() -> None:
    words: list[str] = numbers.load()
    assert len(words) == 30
    assert words[0] == "zéro" and words[-1] == "mille"
    assert words == [numbers.to_fr(n) for n in BASE_NUMBERS]


def test_load_refuses_words_not_built_by_the_rule(tmp_path: Path) -> None:
    p: Path = tmp_path / "n.json"
    p.write_text(
        json.dumps({"words": [{"fr": "vingt-un", "ru": "двадцать один"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        numbers.load(p)
    assert "not built by the rule" in str(err.value) and "vingt-un" in str(err.value)


def test_load_names_every_wrong_word(tmp_path: Path) -> None:
    p: Path = tmp_path / "n.json"
    words: list[dict[str, str]] = [
        {"fr": "un", "ru": "один"},
        {"fr": "vingt-un", "ru": "x"},
        {"fr": "cent-un", "ru": "y"},
    ]
    p.write_text(json.dumps({"words": words}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        numbers.load(p)
    assert str(err.value) == "not built by the rule: vingt-un, cent-un"


def test_load_refuses_an_empty_list(tmp_path: Path) -> None:
    p: Path = tmp_path / "n.json"
    p.write_text(json.dumps({"words": []}), encoding="utf-8")
    err: pytest.ExceptionInfo[ValueError]
    with pytest.raises(ValueError) as err:
        numbers.load(p)
    assert str(err.value) == "no words"


def test_stage_opens_on_two_days() -> None:
    assert numbers.stage([]) == 1
    assert numbers.stage(stage1_days(1) * 2) == 1  # 20 right in one day
    assert numbers.stage(stage1_days(2)) == 2


def test_one_weak_range_keeps_the_stage() -> None:
    h: list[dict[str, Any]] = stage1_days(2)
    almost: list[dict[str, Any]] = h[:-10] + rows([75] * 10, TODAY, ok=False)  # 70-79 last answers wrong
    assert numbers.stage(almost) == 1


def test_retries_do_not_count() -> None:
    h: list[dict[str, Any]] = stage1_days(2)
    retries: list[dict[str, Any]] = [dict(r, ok=False, again=True) for r in rows([75] * 10, TODAY)]
    assert numbers.stage(h + retries) == 2


def test_first_session_is_twenty_numbers_up_to_a_hundred() -> None:
    first: list[int] = numbers.pick([], rng=random.Random(1))
    assert len(first) == 20 and all(0 <= n <= 100 for n in first)
    assert len(set(first)) == 20


def test_stage_one_covers_every_range_and_nothing_above() -> None:
    seen: Counter[str] = Counter(
        numbers.cat_of(n) for s in range(200) for n in numbers.pick([], rng=random.Random(s))
    )
    assert set(seen) == {"0–16", "17–69", "70–79", "80–100"}


def test_weak_range_comes_up_more() -> None:
    seen: Counter[str] = Counter(
        numbers.cat_of(n) for s in range(200) for n in numbers.pick([], rng=random.Random(s))
    )
    weak: list[dict[str, Any]] = stage1_days(1) + rows(list(range(70, 80)) * 2, TODAY, ok=False)
    lean: Counter[str] = Counter(
        numbers.cat_of(n) for s in range(200) for n in numbers.pick(weak, rng=random.Random(s))
    )
    assert lean["70–79"] > 1.5 * seen["70–79"]


def test_missed_number_comes_back_until_it_is_right() -> None:
    missed: list[dict[str, Any]] = rows([97], TODAY, ok=False)
    assert all(97 in numbers.pick(missed, rng=random.Random(s)) for s in range(20))
    assert sum(97 in numbers.pick(missed + rows([97], TODAY), rng=random.Random(s)) for s in range(50)) < 20


def test_stage_two_opens_hundreds_but_not_thousands() -> None:
    up: Counter[str] = Counter(
        numbers.cat_of(n) for s in range(200) for n in numbers.pick(stage1_days(2), rng=random.Random(s))
    )
    assert up["101–999"] > 0 and not up["1000–9999"]


def test_lookalikes_of_97() -> None:
    got: list[int] = numbers.lookalikes(97, 100, random.Random(0))
    assert set(got) <= {96, 98, 87, 77, 79, 100} and len(got) == 3


def test_lookalikes_are_three_distinct_in_range_and_never_the_number() -> None:
    rng: random.Random = random.Random(4)
    n: int
    _rep: int
    for n in range(101):
        for _rep in range(5):
            got: list[int] = numbers.lookalikes(n, 100, rng)
            assert len(set(got)) == 3 and n not in got and all(0 <= c <= 100 for c in got)


def test_sound_alikes_are_offered() -> None:
    assert any(6 in numbers.lookalikes(16, 100, random.Random(s)) for s in range(20))


def test_mistake_kinds() -> None:
    assert [numbers.kind(97, g) for g in (77, 96, 42017, 13, None)] == [
        "tens",
        "units",
        "pieces",
        "other",
        "other",
    ]
    assert numbers.kind(16, 6) == "tens" and numbers.kind(280, 180) == "hundreds"


def test_parse() -> None:
    assert numbers.parse(" 1 990 ") == 1990 and numbers.parse("2.026") == 2026
    assert numbers.parse("97a") is None and numbers.parse("") is None


def test_questions_without_and_with_voice() -> None:
    first: list[int] = numbers.pick([], rng=random.Random(1))
    qs: list[dict[str, Any]] = numbers.questions(first, 100, voice=False, rng=random.Random(5))
    assert {q["mode"] for q in qs} <= {"read", "pick"}
    assert "listen" in {q["mode"] for q in numbers.questions(first, 100, voice=True, rng=random.Random(5))}
    picks: list[dict[str, Any]] = [q for q in qs if q["mode"] == "pick"]
    assert picks and all(
        len(q["options"]) == 4 and numbers.to_fr(q["n"]) in q["options"] and q["prompt"] == str(q["n"])
        for q in picks
    )


def test_attempts_land_in_the_given_file_only(tmp_path: Path) -> None:
    nums: Any = numbers.Numbers(tmp_path / "attempts" / "numbers.jsonl")
    s: dict[str, Any] = nums.session()
    typed: dict[str, Any] = next(i for i in nums.sessions[s["session"]].values() if i["mode"] != "pick")
    nums.check({"session": s["session"], "id": typed["id"], "answer": str(typed["n"])})
    assert (tmp_path / "attempts" / "numbers.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_http_session_and_checks(live: Live, tmp_path: Path) -> None:
    port: int = live.port
    nums: Any = live.t["numbers"]
    assert call(port, "GET", "/numbers/session", host="evil.com")[0] == 403
    st: int
    s: dict[str, Any]
    st, s = call(port, "GET", "/numbers/session")
    assert st == 200 and len(s["items"]) == 20 and s["stage"] == 1 and s["range"] == "0–100"
    assert all("values" not in i and "n" not in i for i in s["items"])
    assert all(i["mode"] != "listen" for i in s["items"])
    real: dict[str, dict[str, Any]] = nums.sessions[s["session"]]
    typed: dict[str, Any] = next(i for i in real.values() if i["mode"] != "pick")
    r: dict[str, Any]
    st, r = call(
        port,
        "POST",
        "/numbers/check",
        {"session": s["session"], "id": typed["id"], "answer": f" {typed['n']} ", "ms": 900},
    )
    assert st == 200 and r["ok"] and r["expected"] == typed["n"] and r["fr"] == numbers.to_fr(typed["n"])
    st, r = call(
        port,
        "POST",
        "/numbers/check",
        {"session": s["session"], "id": typed["id"], "answer": "4207", "again": True},
    )
    assert st == 200 and not r["ok"] and r["kind"] == "pieces"
    pk: dict[str, Any] | None = next((i for i in real.values() if i["mode"] == "pick"), None)
    if pk:
        right: int = pk["values"].index(pk["n"])
        st, r = call(
            port,
            "POST",
            "/numbers/check",
            {"session": s["session"], "id": pk["id"], "answer": (right + 1) % 4},
        )
        assert st == 200 and not r["ok"] and r["answer"] == right
        assert (
            call(port, "POST", "/numbers/check", {"session": s["session"], "id": pk["id"], "answer": 7})[0]
            == 404
        )
    assert call(port, "POST", "/numbers/check", {"session": "x", "id": "0", "answer": "1"})[0] == 404
    logged: list[dict[str, Any]] = [
        json.loads(line) for line in (tmp_path / "numbers.jsonl").read_text().splitlines()
    ]
    assert len(logged) == 2 + bool(pk)
    assert logged[1]["again"] and not logged[0]["again"] and logged[1]["got"] == 4207
    stats: dict[str, Any]
    st, stats = call(port, "GET", "/numbers/stats")
    assert st == 200 and stats["answers"] == 1 + bool(pk) and stats["stage"] == 1
    st, s = call(port, "GET", "/numbers/session?voice=1")
    assert any(i["mode"] == "listen" for i in s["items"])
