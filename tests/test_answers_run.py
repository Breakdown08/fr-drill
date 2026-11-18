"""Answer eval: summary, run file, rescoring, comparison (spec checks 10-15)."""

import dataclasses
import json
import math
import random
from datetime import datetime
from pathlib import Path

import pytest
from answers_builders import grade, row, run_of, scored, summary_of, write_text

from fr_drill import answers
from fr_drill.answers import AnswerError, AnswerItem, AnswerRow, Comparison, Run, Summary


def four_rows() -> list[AnswerRow]:
    return [
        row("a", grade(1.0, True, cited=["1"], cite_in_context=1.0, cite_expected=True), generate_s=1.0),
        row(
            "b",
            grade(0.5, False, forbidden=["x"], cited=["1", "2"], cite_in_context=0.5),
            generate_s=2.0,
        ),
        row("c", grade(0.0, False, refused=True, cited=[], cite_in_context=None), generate_s=3.0),
        row("d", grade(1.0, True), expect="no_answer", generate_s=10.0),
    ]


def test_summarize_hand_calculation() -> None:
    s: Summary = answers.summarize(four_rows())
    assert s.n_answer == 3
    assert s.n_no_answer == 1
    assert s.answer_score == 0.5
    assert s.pass_rate == pytest.approx(1 / 3)
    assert s.wrong_rate == pytest.approx(1 / 3)
    assert s.false_refusal_rate == pytest.approx(1 / 3)
    assert s.no_answer_accuracy == 1.0
    assert s.cited_rate == pytest.approx(2 / 3)
    assert s.cite_precision == 0.75
    assert s.cite_expected_rate == pytest.approx(1 / 3)


def test_summarize_times() -> None:
    s: Summary = answers.summarize(four_rows())
    assert s.generate_median_s == 2.5
    assert s.generate_mean_s == 4.0
    assert s.generate_max_s == 10.0
    assert s.generate_total_s == 16.0
    assert s.retrieve_total_s == pytest.approx(2.0)


def test_summarize_without_no_answer_rows() -> None:
    s: Summary = answers.summarize(four_rows()[:3])
    assert s.no_answer_accuracy is None
    assert s.n_no_answer == 0


def test_summarize_without_answer_rows_and_without_citations() -> None:
    s: Summary = answers.summarize([four_rows()[3]])
    assert s.n_answer == 0
    assert s.answer_score is None
    assert s.pass_rate is None
    assert s.cited_rate is None
    assert s.cite_precision is None
    assert s.no_answer_accuracy == 1.0
    s2: Summary = answers.summarize([four_rows()[2]])
    assert s2.cite_precision is None
    assert s2.cited_rate == 0.0


def test_summarize_empty_is_an_error() -> None:
    with pytest.raises(ValueError):
        answers.summarize([])


def test_format_summary_lines() -> None:
    lines: list[str] = answers.format_summary(answers.summarize(four_rows()))
    assert len(lines) == 4
    assert lines[0].startswith("answers   n=3  score 0.500  pass 0.333")
    assert lines[0] == "answers   n=3  score 0.500  pass 0.333  wrong 0.333  false-refusal 0.333"
    assert lines[1] == "no-answer n=1  accuracy 1.000"
    assert lines[2] == "citations cited 0.667  in-context 0.750  expected-page 0.333"
    assert lines[3] == (
        "time      generate median 2.5 s  mean 4.0 s  max 10.0 s  total 16 s  retrieve total 2.0 s"
    )


def test_format_summary_example_line() -> None:
    s: Summary = summary_of(2, 0, 0.75, 0.5, 0.0, 0.0, None, 0.5, 0.5, 0.5)
    assert (
        answers.format_summary(s)[0]
        == "answers   n=2  score 0.750  pass 0.500  wrong 0.000  false-refusal 0.000"
    )


def test_format_summary_prints_none_as_na() -> None:
    s: Summary = summary_of(0, 0, None, None, None, None, None, None, None, None)
    lines: list[str] = answers.format_summary(s)
    assert len(lines) == 4
    assert lines[0] == "answers   n=0  score n/a  pass n/a  wrong n/a  false-refusal n/a"
    assert lines[1] == "no-answer n=0  accuracy n/a"
    assert lines[2] == "citations cited n/a  in-context n/a  expected-page n/a"


def test_default_run_path_example() -> None:
    got: Path = answers.default_run_path("ollama", "gemma3:4b", datetime(2026, 11, 18, 12, 30, 5))
    assert got == answers.RUNS / "answers-ollama-gemma3-4b-20261118-123005.json"


def test_default_run_path_slug_and_runs_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(answers, "RUNS", tmp_path)
    got: Path = answers.default_run_path("claude", "/odd name//v1.5!", datetime(2026, 1, 2, 3, 4, 5))
    assert got == tmp_path / "answers-claude-odd-name-v1.5-20260102-030405.json"


# write_run, read_run


def test_write_then_read_gives_an_equal_run(tmp_path: Path) -> None:
    run: Run = run_of(four_rows())
    path: Path = tmp_path / "deep" / "er" / "run.json"
    answers.write_run(run, path)
    assert path.is_file()
    assert answers.read_run(path) == run


def test_write_run_keeps_cyrillic_and_overwrites(tmp_path: Path) -> None:
    path: Path = tmp_path / "run.json"
    answers.write_run(run_of([row("a", grade(), answer="Привет")], model="old"), path)
    answers.write_run(run_of([row("a", grade(), answer="Привет")], model="new"), path)
    text: str = path.read_text(encoding="utf-8")
    assert "Привет" in text
    assert json.loads(text)["model"] == "new"
    assert json.loads(text) == dataclasses.asdict(run_of([row("a", grade(), answer="Привет")], model="new"))


def test_read_run_bad_files(tmp_path: Path) -> None:
    good: dict[str, object] = dataclasses.asdict(run_of(four_rows()))
    err: pytest.ExceptionInfo[AnswerError]
    with pytest.raises(AnswerError) as err:
        answers.read_run(tmp_path / "nope.json")
    assert "nope.json" in str(err.value)
    broken: Path = write_text(tmp_path / "broken.json", "{not json")
    with pytest.raises(AnswerError) as err:
        answers.read_run(broken)
    assert "broken.json" in str(err.value)
    schema2: Path = write_text(tmp_path / "s2.json", json.dumps({**good, "schema": 2}))
    with pytest.raises(AnswerError) as err:
        answers.read_run(schema2)
    assert "s2.json" in str(err.value)
    no_rows: dict[str, object] = {k: v for k, v in good.items() if k != "rows"}
    with pytest.raises(AnswerError):
        answers.read_run(write_text(tmp_path / "norows.json", json.dumps(no_rows)))


def test_read_run_wrong_type(tmp_path: Path) -> None:
    good: dict[str, object] = dataclasses.asdict(run_of(four_rows()))
    with pytest.raises(AnswerError):
        answers.read_run(write_text(tmp_path / "t.json", json.dumps({**good, "k": "five"})))
    with pytest.raises(AnswerError):
        answers.read_run(write_text(tmp_path / "l.json", json.dumps([1, 2])))


# rescore

ITEM_A: AnswerItem = AnswerItem("a", "q", "answer", "21", ["1"], [["au Japon"], ["en été"]], [])
ITEM_B: AnswerItem = AnswerItem("b", "q", "no_answer", None, [], [], [])


def test_rescore_drops_unknown_ids_and_counts_them() -> None:
    run: Run = run_of([scored("a", 1.0), scored("gone", 1.0), scored("b", 0.0)])
    rows: list[AnswerRow]
    skipped: int
    rows, skipped = answers.rescore(run, [ITEM_B, ITEM_A])
    assert skipped == 1
    assert [r.id for r in rows] == ["a", "b"]


def test_rescore_recomputes_a_stale_grade_by_the_current_must() -> None:
    stale: AnswerRow = row(
        "a", grade(score=0.0, passed=False), answer="Говорят au Japon (с. 1)", retrieved=["1"]
    )
    rows: list[AnswerRow]
    skipped: int
    rows, skipped = answers.rescore(run_of([stale]), [ITEM_A])
    assert skipped == 0
    assert rows[0].grade.facts_hit == 1
    assert rows[0].grade.facts_total == 2
    assert rows[0].grade.score == 0.5
    assert rows[0].grade.cited == ["1"]
    assert rows[0].answer == stale.answer
    assert rows[0].generate_s == stale.generate_s
    assert rows[0].retrieved == ["1"]


def test_rescore_takes_expect_from_the_item() -> None:
    old: AnswerRow = row("b", grade(score=0.0, passed=False), expect="answer", answer=answers.NO_ANSWER)
    rows: list[AnswerRow]
    skipped: int
    rows, skipped = answers.rescore(run_of([old]), [ITEM_B])
    assert skipped == 0
    assert rows[0].expect == "no_answer"
    assert rows[0].grade.score == 1.0
    assert rows[0].grade.passed is True


# compare


def pairs(a: list[float], b: list[float]) -> tuple[list[AnswerRow], list[AnswerRow]]:
    ra: list[AnswerRow] = [scored(f"i{n}", s) for n, s in enumerate(a)]
    rb: list[AnswerRow] = [scored(f"i{n}", s) for n, s in enumerate(b)]
    return ra, rb


def test_compare_constant_difference() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.0] * 6, [0.5] * 6)
    c: Comparison = answers.compare(ra, rb)
    assert c.n == 6
    assert c.diff == 0.5
    assert c.ci_low == 0.5
    assert c.ci_high == 0.5
    assert c.sd == 0.0
    assert c.better == 6
    assert c.worse == 0
    assert c.same == 0
    assert c.mean_a == 0.0
    assert c.mean_b == 0.5


def test_compare_mixed_hand_calculation() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.0, 0.5, 1.0], [0.5, 0.5, 0.5])
    c: Comparison = answers.compare(ra, rb)
    assert c.n == 3
    assert c.mean_a == 0.5
    assert c.mean_b == 0.5
    assert c.diff == pytest.approx(0.0)
    assert c.sd == pytest.approx(0.5)
    assert (c.better, c.worse, c.same) == (1, 1, 1)
    assert c.ci_low <= c.diff <= c.ci_high
    assert c.ci_low < c.ci_high


def test_compare_is_reproducible_and_follows_the_bootstrap_recipe() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.0, 0.2, 0.9, 0.4, 1.0], [0.3, 0.1, 1.0, 0.9, 0.5])
    c1: Comparison = answers.compare(ra, rb, resamples=1000, seed=3)
    assert answers.compare(ra, rb, resamples=1000, seed=3) == c1
    d: list[float] = [y.grade.score - x.grade.score for x, y in zip(ra, rb, strict=True)]
    rng: random.Random = random.Random(3)
    means: list[float] = sorted(
        sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(1000)
    )
    assert c1.ci_low == pytest.approx(means[math.floor(0.025 * 1000)])
    assert c1.ci_high == pytest.approx(means[math.ceil(0.975 * 1000) - 1])


def test_compare_single_pair_has_zero_sd() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.25], [0.75])
    c: Comparison = answers.compare(ra, rb, resamples=10)
    assert c.sd == 0.0
    assert c.diff == 0.5


def test_compare_skips_no_answer_and_unpaired_ids() -> None:
    ra: list[AnswerRow] = [scored("x", 0.0), scored("n", 0.0, expect="no_answer"), scored("only-a", 1.0)]
    rb: list[AnswerRow] = [scored("only-b", 0.0), scored("n", 1.0, expect="no_answer"), scored("x", 1.0)]
    c: Comparison = answers.compare(ra, rb, resamples=50)
    assert c.n == 1
    assert c.diff == 1.0


def test_compare_counts_ties_with_the_tolerance() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.5, 0.5, 0.5], [0.5 + 1e-12, 0.4, 0.6])
    c: Comparison = answers.compare(ra, rb, resamples=50)
    assert (c.better, c.worse, c.same) == (1, 1, 1)


def test_compare_errors() -> None:
    ra: list[AnswerRow]
    rb: list[AnswerRow]
    ra, rb = pairs([0.5], [0.5])
    with pytest.raises(ValueError):
        answers.compare([scored("a", 1.0)], [scored("b", 1.0)])
    with pytest.raises(ValueError):
        answers.compare([], [])
    with pytest.raises(ValueError):
        answers.compare(ra, rb, resamples=0)
