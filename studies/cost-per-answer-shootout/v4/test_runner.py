"""Frozen scorer and analysis tests for cost-per-answer-shootout v2. Synthetic text only; no study output."""

import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("w18runner", Path(__file__).with_name("runner.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize(
    ("text", "value", "rule"),
    [
        ("42", 42, "bare-number"),
        ("$1,200.", 1200, "bare-number"),
        ("We add 3 and 4.\n#### 7", 7, "hash-delimiter"),
        ("So the total is \\boxed{125}.", 125, "boxed"),
        ("Work: 10 - 4 = 6 apples.\n\nAnswer: 6", 6, "answer-marker"),
        ("Steps 2 * 3 = 6.\n**Answer: 3 + 4 = 7**.", 7, "answer-marker"),
        ("The final answer is $18.", 18, "answer-marker"),
        ("Final answer:\n\n -5", -5, "answer-marker-next"),
        ("She has 16 - 7 = 9 eggs, and 9 x 2 = 18 dollars.", 18, "last-number"),
        ("The price is 1,250,000 dollars in total", 1250000, "last-number"),
        ("Half of 2 is 1. So 2 + 1 = 3 bolts.\n\n**3**", 3, "last-number"),
        ("0.5 of 10 is 5.0", 5, "last-number"),
        ("<think>maybe 99</think>The answer is 12", 12, "answer-marker"),
    ],
)
def test_extractor_frozen_cases(text, value, rule):
    got, why = m.extract_final_number(text)
    assert why == rule
    assert m.same_number(got, value)


def test_truncated_reply_needs_an_explicit_marker():
    assert m.extract_final_number("Step one gives 40, then 41", "length") == (
        None,
        "truncated-no-marker",
    )
    got, why = m.extract_final_number("Answer: 41\nMore text that ran on", "length")
    assert why == "answer-marker" and got == 41


def test_no_number_and_empty_reply_are_incorrect():
    assert m.extract_final_number("") == (None, "no-number")
    assert m.extract_final_number("I cannot tell.") == (None, "no-number")
    assert not m.same_number(None, 3)


def test_word_answers_is_not_a_marker():
    got, why = m.extract_final_number("Three answers were wrong; she kept 8")
    assert why == "last-number" and got == 8


def test_v1_strict_scorer_is_unchanged():
    assert m.normalized_number("#### 1,200") == 1200
    assert m.normalized_number("15 because it is the sum") is None
    assert m.normalized_number("") is None


def test_intervals_and_ratio_bootstrap():
    assert m.wilson(0, 32)[0] == 0 and m.wilson(32, 32)[1] == 1
    rows = {i: {"cost_usd": 0.001, "correct": True} for i in range(32)}
    lo, hi = m.ratio_ci(rows, "cost_usd", "correct")
    assert math.isclose(lo, 0.001) and math.isclose(hi, 0.001)
    none = {i: {"cost_usd": 0.001, "correct": False} for i in range(32)}
    assert m.ratio_ci(none, "cost_usd", "correct") == [math.inf, math.inf]


def test_paired_ratio_decision_inputs():
    a = {i: {"cost_usd": 0.001, "correct": True} for i in range(32)}
    b = {i: {"cost_usd": 0.002, "correct": True} for i in range(32)}
    ci, undefined = m.paired_ratio_ci(a, b)
    assert undefined == 0 and math.isclose(ci[0], 0.5) and math.isclose(ci[1], 0.5)
    zero = {i: {"cost_usd": 0.001, "correct": False} for i in range(32)}
    assert m.paired_ratio_ci(zero, b) == (None, 2000)


def test_summary_reports_primary_secondary_and_contrasts():
    rows = []
    for model, cost in [("4B", 0.001), ("8B", 0.002)]:
        for cache in [False, True]:
            for i in range(32):
                rows.append(
                    {
                        "model": model,
                        "cache": cache,
                        "input_id": i,
                        "correct": i % 4 != 0,
                        "correct_strict_v1": i == 0,
                        "extraction_rule": "answer-marker",
                        "finish_reason": "stop",
                        "cost_usd": cost,
                        "usage": [{"completion_tokens": 50}],
                    }
                )
    run = m.Run.__new__(m.Run)
    summary, verdict = run.summarize(rows)
    g = summary["groups"]["4B-cache-False"]
    assert (g["correct"], g["n"], g["correct_strict_v1"]) == (24, 32, 1)
    c = summary["contrasts"]["4B-vs-8B-cache-False"]
    assert math.isclose(c["ratio"], 0.5) and c["decision"] == "4B-cache-False lower"
    assert summary["contrasts"]["4B-ON-vs-OFF"]["decision"] == "no detectable difference"
    assert all("n=" in line or "/" in line for line in verdict)
