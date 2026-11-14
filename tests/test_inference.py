"""The inference procedure: two-sided-scale p-values, Holm and intersection tests."""

from __future__ import annotations

import numpy as np
import pytest

from reader_audit.inference import (
    ALPHA,
    HypothesisTest,
    decide,
    evaluate,
    intersection,
    p_value,
)

FAMILY = ["H1a", "H1b", "H2", "H3", "H4a", "H4b"]


def test_a_superiority_p_value_is_twice_the_one_sided_share_capped_at_one():
    means = np.concatenate([np.full(90, 0.05), np.full(10, -0.01)])
    p = p_value(means, HypothesisTest("t", "superiority"))
    assert p == pytest.approx(2 * (1 + 10) / (100 + 1))
    assert p_value(np.full(50, -0.1), HypothesisTest("t", "superiority")) == 1.0


def test_non_inferiority_shifts_the_null_to_minus_the_margin():
    means = np.concatenate([np.full(95, -0.005), np.full(5, -0.03)])
    ni = p_value(means, HypothesisTest("t", "noninferiority", margin=0.02))
    assert ni == pytest.approx(2 * (1 + 5) / 101)  # five means are at or below -0.02
    sup = p_value(means, HypothesisTest("t", "superiority"))
    assert sup == 1.0  # the same means do not show superiority


def test_a_two_sided_test_uses_the_audits_two_sided_p():
    means = np.concatenate([np.full(97, 0.04), np.full(3, -0.02)])
    p = p_value(means, HypothesisTest("t", "two-sided"))
    assert p == pytest.approx(min(1.0, 2 * min((1 + 3) / 101, (1 + 97) / 101)))


def test_a_non_inferiority_test_needs_a_margin_and_kinds_are_checked():
    with pytest.raises(ValueError, match="margin"):
        HypothesisTest("t", "noninferiority")
    with pytest.raises(ValueError, match="unknown test kind"):
        HypothesisTest("t", "equivalence")


def test_a_significant_interval_alone_does_not_pass_a_holm_corrected_test():
    # six hypotheses; one has p = 0.02, which is below 0.05 but above 0.05 / 6
    family = {"H1a": 0.02, "H1b": 0.6, "H2": 0.7, "H3": 0.8, "H4a": 0.9, "H4b": 0.5}
    out = decide(family, FAMILY)
    assert out["H1a"]["p_holm"] == pytest.approx(0.12)
    assert out["H1a"]["supported"] is False
    alone = decide({"H1a": 0.02}, ["H1a"])
    assert alone["H1a"]["supported"] is True  # for one hypothesis p < 0.05 suffices


def test_holm_adjusts_step_down_and_keeps_the_order():
    out = decide({"H1a": 0.001, "H1b": 0.004, "H2": 0.01}, ["H1a", "H1b", "H2"])
    assert [round(out[h]["p_holm"], 4) for h in ("H1a", "H1b", "H2")] == [
        0.003,
        0.008,
        0.01,
    ]
    assert all(out[h]["supported"] for h in ("H1a", "H1b", "H2"))


def test_a_hypothesis_that_was_not_run_counts_with_p_one_and_keeps_the_family_size():
    out = decide({"H1a": 0.008}, FAMILY)
    assert out["H1b"]["run"] is False and out["H1b"]["p"] == 1.0
    assert out["H1a"]["p_holm"] == pytest.approx(0.048)  # 0.008 x 6, not x 1
    assert out["H1a"]["supported"] is True
    barely = decide({"H1a": 0.0085}, FAMILY)
    assert barely["H1a"]["supported"] is False  # 0.0085 x 6 = 0.051


def test_hypotheses_outside_the_preregistered_family_are_refused():
    with pytest.raises(ValueError, match="not in the preregistered"):
        decide({"H9": 0.01}, FAMILY)


def test_an_intersection_test_takes_the_largest_component_p():
    parts = [
        {"name": "c0", "p": 0.01},
        {"name": "c1", "p": 0.2},
        {"name": "c2+", "p": 0.03},
    ]
    out = intersection(parts)
    assert out["p"] == 0.2 and out["components"]["c2+"] == 0.03
    with pytest.raises(ValueError):
        intersection([])


def test_evaluate_reports_the_interval_and_the_p_value():
    diffs = {f"p{i}": 0.1 + 0.001 * (i % 5) for i in range(200)}
    out = evaluate(diffs, HypothesisTest("H1a", "superiority"), n_boot=500, seed=0)
    assert out["n_problems"] == 200 and out["mean"] == pytest.approx(0.102, abs=1e-3)
    assert out["ci95_low"] > 0 and out["p"] < 0.01
    with pytest.raises(ValueError, match="not finite"):
        evaluate({"a": float("nan")}, HypothesisTest("t", "superiority"))


@pytest.mark.parametrize(
    "test,true_mean,low,high",
    [
        (HypothesisTest("sup", "superiority"), 0.0, 0.002, 0.05),
        (HypothesisTest("ni", "noninferiority", margin=0.02), -0.02, 0.002, 0.05),
        (HypothesisTest("two", "two-sided"), 0.0, 0.02, 0.09),
    ],
)
def test_the_procedure_is_calibrated_at_the_boundary_of_its_null(
    test, true_mean, low, high
):
    rng = np.random.default_rng(123)
    rejections = 0
    n_sim = 400
    for _ in range(n_sim):
        d = rng.normal(true_mean, 0.2, size=150)
        res = evaluate({str(i): v for i, v in enumerate(d)}, test, n_boot=400, seed=1)
        rejections += res["p"] < ALPHA
    rate = rejections / n_sim
    assert low <= rate <= high, rate  # about 2.5% for one-sided tests, 5% two-sided
