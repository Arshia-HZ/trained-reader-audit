"""The Monte Carlo SC estimate for pools too large to enumerate (PREREG D23)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from reader_audit.sc_estimate import (
    SC_DRAWS,
    sc_at_cost,
    sc_credit_mc,
    sc_curve_mc,
)
from reader_audit.subsets import group_answers, sc_curve, subset_stats


def groups_of(answers, right="1"):
    return group_answers(answers, [a == right for a in answers])


def random_answers(rng, n):
    return [rng.choice(["1", "1", "2", "3", "4", None]) for _ in range(n)]


def test_the_estimate_agrees_with_the_exact_credit_within_monte_carlo_error():
    rng = random.Random(11)
    draws = 20_000
    bound = 4 * 0.5 / draws**0.5  # four standard errors at the worst case p = 0.5
    for i in range(12):
        g = groups_of(random_answers(rng, 18))
        for k in (2, 5, 9, 14):
            exact = subset_stats(g, k).vote
            got = sc_credit_mc(g, k, f"p{i}", n_draws=draws)
            assert got == pytest.approx(exact, abs=bound), (i, k)


def test_the_whole_pool_is_exact_and_needs_no_draws():
    g = groups_of(["1", "1", "2", "3", None, "1"])
    assert sc_credit_mc(g, g.n, "any") == subset_stats(g, g.n).vote


def test_the_same_problem_and_k_always_draw_the_same_subsets():
    g = groups_of(["1", "2", "1", "3", "1", "2", "4", "1"])
    a = sc_credit_mc(g, 4, "problem-a", n_draws=300)
    assert a == sc_credit_mc(g, 4, "problem-a", n_draws=300)
    values = {sc_credit_mc(g, 4, f"problem-{i}", n_draws=300) for i in range(8)}
    assert len(values) > 1  # other problems and other seeds draw other subsets


def test_a_curve_has_one_estimate_per_sample_count_and_follows_the_exact_curve():
    rng = random.Random(5)
    g = groups_of(random_answers(rng, 18))
    curve = sc_curve_mc(g, "p", n_draws=10_000)
    assert len(curve) == 18
    assert np.abs(curve - sc_curve(g)).max() < 0.03


def test_unparsable_pools_have_zero_credit():
    g = groups_of([None, None, None, None])
    assert sc_credit_mc(g, 2, "p") == 0.0


def test_k_must_lie_inside_the_pool():
    g = groups_of(["1", "2", "1"])
    with pytest.raises(ValueError):
        sc_credit_mc(g, 4, "p")
    with pytest.raises(ValueError):
        sc_credit_mc(g, 0, "p")


def test_groups_without_sample_labels_are_refused():
    from reader_audit.subsets import AnswerGroups

    with pytest.raises(ValueError, match="labels"):
        sc_credit_mc(AnswerGroups((2,), (True,), 0), 1, "p")


def test_reading_the_curve_at_a_cost_interpolates_and_flags_out_of_range_costs():
    curve = np.array([0.5, 0.6, 0.8])
    assert sc_at_cost(curve, 1.0) == 0.5
    assert sc_at_cost(curve, 2.5) == pytest.approx(0.7)
    assert sc_at_cost(curve, 3.0) == 0.8
    assert sc_at_cost(curve, 3.01) is None  # more samples than the pool holds
    assert sc_at_cost(curve, 0.99) is None


def test_the_documented_draw_count_is_the_default():
    assert SC_DRAWS == 2000
