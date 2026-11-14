"""Per-problem costs and the matched sample count k* (PREREG section 5)."""

from __future__ import annotations

import pytest

from reader_audit.budget import k_star, reader_cost, sample_cost

CANDS = [(100, 800), (100, 900), (100, 700), (100, 600)]  # (prompt, generated)
READER = (3300, 500)
POOL = [
    (100, 800),
    (100, 700),
    (100, 900),
]  # one sample costs 800 generated, 900 in all


def test_generated_only_counts_the_candidates_and_the_reader_output():
    assert reader_cost("generated_only", CANDS, READER) == 800 + 900 + 700 + 600 + 500
    assert reader_cost("generated_only", CANDS, None) == 3000


def test_prompt_inclusive_counts_every_prompt_and_weights_the_reader():
    own = sum(p + g for p, g in CANDS)
    assert reader_cost("generated_plus_prompt", CANDS, READER) == own + 3800
    assert reader_cost("generated_plus_prompt", CANDS, READER, weight=0.35) == (
        pytest.approx(own + 0.35 * 3800)
    )
    assert reader_cost("generated_plus_prompt", CANDS, None) == own


def test_a_sample_costs_its_mean_under_each_rule():
    assert sample_cost("generated_only", POOL) == 800
    assert sample_cost("generated_plus_prompt", POOL) == 900


def test_k_star_is_the_cost_ratio():
    assert k_star("generated_only", CANDS, READER, POOL) == pytest.approx(3500 / 800)
    got = k_star("generated_plus_prompt", CANDS, READER, POOL)
    assert got == pytest.approx((3400 + 3800) / 900)


def test_a_gated_read_costs_only_its_candidates():
    assert k_star("generated_only", CANDS, None, POOL) == pytest.approx(3000 / 800)


def test_unknown_rules_and_empty_pools_are_errors():
    with pytest.raises(ValueError, match="unknown cost rule"):
        reader_cost("free", CANDS, READER)
    with pytest.raises(ValueError, match="unknown cost rule"):
        sample_cost("free", POOL)
    with pytest.raises(ValueError, match="pool samples"):
        sample_cost("generated_only", [])
