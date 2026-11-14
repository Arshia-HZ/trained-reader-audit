"""Power arithmetic under Holm, checked against hand-computed values."""

from __future__ import annotations

import pytest

from reader_audit import power as P


def test_holm_levels_run_from_alpha_over_m_to_alpha():
    levels = P.holm_levels(6)
    assert levels[0] == pytest.approx(0.05 / 6) and levels[-1] == 0.05
    assert levels == sorted(levels)


def test_the_old_1_6_point_estimate_was_the_unadjusted_single_test():
    # 972 problems, sd 0.18: the number D21 gave on 2026-11-14
    assert P.mde(972, 0.18, 0.05) == pytest.approx(0.01617, abs=2e-5)


def test_the_holm_worst_case_needs_about_two_points():
    assert P.mde(972, 0.18, 0.05 / 6) == pytest.approx(0.02009, abs=3e-5)
    assert P.mde(972, 0.16, 0.05 / 6) == pytest.approx(0.01786, abs=3e-5)
    assert P.mde(972, 0.20, 0.05 / 6) == pytest.approx(0.02232, abs=3e-5)


def test_power_at_the_planning_effect_falls_under_holm():
    single = P.superiority_power(972, 0.18, 0.015, 0.05)
    worst = P.superiority_power(972, 0.18, 0.015, 0.05 / 6)
    assert single == pytest.approx(0.738, abs=0.005)
    assert worst == pytest.approx(0.485, abs=0.01)
    assert worst < single


def test_power_is_a_half_when_the_effect_equals_the_critical_value():
    se = P.standard_error(400, 0.2)
    assert P.superiority_power(400, 0.2, P.z(0.975) * se, 0.05) == pytest.approx(0.5)


def test_non_inferiority_power_grows_with_the_true_difference_and_the_margin():
    base = P.noninferiority_power(900, 0.25, 0.02, 0.0, 0.05)
    assert P.noninferiority_power(900, 0.25, 0.02, 0.01, 0.05) > base
    assert P.noninferiority_power(900, 0.25, 0.03, 0.0, 0.05) > base
    assert P.noninferiority_power(900, 0.25, 0.02, -0.02, 0.05) < 0.03  # on the null


def test_the_sample_size_for_non_inferiority_matches_its_power():
    n = P.n_for_noninferiority(0.25, 0.02, 0.0, 0.05, power=0.8)
    assert P.noninferiority_power(n, 0.25, 0.02, 0.0, 0.05) >= 0.8
    assert P.noninferiority_power(n - 1, 0.25, 0.02, 0.0, 0.05) < 0.8 + 1e-3
    with pytest.raises(ValueError):
        P.n_for_noninferiority(0.25, 0.02, -0.03, 0.05)


def test_the_binomial_model():
    assert P.sd_binomial("diff", 8) == pytest.approx((2 * 0.25 / 8) ** 0.5)
    assert P.sd_binomial("did", 8) == pytest.approx((4 * 0.25 / 8) ** 0.5)
    assert P.sd_binomial("vs_fixed", 8) == pytest.approx((0.25 / 8) ** 0.5)
    with pytest.raises(ValueError):
        P.sd_binomial("sum", 8)


DESIGNS = {
    "c0": P.StratumDesign(0.337, 3.0),
    "c1": P.StratumDesign(0.221, 2.0),
    "c2+": P.StratumDesign(0.702, 3.63),
    "swap_primary": P.StratumDesign(0.519, 2.94),
}


def test_the_family_table_covers_every_hypothesis_and_orders_the_levels():
    rows = P.family1_rows(DESIGNS, n_primary=972, replicates=2)
    names = [r["hypothesis"] for r in rows]
    assert names[:2] == ["H1a", "H1b"] and "H2 (c1)" in names
    assert (
        sum(n.startswith("H3") for n in names) == 3 and "H4b (swap, primary)" in names
    )
    for r in rows:
        assert r["mde_holm_worst"] > r["mde_unadjusted"] > 0
        if r["power_holm_worst"] == r["power_holm_worst"]:  # not NaN
            assert r["power_holm_worst"] <= r["power_unadjusted"]


def test_more_draws_raise_the_power_of_the_stratum_hypotheses():
    few = P.family1_rows(DESIGNS, 972, 2)
    many_designs = {
        k: P.StratumDesign(v.coverage, v.sets_per_covered_problem * 4)
        for k, v in DESIGNS.items()
    }
    many = P.family1_rows(many_designs, 972, 2)
    for a, b in zip(few, many, strict=True):
        if a["hypothesis"] in ("H1a", "H1b"):
            assert a["sd"] == b["sd"]  # the headline sd is a fixed placeholder
        else:
            assert b["sd"] < a["sd"] and b["mde_holm_worst"] < a["mde_holm_worst"]


def test_the_noise_only_row_shows_what_more_draws_can_buy_for_h1():
    few = P.family1_rows(DESIGNS, 972, 2, draws=4)[-1]
    many = P.family1_rows(DESIGNS, 972, 2, draws=16)[-1]
    assert few["hypothesis"] == "H1a, noise-only SD"
    assert few["sd"] == pytest.approx(0.1768, abs=1e-3)  # sqrt(0.25 / 8)
    assert many["sd"] == pytest.approx(0.0884, abs=1e-3)  # sqrt(0.25 / 32)
    assert many["power_holm_worst"] > 0.99 > few["power_holm_worst"]
    assert "draws" not in P.family1_rows(DESIGNS, 972, 2)[-1]["hypothesis"]


def test_a_pooled_h3_is_far_better_powered_than_the_stratum_wise_intersection():
    rows = {r["hypothesis"]: r for r in P.family1_rows(DESIGNS, 972, 2, draws=16)}
    pooled = rows["H3, all sets pooled (alternative), true diff 0"]
    worst_stratum = min(
        r["power_holm_worst"] for k, r in rows.items() if k.startswith("H3 (")
    )
    assert pooled["power_holm_worst"] > 0.95 > worst_stratum


def test_the_sizing_rule_picks_the_smallest_number_of_draws_that_reaches_the_target():
    measured = {
        4: {
            "H1a": (972, 0.18),
            "H1b": (972, 0.18),
            "H2": (215, 0.50),
            "H4a": (682, 0.19),
        },
        8: {
            "H1a": (972, 0.13),
            "H1b": (972, 0.13),
            "H2": (257, 0.42),
            "H4a": (706, 0.13),
        },
        16: {
            "H1a": (972, 0.09),
            "H1b": (972, 0.09),
            "H2": (299, 0.33),
            "H4a": (720, 0.08),
        },
    }
    out = P.draws_by_rule(measured)
    assert out["draws"] == 16 and out["underpowered"] == []
    assert out["power"][4]["H1a"] < 0.70 <= min(out["power"][16].values())


def test_the_sizing_rule_falls_back_to_the_most_draws_and_names_what_is_short():
    measured = {
        4: {"H1a": (972, 0.2), "H2": (215, 0.5), "H4a": (682, 0.25)},
        16: {"H1a": (972, 0.2), "H2": (299, 0.33), "H4a": (720, 0.25)},
    }
    out = P.draws_by_rule(measured)
    assert out["draws"] == 16 and out["underpowered"] == ["H1a", "H4a"]


def test_the_sizing_rule_stops_at_the_first_number_of_draws_that_is_enough():
    easy = {"H1a": (5000, 0.1), "H2": (2000, 0.2), "H4a": (4000, 0.05)}
    assert P.draws_by_rule({4: easy, 8: easy, 16: easy})["draws"] == 4
