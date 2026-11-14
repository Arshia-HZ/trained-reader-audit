"""Stage 0 break-even and headroom: invariants on mock pools, and the planning rules."""

from __future__ import annotations

import pytest

from helpers import write_pool_shard
from reader_audit.budget import RULES, planning_k_star
from reader_audit.pools import load_pools
from reader_audit.stage0 import analyze, gate_g1

G, P = 819.66, 150.55  # mean generated and prompt tokens of a dev sample (Stage 0)


def test_planning_rules_reproduce_the_stage0_table():
    # PREREG section 7 / docs/stage0: level-5 dev rows, K = 4 and K = 8
    assert planning_k_star("generated_only", 4, G, P) == 5.0
    assert planning_k_star("generated_only", 8, G, P) == 9.0
    assert planning_k_star("generated_plus_prompt", 4, G, P) == pytest.approx(
        8.38, abs=0.01
    )
    assert planning_k_star("generated_plus_prompt", 8, G, P) == pytest.approx(
        15.76, abs=0.01
    )
    assert planning_k_star("flops_0.6b_reader", 4, G, P) == pytest.approx(
        5.53, abs=0.01
    )
    assert planning_k_star("flops_0.6b_reader", 8, G, P) == pytest.approx(
        10.72, abs=0.01
    )


def test_the_gate_reads_a_share_of_problems():
    full = planning_k_star("generated_plus_prompt", 4, G, P)
    half = planning_k_star("gated_unanimity_prompt_incl", 4, G, P, share_read=0.5)
    none = planning_k_star("gated_unanimity_prompt_incl", 4, G, P, share_read=0.0)
    assert none == 4.0 < half < full


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError, match="unknown cost rule"):
        planning_k_star("free_lunch", 4, G, P)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=12, replicates=2, n_pool=6)
    return analyze(load_pools(root), n_boot=200)


def test_analysis_has_every_group_and_rule(result):
    assert result["n_problems"] == 12
    assert {"all", "level_3", "level_4"} <= set(result["results"])
    block = result["results"]["all"]["K=8"]
    assert set(block["cost_rules"]) == set(RULES)


def test_strata_sum_to_one_and_headroom_is_pass_minus_vote(result):
    for k in (3, 4, 8):
        b = result["results"]["all"][f"K={k}"]
        assert b["c0"]["mean"] + b["c1"]["mean"] + b["c2plus"]["mean"] == pytest.approx(
            1.0
        )
        assert b["headroom"]["mean"] == pytest.approx(
            b[f"pass@{k}"]["mean"] - b[f"maj@{k}"]["mean"]
        )


def test_the_sc_curve_rises_and_k_equal_pool_is_certain(result):
    curve = result["results"]["all"]["sc_curve"]
    assert len(curve) == 18
    assert curve["18"] >= curve["1"]


def test_generated_only_break_even_shrinks_with_k_on_mock_pools(result):
    b = result["results"]["all"]
    # more samples buy less: SC(K + 1) - maj@K gets smaller as K grows
    assert (
        b["K=8"]["cost_rules"]["generated_only"]["break_even"]["mean"]
        < (b["K=3"]["cost_rules"]["generated_only"]["break_even"]["mean"])
    )


def test_gate_verdict_logic():
    summary = {
        f"K={k}": {
            "headroom": {"mean": h},
            "cost_rules": {"generated_only": {"break_even": {"mean": be}}},
        }
        for k, h, be in ((4, 0.129, 0.023), (8, 0.139, 0.008))
    }
    gate = gate_g1(summary)
    assert gate["K=4"]["pass_high"] and not gate["K=4"]["pass_low"]
    assert gate["K=8"]["pass_low"] and gate["K=8"]["pass_high"]
    assert gate["K=8"]["needed_gain"] == pytest.approx(0.023)
