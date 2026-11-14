"""The reader pipeline end to end on the audit's mock backend: plan, run, resume,
grade, and recover planted arm effects by stratum."""

from __future__ import annotations

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.runner import RunConfig

from helpers import write_pool_shard
from reader_audit import reader_eval as ev
from reader_audit.arms import swap_eligible
from reader_audit.cand_sets import STRATA, draw_set
from reader_audit.pools import load_pools
from reader_audit.reader_run import (
    check_run,
    open_reader_run,
    plan_reader_run,
    run_reader,
)

ARMS = ("reader_full", "candidate_free", "answer_only", "placebo", "answer_swap")
K, DRAWS, REPS = 4, 4, 2
EFFECTS = {
    "reader_full_k4": 0.3,
    "candidate_free_k4": 0.0,
    "answer_only_k4": 0.1,
    "placebo_k4": -0.3,
    "answer_swap_k4": 0.3,
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=30, replicates=2, n_pool=6, base_p=0.5)
    pools = load_pools(root)
    cfg = RunConfig(model_id="mock/reader", max_tokens=512, context_limit=100_000)
    plan = plan_reader_run(
        pools, cfg, arms=ARMS, ks=(K,), n_draws=DRAWS, replicates=REPS
    )
    problems = [p.problem for p in pools.values()]
    backend = MockBackend(MockWorld.from_problems(problems, base_p=0.5), EFFECTS)
    out = tmp_path_factory.mktemp("run") / "reader"
    ctx = open_reader_run(out, cfg, problems, backend)
    n_new = run_reader(ctx, plan)
    return {
        "pools": pools,
        "cfg": cfg,
        "plan": plan,
        "ctx": ctx,
        "n_new": n_new,
        "problems": problems,
        "out": out,
        "backend": backend,
    }


def test_every_planned_generation_is_made_and_graded(world):
    plan, ctx = world["plan"], world["ctx"]
    assert world["n_new"] == len(plan.planned)
    assert check_run(ctx, plan) == []
    keys = [r.gen_key for r in plan.rows]
    assert len(set(keys)) == len(keys)


def test_the_count_follows_the_formula_with_swap_only_where_eligible(world):
    pools, plan = world["pools"], world["plan"]
    per_arm = {a: sum(r.arm == a for r in plan.rows) for a in ARMS}
    n_sets = len(pools) * DRAWS
    for arm in ("reader_full", "candidate_free", "answer_only", "placebo"):
        assert per_arm[arm] == n_sets * REPS
    eligible = sum(
        swap_eligible(pool, draw_set(pool, K, d)) is not None
        for pool in pools.values()
        for d in range(DRAWS)
    )
    assert 0 < eligible < n_sets
    assert per_arm["answer_swap"] == eligible * REPS


def test_a_second_run_makes_nothing_and_does_not_duplicate_the_plan(world):
    ctx = open_reader_run(
        world["out"], world["cfg"], world["problems"], world["backend"]
    )
    assert run_reader(ctx, world["plan"]) == 0
    outcomes = ev.load_outcomes(world["out"])
    assert len(outcomes) == len(world["plan"].rows)


def test_prompts_that_differ_never_share_an_identity(world):
    ctx, plan = world["ctx"], world["plan"]
    hashes: dict[tuple, set] = {}
    for p in plan.planned:
        ident = p.identity
        hashes.setdefault((ident["problem_id"], ident["agent"]), set()).add(
            (ident["arm"], ident["prompt_hash"])
        )
    for cells in hashes.values():
        arms = [a for a, _ in cells]
        assert len(arms) == len(set(arms))  # one prompt hash per arm and set
        assert len({h for _, h in cells}) >= 4  # the arms' prompts differ
    assert len(ctx.gens) == len(plan.planned)


def test_planted_effects_are_recovered_in_every_stratum(world):
    outcomes = ev.load_outcomes(world["out"])
    for stratum in (None, *STRATA):
        gain = ev.contrast(outcomes, "reader_full", "candidate_free", K, stratum)
        if gain is None or gain["n_problems"] < 12:
            continue
        assert gain["mean"] == pytest.approx(0.3, abs=0.15)
        assert gain["ci95_low"] < 0.3 < gain["ci95_high"] + 0.05
    overall = ev.contrast(outcomes, "reader_full", "candidate_free", K)
    assert overall["ci95_low"] > 0.1
    placebo = ev.contrast(outcomes, "placebo", "candidate_free", K)
    assert placebo["mean"] == pytest.approx(-0.3, abs=0.12)
    answers = ev.contrast(outcomes, "answer_only", "candidate_free", K)
    assert answers["mean"] == pytest.approx(0.1, abs=0.12)


def test_contrasts_pair_only_the_cells_both_arms_have(world):
    outcomes = ev.load_outcomes(world["out"])
    swap_cells = set(ev.cell_means(outcomes, "answer_swap", K))
    full_cells = set(ev.cell_means(outcomes, "reader_full", K))
    assert swap_cells < full_cells
    diffs = ev.paired_diffs(outcomes, "answer_swap", "reader_full", K)
    assert {pid for pid, _ in swap_cells} >= set(diffs)
    assert ev.paired_diffs(outcomes, "answer_swap", "reader_full", K, "c0") == {}
    assert ev.contrast(outcomes, "answer_swap", "reader_full", K, "c0") is None


def test_strata_come_from_the_candidate_sets_not_from_the_reader(world):
    outcomes = ev.load_outcomes(world["out"])
    pools = world["pools"]
    for o in outcomes:
        cset = draw_set(pools[o.problem_id], o.k, o.draw)
        assert (o.stratum, o.n_correct) == (cset.stratum, cset.n_correct)


def test_the_plain_vote_is_an_arm_with_one_outcome_per_set(world):
    votes = ev.vote_outcomes(world["pools"], (K,), DRAWS)
    assert len(votes) == len(world["pools"]) * DRAWS
    assert all(0.0 <= v.value <= 1.0 and v.arm == ev.VOTE for v in votes)
    outcomes = ev.load_outcomes(world["out"]) + votes
    diff = ev.contrast(outcomes, ev.VOTE, "reader_full", K)
    assert diff is not None and diff["n_problems"] == len(world["pools"])


def test_unknown_arms_are_refused_when_planning(world):
    with pytest.raises(ValueError, match="unknown reader arms"):
        plan_reader_run(
            world["pools"],
            world["cfg"],
            arms=("telepathy",),
            ks=(K,),
            n_draws=1,
            replicates=1,
        )
