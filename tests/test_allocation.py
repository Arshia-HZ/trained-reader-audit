"""The exploratory allocation analysis (PREREG R20, E1): ESC over a set's own order, the
policies' credit and cost, and the tables on a mock dev run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.runner import RunConfig

from helpers import write_pool_shard
from reader_audit import allocation as al
from reader_audit.baselines import early_stop_vote
from reader_audit.dev_job import run_dev_reader
from reader_audit.pools import load_pools
from reader_audit.subsets import AnswerGroups

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "24_dev_allocation.py"


def groups_of(labels, correct):
    sizes = [labels.count(i) for i in range(len(correct))]
    return AnswerGroups(
        sizes=tuple(sizes),
        correct=tuple(correct),
        n_none=labels.count(None),
        labels=tuple(labels),
    )


def test_esc_on_order_matches_the_baseline_rule_and_says_why_it_stopped():
    g = groups_of([0, 1, 1, 1, 0, 0, None, 1], [True, False])
    for window, cap in [(2, 8), (4, 8), (8, 8), (3, 7)]:
        for order in ([0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 4, 5, 1, 7, 6], [6, 6, 0, 1]):
            drawn, credit, agreed = al.esc_on_order(g, order, window, cap)
            assert (drawn, credit) == early_stop_vote(order, g, window, cap)
            last = [g.labels[i] for i in order[max(drawn - window, 0) : drawn]]
            assert agreed == (
                len(last) == window and last[0] is not None and len(set(last)) == 1
            )
    # samples 1 and 2 agree on the wrong answer: stop after 4 (window 2), credit 0
    assert al.esc_on_order(g, [0, 4, 1, 2, 3], 2, 8) == (2, 1.0, True)
    assert al.esc_on_order(g, [0, 1, 2, 3], 2, 8) == (4, 0.0, True)
    assert al.esc_on_order(g, [0, 1, 4, 2], 2, 4) == (4, 0.5, False)


def view(labels_shown, *, read=1.0, unanimous=False, top=4, vote=0.0, read_cost=2.0):
    n = len(labels_shown)
    prefix = tuple({r: float(i + 1) for r in al.RULES} for i in range(n))
    reads = (al.Read(bool(read), "x", 10, 10),)
    return al.SetView(
        problem_id="p",
        draw=0,
        positions=tuple(range(n)),
        stratum="c1",
        top=top,
        unanimous=unanimous,
        vote=vote,
        prefix_cost=prefix,
        reads=reads,
        frees=(),
        follows=0.0,
        shown=0.0,
        read_cost={r: read_cost for r in al.RULES},
    )


def test_policies_pay_for_a_read_only_where_they_read():
    g = groups_of([0, 0, 1, 1, 0, 1, 1, 0], [True, False])
    pol = al.policies(g)
    v = view(list(range(8)), read=1.0, vote=0.5, top=4)
    assert pol["always_read"](v, "generated") == (1.0, 10.0, True)
    assert pol["vote_k"](v, "generated") == (0.5, 8.0, False)
    assert pol["unanimity_gate"](v, "generated") == (1.0, 10.0, True)
    u = view(list(range(8)), read=0.0, vote=1.0, top=8, unanimous=True)
    assert pol["unanimity_gate"](u, "generated") == (1.0, 8.0, False)
    assert pol["gate_top>=5"](v, "generated")[2] is True
    assert pol["gate_top>=5"](u, "generated") == (1.0, 8.0, False)
    # ESC window 2 over the shown order: samples 0 and 1 agree on the right answer
    assert pol["esc_w2_then_read"](v, "generated") == (1.0, 2.0, False)
    # alternating answers never give a window of two equal ones: the read is used
    g_alt = groups_of([0, 1, 0, 1, 0, 1, 0, 1], [True, False])
    assert al.policies(g_alt)["esc_w2_then_read"](v, "generated") == (1.0, 10.0, True)


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=30, replicates=2, n_pool=6, base_p=0.5)
    pools = load_pools(root)
    cfg = RunConfig(model_id="mock/dev", max_tokens=512, context_limit=100_000)
    problems = [p.problem for p in pools.values()]
    effects = {"reader_full_k4": 0.3, "candidate_free_k4": 0.0}
    backend = MockBackend(MockWorld.from_problems(problems, 0.5), effects)
    out = tmp_path_factory.mktemp("dev")
    report = run_dev_reader(
        out,
        pools,
        cfg,
        backend=backend,
        k=4,
        n_draws=6,
        replicates=2,
        shard_size=10,
        cap=None,
    )
    spec = importlib.util.spec_from_file_location("dev_allocation", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.K = 4
    reads = script.load_reads([out])
    views = {
        pid: al.build_views(
            pool, 4, reads[(pid, "reader_full")], reads.get((pid, "candidate_free"))
        )
        for pid, pool in pools.items()
        if reads.get((pid, "reader_full"))
    }
    assert report["shards"]
    return pools, views


def test_tables_on_a_mock_run(mock_run):
    pools, views = mock_run
    flat = [v for vs in views.values() for v in vs]
    assert len(views) == len(pools) and len(flat) == 6 * len(pools)
    strata = al.stratum_table(flat, n_boot=200, seed=1)
    # the planted reader effect shows up against its own candidate-free solve
    assert strata["all"]["read_minus_free"]["mean"] > 0.1
    lo, hi = strata["all"]["read_minus_free"]["ci"]
    assert lo <= strata["all"]["read_minus_free"]["mean"] <= hi
    agreement = al.agreement_table(flat, n_boot=50, seed=1)
    assert abs(sum(r["share_of_sets"] for r in agreement.values()) - 1.0) < 1e-9
    baselines = {pid: al.baseline_points(pools[pid], n_perm=50) for pid in views}
    table = al.policy_table(views, pools, baselines, n_boot=50, seed=1)
    for rule in al.RULES:
        t = table[rule]
        assert (
            t["always_read"]["read_share"] == 1.0 and t["vote_k"]["read_share"] == 0.0
        )
        assert (
            t["vote_k"]["cost"]
            < t["unanimity_gate"]["cost"]
            <= t["always_read"]["cost"]
        )
        assert 0.0 <= t["esc_w2_then_read"]["read_share"] <= 1.0
        assert np.isfinite(t["vote_k"]["minus_frontier"]["mean"])
    # under the generated rule a vote of 4 costs about 4 samples
    assert abs(table["generated"]["vote_k"]["cost"] - 4.0) < 1.0
