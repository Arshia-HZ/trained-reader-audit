"""The thinking pilot's frontier (PREREG R24, E3): solves at lower caps, the thinking
vote, the cascades, and the script on a fake pilot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import write_pool_shard
from reader_audit import allocation as al
from reader_audit import think_frontier as tf
from reader_audit.pools import load_pools

UNITS = {"generated": 1000.0, "prompt_inclusive": 1100.0}


def test_a_solve_at_a_lower_cap_is_cut_and_wrong():
    s = tf.Solve(True, "1", 100, 5000)
    assert tf.at_cap(s, 8000) == (1.0, 5000)
    assert tf.at_cap(s, 4096) == (0.0, 4096)
    assert tf.solve_cost(s, 4096, UNITS) == {
        "generated": 4.096,
        "prompt_inclusive": (100 + 4096) / 1100,
    }


def test_the_thinking_vote_averages_over_subsets():
    solves = [tf.Solve(a == "7", a, 100, 1000) for a in ("7", "7", "5", None)]
    cost, credit = tf.think_vote(solves, 4, "7", UNITS, "generated")
    assert credit == 1.0 and cost == pytest.approx(4.0)
    cost, credit = tf.think_vote(solves, 2, "7", UNITS, "generated")
    # pairs: (7,7) 1; (7,5) x2 0.5; (7,None) x2 1; (5,None) 0
    assert credit == pytest.approx((1 + 0.5 + 0.5 + 1 + 1 + 0) / 6)
    assert cost == pytest.approx(2.0)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=16, replicates=2, n_pool=6, base_p=0.5)
    return load_pools(root)


def test_cascade_escalates_exactly_where_esc_finds_no_pair(pools):
    for pool in pools.values():
        groups = al.group_answers(
            [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
        )
        cost, credit, up = tf.think_cascade(
            pool,
            4,
            6,
            4,
            (1.0, {"generated": 9.0, "prompt_inclusive": 9.0}),
            "generated",
        )
        expect = np.mean(
            [
                not al.esc_on_order(groups, al.draw_set(pool, 4, d).positions, 2, 4)[2]
                for d in range(6)
            ]
        )
        assert up == pytest.approx(expect)
        assert credit >= 0 and cost > 0
        if up == 1.0:
            assert credit == 1.0 and cost > 9.0


def test_a_policy_on_the_envelope_is_not_above_it(pools):
    pids = sorted(pools)
    env = {"a": {p: (2.0, 0.5) for p in pids}, "b": {p: (4.0, 0.7) for p in pids}}
    boots = np.random.default_rng(0).integers(0, len(pids), (50, len(pids)))
    on = tf.minus_envelope({p: (3.0, 0.6) for p in pids}, env, pids, boots)
    assert on["minus_envelope"] == pytest.approx(0.0)
    above = tf.minus_envelope({p: (3.0, 0.65) for p in pids}, env, pids, boots)
    assert above["minus_envelope"] == pytest.approx(0.05)
    outside = tf.minus_envelope({p: (5.0, 0.9) for p in pids}, env, pids, boots)
    assert not outside["estimable"]


def test_the_script_on_a_fake_pilot(pools, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "27_thinking_frontier.py"
    spec = importlib.util.spec_from_file_location("thinking_frontier", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.K, script.DRAWS = 4, 6
    script.LEVEL = next(iter(pools.values())).problem.level
    pilot = tmp_path / "pilot" / "shard_000"
    pilot.mkdir(parents=True)
    rng = np.random.default_rng(3)
    with (pilot / "generations.jsonl").open("w", encoding="utf-8") as f:
        for pid, pool in pools.items():
            for s in range(4):
                ok = bool(rng.random() < 0.7)
                f.write(
                    json.dumps(
                        {
                            "problem_id": pid,
                            "arm": "candidate_free",
                            "draw": -1,
                            "replicate": s,
                            "correct": ok,
                            "extracted": pool.problem.gold if ok else "0",
                            "prompt_tokens": 150,
                            "completion_tokens": int(rng.integers(500, 20000)),
                        }
                    )
                    + "\n"
                )
            f.write(
                json.dumps(
                    {
                        "problem_id": pid,
                        "arm": "reader_full",
                        "draw": 0,
                        "replicate": 0,
                        "correct": True,
                        "extracted": "1",
                        "prompt_tokens": 4000,
                        "completion_tokens": 3000,
                    }
                )
                + "\n"
            )
    summary = {
        "arms": {"candidate_free": {"n": 4, "seconds": 10.0, "completion_tokens": 100}}
    }
    (pilot / "DONE.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "frontier.json"
    script.load_pools = lambda root: pools
    assert (
        script.main(
            [
                "--pools-root",
                "x",
                "--pilot",
                str(tmp_path / "pilot"),
                "--out",
                str(out),
                "--md",
                str(tmp_path / "f.md"),
                "--n-boot",
                "50",
            ]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    at_level = sum(p.problem.level == script.LEVEL for p in pools.values())
    assert report["problems"] == at_level
    for rule in al.RULES:
        assert "think_1_cap30000" in report["envelope"][rule]
        assert "esc_w2_m2_then_think_cap30000" in report["policies"][rule]
    contrast = report["read_minus_fresh"]["read_cap16384_minus_fresh_cap16384"]
    assert contrast["n"] == at_level
