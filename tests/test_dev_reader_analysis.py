"""The dev reader analysis on a mock run: the planted reader effects come out by stratum
and the sizing rule consumes what the analysis measures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.runner import RunConfig

from helpers import write_pool_shard
from reader_audit import power
from reader_audit import reader_eval as ev
from reader_audit.dev_job import run_dev_reader, shard_name
from reader_audit.pools import load_pools
from reader_audit.subsets import group_answers, sc_curve

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "13_dev_reader_analysis.py"
K, DRAWS = 4, 8


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("dev_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=40, replicates=2, n_pool=6, base_p=0.5)
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
        k=K,
        n_draws=DRAWS,
        replicates=2,
        shard_size=15,
        cap=None,
    )
    outcomes = []
    for name in report["shards"]:
        outcomes += ev.load_outcomes(out / name)
    curves = {
        pid: sc_curve(
            group_answers(
                [c.answer for c in pool.candidates],
                [c.correct for c in pool.candidates],
            )
        )
        for pid, pool in pools.items()
    }
    return module, outcomes, pools, curves


def test_the_summary_covers_every_problem_and_the_requested_draws(analysis):
    module, outcomes, pools, curves = analysis
    out, measured = module.summarise(outcomes, pools, curves, 4, k=K)
    assert out["problems"] == 40 and out["draws"] == 4
    assert sum(out["sets"].values()) == 40 * 4
    assert set(measured) == {"H1a", "H1b", "H2", "H4a"}


def test_more_draws_never_lower_the_stratum_coverage(analysis):
    module, outcomes, pools, curves = analysis
    few, _ = module.summarise(outcomes, pools, curves, 2, k=K)
    many, _ = module.summarise(outcomes, pools, curves, 8, k=K)
    assert many["H4a"]["coverage"] >= few["H4a"]["coverage"]
    assert many["H2_single_arm"]["coverage"] >= few["H2_single_arm"]["coverage"]


def test_the_planted_reader_effect_shows_at_c1_and_the_rule_runs(analysis):
    module, outcomes, pools, curves = analysis
    out, _ = module.summarise(outcomes, pools, curves, 8, k=K)
    assert out["H2_single_arm"]["mean"] > 0  # candidates help the planted reader
    assert out["reader_generated_tokens_mean"] > 0
    measured = {d: module.summarise(outcomes, pools, curves, d, k=K)[1] for d in (4, 8)}
    rule = power.draws_by_rule(measured)
    assert rule["draws"] in (4, 8) and set(rule["power"]) == {4, 8}


def test_the_shard_names_follow_the_job(analysis):
    assert shard_name(2) == "shard_002"
