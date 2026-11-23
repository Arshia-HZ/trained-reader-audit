"""E4 and the headroom figures (PREREG R26): the read-against-fresh table by stratum,
the cascade oracles, pass@m, and the script on a fake pilot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import write_pool_shard
from reader_audit import allocation as al
from reader_audit import read_strata as rs
from reader_audit import think_frontier as tf
from reader_audit.cand_sets import draw_set, members
from reader_audit.pools import load_pools


def record(pid="p", arm="reader_full", draw=0, correct=True, answer="7", **kw):
    defaults = {"think_tokens": 100, "prompt_tokens": 50, "completion_tokens": 120}
    defaults.update(kw)
    return rs.ThinkingRecord(
        pid,
        arm,
        draw,
        correct,
        answer,
        defaults["think_tokens"],
        defaults["prompt_tokens"],
        defaults["completion_tokens"],
        defaults.get("thinking", "Wait, let me check. But fine."),
    )


def test_epistemic_density_counts_whole_words_per_thousand():
    assert rs.epistemic_density("") == 0.0
    # 10 words, three matches: "Wait", "let me check", "but"; "butter" is not one
    text = "Wait so let me check this butter and but then"
    assert rs.epistemic_density(text) == pytest.approx(1000 * 3 / 10)


def test_length_credit_and_copy():
    closed = record(think_tokens=80, completion_tokens=200)
    unclosed = record(think_tokens=None, completion_tokens=500)
    assert rs.thinking_length(closed) == 80 and rs.thinking_length(unclosed) == 500
    assert rs.credit_at(record(completion_tokens=20000)) == 0.0
    assert rs.credit_at(record(correct=True, completion_tokens=100)) == 1.0
    assert rs.copies("7", ["3", None, "7"]) and not rs.copies("7", ["3", None])
    assert not rs.copies(None, ["7"])


def test_strata_table_pairs_reads_with_their_problem():
    def q(credit, tokens, density, copy):
        return dict(credit=credit, think_tokens=tokens, density=density, copy=copy)

    rows = [
        {"problem_id": "a", "stratum": "c0", "read": q(0.0, 10.0, 1.0, 1.0),
         "fresh": q(0.5, 30.0, 2.0, 0.0)},
        {"problem_id": "a", "stratum": "c0", "read": q(1.0, 20.0, 1.0, 0.0),
         "fresh": q(0.5, 30.0, 2.0, 0.0)},
        {"problem_id": "b", "stratum": "c2+", "read": q(1.0, 40.0, 3.0, 1.0),
         "fresh": q(0.25, 50.0, 3.0, 1.0)},
    ]  # fmt: skip
    table = rs.strata_table(rows, 50, 0)
    assert table["c0"]["n_reads"] == 2 and table["c0"]["n_problems"] == 1
    assert table["c0"]["credit"]["diff"] == pytest.approx(0.0)
    assert table["c0"]["think_tokens"]["diff"] == pytest.approx(-15.0)
    assert table["c1"]["n_reads"] == 0
    assert table["c2+"]["credit"]["diff"] == pytest.approx(0.75)
    # pooled: the mean over problems of the per-problem difference
    assert table["all"]["credit"]["diff"] == pytest.approx((0.0 + 0.75) / 2)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=16, replicates=2, n_pool=6, base_p=0.5)
    return load_pools(root)


def test_oracles_are_never_below_the_cascade(pools):
    rng = np.random.default_rng(1)
    for pool in pools.values():
        solves = [
            tf.Solve(bool(rng.random() < 0.6), "1", 100, int(rng.integers(500, 9000)))
            for _ in range(4)
        ]
        for rule in al.RULES:
            out = tf.cascade_oracles(pool, 4, 6, solves, 30000, rule)
            assert out["gate"][1] >= out["dart"][1] - 1e-12
            assert out["gate"][1] >= out["agree"][1] - 1e-12
            think = tf.escalation_by_solve(solves, 30000, al.unit_costs(pool))[0]
            sample = np.mean([c.correct for c in pool.candidates])
            assert out["mode"][1] == pytest.approx(max(think, sample))


def test_pass_at():
    solves = [tf.Solve(c, "1", 0, 10) for c in (True, False, False, False)]
    assert tf.pass_at(solves, 1) == pytest.approx(0.25)
    assert tf.pass_at(solves, 2) == pytest.approx(3 / 6)
    assert tf.pass_at(solves, 4) == 1.0


def test_the_script_on_a_fake_pilot(pools, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "28_read_strata.py"
    spec = importlib.util.spec_from_file_location("read_strata_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.K, script.DRAWS = 4, 6
    script.LEVEL = next(iter(pools.values())).problem.level
    script.load_pools = lambda root: pools
    shard = tmp_path / "pilot" / "shard_000"
    shard.mkdir(parents=True)
    rng = np.random.default_rng(3)
    at_level = [p for p, v in pools.items() if v.problem.level == script.LEVEL]
    with (shard / "generations.jsonl").open("w", encoding="utf-8") as f:
        for pid in at_level:
            pool = pools[pid]
            rows = [("candidate_free", -1, s) for s in range(4)] + [
                ("reader_full", 0, 0)
            ]
            for arm, draw, rep in rows:
                ok = bool(rng.random() < 0.6)
                f.write(
                    json.dumps(
                        {
                            "problem_id": pid,
                            "arm": arm,
                            "draw": draw,
                            "replicate": rep,
                            "correct": ok,
                            "extracted": pool.problem.gold if ok else "0",
                            "think_tokens": int(rng.integers(100, 5000)),
                            "prompt_tokens": 120,
                            "completion_tokens": int(rng.integers(5000, 9000)),
                            "text": "Hmm, wait. But maybe.</think>The answer.",
                        }
                    )
                    + "\n"
                )
    (shard / "DONE.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "strata.json"
    argv = ["--pools-root", "x", "--pilot", str(tmp_path / "pilot"), "--out", str(out)]
    assert script.main([*argv, "--md", str(tmp_path / "s.md"), "--n-boot", "50"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["reads"] == len(at_level)
    expect = {s: 0 for s in ("c0", "c1", "c2+")}
    for pid in at_level:
        expect[draw_set(pools[pid], 4, 0).stratum] += 1
    assert report["strata_counts"] == expect
    assert report["strata"]["all"]["density"]["read"] > 0
    for rule in al.RULES:
        assert set(report["headroom"][rule]) == {"dart", "gate", "agree", "mode"}
    assert report["pass_at"]["1"] <= report["pass_at"]["2"] <= report["pass_at"]["4"]
    # the read's copy flag is checked against the set it was shown
    pid = at_level[0]
    shown = [c.answer for c in members(pools[pid], draw_set(pools[pid], 4, 0))]
    rows = script.rs.read_rows(
        pools, {pid: [record(pid, "candidate_free", -1)] * 4}, [record(pid)], 4
    )
    assert rows[0]["read"]["copy"] == float(rs.copies("7", shown))
