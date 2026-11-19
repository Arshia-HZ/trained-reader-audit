"""The thinking probe's helpers (PREREG R22, E2): the escalation slice, the hashed
choice of problems, ESC continued over a set's seeded order, the grade after </think>."""

from __future__ import annotations

import pytest

from helpers import write_pool_shard
from reader_audit import think_probe as tp
from reader_audit.allocation import esc_on_order
from reader_audit.cand_sets import draw_set
from reader_audit.pools import load_pools


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=20, replicates=2, n_pool=6, base_p=0.4)
    return load_pools(root)


def test_escalated_draws_are_the_sets_esc_cannot_settle(pools):
    k = 4
    for pool in pools.values():
        groups = tp.pool_groups(pool)
        slice_ = tp.escalated_draws(pool, k, 8)
        for draw in range(8):
            agreed = esc_on_order(groups, draw_set(pool, k, draw).positions, 2, k)[2]
            assert (draw in slice_) == (not agreed)


def test_pick_problems_depends_on_the_ids_only():
    ids = [f"p{i}" for i in range(50)]
    a = tp.pick_problems(ids, 10)
    assert a == tp.pick_problems(list(reversed(ids)), 10)
    assert len(set(a)) == 10 and a != ids[:10]
    assert tp.pick_problems(ids, 10, salt="other") != a


def test_esc_continued_starts_with_the_shown_set_and_never_draws_less(pools):
    k = 4
    for pool in pools.values():
        groups = tp.pool_groups(pool)
        for draw in range(6):
            first = esc_on_order(groups, draw_set(pool, k, draw).positions, 2, k)
            drawn, credit = tp.esc_continued(pool, k, draw, cap=pool.n)
            assert drawn >= first[0]
            if first[2]:  # settled inside the shown set: the same stop and credit
                assert (drawn, credit) == first[:2]


def test_grade_thinking_reads_only_after_the_close():
    gold = "7"
    g = tp.grade_thinking(
        "<think>\nmaybe \\boxed{7}\n</think>\n\nSo \\boxed{5}.", "stop", gold
    )
    assert g.closed and g.extracted == "5" and not g.correct
    g = tp.grade_thinking("<think>\nwork</think>\n\\boxed{7}", "stop", gold)
    assert g.closed and g.correct and g.status == "boxed"
    g = tp.grade_thinking("<think>\nlong work \\boxed{7}", "length", gold)
    assert not g.closed and not g.correct and g.status == "unclosed"
    g = tp.grade_thinking("<think>x</think> no box", "stop", gold)
    assert g.closed and g.status == "none" and not g.correct


def test_plan_and_report_of_the_script_on_fake_generations(pools, tmp_path):
    import importlib.util
    import json
    from argparse import Namespace
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "25_thinking_probe.py"
    spec = importlib.util.spec_from_file_location("think_probe_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.K = 4
    script.LEVEL = next(iter(pools.values())).problem.level
    rows = script.plan(pools, n_problems=5, sets_per_problem=2)
    assert rows and len({r["problem_id"] for r in rows}) <= 5
    assert {r["arm"] for r in rows} == set(script.ARMS)
    out = tmp_path / "probe"
    out.mkdir()
    with (out / script.GENS).open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        **r,
                        "prompt_tokens": 100,
                        "completion_tokens": 3000,
                        "think_tokens": 2500,
                        "finish_reason": "stop",
                        "closed": True,
                        "status": "boxed",
                        "extracted": "1",
                        "correct": i % 2 == 0,
                        "text": "",
                    }
                )
                + "\n"
            )
    script.level_pools = lambda root: pools
    assert script.cmd_report(Namespace(pools_root="x", out=str(out), compare=None)) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["arms"]["think_candidate_free"]["n"] == len(rows) // 2
    assert report["arms"]["esc_continued_to_16"]["extra_samples"] >= 0


def test_seeds_differ_by_arm_and_draw():
    seeds = {tp.seed_for("p", d, a, 0) for d in range(4) for a in ("x", "y")}
    assert len(seeds) == 8
