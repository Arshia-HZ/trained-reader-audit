"""Training rows: set selection, rendering, the solver rows and the reward."""

from __future__ import annotations

import pytest
from channel_audit.extract import extract_answer
from channel_audit.grade import grade_generation

from helpers import write_pool_shard
from reader_audit.cand_sets import draw_set
from reader_audit.pools import load_pools
from reader_audit.train_data import (
    HARD,
    TRAIN_KS,
    correctness_reward,
    reader_rows,
    select_sets,
    solver_rows,
)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=30, replicates=2, n_pool=6, base_p=0.5)
    return load_pools(root)


def strata(sets):
    out = {}
    for s in sets:
        out[s.stratum] = out.get(s.stratum, 0) + 1
    return out


def test_every_hard_set_is_kept_and_about_half_of_the_easy_ones(pools):
    everything = select_sets(pools, easy_share=1.0, n_draws=6)
    half = select_sets(pools, easy_share=0.5, n_draws=6)
    for stratum in HARD:
        assert strata(half).get(stratum, 0) == strata(everything).get(stratum, 0)
    easy_all, easy_half = strata(everything)["c2+"], strata(half)["c2+"]
    assert 0.3 * easy_all < easy_half < 0.7 * easy_all
    assert {s.k for s in everything} <= set(TRAIN_KS)


def test_selection_is_seeded_and_the_no_c0_arm_drops_c0(pools):
    a = select_sets(pools, seed=3)
    assert [(s.problem_id, s.k, s.draw) for s in a] == [
        (s.problem_id, s.k, s.draw) for s in select_sets(pools, seed=3)
    ]
    assert [(s.problem_id, s.k, s.draw) for s in a] != [
        (s.problem_id, s.k, s.draw) for s in select_sets(pools, seed=4)
    ]
    assert "c0" not in strata(select_sets(pools, drop_c0=True))
    assert "c0" in strata(select_sets(pools))


def test_easy_share_is_validated(pools):
    with pytest.raises(ValueError):
        select_sets(pools, easy_share=1.5)


def test_reader_rows_use_the_evaluated_prompt_and_hide_who_wrote_what(pools):
    sets = select_sets(pools, n_draws=2)[:12]
    rows = reader_rows(pools, sets, cap=None)
    assert len(rows) == 12
    for row, cset in zip(rows, sets, strict=True):
        pool = pools[cset.problem_id]
        assert row["gold"] == pool.problem.gold and row["k"] == cset.k
        text = " ".join(m["content"] for m in row["prompt"])
        assert pool.problem.question in text
        assert row["stratum"] == draw_set(pool, cset.k, cset.draw).stratum


def test_solver_rows_cover_each_problem_once_per_epoch(pools):
    rows = solver_rows(pools, epochs=2)
    assert len(rows) == 2 * len(pools)
    first = [r["problem_id"] for r in rows[: len(pools)]]
    assert sorted(first) == sorted(pools) and all(r["k"] == 0 for r in rows)


def test_the_reward_is_the_audit_grade(pools):
    pool = next(iter(pools.values()))
    gold = pool.problem.gold
    cases = [
        f"so \\boxed{{{gold}}}",
        "\\boxed{99999}",
        "no final answer here",
        "",
    ]
    rewards = correctness_reward(cases, [gold] * 4)
    assert rewards == [1.0, 0.0, 0.0, 0.0]
    chat = [[{"role": "assistant", "content": cases[0]}]]
    assert correctness_reward(chat, [gold]) == [1.0]
    # the same answer the grader would record for a stored generation
    cand = pool.candidates[0]
    assert extract_answer(cand.text) == cand.answer
    assert correctness_reward([cand.text], [gold]) == [1.0 if cand.correct else 0.0]
    assert grade_generation is not None
