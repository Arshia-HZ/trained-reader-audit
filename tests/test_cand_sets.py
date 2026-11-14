"""Candidate sets: deterministic seeded draws and strata."""

from __future__ import annotations

import pytest

from helpers import write_pool_shard
from reader_audit.cand_sets import (
    STRATA,
    draw_set,
    draw_sets,
    members,
    stratum_of,
    vote_credit,
)
from reader_audit.pools import load_pools


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=6, replicates=2, n_pool=6)
    return load_pools(root)


def test_a_set_is_k_distinct_pool_members_and_is_reproducible(pools):
    pool = next(iter(pools.values()))
    a, b = draw_set(pool, 8, 0), draw_set(pool, 8, 0)
    assert a == b
    assert len(set(a.positions)) == 8 and all(0 <= i < pool.n for i in a.positions)


def test_draws_k_and_problems_give_different_sets(pools):
    first, second = list(pools.values())[:2]
    assert draw_set(first, 4, 0).positions != draw_set(first, 4, 1).positions
    assert draw_set(first, 4, 0).positions != draw_set(first, 5, 0).positions[:4]
    assert draw_set(first, 4, 0).positions != draw_set(second, 4, 0).positions


def test_strata_follow_the_number_of_correct_candidates():
    assert [stratum_of(n) for n in (0, 1, 2, 5)] == ["c0", "c1", "c2+", "c2+"]
    assert STRATA == ("c0", "c1", "c2+")
    with pytest.raises(ValueError):
        stratum_of(-1)


def test_the_stored_count_matches_the_members(pools):
    for pool in pools.values():
        for cset in draw_sets(pool, 4, 6):
            assert cset.n_correct == sum(c.correct for c in members(pool, cset))
            assert cset.stratum == stratum_of(cset.n_correct)


def test_a_full_pool_draw_contains_everyone_in_some_order(pools):
    pool = next(iter(pools.values()))
    cset = draw_set(pool, pool.n, 0)
    assert sorted(cset.positions) == list(range(pool.n))
    assert cset.n_correct == sum(c.correct for c in pool.candidates)


def test_the_share_of_correct_candidates_in_sets_matches_the_pool(pools):
    pool = max(pools.values(), key=lambda p: sum(c.correct for c in p.candidates))
    share = sum(c.correct for c in pool.candidates) / pool.n
    drawn = [draw_set(pool, 4, d).n_correct / 4 for d in range(400)]
    assert sum(drawn) / len(drawn) == pytest.approx(share, abs=0.05)


def test_k_outside_the_pool_is_an_error(pools):
    pool = next(iter(pools.values()))
    with pytest.raises(ValueError):
        draw_set(pool, pool.n + 1, 0)
    with pytest.raises(ValueError):
        draw_set(pool, 0, 0)


def test_vote_credit_is_one_when_every_member_is_right(pools):
    for pool in pools.values():
        right = [i for i, c in enumerate(pool.candidates) if c.correct]
        if len(right) >= 3:
            from reader_audit.cand_sets import CandidateSet

            cset = CandidateSet(pool.problem.problem_id, 3, 0, tuple(right[:3]), 3)
            assert vote_credit(pool, cset) == 1.0
            return
    pytest.skip("no mock problem with three correct samples")
