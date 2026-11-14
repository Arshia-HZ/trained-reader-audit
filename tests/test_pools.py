"""Pool loading from audit-format shards."""

from __future__ import annotations

import pytest

from helpers import write_pool_shard
from reader_audit.pools import ShardIntegrityError, load_pools, pool_sizes


def test_pools_hold_round0_and_pool_samples_in_canonical_order(tmp_path):
    write_pool_shard(tmp_path, n_problems=5, replicates=2, n_pool=6)
    pools = load_pools(tmp_path)
    assert len(pools) == 5
    assert pool_sizes(pools.values()) == {18}  # 2 replicates x (3 agents + 6 pool)
    for pool in pools.values():
        order = [(c.replicate, c.arm, c.index) for c in pool.candidates]
        assert order == sorted(order, key=lambda t: (t[0], t[1] != "round0", t[2]))
        assert [c.arm for c in pool.candidates[:3]] == ["round0"] * 3


def test_candidates_carry_the_stored_grade_and_tokens(tmp_path):
    write_pool_shard(tmp_path, n_problems=3, replicates=1, n_pool=2)
    pool = next(iter(load_pools(tmp_path).values()))
    for c in pool.candidates:
        assert c.completion_tokens > 0 and not c.truncated
        assert c.correct == (c.answer == pool.problem.gold)
        assert c.confidence is not None
        assert "\\boxed" in c.text


def test_a_shard_without_a_done_marker_is_refused(tmp_path):
    shard = write_pool_shard(tmp_path, n_problems=2)
    (shard / "SHARD_DONE.json").unlink()
    with pytest.raises(ShardIntegrityError, match="not done"):
        load_pools(tmp_path)
    assert len(load_pools(tmp_path, verify=False)) == 2


def test_a_changed_generations_file_is_refused(tmp_path):
    shard = write_pool_shard(tmp_path, n_problems=2)
    path = shard / "generations.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ShardIntegrityError, match="changed"):
        load_pools(tmp_path)


def test_a_problem_in_two_shards_is_refused(tmp_path):
    write_pool_shard(tmp_path, name="shard_00", n_problems=3)
    write_pool_shard(tmp_path, name="shard_01", n_problems=3)  # same fake ids
    with pytest.raises(ValueError, match="two shards"):
        load_pools(tmp_path)


def test_shards_with_different_problems_combine(tmp_path):
    write_pool_shard(tmp_path, name="shard_00", n_problems=3)
    write_pool_shard(tmp_path, name="shard_01", n_problems=3, first_problem=3)
    assert len(load_pools(tmp_path)) == 6


def test_an_empty_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pools(tmp_path)
