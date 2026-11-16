"""The sharded pool job on the mock backend: counts, resume, the one-shard stop."""

from __future__ import annotations

import json

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import RunConfig

from reader_audit.pool_job import DONE, run_pool_shards, shard_name
from reader_audit.pools import load_pools
from reader_audit.train_set import train_order


@pytest.fixture
def world(tmp_path):
    problems = fake_problems(25)
    cfg = RunConfig(model_id="mock/pool", max_tokens=512, context_limit=100_000)
    backend = MockBackend(MockWorld.from_problems(problems, 0.5), {})
    return {"problems": problems, "cfg": cfg, "backend": backend, "out": tmp_path}


def run(world, **kw):
    return run_pool_shards(
        world["out"],
        world["problems"],
        world["cfg"],
        backend=world["backend"],
        shard_size=10,
        **kw,
    )


def test_every_problem_gets_sixteen_samples_in_shards_of_ten(world):
    report = run(world)
    assert report["n_shards"] == 3
    assert [s["problems"] for s in report["shards"].values()] == [10, 10, 5]
    assert all(
        s["generations"] == s["problems"] * 16 for s in report["shards"].values()
    )
    assert (world["out"] / shard_name(2) / DONE).is_file()
    pools = load_pools(world["out"])  # the audit's marker and hash check pass
    assert len(pools) == 25 and {p.n for p in pools.values()} == {16}


def test_max_shards_then_resume_gives_the_same_report_and_no_duplicates(world):
    first = run(world, max_shards=1)
    assert list(first["shards"]) == [shard_name(0)]
    full = run(world)
    assert len(full["shards"]) == 3
    assert full["shards"][shard_name(0)] == first["shards"][shard_name(0)]
    keys = [
        json.loads(x)["gen_key"]
        for x in (world["out"] / shard_name(1) / "generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(keys) == len(set(keys))


def test_a_finished_job_never_builds_a_backend(world):
    run(world)

    def boom():
        raise AssertionError("no backend for a finished job")

    again = run_pool_shards(
        world["out"],
        world["problems"],
        world["cfg"],
        backend_factory=boom,
        shard_size=10,
    )
    assert len(again["shards"]) == 3


def test_too_few_samples_are_refused(world):
    with pytest.raises(ValueError, match="more than"):
        run_pool_shards(
            world["out"], world["problems"], world["cfg"], samples=3, backend=None
        )


def test_the_training_order_is_a_seeded_permutation():
    ids = [f"p{i}" for i in range(50)]
    order = train_order(ids)
    assert sorted(order) == sorted(ids) and order == train_order(list(reversed(ids)))
    assert order != sorted(ids) and train_order(ids, seed=1) != order


def test_a_later_batch_continues_the_shard_numbering(world):
    report = run(world, first_index=4, max_shards=1)
    assert list(report["shards"]) == [shard_name(4)]
    assert (world["out"] / shard_name(4) / DONE).is_file()
    assert not (world["out"] / shard_name(0)).exists()
