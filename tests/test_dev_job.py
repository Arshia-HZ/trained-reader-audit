"""The sharded dev reader job on the mock backend: shards, resume, throughput stop."""

from __future__ import annotations

import json

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.runner import RunConfig

from helpers import write_pool_shard
from reader_audit import reader_eval as ev
from reader_audit.dev_job import DONE, run_dev_reader, shard_name, split_shards
from reader_audit.pools import load_pools

K, DRAWS, REPS = 4, 3, 2
EFFECTS = {"reader_full_k4": 0.3, "candidate_free_k4": 0.0}


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "pools"
    write_pool_shard(root, n_problems=10, replicates=2, n_pool=6, base_p=0.5)
    pools = load_pools(root)
    cfg = RunConfig(model_id="mock/dev", max_tokens=512, context_limit=100_000)
    problems = [p.problem for p in pools.values()]
    backend = MockBackend(MockWorld.from_problems(problems, base_p=0.5), EFFECTS)
    return {"pools": pools, "cfg": cfg, "backend": backend, "out": tmp_path / "dev"}


def run(world, **kw):
    return run_dev_reader(
        world["out"],
        world["pools"],
        world["cfg"],
        backend=world["backend"],
        k=K,
        n_draws=DRAWS,
        replicates=REPS,
        shard_size=4,
        cap=None,
        **kw,
    )


def test_shards_are_sorted_and_cover_every_problem_once():
    shards = split_shards(["b", "a", "d", "c", "e"], 2)
    assert shards == [["a", "b"], ["c", "d"], ["e"]]
    with pytest.raises(ValueError):
        split_shards(["a"], 0)


def test_every_shard_gets_a_marker_and_the_counts_follow_the_design(world):
    report = run(world)
    assert report["n_shards"] == 3
    assert sum(s["generations"] for s in report["shards"].values()) == (
        10 * DRAWS * 2 * REPS
    )
    for i in range(3):
        assert (world["out"] / shard_name(i) / DONE).is_file()


def test_a_restart_makes_nothing_new_and_never_builds_a_backend(world):
    first = run(world)

    def boom():
        raise AssertionError("no backend for a finished job")

    again = run_dev_reader(
        world["out"],
        world["pools"],
        world["cfg"],
        backend_factory=boom,
        k=K,
        n_draws=DRAWS,
        replicates=REPS,
        shard_size=4,
        cap=None,
    )
    assert again == first


def test_max_shards_stops_after_the_first_and_the_next_call_finishes(world):
    done: list[int] = []
    partial = run(world, max_shards=1, on_shard_done=lambda p, i: done.append(i))
    assert list(partial["shards"]) == [shard_name(0)] and done == [0]
    full = run(world, on_shard_done=lambda p, i: done.append(i))
    assert len(full["shards"]) == 3 and done == [0, 1, 2]
    assert full["shards"][shard_name(0)] == partial["shards"][shard_name(0)]


def test_a_crash_inside_a_shard_loses_only_that_shard(world):
    run(world, max_shards=1)
    marker = world["out"] / shard_name(1) / DONE
    assert not marker.exists()
    calls = {"n": 0}
    real = world["backend"].generate

    def dying(requests):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt("session ended")
        return real(requests)

    world["backend"].generate = dying
    with pytest.raises(KeyboardInterrupt):
        run(world)
    assert not marker.exists()  # the shard in progress has no marker
    world["backend"].generate = real
    report = run(world)
    assert len(report["shards"]) == 3 and marker.is_file()
    # nothing is generated twice: the generations of a shard are unique by key
    for i in range(3):
        keys = [
            json.loads(x)["gen_key"]
            for x in (world["out"] / shard_name(i) / "generations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(keys) == len(set(keys))


def test_the_planted_reader_effect_is_recovered_across_shards(world):
    run(world)
    outcomes = []
    for i in range(3):
        outcomes += ev.load_outcomes(world["out"] / shard_name(i))
    full = [o.value for o in outcomes if o.arm == "reader_full"]
    free = [o.value for o in outcomes if o.arm == "candidate_free"]
    assert sum(full) / len(full) > sum(free) / len(free)
