"""Candidate pools for a list of problems, in resumable shards (Stage 1 and 2).

Each problem gets 16 independent samples of the pool model: 3 from the audit's round-0
arm and 13 from its pool arm, as in the dev pools. A shard of problems is generated,
graded and checked before the next starts, and each finished shard gets ``DONE.json``;
``on_shard_done`` is where a caller uploads it. A job that dies loses at most the shard
it was in.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from channel_audit.backends.base import Backend
from channel_audit.runner import (
    RunConfig,
    backend_for_run,
    create_run,
    grade_pending,
    run_round0,
    run_sc_pool,
)
from channel_audit.schema import Problem
from channel_audit.shards import LazyBackend

from reader_audit.pools import read_pools

ROUND0_AGENTS = 3
DONE = "DONE.json"


def shard_name(index: int) -> str:
    return f"shard_{index:03d}"


def run_pool_shards(
    out_dir: str | Path,
    problems: Sequence[Problem],
    cfg: RunConfig,
    *,
    backend: Backend | None = None,
    backend_factory: Callable[[], Backend] | None = None,
    samples: int = 16,
    shard_size: int = 100,
    max_shards: int | None = None,
    on_shard_done: Callable[[Path, int], None] | None = None,
) -> dict:
    """Run (or resume) the pool job in the order of ``problems``; return per shard."""
    if samples <= ROUND0_AGENTS:
        raise ValueError(f"need more than {ROUND0_AGENTS} samples per problem")
    out = Path(out_dir)
    pool_cfg = cfg.model_copy(
        update={"replicates": 1, "n_pool": samples - ROUND0_AGENTS}
    )
    if backend is None:

        def factory() -> Backend:
            return backend_for_run(pool_cfg, list(problems))

        backend = LazyBackend(backend_factory or factory)
    shards = [
        list(problems[i : i + shard_size]) for i in range(0, len(problems), shard_size)
    ]
    report: dict = {"n_shards": len(shards), "shards": {}}
    ran = 0
    for index, chunk in enumerate(shards):
        shard = out / shard_name(index)
        marker = shard / DONE
        if marker.is_file():
            report["shards"][shard.name] = json.loads(marker.read_text("utf-8"))
            continue
        if max_shards is not None and ran >= max_shards:
            break
        ctx = create_run(shard, pool_cfg, chunk, backend)
        started = time.monotonic()
        run_round0(ctx)
        run_sc_pool(ctx)
        grade_pending(ctx)
        pools = read_pools(shard)
        sizes = {pool.n for pool in pools.values()}
        if len(pools) != len(chunk) or sizes != {samples}:
            raise RuntimeError(
                f"{shard.name}: {len(pools)} of {len(chunk)} problems, "
                f"pool sizes {sorted(sizes)}"
            )
        gens = [
            json.loads(line)
            for line in (shard / "generations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        summary = {
            "problems": len(chunk),
            "generations": len(gens),
            "seconds": time.monotonic() - started,
            "prompt_tokens": sum(g["prompt_tokens"] for g in gens),
            "completion_tokens": sum(g["completion_tokens"] for g in gens),
            "truncated": sum(g["finish_reason"] != "stop" for g in gens),
        }
        marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report["shards"][shard.name] = summary
        ran += 1
        if on_shard_done is not None:
            on_shard_done(shard, index)
    return report
