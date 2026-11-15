"""The untrained reader on the dev pools, in problem shards (Stage 1).

The generations of one shard are planned, made, graded and checked before the next
shard starts, and each finished shard gets a ``DONE.json``. A job that dies loses at
most the shard it was in, and a restart skips the finished ones. ``on_shard_done`` is
where a caller uploads a finished shard (the compute comes in sessions of about ten
hours, so a shard must not live only on the disk of a machine that may be gone).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from channel_audit.backends.base import Backend
from channel_audit.runner import RunConfig, backend_for_run
from channel_audit.shards import LazyBackend

from reader_audit import reader_prompts as rp
from reader_audit.controls import Truncate
from reader_audit.pools import ProblemPool
from reader_audit.reader_run import (
    check_run,
    open_reader_run,
    plan_reader_run,
    run_reader,
)

DEV_ARMS: tuple[str, ...] = ("reader_full", "candidate_free")  # what H1a, H2, H4a need
DONE = "DONE.json"


def shard_name(index: int) -> str:
    return f"shard_{index:03d}"


def split_shards(ids: Sequence[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("a shard needs at least one problem")
    ordered = sorted(ids)
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def _shard_summary(shard: Path, n_planned: int, seconds: float, new: int) -> dict:
    done = {"generations": n_planned, "new_generations": new, "seconds": seconds}
    rows = [
        json.loads(line)
        for line in (shard / "generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    done["prompt_tokens"] = sum(r["prompt_tokens"] for r in rows)
    done["completion_tokens"] = sum(r["completion_tokens"] for r in rows)
    return done


def run_dev_reader(
    out_dir: str | Path,
    pools: Mapping[str, ProblemPool],
    cfg: RunConfig,
    *,
    backend: Backend | None = None,
    backend_factory: Callable[[], Backend] | None = None,
    clipper: Truncate | None = None,
    arms: Sequence[str] = DEV_ARMS,
    k: int = 8,
    n_draws: int = 16,
    replicates: int = 2,
    shard_size: int = 26,
    cap: int | None = rp.CANDIDATE_CAP,
    max_shards: int | None = None,
    on_shard_done: Callable[[Path, int], None] | None = None,
) -> dict:
    """Run (or resume) the dev reader job; return what each shard did.

    ``max_shards`` stops after that many shards that were *run* now (finished shards do
    not count), so a first call with 1 measures throughput before the rest is paid for.
    """
    out = Path(out_dir)
    shards = split_shards(list(pools), shard_size)
    if backend is None:

        def factory() -> Backend:
            return backend_for_run(cfg, [p.problem for p in pools.values()])

        backend = LazyBackend(backend_factory or factory)
    report: dict = {"n_shards": len(shards), "shards": {}}
    ran = 0
    for index, ids in enumerate(shards):
        shard = out / shard_name(index)
        marker = shard / DONE
        if marker.is_file():
            report["shards"][shard.name] = json.loads(marker.read_text("utf-8"))
            continue
        if max_shards is not None and ran >= max_shards:
            break
        subset = {pid: pools[pid] for pid in ids}
        plan = plan_reader_run(
            subset,
            cfg,
            arms=arms,
            ks=(k,),
            n_draws=n_draws,
            replicates=replicates,
            cap=cap,
            truncate=clipper,
        )
        ctx = open_reader_run(shard, cfg, [p.problem for p in subset.values()], backend)
        started = time.monotonic()
        new = run_reader(ctx, plan)
        problems = check_run(ctx, plan)
        if problems:
            raise RuntimeError(
                f"{shard.name}: {len(problems)} problems, {problems[:3]}"
            )
        summary = _shard_summary(
            shard, len(plan.planned), time.monotonic() - started, new
        )
        marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report["shards"][shard.name] = summary
        ran += 1
        if on_shard_done is not None:
            on_shard_done(shard, index)
    return report
