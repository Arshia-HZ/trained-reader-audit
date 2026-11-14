"""Test helpers: audit-format shards built with the audit's mock backend. No GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import (
    RunConfig,
    create_run,
    grade_pending,
    run_round0,
    run_sc_pool,
)


def write_pool_shard(
    root: Path,
    *,
    name: str = "shard_00",
    n_problems: int = 12,
    replicates: int = 2,
    n_pool: int = 6,
    first_problem: int = 0,
    base_p: float | None = None,
) -> Path:
    """A finished shard holding round 0 and the pool of some fake problems."""
    problems = fake_problems(first_problem + n_problems)[first_problem:]
    mock = MockBackend(MockWorld.from_problems(problems, base_p=base_p))
    cfg = RunConfig(
        model_id="mock/model",
        replicates=replicates,
        n_pool=n_pool,
        max_tokens=512,
        context_limit=100_000,
    )
    ctx = create_run(root / name, cfg, problems, mock)
    run_round0(ctx)
    run_sc_pool(ctx)
    grade_pending(ctx)
    gens = root / name / "generations.jsonl"
    done = {
        "shard": name,
        "n_problems": n_problems,
        "n_generations": len(ctx.gens),
        "generations_sha256": hashlib.sha256(gens.read_bytes()).hexdigest(),
        "arms": [],
    }
    (root / name / "SHARD_DONE.json").write_text(json.dumps(done), encoding="utf-8")
    return root / name
