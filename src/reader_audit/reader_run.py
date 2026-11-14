"""Planning and running reader generations on top of the audit's runner.

A plan lists every generation as ``(identity, messages, parent keys)``, the audit's
``Planned``, so the audit's append-only store, seeds, resume, throughput log and
grading are reused as they are. The identity's ``prompt_hash`` covers the template, the
candidates shown, the token cap and the control's detail, so two prompts that differ
in any way never share a key. A sidecar ``plan.jsonl`` records what the stored
generations do not: the set's draw, stratum and the control's detail.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from channel_audit.backends.base import Backend
from channel_audit.runner import (
    Planned,
    RunConfig,
    RunContext,
    _execute,
    create_run,
    grade_pending,
)
from channel_audit.schema import Problem
from channel_audit.seeds import gen_key
from channel_audit.store import JsonlStore
from pydantic import BaseModel, ConfigDict

from reader_audit import reader_prompts as rp
from reader_audit.arms import ARMS, DonorIndex, build_arm_input
from reader_audit.cand_sets import draw_set
from reader_audit.controls import TAIL_TOKENS, Truncate
from reader_audit.pools import ProblemPool

ROUND = 1  # a reader answers once, after the candidates


class PlanRow(BaseModel):
    """What a stored reader generation was for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_key: str
    problem_id: str
    arm: str
    k: int
    draw: int
    replicate: int
    stratum: str
    n_correct: int
    detail: dict[str, Any]


@dataclass(frozen=True)
class ReaderPlan:
    planned: list[Planned]
    rows: list[PlanRow]


def arm_name(arm: str, k: int) -> str:
    """The arm string stored in the generation: the control and the set size."""
    return f"{arm}_k{k}"


def prompt_hash(
    arm: str,
    parent_keys: Sequence[str],
    cap: int | None,
    detail: Mapping[str, Any],
) -> str:
    """Hash of everything that determines the prompt text."""
    payload = json.dumps(
        [
            rp.template_hash(arm),
            list(parent_keys),
            cap,
            TAIL_TOKENS if cap else None,
            dict(detail),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def reader_identity(
    cfg: RunConfig,
    problem_id: str,
    replicate: int,
    draw: int,
    arm: str,
    k: int,
    p_hash: str,
) -> dict:
    return {
        "problem_id": problem_id,
        "replicate": replicate,
        "agent": draw,
        "model_id": cfg.model_id,
        "arm": arm_name(arm, k),
        "round": ROUND,
        "sample_idx": 0,
        "prompt_version": rp.PROMPT_VERSION,
        "prompt_hash": p_hash,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }


def plan_reader_run(
    pools: Mapping[str, ProblemPool],
    cfg: RunConfig,
    *,
    arms: Sequence[str],
    ks: Sequence[int],
    n_draws: int,
    replicates: int,
    cap: int | None = rp.CANDIDATE_CAP,
    truncate: Truncate | None = None,
) -> ReaderPlan:
    """Every reader generation: problems × K × draws × arms × replicates."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown reader arms {unknown}; expected from {ARMS}")
    donors = DonorIndex(pools, cap) if "placebo" in arms else None
    planned: list[Planned] = []
    rows: list[PlanRow] = []
    for pool in pools.values():
        pid = pool.problem.problem_id
        for k in ks:
            for draw in range(n_draws):
                cset = draw_set(pool, k, draw)
                for arm in arms:
                    inp = build_arm_input(
                        arm, pool, cset, donors=donors, cap=cap, truncate=truncate
                    )
                    if inp is None:
                        continue
                    p_hash = prompt_hash(arm, inp.parent_keys, cap, inp.detail)
                    for rep in range(replicates):
                        identity = reader_identity(cfg, pid, rep, draw, arm, k, p_hash)
                        planned.append(
                            Planned(identity, inp.messages, list(inp.parent_keys))
                        )
                        rows.append(
                            PlanRow(
                                gen_key=gen_key(**identity),
                                problem_id=pid,
                                arm=arm,
                                k=k,
                                draw=draw,
                                replicate=rep,
                                stratum=cset.stratum,
                                n_correct=cset.n_correct,
                                detail=dict(inp.detail),
                            )
                        )
    return ReaderPlan(planned, rows)


def open_reader_run(
    out_dir: str | Path,
    cfg: RunConfig,
    problems: Sequence[Problem],
    backend: Backend | None,
) -> RunContext:
    return create_run(out_dir, cfg, problems, backend)


def run_reader(ctx: RunContext, plan: ReaderPlan) -> int:
    """Generate what the store lacks, grade it and record the plan; return n new."""
    before = len(ctx.gens)
    _execute(ctx, plan.planned)
    grade_pending(ctx)
    rows = JsonlStore(ctx.out_dir / "plan.jsonl", PlanRow, readonly=False)
    for row in plan.rows:
        if row.gen_key not in rows:
            rows.append(row)
    return len(ctx.gens) - before


def check_run(ctx: RunContext, plan: ReaderPlan) -> list[str]:
    """Problems with a finished run: missing or ungraded generations."""
    problems = []
    for row in plan.rows:
        if row.gen_key not in ctx.gens:
            problems.append(f"missing generation {row.gen_key[:12]} ({row.arm})")
        elif row.gen_key not in ctx.graded:
            problems.append(f"ungraded generation {row.gen_key[:12]} ({row.arm})")
    return problems
