"""The plan of a GRPO run: batch arithmetic, rows per step and prompt-length checks.

Nothing here imports torch, so the checks run with the audit's own tests. The trainer
(``grpo_run``) turns a plan into TRL settings. One optimizer step is one generation
batch (``prompts_per_step`` prompts x ``group_size`` samples), which keeps checkpoints
aligned with TRL's generation cycle: TRL does not save buffered generations, so a
checkpoint taken inside a cycle would shift the data on resume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reader_audit import reader_prompts as rp
from reader_audit.controls import Truncate
from reader_audit.pools import ProblemPool
from reader_audit.train_data import reader_rows, select_sets, solver_rows

ARMS = ("reader", "reader_no_c0", "solver")
CHAT_KWARGS = {"enable_thinking": False}  # as in the evaluated reader (D14)


@dataclass(frozen=True)
class GrpoPlan:
    arm: str = "reader"
    steps: int = 300
    prompts_per_step: int = 64
    group_size: int = 8
    per_device_batch: int = 1
    max_completion: int = 2048  # the evaluation's max_tokens
    max_model_len: int = 16384  # the evaluation's window (D14)
    save_every: int = 25
    seed: int = 20261115

    @property
    def completions_per_step(self) -> int:
        return self.prompts_per_step * self.group_size

    @property
    def grad_accum(self) -> int:
        return self.completions_per_step // self.per_device_batch

    def check(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.completions_per_step % self.per_device_batch:
            raise ValueError("per_device_batch must divide prompts x group size")
        if self.group_size < 2:
            raise ValueError("GRPO needs a group of at least 2")
        if self.save_every < 1 or self.steps < 1:
            raise ValueError("steps and save_every must be positive")


def build_rows(
    plan: GrpoPlan,
    pools: Mapping[str, ProblemPool],
    *,
    truncate: Truncate | None,
    cap: int | None = rp.CANDIDATE_CAP,
    n_draws: int = 4,
) -> list[dict]:
    """The prompts of the whole run, in the order the steps consume them."""
    need = plan.steps * plan.prompts_per_step
    if plan.arm == "solver":
        epochs = max(1, math.ceil(need / max(len(pools), 1)))
        rows = solver_rows(pools, epochs=epochs, seed=plan.seed)
    else:
        sets = select_sets(
            pools,
            n_draws=n_draws,
            drop_c0=plan.arm == "reader_no_c0",
            seed=plan.seed,
        )
        rows = reader_rows(pools, sets, cap=cap, truncate=truncate)
    if len(rows) < need:
        raise ValueError(
            f"{len(rows)} rows for {plan.steps} steps of {plan.prompts_per_step} "
            f"prompts ({need} needed); add problems or draws"
        )
    return rows[:need]


def step_rows(rows: Sequence[dict], plan: GrpoPlan, step: int) -> Sequence[dict]:
    """Rows consumed by optimizer step ``step`` (0-based)."""
    a = step * plan.prompts_per_step
    return rows[a : a + plan.prompts_per_step]


def prompt_lengths(rows: Sequence[dict], tokenizer) -> list[int]:
    """Token length of every rendered prompt, as the rollout will see it."""
    out = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **CHAT_KWARGS,
        )
        out.append(len(ids))
    return out


def assert_fits(lengths: Sequence[int], plan: GrpoPlan) -> int:
    """Raise if any prompt plus a full completion would not fit; return the longest."""
    longest = max(lengths)
    if longest + plan.max_completion > plan.max_model_len:
        raise ValueError(
            f"longest prompt {longest} + {plan.max_completion} completion tokens "
            f"exceeds the window {plan.max_model_len}"
        )
    return longest


def rollout_tokens(entry: Mapping[str, float], plan: GrpoPlan) -> int:
    """Completion tokens generated in the step a log entry describes."""
    return round(entry["completions/mean_length"] * plan.completions_per_step)


def cumulative_rollout_tokens(
    log_history: Sequence[Mapping[str, float]], plan: GrpoPlan
) -> int:
    """Total of ``rollout_tokens`` over a log history (rebuilt after a resume)."""
    return sum(
        rollout_tokens(e, plan) for e in log_history if "completions/mean_length" in e
    )
