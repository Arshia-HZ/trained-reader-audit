"""The Stage 3 schedule as a function of the seed count (PREREG D7, section 8).

Budgets are the roadmap's per-run extrapolations (GPU-hours, unmeasured; the Stage 2
pilot replaces them). Nothing here is hard-coded to three seeds: the runs, the budget
and the timeline all follow the seed count the rule of ``decision.seeds_for`` gives.
"""

from __future__ import annotations

from reader_audit.decision import SEED_RULE_HOURS

# (low, high) GPU-hours of one 300-step run
PER_RUN_HOURS: dict[str, tuple[float, float]] = {
    "reader": (15.0, 30.0),
    "solver_grpo": (20.0, 50.0),  # matched on rollout plus pool tokens, 1.3 to 1.7 x
    "joint": (17.5, 35.0),
    "no_c0": (15.0, 30.0),
    "reader_0.6b": (8.0, 15.0),
    "pass_at_k": (15.0, 30.0),
    "reader_marginal": (25.0, 60.0),
}
POOLS_HOURS = (24.0, 24.0)  # test pools at 32 samples, train pools at 16
EVALUATION_HOURS_FOR_NINE = (48.0, 96.0)  # all arms, the grid and the K sweep
BASELINE_CHECKPOINTS = 9  # the evaluation estimate assumes 3 + 3 seeds + 3 single runs
SOLVER_TO_READER_COST = (
    1.7  # the token-matched solver run costs at most this x a reader
)
GPU_WEEK_HOURS = 168.0


def main_runs(n_seeds: int) -> dict[str, int]:
    """The Must training runs of Stage 3."""
    return {
        "reader": n_seeds,
        "solver_grpo": n_seeds,
        "joint": 1,
        "no_c0": 1,
        "reader_0.6b": 1,
    }


def factorial_runs() -> dict[str, int]:
    """The Stage 4 factorial: cut first, by 2027-01-17."""
    return {"pass_at_k": 1, "reader_marginal": 2}


def budget(
    n_seeds: int,
    include_factorial: bool = True,
    reader_run_cap: float | None = None,
) -> dict:
    """GPU-hours by item and in total.

    ``reader_run_cap`` is the measured cost that triggered the five-seed rule: it caps
    the high estimate of reader runs, and of solver runs at 1.7 x that.
    """
    runs = main_runs(n_seeds)
    if include_factorial:
        runs |= factorial_runs()
    by_item: dict[str, tuple[float, float]] = {}
    for name, count in runs.items():
        low, high = PER_RUN_HOURS[name]
        if reader_run_cap is not None and name in ("reader", "no_c0"):
            high = min(high, reader_run_cap)
        if reader_run_cap is not None and name == "solver_grpo":
            high = min(high, SOLVER_TO_READER_COST * reader_run_cap)
        by_item[name] = (low * count, high * count)
    by_item["pools"] = POOLS_HOURS
    checkpoints = sum(main_runs(n_seeds).values())
    scale = checkpoints / BASELINE_CHECKPOINTS
    by_item["evaluation"] = tuple(h * scale for h in EVALUATION_HOURS_FOR_NINE)
    low = sum(v[0] for v in by_item.values())
    high = sum(v[1] for v in by_item.values())
    return {
        "by_item": by_item,
        "low": low,
        "high": high,
        "weeks": (low / GPU_WEEK_HOURS, high / GPU_WEEK_HOURS),
    }


def timeline(n_seeds: int) -> dict[str, tuple[int, int]]:
    """Planning weeks of each stage, counted from the start of Stage 0 (week 1)."""
    extra = 1 if n_seeds > 3 else 0
    return {
        "stage_3_training": (3, 5 + extra),
        "stage_3_evaluation": (6 + extra, 6 + extra),
        "stage_4": (7 + extra, 8 + extra),
    }


def five_seed_budget() -> dict:
    """The budget when the rule's threshold is what triggered five seeds."""
    return budget(5, reader_run_cap=SEED_RULE_HOURS)
