"""The health check of a GRPO run (PREREG R17), computed from the files it logs.

Every trigger means "stop and ask the owner"; none of them is a result. Inputs are the
run's ``steps.jsonl`` (one record per step; a step logged twice by a restart counts
once, by its last record), ``saves.jsonl`` and ``runs.jsonl``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

WINDOW = 25
STEP_SECONDS_LIMIT = 300.0
MILESTONES = (100, 200, 300)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def _finite(x) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


def _windows(values: Sequence[float], size: int = WINDOW):
    for i in range(len(values) - size + 1):
        yield i, sum(values[i : i + size]) / size


def triggers(
    steps: Sequence[dict],
    saves: Sequence[dict] = (),
    runs: Sequence[dict] = (),
    *,
    milestones: Sequence[int] = MILESTONES,
) -> list[str]:
    """The R17 triggers that fire for the logs so far (an empty list: all well)."""
    by_step = {int(r["step"]): r for r in steps}
    ordered = [by_step[s] for s in sorted(by_step)]
    out: list[str] = []
    for r in ordered:
        if not _finite(r.get("loss")) or not _finite(r.get("grad_norm")):
            out.append(f"step {r['step']}: loss or gradient norm not finite")
            break
    grads = [r["grad_norm"] for r in ordered if _finite(r.get("grad_norm"))]
    if len(grads) > WINDOW:
        limit = 10 * statistics.median(grads[:WINDOW])
        run = 0
        for r in ordered[WINDOW:]:
            run = (
                run + 1 if _finite(r.get("grad_norm")) and r["grad_norm"] > limit else 0
            )
            if run >= 3:
                out.append(f"step {r['step']}: gradient norm above 10x median, 3 steps")
                break
    rewards = [r["reward"] for r in ordered if _finite(r.get("reward"))]
    if len(rewards) >= 50 + WINDOW:
        base = sum(rewards[:WINDOW]) / WINDOW
        for i, mean in _windows(rewards[50:]):
            if mean < 0.5 * base:
                out.append(f"steps {51 + i}-{50 + i + WINDOW}: reward fell below half")
                break
    for key, low, high, name in (
        ("mean_length", 100.0, 1500.0, "mean completion length out of range"),
        ("clipped_ratio", None, 0.25, "more than 25% of completions at the cap"),
        ("seconds", None, STEP_SECONDS_LIMIT, "mean step time above the limit"),
    ):
        values = [r[key] for r in ordered if _finite(r.get(key))]
        for i, mean in _windows(values):
            if (low is not None and mean < low) or (high is not None and mean > high):
                out.append(f"steps {i + 1}-{i + WINDOW}: {name} ({mean:.3g})")
                break
    for rec in saves:
        want = {"latest"} | (
            {f"step_{rec['step']:04d}"} if rec["step"] in milestones else set()
        )
        got = {u.rsplit("/", 1)[-1] for u in rec.get("uploaded", [])}
        if want - got:
            out.append(f"step {rec['step']}: not on the hub: {sorted(want - got)}")
    hashes = {r["rows_sha256"] for r in runs if r.get("rows_sha256")}
    if len(hashes) > 1:
        out.append("restart with a different row sequence")
    return out


def check_run(directory: str | Path, **kw) -> list[str]:
    d = Path(directory)
    return triggers(
        read_jsonl(d / "steps.jsonl"),
        read_jsonl(d / "saves.jsonl"),
        read_jsonl(d / "runs.jsonl"),
        **kw,
    )
