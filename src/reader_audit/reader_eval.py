"""Outcomes of reader runs and paired contrasts by stratum (PREREG D9, D15).

An outcome is one scored generation (or, for the plain vote, one set). Contrasts pair
the arms on the same sets: for each problem, the cells (draws) both arms have in the
stratum are averaged over replicates and the arm means are subtracted; the problem is
the unit of the bootstrap, as in the audit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from channel_audit.analysis.contrasts import summarize
from channel_audit.schema import Generation, Graded
from channel_audit.store import JsonlStore

from reader_audit.cand_sets import draw_set, vote_credit
from reader_audit.pools import ProblemPool
from reader_audit.reader_run import PlanRow

VOTE = "vote"  # the plurality vote of the K candidates themselves


@dataclass(frozen=True)
class Outcome:
    problem_id: str
    arm: str
    k: int
    draw: int
    replicate: int
    stratum: str
    n_correct: int
    value: float  # 1.0 or 0.0 for a reader, the vote credit for the plain vote
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_outcomes(out_dir: str | Path) -> list[Outcome]:
    """Join a run's plan, generations and grades."""
    base = Path(out_dir)
    plan = JsonlStore(base / "plan.jsonl", PlanRow, readonly=True)
    gens = JsonlStore(base / "generations.jsonl", Generation, readonly=True)
    graded = JsonlStore(base / "graded.jsonl", Graded, readonly=True)
    out = []
    for row in plan:
        if row.gen_key not in graded:
            continue
        gen = gens.get(row.gen_key)
        out.append(
            Outcome(
                problem_id=row.problem_id,
                arm=row.arm,
                k=row.k,
                draw=row.draw,
                replicate=row.replicate,
                stratum=row.stratum,
                n_correct=row.n_correct,
                value=1.0 if graded.get(row.gen_key).correct else 0.0,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
            )
        )
    return out


def vote_outcomes(
    pools: Mapping[str, ProblemPool], ks: Sequence[int], n_draws: int
) -> list[Outcome]:
    """The plain vote of each candidate set, as an arm to compare readers with."""
    out = []
    for pool in pools.values():
        for k in ks:
            for draw in range(n_draws):
                cset = draw_set(pool, k, draw)
                out.append(
                    Outcome(
                        problem_id=cset.problem_id,
                        arm=VOTE,
                        k=k,
                        draw=draw,
                        replicate=0,
                        stratum=cset.stratum,
                        n_correct=cset.n_correct,
                        value=vote_credit(pool, cset),
                    )
                )
    return out


def cell_means(
    outcomes: Iterable[Outcome], arm: str, k: int, stratum: str | None = None
) -> dict[tuple[str, int], float]:
    """Mean over replicates of each (problem, draw) cell of an arm."""
    sums: dict[tuple[str, int], list[float]] = defaultdict(list)
    for o in outcomes:
        if o.arm == arm and o.k == k and (stratum is None or o.stratum == stratum):
            sums[(o.problem_id, o.draw)].append(o.value)
    return {cell: sum(v) / len(v) for cell, v in sums.items()}


def problem_means(cells: Mapping[tuple[str, int], float]) -> dict[str, float]:
    by_problem: dict[str, list[float]] = defaultdict(list)
    for (pid, _), value in cells.items():
        by_problem[pid].append(value)
    return {pid: sum(v) / len(v) for pid, v in by_problem.items()}


def paired_diffs(
    outcomes: Sequence[Outcome], a: str, b: str, k: int, stratum: str | None = None
) -> dict[str, float]:
    """Per problem, mean(a) - mean(b) over the cells both arms have."""
    cells_a = cell_means(outcomes, a, k, stratum)
    cells_b = cell_means(outcomes, b, k, stratum)
    shared = cells_a.keys() & cells_b.keys()
    return problem_means({cell: cells_a[cell] - cells_b[cell] for cell in shared})


def contrast(
    outcomes: Sequence[Outcome],
    a: str,
    b: str,
    k: int,
    stratum: str | None = None,
    alternative: str | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict | None:
    """Bootstrap summary of a - b, or None if no problem has both arms."""
    diffs = paired_diffs(outcomes, a, b, k, stratum)
    if not diffs:
        return None
    return summarize(diffs, alternative, n_boot=n_boot, seed=seed)
