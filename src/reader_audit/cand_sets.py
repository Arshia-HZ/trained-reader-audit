"""Candidate sets: K samples drawn from a problem's pool, and their strata (PREREG D15).

A set is a seeded draw without replacement; the order of the draw is the order the
candidates are shown in. The stratum counts the correct candidates and is computed
afterwards; it is never used to choose which sets are evaluated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from channel_audit.aggregate import plurality_credit

from reader_audit.pools import Candidate, ProblemPool

SETS_SALT = "reader-audit/sets/v1"
STRATA: tuple[str, ...] = ("c0", "c1", "c2+")


def stratum_of(n_correct: int) -> str:
    if n_correct < 0:
        raise ValueError("n_correct must not be negative")
    return STRATA[min(n_correct, 2)]


@dataclass(frozen=True)
class CandidateSet:
    problem_id: str
    k: int
    draw: int
    positions: tuple[int, ...]  # indices into the pool's candidates, in shown order
    n_correct: int

    @property
    def stratum(self) -> str:
        return stratum_of(self.n_correct)


def _generator(problem_id: str, k: int, draw: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{SETS_SALT}|{problem_id}|{k}|{draw}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def draw_set(pool: ProblemPool, k: int, draw: int) -> CandidateSet:
    """The ``draw``-th set of ``k`` candidates of the pool (deterministic)."""
    if not 1 <= k <= pool.n:
        raise ValueError(f"k={k} outside [1, {pool.n}] for {pool.problem.problem_id}")
    order = _generator(pool.problem.problem_id, k, draw).permutation(pool.n)[:k]
    positions = tuple(int(i) for i in order)
    n_correct = sum(pool.candidates[i].correct for i in positions)
    return CandidateSet(pool.problem.problem_id, k, draw, positions, n_correct)


def draw_sets(pool: ProblemPool, k: int, n_draws: int) -> list[CandidateSet]:
    return [draw_set(pool, k, d) for d in range(n_draws)]


def members(pool: ProblemPool, cset: CandidateSet) -> list[Candidate]:
    return [pool.candidates[i] for i in cset.positions]


def vote_credit(pool: ProblemPool, cset: CandidateSet) -> float:
    """Plurality credit of the set's answers: 1/m on an m-way tie with the right one."""
    answers: Sequence[str | None] = [c.answer for c in members(pool, cset)]
    return plurality_credit(answers, pool.problem.gold)
