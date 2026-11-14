"""Baselines that cost no extra generation, computed on pools (PREREG sections 4 and 7).

- A vote weighted by each sample's stored ``Confidence: N``.
- Early-stopping self-consistency (ESC): draw samples in windows and stop once a whole
  window agrees; the comparator of the gate hypothesis H6.
- The approximate length of a reader prompt, to size the context window.

All of them work on the same exchangeable samples as ``subsets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from reader_audit.pools import Candidate, ProblemPool
from reader_audit.subsets import AnswerGroups, credit_from_votes

DEFAULT_CONFIDENCE = 50  # for a sample whose confidence cannot be read
MAX_SUBSETS = 250_000


@lru_cache(maxsize=32)
def subset_matrix(n: int, k: int, max_subsets: int = MAX_SUBSETS, seed: int = 0):
    """Indicator rows of the k-subsets of n samples.

    Every subset when there are at most ``max_subsets`` of them; otherwise a seeded
    random sample of subsets (the result is then an estimate, not exact).
    """
    if comb(n, k) <= max_subsets:
        rows = np.array(list(combinations(range(n), k)), dtype=np.int32)
    else:
        rng = np.random.default_rng(seed)
        rows = np.argsort(rng.random((max_subsets, n)), axis=1)[:, :k]
    matrix = np.zeros((len(rows), n))
    np.put_along_axis(matrix, rows, 1.0, axis=1)
    matrix.setflags(write=False)
    return matrix


def confidence_weights(candidates: Sequence[Candidate]) -> list[float]:
    """Vote weight = stated confidence + 1 (so a stated 0 still votes)."""
    return [
        (c.confidence if c.confidence is not None else DEFAULT_CONFIDENCE) + 1.0
        for c in candidates
    ]


def weighted_vote_credit(
    groups: AnswerGroups, weights: Sequence[float], k: int
) -> float:
    """Expected credit of the weighted plurality of a random k-subset.

    Each parsable sample adds its weight to its answer group; the heaviest group wins,
    and an m-way tie containing the right answer earns 1/m. With every weight equal
    this is the plain vote.
    """
    n = groups.n
    if len(groups.labels) != n or len(weights) != n:
        raise ValueError("groups built from samples, and one weight per sample, needed")
    n_groups = len(groups.sizes)
    if n_groups == 0:
        return 0.0
    per_group = np.zeros((n, n_groups))
    for i, label in enumerate(groups.labels):
        if label is not None:
            per_group[i, label] = weights[i]
    totals = subset_matrix(n, k) @ per_group  # weighted votes of each group, per subset
    return float(credit_from_votes(totals, np.asarray(groups.correct)).mean())


def _plurality_credit(counts: Sequence[int], correct: Sequence[bool]) -> float:
    top = max(counts, default=0)
    if top == 0:
        return 0.0
    tied = [i for i, c in enumerate(counts) if c == top]
    return 1.0 / len(tied) if any(correct[i] for i in tied) else 0.0


def early_stop_vote(
    order: Sequence[int], groups: AnswerGroups, window: int, max_samples: int
) -> tuple[int, float]:
    """One ESC run over the samples in ``order``: (samples drawn, credit).

    Samples are drawn ``window`` at a time. The run stops after a full window in which
    every sample has the same parsable answer, or at ``max_samples``. The answer is
    the plurality of everything drawn.
    """
    if window < 1 or max_samples < window:
        raise ValueError("need 1 <= window <= max_samples")
    counts = [0] * len(groups.sizes)
    drawn = 0
    while drawn < min(max_samples, len(order)):
        chunk = [groups.labels[i] for i in order[drawn : drawn + window]]
        chunk = chunk[: max_samples - drawn]
        for label in chunk:
            if label is not None:
                counts[label] += 1
        drawn += len(chunk)
        if len(chunk) == window and chunk[0] is not None and len(set(chunk)) == 1:
            break
    return drawn, _plurality_credit(counts, groups.correct)


def esc_means(
    groups: AnswerGroups, window: int, max_samples: int, n_perm: int, seed: int
) -> tuple[float, float]:
    """(mean samples drawn, mean credit) of ESC over random orders of the samples."""
    rng = np.random.default_rng(seed)
    orders = rng.permuted(np.tile(np.arange(groups.n), (n_perm, 1)), axis=1)
    runs = [early_stop_vote(o, groups, window, max_samples) for o in orders]
    return float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs]))


# ---- reader prompt length ---------------------------------------------------------

TEMPLATE_TOKENS = 150  # reader template wording, rounded up (drafts are about 100)
HEADER_TOKENS = 12  # "Solver i:" line and spacing around each candidate


def reader_prompt_lengths(
    pool: ProblemPool, k: int, cap: int, n_draws: int, seed: int
) -> np.ndarray:
    """Approximate prompt tokens of random k-candidate reader prompts for one problem.

    The question and round-0 template are taken from the pool's own prompt length
    (a slight overestimate), and each candidate counts at most ``cap`` tokens.
    """
    rng = np.random.default_rng(seed)
    tokens = np.array([min(c.completion_tokens, cap) for c in pool.candidates])
    base = float(np.mean([c.prompt_tokens for c in pool.candidates]))
    draws = np.argsort(rng.random((n_draws, pool.n)), axis=1)[:, :k]
    return base + TEMPLATE_TOKENS + k * HEADER_TOKENS + tokens[draws].sum(axis=1)


# ---- the best cost-accuracy frontier ------------------------------------------------


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def upper_frontier(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper concave envelope of (cost, accuracy) points.

    Every point of it is reachable by mixing the methods behind the points (use one on
    a share of the problems, another on the rest), so it is the frontier of a family.
    """
    hull: list[tuple[float, float]] = []
    for p in sorted(set(points)):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def frontier_at(points: Sequence[tuple[float, float]], cost: float) -> float:
    hull = upper_frontier(points)
    xs, ys = zip(*hull, strict=True)
    if not xs[0] <= cost <= xs[-1]:
        raise ValueError(f"cost {cost} outside the points' range [{xs[0]}, {xs[-1]}]")
    return float(np.interp(cost, xs, ys))
