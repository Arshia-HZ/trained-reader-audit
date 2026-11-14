"""Exact statistics of a random k-subset of one problem's independent samples.

A problem's samples are exchangeable draws from one prompt, so everything a vote or a
reader sees in a random subset depends only on how many samples fall in each answer
group. These functions sum over the count vector of each group with hypergeometric
weights; there is no sampling and no seed. ``tests/test_subsets.py`` checks them against
an explicit enumeration of subsets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from channel_audit.aggregate import cluster_answers


@dataclass(frozen=True)
class AnswerGroups:
    """The samples of one problem, grouped by answer equivalence.

    ``sizes[i]`` samples share answer group ``i``; ``correct[i]`` says whether that
    answer is right; ``n_none`` samples have no parsable answer and never vote.
    ``inconsistent`` counts samples whose own grade disagrees with their group's.
    ``labels`` gives each sample's group (None if unparsable) when the groups were
    built from samples; statistics that need sample-level weights read it.
    """

    sizes: tuple[int, ...]
    correct: tuple[bool, ...]
    n_none: int
    inconsistent: int = 0
    labels: tuple[int | None, ...] = ()

    @property
    def n(self) -> int:
        return sum(self.sizes) + self.n_none


def group_answers(
    answers: Sequence[str | None], correct: Sequence[bool]
) -> AnswerGroups:
    """Group answers by the audit's equivalence; a group is right if any member is."""
    if len(answers) != len(correct):
        raise ValueError("one correctness flag per answer")
    labels, reps = cluster_answers(answers)
    sizes = [0] * len(reps)
    group_correct = [False] * len(reps)
    for label, ok in zip(labels, correct, strict=True):
        if label is not None:
            sizes[label] += 1
            group_correct[label] = group_correct[label] or bool(ok)
    inconsistent = sum(
        1
        for label, ok in zip(labels, correct, strict=True)
        if label is not None and bool(ok) != group_correct[label]
    )
    return AnswerGroups(
        sizes=tuple(sizes),
        correct=tuple(group_correct),
        n_none=sum(1 for label in labels if label is None),
        inconsistent=inconsistent,
        labels=tuple(labels),
    )


def count_vectors(caps: Sequence[int], total: int) -> Iterator[tuple[int, ...]]:
    """Every vector m with 0 <= m[i] <= caps[i] and sum(m) == total."""
    n = len(caps)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + caps[i]
    current = [0] * n

    def rec(i: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if i == n:
            if remaining == 0:
                yield tuple(current)
            return
        for m in range(max(0, remaining - suffix[i + 1]), min(caps[i], remaining) + 1):
            current[i] = m
            yield from rec(i + 1, remaining - m)

    yield from rec(0, total)


@dataclass(frozen=True)
class SubsetStats:
    """Expectations over a uniformly random k-subset (each a probability or credit)."""

    vote: float  # plurality credit, 1/m on an m-way tie containing the right answer
    pass_at_k: float  # at least one right sample
    c0: float  # no right sample
    c1: float  # exactly one right sample
    c2plus: float  # two or more right samples
    minority: float  # a right sample is present but the vote credit is below 1
    strict_minority: float  # a right sample is present but not in the plurality at all
    unanimous: float  # all k samples give the same parsable answer
    unanimous_right: float  # ... and it is right

    @property
    def headroom(self) -> float:
        """What a perfect reader could add to the vote: pass@k minus the vote."""
        return self.pass_at_k - self.vote


def subset_stats(groups: AnswerGroups, k: int) -> SubsetStats:
    n = groups.n
    if not 1 <= k <= n:
        raise ValueError(f"k={k} outside [1, {n}]")
    sizes = list(groups.sizes) + [groups.n_none]  # the last group never votes
    total = comb(n, k)
    acc = dict.fromkeys(SubsetStats.__dataclass_fields__, 0.0)
    for counts in count_vectors([min(s, k) for s in sizes], k):
        w = 1.0
        for size, m in zip(sizes, counts, strict=True):
            w *= comb(size, m)
        w /= total
        votes = counts[:-1]
        top = max(votes, default=0)
        n_right = sum(m for m, ok in zip(votes, groups.correct, strict=True) if ok)
        tied = [i for i, m in enumerate(votes) if top > 0 and m == top]
        right_tied = any(groups.correct[i] for i in tied)
        credit = 1.0 / len(tied) if right_tied else 0.0
        acc["vote"] += w * credit
        acc["pass_at_k"] += w * (n_right > 0)
        acc["c0"] += w * (n_right == 0)
        acc["c1"] += w * (n_right == 1)
        acc["c2plus"] += w * (n_right >= 2)
        acc["minority"] += w * (n_right > 0 and credit < 1.0)
        acc["strict_minority"] += w * (n_right > 0 and not right_tied)
        if sum(1 for m in counts if m > 0) == 1 and counts[-1] == 0:
            acc["unanimous"] += w
            acc["unanimous_right"] += w * (n_right > 0)
    return SubsetStats(**acc)


def sc_curve(groups: AnswerGroups) -> np.ndarray:
    """Self-consistency credit at k = 1..n samples (index k - 1)."""
    return np.array([subset_stats(groups, k).vote for k in range(1, groups.n + 1)])


def interpolate(curve: Sequence[float], k: float) -> float:
    """A curve indexed by k = 1.. read at a fractional k, linear between integers."""
    ks = np.arange(1, len(curve) + 1)
    if not ks[0] <= k <= ks[-1]:
        raise ValueError(f"k={k} outside [1, {len(curve)}]")
    return float(np.interp(k, ks, curve))


def credit_from_votes(votes: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Plurality credit of each row of a (subsets, answer groups) vote matrix.

    The group with the most votes wins; an m-way tie that contains a right group earns
    1/m, and a row without any vote earns 0.
    """
    top = votes.max(axis=1, keepdims=True)
    tied = (votes == top) & (top > 0)
    wins = (tied & correct[None, :]).any(axis=1)
    return np.where(wins, 1.0 / np.maximum(tied.sum(axis=1), 1), 0.0)
