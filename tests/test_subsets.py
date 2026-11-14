"""Exact subset statistics agree with an explicit enumeration of subsets."""

from __future__ import annotations

import itertools
import random

import pytest

from reader_audit.subsets import (
    AnswerGroups,
    count_vectors,
    group_answers,
    interpolate,
    sc_curve,
    subset_stats,
)


def brute(groups: AnswerGroups, k: int) -> dict[str, float]:
    """Reference: enumerate every k-subset of the individual samples."""
    items: list[int | None] = []  # answer group of each sample, None = unparsable
    for g, size in enumerate(groups.sizes):
        items += [g] * size
    items += [None] * groups.n_none
    acc = dict(vote=0.0, pass_at_k=0.0, c1=0.0, c2plus=0.0, unanimous=0.0, strict=0.0)
    subsets = list(itertools.combinations(range(len(items)), k))
    for sub in subsets:
        picked = [items[i] for i in sub]
        counts: dict[int, int] = {}
        for g in picked:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        top = max(counts.values(), default=0)
        tied = [g for g, m in counts.items() if m == top and top > 0]
        right_tied = any(groups.correct[g] for g in tied)
        n_right = sum(1 for g in picked if g is not None and groups.correct[g])
        acc["vote"] += 1 / len(tied) if right_tied else 0.0
        acc["pass_at_k"] += n_right > 0
        acc["c1"] += n_right == 1
        acc["c2plus"] += n_right >= 2
        acc["unanimous"] += len(set(picked)) == 1 and picked[0] is not None
        acc["strict"] += n_right > 0 and not right_tied
    return {name: value / len(subsets) for name, value in acc.items()}


def random_groups(rng: random.Random) -> AnswerGroups:
    n_groups = rng.randint(1, 4)
    sizes = tuple(rng.randint(1, 4) for _ in range(n_groups))
    correct = [False] * n_groups
    correct[rng.randrange(n_groups)] = True
    if n_groups > 2 and rng.random() < 0.3:  # two right groups, as grading can give
        correct[rng.randrange(n_groups)] = True
    return AnswerGroups(sizes=sizes, correct=tuple(correct), n_none=rng.randint(0, 3))


def test_matches_enumeration_on_random_groups():
    rng = random.Random(7)
    for _ in range(40):
        groups = random_groups(rng)
        for k in range(1, min(groups.n, 7) + 1):
            got, ref = subset_stats(groups, k), brute(groups, k)
            assert got.vote == pytest.approx(ref["vote"], abs=1e-12)
            assert got.pass_at_k == pytest.approx(ref["pass_at_k"], abs=1e-12)
            assert got.c1 == pytest.approx(ref["c1"], abs=1e-12)
            assert got.c2plus == pytest.approx(ref["c2plus"], abs=1e-12)
            assert got.unanimous == pytest.approx(ref["unanimous"], abs=1e-12)
            assert got.strict_minority == pytest.approx(ref["strict"], abs=1e-12)


def test_strata_partition_the_probability():
    groups = AnswerGroups(sizes=(5, 3, 2), correct=(False, True, False), n_none=2)
    for k in (1, 3, 6, 12):
        s = subset_stats(groups, k)
        assert s.c0 + s.c1 + s.c2plus == pytest.approx(1.0)
        assert s.pass_at_k == pytest.approx(1.0 - s.c0)


def test_hand_computed_pair():
    # samples a, a, a, b with a right: a pair is aa (3 of 6, credit 1) or ab (3 of 6,
    # a tie worth 1/2), so the vote at k = 2 is (3 + 1.5) / 6.
    groups = AnswerGroups(sizes=(3, 1), correct=(True, False), n_none=0)
    assert subset_stats(groups, 1).vote == pytest.approx(0.75)
    assert subset_stats(groups, 2).vote == pytest.approx(0.75)
    assert subset_stats(groups, 4).vote == 1.0
    assert (
        subset_stats(groups, 2).strict_minority == 0.0
    )  # a tie keeps a in the plurality


def test_unparsable_samples_never_vote():
    groups = AnswerGroups(sizes=(1,), correct=(True,), n_none=3)
    s = subset_stats(groups, 2)
    # pairs: {right, none} x3 give the right answer alone; {none, none} x3 give nothing
    assert s.vote == pytest.approx(0.5)
    assert s.unanimous == pytest.approx(0.0)  # a pair with a none is not unanimous


def test_headroom_is_pass_minus_vote_and_zero_on_unanimous_sets():
    groups = AnswerGroups(sizes=(4,), correct=(True,), n_none=0)
    s = subset_stats(groups, 3)
    assert s.unanimous == 1.0 and s.unanimous_right == 1.0 and s.headroom == 0.0


def test_sc_curve_ends_at_the_full_pool_vote():
    groups = AnswerGroups(sizes=(4, 3), correct=(True, False), n_none=1)
    curve = sc_curve(groups)
    assert len(curve) == groups.n
    assert curve[-1] == subset_stats(groups, groups.n).vote == 1.0


def test_k_out_of_range_is_an_error():
    groups = AnswerGroups(sizes=(2,), correct=(True,), n_none=0)
    with pytest.raises(ValueError):
        subset_stats(groups, 3)
    with pytest.raises(ValueError):
        subset_stats(groups, 0)


def test_count_vectors_are_the_bounded_compositions():
    got = set(count_vectors([2, 1, 3], 3))
    want = {m for m in itertools.product(range(3), range(2), range(4)) if sum(m) == 3}
    assert got == want


def test_interpolate_is_linear_between_integers():
    curve = [0.6, 0.7, 0.9]
    assert interpolate(curve, 1.0) == 0.6
    assert interpolate(curve, 1.5) == pytest.approx(0.65)
    assert interpolate(curve, 3.0) == 0.9
    with pytest.raises(ValueError):
        interpolate(curve, 3.5)


def test_group_answers_pools_equivalent_answers_and_counts_none():
    answers = ["0.5", r"\frac{1}{2}", "3", None, "3"]
    correct = [True, True, False, False, False]
    g = group_answers(answers, correct)
    assert g.sizes == (2, 2)
    assert g.correct == (True, False)
    assert g.n_none == 1 and g.inconsistent == 0 and g.n == 5


def test_group_answers_flags_a_grade_that_disagrees_with_its_group():
    g = group_answers(["4", "4"], [True, False])
    assert g.correct == (True,)
    assert g.inconsistent == 1
