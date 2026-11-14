"""Confidence-weighted vote, early-stopping SC and prompt lengths."""

from __future__ import annotations

import random

import numpy as np
import pytest

from helpers import write_pool_shard
from reader_audit.baselines import (
    DEFAULT_CONFIDENCE,
    confidence_weights,
    early_stop_vote,
    esc_means,
    reader_prompt_lengths,
    subset_matrix,
    weighted_vote_credit,
)
from reader_audit.pools import load_pools
from reader_audit.subsets import group_answers, subset_stats


def groups_of(answers, right="1"):
    return group_answers(answers, [a == right for a in answers])


def test_unit_weights_give_the_plain_vote():
    rng = random.Random(3)
    for _ in range(25):
        answers = [rng.choice(["1", "2", "3", None]) for _ in range(rng.randint(4, 9))]
        g = groups_of(answers)
        for k in range(1, g.n + 1):
            got = weighted_vote_credit(g, [1.0] * g.n, k)
            assert got == pytest.approx(subset_stats(g, k).vote, abs=1e-12)


def test_a_heavy_wrong_answer_beats_two_light_right_ones():
    g = groups_of(["1", "1", "2"])
    assert weighted_vote_credit(g, [1, 1, 1], 3) == 1.0
    assert weighted_vote_credit(g, [1, 1, 5], 3) == 0.0
    assert weighted_vote_credit(g, [3, 3, 5], 3) == 1.0


def test_equal_weight_groups_tie_with_half_credit():
    g = groups_of(["1", "2"])
    assert weighted_vote_credit(g, [4, 4], 2) == 0.5


def test_unparsable_samples_carry_weight_but_never_vote():
    g = groups_of(["1", None, None])
    assert weighted_vote_credit(g, [1, 99, 99], 3) == 1.0


def test_weighted_vote_needs_groups_built_from_samples():
    from reader_audit.subsets import AnswerGroups

    with pytest.raises(ValueError):
        weighted_vote_credit(AnswerGroups((2,), (True,), 0), [1, 1], 1)


def test_subset_matrix_rows_pick_k_samples_and_sampling_is_seeded():
    exact = subset_matrix(6, 3)
    assert exact.shape == (20, 6) and (exact.sum(axis=1) == 3).all()
    big = subset_matrix(20, 10, 1000, 0)  # C(20,10) = 184,756 > 1,000
    assert big.shape == (1000, 20) and (big.sum(axis=1) == 10).all()
    assert np.array_equal(big, subset_matrix(20, 10, 1000, 0))


def test_confidence_weights_use_a_neutral_default(tmp_path):
    write_pool_shard(tmp_path, n_problems=2, replicates=1, n_pool=2)
    pool = next(iter(load_pools(tmp_path).values()))
    w = confidence_weights(pool.candidates)
    assert all(x > 1 for x in w)
    blank = [type(c)(**{**c.__dict__, "confidence": None}) for c in pool.candidates]
    assert set(confidence_weights(blank)) == {DEFAULT_CONFIDENCE + 1.0}


def test_early_stop_stops_after_the_first_agreeing_window():
    g = groups_of(["1"] * 8)
    assert early_stop_vote(range(8), g, window=2, max_samples=8) == (2, 1.0)
    assert early_stop_vote(range(8), g, window=4, max_samples=8) == (4, 1.0)


def test_early_stop_keeps_drawing_while_windows_disagree():
    g = groups_of(["1", "2", "1", "1", "2", "2", "1", "1"])
    # windows of 2: (1,2) no; (1,1) yes -> stop after 4 samples, plurality 1 (3 of 4)
    assert early_stop_vote(range(8), g, window=2, max_samples=8) == (4, 1.0)
    # never agreeing within the cap: draws max_samples and votes on all of them
    h = groups_of(["1", "2", "1", "2", "1", "2"])
    drawn, credit = early_stop_vote(range(6), h, window=2, max_samples=6)
    assert drawn == 6 and credit == 0.5  # 3 vs 3: a tie worth 1/2


def test_a_window_with_an_unparsable_sample_does_not_count_as_agreeing():
    g = groups_of(["1", None, "1", "1"])
    assert early_stop_vote(range(4), g, window=2, max_samples=4)[0] == 4


def test_the_last_window_is_cut_at_the_cap():
    g = groups_of(["1", "2", "1", "2", "1"])
    drawn, _ = early_stop_vote(range(5), g, window=2, max_samples=3)
    assert drawn == 3


def test_esc_never_beats_the_full_pool_in_cost_and_saves_on_easy_problems():
    easy = groups_of(["1"] * 12)
    hard = groups_of(["1", "2", "3", "4"] * 3)
    easy_n, easy_c = esc_means(easy, 4, 12, n_perm=50, seed=0)
    hard_n, _ = esc_means(hard, 4, 12, n_perm=50, seed=0)
    assert easy_n == 4 and easy_c == 1.0
    assert hard_n > easy_n


def test_esc_is_reproducible():
    g = groups_of(["1", "2", "1", "3", "1", "2", "1", "1"])
    assert esc_means(g, 2, 8, 30, 5) == esc_means(g, 2, 8, 30, 5)


def test_early_stop_rejects_a_bad_window():
    g = groups_of(["1", "1"])
    with pytest.raises(ValueError):
        early_stop_vote(range(2), g, window=3, max_samples=2)


def test_prompt_length_counts_candidates_up_to_the_cap(tmp_path):
    write_pool_shard(tmp_path, n_problems=2, replicates=1, n_pool=4)
    pool = next(iter(load_pools(tmp_path).values()))
    cap = 5
    short = reader_prompt_lengths(pool, k=3, cap=cap, n_draws=20, seed=0)
    full = reader_prompt_lengths(pool, k=3, cap=10_000, n_draws=20, seed=0)
    assert (short <= full).all() and short.max() < full.max()
    more = reader_prompt_lengths(pool, k=5, cap=10_000, n_draws=20, seed=0)
    assert more.mean() > full.mean()


def test_frontier_is_the_upper_concave_envelope():
    from reader_audit.baselines import frontier_at, upper_frontier

    points = [(1, 0.60), (2, 0.62), (3, 0.70), (4, 0.71)]
    # (2, 0.62) lies under the segment from (1, 0.60) to (3, 0.70), so it is dropped
    assert upper_frontier(points) == [(1, 0.60), (3, 0.70), (4, 0.71)]
    assert frontier_at(points, 2) == pytest.approx(0.65)
    assert frontier_at(points, 3.5) == pytest.approx(0.705)
    with pytest.raises(ValueError):
        frontier_at(points, 5)
