"""The answer-swap populations (PREREG D25) and the truncation counters (PREREG D14)."""

from __future__ import annotations

import dataclasses

import pytest

from helpers import write_pool_shard
from reader_audit import arms as A
from reader_audit.cand_sets import CandidateSet
from reader_audit.controls import middle_clipper
from reader_audit.pools import Candidate, ProblemPool, load_pools


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=4, replicates=2, n_pool=6, base_p=0.5)
    return next(iter(load_pools(root).values()))


def cand(key, answer, correct, text=None, truncated=False, tokens=50):
    text = text if text is not None else f"Work for {key}. \\boxed{{{answer}}}"
    return Candidate(
        gen_key=key,
        arm="round0",
        replicate=0,
        index=0,
        text=text,
        answer=answer,
        confidence=80,
        correct=correct,
        prompt_tokens=20,
        completion_tokens=tokens,
        truncated=truncated,
    )


def custom_pool(pool, candidates):
    return ProblemPool(pool.problem, tuple(candidates))


def whole_set(pool_, k=None):
    n = pool_.n if k is None else k
    n_correct = sum(c.correct for c in pool_.candidates[:n])
    return CandidateSet(pool_.problem.problem_id, n, 0, tuple(range(n)), n_correct)


def plan_of(pool, candidates):
    custom = custom_pool(pool, candidates)
    return A.swap_plan(custom, whole_set(custom))


def test_a_strict_correct_plurality_with_a_wrong_answer_is_primary(pool):
    plan = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "7", True),
            cand("d", "9", False),
        ],
    )
    assert plan.target == "9" and plan.correct_plurality is True


def test_a_tie_between_the_correct_and_a_wrong_answer_is_not_primary(pool):
    plan = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "9", False),
            cand("d", "9", False),
        ],
    )
    assert plan is not None and plan.correct_plurality is False


def test_a_wrong_plurality_is_in_the_broad_population_only(pool):
    plan = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "9", False),
            cand("d", "9", False),
            cand("e", "9", False),
        ],
    )
    assert plan.target == "9" and plan.correct_plurality is False


def test_the_target_is_the_most_frequent_wrong_answer_and_ties_go_to_the_first_shown(
    pool,
):
    plan = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "7", True),
            cand("d", "4", False),
            cand("e", "9", False),
            cand("f", "9", False),
        ],
    )
    assert plan.target == "9"  # two votes beat one
    tie = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "7", True),
            cand("d", "4", False),
            cand("e", "9", False),
        ],
    )
    assert tie.target == "4"  # one vote each: the answer shown first


def test_answers_that_are_equivalent_pool_their_votes_when_choosing_the_target(pool):
    plan = plan_of(
        pool,
        [
            cand("a", "7", True),
            cand("b", "7", True),
            cand("c", "7", True),
            cand("d", "0.5", False),
            cand("e", "\\frac{1}{2}", False),
            cand("f", "3", False),
        ],
    )
    assert plan.target == "0.5"  # the pooled group of two, shown first as 0.5


def test_candidates_without_an_answer_are_not_wrong_answers(pool):
    nothing = dataclasses.replace(cand("c", "x", False), answer=None)
    also = dataclasses.replace(cand("d", "x", False), answer=None)
    assert (
        plan_of(pool, [cand("a", "7", True), cand("b", "7", True), nothing, also])
        is None
    )


def test_fewer_than_two_correct_candidates_are_never_swapped(pool):
    assert (
        plan_of(
            pool, [cand("a", "7", True), cand("b", "9", False), cand("c", "4", False)]
        )
        is None
    )


def shown_stats(pool, candidates, **kw):
    custom = custom_pool(pool, candidates)
    return A.build_arm_input("reader_full", custom, whole_set(custom), **kw).detail


def test_generation_truncated_and_prompt_clipped_candidates_are_counted_apart(pool):
    long_text = " ".join(f"s{i}" for i in range(80)) + " so \\boxed{7}"
    cands = [
        cand("a", "7", True, text=long_text),
        cand("b", "9", False, text="short \\boxed{9}"),
        dataclasses.replace(
            cand(
                "c", "x", False, text=" ".join(f"s{i}" for i in range(80)) + " cut off"
            ),
            answer=None,
            truncated=True,
        ),
    ]
    clip = middle_clipper(lambda t: t.split(), lambda ids: " ".join(ids), tail=8)
    detail = shown_stats(pool, cands, cap=30, truncate=clip)
    assert detail["n_gen_truncated"] == 1  # the third hit the generation limit
    assert detail["n_prompt_clipped"] == 2  # the two long ones were cut for display
    assert detail["n_answer_hidden"] == 0  # the tail keeps the first one's final answer


def test_cutting_the_end_would_hide_the_answer_and_is_counted(pool):
    long_text = " ".join(f"s{i}" for i in range(80)) + " so \\boxed{7}"
    cands = [
        cand("a", "7", True, text=long_text),
        cand("b", "9", False, text="short \\boxed{9}"),
    ]
    detail = shown_stats(
        pool, cands, cap=30, truncate=lambda t, n: " ".join(t.split()[:n])
    )
    assert detail["n_prompt_clipped"] == 1 and detail["n_answer_hidden"] == 1


def test_nothing_is_clipped_without_a_cap(pool):
    detail = shown_stats(pool, [cand("a", "7", True), cand("b", "9", False)], cap=None)
    assert detail == {"n_gen_truncated": 0, "n_prompt_clipped": 0, "n_answer_hidden": 0}


def test_the_swap_prompt_shows_the_swapped_answer_after_clipping(pool):
    long_text = " ".join(f"s{i}" for i in range(80)) + " so \\boxed{7}"
    cands = [
        cand("a", "7", True, text=long_text),
        cand("b", "7", True, text=long_text),
        cand("c", "7", True, text="short \\boxed{7}"),
        cand("d", "9", False, text="other \\boxed{9}"),
    ]
    custom = custom_pool(pool, cands)
    clip = middle_clipper(lambda t: t.split(), lambda ids: " ".join(ids), tail=8)
    inp = A.build_arm_input(
        "answer_swap", custom, whole_set(custom), cap=30, truncate=clip
    )
    assert inp.detail["n_swapped"] == 3 and inp.detail["correct_plurality"] is True
    assert (
        inp.detail["n_answer_hidden"] == 0
    )  # the swapped box is still the last thing shown
    assert (
        inp.messages[0]["content"].count("\\boxed{9}") == 4
    )  # three swapped + the wrong one
