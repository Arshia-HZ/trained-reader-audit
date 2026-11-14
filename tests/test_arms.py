"""Reader prompts per arm, on mock pools."""

from __future__ import annotations

import pytest
from channel_audit.extract import extract_answer

from helpers import write_pool_shard
from reader_audit import arms as A
from reader_audit.cand_sets import CandidateSet, draw_set, members
from reader_audit.pools import load_pools


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=12, replicates=2, n_pool=6, base_p=0.5)
    return load_pools(root)


@pytest.fixture(scope="module")
def pool(pools):
    return next(iter(pools.values()))


def mixed_set(pool, n_right, n_wrong):
    right = [i for i, c in enumerate(pool.candidates) if c.correct]
    wrong = [i for i, c in enumerate(pool.candidates) if not c.correct and c.answer]
    positions = tuple(right[:n_right] + wrong[:n_wrong])
    assert len(positions) == n_right + n_wrong, "mock pool lacks the needed samples"
    return CandidateSet(pool.problem.problem_id, len(positions), 0, positions, n_right)


def body(inp):
    return inp.messages[0]["content"]


def test_reader_full_shows_every_candidate_in_order_without_confidence(pool):
    cset = draw_set(pool, 4, 0)
    inp = A.build_arm_input("reader_full", pool, cset, cap=None)
    assert inp.parent_keys == tuple(c.gen_key for c in members(pool, cset))
    last = -1
    for c in members(pool, cset):
        marker = f"[trace {c.text.split('[trace ')[1].split(']')[0]}]"
        at = body(inp).index(marker)
        assert at > last
        last = at
    assert "Confidence:" not in body(inp)


def test_candidate_free_shows_no_candidate(pool):
    inp = A.build_arm_input("candidate_free", pool, draw_set(pool, 4, 0))
    assert inp.parent_keys == () and "trace" not in body(inp)
    assert pool.problem.question in body(inp)


def test_answer_only_shows_answers_and_no_reasoning(pool):
    cset = draw_set(pool, 4, 0)
    inp = A.build_arm_input("answer_only", pool, cset)
    assert "Step 1" not in body(inp) and "trace" not in body(inp)
    for i, c in enumerate(members(pool, cset), start=1):
        assert f"Solver {i}: {c.answer}" in body(inp)


def test_placebo_shows_another_problems_solutions_of_the_same_size(pools, pool):
    donors = A.DonorIndex(pools)
    cset = draw_set(pool, 3, 1)
    inp = A.build_arm_input("placebo", pool, cset, donors=donors)
    donor_id = inp.detail["donor_problem"]
    assert donor_id != pool.problem.problem_id
    assert pools[donor_id].problem.subject == pool.problem.subject
    assert pools[donor_id].problem.level == pool.problem.level
    assert body(inp).count("Solver ") == 3
    for c in members(pool, cset):  # none of the problem's own candidates
        assert c.gen_key not in inp.parent_keys
    donor_set = draw_set(pools[donor_id], 3, 1)
    assert inp.parent_keys == tuple(
        c.gen_key for c in members(pools[donor_id], donor_set)
    )
    # the prompt asks about the real question, with the same template as reader_full
    assert body(inp).startswith(f"Problem:\n{pool.problem.question}")
    assert A.rp.template_hash("placebo") == A.rp.template_hash("reader_full")


def test_placebo_donor_is_the_nearest_in_length_among_the_group(pools, pool):
    donors = A.DonorIndex(pools, cap=None)
    cset = draw_set(pool, 3, 0)
    donor_pool, _ = donors.pick(pool, cset)
    target = sum(c.completion_tokens for c in members(pool, cset))

    def distance(p):
        return abs(
            sum(c.completion_tokens for c in members(p, draw_set(p, 3, 0))) - target
        )

    group = [
        p
        for p in pools.values()
        if p.problem.subject == pool.problem.subject
        and p.problem.level == pool.problem.level
        and p.problem.problem_id != pool.problem.problem_id
    ]
    assert distance(donor_pool) == min(distance(p) for p in group)


def test_placebo_needs_donors(pool):
    with pytest.raises(ValueError, match="DonorIndex"):
        A.build_arm_input("placebo", pool, draw_set(pool, 3, 0))


def test_duplicate_repeats_the_first_candidate(pool):
    cset = draw_set(pool, 4, 0)
    inp = A.build_arm_input("duplicate", pool, cset, cap=None)
    first = members(pool, cset)[0]
    marker = first.text.split("[trace ")[1].split("]")[0]
    assert body(inp).count(f"[trace {marker}]") == 4
    assert len(set(inp.parent_keys)) == 1 and len(inp.parent_keys) == 4


def test_single_slot_fills_the_other_slots_with_a_marker(pool):
    inp = A.build_arm_input("single_slot", pool, draw_set(pool, 4, 0), cap=None)
    assert body(inp).count("(no solution)") == 3
    assert body(inp).count("Solver ") == 4 and len(inp.parent_keys) == 1


def test_dedup_shows_one_candidate_per_distinct_answer(pool):
    cset = mixed_set(pool, 2, 2)
    inp = A.build_arm_input("dedup", pool, cset, cap=None)
    answers = [c.answer for c in members(pool, cset)]
    assert inp.detail["n_shown"] == len(set(answers))
    assert body(inp).count("Solver ") == inp.detail["n_shown"]


def test_answer_swap_replaces_only_the_correct_candidates_answers(pool):
    cset = mixed_set(pool, 2, 1)
    plan = A.swap_plan(pool, cset)
    assert plan is not None
    wrong = plan.target
    inp = A.build_arm_input("answer_swap", pool, cset, cap=None)
    assert inp.detail["swap_to"] == wrong and inp.detail["n_swapped"] == 2
    assert inp.detail["correct_plurality"] is plan.correct_plurality
    shown = [s.split("\n", 1)[1] for s in body(inp).split("Solver ")[1:]]
    shown_answers = [extract_answer(s.split("\n\nWork out")[0]) for s in shown]
    cands = members(pool, cset)
    for c, shown_answer in zip(cands, shown_answers, strict=True):
        assert shown_answer == (wrong if c.correct else c.answer)


def test_answer_swap_does_not_apply_without_two_right_and_one_wrong(pool):
    assert A.build_arm_input("answer_swap", pool, mixed_set(pool, 1, 3)) is None
    assert A.build_arm_input("answer_swap", pool, mixed_set(pool, 3, 0)) is None


def test_unknown_arm_is_an_error(pool):
    with pytest.raises(ValueError, match="unknown reader arm"):
        A.build_arm_input("telepathy", pool, draw_set(pool, 3, 0))


def test_the_token_cap_is_applied_to_every_shown_candidate(pool):
    cset = draw_set(pool, 3, 0)
    inp = A.build_arm_input(
        "reader_full", pool, cset, cap=5, truncate=lambda text, n: text[:n]
    )
    assert body(inp).count("Solver ") == 3
    assert "trace" not in body(inp)  # every candidate cut to its first 5 characters
