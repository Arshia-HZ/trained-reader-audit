"""The training-set builder: normalisation, decontamination and the seeded mix."""

from __future__ import annotations

import pytest

from reader_audit.train_set import (
    Decontaminator,
    Reference,
    TrainProblem,
    build_train_set,
    ids_sha256,
    load_deepscaler,
    load_math_train,
    normalize,
)

LONG = (
    "Let a and b be positive integers such that the sum of their squares is two "
    "hundred and the product is eighty four. Find the value of a plus b."
)


def test_normalisation_ignores_case_markup_and_spacing():
    assert normalize("What is $x^2+1$?  Find  X.") == [
        "what",
        "is",
        "x",
        "2",
        "1",
        "find",
        "x",
    ]


def test_an_identical_problem_is_caught_whatever_its_formatting():
    dec = Decontaminator([Reference("dev", "d1", "Compute 2+2.")])
    assert dec.match("compute  $2 + 2$")[0] == "identical"
    assert dec.match("Compute 2+3.") is None


def test_a_shared_thirteen_word_run_is_caught_and_a_shorter_one_is_not():
    dec = Decontaminator([Reference("primary", "p1", LONG)])
    copied = "Problem: " + LONG + " Show your work."
    reason, ref = dec.match(copied)
    assert reason == "ngram" and ref.set_name == "primary"
    twelve = " ".join(normalize(LONG)[:12]) + " zebra quantum lattice"
    assert dec.match(twelve) is None


def test_short_references_match_only_by_identity():
    dec = Decontaminator([Reference("dev", "d1", "Find x if 3x = 12.")])
    assert (
        dec.match("Find x if 3x = 12 and y = 4, then compute x plus y squared.") is None
    )


def row(level, answer="7"):
    return {
        "level": f"Level {level}",
        "type": "Algebra",
        "problem": f"Question number {level} {answer}",
        "solution": f"so the answer is $\\boxed{{{answer}}}$",
    }


def test_math_train_keeps_the_study_levels_and_lists_unusable_rows():
    rows = {c: [] for c in ("algebra",)}

    def rows_for(config):
        if config == "algebra":
            return [row(2), row(3), row(5), {**row(4), "solution": "no box"}]
        return []

    problems, dropped = load_math_train(rows_for)
    assert {p.level for p in problems} == {3, 5}
    assert dropped == ["math-train-algebra-3"]
    assert problems[0].to_problem().split == "check"
    assert rows == {"algebra": []}


def test_deepscaler_rows_without_an_answer_are_skipped():
    problems = load_deepscaler(
        [{"problem": "q1", "answer": "5"}, {"problem": "q2", "answer": " "}]
    )
    assert [p.gold for p in problems] == ["5"] and problems[0].level is None
    assert problems[0].to_problem().level == 5


def make(source, n, prefix):
    return [
        TrainProblem(
            f"{source}-{i}",
            source,
            "x",
            3 if source == "math-train" else None,
            f"{prefix} problem number {i} about some topic {i * 7}",
            "1",
        )
        for i in range(n)
    ]


def test_the_mix_is_seeded_clean_and_free_of_duplicates():
    math = make("math-train", 5, "alpha")
    deep = make("deepscaler", 20, "beta") + [
        TrainProblem("deepscaler-dup", "deepscaler", "x", None, math[0].question, "1")
    ]
    refs = Decontaminator([Reference("dev", "d1", deep[3].question)])
    kept, report = build_train_set(math, deep, refs, n_deepscaler=8)
    again, _ = build_train_set(math, deep, refs, n_deepscaler=8)
    assert [p.problem_id for p in kept] == [p.problem_id for p in again]
    assert report["kept"] == {"math-train": 5, "deepscaler": 8}
    assert deep[3].problem_id not in {p.problem_id for p in kept}
    assert report["dropped_by_set"].get("dev", 0) in (0, 1)
    assert len({" ".join(normalize(p.question)) for p in kept}) == len(kept)
    other, _ = build_train_set(math, deep, refs, n_deepscaler=8, seed=1)
    assert {p.problem_id for p in other} != {p.problem_id for p in kept}


def test_a_short_source_is_reported_not_hidden():
    kept, report = build_train_set(
        make("math-train", 2, "g"),
        make("deepscaler", 3, "h"),
        Decontaminator([]),
        n_deepscaler=10,
    )
    assert report["short"] == 7 and len(kept) == 5


def test_the_id_hash_does_not_depend_on_order():
    assert ids_sha256(["b", "a"]) == ids_sha256(["a", "b"])
    with pytest.raises(AssertionError):
        assert ids_sha256(["a"]) == ids_sha256(["b"])
