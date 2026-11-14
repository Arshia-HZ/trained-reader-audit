"""Study sets derived from a frozen confirm split."""

from __future__ import annotations

import json

import pytest
from channel_audit.data import write_split
from channel_audit.schema import Problem

from reader_audit import sets


def make_confirm(n: int = 40) -> list[Problem]:
    return [
        Problem(
            problem_id=f"math-test-algebra-{i}",
            dataset="math-test",
            subject="Algebra",
            level=3 + i % 3,
            question=f"q{i}",
            gold=str(i),
            split="confirm",
        )
        for i in range(n)
    ]


def test_dev_is_the_prefix_and_primary_is_later_level_five_only():
    confirm = make_confirm(40)
    derived = sets.derive_math_sets(confirm, n_dev=10)
    assert [p.problem_id for p in derived["dev"]] == [
        p.problem_id for p in confirm[:10]
    ]
    assert {p.level for p in derived["primary"]} == {5}
    assert all(
        p.problem_id not in {d.problem_id for d in derived["dev"]}
        for p in derived["primary"]
    )
    assert len(derived["primary"]) + len(derived["reserve"]) == 30
    assert {p.level for p in derived["reserve"]} == {3, 4}


def test_every_confirm_problem_is_in_exactly_one_set():
    confirm = make_confirm(40)
    derived = sets.derive_math_sets(confirm, n_dev=10)
    ids = [p.problem_id for name in sets.MATH_SETS for p in derived[name]]
    assert sorted(ids) == sorted(p.problem_id for p in confirm)


def test_membership_does_not_depend_on_content_only_on_id_order_and_level():
    confirm = make_confirm(40)
    changed = [p.model_copy(update={"gold": "0", "question": "x"}) for p in confirm]
    a = sets.derive_math_sets(confirm, n_dev=10)
    b = sets.derive_math_sets(changed, n_dev=10)
    assert {k: [p.problem_id for p in v] for k, v in a.items()} == {
        k: [p.problem_id for p in v] for k, v in b.items()
    }


def test_too_small_a_split_is_an_error():
    with pytest.raises(ValueError, match="need >"):
        sets.derive_math_sets(make_confirm(5), n_dev=10)


def test_manifest_is_written_once_and_a_change_is_refused(tmp_path):
    confirm = make_confirm(40)
    derived = sets.derive_math_sets(confirm, n_dev=10)
    manifest = sets.write_sets(tmp_path, derived, b"audit manifest")
    assert manifest["sets"]["dev"]["n"] == 10
    assert (
        sets.write_sets(tmp_path, derived, b"audit manifest") == manifest
    )  # idempotent
    with pytest.raises(ValueError, match="already written"):
        sets.write_sets(tmp_path, derived, b"another audit manifest")
    changed = dict(derived, primary=derived["primary"][:-1])
    with pytest.raises(ValueError, match="already written"):
        sets.write_sets(tmp_path, changed, b"audit manifest")


def test_read_ids_checks_the_manifest(tmp_path):
    derived = sets.derive_math_sets(make_confirm(40), n_dev=10)
    sets.write_sets(tmp_path, derived, b"m")
    assert sets.read_ids(tmp_path, "primary") == [
        p.problem_id for p in derived["primary"]
    ]
    path = tmp_path / "primary.ids.json"
    path.write_text(json.dumps(["tampered"]), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        sets.read_ids(tmp_path, "primary")


def test_load_math_set_reads_the_audits_frozen_file(tmp_path):
    confirm = make_confirm(40)
    # the audit's frozen files need a screen and a pilot file as well
    screen = [
        p.model_copy(update={"split": "screen", "problem_id": f"s{i}"})
        for i, p in enumerate(confirm[:3])
    ]
    pilot = [
        p.model_copy(update={"split": "pilot", "problem_id": f"p{i}"})
        for i, p in enumerate(confirm[:3])
    ]
    from channel_audit.data import SplitResult

    split = SplitResult(screen=screen, pilot=pilot, confirm=confirm, excluded_ids=[])
    write_split(split, tmp_path / "audit", n_rows=46)
    derived = sets.derive_math_sets(confirm, n_dev=10)
    sets.write_sets(tmp_path / "sets", derived, b"m")
    loaded = sets.load_math_set("primary", tmp_path / "sets", tmp_path / "audit")
    assert [p.problem_id for p in loaded] == [p.problem_id for p in derived["primary"]]
