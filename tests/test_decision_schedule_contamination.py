"""Conclusion rules, the seed-conditional schedule and the contamination protocol."""

from __future__ import annotations

import pytest

from reader_audit import contamination as C
from reader_audit import decision as D
from reader_audit import schedule as S

# ---- framing and verdicts (D8, D26) --------------------------------------------


@pytest.mark.parametrize(
    "h1a,h1b,expected",
    [
        (True, True, "C1-full"),
        (True, False, "C1-narrow"),
        (False, False, "C2-lead"),
        (False, True, "C2-lead"),  # H1b cannot rescue H1a
    ],
)
def test_the_framing_follows_h1a_and_h1b(h1a, h1b, expected):
    assert D.framing(h1a, h1b) == expected
    assert expected in D.FRAMINGS


def test_agreement_across_seeds_never_makes_an_unsupported_result_supported():
    assert D.verdict(False, [0.02, 0.03, 0.04]) == "not supported"
    assert D.verdict(True, [0.02, 0.03, 0.04]) == "supported and robust"
    assert D.verdict(True, [0.02, -0.01, 0.04]) == "supported, not robust"
    assert D.verdict(True, []) == "supported, not robust"  # no seed evidence, no claim


def test_the_seed_rule_follows_the_measured_cost_of_a_run():
    assert D.seeds_for(12.0) == 5 and D.seeds_for(20.0) == 5
    assert D.seeds_for(20.1) == 3 and D.seeds_for(35.0) == 3
    with pytest.raises(ValueError):
        D.seeds_for(0)


# ---- the schedule is conditional on the seed rule (D7) ------------------------


def test_the_stage_3_runs_follow_the_seed_count():
    assert S.main_runs(3) == {
        "reader": 3,
        "solver_grpo": 3,
        "joint": 1,
        "no_c0": 1,
        "reader_0.6b": 1,
    }
    assert S.main_runs(5)["reader"] == 5 and S.main_runs(5)["solver_grpo"] == 5


def test_the_budget_for_three_seeds_matches_the_prereg_table():
    b = S.budget(3)
    assert b["low"] == pytest.approx(282.5, abs=0.1)
    assert b["high"] == pytest.approx(590.0, abs=0.1)
    assert b["weeks"][0] == pytest.approx(1.68, abs=0.01)
    assert b["weeks"][1] == pytest.approx(3.51, abs=0.01)


def test_the_budget_for_five_seeds_is_capped_by_the_rule_that_triggered_it():
    b = S.five_seed_budget()
    assert b["low"] == pytest.approx(373.8, abs=0.2)
    assert b["high"] == pytest.approx(652.7, abs=0.2)
    uncapped = S.budget(5)
    assert uncapped["high"] > b["high"] > S.budget(3)["high"]
    assert (
        b["by_item"]["reader"][1] == 100.0 and b["by_item"]["solver_grpo"][1] == 170.0
    )


def test_evaluation_hours_scale_with_the_number_of_checkpoints():
    three, five = (
        S.budget(3)["by_item"]["evaluation"],
        S.budget(5)["by_item"]["evaluation"],
    )
    assert three == (48.0, 96.0)
    assert five[0] == pytest.approx(48.0 * 13 / 9)


def test_dropping_the_factorial_removes_exactly_its_runs():
    with_f, without = S.budget(3), S.budget(3, include_factorial=False)
    assert with_f["low"] - without["low"] == pytest.approx(15.0 + 2 * 25.0)
    assert with_f["high"] - without["high"] == pytest.approx(30.0 + 2 * 60.0)


def test_the_timeline_gains_a_week_with_five_seeds():
    assert S.timeline(3) == {
        "stage_3_training": (3, 5),
        "stage_3_evaluation": (6, 6),
        "stage_4": (7, 8),
    }
    assert S.timeline(5) == {
        "stage_3_training": (3, 6),
        "stage_3_evaluation": (7, 7),
        "stage_4": (8, 9),
    }


# ---- the contamination probe protocol (D13) ------------------------------------


def words(text):
    return text.split(" ")


def test_the_prefix_is_the_first_sixty_percent_of_the_tokens():
    text = " ".join(f"w{i}" for i in range(10))
    prefix, remainder = C.split_statement(text, words, " ".join)
    assert prefix == "w0 w1 w2 w3 w4 w5"
    assert remainder == " w6 w7 w8 w9"  # cut from the original where the prefix ends


def test_the_prefix_length_is_floored():
    text = " ".join(f"w{i}" for i in range(7))  # 0.6 * 7 = 4.2 -> 4
    prefix, _ = C.split_statement(text, words, " ".join)
    assert len(prefix.split(" ")) == 4


def test_a_completion_reproducing_the_remainder_is_memorised_whatever_its_spacing():
    assert C.is_memorised("w6 w7  w8\nw9 and more text", " w6 w7 w8 w9")
    assert C.is_memorised("w6w7w8w9", " w6 w7 w8 w9")
    assert not C.is_memorised("w6 w7 w8 other", " w6 w7 w8 w9")
    assert not C.is_memorised("", " w6 w7")
    assert not C.is_memorised("anything", "   ")  # an empty remainder is never a match


def test_a_completion_must_start_with_the_remainder():
    assert not C.is_memorised("noise w6 w7", " w6 w7")


def test_the_generation_budget_is_one_point_two_times_the_remainder_plus_sixteen():
    assert C.max_new_tokens(100) == 136
    assert C.max_new_tokens(7) == 9 + 16


def test_the_primary_flag_is_the_studied_models_and_the_base_model_is_reported_apart():
    f = C.flags({"studied": False, "base": True})
    assert f == {"primary": False, "base": True, "either": True}
    assert C.flags({"studied": True, "base": False})["primary"] is True
    with pytest.raises(ValueError, match="missing"):
        C.flags({"studied": True})


def test_the_h2_split_keeps_the_problems_the_probe_does_not_flag():
    by_problem = {
        "a": C.flags({"studied": False, "base": False}),
        "b": C.flags({"studied": False, "base": True}),
        "c": C.flags({"studied": True, "base": True}),
    }
    assert C.probe_negative(by_problem) == {"a", "b"}
    assert C.probe_negative(by_problem, "either") == {"a"}
    assert C.probe_negative(by_problem, "base") == {"a"}
    with pytest.raises(ValueError):
        C.probe_negative(by_problem, "union")


def test_the_two_probed_models_are_the_studied_one_and_its_base():
    assert C.MODELS == {"studied": "Qwen/Qwen3-1.7B", "base": "Qwen/Qwen3-1.7B-Base"}
