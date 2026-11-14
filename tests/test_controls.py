"""Answer swap, deduplication and token caps."""

from __future__ import annotations

import pytest
from channel_audit.extract import extract_answer

from reader_audit.controls import (
    answer_span,
    cap_tokens,
    dedup_order,
    plurality_wrong_answer,
    swap_final_answer,
)

TEXTS = [
    "So x = 7.\n\\boxed{7}\nConfidence: 90",
    "Work.\n\\boxed{\\frac{1}{2}}\nConfidence: \\boxed{80}",  # a boxed confidence line
    "A {nested {group}} here: \\boxed{f(x) = \\{1, 2\\}} done",
    "First \\boxed{3} then corrected: \\boxed{4}",
    "The answer is \\boxed{12}.\n\nConfidence: 12",  # confidence equals the answer
]


@pytest.mark.parametrize("text", TEXTS)
def test_swap_changes_exactly_the_extracted_answer(text):
    swapped = swap_final_answer(text, "99")
    assert extract_answer(swapped) == "99"
    original = extract_answer(text)
    assert original is not None and original != "99"
    # everything outside the answer's box content is untouched
    start, end = answer_span(text)
    assert swapped == text[:start] + "99" + text[end:]


def test_swap_keeps_the_reasoning_and_the_confidence_line():
    text = "So x = 7.\n\\boxed{7}\nConfidence: 90"
    swapped = swap_final_answer(text, "8")
    assert swapped == "So x = 7.\n\\boxed{8}\nConfidence: 90"


def test_swap_leaves_an_earlier_box_with_the_same_content_alone():
    text = "Try \\boxed{5}, but really \\boxed{5}"
    assert swap_final_answer(text, "6") == "Try \\boxed{5}, but really \\boxed{6}"


def test_a_text_without_an_answer_cannot_be_swapped():
    with pytest.raises(ValueError, match="no final answer"):
        swap_final_answer("no box here", "1")
    assert answer_span("\\boxed{unclosed") is None


def test_plurality_wrong_answer_ignores_the_right_answer():
    answers = ["4", "4", "5", "5", "5", "6"]
    right = [True, True, False, False, False, False]
    assert plurality_wrong_answer(answers, right) == "5"


def test_plurality_wrong_answer_ties_go_to_the_first_shown_and_none_is_allowed():
    assert plurality_wrong_answer(["1", "2", "2", "1"], [False] * 4) == "1"
    assert plurality_wrong_answer(["3", "3"], [True, True]) is None
    assert plurality_wrong_answer([None, "3"], [False, True]) is None


def test_plurality_wrong_answer_pools_equivalent_answers():
    assert plurality_wrong_answer(["0.5", "\\frac{1}{2}", "9"], [False] * 3) == "0.5"


def test_dedup_orders_by_frequency_then_first_shown_and_puts_no_answer_last():
    answers = ["a", None, "b", "b", "a", "c", "b", None]
    # b x3 (first at 2), a x2 (first at 0), c x1 (at 5), then the no-answer group (1)
    assert dedup_order(answers) == [2, 0, 5, 1]


def test_dedup_of_identical_answers_is_one_candidate():
    assert dedup_order(["7", "7", "7"]) == [0]
    assert dedup_order([None, None]) == [0]


def test_cap_tokens_is_a_noop_without_a_tokenizer_or_cap():
    assert cap_tokens("a b c", None, lambda t, n: t[:n]) == "a b c"
    assert cap_tokens("a b c", 3, None) == "a b c"
    assert cap_tokens("abcdef", 3, lambda t, n: t[:n]) == "abc"


def test_middle_clipper_keeps_the_start_and_the_end_of_a_long_text():
    from reader_audit.controls import OMITTED, middle_clipper

    words = [f"w{i}" for i in range(100)]
    clip = middle_clipper(
        lambda t: t.split(), lambda ids: " ".join(ids), tail=5
    )  # one word is one token here
    text = " ".join(words)
    assert clip(text, 100) == text and clip(text, 200) == text
    out = clip(text, 20)
    assert out.startswith("w0 w1") and out.endswith("w95 w96 w97 w98 w99")
    assert "w14" in out and "w15" not in out  # 20 - 5 = 15 tokens from the start
    assert OMITTED in out


def test_the_boxed_answer_survives_clipping():
    from reader_audit.controls import middle_clipper

    clip = middle_clipper(lambda t: t.split(), lambda ids: " ".join(ids), tail=6)
    text = " ".join(f"step{i}" for i in range(500)) + r" so \boxed{42}"
    assert extract_answer(clip(text, 30)) == "42"
