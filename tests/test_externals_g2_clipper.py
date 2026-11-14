"""External problem sets, the G2 verdict and the tokenizer clipper."""

from __future__ import annotations

import pytest
from channel_audit.extract import extract_answer

from reader_audit import externals
from reader_audit.clipper import clipper_from_tokenizer
from reader_audit.controls import OMITTED
from reader_audit.g2 import CHECKS, REPORT_ONLY, format_verdict, g2_verdict
from reader_audit.reader_eval import VOTE, Outcome

ROWS = [
    {"problem_idx": "1", "problem": "Find $x$.", "answer": "70"},
    {"problem_idx": "2", "problem": "Compute $y$.", "answer": "\\frac{3}{2}"},
]


def test_matharena_rows_become_problems_with_stable_ids():
    problems = externals.matharena_problems(ROWS, "aime25", "aime_2025")
    assert [p.problem_id for p in problems] == ["aime25-01", "aime25-02"]
    assert all(p.subject == "competition" and p.level == 5 for p in problems)
    assert {p.split for p in problems} == {"check"}
    assert problems[1].gold == "\\frac{3}{2}"


@pytest.mark.parametrize(
    "bad",
    [
        {"problem_idx": "3", "problem": "", "answer": "1"},
        {"problem_idx": "3", "problem": "Q", "answer": ""},
        {"problem": "Q", "answer": "1"},
    ],
)
def test_a_row_without_a_question_gold_or_index_is_an_error(bad):
    with pytest.raises(externals.GoldError):
        externals.matharena_problems([*ROWS, bad], "p", "d")


def test_duplicate_indices_are_an_error():
    with pytest.raises(externals.GoldError, match="duplicate"):
        externals.matharena_problems([ROWS[0], ROWS[0]], "p", "d")


def test_golds_that_a_boxed_answer_cannot_match_are_found():
    problems = externals.matharena_problems(ROWS, "p", "d")
    assert externals.check_golds(problems) == []
    broken = problems[0].model_copy(update={"gold": "{1", "problem_id": "p-99"})
    assert externals.check_golds([broken]) == ["p-99"]


def test_content_hash_changes_with_any_gold_or_question():
    a = externals.matharena_problems(ROWS, "p", "d")
    b = [a[0].model_copy(update={"gold": "71"}), a[1]]
    assert externals.content_sha256(a) == externals.content_sha256(list(a))
    assert externals.content_sha256(a) != externals.content_sha256(b)


class WordTokenizer:
    """One word is one token; ids are indices into a growing vocabulary."""

    def __init__(self):
        self.vocab: list[str] = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        ids = []
        for word in text.split(" "):
            if word not in self.vocab:
                self.vocab.append(word)
            ids.append(self.vocab.index(word))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        return " ".join(self.vocab[i] for i in ids)


def test_the_tokenizer_clipper_keeps_the_answer_of_a_long_candidate():
    clip = clipper_from_tokenizer(WordTokenizer(), tail=4)
    text = " ".join(f"s{i}" for i in range(300)) + " so \\boxed{42}"
    out = clip(text, 20)
    assert OMITTED in out and extract_answer(out) == "42"
    assert clip("short text", 20) == "short text"


def outcome(arm, stratum, problem, value, draw=0, k=4):
    return Outcome(problem, arm, k, draw, 0, stratum, 0, value)


def synthetic(signs):
    """Outcomes whose arms differ by the given amounts on 12 problems per stratum."""
    out = []
    for stratum, arms in signs.items():
        for p in range(12):
            for arm, value in arms.items():
                out.append(
                    outcome(arm, stratum, f"{stratum}-{p}", value + 0.01 * (p % 3))
                )
    return out


GOOD = {
    "c0": {"reader_full": 0.1, "candidate_free": 0.3, "placebo": 0.0},
    "c1": {"reader_full": 0.5, "candidate_free": 0.4, "answer_only": 0.5},
    "c2+": {
        "reader_full": 0.8,
        "candidate_free": 0.6,
        VOTE: 0.95,
        "answer_only": 0.9,
        "placebo": 0.5,
    },
}


def test_g2_passes_when_all_four_signs_match():
    v = g2_verdict(synthetic(GOOD), 4, n_boot=500)
    assert v["passed"] is True
    assert set(v["checks"]) == {c.name for c in CHECKS}
    assert set(v["report_only"]) == {c.name for c in REPORT_ONLY}
    assert "PASS" in format_verdict(v)


@pytest.mark.parametrize(
    "stratum,arm,value,failing",
    [
        ("c0", "reader_full", 0.5, "c0_reader_below_candidate_free"),
        ("c0", "placebo", 0.6, "c0_placebo_below_candidate_free"),
        ("c2+", "reader_full", 0.55, "c2_reader_above_candidate_free"),
        ("c2+", "reader_full", 0.99, "c2_vote_above_reader"),
    ],
)
def test_g2_fails_when_one_sign_flips(stratum, arm, value, failing):
    bad = {s: dict(arms) for s, arms in GOOD.items()}
    bad[stratum][arm] = value
    v = g2_verdict(synthetic(bad), 4, n_boot=500)
    assert v["passed"] is False and v["checks"][failing]["passes"] is False


def test_g2_needs_the_interval_to_exclude_zero_at_c2():
    noisy = []
    for p in range(
        12
    ):  # the gain is +0.1 on average but swings by 0.8 between problems
        swing = 0.9 if p % 2 else -0.7
        noisy.append(outcome("reader_full", "c2+", f"n{p}", 0.5 + swing))
        noisy.append(outcome("candidate_free", "c2+", f"n{p}", 0.5))
    v = g2_verdict(noisy, 4, n_boot=500)
    check = v["checks"]["c2_reader_above_candidate_free"]
    assert check["mean"] > 0 and check["ci95_low"] < 0 and check["passes"] is False


def test_a_missing_stratum_is_not_run_and_fails_the_gate():
    only_c2 = {"c2+": GOOD["c2+"]}
    v = g2_verdict(synthetic(only_c2), 4, n_boot=200)
    assert v["passed"] is False
    assert v["checks"]["c0_reader_below_candidate_free"]["status"].startswith("not run")


def fake_loader(name, revision):
    prefix = {"aime_2025": "aime25", "hmmt_feb_2025": "hmmt25"}[name]
    return externals.matharena_problems(ROWS, prefix, name)


def test_the_harness_set_is_pinned_once_and_verified_on_load(tmp_path):
    manifest = externals.pin_harness(tmp_path, lambda repo: "a" * 40, fake_loader)
    assert set(manifest) == set(externals.HARNESS_SETS)
    assert (
        manifest["aime_2025"]["n"] == 2
        and manifest["aime_2025"]["revision"] == "a" * 40
    )
    with pytest.raises(FileExistsError):
        externals.pin_harness(tmp_path, lambda repo: "b" * 40, fake_loader)
    problems = externals.load_harness(tmp_path, fake_loader)
    assert [p.problem_id for p in problems] == [
        "aime25-01",
        "aime25-02",
        "hmmt25-01",
        "hmmt25-02",
    ]


def test_a_changed_set_is_caught_on_load(tmp_path):
    externals.pin_harness(tmp_path, lambda repo: "a" * 40, fake_loader)

    def changed(name, revision):
        rows = [{**ROWS[0], "answer": "71"}, ROWS[1]]
        return externals.matharena_problems(rows, "x", name)

    with pytest.raises(externals.GoldError, match="no longer matches"):
        externals.load_harness(tmp_path, changed)


def test_pinning_refuses_a_gold_a_box_cannot_match(tmp_path):
    def bad_loader(name, revision):
        return externals.matharena_problems(
            [{"problem_idx": "1", "problem": "Q", "answer": "{1"}], "p", name
        )

    with pytest.raises(externals.GoldError, match="cannot match"):
        externals.pin_harness(tmp_path, lambda repo: "a" * 40, bad_loader)
