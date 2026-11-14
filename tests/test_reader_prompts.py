"""Reader prompt templates and builders."""

from __future__ import annotations

import pytest
from channel_audit.models import load_model_config

from reader_audit import reader_prompts as rp

MODELS = rp.PROMPTS_DIR.parents[1] / "configs" / "models"


def test_every_arm_has_a_template_and_a_stable_hash():
    for arm in rp.ARMS:
        assert len(rp.template_hash(arm)) == 64
        assert "<<question>>" in rp.template_text(arm)


def test_arms_that_show_candidates_share_one_template():
    shown = [
        "reader_full",
        "placebo",
        "duplicate",
        "single_slot",
        "dedup",
        "answer_swap",
    ]
    assert len({rp.template_hash(a) for a in shown}) == 1
    assert rp.template_hash("candidate_free") != rp.template_hash("reader_full")
    assert rp.template_hash("answer_only") != rp.template_hash("reader_full")


def test_reader_prompt_lists_the_candidates_in_order():
    msgs = rp.build_prompt(
        "reader_full", "What is 1+1?", texts=["It is 2.", "Maybe 3."]
    )
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    assert body.startswith("Problem:\nWhat is 1+1?")
    assert "Here are 2 candidate solutions" in body
    assert body.index("Solver 1:\nIt is 2.") < body.index("Solver 2:\nMaybe 3.")
    assert "\\boxed{}" in body and "<<" not in body


def test_candidate_free_prompt_has_no_candidates_and_the_same_closing_line():
    body = rp.build_prompt("candidate_free", "Q?")[0]["content"]
    assert "Solver" not in body and "candidate" not in body
    assert body.endswith("Put your final answer in \\boxed{}.")


def test_answer_only_prompt_shows_answers_and_a_marker_for_none():
    body = rp.build_prompt("answer_only", "Q?", answers=["4", None, "5"])[0]["content"]
    assert "Solver 1: 4\nSolver 2: (no final answer)\nSolver 3: 5" in body
    assert "3 solvers" in body


def test_braces_in_the_inserted_text_are_not_expanded():
    msgs = rp.build_prompt("reader_full", "Q <<k>> {x}", texts=["a <<question>> b"])
    body = msgs[0]["content"]
    assert "Q <<k>> {x}" in body and "a <<question>> b" in body


def test_prepare_candidate_drops_the_confidence_line_and_caps_tokens():
    text = "Work.\n\\boxed{4}\nConfidence: 80"
    assert rp.prepare_candidate(text, cap=None) == "Work.\n\\boxed{4}"
    assert rp.prepare_candidate(text, cap=4, truncate=lambda t, n: t[:n]) == "Work"


def test_missing_inputs_and_unknown_arms_are_errors():
    with pytest.raises(ValueError):
        rp.build_prompt("reader_full", "Q")
    with pytest.raises(ValueError):
        rp.build_prompt("answer_only", "Q")
    with pytest.raises(ValueError):
        rp.template_hash("nope")


def test_the_pool_config_is_the_audits_config():
    from channel_audit.channels import PROMPTS_DIR as AUDIT_PROMPTS

    audit_cfg = load_model_config(
        AUDIT_PROMPTS.parent / "configs/models/qwen3-1.7b.yaml"
    )
    assert load_model_config(MODELS / "qwen3-1.7b-pool.yaml") == audit_cfg


def test_reader_configs_load_and_fit_the_prompts():
    reader = load_model_config(MODELS / "qwen3-1.7b-reader.yaml")
    assert reader.max_model_len == 16384 and reader.max_tokens == 2048
    big = load_model_config(MODELS / "qwen3-4b-instruct-2507.yaml")
    assert big.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert (
        big.max_model_len > 4 * 4096 + big.max_tokens
    )  # K = 4 candidates cut at 4,096
