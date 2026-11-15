"""The GRPO plan: batch arithmetic, rows per step, prompt-length and token counters."""

from __future__ import annotations

import pytest

from helpers import write_pool_shard
from reader_audit.grpo_plan import (
    GrpoPlan,
    assert_fits,
    build_rows,
    cumulative_rollout_tokens,
    prompt_lengths,
    rollout_tokens,
    step_rows,
)
from reader_audit.pools import load_pools


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=40, replicates=2, n_pool=6, base_p=0.5)
    return load_pools(root)


def test_the_default_plan_is_one_optimizer_step_per_generation_batch():
    plan = GrpoPlan()
    plan.check()
    assert plan.completions_per_step == 512 and plan.grad_accum == 512


def test_bad_plans_are_refused():
    with pytest.raises(ValueError, match="divide"):
        GrpoPlan(per_device_batch=3).check()
    with pytest.raises(ValueError, match="arm"):
        GrpoPlan(arm="other").check()
    with pytest.raises(ValueError, match="group"):
        GrpoPlan(group_size=1).check()


def test_rows_follow_the_arm_and_each_step_gets_its_own_slice(pools):
    plan = GrpoPlan(steps=3, prompts_per_step=4, group_size=2)
    rows = build_rows(plan, pools, truncate=None, cap=None)
    assert len(rows) == 12 and all(r["k"] in (2, 4, 8) for r in rows)
    first, second = step_rows(rows, plan, 0), step_rows(rows, plan, 1)
    assert len(first) == len(second) == 4 and first[0] is rows[0]
    assert second[0] is rows[4]
    no_c0 = build_rows(
        GrpoPlan(arm="reader_no_c0", steps=3, prompts_per_step=4, group_size=2),
        pools,
        truncate=None,
        cap=None,
    )
    assert all(r["stratum"] != "c0" for r in no_c0)
    solver = build_rows(
        GrpoPlan(arm="solver", steps=3, prompts_per_step=30, group_size=2),
        pools,
        truncate=None,
    )
    assert len(solver) == 90 and {r["stratum"] for r in solver} == {"solver"}


def test_too_few_rows_are_refused(pools):
    with pytest.raises(ValueError, match="needed"):
        build_rows(GrpoPlan(steps=500), pools, truncate=None, cap=None)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        assert kw["add_generation_prompt"] and kw["enable_thinking"] is False
        return list(range(sum(len(m["content"].split()) for m in messages)))


def test_prompt_lengths_use_the_non_thinking_template_and_the_window_is_checked(pools):
    plan = GrpoPlan(steps=1, prompts_per_step=4, group_size=2)
    rows = build_rows(plan, pools, truncate=None, cap=None)
    lengths = prompt_lengths(rows, FakeTokenizer())
    assert len(lengths) == 4 and assert_fits(lengths, plan) == max(lengths)
    with pytest.raises(ValueError, match="window"):
        assert_fits([plan.max_model_len - plan.max_completion + 1], plan)


def test_rollout_tokens_rebuild_from_a_log_history():
    plan = GrpoPlan(prompts_per_step=4, group_size=2)
    history = [
        {"step": 1, "completions/mean_length": 100.0},
        {"step": 1, "eval_loss": 1.0},
        {"step": 2, "completions/mean_length": 50.5},
    ]
    assert rollout_tokens(history[0], plan) == 800
    assert cumulative_rollout_tokens(history, plan) == 800 + 404
    assert cumulative_rollout_tokens([], plan) == 0
