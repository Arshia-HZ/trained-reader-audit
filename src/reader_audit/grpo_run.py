"""GRPO training of the reader (and the solver baseline) with TRL's ``GRPOTrainer``.

Needs the training environment (torch, trl, transformers; vLLM for ``use_vllm``),
which is kept apart from the evaluation one. The plan, rows, prompt checks and counters
live in ``grpo_plan``; the reward is ``train_data.correctness_reward`` (the audit's
own grade).

Per step the run keeps ``steps.jsonl`` (step, seconds, mean K, prompt tokens, rollout
tokens and their running total, peak GPU memory, the problem ids) and ``batches.jsonl``
(the problem ids each generation batch really contained, written by the reward). The
rollout-token total is rebuilt from the trainer's own log history, so it survives a kill
and a resume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from reader_audit.grpo_plan import (
    CHAT_KWARGS,
    GrpoPlan,
    assert_fits,
    cumulative_rollout_tokens,
    prompt_lengths,
    rollout_tokens,
    step_rows,
)
from reader_audit.grpo_upload import CheckpointUploader
from reader_audit.train_data import correctness_reward


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


_SCORING_OPTIONS = ("batch_size", "compute_entropy", "compute_aux_loss")


def trim_padding(inner):
    """Score each sequence at its own length instead of the batch's longest.

    TRL pads a whole generation batch to its longest prompt and longest completion and
    scores every micro-batch at that size. With 512 sequences the longest is near the
    window, so each one costs about twice its own length (and attention grows with the
    square). This wrapper cuts the left padding of the prompt and the right padding of
    the completion, scores the row alone, and pads the result back to the shape the loss
    expects (padded positions are masked there). Relative positions are unchanged, so
    the scores match the padded ones up to numerical noise. Anything that is not plain
    text goes through ``inner`` untouched.
    """

    def trimmed(model, input_ids, attention_mask, logits_to_keep, **kw):
        if any(v is not None for k, v in kw.items() if k not in _SCORING_OPTIONS):
            return inner(model, input_ids, attention_mask, logits_to_keep, **kw)
        kw = dict(kw, batch_size=1)
        width = input_ids.size(1)
        logps, entropies, aux = [], [], []
        for i in range(input_ids.size(0)):
            ids, mask = input_ids[i : i + 1], attention_mask[i : i + 1]
            live = mask[0].nonzero().flatten()
            lo, hi = int(live[0]), int(live[-1]) + 1
            keep = logits_to_keep - (width - hi)
            if keep < 1 or not bool(mask[0, lo:hi].all()):
                lo, hi, keep = 0, width, logits_to_keep
            lp, en, ax = inner(model, ids[:, lo:hi], mask[:, lo:hi], keep, **kw)
            pad = logits_to_keep - keep
            logps.append(F.pad(lp, (0, pad)))
            if en is not None:
                entropies.append(F.pad(en, (0, pad)))
            if ax is not None:
                aux.append(ax)
        return (
            torch.cat(logps, dim=0),
            torch.cat(entropies, dim=0) if entropies else None,
            torch.stack(aux).mean() if aux else None,
        )

    return trimmed


class BatchRecorder:
    """Wraps the reward so each generation batch leaves a record of its prompts."""

    def __init__(self, path: Path, group_size: int, reward=correctness_reward) -> None:
        self.path = path
        self.group_size = group_size
        self.reward = reward
        self.trainer = None
        self.calls = 0

    def __call__(self, completions, gold, **kw):
        ids = list(kw.get("problem_id", []))[:: self.group_size]
        step = self.trainer.state.global_step if self.trainer is not None else -1
        _append(
            self.path,
            {"call": self.calls, "step": step, "n": len(completions), "ids": ids},
        )
        self.calls += 1
        return self.reward(completions, gold)


class PhaseTimer:
    """Seconds spent generating and scoring rollouts since the last read."""

    def __init__(self) -> None:
        self.generation = 0.0

    def wrap(self, trainer) -> None:
        inner = trainer._generate_and_score_completions

        def timed(*args, **kwargs):
            start = time.monotonic()
            try:
                return inner(*args, **kwargs)
            finally:
                self.generation += time.monotonic() - start

        trainer._generate_and_score_completions = timed

    def take(self) -> float:
        out, self.generation = self.generation, 0.0
        return out


class RunLog(TrainerCallback):
    def __init__(
        self,
        plan: GrpoPlan,
        rows: Sequence[dict],
        out_dir: Path,
        uploader: CheckpointUploader | None = None,
        timer: PhaseTimer | None = None,
    ) -> None:
        self.timer = timer
        self.plan = plan
        self.rows = rows
        self.path = out_dir / "steps.jsonl"
        self.uploader = uploader
        self.last = time.monotonic()

    def on_train_begin(self, args, state, control, **kw):
        self.last = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "completions/mean_length" not in logs:
            return
        now = time.monotonic()
        # the callback gets a copy of the entry the trainer stores; leave this step out
        history = [e for e in state.log_history if e.get("step") != state.global_step]
        tokens = rollout_tokens(logs, self.plan)
        total = cumulative_rollout_tokens(history, self.plan) + tokens
        rows = step_rows(self.rows, self.plan, state.global_step - 1)
        peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        )
        generation = self.timer.take() if self.timer is not None else None
        record = {
            "step": state.global_step,
            "seconds": now - self.last,
            "generation_seconds": generation,
            "mean_k": sum(r["k"] for r in rows) / max(len(rows), 1),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows)
            / max(len(rows), 1),
            "rollout_tokens": tokens,
            "rollout_tokens_cum": total,
            "peak_gib": peak,
            "reward": logs.get("reward"),
            "loss": logs.get("loss"),
            "ids": [r["problem_id"] for r in rows],
        }
        _append(self.path, record)
        self.last = now

    def on_save(self, args, state, control, **kw):
        if self.uploader is not None:
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            self.uploader.on_checkpoint(state.global_step, folder)


def make_config(plan: GrpoPlan, out_dir: str | Path, **kw) -> GRPOConfig:
    """TRL settings for a plan; ``kw`` overrides the recorded starting values."""
    plan.check()
    settings = dict(
        output_dir=str(out_dir),
        run_name=Path(out_dir).name,
        per_device_train_batch_size=plan.per_device_batch,
        gradient_accumulation_steps=plan.grad_accum,
        num_generations=plan.group_size,
        max_completion_length=plan.max_completion,
        temperature=1.0,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=False,
        learning_rate=1e-6,
        warmup_steps=10,
        lr_scheduler_type="constant_with_warmup",
        max_steps=plan.steps,
        save_strategy="steps",
        save_steps=plan.save_every,
        save_total_limit=1,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        shuffle_dataset=False,
        remove_unused_columns=False,
        chat_template_kwargs=dict(CHAT_KWARGS),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=plan.max_model_len,
        report_to="none",
        seed=plan.seed,
        data_seed=plan.seed,
    )
    settings.update(kw)
    cfg = GRPOConfig(**settings)
    if cfg.steps_per_generation != cfg.gradient_accumulation_steps:
        raise ValueError("one optimizer step must consume exactly one generation batch")
    return cfg


def train(
    plan: GrpoPlan,
    rows: list[dict],
    out_dir: str | Path,
    *,
    model_id: str,
    revision: str | None = None,
    uploader: CheckpointUploader | None = None,
    resume: bool = True,
    reward=correctness_reward,
    policy_dtype=torch.float32,
    trim=True,
    **config_overrides,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    lengths = prompt_lengths(rows, tokenizer)
    longest = assert_fits(lengths, plan)
    for row, n in zip(rows, lengths, strict=True):
        row["prompt_tokens"] = n
    print(
        f"{len(rows)} prompts, longest {longest} tokens, "
        f"mean {sum(lengths) / len(lengths):.0f}",
        flush=True,
    )
    # fp32 master weights under bf16 autocast: a 1e-6 step on bf16 weights rounds away
    init_kwargs = {"dtype": policy_dtype}
    if revision:
        init_kwargs["revision"] = revision
    config_overrides.setdefault("model_init_kwargs", init_kwargs)
    cfg = make_config(plan, out, **config_overrides)
    recorder = BatchRecorder(out / "batches.jsonl", plan.group_size, reward)
    recorder.__name__ = "correctness"
    timer = PhaseTimer()
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=recorder,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[RunLog(plan, rows, out, uploader, timer)],
    )
    timer.wrap(trainer)
    if trim:
        trainer._get_per_token_logps_and_entropies = trim_padding(
            trainer._get_per_token_logps_and_entropies
        )
    recorder.trainer = trainer
    last = get_last_checkpoint(str(out)) if resume else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    return trainer
