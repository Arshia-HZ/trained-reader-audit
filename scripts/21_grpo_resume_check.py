"""CPU check of the GRPO plumbing: a tiny random model, mock pools, kill and resume.

    python scripts/21_grpo_resume_check.py --work runs/resume_check

Run A trains ``--steps`` steps without a break. Run B is killed (SIGKILL) as soon as its
first checkpoint is written, then started again and left to finish. The two must agree
on which prompts every step used, and B's rollout-token total must equal the sum of its
own steps. Losses and sampled tokens are compared and reported: they agree only if the
checkpoint restores the random state, which is what a resume needs. It uses the real
data path, reward, callbacks and checkpointing; only vLLM and the GPU are not tested.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def make_tiny(path: Path) -> None:
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def mixed_reward(completions, gold, **_):
    """Reward that varies across a group, so the loss and its gradient are not zero."""
    out = []
    for c in completions:
        text = c if isinstance(c, str) else "".join(m["content"] for m in c)
        out.append(float(sum(map(ord, text)) % 2))
    return out


def worker(args) -> int:
    sys.path.insert(0, str(ROOT / "tests"))
    from transformers import AutoTokenizer

    from reader_audit.clipper import clipper_from_tokenizer
    from reader_audit.grpo_plan import GrpoPlan, build_rows
    from reader_audit.grpo_run import train
    from reader_audit.pools import load_pools

    tiny = Path(args.work) / "tiny"
    tok = AutoTokenizer.from_pretrained(tiny)
    plan = GrpoPlan(
        steps=args.steps,
        prompts_per_step=3,
        group_size=2,
        per_device_batch=1,
        max_completion=12,
        save_every=2,
    )
    pools = load_pools(Path(args.work) / "pools")
    rows = build_rows(plan, pools, truncate=clipper_from_tokenizer(tok))
    train(
        plan,
        rows,
        args.out,
        model_id=str(tiny),
        reward=mixed_reward,
        use_vllm=False,
        bf16=False,
        gradient_checkpointing=False,
        use_cpu=True,
        learning_rate=1e-3,
        warmup_steps=0,
        model_init_kwargs={},
        disable_tqdm=True,
    )
    return 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]


def by_step(records: list[dict]) -> dict[int, dict]:
    return {r["step"]: r for r in records}  # a repeated step keeps the last record


def spawn(work: Path, out: Path, steps: int) -> subprocess.Popen:
    cmd = [sys.executable, __file__, "--worker", "--work", str(work)]
    cmd += ["--out", str(out), "--steps", str(steps)]
    return subprocess.Popen(cmd)


def orchestrate(args) -> int:
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import write_pool_shard

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "tiny").exists():
        make_tiny(work / "tiny")
    if not (work / "pools").exists():
        write_pool_shard(
            work / "pools", n_problems=40, replicates=2, n_pool=12, base_p=0.5
        )
    a, b = work / "run_a", work / "run_b"
    assert spawn(work, a, args.steps).wait() == 0, "run A failed"

    proc = spawn(work, b, args.steps)
    while not list(b.glob("checkpoint-*")) and proc.poll() is None:
        time.sleep(0.2)
    time.sleep(0.5)  # let the checkpoint finish writing, then kill hard
    killed = proc.poll() is None
    proc.kill()
    proc.wait()
    first = sorted(p.name for p in b.glob("checkpoint-*"))
    print(f"run B killed: {killed}; checkpoints on disk: {first}")
    assert spawn(work, b, args.steps).wait() == 0, "resumed run B failed"

    sa, sb = (
        by_step(read_jsonl(a / "steps.jsonl")),
        by_step(read_jsonl(b / "steps.jsonl")),
    )
    ok = True

    def check(name: str, cond: bool, hard: bool = True) -> None:
        nonlocal ok
        print(f"{'PASS' if cond else ('FAIL' if hard else 'note')}  {name}")
        ok = ok and (cond or not hard)

    check("every step is logged in both runs", sorted(sa) == sorted(sb))
    check(
        "same prompts at every step",
        all(sa[s]["ids"] == sb[s]["ids"] for s in sa),
    )
    ba, bb = read_jsonl(a / "batches.jsonl"), read_jsonl(b / "batches.jsonl")
    ids_a = {r["step"]: r["ids"] for r in ba}
    ids_b = {r["step"]: r["ids"] for r in bb}
    check("generation batches contain the logged prompts", ids_a == ids_b)
    check(
        "the generation batch of step s holds the prompts of step s+1",
        all(sa[s + 1]["ids"] == ids_a[s] for s in ids_a if s + 1 in sa),
    )
    run = 0
    cum_ok = True
    for s in sorted(sb):
        run += sb[s]["rollout_tokens"]
        cum_ok = cum_ok and sb[s]["rollout_tokens_cum"] == run
    check("rollout-token total continues across the resume", cum_ok)
    from reader_audit.grpo_plan import GrpoPlan, cumulative_rollout_tokens

    state = json.loads(
        (sorted(b.glob("checkpoint-*"))[-1] / "trainer_state.json").read_text("utf-8")
    )
    plan = GrpoPlan(prompts_per_step=3, group_size=2)
    saved = [e for e in state["log_history"] if e["step"] <= state["global_step"]]
    check(
        "the saved trainer state rebuilds the same total",
        cumulative_rollout_tokens(saved, plan)
        == sb[state["global_step"]]["rollout_tokens_cum"],
    )
    check(
        "sampled rollout tokens identical to the unbroken run",
        [sa[s]["rollout_tokens"] for s in sorted(sa)]
        == [sb[s]["rollout_tokens"] for s in sorted(sb)],
        hard=False,
    )
    loss_gap = max(abs((sa[s]["loss"] or 0) - (sb[s]["loss"] or 0)) for s in sa)
    check(f"losses identical (max gap {loss_gap:.2e})", loss_gap < 1e-6, hard=False)

    def history(run: Path) -> dict[int, dict]:
        last = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name[11:]))[-1]
        saved = json.loads((last / "trainer_state.json").read_text("utf-8"))
        return {e["step"]: e for e in saved["log_history"] if "grad_norm" in e}

    ga, gb = history(a), history(b)
    norms_a = [ga[s]["grad_norm"] for s in sorted(ga)]
    norms_b = [gb[s]["grad_norm"] for s in sorted(gb)]
    check("the gradient is not zero", min(norms_a) > 0)
    gap = max(abs(x - y) for x, y in zip(norms_a, norms_b, strict=True))
    check(f"gradient norms identical (max gap {gap:.2e})", gap < 1e-6, hard=False)

    from safetensors.torch import load_file

    def final_weights(run: Path):
        last = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name[11:]))[-1]
        return load_file(last / "model.safetensors")

    wa, wb = final_weights(a), final_weights(b)
    wgap = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    check(f"final weights identical (max gap {wgap:.2e})", wgap < 1e-6, hard=False)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--worker", action="store_true")
    ns = ap.parse_args()
    sys.exit(worker(ns) if ns.worker else orchestrate(ns))
