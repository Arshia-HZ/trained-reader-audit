"""GRPO training of the reader (or the solver baseline) on the training pools.

    python scripts/20_train_grpo.py --arm reader --pools-root runs/train_pools \
        --out runs/grpo_reader --steps 20 --save-every 10

Needs the training environment (torch, trl, vLLM). It resumes from the newest
checkpoint in ``--out`` by itself. With ``--upload-repo`` every checkpoint is copied
to that private model repo (token from ``HF_TOKEN``): the whole state to
``<run>/latest``, the weights alone at ``--milestones``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from channel_audit.models import load_model_config

from reader_audit.clipper import load_hf_clipper
from reader_audit.grpo_plan import ARMS, GrpoPlan, build_rows
from reader_audit.pools import load_pools


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-config", default="configs/models/qwen3-1.7b-reader.yaml")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument(
        "--seed",
        type=int,
        default=20261115,
        help="seed 1 is 20261115, seed 2 20261116, seed 3 20261117 (PREREG R17)",
    )
    ap.add_argument("--expect-pools", type=int, help="refuse to run on another number")
    ap.add_argument("--prompts-per-step", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--vllm-memory", type=float, default=0.3)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument(
        "--keep-flat", action="store_true", help="do not skip zero-advantage rows"
    )
    ap.add_argument(
        "--policy-dtype", default="float32", choices=["float32", "bfloat16"]
    )
    ap.add_argument(
        "--no-freeze-check", action="store_true", help="for tests only: skip the freeze"
    )
    ap.add_argument("--upload-repo")
    ap.add_argument("--milestones", type=int, nargs="*", default=[100, 200, 300])
    args = ap.parse_args(argv)

    import torch

    from reader_audit import training_freeze
    from reader_audit.grpo_run import train
    from reader_audit.grpo_upload import CheckpointUploader

    frozen = None
    if not args.no_freeze_check:
        frozen = training_freeze.verify()["combined_sha256"]  # raises if a file changed
        print(f"training freeze verified: {frozen}", flush=True)
    plan = GrpoPlan(
        arm=args.arm,
        steps=args.steps,
        seed=args.seed,
        prompts_per_step=args.prompts_per_step,
        group_size=args.group_size,
        per_device_batch=args.per_device_batch,
        save_every=args.save_every,
    )
    plan.check()
    model = load_model_config(args.model_config)
    pools = load_pools(args.pools_root)
    if args.expect_pools is not None and len(pools) != args.expect_pools:
        raise SystemExit(f"{len(pools)} pools found, {args.expect_pools} expected")
    clipper = load_hf_clipper(model.model_id, model.revision)
    rows = build_rows(plan, pools, truncate=clipper, n_draws=args.n_draws)
    print(f"{len(pools)} problems, {len(rows)} prompts for {args.steps} steps")

    uploader = None
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        uploader = CheckpointUploader(
            api,
            args.upload_repo,
            Path(args.out).name,
            args.milestones,
            save_every=args.save_every,
        )
    if uploader is not None:
        from reader_audit.grpo_upload import fetch_latest

        step = fetch_latest(
            args.upload_repo,
            Path(args.out).name,
            args.out,
            token=os.environ["HF_TOKEN"],
        )
        if step is not None:
            print(f"fetched the checkpoint of step {step} from the hub", flush=True)
    train(
        plan,
        rows,
        args.out,
        model_id=model.model_id,
        revision=model.revision,
        uploader=uploader,
        run_record={"argv": sys.argv[1:], "freeze_sha256": frozen},
        policy_dtype=getattr(torch, args.policy_dtype),
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        skip_flat=not args.keep_flat,
        use_vllm=not args.no_vllm,
        vllm_gpu_memory_utilization=args.vllm_memory,
    )
    return 0


if __name__ == "__main__":
    code = main()
    # the colocated vLLM engine can keep the process (and the GPU) alive after the end
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
