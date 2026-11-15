"""Candidate pools (16 samples each) of Qwen3-1.7B for the training problems, sharded.

    python scripts/15_train_pools.py --out runs/train_pools \
        --model-config configs/models/qwen3-1.7b-pool.yaml --n-problems 400 --upload

Takes the first ``--n-problems`` training ids of the seeded order (a random sample of
the set; the full set is 7,587) and works through them in shards, each marked when
finished and, with ``--upload``, copied to the private dataset (token from
``HF_TOKEN``). The pool model, prompt and settings are those of the dev pools. No study
set is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from channel_audit.models import load_model_config
from channel_audit.runner import config_for_model

from reader_audit import jobs
from reader_audit.pool_job import run_pool_shards
from reader_audit.train_set import fetch_sources, train_order

RAW_REPO = "Arshia-HZ/trained-reader-audit-raw"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--n-problems", type=int, default=400)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--max-shards", type=int)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--slug", default="train_pools_qwen3-1.7b")
    args = ap.parse_args(argv)

    ids = json.loads((Path(args.train) / "ids.json").read_text("utf-8"))
    order = train_order(ids)[args.offset : args.offset + args.n_problems]
    math_train, _, scaler, _ = fetch_sources()
    by_id = {p.problem_id: p for p in math_train + scaler}
    problems = [by_id[i].to_problem() for i in order]
    print(f"{len(problems)} training problems (offset {args.offset})", flush=True)

    model = load_model_config(args.model_config)
    cfg = config_for_model(model, replicates=1)

    on_done = None
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

        def on_done(shard: Path, index: int) -> None:
            try:
                jobs.upload_folder(api, shard, RAW_REPO, f"{args.slug}/{shard.name}")
                print(f"uploaded {shard.name}", flush=True)
            except Exception as exc:  # the data stays on disk; the job goes on
                print(
                    f"upload of {shard.name} failed: {type(exc).__name__}", flush=True
                )

    report = run_pool_shards(
        args.out,
        problems,
        cfg,
        samples=args.samples,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        on_shard_done=on_done,
    )
    for name, s in report["shards"].items():
        rate = s["completion_tokens"] / max(s["seconds"], 1e-9)
        print(
            f"{name}: {s['problems']} problems, {s['generations']} generations, "
            f"{s['seconds']:.0f} s, {s['completion_tokens']} completion tokens "
            f"({rate:.0f} tok/s), {s['truncated']} truncated"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
