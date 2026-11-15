"""Stage 1: the untrained reader on the dev pools, in resumable problem shards.

    python scripts/12_dev_reader.py --pools-root <dev shards> --out runs/dev_reader \
        --model-config configs/models/qwen3-1.7b-reader.yaml --max-shards 1
    python scripts/12_dev_reader.py --download --out runs/dev_reader ... --upload

The reader (Qwen3-1.7B, untrained, the draft prompt) reads K = 8 of a dev problem's 18
samples, on the level-5 dev problems, for 16 seeded sets per problem and 2 replicates
per set, with its own answer and with no candidates at all (the arms H1a, H2 and H4a
need). ``--max-shards 1`` runs one shard only, to measure throughput before the rest is
paid for. Every finished shard is marked, and with ``--upload`` copied to the private
dataset (the token is read from ``HF_TOKEN`` and never printed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from channel_audit.models import load_model_config
from channel_audit.runner import config_for_model

from reader_audit import jobs
from reader_audit import reader_prompts as rp
from reader_audit.dev_job import DEV_ARMS, run_dev_reader
from reader_audit.pools import load_pools

POOL_REPO = "Arshia-HZ/mas-channel-audit-raw"
POOL_PATH = "confirm/qwen3-1.7b"
RAW_REPO = "Arshia-HZ/trained-reader-audit-raw"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--pools-root")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=26)
    ap.add_argument("--max-shards", type=int)
    ap.add_argument("--arms", nargs="+", default=list(DEV_ARMS))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--slug", default="dev_reader_qwen3-1.7b")
    args = ap.parse_args(argv)

    root = args.pools_root
    if args.download:
        from huggingface_hub import snapshot_download

        local = Path(args.out).parent / "dev_pools"
        snapshot_download(
            POOL_REPO,
            repo_type="dataset",
            allow_patterns=[f"{POOL_PATH}/*"],
            local_dir=str(local),
        )
        root = str(local / POOL_PATH)
    if not root:
        print("error: give --pools-root or --download")
        return 1

    pools = {
        pid: pool
        for pid, pool in load_pools(root).items()
        if pool.problem.level == args.level
    }
    print(f"{len(pools)} dev problems at level {args.level}", flush=True)

    from reader_audit.clipper import load_hf_clipper

    model = load_model_config(args.model_config)
    cfg = config_for_model(model, replicates=args.replicates)
    clipper = load_hf_clipper(model.model_id, model.revision)

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

    report = run_dev_reader(
        args.out,
        pools,
        cfg,
        clipper=clipper,
        arms=args.arms,
        k=args.k,
        n_draws=args.n_draws,
        replicates=args.replicates,
        shard_size=args.shard_size,
        cap=rp.CANDIDATE_CAP,
        max_shards=args.max_shards,
        on_shard_done=on_done,
    )
    for name, s in report["shards"].items():
        rate = s["completion_tokens"] / max(s["seconds"], 1e-9)
        print(
            f"{name}: {s['generations']} generations, {s['new_generations']} new, "
            f"{s['seconds']:.0f} s, {s['prompt_tokens']} prompt and "
            f"{s['completion_tokens']} completion tokens ({rate:.0f} completion tok/s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
