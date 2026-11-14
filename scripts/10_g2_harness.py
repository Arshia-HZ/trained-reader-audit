"""Stage 1 harness check (gate G2): the untrained reader and its controls on AIME 2025
and HMMT Feb 2025 with Qwen3-4B-Instruct-2507.

    python scripts/10_g2_harness.py --out runs/g2_qwen3-4b \
        --model-config configs/models/qwen3-4b-instruct-2507.yaml
    python scripts/10_g2_harness.py --out runs/g2_mock --mock     # rehearsal, no GPU

Resumable: every generation is stored as it is made. Needs the ``vllm``, ``hf`` and
``data`` extras for the real run. The harness problems are read at the revisions pinned
in ``data/sets/harness.json`` and checked against its content hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.models import load_model_config
from channel_audit.runner import RunConfig, config_for_model

from reader_audit import externals
from reader_audit.g2 import format_verdict
from reader_audit.g2_job import run_g2

MOCK_EFFECTS = {
    "reader_full_k4": 0.2,
    "candidate_free_k4": 0.0,
    "answer_only_k4": 0.05,
    "placebo_k4": -0.3,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-config")
    ap.add_argument("--sets", default="data/sets")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n-mock-problems", type=int, default=30)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--cap", type=int, default=4096)
    args = ap.parse_args(argv)

    if args.mock:
        problems = fake_problems(args.n_mock_problems)
        cfg = RunConfig(model_id="mock/g2", max_tokens=512, context_limit=100_000)
        backend = MockBackend(MockWorld.from_problems(problems, 0.5), MOCK_EFFECTS)
        clipper = None
    else:
        if not args.model_config:
            print("error: --model-config is required unless --mock")
            return 1
        from reader_audit.clipper import load_hf_clipper

        problems = externals.load_harness(args.sets, externals.load_matharena)
        model = load_model_config(args.model_config)
        cfg = config_for_model(model, replicates=1)
        backend = None  # built on first use, so a finished job never loads the model
        clipper = load_hf_clipper(model.model_id, model.revision)

    report = run_g2(
        Path(args.out),
        problems,
        cfg,
        backend=backend,
        clipper=clipper,
        samples=args.samples,
        k=args.k,
        n_draws=args.n_draws,
        cap=None if args.mock else args.cap,
    )
    print(format_verdict(report["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
