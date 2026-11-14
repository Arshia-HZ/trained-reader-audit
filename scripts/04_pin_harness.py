"""Pin the harness set (AIME 2025, HMMT Feb 2025) to Hub revisions and hashes.

    python scripts/04_pin_harness.py --out data/sets

Needs the ``data`` extra and the network. Writes ``harness.json`` once; it refuses to
change an existing one. Later runs read the problems at the pinned revisions and stop
if their content differs.
"""

from __future__ import annotations

import argparse
import json
import sys

from huggingface_hub import HfApi

from reader_audit import externals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/sets")
    args = ap.parse_args()
    api = HfApi()
    manifest = externals.pin_harness(
        args.out,
        lambda repo: api.dataset_info(repo).sha,
        externals.load_matharena,
    )
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
