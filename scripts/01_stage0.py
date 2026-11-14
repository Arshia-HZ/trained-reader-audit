"""Stage 0: break-even and headroom on the dev pools (PREREG section 7).

    python scripts/01_stage0.py --root <shards> --out-dir results/stage0 \
        --expect-problems 600 [--compare docs/stage0/stage0.json]

``--root`` holds the audit's ``shard_*`` folders (``confirm/qwen3-1.7b``); each shard is
checked against its done marker. With ``--compare`` the means are checked against the
record of the first, script-less run (2026-11-14). They agree to within 0.001: answer
groups are built in a fixed canonical order here and in file order there, and answer
equivalence is not exactly transitive, so a few groups differ. The 95% intervals come
from a different bootstrap stream and are not compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reader_audit.pools import load_pools, pool_sizes
from reader_audit.stage0 import analyze

TOLERANCE = 1e-3


def flatten(node, prefix=""):
    """(path, mean) pairs of every {"mean": ...} leaf, for comparing two results."""
    if isinstance(node, dict):
        if "mean" in node:
            yield prefix, node["mean"]
        else:
            for key, value in node.items():
                yield from flatten(value, f"{prefix}/{key}")


def compare(new: dict, ref: dict, tol: float = TOLERANCE) -> list[str]:
    ref_means = dict(flatten(ref["results"]))
    new_means = dict(flatten(new["results"]))
    problems = [f"missing in new: {p}" for p in ref_means if p not in new_means]
    for path, value in ref_means.items():
        if path in new_means and abs(new_means[path] - value) > tol:
            problems.append(f"{path}: {new_means[path]} vs {value}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-problems", type=int, required=True)
    ap.add_argument("--compare")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if (out / "stage0.json").exists():
        print(f"error: {out / 'stage0.json'} exists; use a new directory")
        return 1
    pools = load_pools(args.root)
    if len(pools) != args.expect_problems:
        print(f"error: {len(pools)} problems, expected {args.expect_problems}")
        return 1
    print(f"{len(pools)} problems; pool sizes {sorted(pool_sizes(pools.values()))}")
    result = analyze(pools)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage0.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["g1_level_5"], indent=2))
    if args.compare:
        ref = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        problems = compare(result, ref)
        print(f"compared with {args.compare}: {len(problems)} differences")
        for line in problems[:20]:
            print("  ", line)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
