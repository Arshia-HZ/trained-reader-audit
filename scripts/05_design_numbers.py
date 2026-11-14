"""Dev numbers behind the design rules of PREREG sections 4 and 6 (dev only).

    python scripts/05_design_numbers.py --root <shards> \
        --out docs/design_numbers_dev.json

For K = 8 candidate sets: the share of sets in each stratum and in the two answer-swap
populations; for 4, 8 and 16 evaluation draws per problem, the share of problems that
have at least one set of that kind and how many sets they have on average (these set
the power of the stratum-wise hypotheses); and how many candidates are cut for display
or hit the generation limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reader_audit.arms import swap_plan
from reader_audit.cand_sets import draw_set
from reader_audit.pools import ProblemPool, load_pools
from reader_audit.reader_prompts import CANDIDATE_CAP

K = 8
DRAWS = (4, 8, 16)
KINDS = ("c0", "c1", "c2+", "swap_broad", "swap_primary")
MAX_DRAWS = max(DRAWS)


def flags_of(pool: ProblemPool, draw: int) -> dict[str, bool]:
    cset = draw_set(pool, K, draw)
    plan = swap_plan(pool, cset)
    return {
        "c0": cset.stratum == "c0",
        "c1": cset.stratum == "c1",
        "c2+": cset.stratum == "c2+",
        "swap_broad": plan is not None,
        "swap_primary": plan is not None and plan.correct_plurality,
    }


def summarise(pools: list[ProblemPool]) -> dict:
    hits = np.zeros((len(pools), MAX_DRAWS, len(KINDS)), dtype=bool)
    for i, pool in enumerate(pools):
        for d in range(MAX_DRAWS):
            flags = flags_of(pool, d)
            hits[i, d] = [flags[k] for k in KINDS]
    out: dict = {
        "n_problems": len(pools),
        "set_share": {k: float(hits[:, :, j].mean()) for j, k in enumerate(KINDS)},
        "by_draws": {},
    }
    for draws in DRAWS:
        counts = hits[:, :draws, :].sum(axis=1)  # (problems, kinds)
        block = {}
        for j, kind in enumerate(KINDS):
            covered = counts[:, j] >= 1
            block[kind] = {
                "coverage": float(covered.mean()),
                "problems": int(covered.sum()),
                "sets_per_covered_problem": float(
                    counts[covered, j].mean() if covered.any() else 0.0
                ),
            }
        out["by_draws"][str(draws)] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"error: {out} exists; the numbers are recorded once")
        return 1
    pools = load_pools(args.root)
    by_level = {
        "all": list(pools.values()),
        "level_5": [p for p in pools.values() if p.problem.level == 5],
    }
    tokens = np.array(
        [c.completion_tokens for p in pools.values() for c in p.candidates]
    )
    truncated = np.array([c.truncated for p in pools.values() for c in p.candidates])
    over = tokens > CANDIDATE_CAP
    result = {
        "k": K,
        "candidate_cap": CANDIDATE_CAP,
        "sets": {name: summarise(chosen) for name, chosen in by_level.items()},
        "candidates": {
            "n": int(tokens.size),
            "share_generation_truncated": float(truncated.mean()),
            "share_cut_for_display": float(over.mean()),
            "share_complete_but_cut_for_display": float((over & ~truncated).mean()),
            "share_of_cut_that_are_complete": float(
                (over & ~truncated).sum() / max(over.sum(), 1)
            ),
        },
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["candidates"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
