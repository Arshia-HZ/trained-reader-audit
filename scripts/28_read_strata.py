"""Exploratory, dev only (PREREG R26): E4 and the headroom figures, from the E3 pilot.

    python scripts/28_read_strata.py \
        --pools-root runs/dev_pools/confirm/qwen3-1.7b --pilot runs/think_pilot \
        --out docs/dev_read_strata.json --md docs/dev_read_strata.md

E4: each thinking read minus the mean fresh thinking solve of its problem, by the
stratum (c0, c1, c2+) of the 8 candidates it was shown: credit at the 16,384 cap,
thinking tokens, epistemic-word density and the share of answers equal to a shown
answer. Headroom: ESC (2, 2) then one thinking solve and its three oracles, under both
cost rules, and thinking pass@m. Bootstrap over problems. No test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from reader_audit import allocation as al
from reader_audit import read_strata as rs
from reader_audit import think_frontier as tf
from reader_audit.cand_sets import STRATA
from reader_audit.pools import load_pools

K, DRAWS, LEVEL, SOLVES, CAP = 8, 16, 5, 4, 30000
LABELS = {
    "credit": ("Credit at 16,384", 100, ".1f"),
    "think_tokens": ("Thinking tokens", 1, ".0f"),
    "density": ("Epistemic words per 1,000", 1, ".2f"),
    "copy": ("Answer equals a shown one (%)", 100, ".1f"),
}


def load_records(root: Path) -> tuple[dict, list]:
    """(pid -> fresh records, read records) from the finished shards of the pilot."""
    fresh, reads = defaultdict(list), []
    for shard in sorted(root.glob("shard_*")):
        if not (shard / "DONE.json").is_file():
            continue
        for line in (shard / "generations.jsonl").open(encoding="utf-8"):
            r = rs.ThinkingRecord.from_row(json.loads(line))
            if r.arm == "candidate_free":
                fresh[r.problem_id].append(r)
            else:
                reads.append(r)
    return dict(fresh), reads


def as_solve(r: rs.ThinkingRecord) -> tf.Solve:
    return tf.Solve(r.correct, r.answer, r.prompt_tokens, r.completion_tokens)


def markdown(report: dict) -> str:
    lines = [
        "# Thinking read against fresh thinking solve, by stratum (exploratory, E4)",
        "",
        "Level-5 dev; each read paired with the mean of its problem's 4 fresh solves; "
        "means over problems; read minus fresh with a 95% bootstrap over problems.",
        "",
        "| Stratum | Reads | Problems | Quantity | Read | Fresh "
        "| Read − fresh [95% CI] |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, cell in report["strata"].items():
        for q, (label, scale, fmt) in LABELS.items():
            v = cell[q]
            lines.append(
                f"| {name} | {cell['n_reads']} | {cell['n_problems']} | {label} | "
                f"{scale * v['read']:{fmt}} | {scale * v['fresh']:{fmt}} | "
                f"{scale * v['diff']:+{fmt}} [{scale * v['ci'][0]:+{fmt}}, "
                f"{scale * v['ci'][1]:+{fmt}}] |"
            )
    lines += [
        "",
        "## Headroom",
        "",
        "| Policy | Rule | Cost | Credit |",
        "| --- | --- | --- | --- |",
    ]
    for rule, points in report["headroom"].items():
        for name, (cost, credit) in points.items():
            lines.append(f"| {name} | {rule} | {cost:.2f} | {100 * credit:.1f} |")
    passes = ", ".join(f"pass@{m} {100 * v:.1f}" for m, v in report["pass_at"].items())
    lines += ["", f"Thinking solves: {passes}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args(argv)

    pools = {
        p: v for p, v in load_pools(args.pools_root).items() if v.problem.level == LEVEL
    }
    fresh, reads = load_records(Path(args.pilot))
    fresh = {p: v for p, v in fresh.items() if p in pools and len(v) == SOLVES}
    reads = [r for r in reads if r.problem_id in fresh]
    rows = rs.read_rows(pools, fresh, reads, K)
    print(
        f"{len(rows)} reads on {len({r['problem_id'] for r in rows})} problems",
        flush=True,
    )
    report: dict = {
        "reads": len(rows),
        "strata_counts": {s: sum(r["stratum"] == s for r in rows) for s in STRATA},
        "strata": rs.strata_table(rows, args.n_boot, args.seed),
        "headroom": {},
    }
    pids = sorted(fresh)
    for rule in al.RULES:
        cells = [
            tf.cascade_oracles(
                pools[p], K, DRAWS, [as_solve(r) for r in fresh[p]], CAP, rule
            )
            for p in pids
        ]
        report["headroom"][rule] = {
            name: (
                float(np.mean([c[name][0] for c in cells])),
                float(np.mean([c[name][1] for c in cells])),
            )
            for name in cells[0]
        }
    report["pass_at"] = {
        m: float(
            np.mean([tf.pass_at([as_solve(r) for r in fresh[p]], m) for p in pids])
        )
        for m in (1, 2, 4)
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(report), encoding="utf-8")
        print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
