"""Exploratory, dev only (PREREG R20, E1): strata, agreement and allocation of reads.

    python scripts/24_dev_allocation.py --pools-root runs/dev_pools/confirm/qwen3-1.7b \
        --run untrained=runs/dev_untrained/dev_reader_qwen3-1.7b \
        --run step100=runs/dev_trained/dev_trained_reader_s1_step0100_reader_full,\
runs/dev_trained/dev_trained_reader_s1_step0100_candidate_free \
        --out docs/dev_allocation.json --md docs/dev_allocation.md

Each ``--run label=dir[,dir...]`` names the shard folders of one reader (its
``reader_full`` reads and, where run, its ``candidate_free`` solves of the same sets).
For every run it reports, on the level-5 dev problems at K = 8 and 16 draws: raw credit
per arm and stratum with the plain vote of the same sets; how often a read gives the
plurality answer or some shown answer; reader minus vote by the size of the plurality;
and the policies of ``allocation.policies`` (always read, the unanimity gate of H6,
plurality thresholds, ESC first and a read only on disagreement) against plain SC at
their own cost and against the pooled SC + ESC frontier, under both cost rules of D3.
No generation; nothing here is a test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from channel_audit.schema import Generation, Graded
from channel_audit.store import JsonlStore

from reader_audit import allocation as al
from reader_audit.pools import load_pools
from reader_audit.reader_run import PlanRow

K, DRAWS, LEVEL = 8, 16, 5
READER, FREE = "reader_full", "candidate_free"


def load_reads(dirs: list[Path]) -> dict:
    """(problem_id, arm) -> draw -> [Read], from the finished shards of the folders."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for shard in sorted(d.glob("shard_*")):
            if not (shard / "DONE.json").is_file():
                continue
            plan = JsonlStore(shard / "plan.jsonl", PlanRow, readonly=True)
            gens = JsonlStore(shard / "generations.jsonl", Generation, readonly=True)
            graded = JsonlStore(shard / "graded.jsonl", Graded, readonly=True)
            for row in plan:
                if row.k != K or row.draw >= DRAWS or row.arm not in (READER, FREE):
                    continue
                if row.gen_key not in graded:
                    continue
                g, gen = graded.get(row.gen_key), gens.get(row.gen_key)
                out[(row.problem_id, row.arm)][row.draw].append(
                    al.Read(
                        correct=bool(g.correct),
                        answer=g.extracted,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                    )
                )
    return out


def pct(x: float) -> str:
    return f"{100 * x:+.1f}"


def ci(d: dict) -> str:
    return f"{pct(d['mean'])} [{pct(d['ci'][0])}, {pct(d['ci'][1])}]"


def markdown(report: dict) -> str:
    lines = [
        "# Dev allocation analysis (exploratory, PREREG R20 E1)",
        "",
        "Level-5 dev problems, K = 8, 16 draws; points are percentage points,",
        "intervals 95% bootstrap over problems. Descriptive; dev only; no test.",
    ]
    for label, run in report["runs"].items():
        lines += [
            "",
            f"## {label}",
            "",
            "| Stratum | Sets | Read | Vote | Free | Read − vote "
            "| Read − free | Follows plurality | Equals a shown answer |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s, row in run["strata"].items():
            free = "" if "free" not in row else f"{100 * row['free']['mean']:.1f}"
            rmf = "" if "read_minus_free" not in row else ci(row["read_minus_free"])
            lines.append(
                f"| {s} | {row['sets']} | {100 * row['read']['mean']:.1f} | "
                f"{100 * row['vote']['mean']:.1f} | {free} | "
                f"{ci(row['read_minus_vote'])} | {rmf} | "
                f"{100 * row['follows_plurality']['mean']:.0f}% | "
                f"{100 * row['equals_a_shown_answer']['mean']:.0f}% |"
            )
        lines += [
            "",
            "| Plurality of 8 | Share of sets | Vote | Read | Read − vote |",
            "| --- | --- | --- | --- | --- |",
        ]
        for b, row in run["agreement"].items():
            if row["read"]["n"]:
                lines.append(
                    f"| {b} | {100 * row['share_of_sets']:.1f}% | "
                    f"{100 * row['vote']['mean']:.1f} | "
                    f"{100 * row['read']['mean']:.1f} | {ci(row['read_minus_vote'])} |"
                )
        for rule, table in run["policies"].items():
            lines += [
                "",
                f"Policies, {rule} rule (cost in pool samples):",
                "",
                "| Policy | Cost | Credit | Reads | − SC at own cost (n) "
                "| − SC+ESC frontier |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, p in table.items():
                fr = p["minus_frontier"]
                lines.append(
                    f"| {name} | {p['cost']:.2f} | {100 * p['credit']:.1f} | "
                    f"{100 * p['read_share']:.0f}% | {pct(p['minus_sc']['mean'])} "
                    f"({p['minus_sc']['n']}) | "
                    f"{ci(fr) if fr['estimable'] else 'outside the frontier'} |"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=dir[,dir...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--level", type=int, default=LEVEL)
    args = ap.parse_args(argv)

    pools = {
        pid: pool
        for pid, pool in load_pools(args.pools_root).items()
        if pool.problem.level == args.level
    }
    print(f"{len(pools)} pools at level {args.level}", flush=True)
    baselines = {pid: al.baseline_points(pool) for pid, pool in pools.items()}
    report: dict = {
        "settings": {
            "k": K,
            "draws": DRAWS,
            "level": args.level,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "esc": al.ESC_SETTINGS,
        },
        "runs": {},
    }
    for spec in args.run:
        label, dirs = spec.split("=", 1)
        reads = load_reads([Path(d) for d in dirs.split(",")])
        views_by_problem = {}
        for pid, pool in pools.items():
            mine = reads.get((pid, READER))
            if mine:
                frees = reads.get((pid, FREE))
                views_by_problem[pid] = al.build_views(pool, K, mine, frees)
        views = [v for vs in views_by_problem.values() for v in vs]
        print(
            f"{label}: {len(views_by_problem)} problems, {len(views)} sets", flush=True
        )
        report["runs"][label] = {
            "problems": len(views_by_problem),
            "sets": len(views),
            "strata": al.stratum_table(views, args.n_boot, args.seed),
            "agreement": al.agreement_table(views, args.n_boot, args.seed),
            "policies": al.policy_table(
                views_by_problem, pools, baselines, args.n_boot, args.seed
            ),
        }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(report), encoding="utf-8")
        print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
