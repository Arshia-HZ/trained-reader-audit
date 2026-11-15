"""Stage 1: what the untrained dev reader says about the design (dev only).

    python scripts/13_dev_reader_analysis.py --reader-root runs/dev_reader \
        --pools-root runs/dev_pools/confirm/qwen3-1.7b --out docs/dev_reader.json

For 4, 8 and 16 draws per problem (the first d of the 16 seeded sets) it measures, on
the level-5 dev problems, the standard deviations the sizing rule of PREREG section 6.3
needs and the stratum coverage of the primary set's 972 problems:

- H1a: the untrained reader's per-problem accuracy minus self-consistency at the same
  generated tokens (the matched k*, exact on the 18-sample dev pools); H1b has no
  measured SD yet and borrows H1a's;
- H2: per problem with a c1 set, the reader's accuracy with candidates minus its
  candidate-free accuracy on the same sets; the difference-in-differences of a trained
  and an untrained reader is taken to have sqrt(2) times that SD (the two readers
  independent and equally variable: an assumption, stated, not measured);
- H4a: per problem with a c2+ set, the reader's accuracy minus the plain-vote credit of
  the same sets.

It also reports the reader's mean generated tokens, which replaces the planning
assumption that a read costs about one sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from reader_audit import power
from reader_audit import reader_eval as ev
from reader_audit.cand_sets import draw_set, members, vote_credit
from reader_audit.pools import load_pools
from reader_audit.subsets import group_answers, interpolate, sc_curve

K = 8
DRAWS = (4, 8, 16)
PRIMARY_PROBLEMS = 972
READER, FREE = "reader_full", "candidate_free"


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def sd(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")


def summarise(outcomes, pools, curves, draws: int, k: int = K) -> tuple[dict, dict]:
    by = defaultdict(lambda: defaultdict(list))  # (pid, draw) -> arm -> [outcome]
    for o in outcomes:
        if o.k == k and o.draw < draws:
            by[(o.problem_id, o.draw)][o.arm].append(o)
    pids = sorted({pid for pid, _ in by})
    h1a, h2, h4a, k_stars, reader_gen = [], [], [], [], []
    n_h1a_missing = 0
    stratum_counts = defaultdict(int)
    for pid in pids:
        pool = pools[pid]
        read_acc, free_acc, vote, cost_sets = [], [], [], []
        c1_diff, c2_diff = [], []
        for draw in range(draws):
            arms = by.get((pid, draw))
            if not arms or READER not in arms:
                continue
            cset = draw_set(pool, k, draw)
            reader_vals = [o.value for o in arms[READER]]
            gen = np.mean([o.completion_tokens for o in arms[READER]])
            reader_gen.append(gen)
            cand_gen = sum(c.completion_tokens for c in members(pool, cset))
            cost_sets.append(cand_gen + gen)
            v = vote_credit(pool, cset)
            read_acc.append(np.mean(reader_vals))
            vote.append(v)
            stratum_counts[cset.stratum] += 1
            if FREE in arms:
                free_acc.append(np.mean([o.value for o in arms[FREE]]))
                if cset.stratum == "c1":
                    c1_diff.append(read_acc[-1] - free_acc[-1])
            if cset.stratum == "c2+":
                c2_diff.append(read_acc[-1] - v)
        if not read_acc:
            continue
        sample_gen = np.mean([c.completion_tokens for c in pool.candidates])
        k_star = float(np.mean(cost_sets) / sample_gen)
        k_stars.append(k_star)
        if k_star <= len(curves[pid]):
            h1a.append(mean(read_acc) - interpolate(curves[pid], max(k_star, 1.0)))
        else:
            n_h1a_missing += 1
        if c1_diff:
            h2.append(mean(c1_diff))
        if c2_diff:
            h4a.append(mean(c2_diff))
    n = len(pids)
    out = {
        "draws": draws,
        "problems": n,
        "sets": dict(stratum_counts),
        "reader_generated_tokens_mean": mean(reader_gen),
        "k_star_mean": mean(k_stars),
        "H1a": {
            "n": len(h1a),
            "not_estimable": n_h1a_missing,
            "mean": mean(h1a),
            "sd": sd(h1a),
        },
        "H2_single_arm": {
            "n": len(h2),
            "coverage": len(h2) / n,
            "mean": mean(h2),
            "sd": sd(h2),
        },
        "H4a": {
            "n": len(h4a),
            "coverage": len(h4a) / n,
            "mean": mean(h4a),
            "sd": sd(h4a),
        },
    }
    measured = {
        "H1a": (PRIMARY_PROBLEMS, out["H1a"]["sd"]),
        "H1b": (PRIMARY_PROBLEMS, out["H1a"]["sd"]),
        "H2": (
            PRIMARY_PROBLEMS * out["H2_single_arm"]["coverage"],
            out["H2_single_arm"]["sd"] * np.sqrt(2),
        ),
        "H4a": (PRIMARY_PROBLEMS * out["H4a"]["coverage"], out["H4a"]["sd"]),
    }
    return out, measured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reader-root", required=True)
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    outcomes = []
    for shard in sorted(Path(args.reader_root).glob("shard_*")):
        if (shard / "DONE.json").is_file():
            outcomes += ev.load_outcomes(shard)
    done = {o.problem_id for o in outcomes}
    pools = {p: v for p, v in load_pools(args.pools_root).items() if p in done}
    print(f"{len(outcomes)} outcomes over {len(pools)} problems", flush=True)
    curves = {
        pid: sc_curve(
            group_answers(
                [c.answer for c in pool.candidates],
                [c.correct for c in pool.candidates],
            )
        )
        for pid, pool in pools.items()
    }
    report: dict = {"n_problems": len(pools), "k": K, "by_draws": {}}
    measured_all = {}
    for d in DRAWS:
        out, measured = summarise(outcomes, pools, curves, d)
        report["by_draws"][str(d)] = out
        measured_all[d] = measured
        print(json.dumps(out, indent=1), flush=True)
    rule = power.draws_by_rule(measured_all)
    report["sizing_rule"] = {
        "draws": rule["draws"],
        "underpowered": rule["underpowered"],
        "power": {str(d): v for d, v in rule["power"].items()},
    }
    print(json.dumps(report["sizing_rule"], indent=1))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
