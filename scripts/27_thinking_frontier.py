"""Exploratory, dev only (PREREG R24, E3): the thinking pilot read on one frontier.

    python scripts/27_thinking_frontier.py \
        --pools-root runs/dev_pools/confirm/qwen3-1.7b --pilot runs/think_pilot \
        --reads untrained=runs/dev_untrained/dev_reader_qwen3-1.7b \
        --reads step100=runs/dev_trained/dev_trained_reader_s1_step0100_reader_full \
        --out docs/dev_thinking_frontier.json --md docs/dev_thinking_frontier.md

It computes, for the problems whose four fresh thinking solves are all present, the
policies of PREREG R24: the comparison envelope (plain SC, ESC, one thinking solve at
four caps, the thinking vote of 2 and 4); the cascades (ESC with window 2 and cap 2, 4
or 8, then one thinking solve at the 16,384 or 30,000 cap); the reference cascades with
the non-thinking readers and with the thinking read; and the thinking read minus the
fresh thinking solve on the sets the read covers. Every policy is read as pooled credit
minus the envelope at its pooled cost, under both cost rules, with a bootstrap over
problems. No test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from reader_audit import allocation as al
from reader_audit import think_frontier as tf
from reader_audit.pools import load_pools
from reader_audit.subsets import group_answers

K, DRAWS, LEVEL = 8, 16, 5
SOLVES = 4


def load_pilot(root: Path) -> tuple[dict, dict, dict]:
    """(pid -> [fresh Solve]), (pid -> draw -> read Solve), per-arm GPU summary."""
    fresh, reads = defaultdict(list), defaultdict(dict)
    gpu = defaultdict(lambda: {"seconds": 0.0, "completion_tokens": 0, "n": 0})
    for shard in sorted(root.glob("shard_*")):
        done = shard / "DONE.json"
        if not done.is_file():
            continue
        for arm, s in json.loads(done.read_text(encoding="utf-8"))["arms"].items():
            for key in ("seconds", "completion_tokens", "n"):
                gpu[arm][key] += s[key]
        for line in (shard / "generations.jsonl").open(encoding="utf-8"):
            g = json.loads(line)
            solve = tf.Solve(
                bool(g["correct"]),
                g["extracted"],
                g["prompt_tokens"],
                g["completion_tokens"],
            )
            if g["arm"] == "candidate_free":
                fresh[g["problem_id"]].append(solve)
            else:
                reads[g["problem_id"]][g["draw"]] = solve
    return fresh, reads, dict(gpu)


def load_reads_fn():
    path = Path(__file__).resolve().parent / "24_dev_allocation.py"
    spec = importlib.util.spec_from_file_location("dev_allocation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reads


def pct(x: float) -> str:
    return f"{100 * x:+.1f}"


def markdown(report: dict) -> str:
    lines = [
        "# Thinking pilot on one frontier (exploratory, PREREG R24 E3)",
        "",
        f"{report['problems']} level-5 dev problems with all {SOLVES} fresh thinking "
        "solves; cost in samples of the non-thinking pool; credit in percent; "
        "intervals 95% bootstrap over problems. Descriptive; dev only; no test.",
    ]
    for rule in al.RULES:
        lines += [
            "",
            f"## {rule} rule",
            "",
            "| Method (envelope) | Cost | Credit |",
            "| --- | --- | --- |",
        ]
        for name, (c, a) in report["envelope"][rule].items():
            if not name.startswith("sc_") or name in (
                "sc_1",
                "sc_4",
                "sc_8",
                "sc_12",
                "sc_18",
            ):
                lines.append(f"| {name} | {c:.2f} | {100 * a:.1f} |")
        lines += [
            "",
            "| Policy | Cost | Credit | Escalated | − envelope [95% CI] |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, p in report["policies"][rule].items():
            diff = (
                f"{pct(p['minus_envelope'])} [{pct(p['ci'][0])}, {pct(p['ci'][1])}]"
                if p["estimable"]
                else "outside the envelope"
            )
            esc = "" if p.get("escalated") is None else f"{100 * p['escalated']:.0f}%"
            cost, credit = f"{p['cost']:.2f}", f"{100 * p['credit']:.1f}"
            lines.append(f"| {name} | {cost} | {credit} | {esc} | {diff} |")
        lines += ["", "Thinking points minus the non-thinking envelope (SC + ESC):", ""]
        for name, p in report["thinking_vs_nonthinking"][rule].items():
            diff = (
                f"{pct(p['minus_envelope'])} [{pct(p['ci'][0])}, {pct(p['ci'][1])}]"
                if p["estimable"]
                else "outside the envelope"
            )
            lines.append(f"- {name}: cost {p['cost']:.2f}, {diff}")
    c = report["read_minus_fresh"]
    lines += [
        "",
        "## Thinking read minus fresh thinking solve (sets the read covers)",
        "",
    ]
    for name, d in c.items():
        lines.append(
            f"- {name}: {pct(d['mean'])} [{pct(d['ci'][0])}, {pct(d['ci'][1])}] "
            f"over {d['n']} problems"
        )
    lines += ["", "## GPU", ""]
    for arm, g in report["gpu"].items():
        lines.append(
            f"- {arm}: {g['n']} generations, {g['seconds'] / 3600:.2f} h, "
            f"{g['completion_tokens']} tokens, "
            f"{g['completion_tokens'] / max(g['seconds'], 1):.0f} tokens/s"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--reads", action="append", default=[], help="label=dir[,dir...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args(argv)

    pools = {
        p: v for p, v in load_pools(args.pools_root).items() if v.problem.level == LEVEL
    }
    fresh, reads, gpu = load_pilot(Path(args.pilot))
    pids = sorted(p for p in pools if len(fresh.get(p, [])) == SOLVES)
    print(f"{len(pids)} of {len(pools)} problems have all {SOLVES} solves", flush=True)
    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(pids), (args.n_boot, len(pids)))
    baselines = {p: al.baseline_points(pools[p]) for p in pids}
    units = {p: al.unit_costs(pools[p]) for p in pids}
    load_reads = load_reads_fn()
    stored = {}
    for spec in args.reads:
        label, dirs = spec.split("=", 1)
        stored[label] = load_reads([Path(d) for d in dirs.split(",")])

    report: dict = {
        "problems": len(pids),
        "envelope": {},
        "policies": {},
        "thinking_vs_nonthinking": {},
        "gpu": gpu,
    }
    for rule in al.RULES:
        nonthink = {m: {p: baselines[p][m] for p in pids} for m in baselines[pids[0]]}
        thinking = {}
        for cap in tf.THINK_CAPS:
            thinking[f"think_1_cap{cap}"] = {
                p: tf.think_point(fresh[p], cap, units[p], rule) for p in pids
            }
        for m in tf.VOTE_SIZES:
            thinking[f"think_vote_{m}"] = {
                p: tf.think_vote(fresh[p], m, pools[p].problem.gold, units[p], rule)
                for p in pids
            }
        envelope = {**nonthink, **thinking}
        report["envelope"][rule] = {
            m: (
                float(np.mean([v[p][0] for p in pids])),
                float(np.mean([v[p][1] for p in pids])),
            )
            for m, v in envelope.items()
        }
        report["thinking_vs_nonthinking"][rule] = {
            m: tf.minus_envelope(v, nonthink, pids, boots) for m, v in thinking.items()
        }
        policies = {}
        for esc_cap in tf.CASCADE_ESC_CAPS:
            for cap in tf.CASCADE_THINK_CAPS:
                rows = {
                    p: tf.think_cascade(
                        pools[p],
                        K,
                        DRAWS,
                        esc_cap,
                        tf.escalation_by_solve(fresh[p], cap, units[p]),
                        rule,
                    )
                    for p in pids
                }
                name = f"esc_w2_m{esc_cap}_then_think_cap{cap}"
                policies[name] = {
                    **tf.minus_envelope(
                        {p: r[:2] for p, r in rows.items()}, envelope, pids, boots
                    ),
                    "escalated": float(np.mean([r[2] for r in rows.values()])),
                }
        # the thinking read on escalation, from the sets arm B covers
        rows = {}
        for p in pids:
            mine = list(reads.get(p, {}).values())
            if mine:
                credit = float(np.mean([tf.at_cap(r, 16384)[0] for r in mine]))
                costs = [tf.solve_cost(r, 16384, units[p]) for r in mine]
                esc = (
                    credit,
                    {r: float(np.mean([c[r] for c in costs])) for r in al.RULES},
                )
            else:
                esc = (0.0, {r: 0.0 for r in al.RULES})  # never used: nothing escalates
            rows[p] = tf.think_cascade(pools[p], K, DRAWS, 8, esc, rule)
        policies["esc_w2_m8_then_think_read"] = {
            **tf.minus_envelope(
                {p: r[:2] for p, r in rows.items()}, envelope, pids, boots
            ),
            "escalated": float(np.mean([r[2] for r in rows.values()])),
        }
        for label, rds in stored.items():
            per = {}
            for p in pids:
                mine = rds.get((p, "reader_full"))
                if not mine:
                    continue
                views = al.build_views(pools[p], K, mine)
                groups = group_answers(
                    [c.answer for c in pools[p].candidates],
                    [c.correct for c in pools[p].candidates],
                )
                cells = [
                    al.policies(groups)["esc_w2_then_read"](v, rule) for v in views
                ]
                per[p] = (
                    float(np.mean([c[1] for c in cells])),
                    float(np.mean([c[0] for c in cells])),
                )
            policies[f"esc_w2_m8_then_nothink_read_{label}"] = tf.minus_envelope(
                per, envelope, pids, boots
            )
        report["policies"][rule] = policies
    contrast = {}
    for cap_fresh in (16384, 30000):
        vals = {}
        for p in pids:
            mine = list(reads.get(p, {}).values())
            if mine:
                read = np.mean([tf.at_cap(r, 16384)[0] for r in mine])
                solve = np.mean([tf.at_cap(s, cap_fresh)[0] for s in fresh[p]])
                vals[p] = float(read - solve)
        contrast[f"read_cap16384_minus_fresh_cap{cap_fresh}"] = al.mean_ci(
            vals, args.n_boot, args.seed
        )
    report["read_minus_fresh"] = contrast
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(report), encoding="utf-8")
        print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
