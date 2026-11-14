"""Zero-cost baselines on the dev pools (PREREG section 7): vote weighted by the stored
confidence, early-stopping SC, and the length of reader prompts.

    python scripts/03_dev_baselines.py --root <shards> --sets data/sets \
        --out-dir results/dev_baselines

Descriptive and on dev only; these numbers inform the freeze (H6's ESC settings, the
context window) and are not tests. Every shard is checked against its done marker, and
the pools must be exactly the dev set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from channel_audit.analysis.stats import bootstrap_means

from reader_audit import sets
from reader_audit.baselines import (
    confidence_weights,
    esc_means,
    frontier_at,
    reader_prompt_lengths,
    weighted_vote_credit,
)
from reader_audit.pools import load_pools
from reader_audit.subsets import group_answers, interpolate, sc_curve, subset_stats

KS = (4, 8)
ESC_WINDOWS = (2, 4, 8)
ESC_MAX = (8, 16)
N_PERM = 200
N_DRAWS = 100
CAPS = (1536, 2048)
PROMPT_KS = (2, 4, 8)
N_BOOT = 10_000


def interval(values: np.ndarray) -> dict:
    means = bootstrap_means(values, N_BOOT, 0)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)]}


def pct(x: float) -> str:
    return f"{100 * x:+.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", default="data/sets")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if (out / "dev_baselines.json").exists():
        print(f"error: {out / 'dev_baselines.json'} exists; use a new directory")
        return 1
    pools = load_pools(args.root)
    if list(pools) != sets.read_ids(args.sets, "dev") and set(pools) != set(
        sets.read_ids(args.sets, "dev")
    ):
        print("error: the pools are not the dev set")
        return 1
    ids = list(pools)
    levels = np.array([pools[i].problem.level for i in ids])
    groups = [
        group_answers(
            [c.answer for c in pools[i].candidates],
            [c.correct for c in pools[i].candidates],
        )
        for i in ids
    ]
    curves = np.array([sc_curve(g) for g in groups])
    selections = {"all": np.arange(len(ids))} | {
        f"level_{lv}": np.flatnonzero(levels == lv) for lv in (3, 4, 5)
    }
    report: dict = {"n_problems": len(ids)}

    # ---- stored-confidence vote vs the plain vote -------------------------------
    confidences = [c.confidence for i in ids for c in pools[i].candidates]
    correct = np.array([c.correct for i in ids for c in pools[i].candidates])
    stated = np.array([-1 if x is None else x for x in confidences])
    report["confidence_signal"] = {
        "share_missing": float((stated < 0).mean()),
        "mean_when_right": float(stated[(stated >= 0) & correct].mean()),
        "mean_when_wrong": float(stated[(stated >= 0) & ~correct].mean()),
        "share_at_least_90": float((stated >= 90).mean()),
    }
    report["weighted_vote"] = {}
    for k in KS:
        plain = np.array([subset_stats(g, k).vote for g in groups])
        weighted = np.array(
            [
                weighted_vote_credit(g, confidence_weights(pools[i].candidates), k)
                for g, i in zip(groups, ids, strict=True)
            ]
        )
        report["weighted_vote"][f"K={k}"] = {
            name: {
                "plain": interval(plain[idx]),
                "weighted": interval(weighted[idx]),
                "gain": interval(weighted[idx] - plain[idx]),
            }
            for name, idx in selections.items()
        }

    # ---- early-stopping SC vs plain SC at the same mean samples -----------------
    report["esc"] = {}
    for window in ESC_WINDOWS:
        for cap in ESC_MAX:
            runs = np.array(
                [
                    esc_means(g, window, cap, N_PERM, seed=1000 + n)
                    for n, g in enumerate(groups)
                ]
            )  # (problems, 2): mean samples, mean credit
            block = {}
            for name, idx in selections.items():
                used = float(runs[idx, 0].mean())  # SC is read at this selection's cost
                sc_at = np.array([interpolate(curves[j], used) for j in idx])
                block[name] = {
                    "mean_samples": used,
                    "esc": interval(runs[idx, 1]),
                    "sc_same_samples": interval(sc_at),
                    "gain": interval(runs[idx, 1] - sc_at),
                }
            report["esc"][f"window={window},max={cap}"] = block

    # ---- what a reader of K samples must beat: ESC on the frontier at its cost ---
    # The reader costs K + 1 samples under the generated-only rule (PREREG section 5).
    report["esc_bar"] = {}
    for name, idx in selections.items():
        mean_curve = curves[idx].mean(axis=0)
        points = [(float(k), float(mean_curve[k - 1])) for k in range(1, 19)]
        points += [
            (b[name]["mean_samples"], b[name]["esc"]["mean"])
            for b in report["esc"].values()
        ]
        for k in KS:
            cost = k + 1.0
            maj = float(np.mean([subset_stats(groups[j], k).vote for j in idx]))
            sc_star = interpolate(mean_curve, cost)
            esc_star = frontier_at(points, cost)
            report["esc_bar"][f"{name}|K={k}"] = {
                "cost_samples": cost,
                "maj": maj,
                "sc": sc_star,
                "esc_frontier": esc_star,
                "bar_over_vote": esc_star - maj,
                "bar_over_sc": esc_star - sc_star,
            }

    # ---- reader prompt lengths --------------------------------------------------
    report["prompt_tokens"] = {}
    for k in PROMPT_KS:
        for cap in CAPS:
            lengths = np.concatenate(
                [
                    reader_prompt_lengths(pools[i], k, cap, N_DRAWS, seed=n)
                    for n, i in enumerate(ids)
                ]
            )
            q = np.quantile(lengths, [0.5, 0.9, 0.99, 0.999, 1.0])
            report["prompt_tokens"][f"K={k},cap={cap}"] = {
                "mean": float(lengths.mean()),
                "p50": float(q[0]),
                "p90": float(q[1]),
                "p99": float(q[2]),
                "p99.9": float(q[3]),
                "max": float(q[4]),
                "share_over_12k": float((lengths + 2048 > 12288).mean()),
                "share_over_16k": float((lengths + 2048 > 16384).mean()),
            }

    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_baselines.json").write_text(json.dumps(report, indent=2), "utf-8")
    (out / "dev_baselines.md").write_text(render(report), "utf-8")
    print(render(report))
    return 0


def render(r: dict) -> str:
    lines = [
        "# Dev baselines (zero extra generation)",
        "",
        f"Computed on the {r['n_problems']} dev problems (18 samples each) with "
        "`scripts/03_dev_baselines.py`. Descriptive; dev only. Points are percentage "
        "points; intervals are 95% bootstrap over problems.",
        "",
        "## Stored confidence",
        "",
    ]
    c = r["confidence_signal"]
    missing, at90 = 100 * c["share_missing"], 100 * c["share_at_least_90"]
    lines += [
        f"Missing in {missing:.1f}% of samples; mean stated confidence "
        f"{c['mean_when_right']:.1f} when the answer is right and "
        f"{c['mean_when_wrong']:.1f} when wrong; {at90:.0f}% of samples state 90 or "
        "more.",
        "",
        "## Vote weighted by stored confidence, against the plain vote at the same K",
        "",
        "| Set | K | Plain vote | Weighted vote | Gain [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kname, block in r["weighted_vote"].items():
        for name, row in block.items():
            g = row["gain"]
            lines.append(
                f"| {name} | {kname[2:]} | {100 * row['plain']['mean']:.1f} | "
                f"{100 * row['weighted']['mean']:.1f} | {pct(g['mean'])} "
                f"[{pct(g['ci95'][0])}, {pct(g['ci95'][1])}] |"
            )
    lines += [
        "",
        "## Early-stopping self-consistency against plain SC at the same mean samples",
        "",
        "A window of w samples is drawn at a time; the run stops when a whole window "
        "agrees, or at the cap. The comparator is plain SC read at the mean number of "
        "samples ESC used. `gain` is what an adaptive method adds for free; a gated "
        "reader (H6) has to beat ESC, not plain SC.",
        "",
        "| Setting | Mean samples | Set | ESC | SC, same samples | Gain [95% CI] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for setting, block in r["esc"].items():
        for name in ("all", "level_5"):
            row = block[name]
            g = row["gain"]
            esc, sc = 100 * row["esc"]["mean"], 100 * row["sc_same_samples"]["mean"]
            lines.append(
                f"| {setting} | {row['mean_samples']:.2f} | {name} | "
                f"{esc:.1f} | {sc:.1f} "
                f"| {pct(g['mean'])} [{pct(g['ci95'][0])}, {pct(g['ci95'][1])}] |"
            )
    lines += [
        "",
        "## The bar for a K-candidate reader",
        "",
        "A reader of K candidates costs about K + 1 samples (generated-only rule). The "
        "frontier is the best accuracy that mixing the ESC settings above and plain SC "
        "reaches at that cost (pooled means; no interval).",
        "",
        "| Set | K | Cost (samples) | Plain vote of K | SC at that cost | "
        "ESC frontier | Reader must beat the vote by | ... and SC by |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, b in r["esc_bar"].items():
        name, k = key.split("|K=")
        lines.append(
            f"| {name} | {k} | {b['cost_samples']:.0f} | {100 * b['maj']:.1f} | "
            f"{100 * b['sc']:.1f} | {100 * b['esc_frontier']:.1f} | "
            f"{pct(b['bar_over_vote'])} | {pct(b['bar_over_sc'])} |"
        )
    lines += [
        "",
        "## Approximate reader prompt length (tokens)",
        "",
        "Question and template from the pool's own prompt, plus candidates cut at "
        "`cap` tokens. The reader's reply adds up to 2,048. The last two columns give "
        "the share of prompts whose length plus 2,048 exceeds the context window.",
        "",
        "| K, cap | Mean | p50 | p90 | p99 | p99.9 | Max | > 12,288 | > 16,384 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in r["prompt_tokens"].items():
        lines.append(
            f"| {name} | {row['mean']:.0f} | {row['p50']:.0f} | {row['p90']:.0f} | "
            f"{row['p99']:.0f} | {row['p99.9']:.0f} | {row['max']:.0f} | "
            f"{100 * row['share_over_12k']:.2f}% | {100 * row['share_over_16k']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
