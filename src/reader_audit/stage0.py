"""Stage 0: break-even and headroom of a K-candidate reader on the dev pools.

For each problem, its pool gives the exact vote, pass@K, strata and self-consistency
curve of a random K-subset (``subsets``). Averages over problems are bootstrapped over
problems with the audit's frozen ``bootstrap_means``. Gate G1 (PREREG section 3, D2):
the headroom times a recovery rate must clear the break-even plus the planning effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
from channel_audit.analysis.stats import bootstrap_means

from reader_audit.budget import RULES, planning_k_star
from reader_audit.pools import ProblemPool
from reader_audit.subsets import (
    SubsetStats,
    group_answers,
    interpolate,
    sc_curve,
    subset_stats,
)

KS = (3, 4, 8)
RECOVERY = (
    0.27,
    0.57,
)  # AggLM's share of pass@8 - maj@8 recovered (thinking candidates)
PLANNING_EFFECT = 0.015
N_BOOT = 10_000
SEED = 0


def _interval(values: np.ndarray, n_boot: int, seed: int) -> dict:
    means = bootstrap_means(values, n_boot, seed)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)]}


def _column(rows: Sequence[SubsetStats], name: str) -> np.ndarray:
    return np.array([getattr(r, name) for r in rows])


def per_problem(pools: Mapping[str, ProblemPool], ks: Sequence[int] = KS) -> dict:
    """Exact per-problem quantities, in the order of ``pools``."""
    ids, levels, curves, stats = [], [], [], {k: [] for k in ks}
    tokens_g, tokens_p, truncated, inconsistent = [], [], [], 0
    for pid, pool in pools.items():
        groups = group_answers(
            [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
        )
        inconsistent += groups.inconsistent
        ids.append(pid)
        levels.append(pool.problem.level)
        curves.append(sc_curve(groups))
        for k in ks:
            stats[k].append(subset_stats(groups, k))
        tokens_g += [c.completion_tokens for c in pool.candidates]
        tokens_p += [c.prompt_tokens for c in pool.candidates]
        truncated += [c.truncated for c in pool.candidates]
    return {
        "ids": ids,
        "levels": np.array(levels),
        "curves": np.array(curves),
        "stats": stats,
        "gen_tokens": float(np.mean(tokens_g)),
        "prompt_tokens": float(np.mean(tokens_p)),
        "truncated_share": float(np.mean(truncated)),
        "inconsistent": inconsistent,
    }


def summarise(
    data: dict, idx: np.ndarray, *, ks: Sequence[int] = KS, n_boot=N_BOOT, seed=SEED
) -> dict:
    """Metrics on the problems ``idx`` (positions in ``data``)."""
    curves = data["curves"][idx]
    out: dict = {
        "n_problems": int(len(idx)),
        "sc_curve": {
            str(k): float(curves[:, k - 1].mean())
            for k in range(1, curves.shape[1] + 1)
        },
    }
    g, p = data["gen_tokens"], data["prompt_tokens"]
    for k in ks:
        rows = [data["stats"][k][i] for i in idx]
        col = partial(_column, rows)
        vote, unan = col("vote"), col("unanimous")
        block = {
            f"maj@{k}": _interval(vote, n_boot, seed),
            f"pass@{k}": _interval(col("pass_at_k"), n_boot, seed),
            "headroom": _interval(col("pass_at_k") - vote, n_boot, seed),
            "minority_correct": _interval(col("minority"), n_boot, seed),
            "strict_minority": _interval(col("strict_minority"), n_boot, seed),
            "c0": _interval(col("c0"), n_boot, seed),
            "c1": _interval(col("c1"), n_boot, seed),
            "c2plus": _interval(col("c2plus"), n_boot, seed),
            "unanimous": _interval(unan, n_boot, seed),
            "unanimous_wrong": _interval(unan - col("unanimous_right"), n_boot, seed),
        }
        rules = {}
        for rule in RULES:
            share = float(1.0 - unan.mean())
            k_star = planning_k_star(rule, k, g, p, share_read=share)
            at_star = np.array(
                [interpolate(c, min(k_star, len(c))) for c in curves], dtype=float
            )
            entry = {
                "k_star": float(k_star),
                "sc_at_kstar": _interval(at_star, n_boot, seed),
                "break_even": _interval(at_star - vote, n_boot, seed),
            }
            if rule == "gated_unanimity_prompt_incl":
                entry["share_read"] = share
            rules[rule] = entry
        block["cost_rules"] = rules
        out[f"K={k}"] = block
    return out


def gate_g1(summary: dict, ks: Sequence[int] = (4, 8)) -> dict:
    """G1 on one summary (level-5 dev): headroom x recovery against break-even + 1.5."""
    gate = {}
    for k in ks:
        block = summary[f"K={k}"]
        head = block["headroom"]["mean"]
        be = block["cost_rules"]["generated_only"]["break_even"]["mean"]
        need = be + PLANNING_EFFECT
        low, high = head * RECOVERY[0], head * RECOVERY[1]
        gate[f"K={k}"] = {
            "headroom": head,
            "break_even_generated_only": be,
            "needed_gain": need,
            "expected_gain_low": low,
            "expected_gain_high": high,
            "pass_low": bool(low >= need),
            "pass_high": bool(high >= need),
        }
    return gate


def analyze(
    pools: Mapping[str, ProblemPool],
    *,
    ks: Sequence[int] = KS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    data = per_problem(pools, ks)
    levels = data["levels"]
    selections = {"all": np.arange(len(levels))}
    for lv in sorted(set(levels.tolist())):
        selections[f"level_{lv}"] = np.flatnonzero(levels == lv)
    results = {
        name: summarise(data, idx, ks=ks, n_boot=n_boot, seed=seed)
        for name, idx in selections.items()
    }
    out = {
        "n_problems": len(data["ids"]),
        "tokens": {
            "generated_per_sample": data["gen_tokens"],
            "prompt_per_sample": data["prompt_tokens"],
        },
        "truncated_share": data["truncated_share"],
        "clusters_inconsistent_with_grading": data["inconsistent"],
        "results": results,
    }
    if "level_5" in results and all(k in ks for k in (4, 8)):
        out["g1_level_5"] = gate_g1(results["level_5"])
    return out
