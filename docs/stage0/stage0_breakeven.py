"""Stage 0 of the thesis roadmap: break-even and headroom on the 600 dev problems.

Each problem has 18 exchangeable samples (2 replicates x (3 round-0 agents + 6 pool
samples), all with the round-0 prompt). All quantities for a random k-subset are exact:
they sum over the count vector of each answer group, weighted by the hypergeometric
probability, instead of enumerating subsets. A brute-force check runs first.

    python stage0_breakeven.py --root <shards dir> --out-dir <new dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

from channel_audit.aggregate import cluster_answers

N_BOOT = 10_000
SEED = 0
KS = (3, 4, 8)
RECOVERY = (0.27, 0.57)  # AggLM share of pass@8 - maj@8 recovered (thinking candidates)
MARGIN = 0.015  # H1 planning effect
FLOP_WEIGHT_06B = 0.35  # a 0.6B token costs about 0.35 of a 1.7B token (report estimate)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(root: Path) -> list[dict]:
    """One record per problem: level, per-sample group label and tokens."""
    problems: dict[str, dict] = {}
    for shard in sorted(root.glob("shard_*")):
        meta = {p["problem_id"]: p for p in read_jsonl(shard / "problems.jsonl")}
        graded = {g["gen_key"]: g for g in read_jsonl(shard / "graded.jsonl")}
        for g in read_jsonl(shard / "generations.jsonl"):
            if g["arm"] not in ("round0", "sc_pool"):
                continue
            pid = g["problem_id"]
            rec = problems.setdefault(
                pid, {"problem_id": pid, "level": meta[pid]["level"], "samples": []}
            )
            gr = graded[g["gen_key"]]
            rec["samples"].append(
                {
                    "answer": gr["extracted"],
                    "correct": bool(gr["correct"]),
                    "completion_tokens": g["completion_tokens"],
                    "prompt_tokens": g["prompt_tokens"],
                    "truncated": g["finish_reason"] != "stop",
                }
            )
    return list(problems.values())


def group_problem(rec: dict) -> dict:
    """Cluster the 18 answers; return group sizes and which group is correct."""
    labels, _ = cluster_answers([s["answer"] for s in rec["samples"]])
    n_groups = max((lab for lab in labels if lab is not None), default=-1) + 1
    sizes = [0] * n_groups
    correct = [False] * n_groups
    inconsistent = 0
    for s, lab in zip(rec["samples"], labels, strict=True):
        if lab is None:
            continue
        sizes[lab] += 1
        if s["correct"]:
            correct[lab] = True
    for s, lab in zip(rec["samples"], labels, strict=True):
        if lab is not None and s["correct"] != correct[lab]:
            inconsistent += 1
    n_none = sum(1 for lab in labels if lab is None)
    return {"sizes": sizes, "correct": correct, "n_none": n_none, "bad": inconsistent}


def subset_stats(sizes: list[int], correct: list[bool], n_none: int, k: int) -> dict:
    """Exact stats for a uniformly random k-subset of the problem's samples."""
    n = sum(sizes) + n_none
    total = comb(n, k)
    out = dict(vote=0.0, pass_=0.0, c0=0.0, c1=0.0, c2=0.0, minority=0.0, strict=0.0)
    out.update(unanimous=0.0, unanimous_right=0.0)
    groups = sizes + [n_none]  # last group = unparsable, never votes
    ranges = [range(0, min(g, k) + 1) for g in groups]
    for counts in itertools.product(*ranges):
        if sum(counts) != k:
            continue
        w = 1.0
        for g, m in zip(groups, counts, strict=True):
            w *= comb(g, m)
        w /= total
        votes = counts[:-1]
        top = max(votes, default=0)
        n_right = sum(m for m, ok in zip(votes, correct, strict=True) if ok)
        tied = [i for i, m in enumerate(votes) if m == top and top > 0]
        credit = 0.0
        if tied and any(correct[i] for i in tied):
            credit = 1.0 / len(tied)
        out["vote"] += w * credit
        out["pass_"] += w * (n_right > 0)
        out["c0"] += w * (n_right == 0)
        out["c1"] += w * (n_right == 1)
        out["c2"] += w * (n_right >= 2)
        out["minority"] += w * ((n_right > 0) and credit < 1.0)
        out["strict"] += w * ((n_right > 0) and not any(correct[i] for i in tied))
        distinct = sum(1 for m in counts if m > 0)
        if distinct == 1 and counts[-1] == 0:
            out["unanimous"] += w
            out["unanimous_right"] += w * (n_right > 0)
    return out


def brute_force(sizes, correct, n_none, k) -> dict:
    """Reference: enumerate subsets of the 18 samples explicitly."""
    items = [(i, correct[i]) for i, s in enumerate(sizes) for _ in range(s)]
    items += [(None, False)] * n_none
    vote = pass_ = unan = 0.0
    subsets = list(itertools.combinations(range(len(items)), k))
    for sub in subsets:
        picked = [items[i] for i in sub]
        counts: dict[int, int] = {}
        for g, _ in picked:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        top = max(counts.values(), default=0)
        tied = [g for g, m in counts.items() if m == top and top > 0]
        if tied and any(correct[g] for g in tied):
            vote += 1.0 / len(tied)
        pass_ += any(ok for _, ok in picked)
        unan += len({g for g, _ in picked}) == 1 and picked[0][0] is not None
    n = len(subsets)
    return {"vote": vote / n, "pass_": pass_ / n, "unanimous": unan / n}


def self_check(grouped: list[dict]) -> None:
    for g in grouped[:3]:
        # brute force is feasible only for small k on 18 samples
        for k in (2, 3, 4):
            exact = subset_stats(g["sizes"], g["correct"], g["n_none"], k)
            ref = brute_force(g["sizes"], g["correct"], g["n_none"], k)
            for key in ("vote", "pass_", "unanimous"):
                assert abs(exact[key] - ref[key]) < 1e-12, (key, k, exact, ref)


def sc_curve(g: dict) -> np.ndarray:
    n = sum(g["sizes"]) + g["n_none"]
    return np.array(
        [subset_stats(g["sizes"], g["correct"], g["n_none"], k)["vote"]
         for k in range(1, n + 1)]
    )


def interp(curve: np.ndarray, k_star: float) -> float:
    """Linear interpolation of an SC curve indexed k = 1..len (mean over problems)."""
    ks = np.arange(1, len(curve) + 1)
    return float(np.interp(k_star, ks, curve))


def cost_rules(g_tok: float, p_tok: float, k: int) -> dict[str, float]:
    """Matched SC sample count k* for a reader that writes about one sample."""
    per_sample_gen = g_tok
    per_sample_all = g_tok + p_tok
    gen_only = k + 1.0
    # reader prompt = problem + k candidates; its output about one sample
    prompt_incl = ((2 * k + 1) * g_tok + (k + 1) * p_tok) / per_sample_all
    reader_06 = FLOP_WEIGHT_06B * (k * g_tok + p_tok + g_tok)
    flop_06b = (k * per_sample_all + reader_06) / per_sample_all
    return {
        "generated_only": gen_only,
        "generated_plus_prompt": prompt_incl,
        "flops_0.6b_reader": flop_06b,
        "_g": per_sample_gen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} is not empty")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    recs = load(Path(args.root))
    assert len(recs) == 600, len(recs)
    assert all(len(r["samples"]) == 18 for r in recs), "expected 18 samples each"
    grouped = [group_problem(r) for r in recs]
    bad = sum(g["bad"] for g in grouped)
    self_check(grouped)
    print(f"brute-force check passed; clusters inconsistent with grading: {bad}")

    levels = np.array([r["level"] for r in recs])
    g_tok = float(np.mean([s["completion_tokens"] for r in recs for s in r["samples"]]))
    p_tok = float(np.mean([s["prompt_tokens"] for r in recs for s in r["samples"]]))
    trunc = float(np.mean([s["truncated"] for r in recs for s in r["samples"]]))
    print(f"mean generated tokens/sample {g_tok:.0f}, prompt {p_tok:.0f}, truncated {trunc:.3f}")

    curves = np.array([sc_curve(g) for g in grouped])  # (600, 18)
    per_k = {
        k: [subset_stats(g["sizes"], g["correct"], g["n_none"], k) for g in grouped]
        for k in KS
    }

    rng = np.random.default_rng(SEED)
    idx_all = {
        name: np.flatnonzero(mask)
        for name, mask in {
            "all": np.ones(len(recs), bool),
            "level_3": levels == 3,
            "level_4": levels == 4,
            "level_5": levels == 5,
        }.items()
    }

    def summarise(idx: np.ndarray, boot: np.ndarray | None) -> dict:
        """Metrics on the problems in idx; boot is a (B, len(idx)) index matrix."""
        res: dict = {"n_problems": int(len(idx))}
        sc = curves[idx]

        def stat(vals_fn):
            point = float(vals_fn(np.arange(len(idx))))
            if boot is None:
                return {"mean": point}
            bs = np.array([vals_fn(b) for b in boot])
            return {
                "mean": point,
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            }

        res["sc_curve"] = {
            str(k): float(sc[:, k - 1].mean()) for k in range(1, 19)
        }
        for k in KS:
            rows = [per_k[k][i] for i in idx]
            vote = np.array([r["vote"] for r in rows])
            pas = np.array([r["pass_"] for r in rows])
            minority = np.array([r["minority"] for r in rows])
            strict = np.array([r["strict"] for r in rows])
            c0 = np.array([r["c0"] for r in rows])
            c1 = np.array([r["c1"] for r in rows])
            c2 = np.array([r["c2"] for r in rows])
            unan = np.array([r["unanimous"] for r in rows])
            unan_right = np.array([r["unanimous_right"] for r in rows])
            kres: dict = {
                f"maj@{k}": stat(lambda b, v=vote: v[b].mean()),
                f"pass@{k}": stat(lambda b, v=pas: v[b].mean()),
                "headroom": stat(lambda b, v=pas - vote: v[b].mean()),
                "minority_correct": stat(lambda b, v=minority: v[b].mean()),
                "strict_minority": stat(lambda b, v=strict: v[b].mean()),
                "c0": stat(lambda b, v=c0: v[b].mean()),
                "c1": stat(lambda b, v=c1: v[b].mean()),
                "c2plus": stat(lambda b, v=c2: v[b].mean()),
                "unanimous": stat(lambda b, v=unan: v[b].mean()),
                "unanimous_wrong": stat(lambda b, u=unan, r=unan_right: (u[b] - r[b]).mean()),
            }
            # headroom is zero on unanimous sets (right: vote = pass = 1; wrong: both 0),
            # so a unanimity gate gives up none of it; it only gives up fresh solving
            if True:
                rules = cost_rules(g_tok, p_tok, k)
                be = {}
                for name, kstar in rules.items():
                    if name.startswith("_"):
                        continue
                    kstar_c = min(kstar, 18.0)
                    sc_star = lambda b, ks=kstar_c: np.array(
                        [np.interp(ks, np.arange(1, 19), sc[b].mean(axis=0))]
                    )[0]
                    maj = lambda b, v=vote: v[b].mean()
                    be[name] = {
                        "k_star": float(kstar),
                        "sc_at_kstar": stat(sc_star),
                        "break_even": stat(lambda b, f=sc_star, m=maj: f(b) - m(b)),
                    }
                # gated reader under generated + prompt: reads only non-unanimous sets
                r_cost = ((k * g_tok + p_tok + g_tok)) / (g_tok + p_tok)
                nonunan = float(1.0 - unan.mean())
                kstar_gate = k + nonunan * r_cost
                sc_gate = lambda b, ks=kstar_gate: np.array(
                    [np.interp(ks, np.arange(1, 19), sc[b].mean(axis=0))]
                )[0]
                be["gated_unanimity_prompt_incl"] = {
                    "k_star": float(kstar_gate),
                    "share_read": nonunan,
                    "sc_at_kstar": stat(sc_gate),
                    "break_even": stat(lambda b, f=sc_gate, v=vote: f(b) - v[b].mean()),
                }
                kres["cost_rules"] = be
            res[f"K={k}"] = kres
        return res

    results = {}
    for name, idx in idx_all.items():
        boot = rng.integers(0, len(idx), size=(N_BOOT, len(idx)))
        results[name] = summarise(idx, boot)

    # G1 gate on level-5 dev, generated-only rule
    gate = {}
    for k in (4, 8):
        r = results["level_5"][f"K={k}"]
        be = r["cost_rules"]["generated_only"]["break_even"]["mean"]
        head = r["headroom"]["mean"]
        gate[f"K={k}"] = {
            "headroom": head,
            "break_even_generated_only": be,
            "needed_gain": be + MARGIN,
            "recovery_low": RECOVERY[0],
            "recovery_high": RECOVERY[1],
            "expected_gain_low": head * RECOVERY[0],
            "expected_gain_high": head * RECOVERY[1],
            "pass_low": head * RECOVERY[0] >= be + MARGIN,
            "pass_high": head * RECOVERY[1] >= be + MARGIN,
            "max_possible_gain_if_all_headroom_recovered": head,
        }
    payload = {
        "n_problems": len(recs),
        "tokens": {"generated_per_sample": g_tok, "prompt_per_sample": p_tok},
        "truncated_share": trunc,
        "clusters_inconsistent_with_grading": bad,
        "results": results,
        "g1_level_5": gate,
    }
    (out / "stage0.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
