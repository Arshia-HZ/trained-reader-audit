"""The Stage 1 harness check (gate G2) as one resumable job.

Generate a pool of candidates for each harness problem, draw K-candidate sets, run the
untrained reader and its controls on them, and judge the four signs of 2608.18379
(``g2``). Everything goes through the audit's append-only stores, so a restarted job
only does what is missing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from channel_audit.backends.base import Backend
from channel_audit.runner import (
    RunConfig,
    create_run,
    grade_pending,
    run_round0,
    run_sc_pool,
)
from channel_audit.schema import Problem

from reader_audit import reader_eval as ev
from reader_audit.controls import Truncate
from reader_audit.g2 import format_verdict, g2_verdict
from reader_audit.pools import read_pools
from reader_audit.reader_run import open_reader_run, plan_reader_run, run_reader

G2_ARMS: tuple[str, ...] = ("reader_full", "candidate_free", "answer_only", "placebo")
ROUND0_AGENTS = 3


def run_g2(
    out_dir: str | Path,
    problems: Sequence[Problem],
    cfg: RunConfig,
    *,
    backend: Backend | None = None,
    clipper: Truncate | None = None,
    samples: int = 16,
    k: int = 4,
    n_draws: int = 8,
    cap: int | None = 4096,
) -> dict:
    """Run (or resume) the harness check and write ``g2_report.json`` and ``.md``."""
    if samples <= ROUND0_AGENTS or samples < k:
        raise ValueError(f"need more than {max(ROUND0_AGENTS, k)} samples per problem")
    out = Path(out_dir)
    pool_cfg = cfg.model_copy(
        update={"replicates": 1, "n_pool": samples - ROUND0_AGENTS}
    )
    pool_ctx = create_run(out / "pool", pool_cfg, problems, backend)
    run_round0(pool_ctx)
    run_sc_pool(pool_ctx)
    grade_pending(pool_ctx)
    pools = read_pools(out / "pool")

    plan = plan_reader_run(
        pools,
        cfg,
        arms=G2_ARMS,
        ks=(k,),
        n_draws=n_draws,
        replicates=1,
        cap=cap,
        truncate=clipper,
    )
    reader_ctx = open_reader_run(out / "reader", cfg, problems, pool_ctx.backend)
    run_reader(reader_ctx, plan)

    outcomes = ev.load_outcomes(out / "reader") + ev.vote_outcomes(pools, (k,), n_draws)
    verdict = g2_verdict(outcomes, k)
    report = {
        "verdict": verdict,
        "n_problems": len(pools),
        "samples_per_problem": samples,
        "k": k,
        "n_draws": n_draws,
        "candidate_cap": cap,
        "strata_counts": _strata_counts(outcomes, k),
        "pool_truncated_share": _truncated_share(pools),
        "reader_generations": len(plan.planned),
    }
    (out / "g2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "g2_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _strata_counts(outcomes: Sequence[ev.Outcome], k: int) -> dict[str, int]:
    """Candidate sets per stratum (one count per set, from the vote outcomes)."""
    counts: dict[str, int] = {}
    for o in outcomes:
        if o.arm == ev.VOTE and o.k == k:
            counts[o.stratum] = counts.get(o.stratum, 0) + 1
    return dict(sorted(counts.items()))


def _truncated_share(pools) -> float:
    cands = [c for pool in pools.values() for c in pool.candidates]
    return sum(c.truncated for c in cands) / len(cands)


def _markdown(report: dict) -> str:
    r = report
    truncated = 100 * r["pool_truncated_share"]
    return "\n".join(
        [
            "# Gate G2: harness check",
            "",
            f"{r['n_problems']} problems, {r['samples_per_problem']} candidate samples "
            f"each, K = {r['k']}, {r['n_draws']} sets per problem, candidates cut at "
            f"{r['candidate_cap']} tokens. {truncated:.1f}% of the candidates were "
            "truncated.",
            f"Sets per stratum: {r['strata_counts']}.",
            "",
            "```",
            format_verdict(report["verdict"]),
            "```",
            "",
        ]
    )
