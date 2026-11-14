"""Gate G2: does the harness reproduce the signs of 2608.18379? (PREREG section 8)

On AIME 2025 + HMMT 2025 with Qwen3-4B-Instruct-2507 and K = 4, the untrained reader
must show, by stratum of the candidate set:

- c0: the reader is below the candidate-free solve, and so is the placebo;
- c2+: the reader is above the candidate-free solve (interval excluding 0), and the
  plain vote of the candidates is above the reader.

c1 is reported and not judged: the paper's c1 contrast is unresolved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reader_audit.reader_eval import VOTE, Outcome, contrast


@dataclass(frozen=True)
class Check:
    name: str
    stratum: str
    a: str
    b: str
    sign: int  # the required sign of mean(a - b)
    needs_ci: bool = False  # the 95% interval must also exclude 0


CHECKS: tuple[Check, ...] = (
    Check("c0_reader_below_candidate_free", "c0", "reader_full", "candidate_free", -1),
    Check("c0_placebo_below_candidate_free", "c0", "placebo", "candidate_free", -1),
    Check(
        "c2_reader_above_candidate_free",
        "c2+",
        "reader_full",
        "candidate_free",
        +1,
        needs_ci=True,
    ),
    Check("c2_vote_above_reader", "c2+", VOTE, "reader_full", +1),
)
REPORT_ONLY: tuple[Check, ...] = (
    Check("c1_reader_vs_candidate_free", "c1", "reader_full", "candidate_free", 0),
    Check("c2_answer_only_vs_reader", "c2+", "answer_only", "reader_full", 0),
    Check("c2_placebo_vs_candidate_free", "c2+", "placebo", "candidate_free", 0),
)


def _passes(check: Check, summary: dict) -> bool:
    if check.sign < 0:
        ok = summary["mean"] < 0
    else:
        ok = summary["mean"] > 0
        if check.needs_ci:
            ok = ok and summary["ci95_low"] > 0
    return bool(ok)


def g2_verdict(
    outcomes: Sequence[Outcome], k: int, n_boot: int = 10_000, seed: int = 0
) -> dict:
    """The four judged checks, the report-only contrasts and the overall verdict."""
    judged = {}
    for check in CHECKS:
        s = contrast(outcomes, check.a, check.b, k, check.stratum, None, n_boot, seed)
        judged[check.name] = (
            {"status": "not run: no problem has both arms"}
            if s is None
            else {**s, "required_sign": check.sign, "passes": _passes(check, s)}
        )
    reported = {}
    for check in REPORT_ONLY:
        s = contrast(outcomes, check.a, check.b, k, check.stratum, None, n_boot, seed)
        reported[check.name] = s or {"status": "not run"}
    return {
        "k": k,
        "passed": all(r.get("passes") is True for r in judged.values()),
        "checks": judged,
        "report_only": reported,
    }


def format_verdict(v: dict) -> str:
    lines = [f"G2 at K = {v['k']}: {'PASS' if v['passed'] else 'FAIL'}", ""]
    for name, r in v["checks"].items():
        if "mean" not in r:
            lines.append(f"  {name}: {r['status']}")
            continue
        lines.append(
            f"  {name}: {100 * r['mean']:+.1f} points "
            f"[{100 * r['ci95_low']:+.1f}, {100 * r['ci95_high']:+.1f}] over "
            f"{r['n_problems']} problems -> {'ok' if r['passes'] else 'FAILS'}"
        )
    lines.append("")
    lines.append("Reported, not judged:")
    for name, r in v["report_only"].items():
        if "mean" in r:
            lines.append(
                f"  {name}: {100 * r['mean']:+.1f} points "
                f"[{100 * r['ci95_low']:+.1f}, {100 * r['ci95_high']:+.1f}] over "
                f"{r['n_problems']} problems"
            )
        else:
            lines.append(f"  {name}: {r['status']}")
    return "\n".join(lines)
