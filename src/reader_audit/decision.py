"""Conclusion rules (PREREG section 4.2): how results become claims.

Three things are decided here and nowhere else:

- the framing of the paper from H1a and H1b (D26);
- the word used for a supported result, which depends on the seeds (D8);
- the number of seeds, from the measured cost of a 300-step run (D7).

A hypothesis is *supported* only by its Holm-adjusted p-value (``inference.decide``).
Agreement in sign across seeds is a robustness criterion; it never turns an
unsupported result into a supported one.
"""

from __future__ import annotations

from collections.abc import Sequence

SEED_RULE_HOURS = 20.0  # at most this many GPU-hours per 300-step run buys five seeds
FRAMINGS = ("C1-full", "C1-narrow", "C2-lead")


def framing(h1a_supported: bool, h1b_supported: bool) -> str:
    """How the study is framed.

    H1a and H1b supported: the full C1 claim, against base SC and against SC from a
    solver trained on the same tokens. H1a supported, H1b not: a narrower claim against
    base SC only, and the paper says that solver-GRPO matches it. H1a not supported:
    lead with C2 (what GRPO changes in a reader), whatever H1b says.
    """
    if not h1a_supported:
        return "C2-lead"
    return "C1-full" if h1b_supported else "C1-narrow"


def verdict(supported: bool, seed_gains: Sequence[float]) -> str:
    """The word for one hypothesis: not supported, supported, or supported and robust.

    Robust means every individual seed's gain has the claimed sign (here: positive).
    """
    if not supported:
        return "not supported"
    if seed_gains and all(g > 0 for g in seed_gains):
        return "supported and robust"
    return "supported, not robust"


def seeds_for(hours_per_run: float) -> int:
    """Seeds per main arm: five if a 300-step run costs at most 20 GPU-hours, else 3."""
    if hours_per_run <= 0:
        raise ValueError("a run takes a positive number of hours")
    return 5 if hours_per_run <= SEED_RULE_HOURS else 3
