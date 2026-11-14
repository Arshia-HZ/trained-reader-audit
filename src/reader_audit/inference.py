"""The confirmatory inference procedure (PREREG D9, D10, D24).

One procedure for every hypothesis, written so that "a 95% interval above zero" is never
mistaken for a Holm-corrected test:

1. Per problem, a paired difference ``d_p`` (the unit is the problem).
2. 10,000 bootstrap resamples of the problems (numpy ``default_rng(0)``); the means of
   the resamples are ``m*``.
3. The 95% percentile interval of ``m*`` is the effect estimate; it is not the decision.
4. The p-value follows the audit's frozen rule, on the **two-sided scale**:
   superiority (null: mean <= 0): p = 2 * (1 + #{m* <= 0}) / (B + 1);
   non-inferiority with margin delta (null: mean <= -delta):
   p = 2 * (1 + #{m* <= -delta}) / (B + 1); both capped at 1. A two-sided test uses the
   audit's two-sided p. For one hypothesis alone, p < 0.05 on this scale is the same as
   its 95% interval excluding the null.
5. Holm over the preregistered family, on those p-values; a hypothesis is supported
   when its adjusted p is below 0.05. A hypothesis that was not run counts with p = 1,
   so the family size never shrinks.
6. An intersection test (several components must all hold) takes the largest of its
   components' p; H3 itself is one pooled test since R11.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from channel_audit.analysis.contrasts import bootstrap_p, holm_adjust
from channel_audit.analysis.stats import bootstrap_means

ALPHA = 0.05
N_BOOT = 10_000
SEED = 0
KINDS = ("superiority", "noninferiority", "two-sided")


@dataclass(frozen=True)
class HypothesisTest:
    name: str
    kind: str  # superiority, noninferiority or two-sided
    margin: float = (
        0.0  # non-inferiority margin as a positive fraction (0.02 = 2 points)
    )

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown test kind {self.kind!r}; expected one of {KINDS}"
            )
        if self.kind == "noninferiority" and self.margin <= 0:
            raise ValueError("a non-inferiority test needs a positive margin")


def p_value(means: np.ndarray, test: HypothesisTest) -> float:
    """The p-value of a test from bootstrap means, on the two-sided scale."""
    if test.kind == "two-sided":
        return bootstrap_p(means, "two-sided")
    shift = test.margin if test.kind == "noninferiority" else 0.0
    return min(1.0, 2.0 * bootstrap_p(means + shift, "greater"))


def evaluate(
    diffs: Mapping[str, float],
    test: HypothesisTest,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Effect estimate with its interval, and the test's p-value."""
    values = np.array([diffs[k] for k in sorted(diffs)], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("a paired difference is not finite")
    means = bootstrap_means(values, n_boot, seed)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "name": test.name,
        "kind": test.kind,
        "margin": test.margin,
        "n_problems": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p": p_value(means, test),
    }


def intersection(results: Sequence[Mapping]) -> dict:
    """An intersection-union test: it holds only if every component does."""
    if not results:
        raise ValueError("an intersection test needs at least one component")
    return {
        "p": max(r["p"] for r in results),
        "components": {r["name"]: r["p"] for r in results},
    }


def decide(
    family: Mapping[str, float], preregistered: Sequence[str], alpha: float = ALPHA
) -> dict[str, dict]:
    """Holm over the preregistered family; a hypothesis not run counts with p = 1."""
    unknown = set(family) - set(preregistered)
    if unknown:
        raise ValueError(f"not in the preregistered family: {sorted(unknown)}")
    full = {name: family.get(name, 1.0) for name in preregistered}
    adjusted = holm_adjust(full)
    return {
        name: {
            "p": full[name],
            "p_holm": adjusted[name],
            "supported": bool(adjusted[name] < alpha),
            "run": name in family,
        }
        for name in preregistered
    }
