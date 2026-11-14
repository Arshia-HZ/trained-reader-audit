"""Power under the preregistered Holm family (PREREG section 6).

A test on the two-sided scale at level ``a`` rejects when its z statistic exceeds
``z(1 - a / 2)``. Holm compares the j-th smallest of m p-values with
``alpha / (m - j + 1)``, so the smallest p-value of the six-hypothesis Family 1 faces
``alpha / 6``: the worst case for any one hypothesis, and the level used for the
headline numbers. Everything is a normal approximation, with the standard deviation a
placeholder until the freeze replaces it with the one measured on dev.

The stratum-wise standard deviations come from a binomial model: a problem has ``m``
reader outcomes per arm in the stratum (replicates times the number of its sets there),
each a Bernoulli with p = 0.5 (the largest variance), and effects do not vary between
problems. A difference of two arm means then has sd sqrt(2 p (1 - p) / m); a difference
of two such differences sqrt(4 p (1 - p) / m); a reader mean against a fixed vote credit
sqrt(p (1 - p) / m).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist

_NORMAL = NormalDist()
ALPHA = 0.05
FAMILY1_SIZE = 6


def z(probability: float) -> float:
    return _NORMAL.inv_cdf(probability)


def holm_levels(m: int, alpha: float = ALPHA) -> list[float]:
    """The level each p-value, smallest first, is compared with: alpha/m ... alpha."""
    return [alpha / (m - j) for j in range(m)]


def standard_error(n: float, sd: float) -> float:
    return sd / sqrt(n)


def mde(n: float, sd: float, level: float, power: float = 0.8) -> float:
    """The smallest true effect a test at ``level`` (two-sided) finds with ``power``."""
    return (z(1 - level / 2) + z(power)) * standard_error(n, sd)


def superiority_power(n: float, sd: float, effect: float, level: float) -> float:
    return _NORMAL.cdf(effect / standard_error(n, sd) - z(1 - level / 2))


def noninferiority_power(
    n: float, sd: float, margin: float, true_diff: float, level: float
) -> float:
    """Power to show the true difference is above ``-margin`` (null: at or below it)."""
    return _NORMAL.cdf((true_diff + margin) / standard_error(n, sd) - z(1 - level / 2))


def n_for_noninferiority(
    sd: float, margin: float, true_diff: float, level: float, power: float = 0.8
) -> int:
    gap = true_diff + margin
    if gap <= 0:
        raise ValueError("the true difference must lie above -margin")
    return ceil(((z(1 - level / 2) + z(power)) * sd / gap) ** 2)


# ---- the binomial design model ---------------------------------------------------


@dataclass(frozen=True)
class StratumDesign:
    """What the dev pools say about a stratum (or an answer-swap population)."""

    coverage: float  # share of problems with at least one set of the kind
    sets_per_covered_problem: float

    def problems(self, n_primary: int) -> float:
        return self.coverage * n_primary

    def outcomes_per_arm(self, replicates: int) -> float:
        return replicates * self.sets_per_covered_problem


def sd_binomial(kind: str, outcomes: float, p: float = 0.5) -> float:
    """Standard deviation of a per-problem contrast made of arm means of Bernoullis."""
    parts = {"diff": 2, "did": 4, "vs_fixed": 1}
    if kind not in parts:
        raise ValueError(
            f"unknown contrast kind {kind!r}; expected one of {sorted(parts)}"
        )
    return sqrt(parts[kind] * p * (1 - p) / outcomes)


def family1_rows(
    designs: Mapping[str, StratumDesign],
    n_primary: int,
    replicates: int,
    sd_h1: float = 0.18,
    h1_effect: float = 0.015,
    h2_effect: float = 0.10,
    h3_margin: float = 0.02,
    h4a_margin: float = 0.01,
    alpha: float = ALPHA,
    m: int = FAMILY1_SIZE,
    draws: int | None = None,
) -> list[dict]:
    """One row per Family 1 hypothesis (H3 per stratum): n, sd, MDE and power.

    ``designs`` holds the keys c0, c1, c2+ and swap_primary. Levels: the Holm worst case
    ``alpha / m`` and the unadjusted ``alpha``. With ``draws`` an extra informational
    row gives H1a with the standard deviation its binomial noise alone would have at
    ``replicates * draws`` outcomes per problem, to show what more draws can buy.
    """
    worst, single = alpha / m, alpha
    rows: list[dict] = []

    def row(name, n, sd, kind, planning, power_at):
        rows.append(
            {
                "hypothesis": name,
                "n_problems": round(n),
                "sd": sd,
                "kind": kind,
                "planning": planning,
                "mde_holm_worst": mde(n, sd, worst),
                "mde_unadjusted": mde(n, sd, single),
                "power_holm_worst": power_at(worst),
                "power_unadjusted": power_at(single),
            }
        )

    for name in ("H1a", "H1b"):
        row(
            name,
            n_primary,
            sd_h1,
            "superiority",
            h1_effect,
            lambda lv: superiority_power(n_primary, sd_h1, h1_effect, lv),
        )
    c1 = designs["c1"]
    n_c1, sd_c1 = (
        c1.problems(n_primary),
        sd_binomial("did", c1.outcomes_per_arm(replicates)),
    )
    row(
        "H2 (c1)",
        n_c1,
        sd_c1,
        "superiority",
        h2_effect,
        lambda lv: superiority_power(n_c1, sd_c1, h2_effect, lv),
    )
    for stratum in ("c0", "c1", "c2+"):
        d = designs[stratum]
        n_s = d.problems(n_primary)
        sd_s = sd_binomial("diff", d.outcomes_per_arm(replicates))
        row(
            f"H3 ({stratum}), true diff 0",
            n_s,
            sd_s,
            "noninferiority",
            h3_margin,
            lambda lv, n_s=n_s, sd_s=sd_s: noninferiority_power(
                n_s, sd_s, h3_margin, 0.0, lv
            ),
        )
    c2 = designs["c2+"]
    n_c2, sd_c2 = (
        c2.problems(n_primary),
        sd_binomial("vs_fixed", c2.outcomes_per_arm(replicates)),
    )
    row(
        "H4a (c2+), true diff 0",
        n_c2,
        sd_c2,
        "noninferiority",
        h4a_margin,
        lambda lv: noninferiority_power(n_c2, sd_c2, h4a_margin, 0.0, lv),
    )
    sw = designs["swap_primary"]
    n_sw, sd_sw = (
        sw.problems(n_primary),
        sd_binomial("did", sw.outcomes_per_arm(replicates)),
    )
    row("H4b (swap, primary)", n_sw, sd_sw, "two-sided", None, lambda lv: float("nan"))
    if draws is not None:
        sd_pooled = sd_binomial("diff", replicates * draws)
        row(
            "H3, all sets pooled (alternative), true diff 0",
            n_primary,
            sd_pooled,
            "noninferiority",
            h3_margin,
            lambda lv: noninferiority_power(n_primary, sd_pooled, h3_margin, 0.0, lv),
        )
        sd_noise = sd_binomial("vs_fixed", replicates * draws)
        row(
            "H1a, noise-only SD",
            n_primary,
            sd_noise,
            "superiority",
            h1_effect,
            lambda lv: superiority_power(n_primary, sd_noise, h1_effect, lv),
        )
    return rows


# ---- the sizing rule of PREREG section 6.3 ------------------------------------------

DRAW_OPTIONS = (4, 8, 16)
POWER_TARGET = 0.70
RULE_EFFECTS = {"H1a": 0.015, "H1b": 0.015, "H2": 0.10}
H4A_MARGIN = 0.01


def draws_by_rule(
    measured: Mapping[int, Mapping[str, tuple[float, float]]],
    target: float = POWER_TARGET,
    alpha: float = ALPHA,
    m: int = FAMILY1_SIZE,
) -> dict:
    """The number of evaluation draws per problem, by the pre-stated power rule.

    ``measured[d][h]`` is ``(problems, sd)`` of hypothesis h (H1a, H1b, H2, H4a) with
    d draws, as measured on dev with the untrained reader. The result is the smallest d
    for which every one of them has at least ``target`` power at the Holm worst case
    (``alpha / m``). If none does, it is the largest d, and the hypotheses still short
    of the target are listed as underpowered for the owner's decision at the freeze.
    """
    worst = alpha / m

    def power_of(name: str, n: float, sd: float) -> float:
        if name == "H4a":
            return noninferiority_power(n, sd, H4A_MARGIN, 0.0, worst)
        return superiority_power(n, sd, RULE_EFFECTS[name], worst)

    table = {
        d: {name: power_of(name, n, sd) for name, (n, sd) in measured[d].items()}
        for d in sorted(measured)
    }
    for d, powers in table.items():
        if all(p >= target for p in powers.values()):
            return {"draws": d, "underpowered": [], "power": table}
    last = max(table)
    short = [name for name, p in table[last].items() if p < target]
    return {"draws": last, "underpowered": short, "power": table}
