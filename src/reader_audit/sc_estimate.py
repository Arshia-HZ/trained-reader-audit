"""Self-consistency credit for pools too large to enumerate (PREREG D23).

The dev pools hold 18 samples and ``subsets`` computes their SC credit exactly. A test
pool holds 32, and enumerating the answer-count vectors of a diverse level-5 problem
takes seconds per (problem, k) (0.3 s at k = 8 and 2.3 s at k = 16 for 14 answer groups,
measured), hours for a set. So the test sets use this Monte Carlo estimate, with every
parameter fixed here:

- R = 2,000 subsets of k samples, drawn uniformly without replacement, for each
  problem and each integer k = 1..n; the generator is seeded from sha256 of the salt,
  the problem id and k, so a re-run, and every arm and model, uses the same subsets;
- the credit of a subset is the plurality credit (unparsable answers never vote, an
  m-way tie containing the right answer earns 1/m, answers pooled by the audit's
  equivalence), averaged over the subsets;
- k = n is the whole pool, so its credit is exact;
- at a fractional k* the credit is interpolated linearly between the estimates at the
  integers around it; a k* outside [1, n] is "not estimable".

The standard error of one estimate is at most 0.5 / sqrt(R) = 0.011; it is independent
across problems, so on a mean over about a thousand problems it is under 0.04 points and
the bootstrap over problems does not model it.
"""

from __future__ import annotations

import hashlib

import numpy as np

from reader_audit.subsets import AnswerGroups, credit_from_votes, interpolate

SC_DRAWS = 2000
SC_SALT = "reader-audit/sc/v1"


def _generator(problem_id: str, k: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{SC_SALT}|{problem_id}|{k}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _indicator(groups: AnswerGroups) -> np.ndarray:
    """(samples, answer groups) one-hot; an unparsable sample is a row of zeros."""
    onehot = np.zeros((groups.n, len(groups.sizes)))
    for i, label in enumerate(groups.labels):
        if label is not None:
            onehot[i, label] = 1.0
    return onehot


def sc_credit_mc(
    groups: AnswerGroups, k: int, problem_id: str, n_draws: int = SC_DRAWS
) -> float:
    """Estimated plurality credit of k samples drawn from the problem's pool."""
    n = groups.n
    if not 1 <= k <= n:
        raise ValueError(f"k={k} outside [1, {n}]")
    if len(groups.labels) != n:
        raise ValueError("groups must be built from samples (they carry labels)")
    if not groups.sizes:
        return 0.0
    onehot = _indicator(groups)
    correct = np.asarray(groups.correct)
    if k == n:  # the only subset is the whole pool
        return float(credit_from_votes(onehot.sum(axis=0)[None, :], correct)[0])
    rng = _generator(problem_id, k)
    chosen = np.argsort(rng.random((n_draws, n)), axis=1)[:, :k]
    members = np.zeros((n_draws, n))
    np.put_along_axis(members, chosen, 1.0, axis=1)
    return float(credit_from_votes(members @ onehot, correct).mean())


def sc_curve_mc(
    groups: AnswerGroups, problem_id: str, n_draws: int = SC_DRAWS
) -> np.ndarray:
    """Estimated SC credit at k = 1..n samples (index k - 1)."""
    return np.array(
        [sc_credit_mc(groups, k, problem_id, n_draws) for k in range(1, groups.n + 1)]
    )


def sc_at_cost(curve: np.ndarray, k_star: float) -> float | None:
    """The curve read at a fractional sample count, or None if it is out of range."""
    if not 1 <= k_star <= len(curve):
        return None
    return interpolate(curve, k_star)


# ---- early-stopping SC on the test sets (PREREG D23) ------------------------------

ESC_DRAWS = 500
ESC_SALT = "reader-audit/esc/v1"


def esc_seed(problem_id: str, window: int, max_samples: int) -> int:
    """The seed of an ESC estimate: the salt, the problem and the ESC setting."""
    key = f"{ESC_SALT}|{problem_id}|{window}|{max_samples}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def esc_estimate(
    groups: AnswerGroups,
    problem_id: str,
    window: int,
    max_samples: int,
    n_orders: int = ESC_DRAWS,
) -> tuple[float, float]:
    """(mean samples drawn, mean credit) of ESC over ``n_orders`` seeded orders."""
    from reader_audit.baselines import esc_means

    return esc_means(
        groups, window, max_samples, n_orders, esc_seed(problem_id, window, max_samples)
    )
