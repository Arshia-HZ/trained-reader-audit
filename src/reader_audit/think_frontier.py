"""Exploratory, dev only (PREREG R24, E3): the thinking pilot read as policies on one
cost-accuracy frontier.

A thinking solve's credit and cost at a lower cap are exact from its own token count: a
generation that ended within the cap is the same generation, and one that did not is cut
there and wrong. Costs are in samples of the problem's non-thinking pool under the two
rules of D3. The comparison envelope pools, across problems, plain SC and ESC (D22), one
thinking solve at several caps and the thinking vote of 2 and of 4 solves; a policy is
read as its pooled credit minus that envelope at its pooled cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from channel_audit.aggregate import plurality_credit

from reader_audit import allocation as al
from reader_audit.cand_sets import draw_set, members
from reader_audit.pools import ProblemPool
from reader_audit.subsets import group_answers

THINK_CAPS = (4096, 8192, 16384, 30000)
VOTE_SIZES = (2, 4)
CASCADE_ESC_CAPS = (2, 4, 8)
CASCADE_THINK_CAPS = (16384, 30000)


@dataclass(frozen=True)
class Solve:
    """One thinking generation (a fresh solve or a read)."""

    correct: bool
    answer: str | None
    prompt_tokens: int
    completion_tokens: int


def at_cap(s: Solve, cap: int) -> tuple[float, int]:
    """(credit, generated tokens) of the solve had its cap been ``cap``."""
    if s.completion_tokens <= cap:
        return float(s.correct), s.completion_tokens
    return 0.0, cap


def solve_cost(s: Solve, cap: int, units: Mapping[str, float]) -> dict[str, float]:
    gen = at_cap(s, cap)[1]
    return {
        "generated": gen / units["generated"],
        "prompt_inclusive": (s.prompt_tokens + gen) / units["prompt_inclusive"],
    }


def think_point(
    solves: Sequence[Solve], cap: int, units: Mapping[str, float], rule: str
) -> tuple[float, float]:
    """(cost, credit) of one thinking solve at ``cap``, averaged over the solves."""
    cost = np.mean([solve_cost(s, cap, units)[rule] for s in solves])
    credit = np.mean([at_cap(s, cap)[0] for s in solves])
    return float(cost), float(credit)


def think_vote(
    solves: Sequence[Solve], m: int, gold: str, units: Mapping[str, float], rule: str
) -> tuple[float, float]:
    """(cost, credit) of the plurality of ``m`` solves, averaged over all m-subsets."""
    cap = max(THINK_CAPS)
    groups = list(combinations(range(len(solves)), m))
    credit = np.mean(
        [plurality_credit([solves[i].answer for i in g], gold) for g in groups]
    )
    cost = m * np.mean([solve_cost(s, cap, units)[rule] for s in solves])
    return float(cost), float(credit)


def set_prefix_costs(pool: ProblemPool, k: int, draw: int) -> tuple[dict, ...]:
    """Cost of the first i + 1 shown candidates of a set, per rule."""
    units = al.unit_costs(pool)
    running = {r: 0.0 for r in al.RULES}
    out = []
    for c in members(pool, draw_set(pool, k, draw)):
        running = {
            "generated": running["generated"]
            + c.completion_tokens / units["generated"],
            "prompt_inclusive": running["prompt_inclusive"]
            + (c.prompt_tokens + c.completion_tokens) / units["prompt_inclusive"],
        }
        out.append(dict(running))
    return tuple(out)


def think_cascade(
    pool: ProblemPool,
    k: int,
    draws: int,
    esc_cap: int,
    escalate: tuple[float, Mapping[str, float]],
    rule: str,
) -> tuple[float, float, float]:
    """ESC (window 2, ``esc_cap``) over each seeded set's shown order; on agreement its
    plurality, otherwise the escalation ``(credit, cost per rule)``. Returns the mean
    (cost, credit, share escalated) over the draws."""
    groups = group_answers(
        [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
    )
    credit, cost, up = [], [], []
    for draw in range(draws):
        cset = draw_set(pool, k, draw)
        prefix = set_prefix_costs(pool, k, draw)
        drawn, c, agreed = al.esc_on_order(groups, cset.positions, 2, esc_cap)
        if agreed:
            credit.append(c)
            cost.append(prefix[drawn - 1][rule])
            up.append(0.0)
        else:
            credit.append(escalate[0])
            cost.append(prefix[drawn - 1][rule] + escalate[1][rule])
            up.append(1.0)
    return float(np.mean(cost)), float(np.mean(credit)), float(np.mean(up))


def escalation_by_solve(
    solves: Sequence[Solve], cap: int, units: Mapping[str, float]
) -> tuple[float, dict[str, float]]:
    """The expected credit and cost of one thinking solve, over a problem's solves."""
    credit = float(np.mean([at_cap(s, cap)[0] for s in solves]))
    costs = [solve_cost(s, cap, units) for s in solves]
    return credit, {r: float(np.mean([c[r] for c in costs])) for r in al.RULES}


def cascade_oracles(
    pool: ProblemPool,
    k: int,
    draws: int,
    solves: Sequence[Solve],
    cap: int,
    rule: str,
) -> dict[str, tuple[float, float]]:
    """(cost, credit) of ESC (window 2, cap 2) then one thinking solve, and of three
    oracles over it (PREREG R26): ``gate`` accepts the two samples' plurality exactly
    where its credit is at least the problem's mean thinking credit; ``agree`` accepts
    only an agreeing pair and only where that is no worse; ``mode`` takes, per problem,
    one sample or one thinking solve, whichever has more credit (ties to the sample)."""
    units = al.unit_costs(pool)
    t_credit, t_costs = escalation_by_solve(solves, cap, units)
    think = (t_costs[rule], t_credit)
    groups = group_answers(
        [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
    )
    rows: dict[str, list[tuple[float, float]]] = {"dart": [], "gate": [], "agree": []}
    for draw in range(draws):
        prefix = set_prefix_costs(pool, k, draw)
        drawn, c, agreed = al.esc_on_order(
            groups, draw_set(pool, k, draw).positions, 2, 2
        )
        base = prefix[drawn - 1][rule]
        accept, escalate = (base, c), (base + think[0], think[1])
        rows["dart"].append(accept if agreed else escalate)
        rows["gate"].append(accept if c >= t_credit else escalate)
        rows["agree"].append(accept if agreed and c >= t_credit else escalate)
    out = {name: tuple(map(float, np.mean(r, axis=0))) for name, r in rows.items()}
    one = (1.0, float(np.mean([c.correct for c in pool.candidates])))
    out["mode"] = one if one[1] >= think[1] else think
    return out


def pass_at(solves: Sequence[Solve], m: int) -> float:
    """The share of m-subsets of the solves with at least one correct, averaged."""
    groups = list(combinations(range(len(solves)), m))
    return float(np.mean([any(solves[i].correct for i in g) for g in groups]))


def read_vs_fresh(
    reads: Mapping[int, Solve], solves: Sequence[Solve], cap: int
) -> float | None:
    """A problem's thinking read minus its fresh thinking solve, both at ``cap``, over
    the sets that have a read; None if none has."""
    if not reads:
        return None
    fresh = np.mean([at_cap(s, cap)[0] for s in solves])
    return float(np.mean([at_cap(r, cap)[0] for r in reads.values()]) - fresh)


def minus_envelope(
    policy: Mapping[str, tuple[float, float]],
    envelope: Mapping[str, Mapping[str, tuple[float, float]]],
    pids: Sequence[str],
    boots: np.ndarray,
) -> dict:
    """Pooled credit of ``policy`` (pid -> (cost, credit)) minus the pooled envelope of
    the ``envelope`` methods (method -> pid -> (cost, credit)) at the same cost."""
    acc = np.array([policy[p][1] for p in pids])
    cost = np.array([policy[p][0] for p in pids])
    methods = sorted(envelope)
    base_cost = np.array([[envelope[m][p][0] for p in pids] for m in methods])
    base_acc = np.array([[envelope[m][p][1] for p in pids] for m in methods])
    full = al._minus_frontier(acc, cost, base_cost, base_acc, np.arange(len(pids)))
    boot = np.array(
        [al._minus_frontier(acc, cost, base_cost, base_acc, b) for b in boots]
    )
    return {
        "cost": float(cost.mean()),
        "credit": float(acc.mean()),
        "minus_envelope": float(full),
        "ci": al._ci(boot),
        "estimable": bool(np.isfinite(full)),
    }
