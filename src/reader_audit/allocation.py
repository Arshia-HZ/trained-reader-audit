"""Exploratory, dev only (PREREG R20, E1): where a reader gains or loses against the
vote of its own K candidates, and what a read spent only on disagreement costs and
scores.

A set's agreement is the size of its plurality among the K shown answers (unparsable
answers never vote), which a deployed system can see; the stratum needs the gold answer
and only splits the analysis. Costs are in samples of the problem's own pool, under the
two rules of D3: generated tokens only, and prompt plus generated tokens. A policy is
compared with plain SC at its own cost on each problem (as H1a), and with the pooled
frontier of plain SC and early-stopping SC (D22) at the same mean cost, which is what H6
tests; intervals come from a bootstrap over problems.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from channel_audit.aggregate import plurality
from channel_audit.grade import answers_equivalent

from reader_audit.baselines import _plurality_credit, esc_means, frontier_at
from reader_audit.cand_sets import draw_set, members, vote_credit
from reader_audit.pools import ProblemPool
from reader_audit.subsets import AnswerGroups, group_answers, interpolate, sc_curve

RULES = ("generated", "prompt_inclusive")
ESC_SETTINGS = ((2, 8), (2, 16), (4, 8), (4, 16), (8, 8), (8, 16))
CASCADE_WINDOWS = (2, 4)
GATE_THRESHOLDS = (5, 6, 7)
AGREEMENT_BINS = ("8", "7", "6", "5", "4", "<=3")


@dataclass(frozen=True)
class Read:
    """One stored generation of a reader (with or without the candidates)."""

    correct: bool
    answer: str | None
    prompt_tokens: int
    completion_tokens: int


@lru_cache(maxsize=200_000)
def _same(a: str, b: str) -> bool:
    return answers_equivalent(a, b)


def _in(answer: str | None, others: Sequence[str | None]) -> bool:
    return answer is not None and any(
        o is not None and _same(answer, o) for o in others
    )


def unit_costs(pool: ProblemPool) -> dict[str, float]:
    """Tokens of one pool sample on average, per rule: the unit costs are counted in."""
    gen = float(np.mean([c.completion_tokens for c in pool.candidates]))
    full = float(
        np.mean([c.prompt_tokens + c.completion_tokens for c in pool.candidates])
    )
    return {"generated": gen, "prompt_inclusive": full}


def _cost(prompt: int, completion: int, units: Mapping[str, float]) -> dict[str, float]:
    return {
        "generated": completion / units["generated"],
        "prompt_inclusive": (prompt + completion) / units["prompt_inclusive"],
    }


@dataclass(frozen=True)
class SetView:
    """One seeded K-set of a problem with the reads a run made of it."""

    problem_id: str
    draw: int
    positions: tuple[int, ...]
    stratum: str
    top: int  # votes of the plurality group among the K shown answers
    unanimous: bool  # every shown answer parsable and the same
    vote: float  # plurality credit of the K shown answers
    prefix_cost: tuple[
        dict[str, float], ...
    ]  # cost of the first i + 1 shown candidates
    reads: tuple[Read, ...]
    frees: tuple[Read, ...]
    follows: float  # share of reads whose answer is in the plurality
    shown: float  # share of reads whose answer equals some shown answer
    read_cost: dict[str, float]  # mean cost of one read, in pool samples

    @property
    def read(self) -> float:
        return float(np.mean([r.correct for r in self.reads]))

    @property
    def free(self) -> float | None:
        return float(np.mean([r.correct for r in self.frees])) if self.frees else None

    @property
    def agreement_bin(self) -> str:
        return str(self.top) if self.top >= 4 else "<=3"


def build_views(
    pool: ProblemPool,
    k: int,
    reads: Mapping[int, Sequence[Read]],
    frees: Mapping[int, Sequence[Read]] | None = None,
) -> list[SetView]:
    """The views of the draws that have reads (``reads`` maps draw -> its reads)."""
    units = unit_costs(pool)
    out = []
    for draw in sorted(reads):
        if not reads[draw]:
            continue
        cset = draw_set(pool, k, draw)
        shown = members(pool, cset)
        answers = [c.answer for c in shown]
        p = plurality(answers)
        prefix, running = [], {"generated": 0.0, "prompt_inclusive": 0.0}
        for c in shown:
            step = _cost(c.prompt_tokens, c.completion_tokens, units)
            running = {r: running[r] + step[r] for r in RULES}
            prefix.append(dict(running))
        rs = tuple(reads[draw])
        read_costs = [_cost(r.prompt_tokens, r.completion_tokens, units) for r in rs]
        out.append(
            SetView(
                problem_id=cset.problem_id,
                draw=draw,
                positions=cset.positions,
                stratum=cset.stratum,
                top=p.top_count,
                unanimous=all(a is not None for a in answers)
                and p.n_tied == 1
                and p.top_count == k,
                vote=vote_credit(pool, cset),
                prefix_cost=tuple(prefix),
                reads=rs,
                frees=tuple((frees or {}).get(draw, ())),
                follows=float(np.mean([_in(r.answer, p.winners) for r in rs])),
                shown=float(np.mean([_in(r.answer, answers) for r in rs])),
                read_cost={
                    rule: float(np.mean([c[rule] for c in read_costs]))
                    for rule in RULES
                },
            )
        )
    return out


# ---- policies: what one set scores and costs ---------------------------------------


def esc_on_order(
    groups: AnswerGroups, order: Sequence[int], window: int, cap: int
) -> tuple[int, float, bool]:
    """ESC over the samples in ``order``: (samples drawn, credit, stopped by agreement).

    The same rule as ``baselines.early_stop_vote``: windows of ``window`` samples, stop
    after a full window whose samples share one parsable answer, or at ``cap``.
    """
    if window < 1 or cap < window:
        raise ValueError("need 1 <= window <= cap")
    counts = [0] * len(groups.sizes)
    drawn, agreed = 0, False
    while drawn < min(cap, len(order)):
        chunk = [groups.labels[i] for i in order[drawn : drawn + window]]
        chunk = chunk[: cap - drawn]
        for label in chunk:
            if label is not None:
                counts[label] += 1
        drawn += len(chunk)
        if len(chunk) == window and chunk[0] is not None and len(set(chunk)) == 1:
            agreed = True
            break
    return drawn, _plurality_credit(counts, groups.correct), agreed


def _with_read(view: SetView, rule: str) -> tuple[float, float]:
    return view.read, view.prefix_cost[-1][rule] + view.read_cost[rule]


def _vote_only(view: SetView, rule: str) -> tuple[float, float]:
    return view.vote, view.prefix_cost[-1][rule]


def policies(groups: AnswerGroups) -> dict[str, Callable[[SetView, str], tuple]]:
    """Each policy maps (view, rule) to (credit, cost in samples, read or not)."""

    def always_read(v, rule):
        return (*_with_read(v, rule), True)

    def vote_k(v, rule):
        return (*_vote_only(v, rule), False)

    def unanimity_gate(v, rule):
        return (*_vote_only(v, rule), False) if v.unanimous else always_read(v, rule)

    def threshold_gate(t):
        def gate(v, rule):
            return (*_vote_only(v, rule), False) if v.top >= t else always_read(v, rule)

        return gate

    def cascade(window):
        def run(v, rule):
            drawn, credit, agreed = esc_on_order(
                groups, v.positions, window, len(v.positions)
            )
            if agreed:
                return credit, v.prefix_cost[drawn - 1][rule], False
            return always_read(v, rule)

        return run

    out = {
        "always_read": always_read,
        "vote_k": vote_k,
        "unanimity_gate": unanimity_gate,
    }
    out.update({f"gate_top>={t}": threshold_gate(t) for t in GATE_THRESHOLDS})
    out.update({f"esc_w{w}_then_read": cascade(w) for w in CASCADE_WINDOWS})
    return out


# ---- baselines on the pools ---------------------------------------------------------


def baseline_points(
    pool: ProblemPool, n_perm: int = 400, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """(cost in samples, credit) of plain SC at every k and of each ESC setting."""
    groups = group_answers(
        [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
    )
    curve = sc_curve(groups)
    points = {
        f"sc_{k}": (float(k), float(curve[k - 1])) for k in range(1, len(curve) + 1)
    }
    for window, cap in ESC_SETTINGS:
        if cap <= groups.n:
            points[f"esc_w{window}_m{cap}"] = esc_means(
                groups, window, cap, n_perm, seed
            )
    return points


# ---- summaries ----------------------------------------------------------------------


def _ci(boot: np.ndarray) -> list[float]:
    good = boot[np.isfinite(boot)]
    if not len(good):
        return [float("nan"), float("nan")]
    return [float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))]


def per_problem(
    views: Sequence[SetView],
    value: Callable[[SetView], float | None],
    keep: Callable[[SetView], bool] = lambda v: True,
) -> dict[str, float]:
    """The mean of ``value`` over a problem's kept sets, per problem that has one."""
    acc: dict[str, list[float]] = {}
    for v in views:
        if keep(v):
            x = value(v)
            if x is not None:
                acc.setdefault(v.problem_id, []).append(x)
    return {pid: float(np.mean(xs)) for pid, xs in acc.items()}


def mean_ci(values: Mapping[str, float], n_boot: int, seed: int) -> dict:
    xs = np.array(list(values.values()), dtype=float)
    if not len(xs):
        return {"n": 0, "mean": float("nan"), "ci": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    boot = xs[rng.integers(0, len(xs), (n_boot, len(xs)))].mean(axis=1)
    return {"n": len(xs), "mean": float(xs.mean()), "ci": _ci(boot)}


def _minus_frontier(acc, cost, base_cost, base_acc, idx) -> float:
    """Pooled credit of the problems ``idx`` minus the SC + ESC frontier at their pooled
    cost; nan when that cost is outside the frontier's range."""
    points = list(
        zip(base_cost[:, idx].mean(axis=1), base_acc[:, idx].mean(axis=1), strict=True)
    )
    try:
        return float(acc[idx].mean() - frontier_at(points, cost[idx].mean()))
    except ValueError:
        return float("nan")


def policy_table(
    views_by_problem: Mapping[str, Sequence[SetView]],
    pools: Mapping[str, ProblemPool],
    baselines: Mapping[str, Mapping[str, tuple[float, float]]],
    n_boot: int,
    seed: int,
) -> dict:
    """Every policy under both rules: mean cost and credit, its gain over plain SC at
    its own cost on each problem, and its gain over the pooled SC + ESC frontier."""
    pids = sorted(views_by_problem)
    methods = sorted(set.intersection(*(set(baselines[p]) for p in pids)))
    base_cost = np.array([[baselines[p][m][0] for p in pids] for m in methods])
    base_acc = np.array([[baselines[p][m][1] for p in pids] for m in methods])
    curves = [
        [baselines[p][f"sc_{k}"][1] for k in range(1, pools[p].n + 1)] for p in pids
    ]
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(pids), (n_boot, len(pids)))
    out: dict = {}
    for rule in RULES:
        out[rule] = {}
        per_policy: dict[str, list[tuple[float, float, float]]] = {}
        for pid in pids:
            groups = group_answers(
                [c.answer for c in pools[pid].candidates],
                [c.correct for c in pools[pid].candidates],
            )
            for name, policy in policies(groups).items():
                cells = [policy(v, rule) for v in views_by_problem[pid]]
                per_policy.setdefault(name, []).append(
                    (
                        float(np.mean([c[0] for c in cells])),
                        float(np.mean([c[1] for c in cells])),
                        float(np.mean([c[2] for c in cells])),
                    )
                )
        for name, rows in per_policy.items():
            acc = np.array([r[0] for r in rows])
            cost = np.array([r[1] for r in rows])
            sc_gain = [
                a - interpolate(cv, max(c, 1.0))
                for a, c, cv in zip(acc, cost, curves, strict=True)
                if c <= len(cv)
            ]
            base = (base_cost, base_acc)
            full = _minus_frontier(acc, cost, *base, np.arange(len(pids)))
            boot = np.array([_minus_frontier(acc, cost, *base, b) for b in boots])
            out[rule][name] = {
                "cost": float(cost.mean()),
                "credit": float(acc.mean()),
                "read_share": float(np.mean([r[2] for r in rows])),
                "minus_sc": {
                    "n": len(sc_gain),
                    "not_estimable": len(pids) - len(sc_gain),
                    "mean": float(np.mean(sc_gain)) if sc_gain else float("nan"),
                    "sd": float(np.std(sc_gain, ddof=1))
                    if len(sc_gain) > 1
                    else float("nan"),
                },
                "minus_frontier": {
                    "mean": float(full),
                    "ci": _ci(boot),
                    "estimable": bool(np.isfinite(full)),
                },
            }
    return out


def _where(attr: str, value: str) -> Callable[[SetView], bool]:
    return lambda v: getattr(v, attr) == value


def stratum_table(views: Sequence[SetView], n_boot: int, seed: int) -> dict:
    """Raw credit per arm and the paired differences, per stratum and over all sets."""
    out = {}
    for name, keep in [("all", lambda v: True)] + [
        (s, _where("stratum", s)) for s in ("c0", "c1", "c2+")
    ]:
        has_free = any(v.frees for v in views if keep(v))
        row = {
            "sets": sum(1 for v in views if keep(v)),
            "read": mean_ci(per_problem(views, lambda v: v.read, keep), n_boot, seed),
            "vote": mean_ci(per_problem(views, lambda v: v.vote, keep), n_boot, seed),
            "read_minus_vote": mean_ci(
                per_problem(views, lambda v: v.read - v.vote, keep), n_boot, seed
            ),
            "follows_plurality": mean_ci(
                per_problem(views, lambda v: v.follows, keep), n_boot, seed
            ),
            "equals_a_shown_answer": mean_ci(
                per_problem(views, lambda v: v.shown, keep), n_boot, seed
            ),
        }
        if has_free:
            row["free"] = mean_ci(
                per_problem(views, lambda v: v.free, keep), n_boot, seed
            )
            row["read_minus_free"] = mean_ci(
                per_problem(
                    views, lambda v: None if v.free is None else v.read - v.free, keep
                ),
                n_boot,
                seed,
            )
        out[name] = row
    return out


def agreement_table(views: Sequence[SetView], n_boot: int, seed: int) -> dict:
    """Reader minus vote by the size of the plurality among the K shown answers."""
    total = len(views)
    out = {}
    for b in AGREEMENT_BINS:
        keep = _where("agreement_bin", b)
        n = sum(1 for v in views if keep(v))
        out[b] = {
            "share_of_sets": n / total if total else float("nan"),
            "vote": mean_ci(per_problem(views, lambda v: v.vote, keep), n_boot, seed),
            "read": mean_ci(per_problem(views, lambda v: v.read, keep), n_boot, seed),
            "read_minus_vote": mean_ci(
                per_problem(views, lambda v: v.read - v.vote, keep), n_boot, seed
            ),
        }
    return out
