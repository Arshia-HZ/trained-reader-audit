"""Exploratory, dev only (PREREG R26, E4): the thinking read against the fresh thinking
solve, split by how many of the shown candidates are correct.

Each read of the E3 pilot is paired with the mean of its problem's fresh thinking
solves. The stratum (c0, c1, c2+) counts the correct candidates among the shown set and
is a label from candidate correctness, never from the read's outcome. The quantities are
credit at the read's cap, thinking length, a lexical density of epistemic words in the
thinking text, and whether the final answer equals an answer the set showed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from channel_audit.grade import answers_equivalent

from reader_audit import allocation as al
from reader_audit.cand_sets import STRATA, draw_set, members
from reader_audit.pools import ProblemPool

READ_CAP = 16384
EPISTEMIC = re.compile(
    r"\b(?:wait|hmm|alternatively|verify|double-check|let me check|but)\b",
    re.IGNORECASE,
)
THINK_CLOSE = "</think>"
QUANTITIES = ("credit", "think_tokens", "density", "copy")


@dataclass(frozen=True)
class ThinkingRecord:
    """One thinking generation of the pilot, as needed here."""

    problem_id: str
    arm: str
    draw: int
    correct: bool
    answer: str | None
    think_tokens: int | None  # tokens before </think>; None when unclosed
    prompt_tokens: int
    completion_tokens: int
    thinking: str  # the text before </think> (all of it when unclosed)

    @classmethod
    def from_row(cls, row: Mapping) -> ThinkingRecord:
        return cls(
            row["problem_id"],
            row["arm"],
            int(row["draw"]),
            bool(row["correct"]),
            row["extracted"],
            row["think_tokens"],
            int(row["prompt_tokens"]),
            int(row["completion_tokens"]),
            row["text"].split(THINK_CLOSE)[0],
        )


def thinking_length(r: ThinkingRecord) -> int:
    return r.completion_tokens if r.think_tokens is None else r.think_tokens


def epistemic_density(text: str) -> float:
    """Epistemic-word matches per 1,000 whitespace words (0 for an empty text)."""
    words = len(text.split())
    return 1000 * len(EPISTEMIC.findall(text)) / words if words else 0.0


def credit_at(r: ThinkingRecord, cap: int = READ_CAP) -> float:
    return float(r.correct and r.completion_tokens <= cap)


def copies(answer: str | None, shown: Sequence[str | None]) -> bool:
    """Whether ``answer`` equals a ``shown`` answer, by the grader's equivalence."""
    if answer is None:
        return False
    return any(s is not None and answers_equivalent(answer, s) for s in shown)


def read_rows(
    pools: Mapping[str, ProblemPool],
    fresh: Mapping[str, Sequence[ThinkingRecord]],
    reads: Iterable[ThinkingRecord],
    k: int,
) -> list[dict]:
    """One row per read with a fresh solve of its problem: the read's values, the means
    of the fresh solves, and the set's stratum."""
    rows = []
    for r in reads:
        solves = fresh.get(r.problem_id)
        if not solves:
            continue
        pool = pools[r.problem_id]
        cset = draw_set(pool, k, r.draw)
        shown = [c.answer for c in members(pool, cset)]
        rows.append(
            {
                "problem_id": r.problem_id,
                "stratum": cset.stratum,
                "read": {
                    "credit": credit_at(r),
                    "think_tokens": float(thinking_length(r)),
                    "density": epistemic_density(r.thinking),
                    "copy": float(copies(r.answer, shown)),
                },
                "fresh": {
                    "credit": float(np.mean([credit_at(s) for s in solves])),
                    "think_tokens": float(
                        np.mean([thinking_length(s) for s in solves])
                    ),
                    "density": float(
                        np.mean([epistemic_density(s.thinking) for s in solves])
                    ),
                    "copy": float(np.mean([copies(s.answer, shown) for s in solves])),
                },
            }
        )
    return rows


def strata_table(rows: Sequence[Mapping], n_boot: int, seed: int) -> dict:
    """Per stratum and pooled ("all"): counts, and for each quantity the mean over
    problems of read, fresh and read minus fresh, with a bootstrap over problems."""
    out = {}
    for name in (*STRATA, "all"):
        mine = [r for r in rows if name == "all" or r["stratum"] == name]
        by_pid: dict[str, list[Mapping]] = {}
        for r in mine:
            by_pid.setdefault(r["problem_id"], []).append(r)
        cell: dict = {"n_reads": len(mine), "n_problems": len(by_pid)}
        for q in QUANTITIES:
            read = {
                p: float(np.mean([r["read"][q] for r in rs]))
                for p, rs in by_pid.items()
            }
            fresh = {
                p: float(np.mean([r["fresh"][q] for r in rs]))
                for p, rs in by_pid.items()
            }
            diff = al.mean_ci({p: read[p] - fresh[p] for p in by_pid}, n_boot, seed)
            cell[q] = {
                "read": float(np.mean(list(read.values()))) if read else float("nan"),
                "fresh": float(np.mean(list(fresh.values())))
                if fresh
                else float("nan"),
                "diff": diff["mean"],
                "ci": diff["ci"],
            }
        out[name] = cell
    return out
