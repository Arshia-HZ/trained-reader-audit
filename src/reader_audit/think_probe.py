"""Exploratory, dev only (PREREG R22, E2): what the thinking-mode probe needs besides
the engine. The escalation slice (sets on which early-stopping SC over the shown order
finds no agreeing window), the hashed choice of problems, ESC continued over the rest of
a set's seeded order, the split of a thinking generation at ``</think>`` and its grade.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from channel_audit.extract import extract_final
from channel_audit.grade import is_correct

from reader_audit.allocation import esc_on_order
from reader_audit.cand_sets import _generator, draw_set
from reader_audit.pools import ProblemPool
from reader_audit.subsets import AnswerGroups, group_answers

SALT = "reader-audit/think-probe/v1"
THINK_CLOSE = "</think>"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def pool_groups(pool: ProblemPool) -> AnswerGroups:
    return group_answers(
        [c.answer for c in pool.candidates], [c.correct for c in pool.candidates]
    )


def escalated_draws(
    pool: ProblemPool, k: int, n_draws: int, window: int = 2
) -> list[int]:
    """Draws whose K-set ends ESC (``window``, cap K) without an agreeing window."""
    groups = pool_groups(pool)
    out = []
    for draw in range(n_draws):
        cset = draw_set(pool, k, draw)
        if not esc_on_order(groups, cset.positions, window, k)[2]:
            out.append(draw)
    return out


def pick_problems(ids: Sequence[str], n: int, salt: str = SALT) -> list[str]:
    """``n`` of the ids, ordered by a hash of each id (independent of the outcomes)."""
    key = {i: hashlib.sha256(f"{salt}|{i}".encode()).hexdigest() for i in ids}
    return sorted(ids, key=key.__getitem__)[:n]


def seed_for(
    problem_id: str, draw: int, arm: str, replicate: int, salt: str = SALT
) -> int:
    digest = hashlib.sha256(f"{salt}|{problem_id}|{draw}|{arm}|{replicate}".encode())
    return int.from_bytes(digest.digest()[:4], "big")


def esc_continued(
    pool: ProblemPool, k: int, draw: int, window: int = 2, cap: int = 16
) -> tuple[int, float]:
    """ESC over the set's whole seeded order: its K shown samples, then the rest of the
    same permutation, up to ``cap``: (samples drawn, credit)."""
    order = [
        int(i) for i in _generator(pool.problem.problem_id, k, draw).permutation(pool.n)
    ]
    if tuple(order[:k]) != draw_set(pool, k, draw).positions:
        raise AssertionError("the seeded order does not start with the shown set")
    drawn, credit, _ = esc_on_order(pool_groups(pool), order, window, min(cap, pool.n))
    return drawn, credit


@dataclass(frozen=True)
class ThinkingGrade:
    closed: bool  # the generation reached </think>
    think_text: str
    answer_text: str
    extracted: str | None
    status: str  # boxed, none, truncated, or unclosed (thinking cut)
    correct: bool


def grade_thinking(text: str, finish_reason: str, gold: str) -> ThinkingGrade:
    """Grade only what follows ``</think>``; thinking that never closes is wrong."""
    if THINK_CLOSE not in text:
        return ThinkingGrade(False, text, "", None, "unclosed", False)
    think, answer = text.split(THINK_CLOSE, 1)
    extraction = extract_final(answer, finish_reason)
    return ThinkingGrade(
        True,
        think,
        answer,
        extraction.answer,
        extraction.status,
        is_correct(extraction.answer, gold),
    )
