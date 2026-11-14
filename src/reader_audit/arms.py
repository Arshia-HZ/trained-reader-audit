"""Reader inputs: one prompt per arm for a given candidate set (PREREG D16).

Pure functions from a pool and a candidate set to chat messages. Nothing here calls a
model. Every arm that shows solutions renders through the reader's own template.

Every arm that shows candidate text records, in ``detail``, how many of the shown
candidates were cut by the generation limit (``n_gen_truncated``: no final answer, shown
as generated), how many were cut for display by the prompt cap (``n_prompt_clipped``),
and how many shown texts no longer yield their candidate's final answer
(``n_answer_hidden``). The two kinds of truncation are never confused (PREREG D14).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from channel_audit.channels import strip_confidence
from channel_audit.extract import extract_answer
from channel_audit.placebo import pick_donor

from reader_audit import reader_prompts as rp
from reader_audit.cand_sets import CandidateSet, draw_set, members
from reader_audit.controls import (
    Truncate,
    dedup_order,
    plurality_wrong_answer,
    swap_final_answer,
)
from reader_audit.pools import Candidate, ProblemPool
from reader_audit.subsets import group_answers

ARMS = rp.ARMS


@dataclass(frozen=True)
class ArmInput:
    arm: str
    messages: list[dict[str, str]]
    parent_keys: tuple[str, ...]  # generations whose text (or answer) the prompt shows
    detail: dict = field(default_factory=dict)


class DonorIndex:
    """Placebo donors: another problem of the same subject and level, nearest in length.

    A donor shows its own set of the same size and draw index, so a placebo prompt has
    the same number of solutions as the real one. Lengths are the capped candidate
    token counts; no outcome decides the choice.
    """

    def __init__(
        self, pools: Mapping[str, ProblemPool], cap: int | None = rp.CANDIDATE_CAP
    ) -> None:
        self._pools = pools
        self._cap = cap
        self._groups: dict[tuple[str, int], list] = {}
        for pool in pools.values():
            key = (pool.problem.subject, pool.problem.level)
            self._groups.setdefault(key, []).append(pool.problem)
        self._tokens: dict[tuple[int, int], dict[str, int]] = {}

    def _set_tokens(self, pool: ProblemPool, cset: CandidateSet) -> int:
        counts = [c.completion_tokens for c in members(pool, cset)]
        return sum(min(n, self._cap) if self._cap else n for n in counts)

    def _tokens_for(self, k: int, draw: int) -> dict[str, int]:
        if (k, draw) not in self._tokens:
            self._tokens[(k, draw)] = {
                pid: self._set_tokens(pool, draw_set(pool, k, draw))
                for pid, pool in self._pools.items()
                if pool.n >= k
            }
        return self._tokens[(k, draw)]

    def pick(
        self, pool: ProblemPool, cset: CandidateSet
    ) -> tuple[ProblemPool, CandidateSet]:
        problem = pool.problem
        donor = pick_donor(
            problem,
            self._groups[(problem.subject, problem.level)],
            self._set_tokens(pool, cset),
            self._tokens_for(cset.k, cset.draw),
            tiebreak=f"{cset.k}|{cset.draw}",
        )
        donor_pool = self._pools[donor.problem_id]
        return donor_pool, draw_set(donor_pool, cset.k, cset.draw)


@dataclass(frozen=True)
class SwapPlan:
    """How an answer swap applies to one candidate set (PREREG D25)."""

    target: str  # the wrong answer written into the correct candidates' final box
    correct_plurality: bool  # the correct answer was the strict plurality beforehand


def swap_plan(pool: ProblemPool, cset: CandidateSet) -> SwapPlan | None:
    """The swap of a set, or None if the set cannot be swapped.

    A set qualifies (the broad population) with at least two correct candidates and at
    least one wrong parsable answer. Candidates without a final answer are not wrong
    answers. The target is the most frequent wrong answer, answers pooled by the audit's
    equivalence; a tie goes to the answer shown first. The set is in the primary
    population when the correct answer is the *strict* plurality before the swap: more
    candidates than any wrong answer, so a tie with a wrong answer does not count.
    """
    if cset.n_correct < 2:
        return None
    cands = members(pool, cset)
    answers = [c.answer for c in cands]
    correct = [c.correct for c in cands]
    target = plurality_wrong_answer(answers, correct)
    if target is None:
        return None
    groups = group_answers(answers, correct)
    right = sum(s for s, ok in zip(groups.sizes, groups.correct, strict=True) if ok)
    wrong_top = max(
        (s for s, ok in zip(groups.sizes, groups.correct, strict=True) if not ok),
        default=0,
    )
    return SwapPlan(target, right > wrong_top)


def _show(
    cands: Sequence[Candidate],
    sources: Sequence[str],
    intended: Sequence[str | None],
    cap: int | None,
    truncate: Truncate | None,
) -> tuple[list[str], dict]:
    """Prepared texts for display and how the display cut them."""
    texts, clipped, hidden = [], 0, 0
    for source, want in zip(sources, intended, strict=True):
        shown = rp.prepare_candidate(source, cap, truncate)
        texts.append(shown)
        clipped += shown != strip_confidence(source)
        hidden += extract_answer(shown) != want
    stats = {
        "n_gen_truncated": sum(c.truncated for c in cands),
        "n_prompt_clipped": clipped,
        "n_answer_hidden": hidden,
    }
    return texts, stats


def build_arm_input(
    arm: str,
    pool: ProblemPool,
    cset: CandidateSet,
    *,
    donors: DonorIndex | None = None,
    cap: int | None = rp.CANDIDATE_CAP,
    truncate: Truncate | None = None,
) -> ArmInput | None:
    """The prompt of ``arm`` for ``cset``, or None when the arm does not apply."""
    if arm not in ARMS:
        raise ValueError(f"unknown reader arm {arm!r}; expected one of {ARMS}")
    question = pool.problem.question
    cands = members(pool, cset)
    keys = tuple(c.gen_key for c in cands)

    def show(chosen, sources=None, intended=None):
        return _show(
            chosen,
            sources if sources is not None else [c.text for c in chosen],
            intended if intended is not None else [c.answer for c in chosen],
            cap,
            truncate,
        )

    if arm == "candidate_free":
        return ArmInput(arm, rp.build_prompt(arm, question), ())
    if arm == "answer_only":
        msgs = rp.build_prompt(arm, question, answers=[c.answer for c in cands])
        return ArmInput(arm, msgs, keys)
    if arm == "reader_full":
        texts, stats = show(cands)
        return ArmInput(arm, rp.build_prompt(arm, question, texts=texts), keys, stats)
    if arm == "placebo":
        if donors is None:
            raise ValueError("placebo needs a DonorIndex")
        donor_pool, donor_set = donors.pick(pool, cset)
        shown = members(donor_pool, donor_set)
        texts, stats = show(shown)
        return ArmInput(
            arm,
            rp.build_prompt(arm, question, texts=texts),
            tuple(c.gen_key for c in shown),
            {"donor_problem": donor_pool.problem.problem_id, **stats},
        )
    if arm == "duplicate":
        texts, stats = show(cands[:1])
        return ArmInput(
            arm,
            rp.build_prompt(arm, question, texts=texts * cset.k),
            (keys[0],) * cset.k,
            stats,
        )
    if arm == "single_slot":
        texts, stats = show(cands[:1])
        slots = texts + [rp.EMPTY_SLOT] * (cset.k - 1)
        return ArmInput(
            arm, rp.build_prompt(arm, question, texts=slots), (keys[0],), stats
        )
    if arm == "dedup":
        order = dedup_order([c.answer for c in cands])
        texts, stats = show([cands[i] for i in order])
        return ArmInput(
            arm,
            rp.build_prompt(arm, question, texts=texts),
            tuple(keys[i] for i in order),
            {"n_shown": len(order), **stats},
        )
    # answer_swap
    plan = swap_plan(pool, cset)
    if plan is None:
        return None
    sources, intended, swapped = [], [], 0
    for c in cands:
        text, want = c.text, c.answer
        if c.correct and c.answer is not None:
            try:
                text, want = swap_final_answer(c.text, plan.target), plan.target
                swapped += 1
            except ValueError:  # no box to swap: the candidate is shown as it was
                pass
        sources.append(text)
        intended.append(want)
    texts, stats = show(cands, sources, intended)
    return ArmInput(
        arm,
        rp.build_prompt(arm, question, texts=texts),
        keys,
        {
            "swap_to": plan.target,
            "n_swapped": swapped,
            "correct_plurality": plan.correct_plurality,
            **stats,
        },
    )
