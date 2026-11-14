"""Reader inputs: one prompt per arm for a given candidate set (PREREG D16).

Pure functions from a pool and a candidate set to chat messages. Nothing here calls a
model. Every arm that shows solutions renders through the reader's own template.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from channel_audit.placebo import pick_donor

from reader_audit import reader_prompts as rp
from reader_audit.cand_sets import CandidateSet, draw_set, members
from reader_audit.controls import (
    Truncate,
    dedup_order,
    plurality_wrong_answer,
    swap_final_answer,
)
from reader_audit.pools import ProblemPool

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


def swap_eligible(pool: ProblemPool, cset: CandidateSet) -> str | None:
    """The wrong answer to swap in, or None if the set cannot be swapped.

    A set qualifies with at least two correct candidates and at least one wrong
    parsable answer; the swap target is the most frequent wrong answer.
    """
    if cset.n_correct < 2:
        return None
    cands = members(pool, cset)
    return plurality_wrong_answer([c.answer for c in cands], [c.correct for c in cands])


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

    def prep(text: str) -> str:
        return rp.prepare_candidate(text, cap, truncate)

    if arm == "candidate_free":
        return ArmInput(arm, rp.build_prompt(arm, question), ())
    if arm == "answer_only":
        msgs = rp.build_prompt(arm, question, answers=[c.answer for c in cands])
        return ArmInput(arm, msgs, keys)
    if arm == "reader_full":
        texts = [prep(c.text) for c in cands]
        return ArmInput(arm, rp.build_prompt(arm, question, texts=texts), keys)
    if arm == "placebo":
        if donors is None:
            raise ValueError("placebo needs a DonorIndex")
        donor_pool, donor_set = donors.pick(pool, cset)
        shown = members(donor_pool, donor_set)
        texts = [prep(c.text) for c in shown]
        return ArmInput(
            arm,
            rp.build_prompt(arm, question, texts=texts),
            tuple(c.gen_key for c in shown),
            {"donor_problem": donor_pool.problem.problem_id},
        )
    if arm == "duplicate":
        texts = [prep(cands[0].text)] * cset.k
        return ArmInput(
            arm, rp.build_prompt(arm, question, texts=texts), (keys[0],) * cset.k
        )
    if arm == "single_slot":
        texts = [prep(cands[0].text)] + [rp.EMPTY_SLOT] * (cset.k - 1)
        return ArmInput(arm, rp.build_prompt(arm, question, texts=texts), (keys[0],))
    if arm == "dedup":
        order = dedup_order([c.answer for c in cands])
        texts = [prep(cands[i].text) for i in order]
        return ArmInput(
            arm,
            rp.build_prompt(arm, question, texts=texts),
            tuple(keys[i] for i in order),
            {"n_shown": len(order)},
        )
    # answer_swap
    wrong = swap_eligible(pool, cset)
    if wrong is None:
        return None
    texts, swapped = [], 0
    for c in cands:
        text = c.text
        if c.correct and c.answer is not None:
            try:
                text = swap_final_answer(c.text, wrong)
                swapped += 1
            except ValueError:  # no box to swap: the candidate is shown as it was
                pass
        texts.append(prep(text))
    return ArmInput(
        arm,
        rp.build_prompt(arm, question, texts=texts),
        keys,
        {"swap_to": wrong, "n_swapped": swapped},
    )
