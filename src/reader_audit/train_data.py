"""Training prompts and the reward of the GRPO runs (PREREG sections 4 and 9, R15).

Reader rows: for each training problem with a pool, candidate sets with K in {2, 4, 8}
are drawn (seeded, the order of the draw is the order shown, no candidate is marked as
the reader's own), labelled c0, c1 or c2+ by the number of correct candidates, and
rendered with the same template, cap and clipper as the evaluated reader. Every hard set
(c0 and c1) is kept and a seeded share of the easy sets (c2+), as in AggLM; the
no-c0 arm drops the c0 sets. Solver rows hold the plain question under the round-0
prompt, which is what the solver-GRPO baseline trains on.

The reward is the audit's own grade: the answer extracted from the completion with
``extract_answer`` (a complete box counts even if the completion was cut) and compared
with ``is_correct``; anything else is 0. It is the function that writes ``correct`` in
every ``graded.jsonl``, so a reward of 1 and "correct at evaluation" cannot differ.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence

from channel_audit.channels import build_round0_prompt
from channel_audit.extract import extract_answer
from channel_audit.grade import is_correct

from reader_audit import reader_prompts as rp
from reader_audit.arms import build_arm_input
from reader_audit.cand_sets import CandidateSet, draw_set
from reader_audit.controls import Truncate
from reader_audit.pools import ProblemPool

TRAIN_KS: tuple[int, ...] = (2, 4, 8)
HARD = ("c0", "c1")
SELECT_SALT = "reader-audit/train-select/v1"


def _keep_easy(problem_id: str, k: int, draw: int, share: float, seed: int) -> bool:
    digest = hashlib.sha256(
        f"{SELECT_SALT}|{seed}|{problem_id}|{k}|{draw}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def select_sets(
    pools: Mapping[str, ProblemPool],
    *,
    ks: Sequence[int] = TRAIN_KS,
    n_draws: int = 4,
    easy_share: float = 0.5,
    drop_c0: bool = False,
    seed: int = 0,
) -> list[CandidateSet]:
    """Every hard set and a seeded ``easy_share`` of the easy ones, shuffled."""
    if not 0.0 <= easy_share <= 1.0:
        raise ValueError("easy_share must lie in [0, 1]")
    chosen: list[CandidateSet] = []
    for pool in pools.values():
        for k in ks:
            if k > pool.n:
                continue
            for draw in range(n_draws):
                cset = draw_set(pool, k, draw)
                if cset.stratum == "c0" and drop_c0:
                    continue
                if cset.stratum in HARD or _keep_easy(
                    cset.problem_id, k, draw, easy_share, seed
                ):
                    chosen.append(cset)
    random.Random(seed).shuffle(chosen)
    return chosen


def reader_rows(
    pools: Mapping[str, ProblemPool],
    sets: Sequence[CandidateSet],
    *,
    cap: int | None = rp.CANDIDATE_CAP,
    truncate: Truncate | None = None,
) -> list[dict]:
    """GRPO rows (a chat prompt and the gold answer) for the chosen candidate sets."""
    rows = []
    for cset in sets:
        pool = pools[cset.problem_id]
        inp = build_arm_input("reader_full", pool, cset, cap=cap, truncate=truncate)
        rows.append(
            {
                "prompt": inp.messages,
                "gold": pool.problem.gold,
                "problem_id": cset.problem_id,
                "k": cset.k,
                "draw": cset.draw,
                "stratum": cset.stratum,
                "n_correct": cset.n_correct,
            }
        )
    return rows


def solver_rows(
    pools: Mapping[str, ProblemPool], *, epochs: int = 1, seed: int = 0
) -> list[dict]:
    """Rows of the solver-GRPO baseline: the plain question, once per epoch."""
    rows = []
    for epoch in range(epochs):
        ids = sorted(pools)
        random.Random(f"{seed}|{epoch}").shuffle(ids)
        for pid in ids:
            problem = pools[pid].problem
            rows.append(
                {
                    "prompt": build_round0_prompt(problem.question),
                    "gold": problem.gold,
                    "problem_id": pid,
                    "k": 0,
                    "draw": epoch,
                    "stratum": "solver",
                    "n_correct": -1,
                }
            )
    return rows


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") for m in completion if isinstance(m, Mapping))


def correctness_reward(completions, gold, **_) -> list[float]:
    """1.0 if the completion's extracted answer is correct (audit grade), else 0.0."""
    return [
        1.0 if is_correct(extract_answer(_text(c)), g) else 0.0
        for c, g in zip(completions, gold, strict=True)
    ]
