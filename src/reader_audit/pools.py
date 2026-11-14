"""Pools of independent samples per problem, read from finished audit-format shards.

A pool is the round-0 and ``sc_pool`` generations of a problem: independent draws from
the same prompt, so any subset is a fair set of candidates or a self-consistency sample.
Shards are checked against the sha256 recorded when they were finished.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from channel_audit.schema import Generation, Graded, Problem
from channel_audit.store import JsonlStore

POOL_ARMS: tuple[str, ...] = ("round0", "sc_pool")
DONE_FILE = "SHARD_DONE.json"


class ShardIntegrityError(RuntimeError):
    """A shard is missing its done marker, or its generations no longer match it."""


@dataclass(frozen=True)
class Candidate:
    """One graded sample of a problem's pool."""

    gen_key: str
    arm: str  # "round0" or "sc_pool"
    replicate: int
    index: int  # agent for round0, sample_idx for sc_pool
    text: str
    answer: str | None  # the extracted answer as graded (None: no usable answer)
    confidence: int | None
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    truncated: bool


@dataclass(frozen=True)
class ProblemPool:
    problem: Problem
    candidates: tuple[Candidate, ...]  # canonical order: replicate, arm, index

    @property
    def n(self) -> int:
        return len(self.candidates)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_shard(shard: Path) -> dict:
    """The shard's done marker, after checking its generations against the hash."""
    marker = shard / DONE_FILE
    if not marker.is_file():
        raise ShardIntegrityError(
            f"{shard.name}: no {DONE_FILE}; the shard is not done"
        )
    done = json.loads(marker.read_text(encoding="utf-8"))
    if _sha256(shard / "generations.jsonl") != done["generations_sha256"]:
        raise ShardIntegrityError(f"{shard.name}: generations.jsonl changed")
    return done


def _arm_rank(arm: str, arms: Sequence[str]) -> int:
    return arms.index(arm)


def read_pools(
    directory: str | Path, arms: Sequence[str] = POOL_ARMS
) -> dict[str, ProblemPool]:
    """The pools of one run directory (problems, generations and grades)."""
    base = Path(directory)
    problems = JsonlStore(
        base / "problems.jsonl", Problem, key="problem_id", readonly=True
    )
    graded = JsonlStore(base / "graded.jsonl", Graded, readonly=True)
    found: dict[str, list[Candidate]] = defaultdict(list)
    for gen in JsonlStore(base / "generations.jsonl", Generation, readonly=True):
        if gen.arm not in arms:
            continue
        grade = graded.get(gen.gen_key)
        found[gen.problem_id].append(
            Candidate(
                gen_key=gen.gen_key,
                arm=gen.arm,
                replicate=gen.replicate,
                index=gen.agent if gen.arm == "round0" else gen.sample_idx,
                text=gen.text,
                answer=grade.extracted,
                confidence=grade.confidence,
                correct=grade.correct,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                truncated=gen.finish_reason != "stop",
            )
        )
    pools: dict[str, ProblemPool] = {}
    for problem in problems:
        items = found.get(problem.problem_id, [])
        if items:
            items.sort(key=lambda c: (c.replicate, _arm_rank(c.arm, arms), c.index))
            pools[problem.problem_id] = ProblemPool(problem, tuple(items))
    return pools


def load_pools(
    root: str | Path,
    *,
    arms: Sequence[str] = POOL_ARMS,
    verify: bool = True,
) -> dict[str, ProblemPool]:
    """Every problem's pool from the ``shard_*`` folders under ``root``."""
    shards = sorted(p for p in Path(root).glob("shard_*") if p.is_dir())
    if not shards:
        raise FileNotFoundError(f"no shard_* folders under {root}")
    pools: dict[str, ProblemPool] = {}
    for shard in shards:
        if verify:
            verify_shard(shard)
        for pid, pool in read_pools(shard, arms).items():
            if pid in pools:
                raise ValueError(f"{pid} appears in two shards")
            pools[pid] = pool
    return pools


def pool_sizes(pools: Iterable[ProblemPool]) -> set[int]:
    return {p.n for p in pools}
