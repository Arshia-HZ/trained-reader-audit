"""The training problems: MATH train levels 3 to 5 plus a DeepScaleR subset, with every
problem that overlaps a study set removed (PREREG section 2 and the delegated rule of
section 9).

A training problem is dropped when it matches any study problem (dev, primary, reserve,
the audit's screen and pilot sets, the harness set, the secondary set when it is
supplied) in one of two ways: its normalised text is identical, or the two share a run
of 13 consecutive normalised words (the usual n-gram contamination rule). Problems
shorter than 13 words can only match by identity. The study problems' text is read for
this check only; no answer or accuracy is looked at.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from channel_audit.check_problems import CONFIGS, parse_level
from channel_audit.extract import extract_boxed
from channel_audit.schema import Problem

NGRAM = 13
MATH_REPO = "EleutherAI/hendrycks_math"
DEEPSCALER_REPO = "agentica-org/DeepScaleR-Preview-Dataset"
LEVELS = (3, 4, 5)
SEED = 20261115

_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> list[str]:
    """Lower-cased alphanumeric words: spacing, case, markup and punctuation dropped."""
    return _TOKEN.findall(text.lower())


def _shingles(words: Sequence[str], n: int = NGRAM) -> set[int]:
    return {
        int.from_bytes(
            hashlib.sha1(" ".join(words[i : i + n]).encode()).digest()[:8], "big"
        )
        for i in range(len(words) - n + 1)
    }


@dataclass(frozen=True)
class Reference:
    """One study problem's text, for the contamination check."""

    set_name: str
    problem_id: str
    text: str


class Decontaminator:
    def __init__(self, references: Iterable[Reference]) -> None:
        self._exact: dict[str, Reference] = {}
        self._shingles: dict[int, Reference] = {}
        self.sets: Counter[str] = Counter()
        for ref in references:
            words = normalize(ref.text)
            self.sets[ref.set_name] += 1
            self._exact.setdefault(" ".join(words), ref)
            for h in _shingles(words):
                self._shingles.setdefault(h, ref)

    def match(self, text: str) -> tuple[str, Reference] | None:
        """(reason, the study problem it matches), or None if the text is clean."""
        words = normalize(text)
        ref = self._exact.get(" ".join(words))
        if ref is not None:
            return "identical", ref
        for h in _shingles(words):
            ref = self._shingles.get(h)
            if ref is not None:
                return "ngram", ref
        return None


@dataclass(frozen=True)
class TrainProblem:
    problem_id: str
    source: str  # "math-train" or "deepscaler"
    subject: str
    level: int | None
    question: str
    gold: str

    def to_problem(self) -> Problem:
        """As the audit's Problem (split "check"; DeepScaleR has no level: 5)."""
        return Problem(
            problem_id=self.problem_id,
            dataset=self.source,
            subject=self.subject,
            level=self.level or 5,
            question=self.question,
            gold=self.gold,
            split="check",
        )


RowLoader = Callable[[str], Iterable[Mapping]]


def load_math_train(
    rows_for: RowLoader, levels: Sequence[int] = LEVELS
) -> tuple[list[TrainProblem], list[str]]:
    """MATH train problems of the given levels; rows without a usable boxed gold or
    level are returned in the second list (their ids)."""
    out, dropped = [], []
    for config in CONFIGS:
        for i, row in enumerate(rows_for(config)):
            pid = f"math-train-{config}-{i}"
            level = parse_level(row["level"])
            gold = extract_boxed(row["solution"])
            if level is None or gold is None:
                dropped.append(pid)
                continue
            if level in levels:
                out.append(
                    TrainProblem(
                        pid, "math-train", row["type"], level, row["problem"], gold
                    )
                )
    return out, dropped


def load_deepscaler(rows: Iterable[Mapping]) -> list[TrainProblem]:
    out = []
    for i, row in enumerate(rows):
        answer = str(row["answer"]).strip()
        if answer:
            out.append(
                TrainProblem(
                    f"deepscaler-{i}",
                    "deepscaler",
                    "mixed",
                    None,
                    row["problem"],
                    answer,
                )
            )
    return out


def build_train_set(
    math_train: Sequence[TrainProblem],
    deepscaler: Sequence[TrainProblem],
    decontaminator: Decontaminator,
    n_deepscaler: int,
    seed: int = SEED,
) -> tuple[list[TrainProblem], dict]:
    """All clean MATH train problems plus ``n_deepscaler`` clean DeepScaleR ones.

    The DeepScaleR problems are taken in a seeded shuffle, so the subset does not depend
    on the file's order; duplicates of a problem already kept are skipped.
    """
    report: dict = {
        "dropped": Counter(),
        "dropped_by_set": Counter(),
        "kept": Counter(),
    }
    seen: set[str] = set()
    kept: list[TrainProblem] = []

    def take(problem: TrainProblem) -> bool:
        key = " ".join(normalize(problem.question))
        if key in seen:
            report["dropped"]["duplicate"] += 1
            return False
        hit = decontaminator.match(problem.question)
        if hit is not None:
            reason, ref = hit
            report["dropped"][f"{problem.source}:{reason}"] += 1
            report["dropped_by_set"][ref.set_name] += 1
            return False
        seen.add(key)
        kept.append(problem)
        report["kept"][problem.source] += 1
        return True

    for problem in math_train:
        take(problem)
    pool = list(deepscaler)
    random.Random(seed).shuffle(pool)
    added = 0
    for problem in pool:
        if added >= n_deepscaler:
            break
        added += take(problem)
    report["short"] = max(0, n_deepscaler - added)
    return kept, {
        k: dict(v) if isinstance(v, Counter) else v for k, v in report.items()
    }


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def train_order(ids: Sequence[str], seed: int = SEED) -> list[str]:
    """The training ids in a seeded order, so any prefix is a random sample of the set
    (the pilot takes the first few hundred, the pool job works through it in shards)."""
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def fetch_sources() -> tuple[list[TrainProblem], list[str], list[TrainProblem], dict]:
    """MATH train (levels 3 to 5) and DeepScaleR from the Hub at their pinned revisions.

    Returns the MATH problems, the ids of unusable MATH rows, the DeepScaleR problems,
    and the revisions used. Needs ``huggingface_hub`` and ``pyarrow``.
    """
    import pyarrow.parquet as pq
    from channel_audit.data import MATH_REVISION
    from huggingface_hub import HfApi, hf_hub_download

    def math_rows(config: str):
        path = hf_hub_download(
            MATH_REPO,
            f"{config}/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=MATH_REVISION,
        )
        return pq.read_table(path).to_pylist()

    math_train, unusable = load_math_train(math_rows)
    scaler_rev = HfApi().dataset_info(DEEPSCALER_REPO).sha
    path = hf_hub_download(
        DEEPSCALER_REPO, "deepscaler.json", repo_type="dataset", revision=scaler_rev
    )
    scaler = load_deepscaler(json.loads(Path(path).read_text("utf-8")))
    return (
        math_train,
        unusable,
        scaler,
        {"math": MATH_REVISION, "deepscaler": scaler_rev},
    )
