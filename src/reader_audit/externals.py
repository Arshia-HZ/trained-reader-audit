"""Problem sets from outside MATH: loaders and conversion to the audit's ``Problem``.

Conversion is strict: a row without a question or a gold answer is an error, never a
silent drop. Gold answers are checked against themselves with the audit's grader, so a
gold the grader cannot read is found before any generation is spent on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from channel_audit.extract import extract_answer
from channel_audit.grade import is_correct
from channel_audit.schema import Problem

# name -> (Hugging Face dataset, parquet file, problem-id prefix, competition label)
MATHARENA: dict[str, tuple[str, str, str]] = {
    "aime_2025": ("MathArena/aime_2025", "data/train-00000-of-00001.parquet", "aime25"),
    "hmmt_feb_2025": (
        "MathArena/hmmt_feb_2025",
        "data/train-00000-of-00001.parquet",
        "hmmt25",
    ),
}
HARNESS_SETS = ("aime_2025", "hmmt_feb_2025")
COMPETITION_SUBJECT = "competition"  # placebo donors come from any other problem
HARNESS_SPLIT = "check"  # the audit's label for a set used to check the harness


class GoldError(ValueError):
    """A problem's gold answer is missing or unreadable by the grader."""


def matharena_problems(
    rows: Iterable[Mapping], prefix: str, dataset: str
) -> list[Problem]:
    """Convert MathArena rows (``problem_idx``, ``problem``, ``answer``)."""
    problems = []
    for row in rows:
        question = str(row.get("problem") or "").strip()
        gold = str(row.get("answer") or "").strip()
        index = row.get("problem_idx")
        if not question or not gold or index is None:
            raise GoldError(
                f"{dataset}: a row has no problem, answer or index: {row!r}"
            )
        problems.append(
            Problem(
                problem_id=f"{prefix}-{int(index):02d}",
                dataset=dataset,
                subject=COMPETITION_SUBJECT,
                level=5,
                question=question,
                gold=gold,
                split=HARNESS_SPLIT,
            )
        )
    ids = [p.problem_id for p in problems]
    if len(set(ids)) != len(ids):
        raise GoldError(f"{dataset}: duplicate problem ids")
    return problems


def check_golds(problems: Sequence[Problem]) -> list[str]:
    """Ids whose gold answer a model's boxed answer could never match.

    A model that writes the gold in a box must be read back as exactly the gold and
    graded correct; golds with unbalanced braces or an empty box fail that round trip.
    """
    bad = []
    for p in problems:
        boxed = "\\boxed{" + p.gold + "}"
        if extract_answer(boxed) != p.gold.strip() or not is_correct(
            extract_answer(boxed), p.gold
        ):
            bad.append(p.problem_id)
    return bad


def content_sha256(problems: Sequence[Problem]) -> str:
    """Hash of ids, questions and golds, to pin a set that is rebuilt from the Hub."""
    payload = json.dumps(
        [[p.problem_id, p.question, p.gold] for p in problems], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_matharena(name: str, revision: str) -> list[Problem]:
    """One MathArena set from the Hub at a pinned revision (the data extra)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("loading datasets needs: pip install -e .[data]") from exc
    repo, filename, prefix = MATHARENA[name]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return matharena_problems(pq.read_table(path).to_pylist(), prefix, name)


# ---- pinning the harness set ------------------------------------------------------

HARNESS_MANIFEST = "harness.json"


def pin_harness(out_dir, revision_of, loader) -> dict:
    """Write the harness manifest: revision, size and content hash of each set.

    ``revision_of(repo)`` returns a dataset's current commit and ``loader(name,
    revision)`` its problems. An existing manifest is never changed.
    """
    path = Path(out_dir) / HARNESS_MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists: the harness set is already pinned")
    manifest: dict = {}
    for name in HARNESS_SETS:
        revision = revision_of(MATHARENA[name][0])
        problems = loader(name, revision)
        bad = check_golds(problems)
        if bad:
            raise GoldError(f"{name}: golds a boxed answer cannot match: {bad}")
        manifest[name] = {
            "repo": MATHARENA[name][0],
            "revision": revision,
            "n": len(problems),
            "content_sha256": content_sha256(problems),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", "utf-8")
    return manifest


def load_harness(out_dir, loader) -> list[Problem]:
    """The pinned harness problems; the content must match the manifest."""
    manifest = json.loads((Path(out_dir) / HARNESS_MANIFEST).read_text("utf-8"))
    problems: list[Problem] = []
    for name in HARNESS_SETS:
        entry = manifest[name]
        got = loader(name, entry["revision"])
        if len(got) != entry["n"] or content_sha256(got) != entry["content_sha256"]:
            raise GoldError(f"{name} no longer matches its pinned content")
        problems += got
    return problems
