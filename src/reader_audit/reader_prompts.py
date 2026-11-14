"""Reader prompts: templates, hashing and the pure builders (PREREG D14, D16).

Templates live in ``prompts/reader_v0/`` (drafts until the Stage 2 freeze) and are
hashed byte for byte with line endings normalised, as the audit's are. Every control
that shows candidate solutions renders through the same template as the reader, so the
wording is identical by construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from channel_audit.channels import NO_ANSWER, render, strip_confidence

from reader_audit.controls import Truncate, cap_tokens

PROMPT_VERSION = "reader_v0"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / PROMPT_VERSION

EMPTY_SLOT = "(no solution)"
CANDIDATE_CAP = 1536  # tokens kept of each candidate in a reader prompt (PREREG D14)

TEMPLATE_FOR = {
    "reader_full": "reader",
    "placebo": "reader",
    "duplicate": "reader",
    "single_slot": "reader",
    "dedup": "reader",
    "answer_swap": "reader",
    "candidate_free": "reader_no_candidates",
    "answer_only": "reader_answers",
}
ARMS: tuple[str, ...] = tuple(TEMPLATE_FOR)


def _template_bytes(arm: str) -> bytes:
    if arm not in TEMPLATE_FOR:
        raise ValueError(f"unknown reader arm {arm!r}; expected one of {ARMS}")
    path = PROMPTS_DIR / f"{TEMPLATE_FOR[arm]}.txt"
    return path.read_bytes().replace(b"\r\n", b"\n")


@cache
def template_text(arm: str) -> str:
    return _template_bytes(arm).decode("utf-8").rstrip("\n")


def template_hash(arm: str) -> str:
    return hashlib.sha256(_template_bytes(arm)).hexdigest()


def prepare_candidate(
    text: str, cap: int | None = CANDIDATE_CAP, truncate: Truncate | None = None
) -> str:
    """What a reader sees of a candidate: no confidence line, at most ``cap`` tokens."""
    return cap_tokens(strip_confidence(text), cap, truncate)


def format_candidates(texts: Sequence[str]) -> str:
    return "\n\n".join(f"Solver {i + 1}:\n{t}" for i, t in enumerate(texts))


def format_answers(answers: Sequence[str | None]) -> str:
    return "\n".join(
        f"Solver {i + 1}: {a if a is not None else NO_ANSWER}"
        for i, a in enumerate(answers)
    )


def _user(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def build_prompt(
    arm: str,
    question: str,
    *,
    texts: Sequence[str] | None = None,
    answers: Sequence[str | None] | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one arm. ``texts`` must already be prepared candidates."""
    template = template_text(arm)
    if arm == "candidate_free":
        return _user(render(template, question=question))
    if arm == "answer_only":
        if answers is None:
            raise ValueError("answer_only needs the candidates' answers")
        return _user(
            render(
                template,
                question=question,
                k=str(len(answers)),
                answers=format_answers(answers),
            )
        )
    if texts is None:
        raise ValueError(f"{arm} needs candidate texts")
    return _user(
        render(
            template,
            question=question,
            k=str(len(texts)),
            candidates=format_candidates(texts),
        )
    )
