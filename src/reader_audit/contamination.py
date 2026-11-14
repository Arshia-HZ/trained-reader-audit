"""The contamination probe protocol (PREREG D13), frozen here and run after the freeze.

The probe asks whether a model can finish a problem statement it was given 60% of. It
reads problem text only; no problem is ever removed because of it. It runs on the
primary and secondary sets after the Stage 2 freeze, never before (a rehearsal on dev
and on the harness set is allowed: neither is sealed).

Protocol, fixed:

- the prefix is the first ``floor(0.6 * n)`` tokens of the statement (the model's own
  tokenizer, the question text alone, no template); the remainder is what follows it;
- generation is a raw continuation of the prefix (no chat template), greedy
  (temperature 0), at most ``ceil(1.2 * n_remainder) + 16`` new tokens;
- a problem is *memorised* by a model when the completion, with every whitespace
  character removed, starts with the remainder with every whitespace character removed;
- two models are probed. ``studied`` is Qwen3-1.7B, the model the candidates, the reader
  and the solver baselines come from. ``base`` is Qwen3-1.7B-Base and measures
  pre-training exposure. The primary flag, used to split H2, is the studied model's; the
  base model's flag is reported beside it, and "either model" is a sensitivity split.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping

PREFIX_FRACTION = 0.6
MAX_NEW_FACTOR = 1.2
MAX_NEW_EXTRA = 16
MODELS = {"studied": "Qwen/Qwen3-1.7B", "base": "Qwen/Qwen3-1.7B-Base"}
_WHITESPACE = re.compile(r"\s+")


def split_statement(
    text: str,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    fraction: float = PREFIX_FRACTION,
) -> tuple[str, str]:
    """The prefix to prompt with and the remainder to match.

    The remainder is cut from the original text where the decoded prefix ends, so
    nothing depends on a tokenizer round trip of the remainder.
    """
    ids = encode(text)
    cut = math.floor(fraction * len(ids))
    prefix = decode(ids[:cut])
    if text.startswith(prefix):
        return prefix, text[len(prefix) :]
    return prefix, decode(ids[cut:])


def squeeze(text: str) -> str:
    """The text with every whitespace character removed."""
    return _WHITESPACE.sub("", text)


def is_memorised(completion: str, remainder: str) -> bool:
    """The completion reproduces the remainder (whitespace ignored)."""
    wanted = squeeze(remainder)
    return bool(wanted) and squeeze(completion).startswith(wanted)


def max_new_tokens(n_remainder_tokens: int) -> int:
    return math.ceil(MAX_NEW_FACTOR * n_remainder_tokens) + MAX_NEW_EXTRA


def flags(memorised: Mapping[str, bool]) -> dict[str, bool]:
    """The three flags of one problem from the two models' outcomes."""
    missing = set(MODELS) - set(memorised)
    if missing:
        raise ValueError(f"missing probe outcome for {sorted(missing)}")
    return {
        "primary": bool(memorised["studied"]),
        "base": bool(memorised["base"]),
        "either": bool(memorised["studied"] or memorised["base"]),
    }


def probe_negative(
    flags_by_problem: Mapping[str, Mapping[str, bool]], rule: str = "primary"
) -> set[str]:
    """Problems the probe does not flag under ``rule``: the H2 split's population."""
    if rule not in ("primary", "base", "either"):
        raise ValueError(f"unknown flag rule {rule!r}")
    return {pid for pid, f in flags_by_problem.items() if not f[rule]}
