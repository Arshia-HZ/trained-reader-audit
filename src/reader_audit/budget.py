"""Cost rules: how many self-consistency samples a reader costs (PREREG section 5).

Two versions. The planning functions (``planning_k_star``) work from mean token counts
and assume the reader writes about one sample; Stage 0 used them. The per-problem
functions (``reader_cost``, ``sample_cost``, ``k_star``) use the tokens a reader and its
candidates actually used and are what the confirmatory analysis uses.
"""

from __future__ import annotations

from collections.abc import Sequence

RULES: tuple[str, ...] = (
    "generated_only",
    "generated_plus_prompt",
    "flops_0.6b_reader",
    "gated_unanimity_prompt_incl",
)

SIX_TENTHS_B_WEIGHT = 0.35  # a 0.6B token costs about 0.35 of a 1.7B token (estimate)


def planning_k_star(
    rule: str,
    k: int,
    gen_tokens: float,
    prompt_tokens: float,
    *,
    share_read: float = 1.0,
    reader_weight: float = SIX_TENTHS_B_WEIGHT,
) -> float:
    """Self-consistency samples that cost as much as K candidates plus one read.

    ``gen_tokens`` and ``prompt_tokens`` are the means of one pool sample. A reader's
    prompt holds the problem and K candidates; its output is about one sample.
    ``share_read`` is the share of problems a gate sends to the reader.
    """
    g, p = gen_tokens, prompt_tokens
    unit = g + p  # one self-consistency sample under the prompt-inclusive rule
    read = (k * g + p + g) / unit  # the reader's own cost, in samples, prompt included
    if rule == "generated_only":
        return k + 1.0
    if rule == "generated_plus_prompt":
        return k + read
    if rule == "flops_0.6b_reader":
        return k + reader_weight * read
    if rule == "gated_unanimity_prompt_incl":
        return k + share_read * read
    raise ValueError(f"unknown cost rule {rule!r}; expected one of {RULES}")


# ---- per-problem costs (PREREG section 5) -----------------------------------------

Tokens = tuple[int, int]  # (prompt tokens, generated tokens) of one generation
COUNTED_RULES: tuple[str, ...] = ("generated_only", "generated_plus_prompt")


def reader_cost(
    rule: str,
    candidates: Sequence[Tokens],
    reader: Tokens | None,
    weight: float = 1.0,
) -> float:
    """What a read costs: its K candidates plus the reader's own call.

    ``reader`` is None when a gate sent the set to the plain vote. ``weight`` is the
    reader's parameter count over the solver's (1; about 0.35 for 0.6B).
    """
    if rule == "generated_only":
        return sum(g for _, g in candidates) + (reader[1] if reader else 0)
    if rule == "generated_plus_prompt":
        read = weight * (reader[0] + reader[1]) if reader else 0.0
        return sum(p + g for p, g in candidates) + read
    raise ValueError(f"unknown cost rule {rule!r}; expected one of {COUNTED_RULES}")


def sample_cost(rule: str, samples: Sequence[Tokens]) -> float:
    """The mean cost of one self-consistency sample of a problem."""
    if not samples:
        raise ValueError("a problem needs pool samples to price one")
    if rule == "generated_only":
        return sum(g for _, g in samples) / len(samples)
    if rule == "generated_plus_prompt":
        return sum(p + g for p, g in samples) / len(samples)
    raise ValueError(f"unknown cost rule {rule!r}; expected one of {COUNTED_RULES}")


def k_star(
    rule: str,
    candidates: Sequence[Tokens],
    reader: Tokens | None,
    samples: Sequence[Tokens],
    weight: float = 1.0,
) -> float:
    """Self-consistency samples of this problem that cost as much as the read."""
    return reader_cost(rule, candidates, reader, weight) / sample_cost(rule, samples)
