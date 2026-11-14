"""Cost rules: how many self-consistency samples a reader costs (PREREG section 5).

Planning version, from mean token counts: the reader is assumed to write about one
sample. The per-problem exact version, from the tokens a reader actually used, is added
with the reader harness.
"""

from __future__ import annotations

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
