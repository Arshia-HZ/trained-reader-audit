"""Text-level controls on candidate sets: answer swap, deduplication, token caps.

The answer swap replaces the final boxed answer of a candidate and nothing else, so a
reader that follows the stated answers is led to the swapped value, while one that
checks the reasoning can still recover the derived one (PREREG section 4, H4b).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from channel_audit.aggregate import cluster_answers
from channel_audit.extract import _BOXED, _closing_index, extract_answer

Truncate = Callable[[str, int], str]


def answer_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the content of the box that ``extract_answer`` returns."""
    answer = extract_answer(text)
    if answer is None:
        return None
    found = None
    for m in _BOXED.finditer(text):
        end = _closing_index(text, m.end())
        if end is not None and text[m.end() : end].strip() == answer:
            found = (m.end(), end)  # the last box with that content is the answer's
    return found


def swap_final_answer(text: str, new_answer: str) -> str:
    """The text with the final answer's box content replaced; the reasoning is kept."""
    span = answer_span(text)
    if span is None:
        raise ValueError("the text has no final answer to swap")
    start, end = span
    return text[:start] + new_answer + text[end:]


def plurality_wrong_answer(
    answers: Sequence[str | None], correct: Sequence[bool]
) -> str | None:
    """The most frequent wrong answer (ties: the one shown first), or None."""
    labels, reps = cluster_answers(answers)
    group_correct = [False] * len(reps)
    for label, ok in zip(labels, correct, strict=True):
        if label is not None and ok:
            group_correct[label] = True
    counts = Counter(
        label
        for label in labels
        if label is not None and not group_correct[label]  # wrong groups only
    )
    if not counts:
        return None
    best = max(counts.values())
    first = min(label for label, c in counts.items() if c == best)
    return reps[first]


def dedup_order(answers: Sequence[str | None]) -> list[int]:
    """One position per distinct answer, most frequent first (ties: first shown).

    Candidates without a parsable answer count as one group, shown last.
    """
    labels, _ = cluster_answers(answers)
    groups: dict[int | None, list[int]] = {}
    for position, label in enumerate(labels):
        groups.setdefault(label, []).append(position)
    ranked = sorted(
        (g for g in groups.items() if g[0] is not None),
        key=lambda item: (-len(item[1]), item[1][0]),
    )
    order = [positions[0] for _, positions in ranked]
    if None in groups:
        order.append(groups[None][0])
    return order


def cap_tokens(text: str, cap: int | None, truncate: Truncate | None) -> str:
    """The text cut to ``cap`` tokens by ``truncate``, which knows the tokenizer."""
    if cap is None or truncate is None:
        return text
    return truncate(text, cap)


OMITTED = "\n[... middle of the solution omitted ...]\n"
TAIL_TOKENS = 256


def middle_clipper(
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    tail: int = TAIL_TOKENS,
) -> Truncate:
    """A ``Truncate`` that keeps the start and the last ``tail`` tokens of a long text.

    A solution ends with its conclusion and its boxed answer, so cutting the end would
    hide the answer a reader is meant to weigh. A text of at most ``cap`` tokens is
    returned unchanged; a longer one keeps ``cap - tail`` tokens from its start and
    ``tail`` from its end, with a marker between them (about ten more tokens, so the
    result has ``cap`` plus the marker; checked with the Qwen3 tokenizer: 1,545 for a
    cap of 1,536).
    """

    def clip(text: str, cap: int) -> str:
        ids = encode(text)
        if len(ids) <= cap:
            return text
        keep = max(cap - tail, 0)
        return decode(ids[:keep]) + OMITTED + decode(ids[len(ids) - tail :])

    return clip
