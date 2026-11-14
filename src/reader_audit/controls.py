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
    """A ``Truncate`` that keeps the end of a long text and as much of its start as fit.

    A solution ends with its conclusion and its boxed answer, so cutting the end would
    hide the answer a reader is meant to weigh. A text of at most ``cap`` tokens is
    returned unchanged. A longer one becomes ``head + marker + tail`` and is **at most
    ``cap`` tokens including the marker**: the last ``tail`` tokens are kept, the
    marker is counted, and the head takes what is left (with the Qwen3 tokenizer and a
    cap of 1,536, about 1,270 tokens). Re-tokenising the joined text can change a token
    at a seam, so the head is trimmed until the whole result fits.
    """
    marker = len(encode(OMITTED))

    def build(ids: list[int], head: int, tail_n: int) -> str:
        end = decode(ids[len(ids) - tail_n :]) if tail_n else ""
        return decode(ids[:head]) + OMITTED + end

    def clip(text: str, cap: int) -> str:
        ids = encode(text)
        if len(ids) <= cap:
            return text
        room = cap - marker
        if room <= 0:  # no room for the marker: keep only the end
            return decode(ids[len(ids) - cap :]) if cap > 0 else ""
        tail_n = min(tail, room)
        head = room - tail_n
        out = build(ids, head, tail_n)
        over = len(encode(out)) - cap
        while over > 0 and (head > 0 or tail_n > 0):
            if head > 0:
                head = max(head - over, 0)
            else:
                tail_n = max(tail_n - over, 0)
            out = build(ids, head, tail_n)
            over = len(encode(out)) - cap
        return out

    return clip
