"""Clipping candidates by tokens with a real tokenizer (the GPU jobs' Truncate)."""

from __future__ import annotations

from reader_audit.controls import TAIL_TOKENS, Truncate, middle_clipper


def clipper_from_tokenizer(tokenizer, tail: int = TAIL_TOKENS) -> Truncate:
    """A middle clipper for any tokenizer with ``encode`` and ``decode`` (HF style)."""

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    return middle_clipper(encode, decode, tail)


def load_hf_clipper(model_id: str, revision: str, tail: int = TAIL_TOKENS) -> Truncate:
    """The clipper of a model's own tokenizer, pinned to its revision."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("clipping needs: pip install -e .[hf]") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return clipper_from_tokenizer(tokenizer, tail)
