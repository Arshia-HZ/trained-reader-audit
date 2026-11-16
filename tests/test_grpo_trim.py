"""The padding trim of the GRPO scoring step gives the same scores on less work."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from reader_audit.grpo_run import trim_padding  # noqa: E402


class Scorer:
    """Stands in for TRL's scoring: a score that depends on the token and its place."""

    def __init__(self) -> None:
        self.widths: list[int] = []

    def __call__(self, model, input_ids, attention_mask, keep, **kw):
        self.widths.append(input_ids.size(1))
        tail = input_ids[:, -keep:].float()
        mask = attention_mask[:, -keep:].float()
        # a token's score depends only on its own id and the live tokens before it
        before = (input_ids.float() * attention_mask).cumsum(-1)[:, -keep:]
        logps = (tail * 0.01 + before * 1e-4) * mask
        entropy = logps.abs() if kw.get("compute_entropy") else None
        return logps, entropy, None


def padded_batch():
    # prompt left padded to 6, completion right padded to 4; rows of different length
    ids = torch.tensor(
        [
            [0, 0, 5, 6, 7, 8, 9, 10, 11, 0],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 0, 0, 0, 7, 8, 9, 0, 0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ]
    )
    return ids, mask, 4


def test_scores_match_the_padded_ones_and_the_work_shrinks():
    ids, mask, keep = padded_batch()
    full = Scorer()
    want = full(None, ids, mask, keep, compute_entropy=True)
    scorer = Scorer()
    got = trim_padding(scorer)(None, ids, mask, keep, compute_entropy=True)
    assert got[0].shape == want[0].shape == (3, keep)
    assert torch.allclose(got[0], want[0], atol=1e-6)
    assert torch.allclose(got[1], want[1], atol=1e-6)
    assert scorer.widths == [7, 10, 3] and sum(scorer.widths) < 30


def test_a_row_with_a_gap_in_its_mask_is_scored_untrimmed():
    ids, mask, keep = padded_batch()
    mask = mask.clone()
    mask[1, 4] = 0
    scorer = Scorer()
    out = trim_padding(scorer)(None, ids, mask, keep)
    assert scorer.widths[1] == 10 and out[0].shape == (3, keep)


def test_extra_inputs_such_as_images_bypass_the_trim():
    ids, mask, keep = padded_batch()
    scorer = Scorer()
    trim_padding(scorer)(None, ids, mask, keep, pixel_values=torch.zeros(1))
    assert scorer.widths == [10]  # one call on the whole padded batch


def test_gradients_flow_through_the_padding_back_to_the_scores():
    ids, mask, keep = padded_batch()
    weight = torch.ones(1, requires_grad=True)

    def scorer(model, i, m, k, **kw):
        return (i[:, -k:].float() * weight) * m[:, -k:], None, None

    out = trim_padding(scorer)(None, ids, mask, keep)[0]
    out.sum().backward()
    assert weight.grad is not None and weight.grad.item() > 0
