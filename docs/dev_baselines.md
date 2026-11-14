# Dev baselines (zero extra generation)

Computed on the 600 dev problems (18 samples each) with `scripts/03_dev_baselines.py`. Descriptive; dev only. Points are percentage points; intervals are 95% bootstrap over problems.

## Stored confidence

Missing in 9.4% of samples; mean stated confidence 93.9 when the answer is right and 88.8 when wrong; 76% of samples state 90 or more.

## Vote weighted by stored confidence, against the plain vote at the same K

| Set | K | Plain vote | Weighted vote | Gain [95% CI] |
| --- | --- | --- | --- | --- |
| all | 4 | 71.9 | 72.1 | +0.2 [-0.0, +0.5] |
| level_3 | 4 | 84.0 | 84.3 | +0.3 [+0.0, +0.6] |
| level_4 | 4 | 77.8 | 78.1 | +0.3 [-0.2, +0.8] |
| level_5 | 4 | 54.9 | 54.9 | +0.1 [-0.4, +0.5] |
| all | 8 | 76.0 | 76.0 | +0.0 [-0.2, +0.3] |
| level_3 | 8 | 86.5 | 86.7 | +0.1 [-0.1, +0.4] |
| level_4 | 8 | 81.4 | 81.3 | -0.1 [-0.5, +0.3] |
| level_5 | 8 | 61.0 | 61.0 | +0.1 [-0.4, +0.5] |

## Early-stopping self-consistency against plain SC at the same mean samples

A window of w samples is drawn at a time; the run stops when a whole window agrees, or at the cap. The comparator is plain SC read at the mean number of samples ESC used. `gain` is what an adaptive method adds for free; a gated reader (H6) has to beat ESC, not plain SC.

| Setting | Mean samples | Set | ESC | SC, same samples | Gain [95% CI] |
| --- | --- | --- | --- | --- | --- |
| window=2,max=8 | 3.89 | all | 75.4 | 71.7 | +3.7 [+3.1, +4.3] |
| window=2,max=8 | 4.98 | level_5 | 60.2 | 57.1 | +3.1 [+2.3, +4.0] |
| window=2,max=16 | 5.20 | all | 76.9 | 73.7 | +3.2 [+2.4, +4.0] |
| window=2,max=16 | 7.37 | level_5 | 62.6 | 60.4 | +2.3 [+1.2, +3.5] |
| window=4,max=8 | 6.18 | all | 76.0 | 74.7 | +1.3 [+1.1, +1.6] |
| window=4,max=8 | 6.98 | level_5 | 61.0 | 60.0 | +1.0 [+0.7, +1.4] |
| window=4,max=16 | 9.59 | all | 78.5 | 76.8 | +1.7 [+1.2, +2.3] |
| window=4,max=16 | 12.13 | level_5 | 64.5 | 63.6 | +1.0 [+0.3, +1.7] |
| window=8,max=8 | 8.00 | all | 76.0 | 76.0 | -0.0 [-0.1, +0.1] |
| window=8,max=8 | 8.00 | level_5 | 61.0 | 61.0 | +0.0 [-0.1, +0.2] |
| window=8,max=16 | 13.13 | all | 78.6 | 78.0 | +0.6 [+0.3, +0.8] |
| window=8,max=16 | 14.58 | level_5 | 64.6 | 64.3 | +0.2 [-0.0, +0.6] |

## The bar for a K-candidate reader

A reader of K candidates costs about K + 1 samples (generated-only rule). The frontier is the best accuracy that mixing the ESC settings above and plain SC reaches at that cost (pooled means; no interval).

| Set | K | Cost (samples) | Plain vote of K | SC at that cost | ESC frontier | Reader must beat the vote by | ... and SC by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 4 | 5 | 71.9 | 73.5 | 76.7 | +4.7 | +3.1 |
| all | 8 | 9 | 76.0 | 76.6 | 78.3 | +2.3 | +1.8 |
| level_3 | 4 | 5 | 84.0 | 85.1 | 87.0 | +3.0 | +1.9 |
| level_3 | 8 | 9 | 86.5 | 86.8 | 87.7 | +1.2 | +0.9 |
| level_4 | 4 | 5 | 77.8 | 79.2 | 82.5 | +4.6 | +3.2 |
| level_4 | 8 | 9 | 81.4 | 81.9 | 84.1 | +2.8 | +2.2 |
| level_5 | 4 | 5 | 54.9 | 57.1 | 60.3 | +5.4 | +3.1 |
| level_5 | 8 | 9 | 61.0 | 61.8 | 63.3 | +2.3 | +1.5 |

## Approximate reader prompt length (tokens)

Question and template from the pool's own prompt, plus candidates cut at `cap` tokens. The reader's reply adds up to 2,048. The last two columns give the share of prompts whose length plus 2,048 exceeds the context window.

| K, cap | Mean | p50 | p90 | p99 | p99.9 | Max | > 12,288 | > 16,384 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=2,cap=1536 | 1858 | 1650 | 3329 | 3542 | 3824 | 3983 | 0.00% | 0.00% |
| K=2,cap=2048 | 1965 | 1650 | 3692 | 4482 | 4608 | 4855 | 0.00% | 0.00% |
| K=4,cap=1536 | 3414 | 3040 | 6108 | 6606 | 6822 | 6927 | 0.00% | 0.00% |
| K=4,cap=2048 | 3628 | 3044 | 6991 | 8566 | 8719 | 8783 | 0.00% | 0.00% |
| K=8,cap=1536 | 6529 | 5802 | 11788 | 12747 | 12943 | 13119 | 17.90% | 0.00% |
| K=8,cap=2048 | 6958 | 5828 | 13564 | 16743 | 16863 | 16968 | 20.13% | 7.92% |

## Reading (written by hand, 2026-11-14)

- **Confidence weighting buys nothing here.** Weighting each vote by the stated
  confidence changes the vote by +0.2 points at K = 4 and +0.0 at K = 8 (intervals
  include 0 or touch it). Stated confidence barely separates right from wrong (93.9 against
  88.8), and three quarters of the samples say 90 or more. So "the reader just learned to
  weight by confidence" is not a live explanation for Qwen3-1.7B.
- **Adaptive SC is a strong, free baseline.** Early stopping at about 4 samples reaches
  75.4% where plain SC at 4 samples reaches 71.7% (+3.7); at about 9.6 samples it gains
  +1.7. The gain is largest on easy problems, which stop early, and is still +1.5 to +2.3
  points on level 5 at the cost of a K = 8 reader. Mixing settings, the frontier at 9
  samples is 63.3% on level 5 against 61.8% for plain SC. A reader that beats plain SC but
  not ESC has not shown more than adaptive compute does. PREREG D22 therefore reports
  every H1 result against ESC as well, and H6 uses it as its comparator.
- **The window must be 16,384 tokens for K = 8.** With candidates cut at 1,536 tokens, the
  longest K = 8 prompt is about 13.1k tokens, so prompt plus a 2,048-token reply fits
  16,384 with room; 12,288 would cut about 18% of prompts. Without the cap, 7.9% would not
  fit even in 16,384.
- Treat the ESC numbers as planning values: a sample costs one unit here, whatever its
  length, and 200 random orders per problem are averaged.
