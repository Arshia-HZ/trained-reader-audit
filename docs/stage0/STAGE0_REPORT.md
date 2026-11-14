# Stage 0: break-even and headroom on the 600 dev problems

Run on 2026-11-14. Data: the 12 confirmatory shards of `Arshia-HZ/mas-channel-audit-raw`
(`confirm/qwen3-1.7b/`; `generations`, `graded` and `problems` files only). Code:
`stage0_breakeven.py` in this folder; results: `stage0.json`. Dev problems only (never a
test set). The thresholds are those of the revised roadmap, written before this run.

## Method

- Each problem has 18 exchangeable samples: 2 replicates × (3 round-0 agents + 6 pool
  samples), all with the round-0 prompt. 600 problems, 10,800 samples.
- Answers are grouped with the audit's own `cluster_answers`; correctness is the stored grade.
  Seven samples (of 10,800) sit in a group whose grades disagree; they are left as graded.
- Every quantity for a random K-subset is **exact** (sum over each answer group's count
  vector, hypergeometric weights), not sampled. A brute-force enumeration agrees to 1e-12
  on the first three problems for k = 2, 3, 4.
- Vote = plurality with 1/m tie credit, unparsable answers never vote. Headroom =
  pass@K − maj@K (tie credit included). "Strict minority" = a correct answer present but
  not in the plurality at all (the audit's G2 definition).
- Intervals: 10,000 problem-level bootstrap resamples, `default_rng(0)`, percentile.
- Matched SC at fractional k\* is linearly interpolated on the 18-sample SC curve. Measured
  means: 819.7 generated and 150.6 prompt tokens per sample; 7.8% of samples truncated.
- Reader cost rules (reader writes about one sample): generated-only k\* = K + 1; generated +
  prompt k\* = ((2K+1)g + (K+1)p) / (g+p); FLOP rule with a 0.6B reader weights its tokens
  by 0.35 (the report's estimate); the unanimity gate reads only sets whose K answers differ,
  priced with the prompt-inclusive rule.

## Checks against the audit

| Quantity | Audit | Here |
| --- | --- | --- |
| SC accuracy at k = 1 / 3 / 5 / 9 | 63.7 / 69.7 / 73.4 / 76.8 | 63.7 / 69.8 / 73.5 / 76.6 |
| Strict minority share at K = 3 (G2) | 2.6% | 2.8% |

The small gaps are expected: the audit's SC used 9 samples per replicate; this pools 18.

## Results

SC accuracy by sample count on the 600 problems: k=1 63.7%, k=3 69.8%, k=5 73.5%, k=9 76.6%,
k=18 78.8%. By level, k=1 → k=18: L3 76.8 → 87.7, L4 70.2 → 84.6, L5 45.2 → 64.8.

### Headroom and strata (K = 4 and 8; percent of problems)

| Set | K | maj@K | pass@K | Headroom [95% CI] | Strict minority | c0 | c1 | c2+ | Unanimous | Unanimous and wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all 600 | 4 | 71.9 | 81.1 | 9.2 [8.1, 10.3] | 4.0 | 18.9 | 10.6 | 70.5 | 45.5 | 1.6 |
| all 600 | 8 | 76.0 | 85.9 | 9.9 [8.4, 11.5] | 7.1 | 14.1 | 6.2 | 79.7 | 35.9 | 0.7 |
| level 3 | 4 | 84.0 | 90.5 | 6.5 [4.9, 8.2] | 3.2 | 9.5 | 7.3 | 83.2 | 60.3 | 1.3 |
| level 3 | 8 | 86.5 | 94.0 | 7.5 [5.1, 10.0] | 6.0 | 6.0 | 5.1 | 88.9 | 49.9 | 0.5 |
| level 4 | 4 | 77.8 | 86.0 | 8.1 [6.3, 10.1] | 3.7 | 14.0 | 8.5 | 77.5 | 51.8 | 1.4 |
| level 4 | 8 | 81.4 | 89.5 | 8.1 [5.7, 10.8] | 5.6 | 10.5 | 4.5 | 85.0 | 41.0 | 0.5 |
| **level 5 (208)** | 4 | 54.8 | 67.7 | **12.9 [10.9, 14.9]** | 4.9 | 32.3 | 15.7 | 52.1 | 25.6 | 2.1 |
| **level 5 (208)** | 8 | 60.9 | 74.9 | **13.9 [11.0, 17.0]** | 9.6 | 25.1 | 9.0 | 65.9 | 17.7 | 1.1 |

### Break-even, SC(k\*) − maj@K, level 5 (points; 95% CI)

| Cost rule | K = 4: k\* | K = 4: break-even | K = 8: k\* | K = 8: break-even |
| --- | --- | --- | --- | --- |
| Generated tokens only | 5.00 | 2.3 [1.9, 2.7] | 9.00 | 0.8 [0.6, 1.1] |
| Generated + prompt (= FLOPs, same-size reader) | 8.38 | 6.4 [5.1, 7.8] | 15.76 | 3.6 [2.2, 5.0] |
| FLOPs, 0.6B reader | 5.53 | 3.1 [2.5, 3.7] | 10.72 | 1.9 [1.3, 2.5] |
| Prompt-inclusive, unanimity gate (reads 74% / 82%) | 7.26 | 5.4 [4.3, 6.5] | 14.38 | 3.3 [2.1, 4.5] |

The same table for all 600 problems and for levels 3 and 4 is in `stage0.json`.

### Gate G1 (level-5 dev, generated-only rule)

Rule from the roadmap: headroom × recovery ≥ break-even + 1.5 points, for some K, with
recovery 27% (low) to 57% (high).

| K | Headroom | Needed gain (break-even + 1.5) | Expected gain, low / high recovery | Verdict |
| --- | --- | --- | --- | --- |
| 4 | 12.9 | 3.8 | 3.5 / 7.4 | passes at the high end only |
| 8 | 13.9 | 2.3 | 3.8 / 7.9 | **passes at both ends** |

**G1 passes on the roadmap's own rule, at K = 8.** Under the other cost rules, with the
needed gain = break-even + 1.5:

| Level 5 | K = 4 low / high | K = 8 low / high |
| --- | --- | --- |
| Generated + prompt | fail / fail (7.9 needed; 3.5–7.4) | fail / pass (5.1 needed; 3.8–7.9) |
| FLOPs, 0.6B reader | fail / pass (4.6 needed) | pass / pass (3.4 needed) |
| Unanimity gate, prompt-inclusive | fail / pass (6.9 needed; 3.5–7.4) | fail / pass (4.8 needed; 3.8–7.9) |

On the other splits only K = 8 under the generated-only rule passes at both ends. On all 600
problems and on levels 3 and 4, K = 4 passes only at the high end, and K = 4 with prompt
tokens counted fails at both ends everywhere.

## What this says

1. **K = 8 is the right design point.** The break-even shrinks with K under the
   generated-only rule (2.3 → 0.8 points at level 5), while headroom grows (12.9 → 13.9).
   This matches the research report's argument.
2. **Level 5 is the right primary set for H1.** Its headroom is about twice level 3's, and
   unanimity (where reading has nothing to do) is only 18–26%, against 50–60% at level 3.
3. **The prompt-inclusive rule is the hard one.** At K = 4 even the optimistic recovery
   rate (57%) does not clear it. At K = 8 it clears only at the optimistic end. The
   0.6B reader and the gate each help, but neither rescues K = 4.
4. **The unanimity gate costs almost no headroom.** Headroom is zero on unanimous sets, so
   the gate only gives up fresh solving. But at level 5 it still reads 74% (K = 4) to 82%
   (K = 8) of sets, so its saving there is small.
5. **c0 is the largest stratum at level 5 (25–32%)**, and nothing in headroom counts it. Any
   reader gain there is fresh solving, which TTT-Consolidate found in-context aggregators do
   rarely (0.8–6.7%).

## Limits of this gate

- The pass rule counts only recovered headroom. It ignores what the reader breaks on c2+
  sets (the audit's `full_solution` broke about 22 of 801 vote successes). The true expected
  gain is lower.
- The 27–57% recovery range comes from AggLM on **thinking** candidates at K = 8. The reader
  here reads non-thinking samples, so the transfer is an assumption. AggLM's reported gains
  over maj@8 on non-thinking candidates are +4.4 to +11.9 points on competition sets, where
  the vote scores 7–18%, a different regime from level 5 of MATH (maj@8 about 61%).
- Reader length is assumed to be one sample. The pilot must measure it.
- 7.8% of the samples are truncated and count as unparsable; a reader sees them as
  candidates without an answer.
- GPU-seconds, the third accounting axis, waits for the Stage 2 throughput pilot.
- Candidate pools use the round-0 prompt (with the confidence request) on the fp16 /
  vLLM 0.33.0 stack; pools for training and testing should keep this stack.

## Files

- `stage0_breakeven.py`: the analysis (run with `python -I`, repo venv).
- `stage0.json`: every number, for all 600 problems and for levels 3, 4 and 5.
