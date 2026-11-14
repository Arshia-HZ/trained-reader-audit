# Preregistration (DRAFT until the Stage 2 freeze)

Status: **draft**, started 2026-11-14. It is completed, reviewed by the owner and committed
before any trained model is evaluated or any sealed test set is scored (the Stage 2 freeze,
section 9). Until then it records decisions already taken, with their reasons, and the
evidence they rest on. Every change is dated and appended; nothing is rewritten silently.

Items marked **[delegated]** were decided by the coding agent on 2026-11-14 at the owner's
request ("decide the design questions yourself") and need the owner's review before the
freeze. The owner may overrule any of them; the log in section 10 then records the change.

This study follows the closed [Channel Audit](https://github.com/Arshia-HZ/mas-channel-audit)
(tag `audit-final`), whose confirmatory result was a precise null: no message channel beat
budget-matched self-consistency. The audit's grader, answer extractor, seeds, append-only
store, token accounting and bootstrap are imported, not copied (one grader, one extractor).

## 1. Question and claims

Can a small reader, trained with GRPO to read K of its own model's samples and write one
answer, beat self-consistency (SC) at the same cost, and does the gain come from reading
rather than from re-solving, voting or training alone?

- **Part A.** Qwen3-1.7B (non-thinking) is trained to read K candidate solutions drawn from
  base Qwen3-1.7B. Everything is measured against SC matched per problem under three cost
  rules, with counterfactual controls (candidate-free, answer-only, placebo, answer swap)
  split by the number of correct candidates (c0, c1, c2+).
- **Part B.** The solver side: a free grid {base, solver-GRPO candidates} × {vote, untrained
  reader, trained reader}; a joint solver-and-reader model; and, last, solvers trained
  against a frozen reader.
- **Not claimed:** that a trained small aggregator beats voting; that compute matching or
  controls on a trained aggregator are new; that the form effect or a conformity fix is new
  (AggLM, RL-GenSelect, SSA, TTT-Consolidate, SPIRAL, ParaGator, 2608.18379, 2610.21290;
  see `docs/ROADMAP.md`).

## 2. Data and what has been looked at

| Set | Source | Size | Use | Status on 2026-11-14 |
| --- | --- | --- | --- | --- |
| Dev | The audit's 600 confirmatory problems (199 / 193 / 208 at levels 3 / 4 / 5), 18 existing samples each (2 replicates × (3 round-0 + 6 pool), all with the audit's round-0 prompt) | 600 | Stage 0 and 1 analyses, prompt development, paired SD, checkpoint rule. Never a test set. | Read. Stage 0 and the dev baselines of section 7 were computed on it. |
| Harness | AIME 2025 + HMMT Feb 2025 (MathArena) | 60 | Gate G2 with Qwen3-4B-Instruct-2507; reported, not powered | Not yet loaded |
| Primary | MATH test level 5 minus every screen, pilot and dev problem | **972** (1,324 − 82 − 62 − 208) | H1 to H4 | **Sealed** |
| Secondary | OlympiadBench `OE_TO_maths_en_COMP` | 674 | Replication of H1; not in the Holm families | **Sealed** |
| Robustness | MATH-P-Hard | 279 | Perturbed level-5 problems | **Sealed** |
| Post-release | HMMT Nov 2025, AIME 2026, HMMT Feb 2026 | 93 | Contamination check; Bayesian intervals; not powered | **Sealed** |
| Train | MATH train levels 3–5 plus a DeepScaleR subset, decontaminated against every set above | about 8k | Candidate pools and GRPO | Not built |

- "Sealed" means: pools may be generated, and only run quality (overflow, truncation, unboxed
  rate) may be read, until the Stage 2 freeze. No accuracy of any kind is computed on a
  sealed set before then, not even with untrained models.
- The primary set is the 972 level-5 problems of the audit's confirm split that come after
  its first 600 in the frozen hash order. Membership depends on problem ids and levels only.
  The 1,678 level-3 and level-4 confirm problems after the first 600 are an unused reserve.
- The audit's problem split is rebuilt from the pinned MATH revision and checked against
  the audit's committed manifest; the new set manifests are in `data/sets/`.

## 3. Decisions of 2026-11-14

| ID | Question | Decision | Why |
| --- | --- | --- | --- |
| D1 | K for the headline test | **K = 8** (H1 and the main arms). K = 2, 4 and 8 are evaluated for every Must arm. | Stage 0: at K = 8 the generated-only break-even is 0.8 points and G1 passes at both ends of the recovery range; at K = 4 it passes only at the optimistic end. The owner agreed on 2026-11-14. |
| D2 | G1 verdict | **Pass at K = 8** on level-5 dev under the generated-only rule. Both recovery ends pass, so the choice of end no longer matters. | `docs/stage0/STAGE0_REPORT.md` |
| D3 | Cost rules | Generated tokens is the primary rule for H1a and H1b. Prompt-inclusive (parameter-weighted prompt + generated tokens, SC's own prompt counted) is reported for every test as a pre-registered estimate and is the rule of H6. Measured GPU-seconds is descriptive and is measured at the Stage 2 pilot. | The audit's primary rule was generated tokens. Under the prompt-inclusive rule a reader reading K candidates costs about 2K + 1 samples, so that is the rule a reviewer will use. |
| D4 | Pooling the solver clause over primary and secondary | **No pooling.** The solver clause is its own test, H1b, on the primary set. The secondary set replicates H1a and H1b descriptively and is not in a Holm family. All non-dev, non-harness sets stay sealed through Stage 1. | Pooling makes the secondary set confirmatory and mixes populations with different accuracy; it also conflicts with using it in Stage 1. Sealing everything keeps Stage 1 from steering the design. |
| D5 | Hypothesis form | H1a, H1b: superiority (95% CI lower bound above 0). Planning effect 1.5 points, used for sizing only. H1b is powered for 1.5 points; a smaller effect is reported as an estimate. | A "point estimate ≥ 1.5 and CI > 0" rule fails half the time at a true effect of 1.5. |
| D6 | H4 direction | H4a (do no harm): at c2+, reader non-inferior to the plurality vote, margin 1 point. H4b: **two-sided** test of the change in answer-swap drop after training. The two competing predictions are recorded in section 4, but neither sets the test's direction. | The literature predicts RL raises reliance on answer frequency (TTT-Consolidate); the audit's conformity result suggests it could fall. With no basis for a direction, a one-sided test would be a coin flip dressed as a prediction. |
| D7 | Seeds | 3 seeds per main arm (reader, solver-GRPO). **5 seeds** if the Stage 2 pilot measures at most 20 GPU-hours per 300-step run; the rule is fixed now, the measured number goes in the freeze. Joint arm: 1 seed (Must), 2nd seed (Should). | 3 vs 3 gives only 20 relabellings, so a seed-level permutation test cannot go below p = 0.05 (one-sided); 5 vs 5 gives 252 (min p 0.004). |
| D8 | Seeds in the confirmatory test | The test is conditional on the trained checkpoints: a problem-level paired bootstrap of the seed-averaged result. In addition, H1a and H1b are called **robust** only if the gain is positive for every individual seed. The seed-level permutation test and a seed-and-problem bootstrap are robustness checks, reported with their caveats. | Three seeds cannot estimate seed variance well; the per-seed sign rule is honest about that. |
| D9 | Bootstrap | Problem-level paired bootstrap on within-problem averages, 10,000 resamples, numpy `default_rng(0)`, percentile intervals, p-values from the same resamples, as in the audit (its frozen `bootstrap_means` is reused). Strata are problem-level means over the candidate sets of that stratum; problems with no set in a stratum are left out of that stratum's analysis. | A bootstrap over problems already carries the within-problem noise. A nested resample would double-count it (conservatively); it is run as a sensitivity check. |
| D10 | Holm families | Family 1: H1a, H1b, H2, H3, H4a, H4b. Family 2: H5a, H5b, H5c, H6, and the factorial contrasts only if that run is finished by its cut date. Both at 0.05. | |
| D11 | Priorities | Qwen3-0.6B reader: Must (the FLOP-accounting arm). Joint arm: Must, one seed. AggLM-1.7B checkpoint: Must if public (check in Stage 1), otherwise noted as unavailable. | The roadmap listed the joint arm and 0.6B as Must; the AggLM checkpoint's availability is unknown. |
| D12 | Generation stack | Every generation that is evaluated (candidate pools, readers, solver-GRPO samples, the contamination probe) uses vLLM 0.33.0 in fp16 on the RTX PRO 6000, as the dev pools did. Training is bf16. A bf16 checkpoint is cast to fp16 for evaluation after a sanity check (no NaN or inf; greedy agreement with bf16 on 200 prompts is reported). Training rollouts use the training framework's own engine and are never evaluated. | The dev pools are fp16; mixing precisions would put a stack difference inside the comparison. |
| D13 | Contamination probe | Runs in Stage 1 on the primary and secondary problem texts with Qwen3-1.7B and Qwen3-1.7B-Base: greedy completion of the last 40% of the statement from its first 60%, normalised exact match. It reads text only. No problem is removed because of it. H2 is additionally reported on the problems where the probe fails. | Independent of any arm or outcome. |
| D14 | Candidate handling | Candidates are never filtered by outcome. A truncated candidate is shown as generated (it has no answer: "(no final answer)" in answer-only). Each candidate is cut to its first 1,536 tokens inside the reader prompt (K = 8: the longest dev prompt is about 13.1k tokens, so `max_model_len` = 16,384; section 7). `Confidence: N` lines are removed from shown text, as the audit did for `full_solution`. Candidate order is a seeded shuffle. | 2608.18379 cut candidates at their first answer; this study shows them whole. |
| D15 | Candidate sets | K candidates drawn without replacement from the problem's pool by a seeded shuffle keyed on (problem, draw, K). Evaluation: 4 draws × 2 reader replicates per problem and arm. Strata c0, c1, c2+ count the correct candidates (0, 1, 2 or more); the label is computed after the draw and never used to choose sets for evaluation. | |
| D16 | Controls | Candidate-free: the reader template with the candidates block removed (same instructions otherwise). Answer-only: the K extracted answers. Placebo: for each real set, K donor candidates from other problems of the same subject and level, nearest in total token length, in the same slots; the stratum label is inherited from the replaced set. Duplicate: the first candidate repeated K times. Single-slot: one candidate, other slots empty. Deduplicated input: one candidate per distinct answer, in decreasing frequency, counts hidden. Answer swap: see section 4. | |
| D17 | Checkpoint rule | Best of steps 100, 200 and 300 by dev gain over matched SC (generated-only rule), ties to the later step; never the peak of a continuous curve. | Decided now so it cannot move after any test score exists. |
| D18 | Factorial cut | The factorial (pass@k solver, reader-marginal solver) starts only after every Must training arm and the Stage 3 evaluation are frozen. If it is not finished by **2027-01-17**, its contrasts leave family 2 untested and are reported as not run. | |
| D19 | MATH-P-Hard seeds | Problems that seeded MATH-P-Hard items are removed from the train pools; no test item is dropped. | Costs a few training problems, keeps all 279 test items. |
| D20 | Scoring | The boxed final answer, extracted and graded with the audit's `extract.py` and `grade.py` (including the confidence-box rule); no fallback. Candidates and readers are scored the same way. | 2608.18379's answer-only result flips with the scoring rule. |
| D21 | Test-set size | The primary set is fixed at its 972 problems; the confirm problems beyond it are not used. With the audit's paired SD (0.16 to 0.20), 972 problems give a 95% half-width of about 1.1 points and 80% power at about 1.6 points (the dev SD of the reader replaces this at the freeze). | Only 972 level-5 problems exist; level 3–4 would change the population. |
| D22 | Adaptive SC | Early-stopping SC (ESC) is the strongest zero-cost baseline: on dev it gains 1.7 to 3.7 points over plain SC at equal mean samples. The headline of H1a and H1b stays against plain SC at matched cost, as in the audit and in the trained-aggregator literature. Every H1 result is also reported against the ESC frontier at the reader's cost, as an estimate with its interval. H6 is tested against ESC. A vote weighted by stated confidence is reported and is not a hypothesis (it adds 0.0 to 0.2 points on dev). | `docs/dev_baselines.md`: at K = 8 the ESC frontier at 9 samples is 1.8 points above plain SC (1.5 on level 5), so a reader must beat the vote by 2.3 points on level 5 to tie it, not 0.8. |

## 4. Hypotheses and tests

Unit: the problem. Reader accuracy of a problem is the mean correctness over its 8 reader
generations (4 draws × 2 replicates). The matched-SC baseline of a problem is its exact SC
credit at the sample count k\* that costs as much as the reader's generations on that
problem (section 5), interpolated linearly between integers. If k\* falls outside the
problem's pool, that problem is "not estimable" under that rule and is reported.

**Family 1 (Part A).** Trained reader R, untrained reader U (same base model, same prompt),
K = 8 unless stated.

- **H1a.** Gain of R over base SC at matched generated tokens, on the primary set: CI lower
  bound above 0. Reported under the other cost rules as estimates.
- **H1b.** Gain of R over SC from the stronger solver-GRPO variant (chosen on dev between
  the equal-steps and the token-matched checkpoint) at matched generated tokens: CI lower
  bound above 0.
- **H2 (reads, not re-solves).** At c1: [accuracy(R, with candidates) − accuracy(R,
  candidate-free)] − [the same for U] has a CI lower bound above 0. Planning magnitude 10
  points. Secondary: the same difference over all strata; and the c0 gain of R stays at or
  below 0.
- **H3 (form effect).** In each stratum, accuracy(R, placebo) is within 2 points of
  accuracy(R, candidate-free): the lower bound of the CI of the difference is above −2
  points in c0, c1 and c2+ (an intersection test; its p-value is the largest of the three).
- **H4a (do no harm).** At c2+, accuracy(R) − plurality vote of the same K candidates has a
  CI lower bound above −1 point.
- **H4b (answer frequency).** Answer swap: for a set with at least two correct candidates
  and at least one wrong candidate, replace the final boxed answer of every candidate in the
  correct cluster by the plurality wrong answer, leaving the reasoning text unchanged. The
  swap drop is accuracy on the original set minus accuracy on the swapped set. H4b is the
  two-sided test of drop(R) − drop(U) at c2+. Predictions, recorded and not used for the
  test's direction: TTT-Consolidate predicts a larger drop after training (reliance on
  frequency grows); the audit's conformity result and our copy-rate hypothesis allow a
  smaller one. The copy rate (share of wrong outputs equal to a shown answer) and the
  deduplicated-input baseline are reported with intervals, not tested.

**Family 2 (Part B and the gate).**

- **H5a, H5b, H5c (grid).** Two-sided tests of what solver-GRPO candidates change relative to
  base candidates: (a) plurality-vote accuracy at K = 8; (b) the margin of R over the vote;
  (c) answer entropy of the candidate sets. Expectation recorded from RSA, SPIRAL and the
  Olmo diversity result: (a) up, (b) down, (c) down.
- **H6 (gate).** A unanimity-gated reader (vote if all K answers agree, otherwise read)
  beats early-stopping SC at matched mean cost under the prompt-inclusive rule; CI lower
  bound above 0. The ESC settings (window sizes 2, 4, 8) are frozen after the dev curves
  of section 7.
- **Factorial** (pass@k solver; reader-marginal solver trained against the frozen reader),
  each read by vote, U and R: contrasts defined at the freeze, in family 2 only if run
  before the cut date (D18).

**Exploratory, no Holm:** RSA (T ≤ 3) against matched SC; the joint arm's candidate-free gap
on its own weights; a learned gate or verification bit; collusion (the frozen reader's copy
rate on a new solver's wrong candidates); position and length bias; the K sweep; the
secondary, robustness and post-release sets.

## 5. Cost and matching

Per problem, with g the generated tokens and p the prompt tokens of each generation:

- **Generated-only.** B = Σ g(candidates) + g(reader); one SC sample costs its own g.
- **Prompt-inclusive.** B = Σ (p + g)(candidates) + w · (p + g)(reader); one SC sample costs
  p + g. w is the reader's parameter count over the solver's (1; about 0.35 for 0.6B; about 4
  for the 7B verifier).
- **GPU-seconds.** Measured at the Stage 2 pilot; descriptive.
- k\* = B / (cost of one SC sample of that problem). Each problem has at least 32 pool
  samples on the test sets, so that k\* up to about 17 (K = 8, prompt-inclusive) is
  estimable. The 600-problem dev pools have 18.

## 6. Statistics

As D9. One harness runs every baseline and arm; per-sample outputs are released. The new
contrast code is frozen with its hashes at Stage 2, extending the audit's freeze mechanism,
before any trained model is evaluated. The analysis runs once; a later correction is a
dated, labelled deviation published beside the original.

## 7. Stage 0 and dev analyses (record)

- **Stage 0 (2026-11-14), `docs/stage0/STAGE0_REPORT.md`.** Exact break-even and headroom on
  the 600 dev problems. Level 5, generated-only rule: K = 4 headroom 12.9 points, break-even
  2.3; K = 8 headroom 13.9, break-even 0.8. Prompt-inclusive break-even 6.4 (K = 4) and 3.6
  (K = 8). G1 passes at K = 8 (needed gain 2.3; expected 3.8 to 7.9). Assumes the reader
  writes one sample; the recovery range is AggLM's on thinking candidates.
- **Dev baselines (2026-11-14), `docs/dev_baselines.md`.** Computed on dev only, with no
  extra generation; they inform the freeze and are not tests. (1) A vote weighted by the
  stored `Confidence: N` adds +0.2 (K = 4) and +0.0 (K = 8) points over the plain vote.
  (2) Early-stopping SC beats plain SC at equal mean samples by 3.7 points at about 4
  samples and 1.7 at about 9.6; the ESC frontier at the cost of a K = 8 reader (9 samples)
  is 78.3% against 76.6% for plain SC (level 5: 63.3% against 61.8%), so a K = 8 reader
  must beat the vote by 2.3 points on level 5 to tie it (D22). Against that bar the G1
  expectation (3.8 to 7.9) passes at the high end and sits at the line (3.75 against 3.8)
  at the low end. (3) Reader prompts at K = 8 reach 13.1k tokens with candidates cut at
  1,536, so `max_model_len` = 16,384 (D14).

## 8. Stages and gates

| Stage | Work | Gate | Pass |
| --- | --- | --- | --- |
| 0 | Break-even on dev | G1 | Done, passes at K = 8 (D2) |
| 1 | Harness check; pools for every set (run quality only on sealed sets); contamination probe; untrained readers and baselines on dev and harness sets; log-probability features; paired SD on dev | **G2** | On AIME 2025 + HMMT 2025 with Qwen3-4B-Instruct-2507, K = 4, the four point-estimate signs of 2608.18379 are reproduced: aggregation − candidate-free below 0 at c0 and above 0 at c2+; placebo − candidate-free below 0 at c0; plurality vote above aggregation at c2+. The c2+ aggregation − candidate-free CI must also exclude 0. c1 is not judged (unresolved in the paper). Otherwise fix the prompt, extraction or truncation; no training until it passes. |
| 2 | Train pools; 20-step throughput pilot, 50-step GRPO pilot; GPU-seconds; seed rule (D7); **freeze** | **G3** | Stable pilot, throughput fits the budget; this file, the prompts, the analysis code and the set manifests committed and hashed |
| 3 | Reader × 3 (mixed K), solver-GRPO × 3 (step 300 kept), joint × 1, no-c0 × 1, 0.6B × 1; then evaluation of every arm on every set; analysis run once | G4 | H1a holds: frame as C1. Otherwise lead with C2 (what GRPO changes in a reader). |
| 4 | Factorial, transfer pass, write-up | cut date D18 | |

## 9. Not yet decided (frozen at Stage 2)

Reader prompt wording (drafts in `prompts/reader_v0/`; the owner reads real samples first);
GRPO hyperparameters and framework; the training decontamination rule; the train-set mix
(AggLM's easy-set share to be checked); ESC settings; the exact factorial contrasts; the
5-seed decision of D7; the transfer-pass prompts for Llama-3.2-3B and Qwen2.5-Math-1.5B
(plain prompt with a format gate).

## 10. Change log

- 2026-11-14: draft started; decisions D1 to D21 (section 3) taken by the coding agent at
  the owner's request; K = 8 agreed by the owner. Stage 0 recorded. Nothing sealed has been
  touched.
- 2026-11-14 (later): the dev baselines of section 7 were computed; D22 added (adaptive SC
  as a reported baseline and H6's comparator); D14's context window fixed at 16,384.
  Study sets built (`data/sets/`: dev 600, primary 972, reserve 1,678).
