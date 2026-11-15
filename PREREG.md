# Preregistration (DRAFT until the Stage 2 freeze)

Status: **draft**, started 2026-11-14 and revised the same day after the owner's review
(section 10, R1 to R10). It is completed, reviewed by the owner and committed before any
trained model is evaluated and before any sealed test set is scored or probed (the Stage 2
freeze, section 9). Until then it records decisions already taken, with their reasons, and
the evidence they rest on. Every change is dated and logged with what it replaced; nothing
is rewritten silently. In the revision of section 10 no margin, planning effect or
direction of a hypothesis was changed; the changes to a decision rule (D24), a wording
(H3) and a population (H4b) are listed there with their reasons.

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
| Dev | The audit's 600 confirmatory problems (199 / 193 / 208 at levels 3 / 4 / 5), 18 existing samples each (2 replicates × (3 round-0 + 6 pool), all with the audit's round-0 prompt) | 600 | Stage 0 and 1 analyses, prompt development, paired SD, checkpoint rule. Never a test set. | Read. Stage 0, the dev baselines and the design numbers of section 7 were computed on it. |
| Harness | AIME 2025 + HMMT Feb 2025 (MathArena) | 60 | Gate G2 with Qwen3-4B-Instruct-2507; reported, not powered | Pinned (`data/sets/harness.json`); G2 run on 2026-11-14, treated as passed by the owner on 2026-11-15 (R12) |
| Primary | MATH test level 5 minus every screen, pilot and dev problem | **972** (1,324 − 82 − 62 − 208) | H1 to H4 | **Sealed** |
| Secondary | OlympiadBench `OE_TO_maths_en_COMP` | 674 | Replication of H1; not in the Holm families | **Sealed** |
| Robustness | MATH-P-Hard | 279 | Perturbed level-5 problems | **Sealed** |
| Post-release | HMMT Nov 2025, AIME 2026, HMMT Feb 2026 | 93 | Contamination check; Bayesian intervals; not powered | **Sealed** |
| Train | MATH train levels 3–5 plus a DeepScaleR subset, decontaminated against every set above | 7,587 (R14) | Candidate pools and GRPO | Built 2026-11-15 (`data/train/`, **[delegated]**) |

- "Sealed" means: pools may be generated, and only run quality (overflow, truncation, unboxed
  rate) may be read, until the Stage 2 freeze. No accuracy of any kind, **and no model-based
  probe such as the contamination probe of D13**, is computed on a sealed set before then,
  not even with untrained models.
- Dev and the harness set are not sealed; protocols meant for the sealed sets may be
  rehearsed on them.
- The primary set is the 972 level-5 problems of the audit's confirm split that come after
  its first 600 in the frozen hash order. Membership depends on problem ids and levels only.
  The 1,678 level-3 and level-4 confirm problems after the first 600 are an unused reserve.
- The audit's problem split is rebuilt from the pinned MATH revision and checked against
  the audit's committed manifest; the new set manifests are in `data/sets/`.

## 3. Decisions of 2026-11-14

Rows marked "revised" were changed in the review of section 10; the log there gives what
each said before. D23 to D26 are new.

| ID | Question | Decision | Why |
| --- | --- | --- | --- |
| D1 | K for the headline test | **K = 8** (H1 and the main arms). K = 2, 4 and 8 are evaluated for every Must arm. | Stage 0: at K = 8 the generated-only break-even is 0.8 points and G1 passes at both ends of the recovery range; at K = 4 it passes only at the optimistic end. The owner agreed on 2026-11-14. |
| D2 | G1 verdict | **Pass at K = 8** on level-5 dev under the generated-only rule. Both recovery ends pass, so the choice of end no longer matters. | `docs/stage0/STAGE0_REPORT.md` |
| D3 | Cost rules | Generated tokens is the primary rule for H1a and H1b. Prompt-inclusive (parameter-weighted prompt + generated tokens, SC's own prompt counted) is reported for every test as a pre-registered estimate and is the rule of H6. Measured GPU-seconds is descriptive and is measured at the Stage 2 pilot. | The audit's primary rule was generated tokens. Under the prompt-inclusive rule a reader reading K candidates costs about 2K + 1 samples, so that is the rule a reviewer will use. |
| D4 | Pooling the solver clause over primary and secondary (revised, R1) | **No pooling.** The solver clause is its own test, H1b, on the primary set. The secondary set replicates H1a and H1b descriptively and is not in a Holm family. All non-dev, non-harness sets stay sealed through Stage 1, including from the contamination probe (D13). | Pooling makes the secondary set confirmatory and mixes populations with different accuracy; it also conflicts with using it in Stage 1. Sealing everything keeps Stage 1 from steering the design. |
| D5 | Hypothesis form (revised, R6) | H1a, H1b: superiority, decided by the Holm-adjusted p-value of D24, not by a stand-alone interval. Planning effect 1.5 points, used for sizing only. With SD 0.18 the power at 1.5 points is 74% for one unadjusted test and **48% at the Holm worst case** (section 6.3); a smaller true effect is reported as an estimate. | A "point estimate ≥ 1.5 and CI > 0" rule fails half the time at a true effect of 1.5. The earlier claim that H1b is "powered for 1.5 points" held only for one unadjusted test. |
| D6 | H4 direction (revised, R4) | H4a (do no harm): at c2+, reader non-inferior to the plurality vote, margin 1 point. H4b: **two-sided** test of the change in answer-swap drop after training, on the primary swap population of D25. The two competing predictions are recorded in section 4, but neither sets the test's direction. | The literature predicts RL raises reliance on answer frequency (TTT-Consolidate); the audit's conformity result suggests it could fall. With no basis for a direction, a one-sided test would be a coin flip dressed as a prediction. |
| D7 | Seeds (revised, R7) | Seeds per main arm (reader, solver-GRPO): **5** if the Stage 2 pilot measures at most 20 GPU-hours for a 300-step run, otherwise **3**. The rule is fixed now; the measured number and the resulting count go in the freeze. Stage 3's runs, budget and timeline (section 8.1) are written for both counts. Joint arm: 1 seed (Must), 2nd seed (Should). | 3 vs 3 gives only 20 relabellings, so a seed-level permutation test cannot go below p = 0.05 (one-sided); 5 vs 5 gives 252 (min p 0.004). |
| D8 | Seeds in the confirmatory test (revised, R7) | The test is conditional on the trained checkpoints: a problem-level paired bootstrap of the seed-averaged result. A hypothesis is **supported** only by its Holm-adjusted p (D24). H1a and H1b are called **robust** if, in addition, the gain is positive for every individual seed. Sign agreement across seeds is a robustness criterion and never a substitute for the test: a supported result with a negative seed is "supported, not robust", and an unsupported result stays unsupported whatever the seeds say. The seed-level permutation test and a seed-and-problem bootstrap are robustness checks, reported with their caveats. | Three seeds cannot estimate seed variance well; the per-seed sign rule is honest about that and nothing more. |
| D9 | Bootstrap (revised, R6) | Effect estimates: problem-level paired bootstrap on within-problem averages, 10,000 resamples, numpy `default_rng(0)`, percentile 95% intervals, as in the audit (its frozen `bootstrap_means` is reused). Strata are problem-level means over the candidate sets of that stratum; problems with no set in a stratum are left out of that stratum's analysis. **The decision is not the interval; it is D24.** | A bootstrap over problems already carries the within-problem noise. A nested resample would double-count it (conservatively); it is run as a sensitivity check. |
| D10 | Holm families (revised, R6) | Family 1: H1a, H1b, H2, H3, H4a, H4b (m = 6 always; a hypothesis not run counts with p = 1). Family 2: H5a, H5b, H5c, H6 (m = 4), and the factorial contrasts only if that run is finished by its cut date (m is then 4 plus their number). Holm within each family at 0.05 on the two-sided scale of D24. | |
| D11 | Priorities | Qwen3-0.6B reader: Must (the FLOP-accounting arm). Joint arm: Must, one seed. AggLM-1.7B checkpoint: Must if public (check in Stage 1), otherwise noted as unavailable. | The roadmap listed the joint arm and 0.6B as Must; the AggLM checkpoint's availability is unknown. |
| D12 | Generation stack | Every generation that is evaluated (candidate pools, readers, solver-GRPO samples, the contamination probe) uses vLLM 0.33.0 in fp16 on the RTX PRO 6000, as the dev pools did. Training is bf16. A bf16 checkpoint is cast to fp16 for evaluation after a sanity check (no NaN or inf; greedy agreement with bf16 on 200 prompts is reported). Training rollouts use the training framework's own engine and are never evaluated. The G2 harness model (Qwen3-4B-Instruct-2507) runs in bf16: it is in no evaluated comparison. | The dev pools are fp16; mixing precisions would put a stack difference inside the comparison. |
| D13 | Contamination probe (revised, R1) | The probe runs **after the Stage 2 freeze**, in the Stage 3 evaluation window and before the frozen analysis, on the primary and secondary problem texts. Its protocol is frozen at Stage 2 and implemented in `src/reader_audit/contamination.py`: prefix = the first 60% of the statement's tokens (floor, the model's own tokenizer, question text alone, no template); raw continuation, greedy, at most ceil(1.2 × remainder tokens) + 16 new tokens; a problem is **memorised** by a model if the completion, whitespace removed, starts with the remainder, whitespace removed. Two models are probed and **used separately**: the *studied* model Qwen3-1.7B (source of the candidates, the reader's initialisation and the solver baselines) gives the **primary flag**; Qwen3-1.7B-Base is reported beside it as pre-training exposure; "memorised by either" is a sensitivity split. H2's confirmatory test uses all primary problems; as secondary reports the same H2 estimate is given on the problems the primary flag does not mark, and under the "either" flag. No problem is removed because of the probe. Before the freeze only a rehearsal on dev and the harness set is allowed. | A model-based probe on a sealed set is a computation on that set, which section 2 forbids before the freeze; nothing before the freeze depends on it (it only splits a secondary H2 report). The studied model is the one whose behaviour the claims are about; the base model measures where exposure came from. |
| D14 | Candidate handling (revised, R2) | Candidates are never filtered by outcome. **Two kinds of truncation are kept apart.** *Generation-truncated*: the candidate hit `max_tokens` = 2,048 (7.8% of dev candidates). It has no final answer unless a complete box came earlier, counts as wrong in the vote, is shown as generated, and appears as "(no final answer)" in answer-only. *Prompt-truncated*: a candidate longer than the cap of 1,536 tokens (13.6% of dev candidates: the 7.8% above plus 5.8% that finished within the limit) is cut **in the middle** for display. Its last 256 tokens are kept, so a finished candidate keeps its conclusion and final answer; a marker line follows the first part; the first part takes what is left, so the displayed text, **marker included, is at most 1,536 tokens** (about 1,270 tokens of the start; the marker is 10 Qwen3 tokens). Every plan row records, per prompt, how many shown candidates were generation-truncated, prompt-clipped and answer-hidden (the shown text no longer yields the candidate's graded answer); these counts are reported by arm and stratum. `Confidence: N` lines are removed from shown text, as the audit did for `full_solution`. Candidate order is a seeded shuffle. K = 8 prompts reach about 13.1k tokens, so `max_model_len` = 16,384 (section 7). | 2608.18379 cut candidates at their first answer; this study shows them whole, and cutting the end would hide the final answer the reader is meant to weigh. Implemented in `controls.middle_clipper` and `arms.py`. |
| D15 | Candidate sets (revised, R9) | K candidates drawn without replacement from the problem's pool by a seeded shuffle keyed on (problem, draw, K). Evaluation: 2 reader replicates per cell. **Draws per problem on the primary set: 4, 8 or 16, fixed at the freeze by the power rule of section 6.3** (provisionally 16, see O1); 4 draws on the secondary, robustness and post-release sets (replications, descriptive). Strata c0, c1, c2+ count the correct candidates (0, 1, 2 or more); the label is computed after the draw and never used to choose sets for evaluation. The one set-level rule that reads candidate correctness is the answer swap's population (D25), which is a rule on candidates, not on any outcome. | The review's power recalculation (section 6.3) shows that with 4 draws the stratum-wise hypotheses are badly underpowered under Holm. |
| D16 | Controls | Candidate-free: the reader template with the candidates block removed (same instructions otherwise). Answer-only: the K extracted answers. Placebo: for each real set, K donor candidates from other problems of the same subject and level, nearest in total token length, in the same slots; the stratum label is inherited from the replaced set. Duplicate: the first candidate repeated K times. Single-slot: one candidate, other slots empty. Deduplicated input: one candidate per distinct answer, in decreasing frequency, counts hidden. Answer swap: D25. | |
| D17 | Checkpoint rule | Best of steps 100, 200 and 300 by dev gain over matched SC (generated-only rule), ties to the later step; never the peak of a continuous curve. | Decided now so it cannot move after any test score exists. |
| D18 | Factorial cut | The factorial (pass@k solver, reader-marginal solver) starts only after every Must training arm and the Stage 3 evaluation are frozen. If it is not finished by **2027-01-17**, its contrasts leave family 2 untested and are reported as not run. | |
| D19 | MATH-P-Hard seeds | Problems that seeded MATH-P-Hard items are removed from the train pools; no test item is dropped. | Costs a few training problems, keeps all 279 test items. |
| D20 | Scoring | The boxed final answer, extracted and graded with the audit's `extract.py` and `grade.py` (including the confidence-box rule); no fallback. Candidates and readers are scored the same way. | 2608.18379's answer-only result flips with the scoring rule. |
| D21 | Test-set size and power (revised, R6) | The primary set is fixed at its 972 problems; the confirm problems beyond it are not used. With the audit's paired SD (0.16 to 0.20) 972 problems give a 95% half-width of about 1.1 points. The smallest true effect found with 80% power is **1.6 points for one unadjusted test (the figure first given) and 2.0 points at the Holm worst case of Family 1** (0.05 / 6 on the two-sided scale; 1.8 to 2.2 for SD 0.16 to 0.20). The dev SD of the reader replaces the placeholder at the freeze (section 6.3). | Only 972 level-5 problems exist; level 3–4 would change the population. The first figure used the unadjusted threshold of a single test. |
| D22 | Adaptive SC | Early-stopping SC (ESC) is the strongest zero-cost baseline: on dev it gains 1.7 to 3.7 points over plain SC at equal mean samples. The headline of H1a and H1b stays against plain SC at matched cost, as in the audit and in the trained-aggregator literature. Every H1 result is also reported against the ESC frontier at the reader's cost, as an estimate with its interval. H6 is tested against ESC. A vote weighted by stated confidence is reported and is not a hypothesis (it adds 0.0 to 0.2 points on dev). | `docs/dev_baselines.md`: at K = 8 the ESC frontier at 9 samples is 1.8 points above plain SC (1.5 on level 5), so a reader must beat the vote by 2.3 points on level 5 to tie it, not 0.8. |
| D23 | SC and ESC estimation on test sets (new, R5) | **Dev pools (18 samples): exact**, by summing over answer-count vectors with hypergeometric weights (checked against explicit subset enumeration). **Test pools (32 samples): Monte Carlo, not exact.** For each problem and each integer k = 1..32, R = 2,000 subsets of k samples are drawn uniformly without replacement; the generator is numpy `default_rng` seeded with the first 8 bytes of the sha256 of a fixed salt, the problem id and k, so every arm, model and re-run uses the same subsets; k = 32 is the whole pool and exact. The credit of a subset is the plurality credit (unparsable answers never vote, an m-way tie containing the right answer earns 1/m, answers pooled by the audit's equivalence), averaged over the subsets. At a fractional k\* the credit is interpolated linearly between the estimates at ⌊k\*⌋ and ⌈k\*⌉. A problem whose k\* falls outside [1, 32] is **not estimable** under that cost rule: it is dropped from the tests of that rule, the number dropped is reported, and if more than 2% of the primary problems are dropped under a rule a sensitivity analysis that clips k\* at 32 is reported beside it. ESC (D22) is estimated the same way: 500 random orders of the pool per problem, seeded from the salt, the problem id and the ESC setting; the settings (windows 2, 4, 8; caps 8, 16) are fixed at the freeze. The Monte Carlo standard error is at most 0.011 per estimate and under 0.04 points on a mean over 972 problems, so the bootstrap over problems does not model it. | The review: a 32-sample pool cannot be called exactly enumerated. The exact method takes 0.04 s (9 answer groups, k = 8) to 2.3 s (14 groups, k = 16) per problem and k, hours per set (measured). Implemented in `sc_estimate.py`. |
| D24 | Inferential procedure (new, R6) | Per problem a paired difference d; B = 10,000 bootstrap resamples of problems give means m\*. **p-values on the two-sided scale**: superiority (null: mean ≤ 0) p = 2 (1 + #{m\* ≤ 0}) / (B + 1); non-inferiority at margin δ (null: mean ≤ −δ) p = 2 (1 + #{m\* ≤ −δ}) / (B + 1); two-sided (null: mean = 0) the audit's two-sided p; all capped at 1. H3 is one pooled non-inferiority test over all sets (R11; before R11 an intersection test over three strata). **Holm** within each family of D10 on these p-values; a hypothesis is **supported iff its adjusted p < 0.05**. Intervals are effect estimates, never the decision. The audit's frozen `bootstrap_p` and `holm_adjust` do the arithmetic. For a single hypothesis p < 0.05 on this scale is the same as its 95% interval excluding the null; for Family 1 the smallest p must pass 0.05 / 6 = 0.0083. Calibration is tested by simulation: type I error about 2.5% for the one-sided tests at the edge of their null, 5% for the two-sided one. | The review: an interval above zero is not a Holm-corrected test. Doubling the one-sided p keeps directional and two-sided hypotheses on one scale and keeps D5's original meaning for a single test. Implemented in `inference.py`. |
| D25 | Answer swap (new, R4) | **Primary population**: candidate sets with (i) at least one wrong parsable answer and (ii) the correct answer as the **strict plurality** of the original set (more candidates than any wrong answer; a tie does not count). Such sets have at least two correct candidates. **Swap**: every candidate with a correct graded answer has the content of its final boxed answer (the box the audit's extractor returns) replaced by the target; nothing else changes, so the derived value stays visible in the reasoning. **Target**: the most frequent wrong answer, answers pooled by the audit's equivalence; a tie goes to the answer shown first (the order is the seeded shuffle of D15); candidates without a final answer are not wrong answers, never vote and are left as they are. After the swap the target is the strict plurality, so a vote flips from right to wrong. **Drop** = accuracy(reader_full, original set) − accuracy(answer_swap, swapped set), on the same (problem, draw) cells. H4b = two-sided test of drop(R) − drop(U) (problem level). **Secondary** (reported with intervals, not in the family): the broad population, every c2+ set with at least one wrong parsable answer. On dev (level 5, K = 8) 38.5% of sets are in the primary population (52% of problems have one in 4 draws) and 46.7% in the broad one. | The review: sets where the correct answer was tied or outvoted before the swap make a muddled counterfactual. Implemented in `arms.swap_plan` and `reader_eval.swap_cells`; the plan row of every swapped prompt records `correct_plurality`. |
| D26 | Framing and reporting (new, R8) | Section 4.1: the framing of the study depends on H1a and H1b separately; every result is reported with the baselines and cost rules it was tested against. | The review: H1a alone does not establish superiority over the solver-GRPO baseline. Implemented in `decision.py`. |

## 4. Hypotheses and tests

Unit: the problem. A problem's accuracy under an arm is the mean correctness over its
reader generations in the cells considered (on the primary set: the draws of D15 × 2
replicates). The matched-SC baseline of a problem is its SC credit at the sample count k\*
that costs as much as the reader's generations on that problem (section 5): exact on dev
pools, a Monte Carlo estimate on the 32-sample test pools (D23), interpolated linearly
between integers. A problem whose k\* falls outside its pool is "not estimable" under that
rule, is dropped from the tests of that rule and is counted. **The decision rule of every
hypothesis is D24**: supported iff the Holm-adjusted p-value is below 0.05.

**Family 1 (Part A, m = 6).** Trained reader R, untrained reader U (same base model, same
prompt), K = 8 unless stated.

- **H1a.** Superiority of R over base SC at matched generated tokens, on the primary set
  (null: the mean gain is at most 0). Reported under the other cost rules and against the
  ESC frontier (D22) as estimates.
- **H1b.** Superiority of R over SC from the stronger solver-GRPO variant (chosen on dev
  between the equal-steps and the token-matched checkpoint) at matched generated tokens.
- **H2 (reads, not re-solves).** Superiority, at c1, of [accuracy(R, with candidates) −
  accuracy(R, candidate-free)] − [the same for U]. Planning magnitude 10 points. Secondary:
  the same difference over all strata; and the c0 gain of R stays at or below 0.
- **H3 (form effect), a non-inferiority hypothesis (revised, R11).** The trained reader
  under **placebo inputs must not be more than 2 percentage points worse than its own
  candidate-free solve**: the null is mean(accuracy(R, placebo) − accuracy(R,
  candidate-free)) ≤ −0.02, tested one-sided at margin 0.02 as **one pooled test over all
  sets** (problem level). The three strata (c0, c1, c2+) are reported as estimates with
  intervals, not tested. It is **not** an equivalence test: a placebo that helps by more
  than 2 points does not fail it. (Before R11 it was an intersection of three stratum
  tests; `inference.intersection` is kept for such uses.)
- **H4a (do no harm), a non-inferiority hypothesis.** At c2+, the null is mean(accuracy(R)
  − plurality vote of the same K candidates) ≤ −0.01 (margin 1 point).
- **H4b (answer frequency).** Two-sided test of drop(R) − drop(U) on the primary swap
  population of D25. Predictions, recorded and not used for the test's direction:
  TTT-Consolidate predicts a larger drop after training (reliance on frequency grows);
  the audit's conformity result and our copy-rate hypothesis allow a smaller one. The copy
  rate (share of wrong outputs equal to a shown answer), the broad swap population and the
  deduplicated-input baseline are reported with intervals, not tested.

**Family 2 (Part B and the gate, m = 4 plus any factorial contrasts).**

- **H5a, H5b, H5c (grid).** Two-sided tests of what solver-GRPO candidates change relative
  to base candidates: (a) plurality-vote accuracy at K = 8; (b) the margin of R over the
  vote; (c) answer entropy of the candidate sets. Expectation recorded from RSA, SPIRAL and
  the Olmo diversity result: (a) up, (b) down, (c) down.
- **H6 (gate).** Superiority of a unanimity-gated reader (vote if all K answers agree,
  otherwise read) over early-stopping SC at matched mean cost under the prompt-inclusive
  rule. The ESC settings (window sizes 2, 4, 8) are frozen after the dev curves of
  section 7.
- **Factorial** (pass@k solver; reader-marginal solver trained against the frozen reader),
  each read by vote, U and R: contrasts defined at the freeze, in family 2 only if run
  before the cut date (D18).

**Exploratory, no Holm:** RSA (T ≤ 3) against matched SC; the joint arm's candidate-free gap
on its own weights; a learned gate or verification bit; collusion (the frozen reader's copy
rate on a new solver's wrong candidates); position and length bias; the K sweep; the
secondary, robustness and post-release sets.

### 4.1 Conclusion and reporting rules (D26)

1. **Words.** "Supported" means a Holm-adjusted p below 0.05 within the hypothesis's
   family. "Robust" additionally requires the claimed sign in every seed (D8). Anything
   else is an estimate and is called one.
2. **Framing of the study (gate G4).**
   - H1a supported and H1b supported: the full C1 claim, against base SC and against SC
     from a solver trained on the same tokens.
   - H1a supported, H1b not supported: a narrower claim against base SC only. The paper
     states that solver-GRPO matches the reader, and leads with C2 and C3 for what the
     reader adds.
   - H1a not supported: lead with C2 (what GRPO changes in a reader), whatever H1b says.
3. **Every result is reported** with its interval and its adjusted p; H1a and H1b under all
   three cost rules and against ESC; per-seed estimates; the counts of not-estimable
   problems; the truncation counters of D14 by arm and stratum. The abstract names which
   baselines were beaten. A null result gets the same prominence as a positive one.
4. **Secondary and exploratory results are labelled as such** and never replace a
   primary one. A correction after the analysis is a dated, labelled deviation published
   beside the original.

## 5. Cost and matching

Per problem, with g the generated tokens and p the prompt tokens of each generation:

- **Generated-only.** B = Σ g(candidates) + g(reader); one SC sample costs its own g.
- **Prompt-inclusive.** B = Σ (p + g)(candidates) + w · (p + g)(reader); one SC sample costs
  p + g. w is the reader's parameter count over the solver's (1; about 0.35 for 0.6B; about 4
  for the 7B verifier).
- **GPU-seconds.** Measured at the Stage 2 pilot; descriptive.
- k\* = B / (cost of one SC sample of that problem), computed from the problem's own tokens
  (`budget.k_star`). Each problem has at least 32 pool samples on the test sets, so k\* up
  to about 17 (K = 8, prompt-inclusive) is estimable; the 600-problem dev pools have 18.
- SC and ESC are **exact on dev and Monte Carlo on the test sets**, with the parameters,
  seeds, interpolation and the treatment of problems with k\* beyond the pool fixed in D23.

## 6. Statistics

### 6.1 Procedure

Effect estimates and the decision rule are D9 and D24; the framing of conclusions is
section 4.1. One harness runs every baseline and arm; per-sample outputs are released. The
new contrast code is frozen with its hashes at Stage 2, extending the audit's freeze
mechanism, before any trained model is evaluated. The analysis runs once; a later
correction is a dated, labelled deviation published beside the original.

### 6.2 Seeds

The reader and solver-GRPO run with 3 or 5 seeds (D7). Inference is conditional on the
trained checkpoints (D8). The seed-level exact permutation test has 20 relabellings at 3
vs 3 (smallest one-sided p 0.05) and 252 at 5 vs 5 (0.004); it is a robustness check. Sign
agreement across seeds is a robustness criterion, not a test.

### 6.3 Power under the Holm family (revised, R6)

Under Holm the smallest of the six Family 1 p-values faces 0.05 / 6 = 0.0083 on the
two-sided scale, the next 0.05 / 5, and so on; 0.0083 is the worst case for any one
hypothesis. The table is a normal approximation with **placeholder standard deviations**:
0.18 for H1 (the audit's paired SD was 0.16 to 0.20), and for the stratum-wise
hypotheses a binomial model (p = 0.5, no heterogeneity of effects between problems) whose
inputs are the dev design numbers of `docs/design_numbers_dev.json`. The freeze replaces
every placeholder with the SD measured on dev with the untrained reader. The full table is
`docs/power_table.md` (`scripts/06_power.py`); effects are the planning effects of the
hypotheses, a true difference of 0 for the non-inferiority tests.

| Hypothesis (effect or margin) | Problems, 4 / 16 draws | MDE at Holm worst, 4 / 16 | Power at Holm worst, 4 / 16 | Power, one unadjusted test, 4 / 16 |
| --- | --- | --- | --- | --- |
| H1a, H1b (+1.5 points, SD 0.18) | 972 | 2.0 / 2.0 | 48% / 48% | 74% / 74% |
| H1a with the binomial-noise SD (0.177 / 0.088) | 972 | 2.0 / 1.0 | 50% / 99.6% | 75% / 100% |
| H2, c1 (+10 points) | 215 / 299 | 11.9 / 6.6 | 62% / 99.6% | 84% / 100% |
| **H3, one pooled test over all sets (adopted, R11; non-inferior within 2 points, true 0)** | 972 | 2.8 / 1.4 | 44% / 99% | 70% / 99.9% |
| *(superseded) stratum tests, for reference:* H3 c0 | 327 / 393 | 5.6 / 2.8 | 8% / 45% | 24% / 71% |
| *(superseded)* H3 c1 | 215 / 299 | 8.4 / 4.6 | 4% / 13% | 13% / 32% |
| *(superseded)* H3 c2+ | 682 / 720 | 3.5 / 1.7 | 26% / 92% | 51% / 98% |
| *(superseded)* H3 intersection of the three strata | | | 0.1% / 5% | 2% / 22% |
| H4a, c2+ (non-inferior within 1 point, true 0) | 682 / 720 | 2.5 / 1.2 | 11% / 59% | 29% / 82% |
| H4b, primary swap population (two-sided) | 505 / 551 | 6.4 / 3.2 | n/a | n/a |

Findings (dev design numbers, K = 8, level 5; 22% of problems have a c1 set in 4 draws and
31% in 16, because most problems' pools hold either many or few correct samples):

- H1a and H1b are powered at their planning effect of 1.5 points only to 48% at the Holm
  worst case. The effects that Stage 0 makes plausible (3.0 to 7.1 points over SC) give
  at least 98% even for SD 0.20. More draws lower H1's SD toward its binomial-noise value.
- H2, H3 (c0, c1) and H4a are badly underpowered with 4 draws; 16 draws repair H2 and H3
  (c2+) and bring H4a to 59% at the worst case (82% unadjusted).
- **H3 as an intersection over three strata was not testable** (5% power even with 16
  draws, because c1 sets are rare: a property of the design, not of the placebo's
  effect). R11 therefore adopts one pooled test (99% at 16 draws, 44% at 4).
- With 4 draws H4a's margin of 1 point would need about 4,200 problems for 80% power at
  the worst case; with 16 draws its SD falls to 0.094 and the 720 available problems
  reach 59%.

**Sizing rule (fixed now).** The number of evaluation draws on the primary set is the
smallest d in {4, 8, 16} for which, with the SDs measured on dev, H1a, H1b (1.5 points),
H2 (10 points) and H4a (true difference 0 inside its margin) each have at least 70% power
at the Holm worst case; if no d reaches 70% for a hypothesis, d = 16 and the freeze lists
that hypothesis as underpowered with the owner's decision recorded. H3 follows the form
the owner chooses in O2. The rule fixes a design parameter, never a hypothesis, margin or
planning effect.

### 6.4 Where each rule is implemented

| Rule | Code |
| --- | --- |
| D23 SC and ESC estimation, k\* | `sc_estimate.py`, `subsets.py`, `baselines.py`, `budget.py` |
| D24 p-values, Holm, intersection tests | `inference.py` |
| D25 answer swap, D14 clipping and counters | `arms.py`, `controls.py`, `reader_eval.py`, `clipper.py` |
| D13 contamination protocol | `contamination.py` |
| D7, D8, D26 seeds, verdict words, framing | `decision.py` |
| Section 6.3 power, section 8.1 schedule | `power.py`, `schedule.py`, `scripts/05_design_numbers.py`, `scripts/06_power.py` |

Each has tests in `tests/`; the numbers of this file that come from code (power table,
budget table) are reproduced by the scripts above.

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
  1,536 tokens, so `max_model_len` = 16,384 (D14).
- **Design numbers (2026-11-14), `docs/design_numbers_dev.json`.** K = 8, level 5: sets are
  25.4% c0, 9.0% c1 and 65.6% c2+; 38.5% are in the primary swap population and 46.7% in
  the broad one. 13.6% of candidates are cut for display, of which 5.8 points finished
  within the generation limit; 7.8% hit the generation limit. They feed D14, D25 and
  section 6.3.
- **Untrained reader on dev (2026-11-15), `docs/dev_reader.json`, `scripts/12_dev_reader.py`
  and `13_dev_reader_analysis.py`.** Qwen3-1.7B, untrained, draft prompt, non-thinking,
  temperature 0.7, K = 8, candidates cut at 1,536 tokens, the 208 level-5 dev problems, 16
  seeded sets per problem, 2 replicates per set, arms `reader_full` and `candidate_free`
  (13,312 generations, 8 shards of 26 problems, 1.2 GPU-hours on the RTX PRO 6000; raw
  data in `Arshia-HZ/trained-reader-audit-raw`, `dev_reader_qwen3-1.7b/`). Descriptive and
  dev only; the SDs are those of the *untrained* reader and a trained reader's will differ.
  (1) The reader writes 728 tokens per read (a pool sample: about 820), so its matched
  self-consistency count is k\* = 8.75 under the generated-only rule, not 9. (2) Untrained
  reader minus self-consistency at the matched k\*: −3.2 points (SD 0.152 over problems,
  16 draws). (3) c2+ sets: untrained reader minus the plain vote −7.4 points (SD 0.172,
  154 problems, 74% of problems have a c2+ set); the 4B reader of G2 was level with the
  vote, so the 1.7B reader is a weaker reader. (4) c1 sets: the reader with candidates
  minus its candidate-free solve +14.5 points (SD 0.370; 64 of 208 problems, 31%, have a
  c1 set at 16 draws, 22% at 4 draws). (5) The sizing rule of section 6.3 with these
  numbers (draws from {4, 8, 16}; H1b borrows H1a's SD; the difference-in-differences of
  H2 is taken to have √2 times the single-arm SD, both stated assumptions): no number of
  draws reaches 70% power at the Holm worst case for H1a and H1b (67% at 16 draws), H4a
  (14%); H2 reaches 75% at 16 draws. The rule's fallback applies: **16 draws, with H1a, H1b
  and H4a listed as underpowered for the owner's decision at the freeze** (O3, O4).

## 8. Stages and gates

| Stage | Work | Gate | Pass |
| --- | --- | --- | --- |
| 0 | Break-even on dev | G1 | Done, passes at K = 8 (D2) |
| 1 | Harness check; pools for every set (run quality only on sealed sets); untrained readers and baselines on dev and harness sets, including the dev SDs of section 6.3; log-probability features; rehearsal of the contamination protocol on dev and the harness set only | **G2** | On AIME 2025 + HMMT 2025 with Qwen3-4B-Instruct-2507, K = 4, the four point-estimate signs of 2608.18379 are reproduced: aggregation − candidate-free below 0 at c0 and above 0 at c2+; placebo − candidate-free below 0 at c0; plurality vote above aggregation at c2+. The c2+ aggregation − candidate-free CI must also exclude 0. c1 is not judged (unresolved in the paper). Otherwise fix the prompt, extraction or truncation; no training until it passes. **Outcome (R12): the first three signs held; the fourth (vote above aggregation at c2+) did not, because the vote is at ceiling there (99.2% against 99.5%); the owner treated G2 as passed on 2026-11-15, with the rule above left unchanged.** |
| 2 | Train pools; 20-step throughput pilot, 50-step GRPO pilot; GPU-seconds; seed rule (D7); draws by the sizing rule of section 6.3; **freeze** (including the probe protocol, D13) | **G3** | Stable pilot, throughput fits the budget; this file, the prompts, the analysis code and the set manifests committed and hashed |
| 3 | Reader × n_s, solver-GRPO × n_s (step 300 kept), joint × 1, no-c0 × 1, 0.6B × 1, with **n_s = 3 or 5 by D7**; then the contamination probe (D13) and the evaluation of every arm on every set; analysis run once | **G4** | The framing rule of section 4.1: H1a and H1b supported → full C1; H1a only → narrower C1, with the solver-GRPO caveat; H1a not supported → lead with C2. |
| 4 | Factorial, transfer pass, write-up | cut date D18 | |

### 8.1 Stage 3 budget and timeline as a function of the seed count (revised, R7)

GPU-hours, the roadmap's per-run extrapolations (15 to 30 for a reader run; solver-GRPO
matched on rollout plus pool tokens 1.3 to 1.7 times that), **unmeasured**; the Stage 2
pilot replaces them. For five seeds the high end uses the 20-hour threshold that
triggered the rule (reader runs at most 20 h, solver runs at most 34 h). Evaluation hours
scale with the number of checkpoints; with 16 draws on the primary set the evaluation is
estimated at about 35 to 50 of them (unmeasured). `schedule.py` computes the table.

| Item | n_s = 3 | n_s = 5 |
| --- | --- | --- |
| Reader, mixed K | 45–90 | 75–100 |
| Solver-GRPO | 60–150 | 100–170 |
| Joint (1 seed) | 17.5–35 | 17.5–35 |
| Reader without c0 sets (1 seed) | 15–30 | 15–20 |
| Qwen3-0.6B reader (1 seed) | 8–15 | 8–15 |
| Pass@k solver (1 seed, Stage 4) | 15–30 | 15–30 |
| Reader-marginal solver (2 seeds, Stage 4) | 50–120 | 50–120 |
| Pools (test at 32 samples, train at 16) | 24 | 24 |
| Evaluation | 48–96 | 69–139 |
| **Total** | **≈ 283–590 (1.7–3.5 GPU-weeks)** | **≈ 374–653 (2.2–3.9 GPU-weeks)** |

Planning weeks, counted from the start of Stage 0 (week 1, 2026-11-12), assuming
uninterrupted GPU access: with 3 seeds training is weeks 3–5, evaluation week 6 and Stage 4
weeks 7–8 (to about 2027-01-06); with 5 seeds training is weeks 3–6, evaluation week 7 and
Stage 4 weeks 8–9 (to about 2027-01-13). The factorial's cut date, 2027-01-17, is binding
either way and the factorial is cut first.

**Compute available (owner, 2026-11-14).** One RTX PRO 6000 (Blackwell) in short sessions
of about 10 hours, not long continuous access. Consequences: every job is sharded,
resumable and uploaded when it finishes a shard; the order is the main runs first (G2, the
pools of the sets the main hypotheses need, then the reader and solver-GRPO runs and their
evaluation), and the extras (joint arm, 0.6B reader, no-c0 run, secondary, robustness and
post-release sets, K sweep beyond K = 8 and 4, factorial) are decided after the main
runs, not scheduled now. The weeks of the timeline above assume continuous access and
will stretch accordingly; the cut date of the factorial still binds.

## 9. Not yet decided

**Frozen at Stage 2:** reader prompt wording (drafts in `prompts/reader_v0/`; the owner
reads real samples first); GRPO hyperparameters and framework; the training decontamination
rule; the train-set mix (AggLM's easy-set share to be checked); ESC settings; the exact
factorial contrasts; the seed count (D7) and the number of draws (section 6.3), both from
measurements; the SDs of section 6.3; the transfer-pass prompts for Llama-3.2-3B and
Qwen2.5-Math-1.5B (plain prompt with a format gate).

**Open for the owner before the freeze.** The review's power recalculation (section 6.3)
shows four things I did not change, because each would alter a design parameter, a margin
or a hypothesis. No hypothesis, margin or planning effect has been changed.

- **O1 and O2 were accepted by the owner on 2026-11-14 (R11); O3 and O4 stay as stated.**
- **O1. Evaluation draws (accepted).** Recommended: 16 draws per problem on the primary set (4
  elsewhere). It repairs H2 (62% to 99.6% power at the worst case), H3 on c2+ and H4a (11%
  to 59%), and cuts H1's noise. Estimated cost: about 35 to 50 GPU-hours of the evaluation
  budget (unmeasured). D15 carries this as the provisional value of the sizing rule.
- **O2. H3 (accepted: option a).** The intersection of three non-inferiority tests (2 points each) has 5% joint
  power at 16 draws because c1 sets are rare. Options: (a) one pooled non-inferiority test
  over all sets, strata reported as estimates (99% power at 16 draws; recommended); (b)
  test c0 and c2+ only, c1 as an estimate (about 41% joint at the worst case); (c) keep as
  is and accept that H3 cannot be supported.
- **O3. H4a margin.** At 4 draws a margin of 1 point has 11% power at the worst case; a
  margin of 2.5 points would give 81%. At 16 draws the 1-point margin reaches 59%. If O1
  is accepted I would keep 1 point. **Update 2026-11-15 (R13), with the SD measured on dev
  (0.172 at 16 draws, 720 problems):** the 1-point margin has 14% power at the Holm worst
  case; margins of 2, 2.5 and 3 points give 69%, 90% and 98%. A margin of about 2 points
  would reach the 70% target. **Owner's decision 2026-11-15: keep 1 point for now and
  revisit at the freeze with the pilot's SD of the trained reader.**
- **O4. H1 planning effect.** The planning effect of 1.5 points is at 48% power under Holm
  with SD 0.18; the effects Stage 0 makes plausible give at least 98%. No change proposed;
  stated so that the planning effect is not mistaken for the expected one. **Update
  2026-11-15 (R13), SD measured 0.152 at 16 draws:** 1.5 points has 67% power at the worst
  case; 1.54 points would give 70%, 2 points 93%. **Owner's decision 2026-11-15: keep
  1.5 points for now and revisit at the freeze with the pilot's SD.** (O3 and O4 are
  therefore deferred, not closed: the freeze must state them again with the trained
  reader's measured SD.)

## 10. Change log

- 2026-11-14: draft started; decisions D1 to D21 (section 3) taken by the coding agent at
  the owner's request; K = 8 agreed by the owner. Stage 0 recorded. Nothing sealed has been
  touched.
- 2026-11-14 (later): the dev baselines of section 7 were computed; D22 added (adaptive SC
  as a reported baseline and H6's comparator); D14's context window fixed at 16,384.
  Study sets built (`data/sets/`: dev 600, primary 972, reserve 1,678).
- 2026-11-14 (later still): the harness set is pinned (`data/sets/harness.json`: AIME 2025
  and HMMT Feb 2025, 30 problems each, at Hub revisions with content hashes). The G2 job
  (`scripts/10_g2_harness.py`) is written and rehearsed on the mock backend; the first GPU
  run has not happened. A candidate longer than the cap is cut in the middle (first cap − 256
  tokens, last 256, with a marker), not at its end, so its final answer stays visible
  (changes D14's wording from "first 1,536 tokens"; reader prompt drafts in
  `prompts/reader_v0/`).
- **2026-11-14, after the owner's review: R1 to R10.** Each entry gives what was there,
  what is there now, and why. Margins, planning effects and the directions of the
  hypotheses are unchanged; the changes to a population (H4b), a decision rule (D24) and
  a wording (H3) are stated in their entries.
  - **R1, contamination probe (D13, sections 2, 8).** *Was:* run in Stage 1 on the primary and
    secondary problem texts, with Qwen3-1.7B and Qwen3-1.7B-Base, without a rule for
    combining them. *Now:* run after the Stage 2 freeze, protocol and H2 split fixed at the
    freeze, the studied model's flag primary, the base model reported apart, "either" as a
    sensitivity split; rehearsal on dev and the harness set only. *Why:* section 2 forbids
    any computation on a sealed set before the freeze, and the probe is one.
  - **R2, candidate truncation (D14).** *Was:* "cut to its first 1,536 tokens" (the change
    log of the same day already said middle cut: the two disagreed); the marker was extra,
    so a cut candidate had 1,545 tokens. *Now:* generation-truncated and prompt-truncated
    candidates are separate; the middle cut keeps the last 256 tokens; the marker is inside
    the 1,536 (about 1,270 tokens of the start); the plan rows count both kinds and the
    answers the cut hides. *Why:* consistency with the implementation, and so that a cut
    never hides a finished candidate's answer. Code: `controls.middle_clipper`, `arms.py`.
  - **R3, H3 wording (section 4).** *Was:* "within 2 points of" the candidate-free accuracy,
    with a one-sided lower-bound condition. *Now:* a non-inferiority hypothesis (not more
    than 2 points worse), tested one-sided at margin 0.02, as an intersection test over the
    strata (replaced by one pooled test in R11). *Why:* "within" suggests a two-sided equivalence test, which is not what is
    tested. The statistical condition and the margin are unchanged.
  - **R4, answer swap (D6, D25, H4b).** *Was:* every c2+ set with at least one wrong
    candidate, swap target "the plurality wrong answer" without a tie rule. *Now:* the primary
    population requires the correct answer to be the strict plurality and a wrong parsable
    answer to exist; the broad c2+ population is secondary; target, tie-break (answer shown
    first), pooling and treatment of unparsable candidates are fixed. *Why:* in sets where
    the correct answer was tied or outvoted the swap is not a counterfactual about answer
    frequency. Code: `arms.swap_plan`, `reader_eval.swap_cells`.
  - **R5, SC on the test sets (D23, sections 4, 5).** *Was:* "its exact SC credit", for pools
    of 32. *Now:* exact on dev (18 samples), Monte Carlo on the test sets with 2,000
    subsets per problem and k, hashed seeds, linear interpolation, not-estimable problems
    dropped and counted, ESC likewise. *Why:* 32-sample pools are not enumerated (the exact
    method takes seconds per problem and k, measured), so the estimate must not be called
    exact. Code: `sc_estimate.py`.
  - **R6, inference and power (D5, D9, D10, D21, D24, section 6).** *Was:* H1a, H1b and the
    others decided by "CI lower bound above 0" (or above a margin), p-values "as in the
    audit", Holm "at 0.05", and a power of 1.6 points. *Now:* the decision is the
    Holm-adjusted p-value on the two-sided scale (D24), intervals are estimates; the family
    size is fixed; the power is recomputed for the six-hypothesis Family 1 under Holm
    (2.0 points at the worst case, section 6.3). D5's statement that H1b is "powered for
    1.5 points" is corrected to 74% unadjusted and 48% at the Holm worst case. *Why:* an
    interval above zero is not a Holm-corrected test, and the 1.6 points was the unadjusted
    figure for one test. For a single hypothesis the new rule equals the old one.
    Code: `inference.py`, `power.py`.
  - **R7, seeds (D7, D8, sections 6.2, 8, 8.1).** *Was:* D7 allowed 3 or 5 seeds while the
    Stage 3 row hard-coded 3 and 3. *Now:* Stage 3's runs, budget and timeline follow
    n_s = 3 or 5; the budget and weeks are tabulated for both; agreement of signs across
    seeds is stated as a robustness criterion, not a test. *Why:* consistency. Code:
    `decision.py`, `schedule.py`.
  - **R8, G4 and the framing (D26, section 4.1, section 8).** *Was:* "H1a holds: frame as
    C1. Otherwise lead with C2." *Now:* three cases: H1a and H1b supported, full C1; H1a only,
    narrower C1 with the solver-GRPO caveat; H1a not supported, lead with C2 whatever H1b
    says; the reporting rules of section 4.1. *Why:* H1a alone says nothing about the
    solver-GRPO baseline. Code: `decision.framing`.
  - **R9, evaluation draws (D15), my addition.** *Was:* 4 draws × 2 replicates per problem.
    *Now:* the number of draws on the primary set is fixed at the freeze by a power rule
    (4, 8 or 16; provisionally 16), 4 on the other sets. *Why:* the power recalculation of
    R6 shows the stratum-wise hypotheses are underpowered at 4 draws; the rule fixes a design
    parameter by a pre-stated criterion and changes no hypothesis. The owner may overrule
    it (O1). Also corrected: the harness row of section 2 (pinned, not "not yet loaded").
  - **R14, the training set and its decontamination rule (2026-11-15), [delegated].**
    *Rule:* the training problems are all MATH train problems of levels 3 to 5 plus a
    seeded subset of 2,600 DeepScaleR-Preview problems; a problem is removed if its
    normalised text (lower-case alphanumeric words) is identical to a study problem's or
    shares a run of 13 consecutive such words with one, against dev, primary, reserve, the
    audit's screen and pilot sets, the harness set and the secondary set (OlympiadBench
    `OE_TO_maths_en_COMP`, text only); exact duplicates inside the training set are also
    removed. Short problems match by identity only. *Result* (`data/train/manifest.json`,
    `scripts/14_build_train.py`): 7,587 problems (4,987 MATH, 2,600 DeepScaleR; levels
    1,460 / 1,514 / 2,013 for MATH), 680 candidates removed for overlap with a study set (671 by
    n-gram, 9 identical; 644 from the dev, primary, reserve, screen and pilot sets, 29 from
    the harness set and 7 from the secondary set) and 423 duplicates. The n-gram rule removes 584 of the 5,582 usable MATH train
    problems (10%); this is conservative and is kept. *Not done:* AggLM's easy-set share
    (a ratio inside the candidate sets, not in this list) is still to be checked before the
    freeze; the DeepScaleR share (34%) is my choice, to be revisited at the freeze with
    the pilot. The list is frozen at the Stage 2 freeze with the rest of section 9.
  - **R13, untrained reader on dev (2026-11-15).** The job of section 7 ran once, as
    designed in R11: 16 draws, 2 replicates, K = 8, level-5 dev. *Nothing preregistered
    changed:* D15's provisional 16 draws stands (it is also the rule's fallback), and O3
    and O4 carry measured numbers (section 9); on 2026-11-15 the owner kept the H4a margin
    (1 point) and the H1 planning effect (1.5 points) and deferred both to the freeze. The SD of H1b and the
    difference-in-differences of H2 are proxies (stated in section 7), to be replaced by
    the pilot's. The 1.7B pool's generation limit is 2,048 tokens and 14.2% of its level-5
    candidates hit it (R12's item (ii) is thereby bounded for dev: the cap is to be checked
    on the test pools, not here).
  - **R12, G2 result and the owner's decision (run 2026-11-14, decided 2026-11-15).**
    *What happened:* the G2 job ran once on a rented RTX PRO 6000 (Qwen3-4B-Instruct-2507,
    bf16, 60 problems, 16 samples each, K = 4, 8 sets per problem, candidates cut at 4,096
    tokens; 3 h 50 min in all). Result: c0 reader − candidate-free −4.1 points
    [−11.1, +0.5] ok; c0 placebo − candidate-free −7.1 [−15.5, −1.1] ok; c2+ reader −
    candidate-free +27.1 [+15.6, +40.0] ok; **c2+ vote − reader −0.9 [−3.1, +0.9] fails the
    rule as written**. Reported only: c1 reader − candidate-free +25.0 [−10.0, +56.7] (15
    problems); c2+ answer-only − reader −18.4; c2+ placebo − candidate-free −40.8. The
    saved-data diagnostic (no GPU) found a ceiling: on the 192 c2+ sets the vote is right
    99.2% of the time and the reader 99.5%; they agree on 188 sets; the gap is 3 sets where
    the vote tied 2–2 (half credit) and the reader chose the right answer, against 1 set
    where the vote was right and the reader wrong. On the 73 c2+ sets with only 2 or 3
    correct candidates the reader scores 0.99 and the vote 0.98. Extraction is not the
    cause (the reader's boxed answer was found in 480 of 480 outputs, one generation hit
    the length limit); 37.4% of the candidate samples hit the 8,192-token generation limit
    (all 359 wrong) and 304 of 480 reader prompts had all four candidates clipped at the
    4,096 cap without hurting c2+ accuracy. *Decision (owner, 2026-11-15, on my
    recommendation):* treat G2 as passed on the three informative signs, because the
    fourth cannot be told apart from zero when the vote has no headroom. *What did not
    change:* the G2 rule above is not edited; this entry records that it was waived for
    this run and why, so the gate must not be read as having reproduced the paper's fourth
    sign. *Consequences:* (i) the untrained 4B reader is at least as good as the vote at
    K = 4 on c2+, so the main study should expect a trained 1.7B reader's value to come
    from c1 and c2 sets, not from beating a weak untrained reader; (ii) the 8,192-token
    cap loses candidates on hard problems, to be checked on the Qwen3-1.7B pools before the
    freeze (not changed here); (iii) the reader stage costs about 2.7 h for 1,920
    generations at prompts of about 16,500 tokens, which sizes the untrained-reader dev
    runs under the 10-hour session limit. Raw data: private dataset
    `Arshia-HZ/trained-reader-audit-raw`, `g2/g2_qwen3-4b-instruct-2507/`; the diagnostic
    is `scripts/11_g2_diagnostic.py`.
  - **R11, owner's answer to the review (2026-11-14).** The owner accepted O1 (draws
    provisionally 16 on the primary set, fixed at the freeze by the power rule) and O2
    (H3 becomes one pooled non-inferiority test over all sets, strata as estimates; this
    changes H3's test form, not its margin or direction). *Constraint added:* GPU access is
    short sessions of about 10 hours (section 8.1), so the main runs come first and the
    extras are decided after them. The D24 clause "H3 is an intersection test" no longer
    applies.
  - **R10, open items for the owner (section 9).** Four findings of the power
    recalculation that would need a change of a margin, a hypothesis or the design are
    listed with a recommendation and **not applied**: O1 draws, O2 the form of H3, O3 the
    margin of H4a, O4 the reading of H1's planning effect.
