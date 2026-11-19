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
| D12 | Generation stack | Every generation that is evaluated (candidate pools, readers, solver-GRPO samples, the contamination probe) uses vLLM 0.33.0 in fp16 on the RTX PRO 6000, as the dev pools did. Training is bf16. A bf16 checkpoint is cast to fp16 for evaluation after a sanity check (no NaN or inf; greedy agreement with bf16 on 200 prompts is reported). (R17: the trained weights are the fp32 master weights; the stored and evaluated copy is cast once from them to fp16, with no bf16 copy, and the same sanity check applies.) Training rollouts use the training framework's own engine and are never evaluated. The G2 harness model (Qwen3-4B-Instruct-2507) runs in bf16: it is in no evaluated comparison. | The dev pools are fp16; mixing precisions would put a stack difference inside the comparison. |
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

**Update 2026-11-16 (R16, measured at the pilot).** A 300-step reader run is about 20 GPU-hours
and a 300-step solver run about 10.5 (tokens-matched solver runs about 14 to 18, an estimate), so
with 3 seeds the training of the two main arms is about 61 + 41 to 53 = 102 to 114 GPU-hours, inside
the table's 105 to 240. The table is not recomputed here; `schedule.py` is updated at the freeze.

**Compute available (owner, 2026-11-14).** One RTX PRO 6000 (Blackwell) in short sessions
of about 10 hours, not long continuous access. Consequences: every job is sharded,
resumable and uploaded when it finishes a shard; the order is the main runs first (G2, the
pools of the sets the main hypotheses need, then the reader and solver-GRPO runs and their
evaluation), and the extras (joint arm, 0.6B reader, no-c0 run, secondary, robustness and
post-release sets, K sweep beyond K = 8 and 4, factorial) are decided after the main
runs, not scheduled now. The weeks of the timeline above assume continuous access and
will stretch accordingly; the cut date of the factorial still binds.

## 9. Not yet decided

**Frozen at the training freeze (R17, 2026-11-16):** reader prompt wording (the directory
`prompts/reader_v1/`, read and approved by the owner); GRPO framework, hyperparameters and the speed changes (R15, R16); the training
decontamination rule and the train-set mix (R14, R17); the seed count (D7: 3, R16); the checkpoint storage.

**To be fixed before any trained checkpoint is evaluated on a confirmatory set (the second freeze):** the
analysis and contrast code; ESC settings; the exact factorial contrasts; the number of draws (section
6.3) and the SDs of section 6.3 (with the trained reader's dev SD beside them); the contamination-probe
protocol (D13); the transfer-pass prompts for Llama-3.2-3B and Qwen2.5-Math-1.5B (plain prompt with a
format gate).

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
  reader's measured SD.) **Restated at the training freeze (R17, 2026-11-16): values unchanged (margin 1 point,
  planning effect 1.5 points); they are stated again, with the trained reader's dev SD, at the second
  freeze before any confirmatory evaluation.**

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
  - **R20, the dev result of the reader milestones and the D17 selection; the owner's pause
    of Stage 3; exploratory analyses logged before they are computed (2026-11-19),
    [delegated] except where the owner is named.** *Result* (dev, the set-up of R19,
    complete for all three milestones: 208 problems and 6,656 `reader_full` generations
    each; `docs/dev_checkpoint_rule.json`; raw data `dev_trained_reader_s1_*` in the raw
    dataset): the H1a gain over matched SC (generated-only) is +0.5 points at step 100 (SD
    0.144 over problems), −1.7 at step 200 (0.182) and −1.4 at step 300 (0.178), against
    −3.2 for the untrained reader (R13). **D17 selects step 100** (exact argmax, no tie).
    The mean read is 730, 1,239 and 1,266 tokens (untrained 728), k\* 8.74, 9.28 and 9.32;
    no problem is not estimable. At step 100: c2+ sets, reader minus the plain vote −2.5
    points (SD 0.160, 154 problems; untrained −7.4); the `candidate_free` arm (6,656
    generations) gives, on c1 sets, reader with candidates minus its candidate-free solve
    +6.4 points (SD 0.403, 64 problems; untrained +14.5), so the dev estimate of H2 is
    about −8 points against its planning magnitude of +10. The ESC frontier at the reader's
    cost is 1.5 points above plain SC on level 5 (D22, `docs/dev_baselines.md`), so step 100
    is about 1 point below it. *Reading:* dev only and descriptive; the best of three
    steps is biased upward; no intervals here. Training moved the reader about 3.7 points,
    mostly through less harm on c2+ sets, while its c1 advantage over its own fresh solve
    shrank; whether by worse reading or by a better fresh solve is read from E1 (a) below.
    With the planning effect of 1.5 points a supported H1a at the confirmatory stage looks
    unlikely; nothing preregistered changes, and D26 frames that case. The D12 agreement
    check is not available for step 100 (R19). *Owner's decision (2026-11-19):* Stage 3's
    costly runs (reader seeds 2 and 3, the solver-GRPO seeds) are paused until the
    analyses below are read; no hypothesis, margin or rule changes by the pause.
    *Exploratory, logged before any of it is computed* (no Holm, dev only, used to choose
    no confirmatory parameter unless a later dated entry says so). **E1**, no generation,
    the stored dev shards of R13 and of steps 100, 200 and 300: (a) raw accuracy per arm
    and stratum, with the plain vote of the same sets; (b) the reader minus the plain vote
    of the same K = 8 sets over all sets (the quantity AggLM reports, at equal K); (c) the
    share of reads whose answer is in the set's plurality, and the share equal to some
    shown answer; (d) reader minus vote by the size of the plurality among the 8 shown
    answers; (e) a dev estimate of H6: the unanimity gate (vote if all 8 answers agree,
    otherwise read) against ESC at matched mean cost, under the prompt-inclusive rule (H6's
    rule) and the generated-only rule, with ESC (windows 2, 4 and 8, caps 8 and 16) and SC
    computed on the same 208 problems and pooled into the frontier of D22; plurality
    thresholds of 5, 6 and 7 of 8 are reported as a curve and none is selected; (f) ESC
    first, a read only on disagreement: ESC with window 2 or 4 and cap 8 runs over each
    seeded set's shown order; if it stops early, its plurality at its cost; if it reaches 8
    without an agreeing window, the stored read of that same set at the cost of the 8
    samples plus the read; against the same frontier. Intervals by a bootstrap over
    problems (2,000 resamples); a policy whose mean cost is outside the frontier's range is
    reported as not estimable. Code: `allocation.py`, `scripts/24_dev_allocation.py`,
    `tests/test_allocation.py`. **E2**, generation, after E1: a probe of the untrained
    Qwen3-1.7B in thinking mode on about 20 level-5 dev problems, the candidate-free solve
    first, then a read of the same non-thinking sets. Its settings (template, context
    window, length cap, sampling, answer extraction after `</think>`) are logged in a dated
    entry before it runs, with a model config of its own, so the non-thinking check of the
    frozen arms stays as it is.
  - **R19, the dev evaluation of the reader milestones, and a deviation from D12
    (2026-11-18), [delegated]; written before any milestone was scored.** *Set-up:* the
    208 level-5 dev problems of R13, K = 8, 16 seeded sets per problem, 2 replicates,
    temperature 0.7 (R13's, not the training 1.0), candidates cut at 1,536 tokens with
    the base model's tokenizer, non-thinking template, 2,048 new tokens, window 16,384,
    vLLM 0.33.0 in fp16, the fp16 milestone read from a pinned commit of the checkpoint
    repo; 6,656 `reader_full` generations per milestone. The prompts are the frozen
    `reader_v1`, which are byte for byte the draft R13 used (checked against commit
    `fda927d`), so R13's untrained −3.2 points is a fair reference. Only the weights
    differ from R13, and `checkpoint_rule.identity_problems` stops the run on any other
    difference: the generation config, the token ids of a real dev prompt under each
    chat template, the hash of `tokenizer.json`, the main config keys, the weight dtype.
    *Decision quantity (D17, unchanged):* H1a at 16 draws by the same function as R13
    (`summarise` of `13_dev_reader_analysis.py`, loaded from that script): the reader's
    accuracy minus self-consistency at the same generated tokens, averaged over the 208
    problems. Longer trained reads raise the matched count k\* (about 9 to 10 against
    8.75), which is right under the generated-only rule; a problem whose k\* exceeds the
    18 samples of its pool is not estimable, and then nothing is selected. *Ties:* "ties
    to the later step" is read at face value, exact equality of the computed gains, no
    tolerance (a tolerance would be a new rule that favours later steps). *Completeness:*
    nothing is selected unless all three milestones have all 208 problems and 6,656
    generations. *Arms:* `reader_full` for the three milestones, which decides;
    `candidate_free` for the selected one only, for the H2 SD the second freeze states.
    Dev only, no confirmatory data. Code: `checkpoint_rule.py`,
    `scripts/23_dev_checkpoints.py`, `tests/test_checkpoint_rule.py`. *Deviation from D12
    (it amends R17, which said the same sanity check applies):* the agreement check of
    D12 (greedy agreement with the training engine's bf16 on 200 prompts) needs weights
    that are not fp16, and the plan of R17 kept only the fp16 milestones and deleted the
    resumable states, which held the only fp32 weights. So for steps 100 and 200 the check
    cannot be run; what exists for them is the finiteness check at the cast (passed) and,
    on the pilot checkpoint, the fp16 against fp32 comparison of R17 (same top-1 token,
    largest logit difference 0.04). For step 300 the fp32 weights were copied to the
    checkpoint repo before they were lost (`reader_s1/master_0300/`, 6.88 GB, an addition
    to R17's storage plan), so the check can be run for it. If step 100 or 200 is
    selected, the check is reported as not available for it. Every evaluated generation
    still comes from the stored fp16 milestone in vLLM fp16, the stack of D12.
    *Amendment (2026-11-18, the same day, before any generation and any score):* the
    first start stopped at the identity check, as it should. The milestone's
    `generation_config.json` differs from the base's only in `transformers_version`
    (4.51.0 against 5.20.2). Its `tokenizer.json`, written by transformers 5.20.2, differs
    only in serialisation flags (`trim_offsets` of the byte-level pre-tokenizer and three
    flags of the decoder); the split pattern, vocabulary, merges, added tokens and
    normaliser are equal, and 832 real dev prompts (3.68 million tokens) and two edge
    texts encode to the same ids under both. So: the version stamp is no longer compared;
    the tokenizer is compared by the token ids of the 208 dev prompts and not by the hash
    of its file, and a difference still stops the run; and the evaluated folder reads the
    base model's own tokenizer files and chat template (the revision R13 used), so that
    encoding and decoding are the same code as for the untrained reader.
  - **R18, reader seed 1 trained: health check, cost and what is not yet known
    (2026-11-18), [delegated] except where the owner is named.** *What ran:* the reader arm,
    seed 20261115, the recipe of R15, steps 1–300 on the frozen code (tag
    `stage2-training-freeze`, manifest `8842b64b…ea05`), in three molab sessions (steps
    1–50, 51–200, 201–300). The owner paused the run twice; each restart loaded the
    resumable state from the hub (steps 50 and 200), refused a different row hash and found
    the same one; steps 51–55 of the first session were lost and run again. The hub's
    step-25 and step-200 states matched the local checkpoints by file size and by hash of
    the model, trainer state, scheduler and random-number state. *Kept on the hub:* the
    milestones of steps 100, 200 and 300 (fp16, 3.44 GB each, the finiteness check
    passed), the step, batch, run, start and save logs, and `saves_complete.jsonl` (the
    hub copy of `saves.jsonl` lacks the lines of steps 50, 200 and 300, which are written
    after the log upload; the lines of steps 50 and 200 are taken from the session record
    and marked `recovered`). The two resumable-state repos were deleted by the run itself
    once the step-300 milestone was on the hub; the main repo holds 10.3 GB.
    *Health check (R17):* one trigger fired, a mean step time above 300 s (steps 244–268,
    and it held in the last two blocks: 311 s and 302 s). The owner was asked and on
    2026-11-18 chose to continue to step 300. The cause is longer completions, not the
    hardware: the mean length rose from 729 tokens (steps 1–25) to 969 (276–300), the share
    at the 2,048 cap from 1.4% to 8.5%, generation time from 163 s to 215 s of a step; the
    GPU ran at full use, 2,407 MHz and 61 °C, and the share of skipped zero-advantage rows
    stayed at 0.74–0.79. No other trigger fired: no non-finite value, mean gradient norm
    0.10–0.12 in every block, mean reward 0.47–0.53 in every block (never below half of the
    first block's 0.49), length inside 100–1,500, the cap share well under 25%, every
    checkpoint on the hub, one row hash. *Cost (replaces the R16 estimate of 20.26 GPU-h):*
    steps 22.59 h, saves and uploads 0.30 h, start-ups 0.13 h, 23.03 GPU-h in all (mean
    step 271 s against the pilot's 238 s), plus about 0.33 h for the redone steps. The D7
    rule (20 hours or less gives five seeds) already gave three, so nothing changes in the
    design; three reader seeds cost about 70 GPU-h. The solver's 10.45 h estimate (122 s a
    step) may also grow if its completions lengthen, and is to be measured again.
    *Not known yet:* the training reward is flat at about 0.5 over all 300 steps, but each
    step has new problems of different difficulty, so it is not an evaluation, and
    nothing is concluded from it; whether the reader gained anything is for the dev
    evaluation of steps 100, 200 and 300 (the checkpoint rule), which has not been run. If
    it shows no gain over the untrained reader, that is a result for this recipe at this
    scale, to be reported as it is; the recipe (R15) is not changed without a dated
    deviation, and seed 1 would then be reported as a pilot (R17).
  - **R17, the Stage 2 training freeze, the run order and checkpoint storage (2026-11-16),
    [delegated] except where the owner is named.** *Owner:* on 2026-11-16 the owner read all
    three reader templates and a sample of the training pools and judged them fine; the
    wording is unchanged and is now frozen as `prompts/reader_v1/` (the audit's
    `MANIFEST.json`, combined sha256 `0bb74ee5…635da`, checked by a test). The owner also
    approved starting the training. *Frozen now (the training freeze):* the prompts; the
    recipe of R15 with the two exact changes of R16 (own-length scoring, skipping of
    zero-advantage rows) and fp32 master weights; the training set and its decontamination
    (R14); the pool model and settings; the seed count (D7, 3, R16); the files listed in
    `configs/training_freeze.json` with their hashes (combined sha256 `8842b64b…ea05`). `scripts/20_train_grpo.py` verifies
    them at start and refuses to run if one changed; each run records its commit, library
    versions, GPU and the hash of its row sequence in `runs.jsonl`, and a restart refuses to
    continue if its rows differ from those of the first start (the rows are rebuilt from
    the pools on disk, so a machine with other shards would silently change the prompts of
    a step). The freeze commit carries the tag `stage2-training-freeze`. The 50-step pilot
    (R16 addendum) and a further speed-up that would skip the scoring pass of
    zero-advantage groups are not part of it. G3 (stable pilot, throughput fits the budget)
    is passed on the 20-step pilot of R16 and the health check below; the part of G3 that
    asks for the analysis code to be hashed moves to the second freeze. *Seeds:* seed k of
    an arm runs with `--seed` 20261114 + k (20261115, 20261116, 20261117), which sets which
    easy sets are kept, the order of the rows and the trainer's data seed. TRL fixes the
    seed of the vLLM sampler (to 0), so seeds differ in their data and not in sampling
    noise. *Checked at the freeze:* AggLM's easy-set share. The paper's wording is
    ambiguous (the mixture adds "p% of easy examples" with p = 50%, while the caption of
    its Table 4 reads "percentage of easy sets relative to hard examples"); I keep all
    hard sets and half of the easy sets, the roadmap's reading. On the 400 pilot problems
    that is 2,134 hard and 1,375 easy rows (39% easy); the other reading would give about
    1,067 easy rows (33%). The DeepScaleR share (34% of the problems) is kept. *Training
    problems used:* the first 2,400 of the seeded order (the 400 of the pilot pools and
    2,000 more, 24 shards); 300 steps of 64 prompts need 19,200 rows and about 21,000 are
    available (an estimate from the pilot rate); the other 5,187 problems are unused.
    *Not frozen yet, and fixed before any trained checkpoint is evaluated on a confirmatory
    set (the second freeze):* the analysis and contrast code, the evaluation draws, the
    probe protocol (D13), ESC settings, the factorial contrasts and the transfer prompts.
    The margin of H4a (1 point) and the planning effect of H1 (1.5 points) stay as the
    owner decided on 2026-11-15; the trained reader's SD, from the dev evaluation of the
    selected seed-1 checkpoint, is stated beside them at that second freeze, where the
    owner may revisit them (dev is not a confirmatory set). *Run order:* reader seed 1; the
    health check below; the solver seed 1; seeds 2 and 3 of both arms; the other arms
    after the main ones. Seed 1 is a confirmatory run, not a pilot. *Open until the solver
    run:* whether its token match counts rollout tokens only (R15) or rollout plus pool
    tokens (section 8.1); a matched run needs more than 300 steps, so its milestone is its
    final step, not step 300. *Health check* (stop and ask the owner; none of these is a
    result): a non-finite loss or gradient norm; a gradient norm above 10 times its median
    over steps 1–25 for three steps in a row; the mean reward over any 25 steps after step
    50 below half of its mean over steps 1–25; a mean completion length below 100 or above
    1,500 tokens over 25 steps; more than 25% of completions at the 2,048 cap over 25
    steps; a mean step time above 300 s (1.25 times the pilot's) over 25 steps; a
    checkpoint whose resumable state or milestone did not reach the hub (the `uploaded`
    list of `saves.jsonl`); or a restart whose row hash differs. All but the last two are
    computed from `steps.jsonl`, which records the gradient norm, the clipped share and
    the mean length of every step. If the recipe must change after seed 1 has run, that
    run is reported as a pilot and a dated deviation is logged. *Evaluation:* no trained
    checkpoint is evaluated on a confirmatory set until all planned runs are trained and
    the second freeze is taken; the dev evaluation of steps 100, 200 and 300 (the
    checkpoint rule) is the only evaluation before that. *Storage (amends D12):* a
    milestone is stored once, cast from the fp32 master weights to fp16 (no bf16 copy),
    3.4 GB, with a check that every value is finite, and this is the model that is
    evaluated; D12's sanity check (no NaN or inf, greedy agreement with the training
    engine's bf16 on 200 prompts) stays. On a real Qwen3-1.7B checkpoint the conversion
    gave 3.44 GB, 310 tensors, a largest weight of 153, no non-finite value, the same
    top-1 token and a largest logit difference of 0.04 against fp32. The reader keeps
    steps 100, 200 and 300, the solver its final step; the step logs go to the hub with
    every checkpoint. The resumable state (about 20 GB) alternates between two private
    repos, each deleted and created again before its upload, because Hugging Face keeps
    the files of a repo's history in its storage count: the empty checkpoint repo still
    showed 20.66 GB after the test upload was deleted and its history squashed, and
    recreating the repo freed it. At most two states exist (about 40 GB), a failed upload
    never touches the newest complete one, a restart on a new machine resumes from the
    newer of the two, and both are deleted when the run completes. Upload, alternation,
    resume and removal were tested against the real hub with small files. The estimate
    for three seeds of both arms is about 80 GB at its peak (40 GB of states of the run in
    progress, 3 × 10 GB of reader milestones and 3 × 3.4 GB of solver weights), against the
    250 GB of the earlier plan; the owner's quota of about 100 GB for private storage is to
    be checked at each session.
  - **R16, the 20-step GRPO pilot and its cost (2026-11-16), [delegated].** Reader arm,
    the 400 pilot problems, recipe of R15, on one RTX PRO 6000 (molab). *Speed:* the first
    attempt took more than 30 minutes for one step, because TRL pads every micro-batch of
    a generation batch to its longest sequence (about 14,700 tokens in a 512-sample batch,
    against a mean of about 6,000) and attention grows with the square of the length. Two
    changes, neither of which alters the objective: (i) each sequence is scored at its own
    length (`trim_padding`; scores equal the padded ones to 8e-5 in fp32, differences in
    bf16 are rounding); (ii) rows whose advantage is exactly zero (all 8 samples of a
    prompt rewarded alike, 62–86% of groups in the pilot, mean 75%) skip their forward
    and backward pass, which is exact because the dapo normaliser is a constant of the
    generation batch and KL weight and entropy bonus are 0 (CPU check: identical gradient
    norms and final weights, `scripts/21_grpo_resume_check.py`; on the GPU, with the same
    rollouts, losses and rollout tokens at steps 1 and 2, the gradient norm of a skipping run
    differs from two non-skipping runs by 1 to 3 parts in 10,000 (step 1: 0.12510 against
    0.12511 and 0.12512; step 2: 0.13511 against 0.13515 and 0.13513), while the two
    non-skipping runs differ from each other by 1 part in 10,000, so the effect is at the
    level of GPU run-to-run noise). *Measured seconds per
    step:* with (i) alone, 422 on average over 19 steps (range 351–494; generation 39%); with
    (i) and (ii), 280 (including warm-up), 233 and 230 over 3 steps, of which generation
    and scoring of the rollouts are about 70%. *Projection for 300 steps:* about 19 GPU-hours
    with both changes (35 with (i) alone). That is at the 20-hour line of D7, so the seed
    count is **not** decided here: the freeze takes it from a longer measurement (the
    50-step pilot, if the owner approves it), with this figure as the estimate. A lever not
    used: turning off TRL's importance-sampling correction for the gap between the vLLM
    sampler and the trainer would remove the scoring pass of all 512 rollouts (an estimate, not
    measured: about 75 s a step, 17 GPU-hours for 300 steps), but it changes the objective and was left on.
    *Resume on the real stack:* the run was killed after the step-10 checkpoint (20 GB:
    weights, optimizer, scheduler, RNG, trainer state) and restarted; the resumed run
    recomputed step 11 with the same prompts, the same 412,142 rollout tokens and the same
    reward and loss, and its running token total agrees with the sum of its steps.
    *Health over 20 steps:* no NaN; gradient norm 0.08–0.15; clip ratio 0 (every step is
    on-policy); entropy about 0.2; 1–3% of completions reach the 2,048 cap; peak GPU
    memory 60 GiB allocated, vLLM share 0.3. Mean reward 0.49 over steps 1–10 and 0.48 over
    11–20: no movement was expected at learning rate 1e-6 with 10 warm-up steps over 20
    steps, so this pilot says nothing yet about whether the reader learns. *Records:* step logs, batch logs
    and trainer states of the pilot and the timing and noise runs are in the private dataset
    `Arshia-HZ/trained-reader-audit-raw`, `grpo_pilot_2026-11-16/` (the 20 GB checkpoint is
    not kept). *How the D7 figure was to be measured (fixed before the 50-step pilot; amended below):*
    the 19 GPU-hours above count step time only and were seen before this paragraph was
    written, so they are not the figure the rule uses. The figure is `projected_hours`
    (`grpo_plan.py`) on the 50-step pilot of the recipe of R15 with the importance-sampling
    correction on: the mean seconds per step over steps 6 to 50, times 300; plus 12
    checkpoint saves (every 25 steps) at the measured seconds of a save with its upload
    (the upload blocks the GPU); plus one model and vLLM start-up per 10-hour session at the
    measured start-up seconds. It is computed once and not re-extrapolated. The cost of a
    step depends on the share of zero-variance groups, which will drift as reward moves, so
    the 50-step window is not a stationary sample; the figure is a planning number, the
    hours actually spent are reported at the end, and the seed count is not reopened by
    them. Checkpoint uploads were checked against the real `huggingface_hub` matching: the
    deletion pattern of `latest` is relative to its own folder and cannot remove the
    milestone weights (tested).
    *Addendum, later on 2026-11-16 (owner delegated the choice, [delegated]):* **the 50-step
    pilot is not run**, to save GPU time. What it would have added (stability beyond the
    10 warm-up steps, a longer window for the cost figure) is replaced by watching the step
    log of each main run from its first step and stopping a run that diverges; checkpoints
    are resumable. The D7 protocol above is amended accordingly: the window is the four
    steps after the first of the runs that had flat-row skipping (steps 2 and 3 of one run,
    step 2 of three others), with the saves and start-up measured below. *Measured:* a
    reader step takes 238 s on average over those four (230–251), a solver step about 122 s
    (115, 121, 130; the first step, 124). A checkpoint save takes 26 s; uploading the
    20 GB `latest` and the 7 GB milestone weights (fp32) together takes 51 s; model and
    vLLM start-up takes 197 s on a fresh machine (28 s once cached). The private model repo
    `Arshia-HZ/trained-reader-audit-ckpt` was created (owner's go-ahead); the throwaway test
    upload was removed from it. Peak GPU memory 59.7 GiB (reader) and 56.8 GiB (solver).
    *Result of the rule (`projected_hours`, 300 steps, 12 saves of 76 s, three sessions):*
    reader 19.8 h of steps + 0.25 h of saves + 0.16 h of start-up = **20.3 GPU-hours**;
    solver 10.1 + 0.25 + 0.06 = 10.5 GPU-hours for 300 steps. The reader figure is above
    20, so by the rule as written **D7 gives 3 seeds**; the margin is far inside the error
    of a four-step window (the step times of the 19 unskipped pilot steps varied by 10%),
    so this is a rule outcome, not evidence that 5 seeds are unaffordable; the freeze
    restates it, and the actual hours of the main runs are reported at the end. A
    solver-GRPO run matched on tokens needs more steps than 300 (the roadmap's factor is
    1.3 to 1.7, so about 14 to 18 h, an estimate). Storage: a run keeps about 20 GB of
    `latest` and 7 GB per milestone, about 41 GB, so six runs would hold about 250 GB on the
    private repo; the owner should check the storage plan (milestone weights could be saved
    in bf16, halving their size, if needed).
  - **R15, the GRPO set-up (2026-11-15), [delegated]; recorded before any training
    run.** *Framework:* TRL 1.18.0 `GRPOTrainer` with colocated vLLM 0.33.0 (its metadata
    allows the evaluation version), in a separate training environment (the evaluation
    environment is not touched); mixed-precision training (fp32 master weights under bf16 autocast, because a step of 1e-6 on weights stored in bf16 is rounded away). *Starting values* (the roadmap's, with the
    two choices it left open): full fine-tune, AdamW, learning rate 1e-6 after 10 warm-up
    steps, 64 prompts × 8 samples per step, sampling temperature 1.0, completion cap
    2,048 (the evaluation's `max_tokens`), clip 0.2 / 0.28, token-level normalisation
    (`dapo`), advantages scaled within the group, **KL weight 0** (no reference model; the
    roadmap allowed 0 or 0.001), 300 steps, checkpoint every 25. One optimizer step uses
    exactly one generation batch (no reuse), so checkpoints never fall inside TRL's
    generation cycle. Prompts are rendered with the non-thinking template, as in
    evaluation. Completions that hit the cap are **not** masked: the reward is the audit's
    grade, which accepts a complete box in a cut completion, so training and evaluation
    score the same thing. *Rows:* reader prompts are candidate sets with K in {2, 4, 8}
    (4 draws per problem and K, order shuffled, no candidate marked as the reader's own),
    every c0 and c1 set and a seeded half of the c2+ sets; the no-c0 arm drops the c0
    sets; solver-GRPO uses the plain round-0 prompt. The reader and solver runs differ only
    in prompt, and are matched on cumulative rollout tokens, which the run logs per step and
    rebuilds from the trainer's saved log after a resume. *Checkpoint rule:* the best of
    steps 100, 200 and 300 on dev, as in the roadmap. *Measured before the pilot:* on the
    400 pilot pools the reader rows number 3,509 (c0 1,554; c1 580; c2+ 1,375), prompts
    average 4,585 tokens (K = 2: 2,130; K = 4: 4,089; K = 8: 7,941) and the longest is
    12,676, so with the 2,048 cap everything fits the 16,384 window; the pilot pools have
    14–17% of samples at the generation cap. A CPU run with a tiny random model, the real
    rows, reward and callbacks, killed after its first checkpoint and resumed, gave the same
    prompts per step, the same rollout-token totals, the same gradient norms and the same
    final weights as the uninterrupted run (`scripts/21_grpo_resume_check.py`). *Not yet
    checked:* the colocated vLLM path, GPU memory and speed (the 20-step pilot is the first
    test); the cost was an estimate (superseded by R16, which measured about 4 minutes a
    step after two fixes). Changing any of
    these after the pilot needs its own dated entry.
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
