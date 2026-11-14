# Thesis Experimental Roadmap

Nov 13, 2026 · @Arshia (revised; original of Nov 12 kept in docs/history/)

## What changed in this revision

This revision applies the seventeen ranked changes in [Trained aggregators beyond voting](research/trained_aggregators_beyond_voting.md). It also corrects how the Nov 12 text read the audit.

- **The novelty claim is narrower.** Trained small readers are prior work (AggLM, RL-GenSelect, SSA, TTT-Consolidate). The claim is now attribution plus the solver side.
- **The thesis has two parts.** Part A asks what GRPO changes in an in-context reader of its own samples. Part B is the solver side: a free candidate × aggregator grid, a joint arm and a frozen-reader factorial, which runs last and is cut first. A unanimity gate is tested against adaptive SC.
- **The audit misreading is fixed.** Revision does not cost 14 points; the −14.0 is the placebo form effect. Channels lose to SC by 0.8–3.3 points because revision is poor value per token.
- **G1 is a break-even gate** on the 18 existing dev samples per problem. It needs no GPU.
- **There are three compute axes.** Each test problem gets ≥32 samples, verifier FLOPs are counted, and solver-GRPO is matched on rollout + pool tokens.
- **K is a design variable.** Training uses mixed K, and the K sweep is now Must.
- **New controls:** answer swap, placebo on every stratum, and position and length bias.
- **New zero-cost baselines:** DeepConf-offline, self-certainty, a stored-confidence vote and adaptive SC.
- **Test sets:** already-seen problems excluded, a contamination probe (run after the freeze, PREREG D13), MATH-P-Hard, a 93-problem post-release pool and one pinned generation stack.
- **Statistics:** H1 is a superiority test decided by a Holm-adjusted p-value (PREREG D24), with two Holm families, 3 or 5 seeds and a problem-level paired bootstrap. Stage 1 never touches the primary set.
- **Dropped:** distillation, two training ablations, RSA beyond T = 3, and the channel and critic chapters.
- **The budget is about 1.8–3.7 GPU-weeks** (unmeasured). Citations are fixed, and a new repository is recommended.

## Status and decisions (2026-11-14)

This roadmap is the plan. [`PREREG.md`](../PREREG.md) is the binding record, and where the two differ, PREREG wins.

- **Repository split.** The audit is closed at tag `audit-final`. The reader study lives here and imports the audit's grader, extractor, seeds, store and accounting.
- **Stage 0 is done.** G1 passes at K = 8 on level-5 dev under the generated-only rule, so K_H1 = 8 ([Stage 0 record](stage0/STAGE0_REPORT.md)).
- **The open design questions are decided** (PREREG section 3, D1 to D22, marked for the owner's review before the freeze). The ones that change this text:
  - H1b (reader against solver-GRPO SC) is its own test on the primary set; nothing is pooled across primary and secondary.
  - Every set except dev and the harness set stays sealed through Stage 1. Untrained readers run on dev and the harness set only.
  - H4b is two-sided; H4a is a do-no-harm non-inferiority test.
  - Three seeds per main arm, five if a 300-step run measures at most 20 GPU-hours.
  - The joint arm (one seed) and the 0.6B reader are Must. Generation is fp16 on vLLM 0.33.0; training is bf16.
  - The checkpoint rule (best of steps 100, 200, 300 on dev) and the factorial's cut date (2027-01-17) are fixed now.
- **Revised after the owner's PREREG review (2026-11-14; PREREG section 10, R1 to R10).** Where the text below still differs, PREREG wins. What changed:
  - The contamination probe moved from Stage 1 to after the freeze, with a frozen protocol (D13).
  - Candidates over the cap are cut in the middle with the marker counted inside it; generation-truncated and prompt-truncated candidates are counted apart (D14).
  - H3 is a non-inferiority hypothesis ("not more than 2 points worse"). The answer-swap population is the sets where the correct answer was the strict plurality, with fixed tie-breaking (D25).
  - SC on the 32-sample test pools is a Monte Carlo estimate with fixed parameters, exact only on dev (D23).
  - Every hypothesis is decided by a Holm-adjusted p-value on the two-sided scale, not by a stand-alone interval, and the power is recomputed under Holm (D24; PREREG 6.3).
  - Stage 3 follows the seed rule (3 or 5 seeds, PREREG 8.1), and G4 distinguishes H1a from H1b (PREREG 4.1).
  - Open for the owner: the number of evaluation draws, the form of H3 and the margin of H4a (PREREG section 9, O1 to O4).

## Verdict

Keep the reader, but stop claiming it as the contribution. Three Nov 12 claims no longer hold:

- **That no paper trains an aggregator at 1.7B or smaller.** AggLM-1.7B, a GRPO-trained Qwen3-1.7B reading Qwen3-1.7B non-thinking samples, beats maj@8 by 4.4–11.9 points. RL-GenSelect and SSA also train small readers.
- **That none matches total compute.** TTT-Consolidate (NeurIPS 2026) compares a trained 1.7B/4B consolidator with voting at matched FLOPs, with controls. It appeared on Nov 1, and the Nov 12 searches missed it. SPIRAL matches training tokens, and 2610.21290 budget-matches its vote baseline.
- **That all three Nov 12 papers are unreviewed.** 2608.18379 v2 was accepted at the COLM 2026 Workshop on Efficient Reasoning.

What survives is narrower but defensible: the full 2608.18379 battery on an in-context RL reader of its own samples, before and after training, at ≤1.7B and on 1,000+ problems, against per-problem token-matched SC and a token-matched solver-GRPO baseline. The audit's SC curve makes a positive H1 risky. So part of the thesis moves to the solver side, where every joint-training result so far uses one shared model at 4B or larger.

| New rank | Direction (old rank) | Hit by | Decision | Odds of a positive result (judgement from the Nov 13 report) |
| --- | --- | --- | --- | --- |
| 1 | Part A: what GRPO changes in an in-context reader of its own samples, with the full control battery, under three cost rules (old 1) | AggLM, RL-GenSelect, SSA (the method); TTT-Consolidate (controls at 1.7B) | Keep, narrowed to attribution | H1 under generated-only accounting on level 5: roughly a coin flip. Prompt-inclusive: unlikely without the gate or the 0.6B reader. Solver clause: the weakest. C2 is reportable either way. |
| 2 | Part B: the solver side, with the grid, a joint arm and the frozen-reader factorial (new) | SPIRAL, ParaGator, RSA, all with one shared model at ≥4B | Add. The factorial runs last and is cut first. | Not estimated. The grid costs nothing. |
| 3 | A vote-or-read gate, tested against adaptive SC (new) | ESC and Adaptive-Consistency use the same agreement signal | Add as a second-family hypothesis | Not estimated |
| — | Iterated aggregation (RSA) at 1.7B (Nov 12 rank 2) | RSA gains flatten beyond K = 3, and RSA needs 8(T + 1) generations | Should baseline, T ≤ 3 | — |
| — | Distil the reader (Nov 12 rank 3) | Phase 1: full transcripts 49.8% vs rejection-sampled solutions 48.9% | Dropped | — |
| — | Train the receiver to keep or switch (old 2) | Without training, 2610.21290 recovers 53.6% of vote failures and keeps 98.4% of vote successes | Dropped, now with its minority-rescue ablation | — |
| — | Learned, latent or trained-critic channels | Optima loses to SC on MATH at 8B; the smallest trained critic-refiner found is 3B | Not taken up. A verification bit for the gate stays exploratory. | — |

Old Rank 2 stays dropped, for a corrected reason. Revision does not cost 14 points; it is poor value per token (see "Motivation from your audit"). 2610.21290 already gets the conformity benefit without training.

## How the papers overlap with ours

Nine papers now bear on the plan. TTT-Consolidate is closest on controls, AggLM on the reader and 2608.18379 on the measurement standard. The numbers come from the HTML versions read for the Nov 13 report.

| Paper | What it does | Setting | Key numbers | What it takes from our claims |
| --- | --- | --- | --- | --- |
| [2608.18379 v2](https://arxiv.org/html/2608.18379v2), Farmanfarmaian, Aug 2026 (v2 Oct 19; COLM 2026 workshop) | Untrained single-pass aggregation, contrasted with a candidate-free fresh solve, answer-only, an off-topic placebo and plurality voting, by c0/c1/c2+ | Qwen3-4B-Instruct-2507, K = 4, AIME-25 + HMMT-25 (60 problems); candidates cut at their first answer | Aggregation minus fresh solve: −12.3 pts at c0, +24.2 at c1 (p = 0.0755, unresolved), +29.0 at c2+. Vote 0.960 vs aggregation 0.887 at c2+. Placebo −16.4 vs fresh solve, **c0 only**. Answer-only gap at c2+: +0.161 with first-answer scoring, +0.008 with final-answer scoring. | Placebo, candidate-free and answer-only controls as new. It names "a learned selector" as untested. |
| [2610.21290](https://arxiv.org/html/2610.21290v1), Lee et al., Oct 2026 | Training-free bias control (one style-normalised trace per distinct answer) and bias injection | Three-model 8–14B teams; OlymMATH, AIME, HMMT | +3.2–6.2 pts over a combined vote budget-matched at or above its own tokens (it uses 9–12% fewer). Recovers 53.6% of vote failures and keeps 98.4% of successes. | Conformity-robust aggregation |
| [2610.18786 v2](https://arxiv.org/html/2610.18786v2), Kim, Oct 2026 | Splits ensemble gain with Direct, Single-slot and Duplicate controls | API models on RAG QA; no maths, no SC | The repetition gap flips sign between panels: −6.33 vs +3.33 pts | The decomposition as a first. It obliges us to pre-register the control we report. |
| [AggLM](https://arxiv.org/html/2509.06870), Zhao et al., Sep 2025 | GRPO-trained aggregator with a binary `math_verify` reward | AggLM-1.7B from Qwen3-1.7B; K = 8 Qwen3-1.7B non-thinking candidates; AIME/HMMT 24–25 | +4.4 to +11.9 pts over maj@8. "On par" with MV when the majority is large. Its solver fine-tuning control matches data, not tokens. | "A trained reader beats voting" and "a 1.7B model reads its own samples" |
| [RL-GenSelect](https://arxiv.org/html/2602.02143v1), Toshniwal et al., Feb 2026 | DAPO-trained generative selector | Qwen3-1.7B over Qwen3-1.7B candidates; trained on N = 2–16 | HMMT25 from 25.00 (maj@8) to 36.25. All-wrong pools excluded from training; no compute matching against SC. | The same two claims |
| [SSA](https://arxiv.org/html/2506.09014v2), Qi et al., Jun 2025 | GRPO-trained 0.5B–3B aggregators | Qwen2.5 base; K = 5 candidates from 7B+ models | SSA-0.5B averages 54.71 vs 52.97 for a 7B PRM. RL outputs collapse to a template plus an answer, so it selects rather than synthesises. | Small trained readers. A warning that RL readers become selectors. |
| [TTT-Consolidate](https://arxiv.org/html/2611.02104v1) (2611.02104), Wu et al., Nov 1 2026, NeurIPS 2026 | Fits a LoRA adapter on the trajectories at test time, from a start point meta-learned by SFT or GRPO-style RLVR | Frozen Qwen3-1.7B/4B; trajectories from six external LLMs; Minority/Majority/None tiers, roughly c1/c2+/c0 | Minority tier at matched FLOPs: MV@8/13/18 22.7/22.2/22.6 vs 35.4. Answer swap: in-context aggregators lose up to 52 pts when prompted and 64 when RL-trained, against 5–17 for TTT. In-context baselines recover 0.8–6.7% of all-wrong pools. | Controls and compute matching on a trained aggregator at 1.7B. Not covered: own-model samples, in-context reading, token-matched SC, a fresh-solve contrast within strata. |
| [SPIRAL](https://arxiv.org/html/2606.23595v2) (2606.23595), Hamid et al., Jun 2026 | One model trained to produce trace sets useful to its own aggregator | Qwen3-4B-Instruct-2507; GRPO baseline matched at 98,304 training tokens per problem | GRPO and SPIRAL perform similarly under majority voting. SPIRAL pulls ahead only under recursive aggregation. | A solver trained for an aggregator; a matched solver-RL baseline |
| [ParaGator](https://arxiv.org/html/2603.18886v1) (2603.18886), Mar 2026 | Online joint GRPO with a pass@k advantage on the candidates | Qwen3-4B-Base and 4B-Instruct-2507 | Prompted aggregation lowers 4B-Base from 15.45 to 11.54 over three rounds. Trained: Dr.GRPO 28.13, offline AggLM-style 31.46, online multitask 27.46, ParaGator 34.00. | Joint training as new |

Two more papers lower the prior for the solver clause. Standard RL makes Qwen3-4B a worse input to recursive aggregation in 4 of 5 tasks ([RSA](https://arxiv.org/html/2509.26626v2)). At 1.7B, single-agent RL matches trained Voting workflows, 50.5 vs 49.9, matched on steps ([2605.24202](https://arxiv.org/html/2605.24202v2)).

What this means:

- **Parts A and B:** adopt the 2608.18379 strata and contrasts as the measurement standard, add TTT-Consolidate's answer swap, and cite both as such. Joint training exists only for one shared model at ≥4B, so a frozen, separate reader at ≤1.7B is open.
- **Distillation:** dropped. Phase 1 found transcripts barely above rejection-sampled solutions, and at c0 untrained aggregation scored 0.155 against 0.277 for a fresh solve.
- **The Channel Audit:** 2608.18379's placebo cost (−16.4 vs a fresh solve, at c0) is similar in size to the audit's form effect (−14.0 vs self-reflection). The references differ, so the two support each other without being the same quantity. The audit stays the revision half of the thesis.

## Direction, gap and novelty claim

**Part A.** Train Qwen3-1.7B with GRPO to read K of its own samples and give one answer (a reader). Measure what training changes: selection, recombination, fresh solving, form, and reliance on answer frequency. Compare everything against SC matched per problem under three cost rules.

**Part B.** Turn the largest risk, that solver-GRPO matches the reader, into a measured axis. Cross the candidate source with the aggregator, train one joint solver-and-reader, and, last, train solvers against a frozen reader.

**Motivation from your audit.** Revision lost to sampling, not to itself.

- **Content and form.** Plain self-reflection improved agents by +3.2 points net. The −14.0 is the form effect of reading three solutions to a different problem (placebo minus self-reflection). Content measured against placebo (+13.3 to +17.4) mostly cancels that cost. Against self-reflection, content adds between −0.7 and +3.4 points net, and only `full_solution` is above +0.1.
- **Price.** Every real channel lost to matched SC, by 0.8–3.3 points. One revision round lifts the three-sample vote (69.7%) by about 2 points for about 2.2 samples' worth of tokens. 2.2 more SC samples give 3.8 points (73.5%).
- **Headroom.** Only 31 of 1,200 replicates (2.6%) had a right answer the vote missed. Untrained `full_solution` recovered 11 of those 31 (36.6%, 95% CI 20.4–53.8%), but broke about 22 of the 801 replicates the vote already had right.

A reader pays for one output instead of N regenerations, so it attacks the price. The open question is whether training makes one read worth more than the SC samples it costs. Level 5 is the right primary population. Round-0 disagreement was 53% there, against 29% at level 3. `full_solution`'s gap to SC was −0.5 [−3.7, +2.8] at level 5, against −2.9 at level 3.

**The bar the reader must clear.** On the 600 dev problems, SC scores 63.7%, 69.7%, 73.4% and 76.8% at k = 1, 3, 5 and 9. One sample costs about 819 generated tokens. The table is the report's calculation, interpolating about 0.85 points per sample between k = 5 and 9. The last two rows are illustrative.

| Accounting rule, K = 4 | Reader cost in SC samples | k\* | SC at k\* (dev) | Reader accuracy H1 requires |
| --- | --- | --- | --- | --- |
| Generated tokens only (pre-registered primary) | 4 + ≈1 | ≈5 | 73.4% | ≥74.9% |
| Generated + prompt tokens (= FLOPs for a same-size reader) | 4 + ≈4.1 | ≈8 | ≈76.0% | ≥77.5% |
| FLOPs, 0.6B reader | 4 + ≈1.5 | ≈5.5 | ≈73.8% | ≥75.3% |
| Generated + prompt, with a gate that reads 45% of problems | 4 + ≈1.9 | ≈5.9 | ≈74.1% | ≥75.6% |

A four-sample vote lies between 69.7% and 73.4%. To tie SC, the reader must gain about 2 points over its own vote under the generated-only rule, and about 5 under the prompt-inclusive rule. Under the generated-only rule, its overhead stays near one sample whatever K is, while SC's slope falls from about 1.85 points per sample (k = 3 to 5) to 0.85 (k = 5 to 9), so reading gets cheaper as K grows. Under the prompt-inclusive rule, k\* ≈ 2K + 1 and the price does not shrink. K is a design variable, not an ablation.

| Question | Already done by | Still open |
| --- | --- | --- |
| Does a trained aggregator beat majority voting? | AggLM, SSA and RL-GenSelect, at equal K | SC matched per problem on tokens, prompt tokens counted, on 1,000+ problems |
| Can a 0.6–1.7B model read its own samples? | AggLM-1.7B, RL-GenSelect-1.7B, SSA-0.5B/1.5B | Nothing; no longer claimed |
| Is the comparison compute-matched? | AggLM@8 vs MV@16 (in samples); TTT-Consolidate (FLOPs, against MV); SPIRAL (training tokens); 2610.21290 | Generated-token, FLOP and GPU-second SC curves on 1,000+ problems |
| Where does the gain come from? | 2608.18379 (untrained); TTT-Consolidate (trained, external trajectories, in weights) | The full battery by stratum on own samples, in context, before and after RL |
| Is the gain from aggregation or just from training? | RSA, ParaGator, SPIRAL at ≥4B; AggLM (matched on data) | Solver-GRPO matched on training tokens at 1.7B, non-thinking |
| Can a solver be trained to feed an aggregator? | SPIRAL and ParaGator, one shared 4B model | A frozen, separate reader at ≤1.7B, with strata and token-matched SC |
| Can aggregation resist conformity? | 2610.21290 (training-free); ACPO | Not a headline. Open: whether RL raises answer-frequency reliance. |

**Novelty claim:** the first counterfactually controlled account, at ≤1.7B and on 1,000+ problems, of what GRPO changes in an in-context reader of its own model's samples (selection, recombination, fresh solving, form and answer-frequency reliance), and of how the solver's training objective changes what a frozen reader can extract, all against per-problem token-matched SC under three cost rules. If the factorial is cut, the second half rests on the grid's two solver objectives, base and GRPO.

**No longer claimed:**

- a trained small aggregator that beats voting;
- compute matching against voting, or controls on a trained aggregator, as such;
- placebo, candidate-free or answer-only controls as new;
- the form effect as new;
- a conformity fix as new.

**Search before every submission.** The Nov 13 search used the citation graphs of AggLM, RSA and SSA plus keyword searches. It did not browse the NeurIPS 2026 or EMNLP 2026 accepted-paper lists, or ICLR 2027 OpenReview, so do that first. Post a timestamped design pre-registration in Stage 0, and keep the monthly search. AggLM's authors have moved to joint training, and the TTT-Consolidate group already runs placebo-style controls.

## Hypotheses and contributions

H1 is the positive result. H2–H4 make Part A defensible. H5 and H6 cover Part B and the gate. Freeze all of them before any trained model is evaluated, as you did for the audit. Holm runs within two families: H1–H4, and H5–H6.

**Family 1, Part A**

1. **H1, positive effect.** A Qwen3-1.7B GRPO reader, reading K_H1 candidates from base Qwen3-1.7B, beats base SC at per-problem matched generated tokens on the primary set. K_H1 ∈ {4, 8} is fixed at G1.
   - The test is superiority, decided by the Holm-adjusted p-value of the problem-level paired bootstrap (PREREG D24); the 95% interval is the effect estimate, not the decision. The planning effect is 1.5 points. It sizes the test set and is not a threshold on the point estimate. A "≥ 1.5 and CI > 0" test fails half the time when the true effect is exactly 1.5. At SD 0.18 the power at 1.5 points is 74% for one unadjusted test and 48% at the Holm worst case (PREREG 6.3).
   - Second clause, now its own test H1b: the reader also beats SC from the stronger solver-GRPO variant, chosen on dev, on the primary set (not pooled with the secondary set; PREREG D4).
   - Report H1 under the other two cost rules as pre-registered estimates.
2. **H2, it reads rather than re-solves.** After training, the reader's gain over its own candidate-free fresh solve grows at c1 by at least 10 points over the untrained reader's. Its candidate-free accuracy rises by less than its full-input accuracy. Pre-registered prediction: the gain at c0 stays at or below 0. Report H2 also on the problems the contamination probe (run after the freeze, PREREG D13) does not flag for the studied model.
3. **H3, the form effect shrinks, in each stratum (non-inferiority).** With placebo candidates, the trained reader is not more than 2 points worse than its own candidate-free accuracy in each stratum (an intersection test, not an equivalence test; PREREG section 4 and O2). 2608.18379's −16.4 is a c0-only figure, so only the c0 cell compares with it. The audit's −14.0 is not comparable: its reference is self-reflection, not a candidate-free solve.
4. **H4, conformity drops without losing the vote.** At c2+ the trained reader is non-inferior to plurality voting (margin 1 point). Its copy rate falls below the untrained reader's and the deduplicated-input baseline's. Also pre-register TTT-Consolidate's competing prediction: RL *increases* reliance on answer frequency, so the trained reader loses more than the untrained one under the answer swap. The answer-swap contrast, two-sided, on the sets where the correct answer was the strict plurality (PREREG D25), decides between them.

**Family 2, Part B and the gate**

5. **H5, solver objective against reader.** On the grid, solver-GRPO raises maj@K but lowers answer entropy and shrinks the trained reader's margin over the vote, as RSA, SPIRAL and the Olmo diversity result predict. If the factorial runs, reader-marginal training raises the c1 share and the frozen reader's accuracy without raising maj@K. If it is cut, its contrasts leave the family untested.
6. **H6, the gate.** A unanimity-gated reader (vote if all K agree, read otherwise) beats adaptive SC at matched cost. Fix the cost rule for this test at the freeze.

**Exploratory, no Holm:** RSA (T ≤ 3) against matched SC; the joint arm's candidate-free gap on its own weights; a learned gate or verification bit; collusion (the frozen reader's copy rate on a new solver's wrong candidates); position and length bias.

Expected contributions:

- **C1 (if H1 holds):** a compute-matched positive result for an in-context reader at 1.7B, against base SC and a token-matched solver-GRPO baseline, under all three cost rules.
- **C2 (whatever H1 does):** the first stratified, counterfactual breakdown of what GRPO changes in an in-context reader of its own samples. If RL turns the reader into a voter, as TTT-Consolidate predicts, that is the finding.
- **C3 (Part B):** whether the diversity that helps a vote is the diversity that helps a reader, measured with c0/c1/c2+ strata at 1.7B.
- **C4:** how far the verdict depends on the cost rule, the trained-aggregator counterpart of 2610.18786's control dependence, plus the gate against adaptive SC.
- **C5:** the bridge to your audit. The audit showed content moving agents both ways and cancelling at the vote. Part A tests whether RL teaches the reader to lean on the vote instead of reading.

## Models, data and metrics

Stay with Qwen3-1.7B in non-thinking mode, so the study shares its candidate model, grader and token accounting with the audit.

| Role | Model | Why |
| --- | --- | --- |
| Candidates, primary | Qwen3-1.7B, non-thinking | Same model as the audit; about 819 generated tokens per sample |
| Reader, solver-GRPO and joint arm | Qwen3-1.7B, full fine-tune with GRPO | Same size as the solver, so "bigger model" cannot explain the gain |
| Reader, FLOP-accounting arm | Qwen3-0.6B | Costs about 35% per token of a 1.7B reader under FLOPs; SSA-0.5B beat a 7B PRM |
| Harness check and scale point | Qwen3-4B-Instruct-2507 | The model of 2608.18379 and RSA |
| Verifier baseline | Qwen2.5-Math-PRM-7B | About 4× the solver's parameters, so its FLOPs are counted |
| External trained baseline | AggLM-1.7B, if public (not verified) | Rules out a faulty reimplementation |
| Contamination probe | Qwen3-1.7B-Base and Qwen3-1.7B | Qwen3-4B-Base rebuilds MATH-500 problems from 60% prefixes at 21.8% exact match ([2507.10532](https://arxiv.org/html/2507.10532v3)) |
| Transfer candidates | Llama-3.2-3B-Instruct, Qwen2.5-Math-1.5B, with a plain prompt and a format gate | Under the round-0 prompt, Llama left 70.4% of outputs unboxed |

| Split | Source | Size | Samples per problem | Use |
| --- | --- | --- | --- | --- |
| Train | MATH train levels 3–5 plus a DeepScaleR subset, deduplicated against every test set | About 8k | 16 | Pools and GRPO |
| Dev | The audit's 600 confirm problems (208 at level 5) | 600 | 18, existing | G1, Stage 1, paired SD, checkpoint rule; never a test set |
| Primary test | MATH test level 5 minus every screen, pilot and dev problem ([Epoch](https://epoch.ai/benchmarks/math-level-5)) | At most 1,116 (1,324 minus 208), less the screen and pilot level-5 problems | 32 | H1–H4; sealed until Stage 3 |
| Secondary test | OlympiadBench OE\_TO\_maths\_en\_COMP ([EvalScope](https://evalscope.readthedocs.io/en/latest/benchmarks/olympiad_bench.html)) | 674 | 32 | Replicates H1; pooled solver clause |
| Robustness | MATH-P-Hard ([MATH-Perturb](https://arxiv.org/html/2502.06453v2)); drop its items seeded from MATH train, or remove those seeds from the train pools | 279 | 32 | Perturbed level-5 problems |
| Post-release | HMMT Nov 2025, AIME 2026, HMMT Feb 2026 ([MathArena](https://matharena.ai/competitions)) | 93 | 32 | Contamination check with Bayesian intervals ([Don't Pass@k](https://arxiv.org/html/2510.04265v2)); not powered |
| Reported, not powered | AIME 2025 + HMMT Feb 2025 | 60 | 32 | G2 and comparison with 2608.18379 |

The confirm problems beyond the first 600 were never generated on; they are the power reserve.

Metrics, all computed per problem and then averaged:

- **Accuracy:** the boxed final answer under `math_verify`, with the audit's confidence-box rule. Pre-register this rule, because 2608.18379's answer-only result flips with the scoring rule. Average over the candidate-set draws (4, 8 or 16 on the primary set, fixed at the freeze by PREREG 6.3) × 2 replicates per problem.
- **Accuracy against cost, on three axes:** generated tokens (primary, as in the audit); parameter-weighted prompt + generated tokens, with SC's own prompt tokens counted; and measured GPU-seconds in vLLM, with prefix caching on for SC. FLOP-only accounting overrates small models, so wall-clock time is the tie-breaker ([Kinetics](https://arxiv.org/abs/2506.05333v2)).
- **Gain over a fresh solve:** reader with candidates minus reader without, by stratum c0/c1/c2+.
- **Recovery and retention:** the share of vote failures fixed and of vote successes kept, as in 2610.21290.
- **Copy rate:** the share of wrong outputs that equal a shown answer.
- **Form effect:** placebo minus candidate-free accuracy, by stratum.
- **Answer-swap drop:** accuracy lost when the majority answer is swapped, in c1 and c2+ sets.
- **Position and length bias:** accuracy by the slot of the lone correct candidate in c1 sets, and selection against candidate length.
- **Repetition gap:** Duplicate minus Single-slot accuracy, as in 2610.18786.
- **Diversity and health:** pass@k, answer entropy, right-to-wrong flips, output length (to catch SSA-style collapse) and truncation rate.

## Baselines and ablations

Reviewers will attack accounting and baselines before results. Three arms answer "a better solver, not a better reader": the matched solver-GRPO baseline, the candidate-free control and the grid. The zero-cost selectors answer "it just learned confidence weighting". They will likely tie SC, and a tie is useful.

| Arm | Kind | What it rules out | Priority |
| --- | --- | --- | --- |
| SC from base Qwen3-1.7B, matched per problem on all three axes | Baseline | The primary comparison | Must |
| SC from solver-GRPO matched on rollout + pool tokens (≈1.3–1.7× a solver run), plus its step-300 checkpoint. The stronger on dev is the H1 comparator. | Baseline | "Any RL training would have given this" | Must |
| Untrained reader, same prompt | Baseline | "Prompting alone does it" | Must |
| Deduplicated input: one trace per distinct answer, in frequency order (2610.21290's bias control, without style normalisation) | Baseline | "Hiding vote counts fixes conformity for free" | Must |
| DeepConf-offline, a self-certainty Borda vote, and a vote weighted by the stored `Confidence: N` (no extra generation) | Baseline | "The reader learned confidence weighting" | Must |
| Early-stopping or adaptive SC ([ESC](https://arxiv.org/abs/2401.10480), [Adaptive-Consistency](https://arxiv.org/abs/2305.11860)) | Baseline | "The gate is just adaptive compute" | Must |
| Candidate-free: the reader solves from the problem alone, same output cap | Control | Fresh solving dressed up as aggregation | Must |
| Answer-only: shown answers without reasoning | Control | "Only the answer values matter" | Must |
| Placebo on every stratum: length-matched solutions to other problems of the same subject and level | Control | The form effect of reading candidates | Must |
| Answer swap in c1 and c2+ sets | Control | A voter passing as a reader | Must |
| K ∈ {2, 4, 8} at evaluation | Evaluation | Population size; the price of reading depends on K | Must |
| The candidate × aggregator grid: {base, solver-GRPO candidates} × {vote, untrained reader, trained reader} (the report's "2×2") | Evaluation | Tests RSA's "standard RL makes worse inputs" and SPIRAL's "GRPO ties under voting" at 1.7B | Must |
| Unanimity gate: vote if all K agree, read otherwise | Evaluation | Paying for a read where the vote already agrees | Must |
| Train without c0 sets | Training ablation | How much of the gain comes from learning all-wrong pools | Must |
| Qwen3-0.6B reader | Training ablation | A reader smaller than the solver; the FLOP-accounting arm | Must |
| Joint solver-and-reader, with its own candidate-free control | Training arm | The cleanest same-weights attribution | Must |
| Shortest-answer selection ([short-m@k](https://arxiv.org/html/2505.17813v2)); CISC with P(True) ([CISC](https://arxiv.org/abs/2502.06233v2)); USC | Baseline | Length heuristics; prompted self-weighting and selection | Should |
| AggLM-1.7B checkpoint, if public, run through our harness | Baseline | A faulty reimplementation | Should |
| Vote weighted by Qwen2.5-Math-PRM-7B, with verifier FLOPs counted (≈4× generation; it faces SC at about k = 20 under FLOP matching) | Baseline | "A standard verifier does as well" | Should |
| RSA, N = 8, K = 4, T ≤ 3, generated-only accounting | Baseline | "Iteration beats training" | Should |
| Duplicate (one candidate × K) and Single-slot (one candidate, other slots empty) | Control | Diversity vs repetition (2610.18786) | Should |
| Factorial: a pass@k-advantage solver (1 seed) and a reader-marginal solver trained against the frozen reader (2 seeds), each read by all three aggregators | Training arms | Which solver diversity a reader can use | Should; last, cut first |
| Pairwise knockout judged by base Qwen3-1.7B, judge passes counted | Baseline | "Pairwise comparison suffices" | Optional |

"Must" arms go in the main table. "Should" arms go in ablations. If Stage 3 runs late, cut the factorial first. Dropped from Nov 12: training on deduplicated input, minority-rescue upweighting, RSA with T up to 5, and distillation.

## Roadmap and go/no-go gates

As before, the first training run starts after about two weeks of cheap checks. Stage 0 now needs no GPU. Nothing before Stage 3 looks at the primary set beyond run quality.

| Stage | When | Work | Gate | Pass condition | If it fails |
| --- | --- | --- | --- | --- | --- |
| 0 | Days 1–3 | Break-even on the dev logs; design pre-registration; AggLM checks | G1 break-even | On level-5 dev, for some K ∈ {4, 8} under the generated-only rule: headroom × recovery (27–57%) ≥ break-even + 1.5 | Lead with C2 and Part B from the start; H1 becomes a pre-registered estimate |
| 1 | Days 4–10 | Harness check; sealed test pools; untrained readers and every baseline, off the primary set; paired SD on dev; the contamination probe is only rehearsed (on dev and the harness set) | G2 harness | Same signs as 2608.18379 at c0 and c2+ | Fix the prompt, extraction or truncation; no training until it passes |
| 2 | Days 11–14 | Train pools (16 samples) with mixed-K sets; 20-step throughput pilot, then a 50-step GRPO pilot; GPU-seconds measured; freeze | G3 pilot and freeze | Stable pilot; throughput fits the budget; PREREG and analysis freeze committed | Cut the factorial, then other Should arms. Keep ≥3 seeds for reader and solver-GRPO. |
| 3 | Weeks 3–5 (3 seeds) or 3–6 (5 seeds) | Reader × n_s (mixed K), solver-GRPO × n_s (step 300 kept), joint ×1, no-c0 ×1, 0.6B ×1, with n_s = 3 or 5 by the seed rule (PREREG D7, 8.1) | — | — | — |
| 3, evaluation | Week 6 (3 seeds) or 7 (5 seeds) | The contamination probe (frozen protocol); first touch of the primary set: solver-GRPO pools for the grid; all arms, controls, grid, K sweep and gate on every test set; frozen analysis run once | G4 framing | H1a and H1b supported: full C1. H1a only: narrower C1. | H1a not supported: lead with C2 (Findings or TMLR), whatever H1b says (PREREG 4.1) |
| 4 | Weeks 7–8 | Factorial (pass@k ×1, reader-marginal ×2) against the frozen reader; transfer pass; write-up | Cut rule | Factorial done by the pre-registered date | Drop its contrasts from family 2, untested |

Stage 3 is the result the paper rests on. Every earlier gate protects it from a null caused by a reader that costs more than the headroom can repay, or by a broken harness.

1. **Stage 0, days 1–3: break-even on logs you have.**
   - Download the raw shards from the private HF dataset `Arshia-HZ/mas-channel-audit-raw` (`confirm/qwen3-1.7b/`).
   - Each dev problem has 18 independent samples: 2 replicates × 9 (round 0 plus pool), all with the round-0 prompt. Enumerate all 3,060 four-sample subsets, and the 43,758 eight-sample subsets (my arithmetic).
   - This gives SC up to k = 18, plus maj@K, the headroom pass@K − maj@K, the c0/c1/c2+ shares, the minority-correct share and the unanimity rate, all exactly. Report them by level and judge G1 on the 208 level-5 problems.
   - Compute the break-even SC(k\*) − maj@K for each cost rule and K ∈ {4, 8}. Assume the reader writes about one sample until the pilot measures it. The GPU-seconds rule waits for Stage 2.
   - The recovery rate of 27–57% is AggLM's share of the pass@8 − maj@8 gap recovered on 1.7B thinking candidates (the report's calculation). Fix in advance which end G1 uses. I suggest the upper end, because G1 is a screen and Stage 3 is the test.
   - Write down the decomposition: gain = P(minority-correct) · rescue rate − P(majority-correct) · break rate + P(c0) · fresh-solve rate.
2. **Stage 1, days 4–10: harness, pools, untrained baselines.**
   - G2 runs the untrained prompt with Qwen3-4B-Instruct-2507 on AIME-25 + HMMT-25 at K = 4, with candidate-free, answer-only and placebo arms. Judge it only at c0 and c2+: 2608.18379's c1 contrast is unresolved, and its c2+ answer-only gap depends on the scoring rule.
   - Generate the 32-sample test pools on the pinned stack, about 51M generated tokens. On the primary pools, check only run quality (overflow, truncation, unboxed rate), as in M6.
   - Run the untrained 1.7B and 4B readers with every control and baseline on the dev, secondary and harness sets only. Running them on the primary set before H1–H4 are frozen would mean looking at test data before pre-registering. If the solver clause is pooled over primary and secondary, the secondary set is confirmatory too; then keep Stage 1 to dev and the harness set.
   - Measure the reader's paired SD on dev; it fixes the test-set size.
   - Rehearse the contamination probe (greedy raw continuation from a 60% prefix, whitespace-insensitive exact match) on dev and the harness set only. It measures memorisation, not any arm, but it is a computation on the problems, so it runs on the primary and secondary sets only after the freeze, with the protocol and the H2 split fixed first (PREREG D13).
3. **Stage 2, days 11–14: freeze.** After the pilots, freeze H1–H6, every threshold, K_H1, the cost rule for each test, the scoring rule, the checkpoint rule, the test-set size and the factorial's cut rule. Extend the analysis freeze so the new contrast code is hashed, and post a dated amendment.
4. **Stages 3 and 4.** Stage 3 re-runs the untrained arms on the primary set, since the Stage 1 dev table was only a pilot, with the draws of PREREG D15 × 2 replicates. The transfer pass reads Llama-3.2-3B and Qwen2.5-Math-1.5B candidates. The write-up uses the audit as its diagnostic half.

## Implementation details

Reuse the audit harness as it is: the grader with `math_verify` and the confidence-box rule, deterministic seeds, the append-only store and the analysis freeze. The reader is ordinary single-turn GRPO, where a prompt holds the problem plus its candidates.

**Repository (a recommendation for you to approve).** The reader study trains models, which is out of scope for the inference-only `mas-channel-audit` repository. I recommend a new repository with its own `PREREG.md`, depending on `channel_audit` pinned to a tag such as `audit-final`. This mirrors how Phase 1 was frozen at `phase1-final` with a pointer to the audit. The grader and token accounting are then imported, not copied, so there is still one grader. I also recommend a dated note in the audit's `PREREG.md` saying its M8 (second family, Qwen3-4B, mixed teams) is superseded by this roadmap.

**Candidate pools.**

- Draw 32 samples per test problem and 16 per train problem from base Qwen3-1.7B at temperature 0.7, with the audit's frozen round-0 prompt. Every sample then states `Confidence: N`.
- Store per-token log-probabilities for DeepConf and self-certainty. For the dev pools, one prefill pass recovers them.
- Estimate SC at any k by subsampling, so every SC point is free once the pools exist: exact on the 18-sample dev pools, Monte Carlo with fixed parameters on the 32-sample test pools (PREREG D23).
- Cap each candidate at 1,536 tokens inside the reader prompt, cut in the middle: the last 256 tokens are kept so a finished candidate keeps its final answer, and the marker counts inside the cap (PREREG D14). 2608.18379 cut candidates at their first answer, so state the difference.
- Use one pinned stack for every pool. The dev pools came from vLLM 0.33.0 in fp16, and the audit saw vLLM outputs change with batch composition. The simplest choice is to keep that stack for all pools and use BF16 only for training.

**Training sets.** For each train problem, draw candidate sets with K ∈ {2, 4, 8} and label each c0, c1 or c2+. Following AggLM, keep every hard set (majority wrong) and mix in easy sets. Check the ratio in the paper: the fetched text says "50% of easy sets", the Nov 12 text said "an equal number". AggLM found hard-only training worse (AIME24 64.22 vs 70.69). Shuffle candidate order, and never label which candidate is the reader's own.

**GRPO starting values.** Tune these in the pilot; they are not tested:

- full fine-tune in BF16 (1.7B fits on 96 GB with room for vLLM rollouts);
- learning rate 1e-6 with 10 warm-up steps;
- 64 prompts per step, group size 8, rollout temperature 1.0;
- response cap 2k. The prompt cap is 7k for K = 4. For K = 8, raise it to about 13k and `max_model_len` to about 16k (my arithmetic from the 1,536-token candidate cap).
- PPO clip 0.2 low and 0.28 high, KL coefficient 0 or 0.001;
- 300 steps, checkpoint every 25.

**Reward.** 1 if the boxed final answer is correct under `math_verify`, otherwise 0; a missing box counts as wrong. Fix the checkpoint rule before Stage 3: the last checkpoint, or the best of steps 100/200/300 on dev, never the peak of a continuous curve.

**Solver-GRPO baseline.** The same recipe on the same problems, with no candidates in the prompt. Match it on cumulative rollout tokens plus the reader's pool tokens (about 100M), as SPIRAL matched total training tokens. Save step 300 as the equal-steps variant.

**Joint arm.** One Qwen3-1.7B, trained with GRPO on a 50/50 mix of plain and reading prompts, as in RSA's aggregation-aware RL. It reads offline base-model candidates. At 4B, naive online multitask (27.46) trailed the offline AggLM-style aggregator (31.46).

**Factorial.** The pass@k-advantage solver follows ParaGator; pass@k objectives can cost pass@1 ([Tang et al.](https://arxiv.org/html/2503.19595v2)). The reader-marginal solver rewards each trace by the frozen reader's correctness on sampled sets that contain it. Start with SPIRAL's set-average estimator. If variance is high, fall back to the leave-one-out estimator, which [Team-GRPO](https://arxiv.org/html/2611.02404) proves unbiased. Avoid entropy bonuses: 0.005 collapsed training in [Pass@k Training](https://arxiv.org/html/2508.10751v1). A frozen 1.7B reader fits beside training on the 96 GB card ([Qwen docs: verl](https://qwen.readthedocs.io/en/stable/training/verl.html)). Frame it as decomposition at small scale, not as a new method.

**Gate and answer swap.** In the audit, 61.4% of round-0 teams already had at most one distinct answer, so the gate skips many reads. A unanimous-but-wrong set offers the reader only a fresh solve, which in-context aggregators manage under 7% of the time. The answer swap, in the sets where the correct answer was the strict plurality and a wrong parsable answer exists, replaces the final boxed answer of every correct candidate with the most frequent wrong answer (ties to the answer shown first), leaving the reasoning intact (PREREG D25). A voter follows the swap; a reader recovers the derived answer.

**Token accounting.** Count each problem's own tokens; do not use averages across problems.

```latex
B_{\text{gen}} = \sum_{i=1}^{K} g_i + g_{\text{read}}, \qquad
B_{\text{flop}} = \sum_{i=1}^{K} (p_i + g_i) + w \,(p_{\text{read}} + g_{\text{read}}), \qquad
k^{*} = B / \bar{c}_{\text{solve}}
```

Here g is generated tokens, p is prompt tokens, and w is the reader's parameter count over the solver's: 1 for the 1.7B reader, about 0.35 for 0.6B, about 4 for the PRM. c̄\_solve is one SC sample's cost under the same rule. The third axis is measured GPU-seconds. SC at fractional k\* is linearly interpolated, as in the audit; a problem whose k\* exceeds its pool is not estimable and is counted (PREREG D23). The audit's prompt-inclusive contrasts were "not estimable", because budgets passed nine samples. At K = 8 the prompt-inclusive k\* reaches about 17, hence 32 samples per test problem.

**Statistics.**

- The bootstrap is problem-level and paired, as in the audit (replicates and draws are averaged within a problem first); a nested resample of problems, then draws and replicates, is a sensitivity check. The decision is the Holm-adjusted p-value of PREREG D24, and the power is recomputed for the six-hypothesis family under Holm (PREREG 6.3).
- Run three seeds for the reader and solver-GRPO, or five if a 300-step run measures at most 20 GPU-hours (PREREG D7). Compare seed means with a permutation test over seeds and report every seed. With 3 vs 3 seeds there are only 20 relabellings, so the smallest one-sided p is 0.05; with 5 vs 5, 252 and 0.004. Treat that test as a robustness check, and the sign of every seed as a robustness criterion, never as a substitute for the test.
- Run every baseline through one harness, and release per-sample outputs ([Chandak et al.](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37)).

**Compute on the RTX PRO 6000.** This budget is the report's extrapolation from the Nov 12 estimate of 15–30 hours per 1.7B GRPO run, with 1.5–2× that for reader-marginal steps. It is unmeasured. Measure throughput in the Stage 2 pilot before committing.

| Job | Runs | Estimated GPU-hours |
| --- | --- | --- |
| Test pools at 32 samples; train pools at 16 | — | ≈24 |
| Reader GRPO, mixed K | 3 seeds | 45–90 |
| Solver GRPO, matched on rollout + pool tokens | 3 seeds | 60–150 |
| Joint solver-and-reader | 2 seeds | 35–70 |
| Solver trained against the frozen reader | 2 seeds | 50–120 |
| Pass@k-advantage solver | 1 seed | 15–30 |
| Reader trained without c0 sets | 1 seed | 15–30 |
| 0.6B reader | 1 seed | 8–15 |
| Evaluation of all arms, controls, the grid and the K sweep | — | 48–96 |
| **Total** | | **≈300–625 h, or ≈1.8–3.7 GPU-weeks** |

This table is the Nov 13 estimate. PREREG 8.1 gives the budget and timeline for both seed counts with the joint arm at one seed: about 283–590 h with 3 seeds and 374–653 h with 5.

**Software.**

- Stay in BF16 for training and avoid FP8: block-scaled FP8 crashes at load on this GPU family under vLLM unless `VLLM_USE_DEEP_GEMM=0` is set ([vLLM #47436](https://github.com/vllm-project/vllm/issues/47436)).
- Pin one torch, CUDA and vLLM combination for all generation, and fall back to SDPA attention if FlashAttention wheels fail.
- Run rollouts on the same GPU as training (colocated).

## Risks, limitations and fallbacks

The biggest risk is still that training makes a better solver rather than a better reader. Part B now measures that outcome on its own axis, so the paper survives either answer.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Reading costs more than the headroom can repay | G1 on level-5 dev | Lead with C2 and Part B; H1 becomes a pre-registered estimate |
| Gain is fresh solving | Candidate-free accuracy rises as much as full-input; no-c0 keeps the gain only on c1/c2+ | Report C2 as the headline ("trained readers mostly re-solve"), as a correction to AggLM and SSA. Aim for Findings or TMLR. |
| RL turns the reader into a voter | A large answer-swap drop | Report it as C2's main finding; it matches the audit's content cancelling at the vote |
| RL collapses the reader into a selector, as in SSA | Output length falls in training | Cheaper reading, which helps under prompt-inclusive accounting, but no recombination claim |
| Solver-GRPO matches the reader | H1's second clause; SPIRAL and 2605.24202 predict it | Part B measures it. Never drop the baseline to rescue H1. |
| Training collapses diversity, or the reader copies the majority | Copy rate and answer entropy on dev every 25 steps | KL 0.001 or a lower learning rate; deduplicated-input training as a fallback only |
| Contamination | The probe's exact-match rate | Split H2 by probe outcome; lean on MATH-P-Hard and the post-release pool |
| Scooped: related preprints arrived roughly monthly in 2026 | Monthly search, plus the NeurIPS, EMNLP and ICLR lists | Timestamped pre-registration now; preprint as soon as Stage 3 numbers are final |
| Reviewers attack the accounting or the baselines | Three axes and one harness exist by design | Lead with the prompt-inclusive axis if the gap holds there |
| Forking paths from wanting a positive result | Any change to H1–H6 after the freeze | Freeze first; label everything else exploratory |
| The stack breaks on Blackwell, or pools differ across stacks | Errors in the pilot; precision differences from the fp16 dev pools | BF16 training, pinned versions, SDPA, one generation stack |

Limits to state in the paper:

- one model family for candidates in the main result;
- maths only, non-thinking mode only, one GPU type;
- three seeds per main arm;
- most cited papers are arXiv preprints whose numbers may change; TTT-Consolidate and 2608.18379 v2 are the exceptions.

If both H1 and C2 come out weak, the thesis falls back to diagnostic chapters on one shared harness: the Channel Audit, the untrained-reader study, and the grid with the matched solver-GRPO comparison.

## Sources

- [2608.18379 v2: Selection, Recombination, or a Fresh Solve? (COLM 2026 Workshop on Efficient Reasoning)](https://arxiv.org/html/2608.18379v2)
- [2610.21290: Bias Control and Injection for Multi-Agent Aggregation](https://arxiv.org/html/2610.21290v1)
- [2610.18786: Where Do LLM Ensemble Gains Come From?](https://arxiv.org/html/2610.18786v2)
- [AggLM: RL training for solution aggregation](https://arxiv.org/html/2509.06870)
- [RL-GenSelect: Learning Generative Selection for Best-of-N](https://arxiv.org/html/2602.02143v1)
- [SSA: learning to reason across parallel samples](https://arxiv.org/html/2506.09014v2)
- [TTT-Consolidate (2611.02104): Consolidating Reasoning with Test-Time Training, NeurIPS 2026](https://arxiv.org/html/2611.02104v1)
- [SPIRAL (2606.23595): Learning to Search and Aggregate](https://arxiv.org/html/2606.23595v2)
- [ParaGator (2603.18886): on-policy reward modeling and test-time aggregation](https://arxiv.org/html/2603.18886v1)
- [RSA: Recursive Self-Aggregation](https://arxiv.org/html/2509.26626v2)
- [2605.24202: When does multi-agent RL improve LLM workflows?](https://arxiv.org/html/2605.24202v2)
- [Tang et al.: inference-time objectives with RL](https://arxiv.org/html/2503.19595v2), [Pass@k Training](https://arxiv.org/html/2508.10751v1), [Team-GRPO](https://arxiv.org/html/2611.02404)
- [ESC](https://arxiv.org/abs/2401.10480), [Adaptive-Consistency](https://arxiv.org/abs/2305.11860), [CISC](https://arxiv.org/abs/2502.06233v2), [short-m@k](https://arxiv.org/html/2505.17813v2)
- [Kinetics: test-time scaling cost beyond FLOPs](https://arxiv.org/abs/2506.05333v2)
- [Chandak et al.: incorrect baseline evaluations](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37)
- [2507.10532: contamination of Qwen3 on MATH-500](https://arxiv.org/html/2507.10532v3)
- [MATH-Perturb (MATH-P-Hard)](https://arxiv.org/html/2502.06453v2), [MathArena competitions](https://matharena.ai/competitions), [Don't Pass@k](https://arxiv.org/html/2510.04265v2)
- [Epoch: MATH Level 5](https://epoch.ai/benchmarks/math-level-5)
- [EvalScope: OlympiadBench subsets](https://evalscope.readthedocs.io/en/latest/benchmarks/olympiad_bench.html)
- [Qwen docs: verl](https://qwen.readthedocs.io/en/stable/training/verl.html)
- [vLLM issue #47436](https://github.com/vllm-project/vllm/issues/47436)
- [Trained aggregators beyond voting (Nov 13 report)](research/trained_aggregators_beyond_voting.md); the other papers it cites (DeepConf, self-certainty, USC, Optima, Critique-RL and others) are linked there
- [Channel Audit report, M0–M7](https://github.com/Arshia-HZ/mas-channel-audit/blob/audit-final/reports/channel_audit_M0-M7_report.md)
- Nov 12 roadmap: `docs/history/roadmap_2026-11-12.md`
- Previous report: `reports/Thesis pivot after channel audit.md` (cited on Nov 12; not in `reports/` on Nov 13)
