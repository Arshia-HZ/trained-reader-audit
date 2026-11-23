# Paper plan: where should a small model's test-time tokens go?

State on 2026-11-24 (PREREG R26). Everything below that is not marked confirmatory is from dev
data and is exploratory. Literature claims marked "abstract" were read from abstracts or
search summaries only; the two papers marked "read" were read in full text.

## 1. Claim and evidence so far

**Working claim.** For a small hybrid model on hard maths, at matched generated tokens, the
three channels that let a model use other samples' work do not beat spending the same tokens
on its own reasoning:
- messages between agents;
- a GRPO-trained reader of K = 8 samples;
- candidates shown to a thinking solver.

The switch to thinking mode dominates all of them.

| Channel or policy | Result | Where |
| --- | --- | --- |
| Messages between agents (7 channels, preregistered) | precise null against budget-matched SC | Channel Audit, tag `audit-final` |
| GRPO-trained reader, step 100 (D17) | H1a +0.5 points against matched SC on dev; skill only local (+6.6 against the vote when the plurality is at most 3 of 8) | R20, R21 |
| Reader only on disagreement, after ESC (2, 8) | +1.7 [+0.6, +2.9] against ESC alone; +1.1 [−0.2, +2.5] against the SC/ESC envelope (generated rule only) | R21 (f) |
| One thinking solve (Qwen3-1.7B, cap 30,000) | 83.4 at 8.68 samples; +20.2 [+14.6, +25.8] over the SC/ESC envelope | R25 |
| The reader cascade, once thinking is on the envelope | −4.4 [−8.8, −0.2] (generated), −21.6 (prompt-inclusive) | R25 |
| ESC (2, 2) then think (DART's policy) | +2.7 [+0.2, +5.0] over the full envelope (generated rule only; +0.7 prompt-inclusive) | R25 |
| Thinking read of the 8 candidates minus a fresh thinking solve | −11.6 [−18.1, −5.2]; paired in E2, the read generated half the tokens (5,043 against 9,988) | R23, R25 |
| Headroom | perfect gate about 5 points over DART; perfect selector over 2 or 4 thinking solves about 5 to 6 points over their vote | R26 |
| Thinking read by stratum (E4) | thinking shortened everywhere (to a third or half). Credit: c0 −29.5 (half the reads adopt a wrong shown answer); c1 −26.5 (a lone right candidate is not recovered); c2+ +5.5 [−2.1, +12.7] at 37% of the thinking tokens | R27 |

## 2. Position against prior work

- **Single agent against multi-agent at equal thinking tokens:** Tran & Kiela, arXiv
  2604.02460 (abstract). Single agents match or beat multi-agent setups on multi-hop QA with
  larger models. Our cell differs: maths, small hybrid models, a trained aggregator,
  preregistration, and causal controls (placebo, answer swap, candidate-free). We do not
  claim to be first.
- **Automatic multi-agent systems against CoT-SC:** "The Illusion of Multi-Agent
  Advantage", arXiv 2606.13003 (read). Frontier models; SC fixed at 5 samples; dollar cost
  compared, not matched; no heterogeneous teams and no small models. It supports the audit's
  direction.
- **Debate against vote:** arXiv 2508.17536 (abstract). Cross-family teams: "Beyond
  Symmetric Agents", arXiv 2609.35875 (abstract). Across 23 models, mixed teams lose to the
  vote over their own roster at equal compute.
- **Gating thinking on cheap drafts:** DART, arXiv 2606.23181 (abstract). Our cascade is
  DART's first stage. We claim only its measured value against a random mixture of modes;
  Route to Reason, arXiv 2505.19435, already reports a random router.
- **Errors in context pull reasoning:** Contextual Drag, arXiv 2602.04288 (abstract), with
  partial fixes (denoising, a clean-slate SFT). The fork vocabulary comes from arXiv
  2607.05184 (read). Our E4 asks a narrower question: mixed-correctness peer sets, by
  stratum, at matched cost.
- **Thinking and its low-budget regime:** NoThinking, arXiv 2504.09858 (abstract). Our
  crossover between the 8,192 and 16,384 caps agrees with it.
- **Trained aggregators:** AggLM, arXiv 2509.06870, and the notes in `docs/research_notes/`.

**Directions checked and not pursued (2026-11-24):**
- *Merging debate-role adapters, then GRPO.*
  - All role data is in hand, so joint training on the union is the baseline merging must
    beat, together with solver-GRPO.
  - AgentArk (2602.03955) and IMAD (2604.24881) already distil a multi-agent system into
    one model.
  - A non-thinking student starts about 20 points below one thinking solve.
- *A latency-aware router as a method.*
  - Per-query strategy selection with token and wall-clock terms exists (2509.09864, and
    2505.19634).
  - The gate headroom is about 5 points.
  - What survives is a wall-clock cost rule in the audit.
- *Cross-family teams.* 2609.35875 answers the question, negatively. Token matching across
  tokenizers is ill-defined.
- *Repairing anchoring with RL in thinking mode.* The phenomenon is published (Contextual
  Drag). Thinking rollouts of about 5,000 tokens are beyond one GPU in 10-hour sessions at
  the scale needed. This is a judgement from the 23 GPU-hours of the 800-token reader run.

## 3. Plan

1. **Step 1 (now, no generation).** R26; E4 (the stratified read against fresh); this plan.
2. **Step 2, confirmatory.** A second freeze, then the main claims on the primary set (972
   problems), logged before the freeze:
   - (a) one thinking solve against the SC/ESC envelope;
   - (b) the reader cascade against the envelope with thinking;
   - (c) the thinking read against the fresh solve;
   - (d) the gate's value against a random mixture of modes;
   - (e) H1a of section 4 as preregistered, with the trained reader at step 100.

   Under the generated and prompt-inclusive rules, plus a measured wall-clock rule.
   Inference only; estimate one to two molab sessions (judgement).
3. **Step 3, for a paper.**
   - Qwen3-0.6B and 4B.
   - AIME 2025 and HMMT February 2025 (the pinned harness set).
   - The E4 mechanism with controlled shown sets (all right, mixed, all wrong; answers only;
     placebo).
   - Inference only; estimate a few tens of GPU-hours (judgement).
4. **Option E5.** Recombining partial programs on code (R26), only after its own entry and
   a kill test.

**Venues, realistically.** Findings of ACL/EMNLP, COLM, or a NeurIPS/ICLR workshop. A main
track needs step 3 and stays uncertain.

## 4. Risks

- The dev effects may shrink on the primary set. The thinking effect is large, so that risk
  falls mostly on the small effects (DART's +2.7, the reader's +1.7).
- Reviewers may call the mode-switch result known. The contribution must rest on the
  matched-cost audit across channels and on the controls, not on that number.
- Latency is not yet measured; one serial thinking chain is slower than parallel samples.
