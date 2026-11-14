# trained-reader-audit

**Does RL training make a small reader beat self-consistency, and does it do so because it reads?**
A compute-matched, counterfactually controlled study of a Qwen3-1.7B "reader" trained with
GRPO to read K of its own model's samples and write one answer.

Successor to the [Channel Audit](https://github.com/Arshia-HZ/mas-channel-audit) (closed at tag
`audit-final`), where no inter-agent message channel beat budget-matched self-consistency.
Phase 1 is archived at [`causal-mas-distill`](https://github.com/Arshia-HZ/causal-mas-distill)
(tag `phase1-final`).

## Status

Stage 0 is done: on the audit's 600 dev problems, the break-even for a K = 8 reader is
0.8 points (generated tokens) and the gate G1 passes. The design decisions, hypotheses and
gates are in [`PREREG.md`](PREREG.md) (a draft until the Stage 2 freeze); the plan is
[`docs/ROADMAP.md`](docs/ROADMAP.md); the literature check behind it is
[`docs/research/`](docs/research/trained_aggregators_beyond_voting.md). Stage 1 (harness
check, pools, baselines) is in progress. No sealed test set has been scored.

## Milestones

| Milestone | Work | State |
| --- | --- | --- |
| R0 | Repository split, design decisions, scaffold, Stage 0 carried over | done |
| R1 | Set manifests; Stage 0 code under tests; dev pool loader | in progress |
| R2 | Dev baselines (stored-confidence vote, early-stopping SC), prompt lengths | |
| R3 | Reader harness: candidate sets, strata, prompts, controls, answer swap, planner | |
| R4 | GPU jobs of Stage 1 (G2 harness check, pools, contamination probe, untrained readers) | |
| R5 | Stage 2: train pools, GRPO pilot, freeze | |
| R6 | Stage 3: training runs, evaluation, frozen analysis | |

## Setup

The audit package reads its frozen prompts from its repository, so install it as an
editable checkout of the tag, then this package:

```bash
git clone --branch audit-final https://github.com/Arshia-HZ/mas-channel-audit.git ../mas-channel-audit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ../mas-channel-audit
pip install -e ".[dev]"
ruff check . && pytest
```

Extras: `[data]` (datasets, huggingface_hub), `[vllm]`, `[hf]`, `[figures]`. Raw generations,
pools and checkpoints go to private Hugging Face repositories, never to Git. No token is ever
written to a file: use the `HF_TOKEN` environment variable.

## Layout

```text
PREREG.md          decisions, hypotheses, gates (draft until the Stage 2 freeze)
AGENTS.md          rules for contributors and coding agents
docs/              roadmap, literature report, Stage 0 record, dev baselines
configs/models/    model configs (revision pinned)
prompts/           reader prompts (drafts until frozen)
data/sets/         set manifests and id lists (problem files are rebuilt, not committed)
src/reader_audit/  the package
scripts/           numbered scripts, one per step
tests/
```

Rules: see [`AGENTS.md`](AGENTS.md). Sealed sets are not scored before the freeze.
