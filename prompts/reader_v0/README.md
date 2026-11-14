# Reader prompts, version 0 (DRAFT)

Drafts for the Stage 1 harness check and the dev runs. They are **not frozen**: the owner
reads real samples first, and the wording is frozen as `reader_v1` at the Stage 2 freeze
(PREREG section 9). Until then every generation records the hash of the draft it used.

- `reader.txt`: the problem and K candidate solutions (the reader, the placebo, the
  duplicate, single-slot, deduplicated and answer-swap controls all render through it).
- `reader_no_candidates.txt`: the same instructions with the candidates removed (the
  candidate-free control).
- `reader_answers.txt`: the K extracted answers only (the answer-only control).
