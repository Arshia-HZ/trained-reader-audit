# Reader prompts, version 1 (frozen)

The wording of `reader_v0`, unchanged, frozen as `reader_v1` at the Stage 2 training
freeze (PREREG R17) after the owner read all three templates and real pool samples
(2026-11-16). `MANIFEST.json` holds the hash of each template; a test checks the files
against it, so a change fails CI. Every generation records the hash of the template it
used, and these hashes are the ones the dev runs recorded for `reader_v0`.

- `reader.txt`: the problem and K candidate solutions (the reader, the placebo, the
  duplicate, single-slot, deduplicated and answer-swap controls all render through it).
- `reader_no_candidates.txt`: the same instructions with the candidates removed (the
  candidate-free control).
- `reader_answers.txt`: the K extracted answers only (the answer-only control).
