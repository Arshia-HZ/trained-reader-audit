# Rules for this repository

This is a scientific measurement and training pipeline. A silent change can invalidate
results. It builds on the closed `mas-channel-audit` (tag `audit-final`), whose rules
carry over.

1. Implement only the current milestone. Do not refactor other code.
2. Never edit a frozen file (a frozen prompt directory, a freeze manifest, the frozen
   analysis files, a set manifest after the freeze) and never change `PREREG.md` except by a
   dated, appended entry the owner has approved. Decisions marked **[delegated]** stay open
   to the owner's review until the Stage 2 freeze.
3. One grader and one answer extractor: import `channel_audit.grade` and
   `channel_audit.extract` (pinned at `audit-final`). Never add a second one.
4. Never catch an exception to hide a failed generation. Record it.
5. The store is append-only. Never overwrite or delete a run.
6. Never use an arm's outcome to filter problems, candidates or candidate sets of an
   evaluation. The strata c0, c1 and c2+ are labels computed from candidate correctness.
7. Sealed sets (primary, secondary, robustness, post-release) are not scored before the
   Stage 2 freeze: only run quality (overflow, truncation, unboxed rate) may be read. Never
   train on a problem that is in dev, harness or any test set; the decontamination check is
   a test.
8. Every new behaviour needs a test. Run `pytest` and `ruff` before finishing.
9. Add no dependency without saying why.
10. Tests use mock backends and tiny fixtures. They must not download models or datasets.
11. No token or secret in any file, log or commit. Hugging Face and molab tokens are
    environment variables or typed prompts only.
12. Report what you ran and its output. Do not claim a test passed without running it.
