"""The checkpoint rule (PREREG D17) and the set-up of its dev evaluation (R19).

The rule is: the best of the milestones 100, 200 and 300 by the dev gain over matched
self-consistency (H1a of ``13_dev_reader_analysis.py``, the generated-only rule), an
exact tie going to the later step. ``select_checkpoint`` is the whole rule: no
tolerance, no smoothing, no other quantity, and a milestone that is missing or not
finite stops it.

The evaluation differs from the untrained reader's dev run (R13) in the weights only.
``identity_problems`` lists everything that could differ besides them: the sampling
defaults, the chat template, the tokenizer and the weight format.

``seeds_report`` describes one milestone across the reader's seeds (R28); it selects
nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from channel_audit.models import ModelConfig

from reader_audit import reader_eval as ev
from reader_audit.dev_job import DONE
from reader_audit.pools import ProblemPool
from reader_audit.subsets import group_answers, sc_curve

MILESTONES: tuple[int, ...] = (100, 200, 300)
LEVEL, K, DRAWS, REPLICATES = 5, 8, 16, 2  # R13's design
TEMPERATURE = 0.7  # R13's, not the training 1.0
SELECTION_ARM, FREE_ARM = "reader_full", "candidate_free"
# docs/dev_reader.json, 16 draws, for reference only
UNTRAINED_H1A = -0.03218233009773353
REQUIRED_FILES: tuple[str, ...] = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
BASE_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
CONFIG_KEYS = ("architectures", "hidden_size", "num_hidden_layers", "vocab_size")


def select_checkpoint(gains: Mapping[int, float]) -> int:
    """The step with the highest gain; an exact tie goes to the later step."""
    if set(gains) != set(MILESTONES):
        raise ValueError(f"the rule needs exactly the milestones {MILESTONES}")
    bad = [s for s, g in gains.items() if not math.isfinite(g)]
    if bad:
        raise ValueError(f"gain not finite at steps {sorted(bad)}")
    best = MILESTONES[0]
    for step in MILESTONES[1:]:
        if gains[step] >= gains[best]:
            best = step
    return best


def milestone_folder(root: str | Path, run: str, step: int) -> Path:
    """Where a milestone lives on a machine; the step is in the path, so the stored
    generations' keys (which carry the model id) never mix two milestones."""
    return Path(root) / run / f"step_{step:04d}"


def milestone_model_config(
    base: ModelConfig, folder: str | Path, revision: str
) -> ModelConfig:
    """The reader's model config with the weights of a local milestone.

    The audit's Qwen3 check keys on 'qwen3' in the model id and a local path has none,
    so the non-thinking setting is asserted here.
    """
    if base.chat_template_kwargs.get("enable_thinking") is not False:
        raise ValueError("a Qwen3 reader must run with enable_thinking: false")
    data = base.model_dump()
    data.update(model_id=Path(folder).as_posix(), revision=revision)
    return ModelConfig(**data)


def missing_files(folder: str | Path) -> list[str]:
    return [n for n in REQUIRED_FILES if not (Path(folder) / n).is_file()]


def ids_digest(prompts_ids: Sequence[Sequence[int]]) -> str:
    """One hash of the token ids of many rendered prompts."""
    h = hashlib.sha256()
    for ids in prompts_ids:
        h.update(json.dumps(list(ids)).encode())
    return h.hexdigest()


def use_base_tokenizer(
    folder: str | Path, base_files: Mapping[str, str | Path]
) -> None:
    """Make a milestone folder read its tokenizer and chat template from the base model.

    Weights come from the milestone and the tokenizer files from the base revision R13
    used, so encoding and decoding are the same code as for the untrained reader. The
    milestone's own tokenizer.json was written by a newer transformers and differs from
    the base's in serialisation flags; the base's chat template lives in
    tokenizer_config.json, so the separate template file is removed.
    """
    folder = Path(folder)
    for name in BASE_TOKENIZER_FILES:
        shutil.copyfile(base_files[name], folder / name)
    (folder / "chat_template.jinja").unlink(missing_ok=True)


def identity_problems(
    *,
    base_generation_config: Mapping,
    milestone_generation_config: Mapping,
    base_config: Mapping,
    milestone_config: Mapping,
    base_ids_digest: str,
    milestone_ids_digest: str,
) -> list[str]:
    """What differs between the base model's set-up and a milestone's, but weights.

    The library version that wrote a file is not a setting and is left out. The
    milestone's own tokenizer must encode real dev prompts to the base's token ids.
    """
    out = []
    drop = {"transformers_version"}
    if {k: v for k, v in base_generation_config.items() if k not in drop} != {
        k: v for k, v in milestone_generation_config.items() if k not in drop
    }:
        out.append("generation_config.json differs from the base model's")
    if base_ids_digest != milestone_ids_digest:
        out.append("the milestone's tokenizer encodes dev prompts differently")
    for key in CONFIG_KEYS:
        if base_config.get(key) != milestone_config.get(key):
            out.append(f"config.json {key} differs from the base model's")
    dtype = milestone_config.get("dtype", milestone_config.get("torch_dtype"))
    if dtype != "float16":
        out.append(f"config.json dtype is {dtype!r}, not float16")
    return out


def _curves(pools: Mapping[str, ProblemPool]) -> dict:
    return {
        pid: sc_curve(
            group_answers(
                [c.answer for c in pool.candidates],
                [c.correct for c in pool.candidates],
            )
        )
        for pid, pool in pools.items()
    }


def finished_shards(root: str | Path) -> list[Path]:
    return [s for s in sorted(Path(root).glob("shard_*")) if (s / DONE).is_file()]


def load_step_outcomes(roots: Sequence[str | Path]) -> list[ev.Outcome]:
    outcomes: list[ev.Outcome] = []
    for root in roots:
        for shard in finished_shards(root):
            outcomes += ev.load_outcomes(shard)
    return outcomes


def summarise_step(
    roots: Sequence[str | Path],
    pools: Mapping[str, ProblemPool],
    summarise: Callable,
    *,
    draws: int = DRAWS,
    k: int = K,
) -> dict:
    """The dev analysis of one milestone, by the same function as R13's."""
    outcomes = load_step_outcomes(roots)
    done = {o.problem_id for o in outcomes}
    use = {pid: p for pid, p in pools.items() if pid in done}
    out, _ = summarise(outcomes, use, _curves(use), draws, k=k)
    out["n_outcomes"] = len(outcomes)
    by_arm: dict[str, int] = defaultdict(int)
    for o in outcomes:
        by_arm[o.arm] += 1
    out["outcomes_by_arm"] = dict(by_arm)
    return out


def rule_report(
    per_step: Mapping[int, dict], n_problems: int, *, arm_generations: int
) -> dict:
    """Select by the rule if every milestone is complete; say what is missing if not."""
    gains = {s: per_step[s]["H1a"]["mean"] for s in per_step}
    missing = []
    for step, r in sorted(per_step.items()):
        if r["problems"] != n_problems:
            missing.append(f"step {step}: {r['problems']} of {n_problems} problems")
        if r["H1a"]["not_estimable"]:
            missing.append(f"step {step}: {r['H1a']['not_estimable']} not estimable")
        if r["outcomes_by_arm"].get(SELECTION_ARM, 0) != arm_generations:
            missing.append(
                f"step {step}: {r['outcomes_by_arm'].get(SELECTION_ARM, 0)} of "
                f"{arm_generations} {SELECTION_ARM} generations"
            )
    complete = not missing and set(gains) == set(MILESTONES)
    return {
        "rule": "D17: best of steps 100, 200 and 300 by H1a gain over matched SC "
        "(generated-only), an exact tie to the later step",
        "gains": {str(s): gains[s] for s in sorted(gains)},
        "untrained_reference": UNTRAINED_H1A,
        "complete": complete,
        "incomplete": missing,
        "selected": select_checkpoint(gains) if complete else None,
        "per_step": {str(s): per_step[s] for s in sorted(per_step)},
    }


SEED_QUANTITIES = ("H1a", "H2_single_arm", "H4a")


def seeds_report(
    per_run: Mapping[str, dict | None],
    n_problems: int,
    *,
    arm_generations: int,
    arms: Sequence[str] = (SELECTION_ARM, FREE_ARM),
) -> dict:
    """One milestone across seeds (R28): descriptive, it selects nothing.

    A seed is complete when it has every problem, no H1a that is not estimable and
    ``arm_generations`` generations in each arm (``None``: no finished shard). Only
    complete seeds enter the mean over seeds and the SD over seeds of each quantity's
    per-seed mean; the per-seed SD over problems is what the second freeze restates
    O3 and O4 with.
    """
    incomplete: dict[str, list[str]] = {}
    for run, r in per_run.items():
        if r is None:
            incomplete[run] = ["no finished shard"]
            continue
        why = []
        if r["problems"] != n_problems:
            why.append(f"{r['problems']} of {n_problems} problems")
        if r["H1a"]["not_estimable"]:
            why.append(f"{r['H1a']['not_estimable']} not estimable")
        for arm in arms:
            got = r["outcomes_by_arm"].get(arm, 0)
            if got != arm_generations:
                why.append(f"{got} of {arm_generations} {arm} generations")
        if why:
            incomplete[run] = why
    complete = [run for run in per_run if run not in incomplete]

    def finite(x) -> float | None:
        return x if x is not None and math.isfinite(x) else None

    across = {}
    for q in SEED_QUANTITIES:
        means = [
            m for run in complete if (m := finite(per_run[run][q]["mean"])) is not None
        ]
        across[q] = {
            "seeds": len(means),
            "mean": statistics.fmean(means) if means else None,
            "sd_over_seeds": statistics.stdev(means) if len(means) > 1 else None,
            "sd_over_problems_by_seed": {
                run: finite(per_run[run][q]["sd"]) for run in complete
            },
        }
    return {
        "note": "descriptive (R28): each seed by R13's summarise; selects nothing",
        "complete": complete,
        "incomplete": incomplete,
        "across_seeds": across,
        "per_run": dict(per_run),
    }


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
