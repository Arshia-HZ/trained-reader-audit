"""The checkpoint rule (D17), its identity checks and its report on mock runs."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.models import ModelConfig, load_model_config
from channel_audit.runner import RunConfig

from helpers import write_pool_shard
from reader_audit import checkpoint_rule as cr
from reader_audit.dev_job import run_dev_reader
from reader_audit.pools import load_pools

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "13_dev_reader_analysis.py"
K, DRAWS, REPS, N_PROBLEMS = 4, 8, 2, 40


def test_the_best_milestone_is_selected():
    assert cr.select_checkpoint({100: -0.05, 200: -0.01, 300: -0.03}) == 200
    assert cr.select_checkpoint({100: 0.02, 200: -0.01, 300: -0.03}) == 100
    assert cr.select_checkpoint({100: -0.05, 200: -0.04, 300: 0.01}) == 300


def test_an_exact_tie_goes_to_the_later_step_and_nothing_else_does():
    assert cr.select_checkpoint({100: 0.01, 200: 0.01, 300: 0.0}) == 200
    assert cr.select_checkpoint({100: 0.01, 200: 0.01, 300: 0.01}) == 300
    # no tolerance: a difference of one millionth of a point is a difference
    assert cr.select_checkpoint({100: 0.01, 200: 0.01 - 1e-12, 300: 0.0}) == 100


def test_the_rule_needs_all_three_milestones_and_finite_gains():
    with pytest.raises(ValueError):
        cr.select_checkpoint({100: 0.0, 200: 0.1})
    with pytest.raises(ValueError):
        cr.select_checkpoint({100: 0.0, 200: math.nan, 300: 0.1})


def test_a_milestone_config_keeps_the_base_settings_and_pins_a_local_path():
    base = load_model_config(ROOT / "configs" / "models" / "qwen3-1.7b-reader.yaml")
    folder = cr.milestone_folder("/home/marimo/milestones", "reader_s1", 300)
    assert folder.as_posix() == "/home/marimo/milestones/reader_s1/step_0300"
    model = cr.milestone_model_config(base, folder, "a" * 40)
    assert model.model_id.endswith("reader_s1/step_0300") and model.revision == "a" * 40
    assert model.chat_template_kwargs == {"enable_thinking": False}
    for field in ("dtype", "max_tokens", "max_model_len", "stop_token_ids"):
        assert getattr(model, field) == getattr(base, field)
    other = cr.milestone_model_config(
        base, cr.milestone_folder("/m", "reader_s1", 100), "a" * 40
    )
    assert other.model_id != model.model_id  # the step is in the generations' keys


def test_a_reader_config_without_non_thinking_is_refused():
    bare = ModelConfig(
        model_id="x/y",
        backend="vllm",
        dtype="float16",
        revision="b" * 40,
        max_tokens=10,
        max_model_len=100,
    )
    with pytest.raises(ValueError):
        cr.milestone_model_config(bare, "/m/step_0100", "a" * 40)


def test_missing_milestone_files_are_listed(tmp_path):
    assert set(cr.missing_files(tmp_path)) == set(cr.REQUIRED_FILES)
    for name in cr.REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"x")
    assert cr.missing_files(tmp_path) == []


def _same(**over):
    base = {
        "base_generation_config": {"top_p": 0.95, "top_k": 20},
        "milestone_generation_config": {"top_p": 0.95, "top_k": 20},
        "base_config": {
            "architectures": ["Qwen3"],
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
        },
        "milestone_config": {
            "architectures": ["Qwen3"],
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "dtype": "float16",
        },
        "base_ids_digest": cr.ids_digest([[1, 2, 3], [4]]),
        "milestone_ids_digest": cr.ids_digest([[1, 2, 3], [4]]),
    }
    base.update(over)
    return base


def test_an_identical_setup_has_no_problems():
    assert cr.identity_problems(**_same()) == []


def test_the_library_version_that_wrote_a_generation_config_is_not_a_setting():
    base = {"top_p": 0.95, "top_k": 20, "transformers_version": "4.51.0"}
    ms = {"top_p": 0.95, "top_k": 20, "transformers_version": "5.20.2"}
    assert (
        cr.identity_problems(
            **_same(base_generation_config=base, milestone_generation_config=ms)
        )
        == []
    )


def test_a_digest_of_token_ids_tells_one_changed_token_apart():
    a = cr.ids_digest([[1, 2, 3], [4]])
    assert a == cr.ids_digest([[1, 2, 3], [4]])
    assert a != cr.ids_digest([[1, 2, 4], [4]])
    assert a != cr.ids_digest([[1, 2, 3, 4]])  # the split between prompts counts


def test_the_milestone_folder_reads_the_base_tokenizer_and_template(tmp_path):
    folder, base = tmp_path / "ms", tmp_path / "base"
    folder.mkdir()
    base.mkdir()
    for name in cr.BASE_TOKENIZER_FILES:
        (folder / name).write_text("milestone")
        (base / name).write_text("base")
    (folder / "chat_template.jinja").write_text("template")
    (folder / "model.safetensors").write_bytes(b"weights")
    cr.use_base_tokenizer(folder, {n: base / n for n in cr.BASE_TOKENIZER_FILES})
    assert all((folder / n).read_text() == "base" for n in cr.BASE_TOKENIZER_FILES)
    assert not (folder / "chat_template.jinja").exists()
    assert (folder / "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize(
    "change, word",
    [
        (
            {"milestone_generation_config": {"top_p": 0.8, "top_k": 20}},
            "generation_config",
        ),
        (
            {"milestone_ids_digest": cr.ids_digest([[1, 2, 4], [4]])},
            "tokenizer encodes",
        ),
        (
            {
                "milestone_config": {
                    "architectures": ["Qwen3"],
                    "hidden_size": 1024,
                    "num_hidden_layers": 28,
                    "vocab_size": 151936,
                    "dtype": "float16",
                }
            },
            "hidden_size",
        ),
        (
            {
                "milestone_config": {
                    "architectures": ["Qwen3"],
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "vocab_size": 151936,
                    "dtype": "bfloat16",
                }
            },
            "float16",
        ),
    ],
)
def test_every_difference_besides_the_weights_is_reported(change, word):
    problems = cr.identity_problems(**_same(**change))
    assert len(problems) == 1 and word in problems[0]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("dev_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("pools")
    write_pool_shard(root, n_problems=N_PROBLEMS, replicates=2, n_pool=6, base_p=0.5)
    pools = load_pools(root)
    problems = [p.problem for p in pools.values()]
    effects = {100: 0.0, 200: 0.4, 300: 0.2}  # the planted gain of each milestone
    out = tmp_path_factory.mktemp("dev")
    dirs = {}
    for step, effect in effects.items():
        cfg = RunConfig(
            model_id=f"mock/step{step}", max_tokens=512, context_limit=100_000
        )
        backend = MockBackend(
            MockWorld.from_problems(problems, 0.5), {f"reader_full_k{K}": effect}
        )
        dirs[step] = out / f"step{step}"
        run_dev_reader(
            dirs[step],
            pools,
            cfg,
            backend=backend,
            arms=["reader_full"],
            k=K,
            n_draws=DRAWS,
            replicates=REPS,
            shard_size=15,
            cap=None,
        )
    return module.summarise, pools, dirs


def test_the_rule_picks_the_planted_best_milestone_from_finished_shards(runs):
    summarise, pools, dirs = runs
    per_step = {
        s: cr.summarise_step([d], pools, summarise, draws=DRAWS, k=K)
        for s, d in dirs.items()
    }
    report = cr.rule_report(
        per_step, N_PROBLEMS, arm_generations=N_PROBLEMS * DRAWS * REPS
    )
    assert report["complete"] and report["incomplete"] == []
    assert report["selected"] == 200
    gains = report["gains"]
    assert gains["200"] > gains["300"] > gains["100"]
    assert report["untrained_reference"] == pytest.approx(-0.0322, abs=1e-4)


def test_an_unfinished_milestone_stops_the_selection(runs):
    summarise, pools, dirs = runs
    per_step = {
        s: cr.summarise_step([d], pools, summarise, draws=DRAWS, k=K)
        for s, d in dirs.items()
        if s != 300
    }
    report = cr.rule_report(
        per_step, N_PROBLEMS, arm_generations=N_PROBLEMS * DRAWS * REPS
    )
    assert not report["complete"] and report["selected"] is None


def test_a_milestone_with_missing_problems_is_reported_not_selected(runs, tmp_path):
    summarise, pools, dirs = runs
    short = tmp_path / "short"
    short.mkdir()
    shards = sorted(dirs[300].glob("shard_*"))
    (short / shards[0].name).symlink_to(shards[0], target_is_directory=True)
    per_step = {
        s: cr.summarise_step([d], pools, summarise, draws=DRAWS, k=K)
        for s, d in dirs.items()
    }
    per_step[300] = cr.summarise_step([short], pools, summarise, draws=DRAWS, k=K)
    report = cr.rule_report(
        per_step, N_PROBLEMS, arm_generations=N_PROBLEMS * DRAWS * REPS
    )
    assert not report["complete"] and report["selected"] is None
    assert any("step 300" in m for m in report["incomplete"])
