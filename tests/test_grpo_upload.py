"""Checkpoint uploads: alternating state repos, fp16 milestones, resume, failures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reader_audit.grpo_upload import CheckpointUploader, fetch_latest, state_repos


class FakeApi:
    def __init__(self, fail_first: int = 0) -> None:
        self.events: list[tuple] = []
        self.calls: list[dict] = []
        self.fail_first = fail_first

    def create_repo(self, repo_id, **kw):
        self.events.append(("create", repo_id))

    def delete_repo(self, repo_id, **kw):
        self.events.append(("delete", repo_id))

    def upload_folder(self, **kw):
        if self.fail_first > 0:
            self.fail_first -= 1
            raise OSError("network down")
        # the folder may be temporary: note what it holds at the time of the call
        kw["files"] = sorted(os.listdir(kw["folder_path"]))
        self.calls.append(kw)
        self.events.append(("upload", kw["repo_id"], kw["path_in_repo"]))


def uploader(api, **kw):
    return CheckpointUploader(api, "me/ckpt", "run1", wait=0.0, **kw)


def test_state_goes_to_the_two_repos_in_turn_and_logs_to_the_main_repo(tmp_path):
    api = FakeApi()
    up = uploader(api, milestones={100})
    a, b = state_repos("me/ckpt", "run1")
    assert up.on_checkpoint(25, tmp_path, logs_dir=tmp_path) == [
        "run1/logs",
        "run1/latest",
    ]
    assert up.on_checkpoint(50, tmp_path, logs_dir=tmp_path)
    uploads = [e for e in api.events if e[0] == "upload"]
    assert uploads == [
        ("upload", "me/ckpt", "run1/logs"),
        ("upload", b, "run1/latest"),
        ("upload", "me/ckpt", "run1/logs"),
        ("upload", a, "run1/latest"),
    ]
    # a state repo is emptied (deleted and created) before its upload, never after
    seq = [e for e in api.events if e[1] == a]
    assert [e[0] for e in seq] == ["delete", "create", "upload"]
    assert api.calls[0]["allow_patterns"] == ["*.jsonl"]


def test_a_transient_failure_is_retried_and_a_lasting_one_is_reported(tmp_path, capsys):
    api = FakeApi(fail_first=2)
    assert uploader(api).on_checkpoint(25, tmp_path) == ["run1/latest"]
    assert "failed (1/3)" in capsys.readouterr().out
    broken = FakeApi(fail_first=99)
    assert uploader(broken).on_checkpoint(25, tmp_path) == []
    assert "failed (3/3)" in capsys.readouterr().out


def checkpoint_with_weights(folder, scale=1.0):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    save = pytest.importorskip("safetensors.numpy").save_file
    folder.mkdir()
    weights = {
        "w": (np.arange(6, dtype=np.float32) * 0.1 * scale).reshape(2, 3),
        "norm": np.ones(4, dtype=np.float32),
    }
    save(weights, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(json.dumps({"dtype": "float32", "n": 1}))
    (folder / "tokenizer.json").write_text("{}")
    (folder / "optimizer.pt").write_bytes(b"x" * 100)
    return folder


def test_a_milestone_is_fp16_weights_without_the_training_state(tmp_path):
    folder = checkpoint_with_weights(tmp_path / "checkpoint-100")
    api = FakeApi()
    up = uploader(api, milestones={100})
    done = up.on_checkpoint(100, folder)
    assert done == ["run1/latest", "run1/step_0100"]
    milestone = [c for c in api.calls if c["path_in_repo"] == "run1/step_0100"][0]
    assert milestone["repo_id"] == "me/ckpt"
    assert milestone["files"] == ["config.json", "model.safetensors", "tokenizer.json"]
    assert up.on_hub == {100} and up.last_cast["nonfinite"] == 0


def test_the_fp16_copy_holds_fp16_values_and_says_so_in_its_config(tmp_path):
    from reader_audit.grpo_upload import fp16_copy

    safe = pytest.importorskip("safetensors.numpy")
    src = checkpoint_with_weights(tmp_path / "checkpoint-100")
    report = fp16_copy(src, tmp_path / "out")
    loaded = safe.load_file(str(tmp_path / "out" / "model.safetensors"))
    assert all(v.dtype.name == "float16" for v in loaded.values())
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["dtype"] == "float16"
    assert report["tensors"] == 2 and report["nonfinite"] == 0
    assert not (tmp_path / "out" / "optimizer.pt").exists()


def test_weights_that_overflow_fp16_are_not_uploaded(tmp_path, capsys):
    folder = checkpoint_with_weights(tmp_path / "checkpoint-100", scale=1e6)
    api = FakeApi()
    up = uploader(api, milestones={100})
    done = up.on_checkpoint(100, folder)
    assert done == ["run1/latest"] and up.on_hub == set()
    assert "not finite" in capsys.readouterr().out
    deleted = [e for e in api.events if e[0] == "delete"]
    assert up.finish(100) is False
    assert [e for e in api.events if e[0] == "delete"] == deleted  # states kept


def test_the_states_go_only_when_the_run_is_complete_and_the_last_milestone_is_up():
    api = FakeApi()
    up = uploader(api, milestones={100, 200})
    up.on_hub = {100}
    assert up.finish(150) is False and up.finish(200) is False
    assert not [e for e in api.events if e[0] == "delete"]
    up.on_hub = {100, 200}
    assert up.finish(150) is False
    assert up.finish(200) is True
    assert sorted(e[1] for e in api.events if e[0] == "delete") == sorted(
        state_repos("me/ckpt", "run1")
    )


def fake_hub(steps: dict[str, int | None]):
    """A download function over repos holding a state at the given step (or nothing)."""

    def download(repo_id, repo_type, allow_patterns, local_dir, token):
        root = Path(local_dir)
        if repo_id == "me/ckpt":  # the main repo: logs
            logs = root / "run1" / "logs"
            logs.mkdir(parents=True)
            (logs / "steps.jsonl").write_text('{"step": 75}\n')
            return local_dir
        step = steps.get(repo_id)
        if step is None:
            raise OSError("no such repo")
        latest = root / "run1" / "latest"
        latest.mkdir(parents=True)
        (latest / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        if allow_patterns == ["run1/latest/*"]:
            (latest / "model.safetensors").write_bytes(f"w{step}".encode())
        return local_dir

    return download


def test_fetch_latest_takes_the_newer_of_the_two_states(tmp_path):
    a, b = state_repos("me/ckpt", "run1")
    out = tmp_path / "out"
    assert fetch_latest("me/ckpt", "run1", out, download=fake_hub({a: 50, b: 75})) == 75
    assert (out / "checkpoint-75" / "model.safetensors").read_bytes() == b"w75"
    assert (out / "steps.jsonl").read_text() == '{"step": 75}\n'
    # a machine that already has a checkpoint is left alone
    assert fetch_latest("me/ckpt", "run1", out, download=fake_hub({a: 99})) is None


def test_fetch_latest_works_with_one_state_and_with_none(tmp_path):
    a, _ = state_repos("me/ckpt", "run1")
    got = fetch_latest("me/ckpt", "run1", tmp_path / "o1", download=fake_hub({a: 25}))
    assert got == 25
    nothing = fetch_latest("me/ckpt", "run1", tmp_path / "o2", download=fake_hub({}))
    assert nothing is None
