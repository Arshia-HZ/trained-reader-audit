"""Checkpoint uploads: what goes where, fp16 milestones, resume, and failures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reader_audit.grpo_upload import CheckpointUploader, fetch_latest


class FakeApi:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.fail = fail

    def create_repo(self, repo_id, **kw):
        self.created.append(repo_id)

    def upload_folder(self, **kw):
        if self.fail:
            raise OSError("network down")
        # the folder may be temporary: note what it holds at the time of the call
        kw["files"] = sorted(os.listdir(kw["folder_path"]))
        self.calls.append(kw)

    def delete_folder(self, path_in_repo, **kw):
        self.deleted.append(path_in_repo)


def test_every_checkpoint_goes_to_latest_with_the_logs(tmp_path):
    api = FakeApi()
    up = CheckpointUploader(api, "me/ckpt", "run1", milestones={100})
    done = up.on_checkpoint(25, tmp_path, logs_dir=tmp_path)
    assert done == ["run1/latest", "run1/logs"]
    assert [c["path_in_repo"] for c in api.calls] == ["run1/latest", "run1/logs"]
    assert api.calls[0]["delete_patterns"] == ["*"]
    assert api.calls[1]["allow_patterns"] == ["*.jsonl"]
    assert api.created == ["me/ckpt"] and all(
        c["repo_type"] == "model" for c in api.calls
    )


def test_a_failed_upload_is_reported_not_raised(tmp_path, capsys):
    up = CheckpointUploader(FakeApi(fail=True), "me/ckpt", "run1", milestones={25})
    assert up.on_checkpoint(25, tmp_path) == []
    assert "failed" in capsys.readouterr().out


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
    up = CheckpointUploader(api, "me/ckpt", "run1", milestones={100})
    done = up.on_checkpoint(100, folder)
    assert done == ["run1/latest", "run1/step_0100"]
    milestone = api.calls[1]
    assert milestone["files"] == ["config.json", "model.safetensors", "tokenizer.json"]
    assert up.on_hub == {100} and up.last_cast["nonfinite"] == 0


def test_the_fp16_copy_holds_fp16_values_and_says_so_in_its_config(tmp_path):
    from reader_audit.grpo_upload import fp16_copy

    safe = pytest.importorskip("safetensors.numpy")
    src = checkpoint_with_weights(tmp_path / "checkpoint-100")
    report = fp16_copy(src, tmp_path / "out")
    loaded = safe.load_file(str(tmp_path / "out" / "model.safetensors"))
    assert all(v.dtype.name == "float16" for v in loaded.values())
    assert json.loads((tmp_path / "out" / "config.json").read_text())["dtype"] == (
        "float16"
    )
    assert report["tensors"] == 2 and report["nonfinite"] == 0
    assert not (tmp_path / "out" / "optimizer.pt").exists()


def test_weights_that_overflow_fp16_are_not_uploaded(tmp_path, capsys):
    folder = checkpoint_with_weights(tmp_path / "checkpoint-100", scale=1e6)
    api = FakeApi()
    up = CheckpointUploader(api, "me/ckpt", "run1", milestones={100})
    done = up.on_checkpoint(100, folder)
    assert done == ["run1/latest"] and up.on_hub == set()
    assert "not finite" in capsys.readouterr().out
    assert up.finish(100) is False and api.deleted == []


def test_latest_is_removed_only_when_the_run_is_complete_and_the_last_milestone_is_up():
    api = FakeApi()
    up = CheckpointUploader(api, "me/ckpt", "run1", milestones={100, 200})
    up.on_hub = {100}
    assert up.finish(150) is False and up.finish(200) is False and api.deleted == []
    up.on_hub = {100, 200}
    assert up.finish(150) is False
    assert up.finish(200) is True and api.deleted == ["run1/latest"]


def test_fetch_latest_brings_the_checkpoint_and_logs_to_where_the_trainer_looks(
    tmp_path,
):
    def download(repo_id, repo_type, allow_patterns, local_dir, token):
        latest = Path(local_dir) / "run1" / "latest"
        latest.mkdir(parents=True)
        (latest / "trainer_state.json").write_text(json.dumps({"global_step": 75}))
        (latest / "model.safetensors").write_bytes(b"w")
        logs = Path(local_dir) / "run1" / "logs"
        logs.mkdir()
        (logs / "steps.jsonl").write_text('{"step": 75}\n')
        return local_dir

    out = tmp_path / "out"
    assert fetch_latest("me/ckpt", "run1", out, download=download) == 75
    assert (out / "checkpoint-75" / "model.safetensors").is_file()
    assert (out / "steps.jsonl").read_text() == '{"step": 75}\n'
    # a machine that already has a checkpoint is left alone
    assert fetch_latest("me/ckpt", "run1", out, download=download) is None


def test_fetch_latest_with_nothing_on_the_hub_returns_none(tmp_path):
    def download(**kw):
        raise OSError("not found")

    assert fetch_latest("me/ckpt", "run1", tmp_path / "o", download=download) is None


def test_deleting_latest_never_touches_the_milestone_weights():
    # the real huggingface_hub matching: delete_patterns are relative to path_in_repo
    hub = pytest.importorskip("huggingface_hub")
    api = hub.HfApi()
    files = [
        "run1/latest/model.safetensors",
        "run1/latest/sub/optimizer.pt",
        "run1/step_0100/model.safetensors",
        "run1/latest_old/model.safetensors",
        "run2/latest/model.safetensors",
        ".gitattributes",
    ]
    api.list_repo_files = lambda **kw: files
    gone = api._prepare_folder_deletions(
        repo_id="me/ckpt",
        repo_type="model",
        revision=None,
        path_in_repo="run1/latest",
        delete_patterns=["*"],
    )
    assert sorted(op.path_in_repo for op in gone) == [
        "run1/latest/model.safetensors",
        "run1/latest/sub/optimizer.pt",
    ]
