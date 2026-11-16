"""Checkpoint uploads: what goes where, and that a failure never propagates."""

from __future__ import annotations

import pytest

from reader_audit.grpo_upload import CheckpointUploader


class FakeApi:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.created: list[str] = []
        self.fail = fail

    def create_repo(self, repo_id, **kw):
        self.created.append(repo_id)

    def upload_folder(self, **kw):
        if self.fail:
            raise OSError("network down")
        self.calls.append(kw)


def test_every_checkpoint_goes_to_latest_and_milestones_also_keep_weights(tmp_path):
    api = FakeApi()
    up = CheckpointUploader(api, "me/ckpt", "run1", milestones={100})
    assert up.on_checkpoint(25, tmp_path) == ["run1/latest"]
    assert up.on_checkpoint(100, tmp_path) == ["run1/latest", "run1/step_0100"]
    paths = [c["path_in_repo"] for c in api.calls]
    assert paths == ["run1/latest", "run1/latest", "run1/step_0100"]
    assert api.calls[0]["delete_patterns"] == ["*"]
    assert "*.safetensors" in api.calls[2]["allow_patterns"]
    assert "optimizer.pt" not in " ".join(api.calls[2]["allow_patterns"])
    assert api.created == ["me/ckpt"] and all(
        c["repo_type"] == "model" for c in api.calls
    )


def test_a_failed_upload_is_reported_not_raised(tmp_path, capsys):
    up = CheckpointUploader(FakeApi(fail=True), "me/ckpt", "run1", milestones={25})
    assert up.on_checkpoint(25, tmp_path) == []
    assert "failed" in capsys.readouterr().out


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
