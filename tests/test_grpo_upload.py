"""Checkpoint uploads: what goes where, and that a failure never propagates."""

from __future__ import annotations

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
