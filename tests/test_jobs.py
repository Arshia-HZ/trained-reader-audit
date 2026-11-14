"""Job commands: pinned stack, no token in any command, uploads go to a private repo."""

from __future__ import annotations

import ast
from pathlib import Path

from reader_audit import jobs

ROOT = Path(__file__).resolve().parents[1]


def test_setup_installs_the_pinned_stack_with_both_repositories_editable():
    commands = jobs.setup_commands("/r/reader", "/r/audit", "/venv")
    flat = [part for cmd in commands for part in cmd]
    assert f"vllm=={jobs.VLLM_VERSION}" in flat and jobs.VLLM_VERSION == "0.33.0"
    assert flat.count("-e") == 2 and "/r/audit" in flat and "/r/reader[data,hf]" in flat


def test_the_g2_command_carries_the_settings_and_no_token():
    job = jobs.G2Job(out="/runs/g2", model_config="cfg.yaml", sets="data/sets")
    cmd = job.command(python="/venv/bin/python")
    assert cmd[:2] == ["/venv/bin/python", str(ROOT / "scripts" / "10_g2_harness.py")]
    assert cmd[cmd.index("--cap") + 1] == "4096" and cmd[cmd.index("--k") + 1] == "4"
    assert not any("token" in part.lower() or part.startswith("hf_") for part in cmd)
    assert job.env() == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def test_upload_creates_the_dataset_private_and_skips_torn_files(tmp_path):
    api = FakeApi()
    jobs.upload_folder(api, tmp_path, "me/raw", "g2/run")
    (_, repo, kw), (_, up) = api.calls
    assert repo == "me/raw" and kw["private"] is True and kw["repo_type"] == "dataset"
    assert up["path_in_repo"] == "g2/run" and "*.torn" in up["ignore_patterns"]


def test_the_molab_notebook_parses_and_holds_no_secret():
    path = ROOT / "notebooks" / "stage1_molab.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    assert "hf_" not in lowered.replace("hf_repo", "").replace("hf_token", "")
    assert 'kind="password"' in source  # a token is typed, never stored
