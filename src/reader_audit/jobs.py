"""Commands and uploads of the hosted GPU jobs (molab).

A notebook only starts and watches these jobs: each is a separate process that logs to
a file and is resumable, so a lost kernel or an ended session costs only a restart.
Process control is the audit's (``channel_audit.launcher``). Tokens are never an
argument or a file: the upload reads one from the environment or a typed prompt.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from channel_audit import launcher

VLLM_VERSION = launcher.VLLM_VERSION
JOB_ENV = dict(launcher.JOB_ENV)
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_REPO = "Arshia-HZ/trained-reader-audit-raw"  # private dataset of raw generations


def setup_commands(this_repo: str, audit_repo: str, venv: str) -> list[list[str]]:
    """Make an environment for the job and install the pinned stack into it.

    The job gets its own environment so that installing vLLM never replaces what the
    notebook kernel runs on. Both repositories are installed as editable checkouts: the
    audit reads its frozen prompts from its repository.
    """
    py = launcher.venv_python(venv)
    pins = [
        f"vllm=={VLLM_VERSION}",
        "-e",
        str(audit_repo),
        "-e",
        f"{this_repo}[data,hf]",
    ]
    if shutil.which("uv"):
        return [
            ["uv", "venv", "--python", "3.12", venv],
            ["uv", "pip", "install", "--python", py, *pins],
        ]
    return [
        [sys.executable, "-m", "venv", venv],
        [py, "-m", "pip", "install", "-q", *pins],
    ]


@dataclass(frozen=True)
class G2Job:
    """The arguments of ``scripts/10_g2_harness.py``; the token is never one."""

    out: str
    model_config: str
    sets: str
    samples: int = 16
    k: int = 4
    n_draws: int = 8
    cap: int = 4096

    def command(self, python: str | None = None) -> list[str]:
        return [
            python or sys.executable,
            str(REPO_ROOT / "scripts" / "10_g2_harness.py"),
            "--out",
            self.out,
            "--model-config",
            self.model_config,
            "--sets",
            self.sets,
            "--samples",
            str(self.samples),
            "--k",
            str(self.k),
            "--n-draws",
            str(self.n_draws),
            "--cap",
            str(self.cap),
        ]

    def env(self) -> dict[str, str]:
        return dict(JOB_ENV)


def upload_folder(api, folder: str | Path, repo_id: str, path_in_repo: str) -> None:
    """Upload a finished run to the private dataset (created private if missing)."""
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        ignore_patterns=["*.torn", "__pycache__/*"],
    )
