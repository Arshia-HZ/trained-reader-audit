"""Copies of training checkpoints on the private model repo, one run per folder.

``latest`` holds the whole checkpoint (weights, optimizer, scheduler, RNG, trainer
state) and is overwritten each time, so a run can resume on another machine.
``step_NNNN`` holds the weights only, kept for the steps that are evaluated. A failed
upload is reported and never stops training; the checkpoint stays on disk.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

MODEL_FILES = [
    "*.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template*",
]


class CheckpointUploader:
    def __init__(
        self,
        api,
        repo_id: str,
        run_name: str,
        milestones: Collection[int] = (),
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.run_name = run_name
        self.milestones = set(milestones)
        self._made = False

    def _repo(self) -> None:
        if not self._made:
            self.api.create_repo(
                self.repo_id, repo_type="model", private=True, exist_ok=True
            )
            self._made = True

    def _put(self, folder: Path, path_in_repo: str, **kw) -> None:
        self._repo()
        self.api.upload_folder(
            folder_path=str(folder),
            repo_id=self.repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            **kw,
        )

    def on_checkpoint(self, step: int, folder: str | Path) -> list[str]:
        """Upload what a checkpoint at ``step`` calls for; return what went up."""
        folder = Path(folder)
        done: list[str] = []
        jobs = [(f"{self.run_name}/latest", {"delete_patterns": ["*"]})]
        if step in self.milestones:
            jobs.append(
                (f"{self.run_name}/step_{step:04d}", {"allow_patterns": MODEL_FILES})
            )
        for path, kw in jobs:
            try:
                self._put(folder, path, **kw)
                done.append(path)
            except Exception as exc:  # the checkpoint stays on disk
                print(f"upload of {path} failed: {type(exc).__name__}", flush=True)
        return done
