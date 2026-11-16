"""Copies of training checkpoints on private model repos.

Resumable state (weights, optimizer, scheduler, RNG, trainer state: about 20 GB) goes to
one of two repos in turn, ``<repo>-state-<run>-a`` and ``-b``. Hugging Face keeps the
files of a repo's history in its storage count even after they are deleted, so a repo is
deleted and created again before each upload instead of being overwritten: at most two
states exist (about 40 GB), a failed upload never touches the newest complete one, and
``fetch_latest`` resumes from whichever repo holds the higher step. Both are removed
when the run completes.

The main repo ``<repo>`` keeps, per run, the weights of the chosen steps in
``step_NNNN`` (cast once from the fp32 master weights to fp16, which is what the
evaluation runs, PREREG D12 and R17: 3.4 GB instead of 7 GB, with no further rounding)
and the step logs in ``logs``. A failed upload is retried, then reported, and never
stops training; the checkpoint stays on disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable, Collection
from pathlib import Path

WEIGHT_SUFFIX = ".safetensors"
KEEP_BESIDE_WEIGHTS = (
    "config.json",
    "generation_config.json",
    "tokenizer",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template",
    "model.safetensors.index.json",
)


def state_repos(repo_id: str, run_name: str) -> tuple[str, str]:
    return (f"{repo_id}-state-{run_name}-a", f"{repo_id}-state-{run_name}-b")


def fp16_copy(src: Path, dst: Path) -> dict:
    """Write the weights of checkpoint ``src`` as fp16 into ``dst`` (with small files).

    Returns the largest absolute weight and the number of values that are not finite
    after the cast; the caller refuses to upload a copy with any.
    """
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    largest, bad, tensors = 0.0, 0, 0
    for path in sorted(src.glob(f"*{WEIGHT_SUFFIX}")):
        weights = load_file(str(path))
        out = {}
        for name, value in weights.items():
            if value.is_floating_point():
                largest = max(largest, float(value.abs().max()))
                value = value.to(torch.float16)
                bad += int((~torch.isfinite(value)).sum())
            out[name] = value.contiguous()
            tensors += 1
        save_file(out, str(dst / path.name), metadata={"format": "pt"})
    for path in src.iterdir():
        if path.is_file() and path.name.startswith(KEEP_BESIDE_WEIGHTS):
            if path.name == "config.json":
                config = json.loads(path.read_text(encoding="utf-8"))
                for key in ("dtype", "torch_dtype"):
                    if key in config:
                        config[key] = "float16"
                (dst / path.name).write_text(json.dumps(config, indent=2), "utf-8")
            else:
                shutil.copy(path, dst / path.name)
    return {"largest_weight": largest, "nonfinite": bad, "tensors": tensors}


class CheckpointUploader:
    def __init__(
        self,
        api,
        repo_id: str,
        run_name: str,
        milestones: Collection[int] = (),
        *,
        save_every: int = 25,
        attempts: int = 3,
        wait: float = 15.0,
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.run_name = run_name
        self.milestones = set(milestones)
        self.save_every = save_every
        self.attempts = attempts
        self.wait = wait
        self.states = state_repos(repo_id, run_name)
        self.on_hub: set[int] = set()  # milestones that reached the repo
        self.last_cast: dict | None = None
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

    def _put_state(self, step: int, folder: Path) -> None:
        target = self.states[(step // self.save_every) % 2]
        self.api.delete_repo(target, repo_type="model", missing_ok=True)
        self.api.create_repo(target, repo_type="model", private=True, exist_ok=True)
        self.api.upload_folder(
            folder_path=str(folder),
            repo_id=target,
            repo_type="model",
            path_in_repo=f"{self.run_name}/latest",
        )

    def _milestone(self, step: int, folder: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="fp16_") as tmp:
            report = fp16_copy(folder, Path(tmp))
            self.last_cast = report
            if report["nonfinite"]:
                raise ValueError(f"{report['nonfinite']} weights not finite in fp16")
            self._put(Path(tmp), f"{self.run_name}/step_{step:04d}")
        self.on_hub.add(step)

    def _attempt(self, name: str, job: Callable[[], None]) -> bool:
        for attempt in range(1, self.attempts + 1):
            try:
                job()
                return True
            except Exception as exc:  # the checkpoint stays on disk
                print(
                    f"upload of {name} failed ({attempt}/{self.attempts}): "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt < self.attempts:
                    time.sleep(self.wait * attempt)
        return False

    def on_checkpoint(
        self, step: int, folder: str | Path, logs_dir: str | Path | None = None
    ) -> list[str]:
        """Upload what a checkpoint at ``step`` calls for; return what went up."""
        folder = Path(folder)
        jobs: list[tuple[str, Callable[[], None]]] = []
        if logs_dir is not None:
            jobs.append(
                (
                    f"{self.run_name}/logs",
                    lambda: self._put(
                        Path(logs_dir),
                        f"{self.run_name}/logs",
                        allow_patterns=["*.jsonl"],
                    ),
                )
            )
        jobs.append((f"{self.run_name}/latest", lambda: self._put_state(step, folder)))
        if step in self.milestones:
            jobs.append(
                (
                    f"{self.run_name}/step_{step:04d}",
                    lambda: self._milestone(step, folder),
                )
            )
        return [name for name, job in jobs if self._attempt(name, job)]

    def finish(self, step: int) -> bool:
        """Remove the resumable state once the run is complete and its last milestone
        is on the hub."""
        if not self.milestones or step < max(self.milestones):
            return False
        if max(self.milestones) not in self.on_hub:
            print("final milestone not on the hub: keeping the state", flush=True)
            return False
        ok = True
        for repo in self.states:
            try:
                self.api.delete_repo(repo, repo_type="model", missing_ok=True)
            except Exception as exc:
                print(f"removal of {repo} failed: {type(exc).__name__}", flush=True)
                ok = False
        return ok


def fetch_latest(
    repo_id: str,
    run_name: str,
    out_dir: str | Path,
    *,
    download: Callable[..., str] | None = None,
    token: str | None = None,
) -> int | None:
    """Bring the newest resumable state and the logs of a run back from the hub.

    The checkpoint lands in ``out_dir/checkpoint-<step>`` (the name the trainer looks
    for) and the logs beside it, unless they exist. Returns the step, or None if the
    run has nothing on the hub or already has a checkpoint here.
    """
    out = Path(out_dir)
    if list(out.glob("checkpoint-*")):
        return None
    if download is None:
        from huggingface_hub import snapshot_download as download

    def get(repo: str, patterns: list[str], where: str) -> bool:
        try:
            download(
                repo_id=repo,
                repo_type="model",
                allow_patterns=patterns,
                local_dir=where,
                token=token,
            )
            return True
        except Exception as exc:
            print(f"nothing fetched from {repo}: {type(exc).__name__}", flush=True)
            return False

    best: tuple[int, str] | None = None
    with tempfile.TemporaryDirectory(prefix="peek_") as peek:
        for repo in state_repos(repo_id, run_name):
            where = str(Path(peek) / repo.replace("/", "__"))
            if get(repo, [f"{run_name}/latest/trainer_state.json"], where):
                state = Path(where) / run_name / "latest" / "trainer_state.json"
                if state.is_file():
                    step = int(json.loads(state.read_text("utf-8"))["global_step"])
                    if best is None or step > best[0]:
                        best = (step, repo)
    if best is None:
        return None
    step, repo = best
    with tempfile.TemporaryDirectory(prefix="fetch_") as tmp:
        if not get(repo, [f"{run_name}/latest/*"], tmp):
            return None
        out.mkdir(parents=True, exist_ok=True)
        latest = Path(tmp) / run_name / "latest"
        shutil.move(str(latest), str(out / f"checkpoint-{step}"))
    with tempfile.TemporaryDirectory(prefix="logs_") as tmp:
        if get(repo_id, [f"{run_name}/logs/*"], tmp):
            for path in (Path(tmp) / run_name / "logs").glob("*.jsonl"):
                if not (out / path.name).exists():
                    shutil.move(str(path), str(out / path.name))
    return step
