"""Copies of training checkpoints on the private model repo, one run per folder.

``latest`` holds the whole checkpoint (weights, optimizer, scheduler, RNG, trainer
state) and is overwritten each time, so a run can resume on another machine
(``fetch_latest``); it is removed once the run is complete. ``step_NNNN`` holds the
weights only, cast once from the fp32 master weights to fp16, which is what the
evaluation runs (PREREG D12, R17): 3.4 GB instead of 7 GB, and no further rounding. The
step logs go to ``logs`` with every checkpoint. A failed upload is reported and never
stops training; the checkpoint stays on disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
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
FP16_MAX = 65504.0


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
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.run_name = run_name
        self.milestones = set(milestones)
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

    def _milestone(self, step: int, folder: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="fp16_") as tmp:
            report = fp16_copy(folder, Path(tmp))
            self.last_cast = report
            if report["nonfinite"]:
                raise ValueError(f"{report['nonfinite']} weights not finite in fp16")
            self._put(Path(tmp), f"{self.run_name}/step_{step:04d}")
        self.on_hub.add(step)

    def on_checkpoint(
        self, step: int, folder: str | Path, logs_dir: str | Path | None = None
    ) -> list[str]:
        """Upload what a checkpoint at ``step`` calls for; return what went up."""
        folder = Path(folder)
        done: list[str] = []
        jobs: list[tuple[str, Callable[[], None]]] = [
            (
                f"{self.run_name}/latest",
                lambda: self._put(
                    folder, f"{self.run_name}/latest", delete_patterns=["*"]
                ),
            )
        ]
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
        if step in self.milestones:
            jobs.append(
                (
                    f"{self.run_name}/step_{step:04d}",
                    lambda: self._milestone(step, folder),
                )
            )
        for path, job in jobs:
            try:
                job()
                done.append(path)
            except Exception as exc:  # the checkpoint stays on disk
                print(
                    f"upload of {path} failed: {type(exc).__name__}: {exc}", flush=True
                )
        return done

    def finish(self, step: int) -> bool:
        """Remove ``latest`` when the run is complete and its last milestone is up."""
        if not self.milestones or step < max(self.milestones):
            return False
        if max(self.milestones) not in self.on_hub:
            print("final milestone not on the hub: keeping latest", flush=True)
            return False
        try:
            self.api.delete_folder(
                f"{self.run_name}/latest", repo_id=self.repo_id, repo_type="model"
            )
        except Exception as exc:
            print(f"removal of latest failed: {type(exc).__name__}", flush=True)
            return False
        return True


def fetch_latest(
    repo_id: str,
    run_name: str,
    out_dir: str | Path,
    *,
    download: Callable[..., str] | None = None,
    token: str | None = None,
) -> int | None:
    """Bring ``latest`` and the logs of a run back from the hub to resume it here.

    The checkpoint lands in ``out_dir/checkpoint-<step>`` (the name the trainer looks
    for) and the logs beside it, unless they exist. Returns the step, or None if the
    run has nothing on the hub or already has a checkpoint here.
    """
    out = Path(out_dir)
    if list(out.glob("checkpoint-*")):
        return None
    if download is None:
        from huggingface_hub import snapshot_download as download
    with tempfile.TemporaryDirectory(prefix="fetch_") as tmp:
        try:
            download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"{run_name}/latest/*", f"{run_name}/logs/*"],
                local_dir=tmp,
                token=token,
            )
        except Exception as exc:
            print(f"nothing fetched for {run_name}: {type(exc).__name__}", flush=True)
            return None
        latest = Path(tmp) / run_name / "latest"
        state = latest / "trainer_state.json"
        if not state.is_file():
            return None
        step = int(json.loads(state.read_text(encoding="utf-8"))["global_step"])
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(latest), str(out / f"checkpoint-{step}"))
        logs = Path(tmp) / run_name / "logs"
        if logs.is_dir():
            for path in logs.glob("*.jsonl"):
                if not (out / path.name).exists():
                    shutil.move(str(path), str(out / path.name))
    return step
