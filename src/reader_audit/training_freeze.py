"""The training freeze: content hashes of everything a GRPO run depends on (Stage 2).

``configs/training_freeze.json`` lists the training code, configs, prompts and the
training set with line-ending-normalised sha256 hashes (the audit's convention for its
prompts and analysis code). ``scripts/20_train_grpo.py`` refuses to start when a listed
file differs, and a test checks the live files, so a silent change fails before it costs
GPU hours. A deliberate change is a dated deviation in PREREG and a new manifest. The
analysis code is frozen separately, before any trained model is evaluated on a
confirmatory set.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE_FILE = "configs/training_freeze.json"

FILES = (
    "prompts/reader_v1/MANIFEST.json",
    "configs/models/qwen3-1.7b-pool.yaml",
    "configs/models/qwen3-1.7b-reader.yaml",
    "data/train/ids.json",
    "data/train/manifest.json",
    "scripts/15_train_pools.py",
    "scripts/20_train_grpo.py",
    "src/reader_audit/arms.py",
    "src/reader_audit/cand_sets.py",
    "src/reader_audit/clipper.py",
    "src/reader_audit/controls.py",
    "src/reader_audit/grpo_plan.py",
    "src/reader_audit/grpo_run.py",
    "src/reader_audit/grpo_upload.py",
    "src/reader_audit/pool_job.py",
    "src/reader_audit/pools.py",
    "src/reader_audit/reader_prompts.py",
    "src/reader_audit/train_data.py",
    "src/reader_audit/train_set.py",
    "src/reader_audit/training_freeze.py",
)


class TrainingFreezeError(RuntimeError):
    """A frozen training file differs from the manifest."""


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build(root: Path = ROOT) -> dict:
    files = {name: file_hash(root / name) for name in FILES}
    listing = "\n".join(f"{name}:{digest}" for name, digest in files.items())
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(listing.encode()).hexdigest(),
    }


def write(root: Path = ROOT) -> Path:
    path = root / FREEZE_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists: the freeze is never rewritten")
    path.write_text(json.dumps(build(root), indent=1) + chr(10), encoding="utf-8")
    return path


def verify(root: Path = ROOT) -> dict:
    """The manifest, after checking every listed file against it."""
    manifest = json.loads((root / FREEZE_FILE).read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in manifest["files"].items()
        if file_hash(root / name) != digest
    ]
    if changed:
        raise TrainingFreezeError("changed since the freeze: " + ", ".join(changed))
    if set(manifest["files"]) != set(FILES):
        raise TrainingFreezeError("the list of frozen files changed")
    return manifest


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_report() -> dict:
    """What a run records about itself: code versions, library versions, the GPU."""
    import channel_audit

    report: dict = {
        "python": sys.version.split()[0],
        "commit": _git(["rev-parse", "HEAD"], ROOT),
        "dirty": bool(_git(["status", "--porcelain"], ROOT)),
        "audit_commit": _git(
            ["rev-parse", "HEAD"], Path(channel_audit.__file__).resolve().parents[2]
        ),
    }
    for name in ("torch", "transformers", "trl", "vllm", "accelerate"):
        try:
            report[name] = __import__(name).__version__
        except Exception:  # a library that is not installed is simply absent
            report[name] = None
    try:
        import torch

        report["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        report["gpu"] = None
    return report


if __name__ == "__main__":
    # python -m reader_audit.training_freeze  (write the manifest once, then verify it)
    print(f"wrote {write()}")
    print(f"combined sha256: {verify()['combined_sha256']}")
