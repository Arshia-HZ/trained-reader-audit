"""The study imports the closed audit as it was frozen: same prompt, same tag."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from channel_audit import channels

import reader_audit


def test_round0_prompt_is_the_frozen_one():
    assert channels.template_hash("round0") == reader_audit.ROUND0_PROMPT_SHA256


def test_prompts_directory_is_the_audits_checkout():
    assert channels.PROMPTS_DIR.is_dir()
    assert (channels.PROMPTS_DIR / "v1" / "MANIFEST.json").is_file()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def test_audit_checkout_is_at_the_tag():
    root = channels.PROMPTS_DIR.parent
    head = _git(root, "rev-parse", "HEAD")
    tag = _git(root, "rev-parse", "audit-final^{commit}")
    if head is None or tag is None:
        pytest.skip("the audit is not a git checkout with the audit-final tag")
    assert head == tag, "the audit checkout moved off the tag audit-final"
