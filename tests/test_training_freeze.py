"""The frozen prompts and training files still match their manifests."""

from __future__ import annotations

import json

import pytest

from reader_audit import reader_prompts as rp
from reader_audit import training_freeze as tf


def test_the_reader_prompts_match_their_manifest():
    from channel_audit.prompt_freeze import verify_manifest

    manifest = verify_manifest(rp.PROMPTS_DIR)
    assert manifest["version"] == "reader_v1" == rp.PROMPT_VERSION
    assert set(manifest["files"]) == {
        "reader.txt",
        "reader_answers.txt",
        "reader_no_candidates.txt",
    }


def test_the_training_files_match_the_freeze():
    manifest = tf.verify()
    assert list(manifest["files"]) == list(tf.FILES)


def test_a_changed_file_is_named_and_the_freeze_is_never_rewritten(tmp_path):
    for name in tf.FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"content of {name}\n")
    (tmp_path / "configs").mkdir(exist_ok=True)
    path = tf.write(tmp_path)
    assert (
        json.loads(path.read_text())["combined_sha256"]
        == tf.build(tmp_path)["combined_sha256"]
    )
    with pytest.raises(FileExistsError):
        tf.write(tmp_path)
    assert tf.verify(tmp_path)
    (tmp_path / "scripts/20_train_grpo.py").write_text("edited\n")
    with pytest.raises(tf.TrainingFreezeError, match="20_train_grpo"):
        tf.verify(tmp_path)


def test_line_endings_do_not_change_a_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x\ny\n")
    b.write_bytes(b"x\r\ny\r\n")
    assert tf.file_hash(a) == tf.file_hash(b)
