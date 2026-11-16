"""The R17 health triggers fire on what they should and are quiet on a healthy run."""

from __future__ import annotations

import json

from reader_audit.grpo_health import check_run, triggers


def step(i, **kw):
    base = {
        "step": i,
        "loss": 0.01,
        "grad_norm": 0.1,
        "reward": 0.5,
        "mean_length": 700.0,
        "clipped_ratio": 0.02,
        "seconds": 240.0,
    }
    return {**base, **kw}


def healthy(n=120):
    return [step(i) for i in range(1, n + 1)]


def test_a_healthy_run_has_no_trigger():
    saves = [{"step": 25, "uploaded": ["r/logs", "r/latest"]}]
    assert (
        triggers(healthy(), saves, [{"rows_sha256": "a"}, {"rows_sha256": "a"}]) == []
    )


def test_non_finite_loss_or_gradient_fires():
    bad = healthy()
    bad[10]["loss"] = float("nan")
    assert "not finite" in triggers(bad)[0]
    bad = healthy()
    bad[5]["grad_norm"] = None
    assert "not finite" in triggers(bad)[0]


def test_three_gradient_spikes_in_a_row_fire_but_two_do_not():
    spikes = healthy()
    for i in (40, 41, 42):
        spikes[i]["grad_norm"] = 5.0
    assert any("gradient norm above" in t for t in triggers(spikes))
    two = healthy()
    for i in (40, 41):
        two[i]["grad_norm"] = 5.0
    assert not any("gradient norm above" in t for t in triggers(two))


def test_a_reward_collapse_after_step_50_fires():
    run = healthy(140)
    for r in run[60:]:
        r["reward"] = 0.1
    assert any("reward fell" in t for t in triggers(run))
    assert not any("reward fell" in t for t in triggers(healthy(140)))


def test_length_cap_and_speed_windows():
    short = [step(i, mean_length=50.0) for i in range(1, 40)]
    assert any("completion length" in t for t in triggers(short))
    clipped = [step(i, clipped_ratio=0.4) for i in range(1, 40)]
    assert any("at the cap" in t for t in triggers(clipped))
    slow = [step(i, seconds=320.0) for i in range(1, 40)]
    assert any("step time" in t for t in triggers(slow))


def test_a_step_logged_twice_counts_once_by_its_last_record():
    run = healthy(30) + [step(10, seconds=999.0)]
    assert not any("step time" in t for t in triggers(run))  # 1 of 25 only


def test_missing_uploads_and_a_changed_row_hash_fire():
    saves = [
        {"step": 25, "uploaded": ["r/logs"]},
        {"step": 100, "uploaded": ["r/logs", "r/latest"]},
    ]
    found = triggers(healthy(30), saves, [{"rows_sha256": "a"}, {"rows_sha256": "b"}])
    assert any("step 25: not on the hub" in t for t in found)
    assert any("step 100" in t and "step_0100" in t for t in found)
    assert "restart with a different row sequence" in found


def test_check_run_reads_the_files_of_a_run_directory(tmp_path):
    (tmp_path / "steps.jsonl").write_text(
        chr(10).join(json.dumps(r) for r in healthy(30)) + chr(10)
    )
    assert check_run(tmp_path) == []
    assert check_run(tmp_path / "missing") == []
