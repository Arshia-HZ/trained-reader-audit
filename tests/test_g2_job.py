"""The G2 job end to end on the mock backend (flow, resume, report)."""

from __future__ import annotations

import json

import pytest
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import RunConfig

from reader_audit.g2_job import G2_ARMS, run_g2

EFFECTS = {
    "reader_full_k4": 0.2,
    "candidate_free_k4": 0.0,
    "answer_only_k4": 0.05,
    "placebo_k4": -0.3,
}


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    problems = fake_problems(24)
    cfg = RunConfig(model_id="mock/g2", max_tokens=512, context_limit=100_000)
    backend = MockBackend(MockWorld.from_problems(problems, base_p=0.5), EFFECTS)
    out = tmp_path_factory.mktemp("g2")
    report = run_g2(
        out, problems, cfg, backend=backend, samples=16, k=4, n_draws=3, cap=None
    )
    return {
        "out": out,
        "report": report,
        "problems": problems,
        "cfg": cfg,
        "backend": backend,
    }


def test_the_job_writes_both_reports_and_a_verdict(job):
    out, report = job["out"], job["report"]
    assert (out / "g2_report.json").is_file() and (out / "g2_report.md").is_file()
    on_disk = json.loads((out / "g2_report.json").read_text(encoding="utf-8"))
    assert on_disk["verdict"]["checks"].keys() == report["verdict"]["checks"].keys()
    assert "Gate G2" in (out / "g2_report.md").read_text(encoding="utf-8")
    assert isinstance(report["verdict"]["passed"], bool)


def test_the_counts_follow_the_design(job):
    report = job["report"]
    assert report["n_problems"] == 24 and report["samples_per_problem"] == 16
    assert report["reader_generations"] == 24 * 3 * len(G2_ARMS)
    assert sum(report["strata_counts"].values()) == 24 * 3


def test_a_restart_changes_nothing(job):
    again = run_g2(
        job["out"],
        job["problems"],
        job["cfg"],
        backend=job["backend"],
        samples=16,
        k=4,
        n_draws=3,
        cap=None,
    )
    assert again == job["report"]


def test_too_few_samples_are_refused(tmp_path):
    cfg = RunConfig(model_id="mock/g2", max_tokens=512, context_limit=100_000)
    with pytest.raises(ValueError, match="more than"):
        run_g2(tmp_path, fake_problems(4), cfg, samples=3, k=4)


def test_a_finished_job_never_builds_a_backend(job):
    def boom():
        raise AssertionError("the backend must not be built for a finished job")

    again = run_g2(
        job["out"],
        job["problems"],
        job["cfg"],
        backend_factory=boom,
        samples=16,
        k=4,
        n_draws=3,
        cap=None,
    )
    assert again == job["report"]


def test_an_unfinished_job_builds_one_backend_for_both_runs(tmp_path):
    problems = fake_problems(10)
    cfg = RunConfig(model_id="mock/g2", max_tokens=512, context_limit=100_000)
    built = []

    def factory():
        built.append(1)
        return MockBackend(MockWorld.from_problems(problems, base_p=0.5), EFFECTS)

    run_g2(
        tmp_path,
        problems,
        cfg,
        backend_factory=factory,
        samples=16,
        k=4,
        n_draws=2,
        cap=None,
    )
    assert built == [1]
