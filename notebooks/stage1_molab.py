import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reader study on molab: Stage 1, harness check (gate G2)

    This notebook only **starts and watches** `scripts/10_g2_harness.py` as a background
    job; the job generates candidates with Qwen3-4B-Instruct-2507 on AIME 2025 and
    HMMT Feb 2025, runs the untrained reader and its controls, and judges the four signs
    of 2608.18379. It touches no sealed set. Everything is resumable: if the session ends,
    start the job again and it continues.

    Attach the GPU from the notebook specs button first. A session ends after 12 hours or
    90 idle minutes. Tokens are typed into the password box or read from the `HF_TOKEN`
    variable; they are never written to a file, a log or a command line.
    """)
    return


@app.cell
def _():
    RAW_REPO = (
        "Arshia-HZ/trained-reader-audit-raw"  # private dataset of raw generations
    )
    READER_URL = "https://github.com/Arshia-HZ/trained-reader-audit.git"
    READER_REF = "main"  # branch, tag or commit of this study's repository
    AUDIT_URL = "https://github.com/Arshia-HZ/mas-channel-audit.git"
    AUDIT_REF = "audit-final"
    MODEL_CONFIG = "configs/models/qwen3-4b-instruct-2507.yaml"
    SLUG = "g2_qwen3-4b-instruct-2507"
    return AUDIT_REF, AUDIT_URL, MODEL_CONFIG, RAW_REPO, READER_REF, READER_URL, SLUG


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Clone, install, show the GPU")
    token_input = mo.ui.text(
        kind="password",
        label="Hugging Face token (only for the upload; not needed if HF_TOKEN is set)",
    )
    mo.vstack([setup_button, token_input])
    return setup_button, token_input


@app.cell
def _(AUDIT_REF, AUDIT_URL, READER_REF, READER_URL, mo, setup_button):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    mo.stop(not setup_button.value, mo.md("Press **1** to clone and install."))

    READER = Path.home() / "trained-reader-audit"
    AUDIT = Path.home() / "mas-channel-audit"
    RUN_ROOT = Path.home() / "trained-reader-audit-runs"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    def sh(*cmd, cwd=None):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode:
            raise RuntimeError(f"{' '.join(cmd)} exited with {proc.returncode}")

    # the audit at its tag (a detached checkout; its frozen prompts live in the repository)
    if not (AUDIT / ".git").is_dir():
        sh("git", "clone", "-q", AUDIT_URL, str(AUDIT))
    sh("git", "fetch", "-q", "--all", "--tags", cwd=AUDIT)
    sh("git", "checkout", "-q", AUDIT_REF, cwd=AUDIT)

    if not (READER / ".git").is_dir():
        sh("git", "clone", "-q", READER_URL, str(READER))
    sh("git", "fetch", "-q", "--all", "--tags", cwd=READER)
    sh("git", "checkout", "-q", READER_REF, cwd=READER)
    if (
        subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=READER).returncode
        == 0
    ):
        sh("git", "pull", "-q", "--ff-only", cwd=READER)  # on a branch: take the latest

    # jobs.py needs only the standard library and channel_audit.launcher, so it is loaded
    # by path: importing reader_audit itself would need the grader's packages here
    sys.path.insert(0, str(AUDIT / "src"))
    _spec = importlib.util.spec_from_file_location(
        "reader_jobs", READER / "src" / "reader_audit" / "jobs.py"
    )
    jobs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(jobs)
    from channel_audit import launcher

    # the job gets its own environment: installing vLLM must not replace the packages
    # this notebook kernel runs on
    VENV = Path.home() / "reader-venv"
    _cmds = jobs.setup_commands(str(READER), str(AUDIT), str(VENV))
    if not Path(launcher.venv_python(VENV)).exists():
        sh(*_cmds[0])
    sh(*_cmds[1])
    PYTHON = launcher.venv_python(VENV)
    sh("git", "log", "--oneline", "-1", cwd=READER)
    sh("git", "log", "--oneline", "-1", cwd=AUDIT)
    subprocess.run(["nvidia-smi"])
    mo.md(f"Both repositories ready under `{Path.home()}`. Runs go to `{RUN_ROOT}`.")
    return AUDIT, PYTHON, Path, READER, RUN_ROOT, jobs, launcher


@app.cell
def _(mo):
    launch_button = mo.ui.run_button(label="2. Start the harness check")
    stop_button = mo.ui.run_button(label="Stop the job")
    mo.hstack([launch_button, stop_button], justify="start")
    return launch_button, stop_button


@app.cell
def _(
    MODEL_CONFIG,
    PYTHON,
    Path,
    READER,
    RUN_ROOT,
    SLUG,
    jobs,
    launch_button,
    launcher,
    mo,
    token_input,
):
    import os

    mo.stop(not launch_button.value, mo.md("Press **2** to start the job."))

    job = jobs.G2Job(
        out=str(RUN_ROOT / SLUG),
        model_config=str(READER / MODEL_CONFIG),
        sets=str(READER / "data" / "sets"),
    )
    _env = job.env()
    _token = os.environ.get("HF_TOKEN") or token_input.value
    if _token:
        _env["HF_TOKEN"] = _token  # read-only use here; the job never prints it
    launcher.record_hardware(job.out, python=PYTHON)
    LOG = Path(job.out).parent / f"{SLUG}.log"
    PID = Path(job.out).parent / f"{SLUG}.pid"
    _pid = launcher.start(job.command(python=PYTHON), LOG, PID, env=_env)
    mo.md(f"Started as process {_pid}. Log: `{LOG}`")
    return LOG, PID, job


@app.cell
def _(PID, launcher, mo, stop_button):
    mo.stop(not stop_button.value, None)
    launcher.stop(PID)
    mo.md("Stop signal sent. Press **2** again later; finished work is kept.")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="30s", label="Refresh the log")
    refresh
    return (refresh,)


@app.cell
def _(LOG, PID, RUN_ROOT, SLUG, launcher, mo, refresh):
    refresh.value  # rerun on every refresh
    _state = "running" if launcher.is_running(PID) else "not running"
    _report = RUN_ROOT / SLUG / "g2_report.md"
    mo.vstack(
        [
            mo.md(f"**{_state}**"),
            mo.md(f"```\n{launcher.tail(LOG, 40)}\n```"),
            mo.md(_report.read_text(encoding="utf-8"))
            if _report.exists()
            else mo.md("The report appears here when the job finishes."),
        ]
    )
    return


@app.cell
def _(mo):
    upload_button = mo.ui.run_button(label="3. Upload the run to the private dataset")
    upload_button
    return (upload_button,)


@app.cell
def _(RAW_REPO, RUN_ROOT, SLUG, jobs, mo, token_input, upload_button):
    import os as _os

    mo.stop(not upload_button.value, mo.md("Press **3** when the job has finished."))
    _token = _os.environ.get("HF_TOKEN") or token_input.value
    if not _token:
        raise RuntimeError("the upload needs a Hugging Face token with write access")
    from huggingface_hub import HfApi

    jobs.upload_folder(HfApi(token=_token), RUN_ROOT / SLUG, RAW_REPO, f"g2/{SLUG}")
    mo.md(f"Uploaded to `{RAW_REPO}` under `g2/{SLUG}`.")
    return


if __name__ == "__main__":
    app.run()
