"""Stage 2: the dev evaluation of a trained reader's milestones (PREREG D17, R19).

    # one milestone, its own process (the engine is not freed inside a process)
    python scripts/23_dev_checkpoints.py run --run reader_s1 --step 300 \
        --download-pools --upload --resume
    # the selected milestone only, the arm the second freeze needs for H2
    python scripts/23_dev_checkpoints.py run --run reader_s1 --step 300 \
        --arms candidate_free --download-pools --upload --resume
    # the rule, from the finished shards of the three milestones
    python scripts/23_dev_checkpoints.py select --run reader_s1 --download-pools \
        --download-shards --report docs/dev_checkpoint_rule.json
    # step 100 of every seed, both arms, descriptive (R28)
    python scripts/23_dev_checkpoints.py seeds --runs reader_s1 reader_s2 reader_s3 \
        --step 100 --download-pools --download-shards \
        --report docs/dev_seeds_step100.json --md docs/dev_seeds_step100.md

The reader is the trained milestone read from the private checkpoint repo (a pinned
commit), on the 208 level-5 dev problems of R13: K = 8, 16 seeded sets per problem, 2
replicates per set, temperature 0.7, candidates cut at 1,536 tokens by the base model's
tokenizer, the frozen ``reader_v1`` prompts (identical in text to the draft R13 used).
Before any generation the milestone is checked against the base model for everything but
the weights (generation config, weight format, the token ids of 208 dev prompts under
the milestone's own tokenizer), and its folder then reads the base model's tokenizer and
chat template, as R13 did. The tokens are read from ``HF_TOKEN`` and never printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from channel_audit.backends.chat import render_prompt_ids
from channel_audit.models import load_model_config
from channel_audit.runner import config_for_model

from reader_audit import checkpoint_rule as cr
from reader_audit import jobs
from reader_audit import reader_prompts as rp
from reader_audit.dev_job import run_dev_reader
from reader_audit.pools import load_pools
from reader_audit.reader_run import plan_reader_run

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "models" / "qwen3-1.7b-reader.yaml"
CKPT_REPO = "Arshia-HZ/trained-reader-audit-ckpt"
POOL_REPO = "Arshia-HZ/mas-channel-audit-raw"
POOL_PATH = "confirm/qwen3-1.7b"
RAW_REPO = jobs.RAW_REPO


def slug_for(run: str, step: int, arms: list[str]) -> str:
    return f"dev_trained_{run}_step{step:04d}_{'+'.join(arms)}"


def load_summarise():
    """R13's own function, loaded from its script, so the rule uses the same code."""
    path = REPO_ROOT / "scripts" / "13_dev_reader_analysis.py"
    spec = importlib.util.spec_from_file_location("dev_reader_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarise


def get_pools(args) -> dict:
    root = args.pools_root
    if args.download_pools:
        from huggingface_hub import snapshot_download

        local = Path(args.out).parent / "dev_pools"
        snapshot_download(
            POOL_REPO,
            repo_type="dataset",
            allow_patterns=[f"{POOL_PATH}/*"],
            local_dir=str(local),
            token=os.environ.get("HF_TOKEN"),
        )
        root = str(local / POOL_PATH)
    if not root:
        raise SystemExit("error: give --pools-root or --download-pools")
    pools = {
        pid: pool
        for pid, pool in load_pools(root).items()
        if pool.problem.level == cr.LEVEL
    }
    print(f"{len(pools)} dev problems at level {cr.LEVEL}", flush=True)
    return pools


def fetch_milestone(args, folder: Path) -> str:
    """Download one milestone from a pinned commit of the checkpoint repo."""
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ["HF_TOKEN"]
    sha = HfApi(token=token).model_info(args.repo).sha
    snapshot_download(
        args.repo,
        repo_type="model",
        allow_patterns=[f"{args.run}/step_{args.step:04d}/*"],
        revision=sha,
        local_dir=str(args.milestones_root),
        token=token,
    )
    gone = cr.missing_files(folder)
    if gone:
        raise SystemExit(f"error: {folder} lacks {gone}")
    return sha


def check_identity(base, folder: Path, pools: dict, cfg) -> list[str]:
    """What could differ between the milestone and the base besides the weights."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(base.model_id, name, revision=base.revision))

    jl = lambda p: json.loads(Path(p).read_text("utf-8"))  # noqa: E731
    plan = plan_reader_run(
        pools, cfg, arms=["reader_full"], ks=(cr.K,), n_draws=1, replicates=1
    )
    kwargs = base.chat_template_kwargs
    base_tok = AutoTokenizer.from_pretrained(base.model_id, revision=base.revision)
    ms_tok = AutoTokenizer.from_pretrained(str(folder))
    digest = lambda tok: cr.ids_digest(  # noqa: E731
        [render_prompt_ids(tok, p.messages, kwargs) for p in plan.planned]
    )
    problems = cr.identity_problems(
        base_generation_config=jl(fetch("generation_config.json")),
        milestone_generation_config=jl(folder / "generation_config.json"),
        base_config=jl(fetch("config.json")),
        milestone_config=jl(folder / "config.json"),
        base_ids_digest=digest(base_tok),
        milestone_ids_digest=digest(ms_tok),
    )
    if not problems:  # from here on the folder reads the base tokenizer, as R13 did
        cr.use_base_tokenizer(
            folder, {name: fetch(name) for name in cr.BASE_TOKENIZER_FILES}
        )
    return problems


def cmd_run(args) -> int:
    from reader_audit.clipper import load_hf_clipper

    if args.step not in cr.MILESTONES:
        raise SystemExit(f"error: --step must be one of {cr.MILESTONES}")
    args.milestones_root = Path(args.milestones_root).expanduser().resolve()
    folder = cr.milestone_folder(args.milestones_root, args.run, args.step)
    pools = get_pools(args)
    sha = fetch_milestone(args, folder)
    base = load_model_config(BASE_CONFIG)
    model = cr.milestone_model_config(base, folder, sha)
    cfg = config_for_model(model, replicates=cr.REPLICATES, temperature=cr.TEMPERATURE)
    problems = check_identity(base, folder, pools, cfg)
    if problems:
        raise SystemExit(
            "error: the milestone differs from the base: " + "; ".join(problems)
        )
    print(f"identity with the base model checked; hub commit {sha[:10]}", flush=True)

    slug = slug_for(args.run, args.step, args.arms)
    out_root = Path(args.out)
    out = out_root / slug
    if args.resume:
        from huggingface_hub import snapshot_download

        snapshot_download(
            RAW_REPO,
            repo_type="dataset",
            allow_patterns=[f"{slug}/*"],
            local_dir=str(out_root),
            token=os.environ["HF_TOKEN"],
        )
    api = None
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

    def on_done(shard: Path, index: int) -> None:
        try:
            jobs.upload_folder(api, shard, RAW_REPO, f"{slug}/{shard.name}")
            print(f"uploaded {shard.name}", flush=True)
        except Exception as exc:  # the data stays on disk; the job goes on
            print(f"upload of {shard.name} failed: {type(exc).__name__}", flush=True)

    started = time.monotonic()
    clipper = load_hf_clipper(
        base.model_id, base.revision
    )  # the base tokenizer, as R13
    report = run_dev_reader(
        out,
        pools,
        cfg,
        clipper=clipper,
        arms=args.arms,
        k=cr.K,
        n_draws=cr.DRAWS,
        replicates=cr.REPLICATES,
        shard_size=args.shard_size,
        cap=rp.CANDIDATE_CAP,
        max_shards=args.max_shards,
        on_shard_done=on_done if api else None,
    )
    for name, s in report["shards"].items():
        rate = s["completion_tokens"] / max(s["seconds"], 1e-9)
        print(
            f"{name}: {s['generations']} generations, {s['new_generations']} new, "
            f"{s['seconds']:.0f} s, {s['completion_tokens']} completion tokens "
            f"({rate:.0f} tok/s)",
            flush=True,
        )
    record = {
        "run": args.run,
        "step": args.step,
        "arms": args.arms,
        "hub_commit": sha,
        "temperature": cr.TEMPERATURE,
        "draws": cr.DRAWS,
        "replicates": cr.REPLICATES,
        "k": cr.K,
        "prompt_version": rp.PROMPT_VERSION,
        "seconds_this_call": time.monotonic() - started,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_record.json").write_text(json.dumps(record, indent=2), "utf-8")
    if api:
        try:
            api.upload_file(
                path_or_fileobj=str(out / "run_record.json"),
                path_in_repo=f"{slug}/run_record.json",
                repo_id=RAW_REPO,
                repo_type="dataset",
            )
        except Exception as exc:
            print(f"upload of the run record failed: {type(exc).__name__}", flush=True)
    return 0


def cmd_select(args) -> int:
    pools = get_pools(args)
    out_root = Path(args.out)
    if args.download_shards:
        from huggingface_hub import snapshot_download

        snapshot_download(
            RAW_REPO,
            repo_type="dataset",
            allow_patterns=[f"dev_trained_{args.run}_step*/*"],
            local_dir=str(out_root),
            token=os.environ["HF_TOKEN"],
        )
    summarise = load_summarise()
    per_step = {}
    for step in cr.MILESTONES:
        root = out_root / slug_for(args.run, step, [cr.SELECTION_ARM])
        if cr.finished_shards(root):
            per_step[step] = cr.summarise_step([root], pools, summarise)
    n = len(pools)
    report = cr.rule_report(per_step, n, arm_generations=n * cr.DRAWS * cr.REPLICATES)
    report["settings"] = {
        "level": cr.LEVEL,
        "k": cr.K,
        "draws": cr.DRAWS,
        "replicates": cr.REPLICATES,
        "temperature": cr.TEMPERATURE,
        "prompt_version": rp.PROMPT_VERSION,
    }
    chosen = report["selected"]
    if chosen is not None:
        free = out_root / slug_for(args.run, chosen, [cr.FREE_ARM])
        if cr.finished_shards(free):
            both = [out_root / slug_for(args.run, chosen, [cr.SELECTION_ARM]), free]
            report["selected_with_candidate_free"] = cr.summarise_step(
                both, pools, summarise
            )
    print(json.dumps({k: v for k, v in report.items() if k != "per_step"}, indent=1))
    cr.write_report(report, args.report)
    return 0


def _pct(x, sd=None, n=None) -> str:
    if x is None:
        return "-"
    out = f"{100 * x:+.1f}"
    if sd is not None:
        out += f" (SD {sd:.3f}"
        out += f", {n} problems)" if n is not None else ")"
    return out


def seeds_markdown(report: dict) -> str:
    """The table of ``seeds_report``: one row per seed, then the mean over seeds."""
    rows = [
        f"# Dev evaluation of step {report['step']} across reader seeds (R28)",
        "",
        "Descriptive, dev only; it selects nothing. Points, means over the 208",
        "level-5 dev problems (H2 and H4a over the problems that have a c1 or c2+",
        "set), SD over problems in brackets. Seed 1's step 100 was chosen by D17 as",
        "the best of three steps, so its value is biased upward; seeds 2 and 3 were",
        "fixed at step 100 by R28.",
        "",
        "| Seed | H1a, reader minus matched SC | H2, reader minus its candidate-free"
        " solve (c1) | H4a, reader minus vote (c2+) | Read tokens | k* |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run, r in report["per_run"].items():
        if r is None or run in report["incomplete"]:
            why = "; ".join(report["incomplete"].get(run, []))
            rows.append(f"| {run} | incomplete: {why} | | | | |")
            continue
        h1, h2, h4 = (r[q] for q in cr.SEED_QUANTITIES)
        rows.append(
            f"| {run} | {_pct(h1['mean'], h1['sd'])}"
            f" | {_pct(h2['mean'], h2['sd'], h2['n'])}"
            f" | {_pct(h4['mean'], h4['sd'], h4['n'])}"
            f" | {r['reader_generated_tokens_mean']:.0f} | {r['k_star_mean']:.2f} |"
        )
    a = report["across_seeds"]
    rows.append(
        "| mean over seeds | "
        + " | ".join(_pct(a[q]["mean"]) for q in cr.SEED_QUANTITIES)
        + " | | |"
    )
    rows.append(
        "| SD over seeds (points) | "
        + " | ".join(
            "-"
            if a[q]["sd_over_seeds"] is None
            else f"{100 * a[q]['sd_over_seeds']:.1f}"
            for q in cr.SEED_QUANTITIES
        )
        + " | | |"
    )
    return "\n".join(rows) + "\n"


def cmd_seeds(args) -> int:
    pools = get_pools(args)
    out_root = Path(args.out)
    if args.download_shards:
        from huggingface_hub import snapshot_download

        snapshot_download(
            RAW_REPO,
            repo_type="dataset",
            allow_patterns=[
                f"dev_trained_{run}_step{args.step:04d}_*/*" for run in args.runs
            ],
            local_dir=str(out_root),
            token=os.environ["HF_TOKEN"],
        )
    summarise = load_summarise()
    arms = [cr.SELECTION_ARM, cr.FREE_ARM]
    per_run = {}
    for run in args.runs:
        roots = [out_root / slug_for(run, args.step, [arm]) for arm in arms]
        found = any(cr.finished_shards(root) for root in roots)
        per_run[run] = cr.summarise_step(roots, pools, summarise) if found else None
    n = len(pools)
    report = cr.seeds_report(
        per_run, n, arm_generations=n * cr.DRAWS * cr.REPLICATES, arms=arms
    )
    report["step"] = args.step
    report["settings"] = {
        "level": cr.LEVEL,
        "k": cr.K,
        "draws": cr.DRAWS,
        "replicates": cr.REPLICATES,
        "temperature": cr.TEMPERATURE,
        "prompt_version": rp.PROMPT_VERSION,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, indent=1))
    cr.write_report(report, args.report)
    if args.md:
        Path(args.md).write_text(seeds_markdown(report), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "select", "seeds"):
        p = sub.add_parser(name)
        if name != "seeds":
            p.add_argument("--run", default="reader_s1")
        p.add_argument("--out", default="runs/dev_trained")
        p.add_argument("--pools-root")
        p.add_argument("--download-pools", action="store_true")
        p.add_argument("--repo", default=CKPT_REPO)
    run = sub.choices["run"]
    run.add_argument("--step", type=int, required=True)
    run.add_argument("--arms", nargs="+", default=[cr.SELECTION_ARM])
    run.add_argument("--milestones-root", default="~/milestones")
    run.add_argument("--shard-size", type=int, default=26)
    run.add_argument("--max-shards", type=int)
    run.add_argument("--upload", action="store_true")
    run.add_argument("--resume", action="store_true")
    select = sub.choices["select"]
    select.add_argument("--download-shards", action="store_true")
    select.add_argument("--report", default="docs/dev_checkpoint_rule.json")
    seeds = sub.choices["seeds"]
    seeds.add_argument("--runs", nargs="+", required=True)
    seeds.add_argument("--step", type=int, default=100)
    seeds.add_argument("--download-shards", action="store_true")
    seeds.add_argument("--report", default="docs/dev_seeds_step100.json")
    seeds.add_argument("--md")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "select": cmd_select, "seeds": cmd_seeds}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
