"""Exploratory, dev only (PREREG R22, E2): Qwen3-1.7B in thinking mode on the escalation
slice, the sets on which early-stopping SC (window 2, cap 8) finds no agreeing pair.

    # generate (resumable: generations already in the output are kept)
    python scripts/25_thinking_probe.py run \
        --pools-root runs/dev_pools/confirm/qwen3-1.7b --out runs/think_probe
    # summarise, with the stored non-thinking reads of the same sets
    python scripts/25_thinking_probe.py report \
        --pools-root runs/dev_pools/confirm/qwen3-1.7b --out runs/think_probe \
        --compare untrained=runs/dev_untrained/dev_reader_qwen3-1.7b \
        --compare step100=runs/dev_trained/dev_trained_reader_s1_step0100_reader_full

Two arms, ``candidate_free`` and ``reader_full``, with the frozen ``reader_v1`` prompts
(candidates cut at 1,536 tokens by the base tokenizer, as in R13); thinking on, Qwen's
stated sampling settings for that mode, at most 16,384 new tokens in a 32,768 window.
The answer is graded from the text after ``</think>``; thinking that never closes is
wrong. This is a separate path from the frozen non-thinking harness, which it does not
touch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from reader_audit import allocation as al
from reader_audit import reader_prompts as rp
from reader_audit import think_probe as tp
from reader_audit.arms import build_arm_input
from reader_audit.cand_sets import draw_set, vote_credit
from reader_audit.pools import load_pools

MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
K, DRAWS, LEVEL = 8, 16, 5
ARMS = ("candidate_free", "reader_full")
MAX_TOKENS, WINDOW = 16384, 32768
STOP_IDS = [151645, 151643]
GENS = "generations.jsonl"


def level_pools(root: str) -> dict:
    return {p: v for p, v in load_pools(root).items() if v.problem.level == LEVEL}


def plan(pools: dict, n_problems: int, sets_per_problem: int) -> list[dict]:
    with_slice = {
        pid: tp.escalated_draws(pool, K, DRAWS) for pid, pool in pools.items()
    }
    chosen = tp.pick_problems(sorted(p for p, d in with_slice.items() if d), n_problems)
    rows = []
    for pid in chosen:
        for draw in with_slice[pid][:sets_per_problem]:
            for arm in ARMS:
                rows.append(
                    {"problem_id": pid, "draw": draw, "arm": arm, "replicate": 0}
                )
    return rows


def key(row: dict) -> str:
    return f"{row['problem_id']}|{row['draw']}|{row['arm']}|{row['replicate']}"


def cmd_run(args) -> int:
    import vllm
    from transformers import AutoTokenizer

    from reader_audit.clipper import clipper_from_tokenizer

    pools = level_pools(args.pools_root)
    rows = plan(pools, args.problems, args.sets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gens_path = out / GENS
    done = set()
    if gens_path.is_file():
        done = {key(json.loads(line)) for line in gens_path.open(encoding="utf-8")}
    todo = [r for r in rows if key(r) not in done]
    (out / "plan.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} planned, {len(todo)} to generate", flush=True)
    if not todo:
        return 0
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    truncate = clipper_from_tokenizer(tok)
    close_id = tok.convert_tokens_to_ids(tp.THINK_CLOSE)
    prompts, params = [], []
    for r in todo:
        pool = pools[r["problem_id"]]
        cset = draw_set(pool, K, r["draw"])
        inp = build_arm_input(
            r["arm"], pool, cset, cap=rp.CANDIDATE_CAP, truncate=truncate
        )
        text = tok.apply_chat_template(
            list(inp.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append({"prompt_token_ids": ids})
        params.append(
            vllm.SamplingParams(
                **tp.SAMPLING,
                repetition_penalty=1.0,
                max_tokens=MAX_TOKENS,
                seed=tp.seed_for(r["problem_id"], r["draw"], r["arm"], r["replicate"]),
                stop_token_ids=STOP_IDS,
                n=1,
            )
        )
    llm = vllm.LLM(
        model=MODEL_ID,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="float16",
        max_model_len=WINDOW,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.time() - t0
    with gens_path.open("a", encoding="utf-8") as f:
        for r, p, o in zip(todo, prompts, outputs, strict=True):
            c = o.outputs[0]
            token_ids = list(c.token_ids)
            think_tokens = token_ids.index(close_id) if close_id in token_ids else None
            text = c.text
            if think_tokens is not None and tp.THINK_CLOSE not in text:
                text = (
                    tok.decode(token_ids[:think_tokens])
                    + tp.THINK_CLOSE
                    + tok.decode(token_ids[think_tokens + 1 :])
                )
            g = tp.grade_thinking(
                text, c.finish_reason, pools[r["problem_id"]].problem.gold
            )
            f.write(
                json.dumps(
                    {
                        **r,
                        "prompt_tokens": len(p["prompt_token_ids"]),
                        "completion_tokens": len(token_ids),
                        "think_tokens": think_tokens,
                        "finish_reason": c.finish_reason,
                        "closed": g.closed,
                        "status": g.status,
                        "extracted": g.extracted,
                        "correct": g.correct,
                        "text": text,
                    }
                )
                + "\n"
            )
    n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"{len(todo)} generations, {seconds:.0f} s, {n_tok} tokens", flush=True)
    return 0


def load_reads_script():
    path = Path(__file__).resolve().parent / "24_dev_allocation.py"
    spec = importlib.util.spec_from_file_location("dev_allocation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reads


def cmd_report(args) -> int:
    pools = level_pools(args.pools_root)
    out = Path(args.out)
    gens = [json.loads(line) for line in (out / GENS).open(encoding="utf-8")]
    units = {pid: al.unit_costs(pool)["generated"] for pid, pool in pools.items()}
    sets = sorted({(g["problem_id"], g["draw"]) for g in gens})
    by_arm = {a: [g for g in gens if g["arm"] == a] for a in ARMS}
    report: dict = {
        "sets": len(sets),
        "problems": len({p for p, _ in sets}),
        "arms": {},
    }
    for arm, gs in by_arm.items():
        report["arms"][f"think_{arm}"] = {
            "n": len(gs),
            "credit": float(np.mean([g["correct"] for g in gs])),
            "completion_tokens": float(np.mean([g["completion_tokens"] for g in gs])),
            "extra_samples": float(
                np.mean([g["completion_tokens"] / units[g["problem_id"]] for g in gs])
            ),
            "prompt_tokens": float(np.mean([g["prompt_tokens"] for g in gs])),
            "cut_by_cap": float(np.mean([g["finish_reason"] == "length" for g in gs])),
            "unclosed": float(np.mean([not g["closed"] for g in gs])),
        }
    # the same sets without thinking: the vote, ESC continued, stored reads
    vote, cont_credit, cont_extra = [], [], []
    for pid, draw in sets:
        pool = pools[pid]
        vote.append(vote_credit(pool, draw_set(pool, K, draw)))
        drawn, credit = tp.esc_continued(pool, K, draw)
        cont_credit.append(credit)
        cont_extra.append(drawn - K)
    report["arms"]["vote_of_8"] = {"n": len(sets), "credit": float(np.mean(vote))}
    report["arms"]["esc_continued_to_16"] = {
        "n": len(sets),
        "credit": float(np.mean(cont_credit)),
        "extra_samples": float(np.mean(cont_extra)),
    }
    load_reads = load_reads_script()
    for spec in args.compare or []:
        label, dirs = spec.split("=", 1)
        reads = load_reads([Path(d) for d in dirs.split(",")])
        vals, extra = [], []
        for pid, draw in sets:
            rs = reads.get((pid, "reader_full"), {}).get(draw, [])
            if rs:
                vals.append(np.mean([r.correct for r in rs]))
                extra.append(np.mean([r.completion_tokens for r in rs]) / units[pid])
        report["arms"][f"nothink_read_{label}"] = {
            "n": len(vals),
            "credit": float(np.mean(vals)) if vals else float("nan"),
            "extra_samples": float(np.mean(extra)) if extra else float("nan"),
        }
    text = json.dumps(report, indent=1)
    (out / "report.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pools-root", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--problems", type=int, default=24)
        p.add_argument("--sets", type=int, default=2)
    sub.choices["report"].add_argument("--compare", action="append")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
