"""Exploratory, dev only (PREREG R24, E3): the thinking pilot, in shards uploaded as
they finish.

    python scripts/26_thinking_pilot.py --pools-root runs/dev_pools/confirm/qwen3-1.7b \
        --out runs/think_pilot --upload

Arm A: the fresh thinking solve (``candidate_free``) of all 208 level-5 dev problems, 4
samples each, at most 30,000 new tokens. Arm B: the thinking read (``reader_full``) of
the first two sets of each problem that ESC (window 2, cap 8) leaves without an agreeing
pair, at most 16,384 new tokens. Untrained Qwen3-1.7B, thinking on, the settings of R22
with new seeds. Four shards of 52 problems; a shard already on the hub (or finished
locally) is not generated again. Each arm of a shard is one engine call, so its wall
time is measured apart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from reader_audit import reader_prompts as rp
from reader_audit import think_probe as tp
from reader_audit.arms import build_arm_input
from reader_audit.cand_sets import draw_set
from reader_audit.pools import load_pools

MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
K, DRAWS, LEVEL = 8, 16, 5
SALT = "reader-audit/think-pilot/v1"
SOLVES, READ_SETS = 4, 2
MAX_TOKENS = {"candidate_free": 30000, "reader_full": 16384}
WINDOW, N_SHARDS = 32768, 4
STOP_IDS = [151645, 151643]
RAW_REPO = "Arshia-HZ/trained-reader-audit-raw"
HUB_DIR = "think_pilot_qwen3-1.7b"
DONE, GENS = "DONE.json", "generations.jsonl"


def level_pools(root: str) -> dict:
    return {p: v for p, v in load_pools(root).items() if v.problem.level == LEVEL}


def plan(pools: dict) -> list[list[dict]]:
    """The rows of each shard: problems in sorted order, split into N_SHARDS."""
    ids = sorted(pools)
    size = -(-len(ids) // N_SHARDS)
    shards = []
    for start in range(0, len(ids), size):
        rows = []
        for pid in ids[start : start + size]:
            for s in range(SOLVES):
                rows.append(
                    {
                        "problem_id": pid,
                        "arm": "candidate_free",
                        "draw": -1,
                        "replicate": s,
                    }
                )
            for draw in tp.escalated_draws(pools[pid], K, DRAWS)[:READ_SETS]:
                rows.append(
                    {
                        "problem_id": pid,
                        "arm": "reader_full",
                        "draw": draw,
                        "replicate": 0,
                    }
                )
        shards.append(rows)
    return shards


def shard_name(i: int) -> str:
    return f"shard_{i:03d}"


def generate(llm, vllm, tok, pools, rows, close_id) -> tuple[list[dict], float]:
    from reader_audit.clipper import clipper_from_tokenizer

    truncate = clipper_from_tokenizer(tok)
    prompts, params = [], []
    for r in rows:
        pool = pools[r["problem_id"]]
        draw = max(r["draw"], 0)  # the candidate-free prompt does not use the set
        inp = build_arm_input(
            r["arm"],
            pool,
            draw_set(pool, K, draw),
            cap=rp.CANDIDATE_CAP,
            truncate=truncate,
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
                max_tokens=min(MAX_TOKENS[r["arm"]], WINDOW - len(ids)),
                seed=tp.seed_for(
                    r["problem_id"], r["draw"], r["arm"], r["replicate"], SALT
                ),
                stop_token_ids=STOP_IDS,
                n=1,
            )
        )
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.time() - t0
    records = []
    for r, p, o in zip(rows, prompts, outputs, strict=True):
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
        records.append(
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
    return records, seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--max-shards", type=int)
    args = ap.parse_args(argv)

    pools = level_pools(args.pools_root)
    shards = plan(pools)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(shards, indent=1), encoding="utf-8")
    print(
        f"{len(pools)} problems, {sum(map(len, shards))} rows in {len(shards)} shards",
        flush=True,
    )
    api = None
    if args.upload:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi(token=os.environ["HF_TOKEN"])
        snapshot_download(
            RAW_REPO,
            repo_type="dataset",
            allow_patterns=[f"{HUB_DIR}/*"],
            local_dir=str(out.parent / "hub_copy"),
            token=os.environ["HF_TOKEN"],
        )
        for i in range(len(shards)):
            hub = out.parent / "hub_copy" / HUB_DIR / shard_name(i)
            if (hub / DONE).is_file() and not (out / shard_name(i) / DONE).is_file():
                (out / shard_name(i)).mkdir(exist_ok=True)
                for name in (GENS, DONE):
                    (out / shard_name(i) / name).write_bytes((hub / name).read_bytes())
    todo = [i for i in range(len(shards)) if not (out / shard_name(i) / DONE).is_file()]
    if args.max_shards is not None:
        todo = todo[: args.max_shards]
    print("shards to run:", [shard_name(i) for i in todo], flush=True)
    if not todo:
        return 0

    import vllm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    close_id = tok.convert_tokens_to_ids(tp.THINK_CLOSE)
    llm = vllm.LLM(
        model=MODEL_ID,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="float16",
        max_model_len=WINDOW,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    for i in todo:
        folder = out / shard_name(i)
        folder.mkdir(exist_ok=True)
        summary = {"shard": shard_name(i), "arms": {}}
        records = []
        for arm in ("candidate_free", "reader_full"):
            rows = [r for r in shards[i] if r["arm"] == arm]
            if not rows:
                continue
            recs, seconds = generate(llm, vllm, tok, pools, rows, close_id)
            records += recs
            summary["arms"][arm] = {
                "n": len(recs),
                "seconds": seconds,
                "completion_tokens": sum(r["completion_tokens"] for r in recs),
                "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            }
            print(
                f"{shard_name(i)} {arm}: {len(recs)} generations, {seconds:.0f} s, "
                f"{summary['arms'][arm]['completion_tokens']} tokens",
                flush=True,
            )
        with (folder / GENS).open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        (folder / DONE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if api is not None:
            for attempt in range(3):
                try:
                    api.upload_folder(
                        folder_path=str(folder),
                        repo_id=RAW_REPO,
                        repo_type="dataset",
                        path_in_repo=f"{HUB_DIR}/{shard_name(i)}",
                        commit_message=f"E3 thinking pilot {shard_name(i)}",
                    )
                    print(f"uploaded {shard_name(i)}", flush=True)
                    break
                except Exception as exc:  # keep going; the shard is on disk
                    print(
                        f"upload failed ({type(exc).__name__}), try {attempt + 1}",
                        flush=True,
                    )
    print("PILOT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
