"""Build the training problem list: MATH train levels 3 to 5 plus a DeepScaleR subset,
with every problem that overlaps a study set removed (PREREG section 9, delegated rule).

    python scripts/14_build_train.py \
        --audit-problems ../mas-channel-audit/data/problems --out data/train \
        --n-deepscaler 2600

Writes ``data/train/ids.json`` and ``manifest.json`` (sources and revisions, counts,
what was dropped and why, the hash of the ids). The problem texts are rebuilt from the
sources, not committed. Study sets are read for their text only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from channel_audit import data as audit_data
from channel_audit.check_problems import CONFIGS

from reader_audit import externals, sets
from reader_audit.train_set import (
    DEEPSCALER_REPO,
    MATH_REPO,
    NGRAM,
    SEED,
    Decontaminator,
    Reference,
    build_train_set,
    ids_sha256,
    load_deepscaler,
    load_math_train,
)

OLYMPIAD_REPO = "Hothan/OlympiadBench"
OLYMPIAD_FILE = "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"


def hub_file(repo: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def head(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo).sha


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit-problems", required=True)
    ap.add_argument("--sets", default="data/sets")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--n-deepscaler", type=int, default=2600)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if (out / "manifest.json").exists():
        print(f"error: {out / 'manifest.json'} exists; the train set is built once")
        return 1

    import pyarrow.parquet as pq

    # ---- the study sets, as text -------------------------------------------------
    refs: list[Reference] = []
    for name in ("dev", "primary", "reserve"):
        for p in sets.load_math_set(name, args.sets, args.audit_problems):
            refs.append(Reference(name, p.problem_id, p.question))
    for name in ("screen", "pilot"):
        for p in audit_data.load_frozen(args.audit_problems, name):
            refs.append(Reference(name, p.problem_id, p.question))
    for p in externals.load_harness(args.sets, externals.load_matharena):
        refs.append(Reference("harness", p.problem_id, p.question))
    olympiad_rev = head(OLYMPIAD_REPO)
    table = pq.read_table(hub_file(OLYMPIAD_REPO, OLYMPIAD_FILE, olympiad_rev))
    for i, row in enumerate(table.to_pylist()):
        refs.append(Reference("secondary", f"olympiadbench-{i}", row["question"]))
    decon = Decontaminator(refs)

    # ---- the sources ---------------------------------------------------------------
    def math_rows(config: str):
        path = hub_file(
            MATH_REPO,
            f"{config}/train-00000-of-00001.parquet",
            audit_data.MATH_REVISION,
        )
        return pq.read_table(path).to_pylist()

    math_train, unusable = load_math_train(math_rows)
    scaler_rev = head(DEEPSCALER_REPO)
    scaler_path = hub_file(DEEPSCALER_REPO, "deepscaler.json", scaler_rev)
    scaler = load_deepscaler(json.loads(Path(scaler_path).read_text("utf-8")))

    kept, report = build_train_set(math_train, scaler, decon, args.n_deepscaler)
    ids = [p.problem_id for p in kept]
    manifest = {
        "n": len(kept),
        "ids_sha256": ids_sha256(ids),
        "ngram": NGRAM,
        "seed": SEED,
        "n_deepscaler_requested": args.n_deepscaler,
        "sources": {
            "math-train": {
                "repo": MATH_REPO,
                "revision": audit_data.MATH_REVISION,
                "configs": list(CONFIGS),
                "levels": [3, 4, 5],
                "candidates": len(math_train),
                "rows_without_usable_level_or_gold": len(unusable),
            },
            "deepscaler": {
                "repo": DEEPSCALER_REPO,
                "revision": scaler_rev,
                "candidates": len(scaler),
            },
        },
        "references": dict(decon.sets) | {"secondary_revision": olympiad_rev},
        "report": report,
        "levels": {str(lv): sum(p.level == lv for p in kept) for lv in (3, 4, 5)},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps(sorted(ids)) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
