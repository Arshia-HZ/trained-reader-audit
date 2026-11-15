"""Saved-data diagnostic of a G2 run (no GPU, no network): pool truncation, the c2+
vote-versus-reader gap by number of correct candidates, and the effect of the cap.

Usage: python scripts/11_g2_diagnostic.py <run dir containing pool/ and reader/>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from reader_audit import pools as P
from reader_audit import reader_eval as ev

BASE = Path(sys.argv[1])


def jl(p):
    return [
        json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x
    ]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


# ---- A. the candidate pool ------------------------------------------------------
pg = {r["gen_key"]: r for r in jl(BASE / "pool/generations.jsonl")}
pgr = {r["gen_key"]: r for r in jl(BASE / "pool/graded.jsonl")}
print(f"A. POOL: {len(pg)} generations")
by_ds = defaultdict(list)
for k, g in pg.items():
    by_ds[g["problem_id"].split("-")[0] if "-" in g["problem_id"] else "?"].append(k)
for ds, keys in sorted(by_ds.items()):
    fr = Counter(pg[k]["finish_reason"] for k in keys)
    st = Counter(pgr[k]["extraction_status"] for k in keys if k in pgr)
    cor = sum(pgr[k]["correct"] for k in keys if k in pgr)
    toks = sorted(pg[k]["completion_tokens"] for k in keys)
    print(
        f"  {ds}: n={len(keys)} finish={dict(fr)} extraction={dict(st)} "
        f"correct={pct(cor, len(keys))} median_tokens={toks[len(toks) // 2]}"
    )
fr_cor = defaultdict(lambda: [0, 0])
for k, g in pg.items():
    fr_cor[g["finish_reason"]][0] += 1
    fr_cor[g["finish_reason"]][1] += bool(pgr[k]["correct"])
print("  correct by finish_reason:", {a: f"{c}/{n}" for a, (n, c) in fr_cor.items()})

pools = P.read_pools(BASE / "pool")
hist = Counter()
for pool in pools.values():
    hist[sum(bool(c.correct) for c in pool.candidates)] += 1
print(
    "  problems by number of correct candidates out of 16:", dict(sorted(hist.items()))
)
trunc_by_p = {
    pid: sum(c.truncated for c in pool.candidates) for pid, pool in pools.items()
}
print(
    "  problems by number of truncated candidates (of 16):",
    dict(sorted(Counter(trunc_by_p.values()).items())),
)

# ---- B. the reader stage --------------------------------------------------------
rp = jl(BASE / "reader/plan.jsonl")
rg = {r["gen_key"]: r for r in jl(BASE / "reader/generations.jsonl")}
rgr = {r["gen_key"]: r for r in jl(BASE / "reader/graded.jsonl")}
print(f"\nB. READER: {len(rp)} plan rows")
for arm in ("reader_full", "candidate_free", "answer_only", "placebo"):
    rows = [r for r in rp if r["arm"] == arm]
    fr = Counter(rg[r["gen_key"]]["finish_reason"] for r in rows)
    st = Counter(rgr[r["gen_key"]]["extraction_status"] for r in rows)
    cor = sum(rgr[r["gen_key"]]["correct"] for r in rows)
    print(
        f"  {arm}: n={len(rows)} correct={pct(cor, len(rows))} "
        f"finish={dict(fr)} extraction={dict(st)}"
    )

# ---- C. c2+ sets: vote vs reader, by number of correct candidates --------------
out = ev.load_outcomes(BASE / "reader") + ev.vote_outcomes(pools, (4,), 8)
cell = {}
for o in out:
    cell[(o.problem_id, o.draw, o.arm)] = o
print("\nC. c2+ SETS (K=4), mean credit by arm and by number of correct candidates")
rows = defaultdict(lambda: defaultdict(list))
for (_pid, _d, arm), o in cell.items():
    if o.stratum == "c2+":
        rows[o.n_correct][arm].append(o.value)
        rows["all"][arm].append(o.value)
for nc in sorted(rows, key=str):
    r = rows[nc]
    n = len(r[ev.VOTE])
    print(
        f"  n_correct={nc} sets={n}: "
        + " ".join(f"{a}={sum(v) / len(v):.3f}" for a, v in sorted(r.items()))
    )

print("\n  disagreements on c2+ sets (reader_full vs vote):")
agree = Counter()
for (pid, d, arm), o in cell.items():
    if o.stratum == "c2+" and arm == "reader_full":
        v = cell[(pid, d, ev.VOTE)].value
        agree[(round(v, 2), o.value)] += 1
for kk, n in sorted(agree.items()):
    print(f"    vote credit {kk[0]}, reader {kk[1]}: {n}")

# per problem on c2+: where does the reader differ from the vote
print("\n  per problem on c2+ (vote, reader, candidate_free), problems with a gap:")
pp = defaultdict(lambda: defaultdict(list))
for (pid, _d, arm), o in cell.items():
    if o.stratum == "c2+":
        pp[pid][arm].append(o.value)
for pid, r in sorted(pp.items()):
    m = {a: sum(v) / len(v) for a, v in r.items()}
    if abs(m.get(ev.VOTE, 0) - m.get("reader_full", 0)) > 0.01:
        print(
            f"    {pid}: sets={len(r[ev.VOTE])} vote={m[ev.VOTE]:.2f} "
            f"reader={m['reader_full']:.2f} "
            f"cand_free={m['candidate_free']:.2f} answer_only={m['answer_only']:.2f}"
        )

# ---- D. what the display cap did on the reader prompts --------------------------
print("\nD. DISPLAY CAP on reader_full prompts (detail counters)")
det = [r["detail"] for r in rp if r["arm"] == "reader_full"]
for key in ("n_gen_truncated", "n_prompt_clipped", "n_answer_hidden"):
    c = Counter(d.get(key, 0) for d in det)
    print(f"  {key}: {dict(sorted(c.items()))}")
print("  reader_full accuracy by n_prompt_clipped and stratum:")
acc = defaultdict(list)
for r in rp:
    if r["arm"] == "reader_full":
        acc[(r["stratum"], r["detail"].get("n_prompt_clipped", 0))].append(
            rgr[r["gen_key"]]["correct"]
        )
for kk, v in sorted(acc.items()):
    print(f"    {kk}: n={len(v)} acc={sum(v) / len(v):.3f}")
print("  reader_full accuracy by n_gen_truncated (candidates cut by max_tokens):")
acc = defaultdict(list)
for r in rp:
    if r["arm"] == "reader_full":
        acc[(r["stratum"], r["detail"].get("n_gen_truncated", 0))].append(
            rgr[r["gen_key"]]["correct"]
        )
for kk, v in sorted(acc.items()):
    print(f"    {kk}: n={len(v)} acc={sum(v) / len(v):.3f}")
ptoks = [rg[r["gen_key"]]["prompt_tokens"] for r in rp if r["arm"] == "reader_full"]
ptoks.sort()
print(f"  reader_full prompt tokens: median={ptoks[len(ptoks) // 2]} max={ptoks[-1]}")
