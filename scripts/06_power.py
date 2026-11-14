"""Power of the six-hypothesis Family 1 under Holm (PREREG section 6), as a table.

    python scripts/06_power.py --design docs/design_numbers_dev.json \
        --out docs/power_table.md

Reads the dev design numbers (``05_design_numbers.py``) and prints, for 4, 8 and 16
evaluation draws per problem, each hypothesis's problems, placeholder standard
deviation, minimum detectable effect and power at the Holm worst case (alpha / 6 on the
two-sided scale) and unadjusted. The standard deviations are placeholders from a
binomial model and the headline SD 0.18; the freeze replaces them with the ones
measured on dev.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reader_audit import power as P

N_PRIMARY = 972
REPLICATES = 2


def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}"


def render(design: dict, sd_h1: float) -> str:
    level_5 = design["sets"]["level_5"]
    lines = [
        "# Power of Family 1 under Holm (placeholders; level-5 dev design numbers)",
        "",
        f"972 primary problems, {REPLICATES} replicates, K = 8. Headline SD {sd_h1}. "
        "Levels are on the two-sided scale: the Holm worst case is 0.05 / 6 = 0.0083. "
        "Power is for the planning effect (H1, H2) or for a true difference of 0 "
        "inside "
        "the margin (H3: 2 points, H4a: 1 point). MDE = smallest true effect found "
        "with 80% power. Percentages are points of accuracy.",
        "",
    ]
    for draws, block in level_5["by_draws"].items():
        designs = {
            kind: P.StratumDesign(v["coverage"], v["sets_per_covered_problem"])
            for kind, v in block.items()
        }
        rows = P.family1_rows(
            designs, N_PRIMARY, REPLICATES, sd_h1=sd_h1, draws=int(draws)
        )
        lines += [
            f"## {draws} draws per problem",
            "",
            "| Hypothesis | Problems | SD | MDE, Holm worst | MDE, unadjusted | "
            "Power, Holm worst | Power, unadjusted |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['hypothesis']} | {r['n_problems']} | {r['sd']:.3f} | "
                f"{pct(r['mde_holm_worst'])} | {pct(r['mde_unadjusted'])} | "
                f"{pct(r['power_holm_worst'])} | {pct(r['power_unadjusted'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--out")
    ap.add_argument("--sd-h1", type=float, default=0.18)
    args = ap.parse_args()
    design = json.loads(Path(args.design).read_text(encoding="utf-8"))
    text = render(design, args.sd_h1)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
