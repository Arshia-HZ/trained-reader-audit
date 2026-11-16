"""Print the R17 health triggers of a GRPO run, and nothing if there are none.

    python scripts/22_health_check.py runs/reader_s1 [--verbose]

Run it against the run directory (or a copy of its logs from the hub). The exit code is
1 when a trigger fires, so a monitor can watch it.
"""

from __future__ import annotations

import argparse
import sys

from reader_audit.grpo_health import check_run, read_jsonl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    found = check_run(args.run_dir)
    for line in found:
        print("TRIGGER", line)
    if args.verbose and not found:
        n = len(
            {
                r["step"]
                for r in read_jsonl(
                    __import__("pathlib").Path(args.run_dir) / "steps.jsonl"
                )
            }
        )
        print(f"HEALTH ok after {n} steps")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
