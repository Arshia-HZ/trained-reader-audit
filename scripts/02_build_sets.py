"""Write the study's MATH sets (dev, primary, reserve) from the audit's frozen split.

    python scripts/02_build_sets.py --audit-problems <audit checkout>/data/problems \
        --out data/sets

The audit's problem files are rebuilt, not committed: in its repository run
``python scripts/04_freeze_problems.py --out data/problems`` (needs the ``data`` extra).
They are checked against the sha256 in the audit's committed split manifest. Only the id
lists and the manifest of this study are committed. An existing, different manifest is
never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from channel_audit import data as audit_data

from reader_audit import sets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit-problems", required=True)
    ap.add_argument("--out", default="data/sets")
    args = ap.parse_args()

    base = Path(args.audit_problems)
    confirm = audit_data.load_frozen(base, "confirm")  # checks the sha256
    derived = sets.derive_math_sets(confirm)
    manifest_bytes = (base / audit_data.MANIFEST_NAME).read_bytes()
    manifest = sets.write_sets(args.out, derived, manifest_bytes)
    for name, entry in manifest["sets"].items():
        print(f"{name:8s} n={entry['n']:5d} levels={entry['levels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
