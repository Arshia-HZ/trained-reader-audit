"""The study's problem sets from the audit's frozen MATH split (PREREG section 2).

Dev is the audit's first 600 confirmatory problems (frozen hash order). The primary test
set is the level-5 problems of the confirm split after them, so it excludes every
screen, pilot and dev problem. The remaining level 3 and 4 confirm problems are an
unused reserve. Membership depends on problem ids and levels only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from channel_audit import data as audit_data
from channel_audit.schema import Problem

N_DEV = audit_data.N_CONFIRMATORY  # 600
PRIMARY_LEVEL = 5
MANIFEST_NAME = "manifest.json"
MATH_SETS = ("dev", "primary", "reserve")


def derive_math_sets(
    confirm: Sequence[Problem], n_dev: int = N_DEV
) -> dict[str, list[Problem]]:
    """Split the confirm problems (already in frozen hash order) into the study sets."""
    if len(confirm) <= n_dev:
        raise ValueError(
            f"the confirm split has {len(confirm)} problems, need > {n_dev}"
        )
    rest = list(confirm[n_dev:])
    return {
        "dev": list(confirm[:n_dev]),
        "primary": [p for p in rest if p.level == PRIMARY_LEVEL],
        "reserve": [p for p in rest if p.level != PRIMARY_LEVEL],
    }


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def build_manifest(
    sets: Mapping[str, Sequence[Problem]], audit_manifest: bytes
) -> dict:
    """Counts, level mix and the sha256 of each id list, tied to the audit's split."""
    manifest: dict = {
        "audit_split_manifest_sha256": hashlib.sha256(audit_manifest).hexdigest(),
        "n_dev": N_DEV,
        "sets": {},
    }
    for name, problems in sets.items():
        ids = [p.problem_id for p in problems]
        levels: dict[str, int] = {}
        for p in problems:
            levels[str(p.level)] = levels.get(str(p.level), 0) + 1
        manifest["sets"][name] = {
            "n": len(ids),
            "levels": dict(sorted(levels.items())),
            "ids_sha256": ids_sha256(ids),
        }
    return manifest


def write_sets(
    out_dir: str | Path, sets: Mapping[str, Sequence[Problem]], audit_manifest: bytes
) -> dict:
    """Write id lists and the manifest; refuse to change an existing freeze."""
    out = Path(out_dir)
    manifest = build_manifest(sets, audit_manifest)
    path = out / MANIFEST_NAME
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError(
                f"{path} is already written with different content; the sets are fixed"
            )
    out.mkdir(parents=True, exist_ok=True)
    for name, problems in sets.items():
        ids_path = out / f"{name}.ids.json"
        if not ids_path.exists():
            ids = [p.problem_id for p in problems]
            ids_path.write_text(json.dumps(ids, indent=0) + "\n", encoding="utf-8")
    if not path.exists():
        path.write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def read_ids(sets_dir: str | Path, name: str) -> list[str]:
    """One set's ids, checked against the manifest."""
    base = Path(sets_dir)
    manifest = json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
    ids = json.loads((base / f"{name}.ids.json").read_text(encoding="utf-8"))
    if ids_sha256(ids) != manifest["sets"][name]["ids_sha256"]:
        raise ValueError(f"{name}.ids.json does not match the manifest")
    return ids


def load_math_set(
    name: str, sets_dir: str | Path, audit_problems_dir: str | Path
) -> list[Problem]:
    """The problems of one MATH set, from the audit's frozen (sha-checked) files."""
    wanted = read_ids(sets_dir, name)
    pool = {
        p.problem_id: p for p in audit_data.load_frozen(audit_problems_dir, "confirm")
    }
    missing = [i for i in wanted if i not in pool]
    if missing:
        raise ValueError(
            f"{len(missing)} ids of {name} are not in the audit's confirm split"
        )
    return [pool[i] for i in wanted]
