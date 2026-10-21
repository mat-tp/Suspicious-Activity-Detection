"""Checks that the eight cells of the factorial design differ only in the intended factors.

The factorial analysis is only valid if nothing else changes between cells. Each cell's config is
resolved (including ``extends``), flattened and checked: outside the three factors every setting must be
the same in all cells, and the settings that define a factor level must be the same in all cells that
share that level. The resolved configs are saved, so the settings behind the paper's numbers are on
record. The exit status is 1 when a check fails.

    python scripts/audit_configs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import bootstrap  # noqa: F401  (puts source/ on the import path)
from activity_detection.config import load_config
from activity_detection.paths import EXPERIMENTS_DIR

# (detector, preprocessing, features) -> config name
CELLS: Dict[Tuple[int, int, int], str] = {
    (0, 0, 0): "group_a_classical",
    (1, 0, 0): "ablation_1_detector_only",
    (0, 1, 0): "ablation_2_preprocessing_only",
    (0, 0, 1): "ablation_3_features_only",
    (1, 1, 0): "ablation_5_detector_preprocessing",
    (1, 0, 1): "ablation_6_detector_features",
    (0, 1, 1): "ablation_4_full_pipeline_fixed_mog2",
    (1, 1, 1): "group_i_adaptive_mog2",
}
# These are the three two-level factors of the factorial ablation.
FACTORS = ("detector", "preprocessing", "features")

# Keys (or key prefixes ending in '.') that MAY differ because of each factor.
FACTOR_KEYS = {
    "detector": ("detection.type", "detection.adaptive_mog2."),
    "preprocessing": ("preprocessing.type",),
    "features": ("features.modules", "features.enable_multifamily"),
}
# Descriptive metadata that is expected to differ between configs.
IGNORED_KEYS = ("experiment.name", "experiment.description", "experiment.group")


def flatten(node, prefix="") -> Dict[str, object]:
    """{'a.b.c': value}; lists are kept whole as leaf values."""
    if isinstance(node, dict):
        flat = {}
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: node}


def belongs_to(key: str, factor: str) -> bool:
    return any(key == rule or (rule.endswith(".") and key.startswith(rule))
               for rule in FACTOR_KEYS[factor])


def differing_keys(a: Dict, b: Dict) -> List[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k, "<absent>") != b.get(k, "<absent>")
                  and k not in IGNORED_KEYS)


def audit(resolved: Dict[Tuple[int, int, int], Dict]) -> List[str]:
    """Return a list of violation messages (empty when the design is clean)."""
    flat = {cell: flatten(cfg) for cell, cfg in resolved.items()}
    violations: List[str] = []

    # Check 1: any key outside the three factors must be identical in all cells.
    reference_cell = (0, 0, 0)
    for cell, settings in flat.items():
        for key in differing_keys(flat[reference_cell], settings):
            if not any(belongs_to(key, factor) for factor in FACTORS):
                violations.append(f"{CELLS[cell]} differs from {CELLS[reference_cell]} "
                                  f"in '{key}' (not part of any factor): "
                                  f"{flat[reference_cell].get(key, '<absent>')!r} vs "
                                  f"{settings.get(key, '<absent>')!r}")

    # Check 2: a factor's own settings must agree among cells at the same level of it,
    # and must differ between levels (otherwise the factor does nothing).
    for index, factor in enumerate(FACTORS):
        for level in (0, 1):
            members = [c for c in flat if c[index] == level]
            first = members[0]
            for other in members[1:]:
                for key in differing_keys(flat[first], flat[other]):
                    if belongs_to(key, factor):
                        violations.append(f"{factor} level {level}: {CELLS[first]} and "
                                          f"{CELLS[other]} disagree on '{key}'")
        low = flat[next(c for c in flat if c[index] == 0)]
        high = flat[next(c for c in flat if c[index] == 1)]
        if not any(belongs_to(k, factor) for k in differing_keys(low, high)):
            violations.append(f"{factor}: levels 0 and 1 have identical settings (factor has no effect)")
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(EXPERIMENTS_DIR / "config_audit"),
                        help="folder for the resolved configs and the audit report")
    args = parser.parse_args(argv)

    resolved = {cell: load_config(name).to_dict() for cell, name in CELLS.items()}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell, name in CELLS.items():
        (out_dir / f"{name}.resolved.yaml").write_text(yaml.safe_dump(resolved[cell], sort_keys=True))

    violations = audit(resolved)
    flat = {cell: flatten(cfg) for cell, cfg in resolved.items()}
    print("Cell (detector, preprocessing, features) -> config / detector / preprocessing / modules")
    for cell, name in sorted(CELLS.items()):
        s = flat[cell]
        print(f"  {cell}  {name:40s} {s['detection.type']:14s} {s['preprocessing.type']:16s} "
              f"{len(s['features.modules'])} modules, multifamily={s.get('features.enable_multifamily', False)}")

    report = {"cells": {str(c): n for c, n in CELLS.items()}, "violations": violations}
    (out_dir / "config_audit.json").write_text(json.dumps(report, indent=2))
    if violations:
        print(f"\nFAILED: {len(violations)} violation(s)")
        for message in violations:
            print(f"  - {message}")
        return 1
    print(f"\nOK: the eight cells differ only in detector, preprocessing and features.\n"
          f"Resolved configs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
