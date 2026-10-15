"""The factorial audit passes on the shipped configs and catches a hidden confound."""
import copy
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_configs.py"
spec = importlib.util.spec_from_file_location("audit_configs", SCRIPT)
audit_configs = importlib.util.module_from_spec(spec)
sys.modules["audit_configs"] = audit_configs
spec.loader.exec_module(audit_configs)


def resolved_cells():
    return {cell: audit_configs.load_config(name).to_dict()
            for cell, name in audit_configs.CELLS.items()}


def test_shipped_factorial_cells_are_clean():
    assert audit_configs.audit(resolved_cells()) == []


def test_detects_a_change_outside_the_three_factors():
    cells = resolved_cells()
    cells[(1, 0, 0)]["tracking"]["iou_hungarian"]["iou_threshold"] = 0.5
    messages = audit_configs.audit(cells)
    assert any("iou_threshold" in m for m in messages)


def test_detects_inconsistent_adaptive_parameters_between_adaptive_cells():
    cells = resolved_cells()
    cells[(1, 1, 1)]["detection"]["adaptive_mog2"]["quality_threshold"] = 0.6
    assert any("quality_threshold" in m for m in audit_configs.audit(cells))


def test_detects_a_factor_with_no_effect():
    cells = copy.deepcopy(resolved_cells())
    for cell in cells:
        cells[cell]["preprocessing"]["type"] = "standard"
    assert any("preprocessing" in m and "no effect" in m for m in audit_configs.audit(cells))


def test_experiment_folder_names_match_config_names():
    """factorial_analysis.py and compare_experiments.py look runs up by config name, so the
    experiment folder (experiment.name) must be the same string."""
    for name in audit_configs.CELLS.values():
        assert audit_configs.load_config(name).experiment.name == name
