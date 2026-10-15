"""verify_outputs flags missing files, failed models and mismatched video universes."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_outputs.py"
spec = importlib.util.spec_from_file_location("verify_outputs", SCRIPT)
verify_outputs = importlib.util.module_from_spec(spec)
sys.modules["verify_outputs"] = verify_outputs
spec.loader.exec_module(verify_outputs)


def make_run(root, experiment, videos, results=None, with_features=True):
    run = root / experiment / "run_001"
    run.mkdir(parents=True)
    (run / "results_activity.json").write_text(json.dumps(results or {"knn": {"accuracy": 0.5}}))
    (run / "predictions_activity.csv").write_text("video,true\n")
    pd.DataFrame({"video": videos}).to_csv(run / "videos_attempted.csv", index=False)
    (run / "run_meta.json").write_text(json.dumps({"n_videos": len(videos), "git_commit": "abc"}))
    if with_features:
        (run / "video_features.csv").write_text("video_name\n")
    return run


def test_clean_run_passes(tmp_path):
    make_run(tmp_path, "group_a_classical", ["a", "b"])
    report = verify_outputs.verify(["group_a_classical"], ["activity"], tmp_path, [])
    assert report["problems"] == []


def test_failed_model_and_missing_file_are_reported(tmp_path):
    make_run(tmp_path, "group_a_classical", ["a"], results={"svm_rbf": {"error": "boom"}},
             with_features=False)
    problems = verify_outputs.verify(["group_a_classical"], ["activity"], tmp_path, [])["problems"]
    assert any("svm_rbf failed" in p for p in problems)
    assert any("missing video_features.csv" in p for p in problems)


def test_different_video_universes_between_cells_are_reported(tmp_path):
    make_run(tmp_path, "group_a_classical", ["a", "b"])
    make_run(tmp_path, "ablation_1_detector_only", ["a"])
    names = ["group_a_classical", "ablation_1_detector_only"]
    problems = verify_outputs.verify(names, ["activity"], tmp_path, names)["problems"]
    assert any("different video sets" in p for p in problems)
