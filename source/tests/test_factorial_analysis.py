"""Tests for the 2x2x2 factorial ablation (synthetic runs, no videos needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("factorial_analysis", SCRIPTS / "factorial_analysis.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)

CLASSES = ["a", "b", "c"]


def make_experiments(root, detector_gain=0.2, n=240, seed=0):
    """Accuracy = base + detector_gain * D (only the detector matters); fixed MOG2 loses some videos."""
    rng = np.random.default_rng(seed)
    videos = [f"v{i}" for i in range(n)]
    true = rng.choice(CLASSES, n)
    for cell, name in fa.CELLS.items():
        d = cell[0]
        p_correct = 0.5 + detector_gain * d
        pred = [t if rng.random() < p_correct else rng.choice([c for c in CLASSES if c != t]) for t in true]
        ok = np.ones(n, bool)
        if d == 0:
            ok[: n // 10] = False                    # fixed MOG2 yields fewer videos
        run = root / name / "run_001"
        run.mkdir(parents=True)
        pd.DataFrame({"video": videos, "true": true, "lightgbm": pred})[ok].to_csv(run / "predictions_activity.csv", index=False)
        pd.DataFrame({"video": videos, "activity": true, "distortion": "x",
                      "status": np.where(ok, "ok", "failed: no_tracks")}).to_csv(run / "videos_attempted.csv", index=False)


def test_effects_recover_a_pure_detector_effect():
    cells = {c: 0.5 + 0.2 * c[0] for c in fa.CELLS}
    e = fa.effects(cells)
    assert e["main: detector"] == pytest.approx(0.2)
    assert e["main: preprocessing"] == pytest.approx(0) and e["main: features"] == pytest.approx(0)
    assert all(v == pytest.approx(0) for k, v in e.items() if k.startswith("interaction"))
    assert sum(k.startswith("simple: detector") for k in e) == 4


def test_effects_detect_an_interaction():
    cells = {c: 0.5 + 0.2 * (c[0] and c[1]) for c in fa.CELLS}
    e = fa.effects(cells)
    assert e["interaction: detector x preprocessing"] == pytest.approx(0.2)
    assert e["interaction: detector x features"] == pytest.approx(0)


@pytest.mark.parametrize("universe", ["common", "all"])
def test_end_to_end_detector_effect_is_significant(tmp_path, universe):
    make_experiments(tmp_path)
    out = tmp_path / "out"
    df = fa.main(["--experiments-dir", str(tmp_path), "--universe", universe, "--n-boot", "200",
                  "--model", "lightgbm", "--out", str(out)])
    main = df[df.kind == "main"].set_index("name")
    assert main.loc["detector", "estimate"] > 0.1 and bool(main.loc["detector", "excludes_0"])
    assert not bool(main.loc["preprocessing", "excludes_0"])
    assert len(df[df.kind == "cell"]) == 8
    assert (out / f"factorial_{universe}_lightgbm.csv").exists()
    assert (out / f"factorial_{universe}_lightgbm.png").exists()


def test_all_universe_counts_lost_videos_against_fixed_mog2(tmp_path):
    make_experiments(tmp_path, detector_gain=0.0)
    df = fa.main(["--experiments-dir", str(tmp_path), "--universe", "all", "--n-boot", "100",
                  "--out", str(tmp_path / "o")])
    assert df[df.kind == "main"].set_index("name").loc["detector", "estimate"] > 0.03
