# Suspicious Activity Detection in Authentically Distorted Surveillance Video

Classifies activities (walking, running, fighting, ...) and imaging conditions
(pristine, exposure, focus, exposure + focus) in the
[AD-SVD](https://ieee-dataport.org/open-access/authentically-distorted-surveillance-videos-dataset) surveillance dataset, using
foreground detection, object tracking, and handcrafted trajectory features
with classical machine-learning classifiers.

```
my project root
├── source/          code (package `activity_detection`) and YAML configs
├── docs/            paper, bibliography, how-to-run guide
├── ss/              slides and videos
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python source/run.py --list-configs
python source/run.py --config group_a_classical --target activity --sample
```

The dataset is not included. See [docs/howToRun.md](docs/howToRun.md) for
where to put it and for the full list of experiments.

## Review experiments (ablation study)

Everything the SACAIR 2026 reviewers asked for runs from one script. Smoke test first:

```bash
bash source/scripts/run_review_experiments.sh --smoke   # tiny subset, minutes
bash source/scripts/run_review_experiments.sh           # full run, hours
python source/scripts/collect_paper_figures.py          # copy plots to docs/images/
bash docs/compile.sh                                    # rebuild the paper
```

| Script                                    | What it does                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `source/scripts/factorial_analysis.py`    | 2x2x2 ablation: effect of detector, preprocessing and features with the others fixed                   |
| `source/scripts/mog2_threshold_sweep.py`  | `var_threshold` ablation, fixed vs adaptive MOG2, with common-subset macro-F1 and per-distortion yield |
| `source/scripts/sensitivity_sweep.py`     | one-at-a-time sensitivity of q, w, tau_r, tau_min                                                      |
| `source/scripts/compare_experiments.py`   | paired bootstrap differences and McNemar tests between two runs                                        |
| `source/scripts/collect_paper_figures.py` | copies the plots to `docs/images/` under the names `paper.tex` expects                                 |

Details, the experiment-to-reviewer mapping and known gaps: [docs/ablation_study.md](docs/ablation_study.md).
Run the tests with `cd source && python -m pytest tests` (install `imbalanced-learn` to run the resampling tests too).

## Status

Under revision after peer review. Planned changes are listed in
[docs/review_checklist.md](docs/review_checklist.md).
