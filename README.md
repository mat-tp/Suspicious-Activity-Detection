# Suspicious Activity Detection in Authentically Distorted Surveillance Video

Classifies **activities** (walking, running, fighting, robbery, ...) and **imaging conditions**
(pristine, exposure, focus, exposure + focus) in the
[AD-SVD dataset](https://ieee-dataport.org/open-access/authentically-distorted-surveillance-videos-dataset)
of authentically distorted surveillance videos. The pipeline uses foreground detection, object
tracking and handcrafted trajectory features with classical machine-learning classifiers, and
asks whether a confidence-driven **adaptive MOG2** detector helps under real blur and exposure
problems.

## Contents

1. [Pipeline](#pipeline)
2. [Quick start](#quick-start)
3. [Detectors and why](#detectors-and-why)
4. [Preprocessing](#preprocessing)
5. [Tracking and why](#tracking-and-why)
6. [Features](#features)
7. [Classifiers and evaluation](#classifiers-and-evaluation)
8. [Experiments](#experiments)
9. [Scripts](#scripts)
10. [What `build.sh` runs](#what-buildsh-runs)
11. [Project layout](#project-layout)
12. [Tests and status](#tests-and-status)

## Pipeline

```
video -> preprocessing -> foreground detection -> tracking -> per-track features
      -> pooled per video (mean / max / std) -> classifier -> cross-validated scores
```

Every stage is a plug-in chosen in a YAML config (`source/configs/`), so a change of detector,
tracker, feature set or classifier is a one-line edit and never a code change.

## Audit and publication notes

* [docs/notes/AUDIT_REPORT.md](docs/notes/AUDIT_REPORT.md) - defects found and fixed before submission, and what must be re-run.
* [docs/notes/PUBLICATION_STRATEGY.md](docs/notes/PUBLICATION_STRATEGY.md) - novelty, the runs a stronger paper needs, venue fit.
* `python scripts/check_docstrings.py` - docstring coverage (enforced by a test).

## Quick start

Put the dataset in `data/` (layout in [docs/notes/howToRun.md](docs/notes/howToRun.md)), then:

```bash
./scripts/build.sh --smoke                  # Linux/macOS: every stage on a tiny sample (minutes), to check the setup
./scripts/build.sh --workers auto --resume  # full run (many hours); --resume continues after an interruption
scripts\build.bat --smoke                   # Windows equivalent
```

`build.sh` / `build.bat` create the virtual environment `sdr-env`, activate it, check and install
the libraries, run all training and testing, and write a log for every step under
`docs/logs/<time>/` (with `pip freeze`, git commit and config checksums). A failing
step does not stop the batch; every failure is listed at the end and the exit code is 1.
`./scripts/build.sh --help` lists all stages and options; stages can run alone, e.g. `./scripts/build.sh env tests`.

### Using several cores

* **Videos** are processed one per worker process: `--workers N` (or `--workers auto` for all cores
  but one, or `experiment.n_workers` in a config). Each worker limits OpenCV and the math libraries to one thread, builds
  its pipeline once, and the features are sorted by video name afterwards, so the result does not depend on how many workers ran.
  This is the slow part of a run, so it is where the cores matter. The default is 1.
* **Cross-validation** folds can be trained at the same time with `--set training.n_jobs=N` (or `build.sh --cv-jobs N`; `-1` = all cores).
* The pure-NumPy forest (`random_forest_custom`) takes `n_jobs` to grow its trees in parallel and gives the same forest for any value.

One experiment by hand:

```bash
source sdr-env/bin/activate                     # Windows: sdr-env\Scripts\activate
python source/run.py --list-configs
python source/run.py --config group_a_classical --target both --sample
python source/run.py --config group_i_adaptive_mog2 --set detection.adaptive_mog2.quality_threshold=0.5
python source/run.py --config group_i_adaptive_mog2 --features-csv experiments/<run>/video_features.csv
```

`--set key.path=value` overrides any config value, `--workers N` sets the number of
video-processing processes, and `--features-csv` re-trains classifiers from cached features
without reprocessing the videos.

## Detectors and why

The detector turns each frame into bounding boxes of moving objects. The camera is static in
surveillance footage, so background subtraction is the natural fit: it needs no training data and
no labels, and its output is directly usable by the tracker.

| Detector (`detection.type`)                     | What it is                                                                                                                                                                                                                                                     | Why it is here                                                                                                                                                                                               |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mog2`                                          | Gaussian-mixture background model (Zivkovic 2004; Zivkovic and van der Heijden 2006), fixed variance threshold                                                                                                                                                 | The classical baseline: fast, widely available in OpenCV, and the reference every other detector is compared with                                                                                            |
| `adaptive_mog2`                                 | MOG2 plus a quality check. It scores the mask (coverage, fragmentation, blob size) and the recent track quality; when the combined score falls below `q` it looks at the same frame again with a lower threshold and keeps the result only if it scores better | Blur and exposure change make a fixed threshold lose objects. The retry runs at learning rate 0, so the background model is updated once per frame, exactly as in plain MOG2. This is the method under study |
| `vibe`, `pbas`, `subsense`                      | Other adaptive background models: ViBe (sample-based), PBAS (per-pixel adaptive thresholds), SuBSENSE (colour + local binary patterns with feedback)                                                                                                           | Tests whether any gain is specific to the confidence-driven variant or just a generic effect of adaptivity. ViBe and PBAS are our own NumPy implementations; SuBSENSE needs the optional `pybgs` package     |
| `dense_flow`, `frame_diff`, `cascaded_fallback` | Optical-flow detector, frame differencing, and a MOG2 -> flow -> differencing cascade                                                                                                                                                                          | Detectors with no background model, to check robustness to illumination and blur independently of MOG2 (Group H)                                                                                             |
| `ground_truth`                                  | The annotated boxes of AD-SVD                                                                                                                                                                                                                                  | An upper bound: the gap to a real detector is the cost of imperfect detection (Group E). Also the reference for validating the mask-quality score                                                            |

The adaptive parameters are `q` (adapt below this score), `w` (weight of mask vs track quality),
`tau_r` (threshold under full relaxation), `tau_min` (floor), the EMA smoothing of the track
quality, and the minimum track length that counts as reliable. They live once in
`source/configs/base.yaml`. Because the quality score is never negative, the relaxed threshold
never drops below `tau_r`: `tau_min <= tau_r` and `tau_r >= tau` change nothing.

## Preprocessing

| Type               | What it does                                                                                                                  | Why                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `standard`         | Resize to 640x480, grey, Gaussian blur                                                                                        | Plain baseline                                                                                                           |
| `distortion_aware` | Measures sharpness and brightness per frame; sharpens blurred frames (unsharp mask) and equalises poorly exposed ones (CLAHE) | AD-SVD contains real blur and exposure faults; correcting them before detection gives the background model cleaner input |
| `passthrough`      | Nothing                                                                                                                       | Ablation                                                                                                                 |

## Tracking and why

| Tracker (`tracking.type`) | How it matches                                                                                            | Why                                                                                                                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iou_hungarian` (default) | Overlap (IoU >= 0.3) between existing tracks and new boxes, solved optimally with the Hungarian algorithm | The detector gives blob boxes with no appearance model, and in a static scene overlap between consecutive frames is a strong cue. The method is simple, deterministic, has almost no parameters and is cheap enough to run on thousands of videos |
| `centroid`                | Nearest centre point                                                                                      | Simplest reference; the legacy tracker                                                                                                                                                                                                            |
| `sort`                    | Kalman prediction plus IoU matching                                                                       | A standard stronger baseline                                                                                                                                                                                                                      |
| `tfcr`                    | A tracker inspired by TFCR (Yuan et al. 2020), not the original network                                   | Tests whether a tracker designed for target/background imbalance improves recognition (Group J)                                                                                                                                                   |

Group F compares centroid, IoU-Hungarian and SORT at several minimum trajectory lengths, so the
default is checked by an experiment and not just assumed. Tracks shorter than
`min_trajectory_len` are dropped before feature extraction when `tracking.filter_short_tracks`
is true (`group_i_minlen30_filtered`); by default every track with at least two points is used.

## Features

Per-track descriptors are pooled over each video's tracks by mean, max and std, then joined with
video-level features.

| Family (`features.modules`)                  | Captures                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `trajectory`                                 | Speed, straightness, direction, duration, bounding-box size                                            |
| `kinematics`                                 | Acceleration and jerk                                                                                  |
| `shape`                                      | Curvature, turning rate, convex hull, compactness                                                      |
| `circular_motion`                            | Circular statistics of heading (correct near 0/360 degrees)                                            |
| `wavelet`                                    | Wavelet energy per scale of the trajectory signal (`db4`, level 3)                                     |
| `interaction`                                | Pairwise distance and relative speed, aligned on shared frames                                         |
| multi-family (`features.enable_multifamily`) | Image quality (sharpness, brightness), global optical flow, appearance, track counts and file metadata |

Sizes: 54 (trajectory only), 147 (six motion families) and 221 (everything). The multi-family
set contains blur and exposure measures that need no track, so imaging-condition results on it
must be read together with the feature-family ablations (`ablation_7`, `ablation_8`). Every
feature, parameter and aggregation is listed in [docs/manuscript/feature_appendix.tex](docs/manuscript/feature_appendix.tex).

To identify a column of `video_features.csv` or of an importance file, look it up in
`source/activity_detection/features/catalogue.py` (`describe("std_dwt_entropy")` gives its family and meaning).
Every run also writes `feature_catalogue_<target>.csv` with one row per feature of the matrix,
`python scripts/list_features.py` writes the table of all 221 features, and a test fails if the code
produces a feature that the catalogue does not describe.

## Classifiers and evaluation

Seven classifiers run in every configuration: random forest (scikit-learn and a from-scratch
version), SVM (RBF), KNN, LightGBM, XGBoost and an MLP on the same handcrafted features.
Evaluation is stratified 5-fold cross-validation on out-of-fold predictions, with **macro-F1**
as the main metric (the classes are imbalanced), plus balanced accuracy, per-class results and
bootstrap 95% confidence intervals. Re-encoded copies of one recording must not sit in both
train and test, so `audit_duplicates.py` builds groups and cross-validation then keeps each group
in one fold, with the bootstrap resampling whole groups. Class imbalance is handled by class
weights or resampling inside the training folds only.

## Experiments

| Config                                                                | What it tests                                                                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `group_a_classical`                                                   | Baseline: MOG2, standard preprocessing, trajectory features                                                                |
| `group_b_distortion_aware`                                            | Adds distortion-aware preprocessing                                                                                        |
| `group_c_neural`, `group_d_hybrid`                                    | CNN / CNN-LSTM / 3D-CNN baselines and handcrafted + neural fusion (need PyTorch)                                           |
| `group_e_gt_upper_bound`                                              | Ground-truth boxes: the performance ceiling                                                                                |
| `group_f_tracking_ablation`                                           | Centroid vs IoU-Hungarian vs SORT                                                                                          |
| `group_g_feature_comparison`                                          | Blob trajectory features vs dense trajectories                                                                             |
| `group_h_*`                                                           | Detectors without a background model (cascade, optical flow)                                                               |
| `group_i_adaptive_mog2`                                               | Adaptive MOG2 + distortion-aware preprocessing + all features (the proposed system)                                        |
| `group_i_minlen30_filtered`                                           | The same with tracks under 30 frames dropped                                                                               |
| `group_j_tfcr_tracker`                                                | Replaces the tracker with TFCR                                                                                             |
| `ablation_1` ... `ablation_6`                                         | With `group_a` and `group_i`: the 2x2x2 design (detector x preprocessing x features); each cell differs in one factor only |
| `ablation_0`, `ablation_7`, `ablation_8`                              | Without metadata / without image-quality features / motion features only                                                   |
| `detector_vibe`, `detector_pbas`, `detector_subsense` (+ `_fullfeat`) | Other adaptive background models on trajectory-only and full features                                                      |

## Scripts

| Script                                        | What it does                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `source/run.py`                               | Runs one config (or all groups / ablations / detectors); `--set`, `--workers`, `--features-csv`                                       |
| `scripts/audit_configs.py`             | Proves the eight factorial cells differ only in detector, preprocessing and features                                                  |
| `scripts/audit_duplicates.py`          | Finds re-encoded copies of one recording, writes `data/groups.csv`, a review listing and a metadata file with thresholds and checksum |
| `scripts/factorial_analysis.py`        | 2x2x2 analysis: effect of each factor with the others fixed, with bootstrap intervals                                                 |
| `scripts/mog2_threshold_sweep.py`      | `var_threshold` ablation, fixed vs adaptive MOG2, common-subset macro-F1 and per-distortion yield                                     |
| `scripts/parameter_ablation.py`        | One-at-a-time ablation of every selected parameter: preprocessing, detector, adaptive MOG2, tracker, features, classifiers, imbalance |
| `scripts/validate_mask_quality.py`     | Spearman correlation of the mask-quality score with agreement to ground-truth boxes, bootstrap over videos                            |
| `scripts/compare_experiments.py`       | Paired bootstrap differences and McNemar tests between two runs                                                                       |
| `scripts/feature_family_importance.py` | Share of classifier importance carried by each feature family                                                                         |
| `scripts/verify_outputs.py`            | Checks that every required result exists, no classifier failed, and the factorial cells used the same videos                          |
| `scripts/check_dataset.py`             | Checks that the dataset is in place                                                                                                   |
| `scripts/collect_paper_figures.py`     | Copies the plots to `docs/manuscript/images/` under the names `paper.tex` expects                                                     |
| `scripts/list_features.py`             | Writes a table that says what every feature column is (family and description); `--csv` checks a `video_features.csv`                  |
| `scripts/check_docstrings.py`          | Reports the docstring coverage of the package                                                                                         |

Details of the experiment design and known gaps: [docs/notes/ablation_study.md](docs/notes/ablation_study.md).

## What `build.sh` runs

| Stage         | Purpose                                                            |
| ------------- | ------------------------------------------------------------------ |
| `env`         | Create and activate `sdr-env`, check and install libraries         |
| `record`      | Save `pip freeze`, git commit and config checksums                 |
| `dataset`     | Check the dataset paths                                            |
| `tests`       | Unit tests on synthetic data                                       |
| `configs`     | Audit the factorial configs                                        |
| `duplicates`  | Build the recording groups used by cross-validation                |
| `factorial`   | Train and evaluate the eight cells                                 |
| `detectors`   | ViBe, PBAS, SuBSENSE                                               |
| `features`    | Feature-family ablations and the minimum-track-length filter       |
| `groups`      | Remaining groups (B, E, F, G, H, J, and C, D with PyTorch)         |
| `threshold`   | Fixed vs adaptive MOG2 over a range of thresholds                  |
| `ablation`    | Every selected parameter, one at a time                            |
| `maskquality` | Validate the mask-quality score                                    |
| `verify`      | Inspect the outputs for missing or failed runs                     |
| `analysis`    | Factorial effects, pairwise comparisons, feature-family importance |
| `figures`     | Copy the paper figures                                             |

`--smoke` runs the same stages on a small sample and writes to `experiments_smoke/`, so test
runs never mix with real results.

## Project layout

```
.
├── source/                 the main code
│   ├── run.py              entry point
│   ├── activity_detection/ the package (detection, tracking, features, models, evaluation)
│   ├── configs/            YAML experiment configs (base.yaml holds every default)
│   └── tests/              unit tests
├── scripts/                build.sh, build.bat and the analysis and ablation scripts
├── docs/
│   ├── manuscript/         the paper, bibliography and feature appendix
│   ├── notes/              how to run, audit report, publication strategy, reviewer notes
│   └── logs/               logs of the training runs
├── ss/                     the presentation video and slides
├── requirements.txt
└── README.md
```

The results of a run go to `experiments/` (not tracked by git, see `.gitignore`).

## Tests and status

```bash
python -m pytest source/tests        # install imbalanced-learn to run the resampling tests too
```
