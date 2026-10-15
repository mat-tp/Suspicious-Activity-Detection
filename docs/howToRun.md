# How to run

## 1. Install

Python 3.10 or newer. The build scripts do this for you: they create a virtual
environment called `sdr-env`, activate it, and install whatever is missing.

```bash
./build.sh env          # Linux / macOS
build.bat env           # Windows
```

Or by hand: `python -m venv sdr-env`, activate it, then `pip install -r requirements.txt`.
To run everything (environment, tests, all experiments, analysis, figures) use `./build.sh`
(see the README); `./build.sh --smoke` first checks the setup on a tiny sample.

Only the first block of `requirements.txt` is required. Missing optional
packages (LightGBM, XGBoost, PyTorch, ...) just disable the parts that need them.

## 2. Add the dataset

Download AD-SVD and arrange it like this, in the project root:

```
data/
├── datasetInfo.xlsx
└── surveillanceVideosDataset/
    ├── surveillanceVideos/        the video files (.mp4 in the IEEE DataPort download)
    └── surveillanceVideosGT/      the *_gt.txt annotation files
```

These paths come from `source/configs/base.yaml` and can be changed there.

## 3. Run an experiment

Always run from the project root.

```bash
python source/run.py --list-configs        # what experiments exist
python source/run.py --list-components     # detectors, trackers, features, models

python source/run.py --config group_a_classical --target activity
python source/run.py --config group_i_adaptive_mog2 --target both
python source/run.py --all-groups --target activity
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--target activity / distortion / both` | what to classify |
| `--sample --n 5` | quick run on 5 videos per class |
| `--detection-source ground_truth` | use the annotated boxes instead of MOG2 |
| `--diagnose` | extract features only, skip training |
| `--visualise` | write FiftyOne visualisations |

Results are written to `experiments/<run name>/` in the project root.

## 4. Experiment groups

| Config | What it tests |
|---|---|
| `group_a_classical` | baseline: MOG2 + IoU-Hungarian tracking + trajectory features |
| `group_b_distortion_aware` | adds distortion-aware preprocessing |
| `group_c_neural` | CNN, CNN-LSTM, 3D-CNN baselines (needs PyTorch) |
| `group_d_hybrid` | handcrafted + neural features |
| `group_e_gt_upper_bound` | ground-truth boxes (upper bound) |
| `group_f_tracking_ablation` | centroid vs IoU-Hungarian vs SORT |
| `group_g_feature_comparison` | tracked vs dense trajectories |
| `group_h_*` | optical-flow and cascaded detectors |
| `group_i_adaptive_mog2` | adaptive MOG2 (proposed) |
| `group_j_tfcr_tracker` | lightweight TFCR-style tracker |

## 5. Ablations

Each ablation changes exactly one thing relative to its reference, so a difference in results can be attributed to that one thing.

| Config | Reference | Only change |
|---|---|---|
| `ablation_1_detector_only` | `group_a_classical` | fixed MOG2 -> adaptive MOG2 |
| `ablation_2_preprocessing_only` | `group_a_classical` | standard -> distortion-aware preprocessing |
| `ablation_3_features_only` | `group_a_classical` | trajectory-only -> full feature set |
| `ablation_4_full_pipeline_fixed_mog2` | `group_i_adaptive_mog2` | adaptive MOG2 -> fixed MOG2 |

```bash
python source/run.py --ablations --target activity
```

### Detector comparison (ViBe, PBAS, SuBSENSE)

A natural question is whether the gains come from adaptive background subtraction in general or
from this particular adaptive MOG2. These configs swap in other adaptive methods, changing
only the background model (same preprocessing, tracker, mask cleaning and features as Group A):

| Config | Method | Implementation |
|---|---|---|
| `detector_vibe` | ViBe (Barnich and Van Droogenbroeck, 2011) | NumPy, written for this project from the paper, paper-default parameters |
| `detector_pbas` | PBAS (Hofmann et al., 2012) | NumPy, written for this project from the paper, paper-default parameters |
| `detector_subsense` | SuBSENSE (St-Charles et al., 2015) | BGSLibrary reference code through `pybgs` (optional) |

```bash
python source/run.py --detectors --target activity
```

References to compare against: `group_a_classical` (fixed MOG2) and `ablation_1_detector_only`
(adaptive MOG2).

Things to know:
- OpenCV does not ship any of these three. ViBe and PBAS are re-implementations, so say so in
  the paper. They are covered by tests but have not been checked against the authors' code.
- SuBSENSE needs `pip install pybgs`, which compiles C++ code and may fail on your system. If
  it is missing, `detector_subsense` prints "SKIPPING" and the other runs carry on.
- PBAS is about 3.4 times slower than MOG2 per video (ViBe about 1.3 times). Try
  `--sample --n 10` before a full run, and report the running times too.

The adaptive detector's parameters (q, w, tau_r, tau_min) are in the `adaptive_mog2:` block of `group_i_adaptive_mog2.yaml`.

## 6. Evaluation and class imbalance

Every model is scored with stratified k-fold cross-validation (`training:` in the config).
Each run writes, to `experiments/<name>/<run>/`:

| File | Contents |
|---|---|
| `summary_<target>.csv` | one row per model: accuracy, balanced accuracy, macro-F1 with 95% intervals, MCC, AUROC |
| `per_class_<target>.csv` | precision, recall, F1 and support for every class |
| `predictions_<target>.csv` | the out-of-fold prediction for every video, one column per model |
| `videos_attempted.csv` | every video the run tried, and whether it produced features |
| `results_<target>.json` | everything above plus fold-by-fold scores and confusion matrices |

Report **macro-F1 and balanced accuracy** alongside accuracy. Imaging-condition classes are
very uneven (160 pristine videos against about 1,450 for each other condition), so accuracy
alone can look good while the pristine class is ignored. Check the per-class table.

How the numbers are computed:
- **Fold scores** are the mean and standard deviation over the folds. Training sets overlap
  between folds, so these are not independent samples. Do not run t-tests on them.
- **Pooled scores** use the out-of-fold predictions, where every video is predicted once by a
  model that did not see it. The 95% intervals come from bootstrapping these videos.
- Set `training.cv_repeats: 3` (or more) to repeat the cross-validation with new shuffles. The
  spread between repeats is reported as `repeats`.

### Choosing how imbalance is handled

Edit the `imbalance:` block in your config (defaults are in `base.yaml`):

```yaml
imbalance:
  class_weight: balanced     # balanced | none
  resampling: smote          # see the table below
  sampling_ratio: 1.0        # 1.0 = fully balanced, 0.5 = minority up to half the majority
  k_neighbors: 5
  params: {}                 # e.g. {noise_std: 0.1} for gaussian_noise, {alpha: 0.4} for mixup
```

| `resampling` | What it does |
|---|---|
| `none` | leaves the data alone (use `class_weight` only) |
| `random_over` | duplicates minority samples |
| `smote`, `borderline_smote`, `adasyn` | synthetic minority samples between neighbours |
| `smote_tomek`, `smote_enn` | SMOTE followed by cleaning (ENN removes many samples) |
| `random_under` | drops majority samples |
| `gaussian_noise` | augmentation: minority samples copied with small noise added |
| `mixup` | augmentation: blends of two samples of the same class |

Resampling and augmentation happen **inside each training fold only**. The test fold is never
resampled, so scores stay honest. A `class_weight` written on a single model in the config
overrides the global setting for that model. KNN has no class weights. XGBoost's "balanced"
is applied as per-sample weights.

Run one config under different strategies from the command line by copying the config and
changing the `imbalance:` block, or with a one-line override file extending it.

These methods work on the handcrafted feature vectors. They do **not** augment the videos
themselves (flips, brightness changes, temporal crops). That would mean re-running detection
and tracking on the augmented videos, and is not implemented yet.

## 7. Comparing experiments

Use this instead of t-tests on fold scores. It compares two or more experiments video by video.

```bash
# differences are candidate minus reference
python source/scripts/compare_experiments.py group_a_classical ablation_1_detector_only

# several candidates against one reference (Holm-corrected p-values)
python source/scripts/compare_experiments.py group_a_classical \
    ablation_1_detector_only detector_vibe detector_pbas --model lightgbm --per-class

# two classifiers within one experiment
python source/scripts/compare_experiments.py group_i_adaptive_mog2:svm_rbf group_i_adaptive_mog2:lightgbm
```

Names are experiment folders in `experiments/`; the latest run is used. Runs made before this
feature existed have no `predictions_<target>.csv`, so run them again.

For each metric (accuracy, balanced accuracy, macro-F1) it reports both scores, the difference,
and a 95% paired-bootstrap interval. It also gives an exact McNemar test on accuracy and
per-class F1 differences. Say "the interval excludes zero", not "statistically significant".

**Two sets of videos are compared, and both matter:**

| Universe | Videos used | What it answers |
|---|---|---|
| `common` | videos both experiments made a prediction for | is the classifier better on the same easy videos? |
| `all` | every video both attempted; no prediction counts as a miss | is the whole pipeline better, including videos it loses? |

A detector that finds trajectories in more videos can look the same on `common` and far better
on `all`. Report both, and say which one a claim comes from. This is the fair way to handle the
baseline producing features for only 46.5% of videos against 99.7% for adaptive MOG2.

Cautions:
- Choose the classifier (`--model`) before looking at results. Picking the best classifier
  per experiment and then comparing them favours whichever experiment had more to choose from.
- The intervals show how much a difference depends on which videos were sampled. They do not
  include the randomness of retraining, and cross-validation folds share training data, so be
  careful with small differences. Repeating the cross-validation (`training.cv_repeats`) shows
  how stable one experiment's own score is.
- When comparing many experiments, the Holm adjustment is applied to the McNemar p-values only.
  The intervals are not adjusted.

## 8. Sensitivity analysis

```bash
python source/scripts/parameter_ablation.py --list                 # every ablated parameter and its values
python source/scripts/parameter_ablation.py --dry-run              # show the plan
python source/scripts/parameter_ablation.py --sample --n 5         # quick trial
python source/scripts/parameter_ablation.py --workers 8            # full ablation (hours)
python source/scripts/parameter_ablation.py --groups adaptive      # only q, w, tau_r, tau_min, ...
python source/scripts/parameter_ablation.py --groups classifier imbalance   # fast: reuses cached features
python source/scripts/parameter_ablation.py --resume               # continue after a stop
```

Varies q, w, tau_r and tau_min one at a time on `ablation_1_detector_only` (the adaptive
detector with baseline features), scoring each setting with one classifier. The default
setting is run once and shared by all four curves. Output is
`experiments/sensitivity_<config>_<target>/sensitivity_results.csv` and `sensitivity.png`,
written after each setting so an interrupted run can be resumed.

## 9. Tests

```bash
cd source && python -m pytest tests
```

## 10. Build the paper and slides

```bash
bash docs/compile.sh     # docs/paper.pdf
bash ss/compile.sh       # slides, output in ss/output/
```

Both need a LaTeX installation with `latexmk`.

## 11. Code layout

```
source/
├── run.py
├── configs/                  YAML experiment configs (group_*.yaml, ablation_*.yaml)
├── scripts/                  parameter_ablation.py, compare_experiments.py, audit_configs.py, ...
├── tests/                    pytest tests
└── activity_detection/
    ├── cli.py                command-line interface
    ├── config.py             config loading
    ├── dataset_info.py       reads datasetInfo.xlsx
    ├── preprocessing.py
    ├── detection/            MOG2, adaptive MOG2, ground truth, optical flow, cascade
    ├── tracking/             centroid, IoU-Hungarian, SORT, TFCR-style
    ├── features/             trajectory, wavelet, interaction, dense, video-level
    ├── pipeline/             builds and runs one video through the steps above
    ├── models/               decision tree / forest, sklearn, boosting, neural, imbalance handling
    ├── evaluation/           metrics, cross-validation, result reports, experiment comparison
    ├── experiment/           runs a whole experiment: videos -> features -> models -> results
    └── viz/                  plots and FiftyOne
```
