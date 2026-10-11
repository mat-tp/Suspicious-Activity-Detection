# Ablation study guide (SACAIR 2026 reviews)

What each experiment answers, which file implements it, and how to run it. Run everything from the project root. **Nothing here has been run on AD-SVD yet** (the videos are not in the repo); the code is tested on synthetic data only (`cd source && python -m pytest tests`).

## 1. The experiments

| Question (reviewer) | Experiment | Code | Command |
|---|---|---|---|
| Is the RQ2 gain due to adaptive MOG2, or to preprocessing/features changing at the same time? (R3) | **2x2x2 factorial**: detector x preprocessing x features, 8 cells | configs `group_a_classical`, `ablation_1` to `ablation_6`, `group_i_adaptive_mog2`; analysis `source/scripts/factorial_analysis.py` | `bash source/scripts/run_review_experiments.sh`, then `python source/scripts/factorial_analysis.py --target activity` |
| Does adaptive MOG2 win only because it uses a lower threshold? | **var_threshold ablation**: fixed and adaptive MOG2 on the same grid (8 to 40) | `source/scripts/mog2_threshold_sweep.py` | `python source/scripts/mog2_threshold_sweep.py --target activity --resume` |
| Are q, w, tau_r, tau_min hand-tuned? (R1) | One-at-a-time sensitivity | `source/scripts/sensitivity_sweep.py` | `python source/scripts/sensitivity_sweep.py --target activity --resume` |
| Generic adaptivity or this method? ViBe, PBAS, SuBSENSE (R1) | Same pipeline, other background model, on trajectory features **and** on full features | `detector_{vibe,pbas,subsense}` and `detector_*_fullfeat` | in the run script, step 2 |
| Do SVM/KNN/RF close the gap under the proposed pipeline? MLP? (R1, R3) | All 7 classifiers in every config | `models.active` in `base.yaml` | automatic |
| Fold-dependence, 5-fold t-tests (R1) | Repeated CV, bootstrap CIs, paired bootstrap, McNemar | `evaluation/`, `scripts/compare_experiments.py` | `python source/scripts/compare_experiments.py group_a_classical ablation_1_detector_only` |

### Factorial design

| cell | detector | preprocessing | features | experiment |
|---|---|---|---|---|
| d0p0f0 | fixed | standard | trajectory | `group_a_classical` |
| d1p0f0 | adaptive | standard | trajectory | `ablation_1_detector_only` |
| d0p1f0 | fixed | distortion-aware | trajectory | `ablation_2_preprocessing_only` |
| d0p0f1 | fixed | standard | 221-dim | `ablation_3_features_only` |
| d1p1f0 | adaptive | distortion-aware | trajectory | `ablation_5_detector_preprocessing` (new) |
| d1p0f1 | adaptive | standard | 221-dim | `ablation_6_detector_features` (new) |
| d0p1f1 | fixed | distortion-aware | 221-dim | `ablation_4_full_pipeline_fixed_mog2` |
| d1p1f1 | adaptive | distortion-aware | 221-dim | `group_i_adaptive_mog2` |

`factorial_analysis.py` reports, with 95% bootstrap intervals over videos (same resampled videos in all cells): the 8 cell scores, the **simple effect** of each factor at every setting of the other two, the **main effect**, and **two-way interactions**. The key row for R3 is `simple: detector (1-0) at P=1,F=1`, which equals `group_i - ablation_4` with everything else fixed. Use `--universe all` (missing predictions count as misses, so detector yield is included) and `--universe common` (only videos every cell produced).

## 2. The var_threshold ablation

`mog2_threshold_sweep.py` runs fixed MOG2 and adaptive MOG2 over the same `var_threshold` grid. For adaptive MOG2 `var_threshold` is the ceiling of its range, so values <= tau_r are skipped (no-ops). Output per base config is `threshold_results.csv` and one figure with three panels:

1. macro-F1 with 95% CI (solid) and **macro_f1_common** (dashed). The common-subset score is the fair one, because different thresholds keep different videos.
2. Overall video yield.
3. **Yield per distortion class** (`yield_<Distortion>` columns). Shows whether a threshold helps Focus-distorted videos more than Pristine ones, which is the detection-bottleneck claim.

If the predictions file or the model's column is missing, the script now **warns loudly** instead of silently leaving `macro_f1_common` empty. Pass `--model` with a column that exists in `predictions_<target>.csv`.

Choose the final tau from this sweep, then re-run everything at that single value. The paper's numbers used 25; the code default is 16.

## 3. Paper / code mismatches found

| Item | Paper | Code | Action |
|---|---|---|---|
| tau_min | 15 | 8 | **Fixed in `paper.tex`** (value is not touched by any sweep) |
| tau | 25 | 16 (base.yaml) | TODO marker in `paper.tex` line 392; set after the sweep, then re-run all tables |
| "Ablation" features | 147-dim | `ablation_4` is 221-dim | Keep the code: 221-dim is the correct control (same as the proposed pipeline except the detector). The paper's 62.9% came from a different (147-dim) config, so re-run and replace Table 3 / the 147 mentions |

## 4. Reviewer comment coverage

| Comment | Status |
|---|---|
| R3: confounded configurations | Code ready (8-cell factorial + analysis). **Needs the runs.** |
| R3: unequal video counts | `--universe all/common`, per-distortion yield. Ready |
| R3: no MLP | `mlp` in every config. Ready |
| R3: features not specified | **Paper**: appendix with every feature, parameters and aggregation (source: `features/`) |
| R3: MOG2 citation | **Paper**: cite Zivkovic 2004; Zivkovic and van der Heijden 2006 |
| R3: "reformulate" / "distortion classification" | **Paper**: wording |
| R3: justify omitting deep models | **Paper**: cite comparable video-classification dataset sizes; add MLP result |
| R1: sensitivity of q, w, tau_r, tau_min | Code ready (`sensitivity_sweep.py`). Needs runs |
| R1: ViBe/SuBSENSE/PBAS | Configs ready, both feature sets. SuBSENSE needs `pybgs`. Needs runs |
| R1: classifiers only under baseline | All 7 run under every config. Needs runs |
| R1: t-tests on 5 folds | Replace with repeated CV + bootstrap CIs (`compare_experiments.py`). Ready |
| R1: class imbalance / 97.7% | Balanced accuracy, macro-F1, per-class tables saved. **Paper**: add a sentence and the per-class table |
| R1: Eq. 1 mask quality not validated | **Not done.** Needs a GT-mask experiment (`detection/ground_truth.py`): correlate Q_mask with IoU against GT. Otherwise state as a limitation |
| R1: code/data statement, feature-importance paragraph, citation order, weighted-F1 citation | **Paper** |
| R2: novelty overstated (Shaban 2024/2025, Gomez-Nieto 2021 and thesis) | **Paper**: narrow claim, add to Table 1 |
| R2: Ahmed 2019 citation, conventional-method literature, split/SVM/imbalance details, writing | **Paper** (details are in `base.yaml`: stratified 5-fold CV, `class_weight: balanced`, SVM RBF C=10) |

## 5. Known gaps

* Eq. 1 validation (above).
* Video-level augmentation not implemented.
* SORT `predict()` bug is documented with an xfail test; SORT is not used by the main pipeline.
* `scripts/` and `tests/` in the project root are older copies of `source/scripts` and `source/tests`. Use the `source/` ones.

## 6. Getting the plots into the paper

`python source/scripts/collect_paper_figures.py` copies the plots to `docs/images/` as `fig_factorial_*`, `fig_threshold_*` and `fig_sensitivity_*`. `paper.tex` has a new subsection "Controlled ablations and sensitivity analyses" (before the Conclusion) that includes them; any plot not yet present shows a red placeholder, and the text to write is marked `[TODO ...]` in red. Search for `TODO` before submitting.
