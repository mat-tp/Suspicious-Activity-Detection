# Ablation study guide

What each experiment answers, which file implements it, and how to run it. Run everything from the project root. **Nothing here has been run on AD-SVD yet** (the videos are not in the repo); the code is tested on synthetic data only (`cd source && python -m pytest tests`).

## 1. The experiments

| Question | Experiment | Code | Command |
|---|---|---|---|
| Is the RQ2 gain due to adaptive MOG2, or to preprocessing/features changing at the same time? | **2x2x2 factorial**: detector x preprocessing x features, 8 cells | configs `group_a_classical`, `ablation_1` to `ablation_6`, `group_i_adaptive_mog2`; analysis `source/scripts/factorial_analysis.py` | `./build.sh factorial analysis` (or `build.bat`), which runs the 8 cells and then `source/scripts/factorial_analysis.py` |
| Does adaptive MOG2 win only because it uses a lower threshold? | **var_threshold ablation**: fixed and adaptive MOG2 on the same grid (8 to 40) | `source/scripts/mog2_threshold_sweep.py` | `python source/scripts/mog2_threshold_sweep.py --target activity --resume` |
| Are q, w, tau_r, tau_min hand-tuned? | One-at-a-time sensitivity | `source/scripts/parameter_ablation.py` | `./build.sh ablation` (every selected parameter, one at a time; `--resume` continues) |
| Generic adaptivity or this method? ViBe, PBAS, SuBSENSE | Same pipeline, other background model, on trajectory features **and** on full features | `detector_{vibe,pbas,subsense}` and `detector_*_fullfeat` | in the run script, step 2 |
| Do SVM/KNN/RF close the gap under the proposed pipeline? MLP? | All 7 classifiers in every config | `models.active` in `base.yaml` | automatic |
| Fold-dependence, 5-fold t-tests | Repeated CV, bootstrap CIs, paired bootstrap, McNemar | `evaluation/`, `scripts/compare_experiments.py` | `python source/scripts/compare_experiments.py group_a_classical ablation_1_detector_only` |

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

## 4. Known gaps

* Eq. 1 validation (above).
* Video-level augmentation not implemented.
* SORT `predict()` bug is documented with an xfail test; SORT is not used by the main pipeline.
* `scripts/` and `tests/` in the project root are older copies of `source/scripts` and `source/tests`. Use the `source/` ones.

## 6. Getting the plots into the paper

`python source/scripts/collect_paper_figures.py` copies the plots to `docs/images/` as `fig_factorial_*`, `fig_threshold_*` and `fig_ablation_<group>_*` (one per parameter group). `paper.tex` has a new subsection "Controlled ablations and sensitivity analyses" (before the Conclusion) that includes them; any plot not yet present shows a red placeholder, and the text to write is marked `[TODO ...]` in red. Search for `TODO` before submitting.
