# SACAIR 2026, Paper 29: review mapping and draft response

Status key: **Run** = the code/config exists, only an experiment run is needed. **Write** = text change only. **Build** = new work needed. Nothing below has been run on the real dataset yet.

## 1. Mapping

| # | Reviewer point | Status | What to do |
|---|---|---|---|
| R1 | No sensitivity analysis for q, w, tau_r, tau_min | Run | `source/scripts/sensitivity_sweep.py` |
| R1 | Fixed MOG2 threshold unjustified (new, from our ablation) | Run | `source/scripts/mog2_threshold_sweep.py` (fixed and adaptive, same grid) |
| R1 | Mask quality function (Eq. 1) not validated | Build | Needs GT masks. `detection/ground_truth.py` loads them; correlate mask-quality score with IoU against GT on the videos that have it |
| R1 | SVM/KNN/custom RF only run under baseline (RQ3 "comprehensive") | Run | `group_i_adaptive_mog2` inherits the full 7-model list from `base.yaml`; run it and report all models |
| R1 | No comparison with ViBe, SuBSENSE, PBAS | Run | `detector_vibe`, `detector_pbas`, `detector_subsense` configs exist. SuBSENSE needs `pybgs` (may not install) |
| R1 | 5-fold paired t-tests, folds not independent | Run + Write | `cv_repeats` and bootstrap CIs exist (`evaluation/`). Use repeated CV and bootstrap CIs, drop or soften t-tests |
| R1 | Class imbalance (160 Pristine vs ~1,450) may inflate 97.7% | Run | Balanced accuracy and macro-F1 are already computed; add per-class table (`per_class_*.csv`) |
| R1 | No code/data availability statement | Write | Add statement, release repo |
| R1 | Feature importance paragraph thin | Run | `feature_importance_*.json` already saved; report top features per family |
| R1 | Citation order, weighted-F1 citation | Write | |
| R2 | Reformulation / "first" novelty overstated; Shaban 2024/2025 and Gomez-Nieto work on AD-SVD | Write | Narrow the claim, add them to Table 1 |
| R2 | Introduction citation (Ahmed 2019) does not support claim | Write | Replace |
| R2 | Missing literature on conventional activity recognition and its limits | Write | Add synthesis in logical order |
| R2 | Split ratio, SVM training, model selection rationale, imbalance handling | Write | All in config: 5-fold stratified CV, `class_weight: balanced`, SVM RBF C=10 gamma=scale. State them |
| R2 | Terminology (features/representations/descriptors), writing quality | Write | |
| R3 | Three configs change detector, preprocessing and features together, so RQ2 gain not attributable to adaptive MOG2 | Run | **Strongest point; may be fixable with existing configs.** `ablation_1` (detector only), `ablation_2` (preprocessing only), `ablation_3` (features only), `ablation_4` (everything except detector) isolate each factor |
| R3 | Different numbers of evaluated videos across configs | Run | Report video yield (already logged) and `macro_f1_common` on the shared video subset |
| R3 | Handcrafted features not specified | Write | Add a feature table or appendix from `features/` and the config |
| R3 | No neural comparison; MLP on handcrafted features | Run | `mlp` is in `base.yaml`'s model list, so it runs with everything else |
| R3 | MOG2 citation missing | Write | Cite Zivkovic 2004 / Zivkovic & van der Heijden 2006 |
| R3 | "Reformulating" and "distortion classification" terminology | Write | Use "imaging-condition classification" |

**Key observation.** R3's central objection (confounded configurations) is the one that sank the paper, and the four ablation configs already isolate each factor. If ablations 1 to 4 show the detector effect holds when everything else is fixed, the rebuttal is an experiment, not an argument.

## 2. Suggested run order

1. `ablation_1` to `ablation_4` plus `group_a` and `group_i` (all 7 models). Addresses R3's main point and R1's RQ3 point.
2. `detector_vibe`, `detector_pbas`, `detector_subsense` (R1).
3. `mog2_threshold_sweep.py`, then `sensitivity_sweep.py` (R1).
4. Repeated CV (`training.cv_repeats: 3` to `5`) for all of the above.

Fix the SORT predict() bug **only if** SORT numbers appear in the paper; the main pipeline uses `iou_hungarian`.

## 3. Draft response (for a resubmission cover note or journal version)

We thank the reviewers for their detailed comments. The main changes are:

**Attribution of the improvement (R3, R1).** The earlier comparison changed the detector, preprocessing and features together. We now report four single-factor ablations: detector only, preprocessing only, features only, and the full pipeline with fixed MOG2. We also compare against ViBe, PBAS and SuBSENSE, and against fixed MOG2 at every threshold in a sweep (8 to 40), so the gain can be separated from simply lowering the threshold. [Insert results.]

**Classifier comparison (R1, R3).** All seven classifiers, including an MLP on the same handcrafted features, are now evaluated under the proposed pipeline. [Insert results.]

**Sensitivity and statistics (R1).** We added one-at-a-time sensitivity analyses of q, w, tau_r, tau_min and the base variance threshold. We replaced paired t-tests on five folds with repeated stratified cross-validation and bootstrap 95% confidence intervals, and we report macro-F1, balanced accuracy and per-class results to address class imbalance. Because video yield varies between configurations, we also report scores on the common subset of videos.

**Novelty and framing (R2, R3).** We narrowed our claims. AD-SVD annotations had been used for human-motion classification (Shaban et al. 2024, 2025) and for pristine versus exposure-distorted classification (Gomez-Nieto). We now describe our contribution as a systematic evaluation of activity and imaging-condition classification with traditional classifiers on AD-SVD, and we added these works to Table 1 and the related work.

**Reproducibility (R1, R3).** We added a full feature table with parameters and aggregation, the split and training protocol, the MOG2 reference, and a code and data availability statement.

**Not addressed.** Eq. 1 is not validated against ground-truth mask quality: [either add the GT correlation experiment or state it as a limitation].

## 4. Caveat

The paper's reported numbers used `var_threshold = 25`. The repo default is now 16. Re-run everything at one consistent value (the sweep will tell you which), and don't mix old and new numbers in the revised tables.
