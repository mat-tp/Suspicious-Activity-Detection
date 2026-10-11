# SACAIR 2026 review - what to change

Work through these in order. Tick each one off when done.

## Code
- [x] Restore missing `models` package; repo runs from a clean checkout
- [x] Clean package structure, explicit imports
- [x] Fix adaptive MOG2 calling `apply()` twice on one frame (double learning update); `update_once: false` reproduces the old behaviour
- [x] Adaptive parameters in the YAML were silently ignored by the builder - fixed (needed for the sensitivity sweep)
- [x] Group I's multi-family features were switched on by the group name - now explicit (`enable_multifamily`)
- [x] Config: adaptive MOG2 with the *baseline* feature set (isolates the detector) - `ablation_1_detector_only`; also ablations 2-4
- [x] Full 2x2x2 factorial (adds `ablation_5`, `ablation_6`) and `source/scripts/factorial_analysis.py`; see [ablation_study.md](ablation_study.md)
- [x] var_threshold ablation: macro_f1_common plotted, per-distortion yield, loud column check
- [x] Skip guards for tests that need imbalanced-learn
- [x] Run SVM, KNN, RF and an MLP on the proposed pipeline too - Group I now inherits the full model list from `base.yaml`, which includes the new `mlp`
- [x] Compare against ViBe / SuBSENSE / PBAS - `detector_vibe`, `detector_pbas` (own NumPy implementations), `detector_subsense` (needs pybgs); still to run on the real data
- [x] Detector comparison on the full-pipeline features as well - `detector_*_fullfeat`
- [x] Sensitivity sweep: q, w, tau_r, tau_min - `source/scripts/sensitivity_sweep.py` (written and tested; still to run on the real data)
- [x] Class imbalance: macro-F1, balanced accuracy, per-class results, configurable class weights / resampling / augmentation
- [ ] Video-level augmentation (flip, brightness, temporal crop) - not done; needs detection to be re-run on augmented videos
- [ ] Choose the imbalance strategy by comparing settings on the real data, and report all of them
- [x] Replace five-fold paired t-tests - bootstrap intervals, repeated CV, and `compare_experiments.py` (difference + paired interval + McNemar, on common and all videos)
- [ ] Re-run Group A and the proposed pipeline with the current code so they have per-video predictions, then compare them on `all` videos (the yield difference)
- [ ] Audit the 99.7% yield: count missing files, zero detections, short tracks and feature failures separately (`videos_attempted.csv` has the reason for each failure)
- [ ] Report results with and without video metadata features
- [ ] Make `dynamic_min_len` actually used in the feature extractors

## Paper
- [ ] Acknowledge Shaban et al. (2024, 2025) and Gomez-Nieto et al. (2021, thesis); narrow the novelty claim
- [ ] Say "evaluate", not "reformulate"; rename to "imaging-condition classification"
- [ ] Appendix: every handcrafted feature, its formula, parameters and aggregation
- [ ] State the train/test split and how imbalance was handled
- [ ] Justify the mask-quality function (Eq. 1 is still not validated against GT masks); add the MOG2 citation
- [x] Paper: tau_min corrected to 8 (tau and the 147-dim ablation row wait on the re-runs)
- [ ] Justify omitting deep models; add the MLP result
- [ ] Code and data availability statement
- [ ] Fix citation order and the weighted-F1 citation
