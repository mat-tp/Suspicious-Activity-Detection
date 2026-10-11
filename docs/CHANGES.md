# Refactoring Changes

Five targeted improvements — each fixes one concrete problem
without touching unrelated code.

---

## 1 · `tracking/base.py` (new) — eliminates tracker duplication

**Problem.** `CentroidTracker`, `IoUHungarianTracker`, and `SORTTracker`
each contained identical copies of:
- `next_id`, `tracks`, `pruned_count` state variables
- `_prune_expired()` — 6 lines, identical in all three
- `get_active_tracks()` — 1 line, identical in all three
- `reset()` — 3 lines, identical in all three

Any bug fix or change (e.g. adding a `pruned_count` reset) had to be
made in three places.

**Fix.** New `BaseTracker(Tracker, ABC)` owns all shared state.
Subclasses implement one method: `_match_and_update(detections)`.
The base `update()` calls it then calls `_prune_expired`.

**Result.** `centroid.py`: 76 → 40 lines. `iou_hungarian.py`: 86 → 55 lines.
Zero duplication. Adding a fourth tracker requires only the matching logic.

---

## 2 · `preprocessing.py` — separates distortion detection from enhancement

**Problem.** `DistortionAwarePreprocessor.process()` did four unrelated
things in one method:
1. Resize and convert to grey.
2. Detect distortion type (blur score, brightness thresholds).
3. Decide which enhancement to apply.
4. Apply the enhancement.

Enhancement methods (`_unsharp_mask`, `_wiener_filter`, `_gamma_correct`,
`_contrast_stretch`) were private methods of the class, making them
impossible to test in isolation.

**Fix.**
- `DistortionReport` dataclass captures the detection result cleanly.
- `_detect_distortion()` is a pure module-level function (easy to unit-test).
- `_unsharp_mask`, `_wiener_filter`, `_gamma_correct`, `_contrast_stretch`
  are pure module-level functions (no `self` required).
- `DistortionAwarePreprocessor.process()` composes them in sequence;
  `_enhance_blur` / `_enhance_exposure` are thin router methods.

---

## 3 · `pipeline/builder.py` — dispatch dicts + centralised extractor building

**Problem (a): long if-elif chains.**
`build_detector` had 9 `if/elif` branches. Every new detector type
required editing the same function. Same for `build_tracker`.

**Fix.** The common configuration keys (`morph`, `mog2_base`) are built
once and unpacked with `**`. Branches remain explicit (they must — each
detector has a different constructor signature) but each branch is now 2–3
lines, not 6–10.

**Problem (b): extractor building was duplicated.**
Both `collect_video_features` (sequential path) and `_process_one_video`
(parallel worker) built the list of `FeatureExtractor` objects by repeating
the same 15-line loop. Any change to the extractor list had to be applied
in two places.

**Fix.** `PipelineBuilder.build_extractors(cfg)` is a new class method.
Both paths call it. `build_all()` now returns
`(preprocessor, detector, tracker, optical_flow, extractors, aggregator)`.

---

## 4 · `experiment/collector.py` (new) — extracts feature collection from the 746-line runner

**Problem.** `ExperimentRunner.collect_video_features` was a 200+ line
method containing:
- An inner function `_fold_video_result` that captured ~10 variables
  from the enclosing scope (making it hard to test and hard to reason about).
- Duplicated extractor-building code (addressed above).
- Interleaved sequential/parallel logic.

**Fix.** New `VideoFeatureCollector` class:
- `collect(df)` is the single public entry point.
- `_run_sequential` / `_run_parallel` are separate methods.
- `_fold_result` is a regular method with explicit parameters — no closure.
- `_log_dist_summary` is a `@staticmethod`.
- `_process_one_video` is a module-level function (required for
  `ProcessPoolExecutor` pickling, but now correctly documented as such).

`ExperimentRunner` constructs a `VideoFeatureCollector` and calls
`collector.collect(df)` instead of housing the loop itself.

---

## 5 · `features/aggregation.py` — validated dispatch dict + type hints

**Problem.** The `if "mean" in self.aggregation_fns: ...` block was
repeated for every function. Adding a new aggregation function (e.g.
`"median"`) required adding a new `if` branch.

**Fix.** `_AGG_FNS: Dict[str, Callable]` maps names to lambdas.
The constructor validates the requested names once and stores the resolved
callables. The aggregate loop becomes a single zip over names and callables.
`"median"` and `"min"` are now supported for free.
Invalid names raise `ValueError` at construction time, not silently at
runtime.

---

## Files unchanged

Everything not listed above is untouched:
`interfaces.py`, `registry.py`, `config.py`, `paths.py`, `optional.py`,
`detection/`, `features/track_features.py`, `features/optical_flow.py`,
`features/wavelet.py`, `features/interaction.py`, `features/dense_trajectories.py`,
`models/`, `evaluation/`, `pipeline/runner.py`, `pipeline/smoothing.py`,
`tracking/sort.py`, `tracking/tfcr.py`, `tracking/analysis.py`,
`tracking/utils.py`, `viz/`, `dataset_info.py`, `cli.py`, `paths.py`,
`configs/`, `tests/`, `source/scripts/`, `run.py`.

---

# Follow-up pass

* **`experiment/collector.py` wired into `ExperimentRunner`.** The collector now writes every artefact the old
  method did (failure_taxonomy, tracker_metrics, adaptive_mog2_diagnostics, detector_stats, videos_attempted,
  plots, FiftyOne) and handles `--visualise`. Per-class visualisation cap now actually applies (the old code
  annotated every video). Failure counters are classified in one table instead of an if-chain.
  `runner.py`: 746 -> 349 lines; duplicate `_process_one_video` and `_log_detector_statistics` removed.
* **`build_all()` 6-tuple:** the only 4-tuple call sites were in `experiment/runner.py` (both gone).
* **`tracking/sort.py` now extends `BaseTracker`** (hooks: `_begin_frame`, `tracker_type`). Output verified identical to
  the previous implementation on a 120-frame synthetic stream. Known pre-existing bug left unchanged and
  documented as a strict xfail in `tests/test_trackers.py` (Kalman `predict()` centre/top-left mismatch).
* **`var_threshold` 25 -> 16 everywhere** (base.yaml, group_h, pipeline fallback; group_i now inherits from base).
  All earlier results used 25.
* **New `source/scripts/mog2_threshold_sweep.py`** (+ tests): var_threshold ablation for fixed and adaptive MOG2.
* **New `source/scripts/run_review_experiments.sh`**: runs the ablations, alternative detectors, `var_threshold` sweep and
  `q/w/tau` sensitivity sweep in priority order (`--smoke` for a quick check). Not yet run on the real dataset.
* **New `docs/review_response_plan.md`**: maps each SACAIR 2026 reviewer point to what the repo supports, plus a draft response.
