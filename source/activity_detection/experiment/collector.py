"""Collects the feature vector of every video, separate from training.

``VideoFeatureCollector`` loops over the videos (in one process or several), adds up the
per-distortion statistics, writes the diagnostic files of the run (failure taxonomy, tracker metrics,
adaptive MOG2 diagnostics, detector statistics, attempted videos, track-length plots) and returns the
list of feature dicts that training uses.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..dataset_info import DatasetInfoHandler
from ..evaluation.report import write_attempted_videos
from ..features.video_features import extract_video_metadata_features
from ..interfaces import FailureReason, FeatureDict
from ..pipeline.builder import DetectionSource, PipelineBuilder
from ..pipeline.runner import PipelineRunner
from ..tracking.analysis import TrackerAnalyzer

# This is the logger of this module.
log = logging.getLogger(__name__)


# This is the pipeline runner that each worker process builds once and reuses for all its videos.
_WORKER_RUNNER: Optional[PipelineRunner] = None

# These are the settings that stop every worker from starting its own pool of math threads.
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def resolve_worker_count(requested: Any, n_tasks: int) -> int:
    """Turn the ``n_workers`` setting (a number or ``"auto"``) into a usable process count."""
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        requested = (os.cpu_count() or 2) - 1
    return max(1, min(int(requested or 1), n_tasks or 1))


def _build_runner(config_dict: Dict[str, Any], detection_source: Optional[str],
                  runner_args: Dict[str, Any]) -> PipelineRunner:
    cfg = Config(config_dict)
    source = DetectionSource(detection_source) if detection_source else None
    preprocessor, detector, tracker, optical_flow, extractors, aggregator = (
        PipelineBuilder.build_all(cfg, source)
    )
    return PipelineRunner(preprocessor, detector, tracker, optical_flow, extractors,
                          aggregator, **runner_args)


def _init_worker(config_dict: Dict[str, Any], detection_source: Optional[str],
                 runner_args: Dict[str, Any]) -> None:
    """Runs once in each worker process: one OpenCV thread, then build the pipeline."""
    global _WORKER_RUNNER
    cv2.setNumThreads(1)
    _WORKER_RUNNER = _build_runner(config_dict, detection_source, runner_args)


def _process_one_video(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pipeline on one video inside a worker process."""
    runner = _WORKER_RUNNER
    vis_path = Path(task["vis_path"]) if task.get("vis_path") else None
    result = runner.process_video(
        Path(task["vpath"]),
        collect_diagnostics=True,
        frame_sampling=task["frame_sampling"],
        visualise_path=vis_path,
        collect_frame_detections=vis_path is not None,
    )
    adaptive_diag = (
        runner.detector.get_diagnostics()
        if hasattr(runner.detector, "get_diagnostics") else None
    )
    return {
        "activity": task["activity"],
        "distortion": task["distortion"],
        "vname": task["vname"],
        "vpath": task["vpath"],
        "result": result,
        "adaptive_diagnostics": adaptive_diag,
        "vis_path": str(vis_path) if vis_path else None,
    }


def _empty_dist_stats() -> Dict[str, Any]:
    return {
        "total": 0, "successful": 0, "failed": 0,
        "failure_detection": 0, "failure_tracking": 0,
        "failure_feature": 0, "failure_track_too_short": 0,
        "failure_unknown": 0,
        "n_tracks_total": 0, "n_frames_total": 0,
        "with_trajectory": 0,      # videos that kept at least one track (true trajectory yield)
        "with_usable_trajectory": 0,   # videos with a track of >= min_trajectory_len points
        "track_lengths": [],
    }


# failure flag on FailureReason -> (reason label, dist_stats counter)
_FAILURE_KINDS = (
    ("detection_failure", "detection_failure", "failure_detection"),
    ("track_too_short", "track_too_short", "failure_track_too_short"),
    ("tracking_failure", "tracking_failure", "failure_tracking"),
    ("feature_failure", "feature_failure", "failure_feature"),
)


def _classify_failure(failure: Optional[FailureReason]) -> Tuple[str, Optional[str]]:
    """Return (reason_label, dist_stats_counter_to_increment)."""
    if failure is None:
        return "unknown", None
    for attr, label, counter in _FAILURE_KINDS:
        if getattr(failure, attr, False):
            return label, counter
    if failure.unknown_error:
        return f"unknown: {failure.error_message[:50]}", "failure_unknown"
    return "unknown", None


@dataclass
class _Accumulator:
    """Everything one collection run builds up, in one place."""

    video_features: List[FeatureDict] = field(default_factory=list)
    failure_records: List[Dict] = field(default_factory=list)
    dist_stats: Dict[str, Dict] = field(default_factory=dict)
    track_lengths: List[int] = field(default_factory=list)
    tracker_metrics: List[Dict] = field(default_factory=list)
    adaptive_stats: List[Dict] = field(default_factory=list)
    unknown_by_class: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    visualised_counts: Dict[str, int] = field(default_factory=dict)
    processed: int = 0

    def stats(self, dkey: str) -> Dict:
        return self.dist_stats.setdefault(dkey, _empty_dist_stats())


class VideoFeatureCollector:
    """Run the detection-tracking-feature pipeline over every video in *df*."""

    def __init__(
        self,
        config: Config,
        detection_source: Optional[DetectionSource] = None,
        dataset_info: Optional[DatasetInfoHandler] = None,
        *,
        run_dir: Optional[Path] = None,
        plots=None,
        fo_visualizer=None,
        visualise: bool = False,
        n_visualise: int = 3,
    ) -> None:
        self.config = config
        self.detection_source = detection_source
        self.dataset_info = dataset_info
        # This is the folder where the diagnostic files of the run are written (None
        # means no files).
        self.run_dir = Path(run_dir) if run_dir else None
        self.plots = plots
        self.fo_visualizer = fo_visualizer
        self.visualise = visualise
        self.n_visualise = n_visualise
        # These are the settings that are read once from the config and handed to every
        # pipeline runner.
        self.min_trajectory_len = config.tracking.min_trajectory_len
        self.use_dense = config.features.get("dense_trajectories", {}).get("enabled", False)
        self.enable_mf = config.features.get("enable_multifamily", False)
        self.filter_short = bool(config.tracking.get("filter_short_tracks", False))
        self.global_flow_step = int(config.features.get("global_flow_step", 5))
        self.appearance_frame_count = int(config.features.get("appearance_frame_count", 30))

    def collect(self, df: pd.DataFrame, use_sample: bool = False) -> List[FeatureDict]:
        """Return a flat list of video feature dicts ready for training."""
        groups = self._build_groups(df, use_sample)
        total = sum(len(v) for v in groups.values())
        print(f"\n  Videos to process: {total}")

        acc = _Accumulator(
            dist_stats={str(d): _empty_dist_stats() for d in df["Distortion"].unique()},
            visualised_counts={a: 0 for a in df["Activity"].unique()},
        )
        tasks = self._build_task_list(groups, acc, total)
        frame_sampling = self._get_frame_sampling(groups)
        n_workers = resolve_worker_count(self.config.experiment.get("n_workers", 1), len(tasks))

        if n_workers <= 1:
            self._run_sequential(tasks, df, frame_sampling, total, acc)
        else:
            self._run_parallel(tasks, df, frame_sampling, total, n_workers, acc)

        # Parallel workers finish in any order, and the CV folds depend on the row order,
        # so the rows are sorted to get the same folds for the same videos on every run.
        acc.video_features.sort(key=lambda fv: fv["video_name"])
        self._write_artifacts(acc, groups, total)
        return acc.video_features

    def _build_groups(self, df: pd.DataFrame, use_sample: bool) -> Dict[Tuple, List[str]]:
        groups: Dict[Tuple, List[str]] = {}
        n = self.config.dataset.get("n_random_videos_per_group", 5)
        for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
            pool = grp["Name of Video Series"].tolist()
            if use_sample:
                pool = random.sample(pool, min(n, len(pool)))
            groups[(act, dist)] = pool
        return groups

    def _build_task_list(self, groups: Dict[Tuple, List[str]], acc: _Accumulator,
                         total: int) -> List[Dict[str, Any]]:
        """One task per video that exists on disk; missing files are recorded as failures here."""
        tasks: List[Dict[str, Any]] = []
        video_dir = Path(self.config.paths.video_dir)
        vis_root = (self.run_dir / "annotated_videos") if (self.visualise and self.run_dir) else None
        if vis_root is not None:
            vis_root.mkdir(parents=True, exist_ok=True)
            print(f"\n  [visualise] Annotated videos → {vis_root}")
            print(f"  [visualise] Cap: {self.n_visualise} per activity class")
        assigned: Dict[str, int] = {}

        for (activity, distortion), names in groups.items():
            for vname in names:
                dkey = str(distortion)
                acc.stats(dkey)["total"] += 1
                vpath = video_dir / vname
                if not vpath.exists():
                    acc.processed += 1
                    print(f"  [{acc.processed}/{total}] MISSING  {vname}")
                    acc.failure_records.append({"video": vname, "activity": activity,
                                                "distortion": distortion, "reason": "file_missing"})
                    acc.stats(dkey)["failed"] += 1
                    continue
                vis_path = None
                if vis_root is not None and assigned.get(activity, 0) < self.n_visualise:
                    class_dir = vis_root / str(activity)
                    class_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = str(class_dir / f"{Path(vname).stem}_annotated.mp4")
                    assigned[activity] = assigned.get(activity, 0) + 1
                tasks.append({"activity": activity, "distortion": distortion, "vname": vname,
                              "vpath": str(vpath), "vis_path": vis_path})
        return tasks

    def _get_frame_sampling(self, groups: Dict) -> int:
        if self.dataset_info:
            first = next(iter(groups.values()), [None])[0]
            if first:
                return self.dataset_info.get_adaptive_frame_sampling(first, 100)
        return 1

    def _runner_args(self) -> Dict[str, Any]:
        return dict(
            min_trajectory_len=self.min_trajectory_len,
            use_dense_trajectories=self.use_dense,
            enable_multifamily=self.enable_mf,
            filter_short_tracks=self.filter_short,
            global_flow_step=self.global_flow_step,
            appearance_frame_count=self.appearance_frame_count,
        )

    def _run_sequential(self, tasks, df, frame_sampling, total, acc: _Accumulator) -> None:
        source = self.detection_source.value if self.detection_source else None
        runner = _build_runner(self.config.to_dict(), source, self._runner_args())
        for task in tasks:
            vis_path = Path(task["vis_path"]) if task["vis_path"] else None
            result = runner.process_video(
                Path(task["vpath"]), collect_diagnostics=True,
                frame_sampling=frame_sampling, visualise_path=vis_path,
                collect_frame_detections=vis_path is not None,
            )
            adaptive_diag = (runner.detector.get_diagnostics()
                             if hasattr(runner.detector, "get_diagnostics") else None)
            self._fold_result(task, total, result, adaptive_diag, df, acc)

    def _run_parallel(self, tasks, df, frame_sampling, total, n_workers, acc: _Accumulator) -> None:
        print(f"\n  [parallel] Processing {len(tasks)} videos across {n_workers} worker processes")
        payloads = [{**task, "frame_sampling": frame_sampling} for task in tasks]
        init_args = (
            self.config.to_dict(),
            self.detection_source.value if self.detection_source else None,
            self._runner_args(),
        )
        # The workers inherit the environment of this process when they are spawned, so the
        # thread limits are set here and put back afterwards.
        saved_env = {name: os.environ.get(name) for name in _SINGLE_THREAD_ENV}
        os.environ.update({name: "1" for name in _SINGLE_THREAD_ENV})
        try:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_worker, initargs=init_args) as pool:
                futures = {pool.submit(_process_one_video, p): p for p in payloads}
                for fut in as_completed(futures):
                    payload = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as exc:
                        acc.processed += 1
                        log.error("Worker failed for %s: %s", payload["vname"], exc)
                        acc.failure_records.append({"video": payload["vname"], "activity": payload["activity"],
                                                    "distortion": payload["distortion"],
                                                    "reason": f"worker_error: {exc}"})
                        acc.stats(str(payload["distortion"]))["failed"] += 1
                        continue
                    self._fold_result(out, total, out["result"], out["adaptive_diagnostics"], df, acc)
        finally:
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _fold_result(self, task: Dict, total: int, result, adaptive_diag,
                     df: pd.DataFrame, acc: _Accumulator) -> None:
        """Apply one video's result to the accumulators (sequential or parallel alike)."""
        activity, distortion, vname = task["activity"], task["distortion"], task["vname"]
        vis_path = task.get("vis_path")
        dkey = str(distortion)
        stats = acc.stats(dkey)
        acc.processed += 1
        idx = acc.processed

        if result.track_lengths:
            acc.track_lengths.extend(result.track_lengths)
            stats["track_lengths"].extend(result.track_lengths)
        stats["n_frames_total"] += result.n_frames_processed
        stats["n_tracks_total"] += len(result.tracks)

        if result.tracker_metrics:
            result.tracker_metrics.update({"video": vname, "activity": activity, "distortion": distortion})
            acc.tracker_metrics.append(result.tracker_metrics)

        if not result.video_feature:
            self._record_failure(task, result, stats, acc)
            print(f"  [{idx}/{total}] {activity}/{distortion}  {vname}  →  FAILED "
                  f"({acc.failure_records[-1]['reason']})")
            return

        stats["successful"] += 1
        has_traj = bool(result.track_lengths)
        stats["with_trajectory"] += int(has_traj)
        stats["with_usable_trajectory"] += int(result.n_usable_tracks > 0)
        if vis_path is not None:
            self._record_visualisation(task, result, acc)

        fv = result.video_feature
        fv.update({
            "activity": activity, "distortion": distortion, "video_name": vname,
            "n_tracks": len(result.tracks), "frames_used": result.n_frames_processed,
            "has_trajectory": float(has_traj),
            "n_usable_tracks": result.n_usable_tracks,
            "track_length": float(np.mean(result.track_lengths)) if result.track_lengths else 0.0,
            "feature_count": result.feature_count,
            "track_confidence": self._track_confidence(result.track_lengths),
        })
        fv.update(self._metadata_features(df, vname, result.n_frames_processed))
        acc.video_features.append(fv)

        suffix = ""
        if adaptive_diag is not None:
            acc.adaptive_stats.append({"video": vname, "activity": activity,
                                       "distortion": distortion, **adaptive_diag})
            suffix = (f"  trig={adaptive_diag['adaptive_trigger_rate']:.0%}"
                      f" thr={adaptive_diag['avg_threshold_used']:.1f}"
                      f" acc={adaptive_diag['rerun_accept_rate']:.0%}"
                      f" mq={adaptive_diag['avg_mask_quality']:.2f}"
                      f" tq={adaptive_diag['avg_track_quality']:.2f}")
        print(f"  [{idx}/{total}] {activity}/{distortion}  {vname}"
              f"  →  tracks={len(result.tracks)} frames={result.n_frames_processed}"
              f" features={result.feature_count}{suffix}")

    def _track_confidence(self, track_lengths) -> float:
        if not track_lengths:
            return 0.0
        return float(np.mean([min(t / self.min_trajectory_len, 1.0) for t in track_lengths]))

    @staticmethod
    def _metadata_features(df: pd.DataFrame, vname: str, n_frames_processed: int) -> Dict:
        """Frame count / fps / duration from datasetInfo.xlsx, with safe fallbacks."""
        mask = df["Name of Video Series"] == vname
        if mask.any():
            row = df[mask].iloc[0]
            n_frames = int(row.get("Nframes", n_frames_processed))
            fps = float(row.get("Frame Rate (FPS)", 10.0))
            duration = float(row.get("Duration (s)", n_frames / max(fps, 1.0)))
        else:
            n_frames, fps, duration = n_frames_processed, 10.0, n_frames_processed / 10.0
        return extract_video_metadata_features(n_frames, fps, duration)

    @staticmethod
    def _record_failure(task: Dict, result, stats: Dict, acc: _Accumulator) -> None:
        stats["failed"] += 1
        reason, counter = _classify_failure(result.failure)
        if counter:
            stats[counter] += 1
        if counter == "failure_unknown":
            acc.unknown_by_class.setdefault(task["activity"], []).append(
                (task["vname"], result.failure.error_message))
        acc.failure_records.append({"video": task["vname"], "activity": task["activity"],
                                    "distortion": task["distortion"], "reason": reason})

    def _record_visualisation(self, task: Dict, result, acc: _Accumulator) -> None:
        activity = task["activity"]
        acc.visualised_counts[activity] = acc.visualised_counts.get(activity, 0) + 1
        print(f"\n  [visualise] Saved {Path(task['vis_path']).name}"
              f"  ({acc.visualised_counts[activity]}/{self.n_visualise} for '{activity}')", end="")
        if self.fo_visualizer and result.frame_detections and result.frame_size:
            self.fo_visualizer.add_video_sample(
                video_path=task["vpath"], frame_detections=result.frame_detections,
                frame_size=result.frame_size, activity=activity,
                distortion=task["distortion"], video_name=task["vname"], tracks=result.tracks)

    def _write_artifacts(self, acc: _Accumulator, groups: Dict, total: int) -> None:
        if acc.unknown_by_class:
            log.warning("Unknown failures by activity class:")
            for cls, failures in acc.unknown_by_class.items():
                log.warning("  Class %s: %d unknown failures", cls, len(failures))
                for name, msg in failures[:3]:
                    log.warning("    %s — %s", name, msg)

        self._log_detector_statistics(acc.dist_stats, total, len(acc.video_features))
        if self.run_dir is None:
            return
        if self.fo_visualizer:
            self.fo_visualizer.finalize()
        self._dump("failure_taxonomy.json", acc.failure_records)
        if acc.tracker_metrics:
            self._dump("tracker_metrics.json", acc.tracker_metrics)
        if acc.adaptive_stats:
            self._dump("adaptive_mog2_diagnostics.json", acc.adaptive_stats)
            agg = {k: float(np.mean([r[k] for r in acc.adaptive_stats]))
                   for k in ("adaptive_trigger_rate", "avg_threshold_used",
                             "rerun_accept_rate", "avg_mask_quality", "avg_track_quality")}
            print(f"\n  [AdaptiveMOG2] across {len(acc.adaptive_stats)} videos:"
                  f"  trigger={agg['adaptive_trigger_rate']:.0%}"
                  f"  threshold={agg['avg_threshold_used']:.2f}"
                  f"  rerun_accept={agg['rerun_accept_rate']:.0%}"
                  f"  mask_q={agg['avg_mask_quality']:.3f}"
                  f"  track_q={agg['avg_track_quality']:.3f}")
        if acc.track_lengths:
            if self.plots:
                self.plots.track_length_histogram(
                    acc.track_lengths, f"Track length distribution — {self.config.experiment.name}",
                    "track_length_histogram.png")
            self._dump("trajectory_length_justification.json",
                       TrackerAnalyzer.justify_min_trajectory_length(acc.track_lengths), default=None)
        if self.plots:
            self.plots.detector_stats_plot(
                {k: {"success_rate": s["successful"] / s["total"] if s["total"] > 0 else 0.0}
                 for k, s in acc.dist_stats.items()},
                "detector_stats_by_distortion.png")
        write_attempted_videos(self.run_dir / "videos_attempted.csv", groups,
                               [fv["video_name"] for fv in acc.video_features],
                               {r["video"]: r["reason"] for r in acc.failure_records})

    def _dump(self, name: str, obj, default=str) -> None:
        with open(self.run_dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=default)

    def _log_detector_statistics(self, dist_stats: Dict, total: int, n_success: int) -> None:
        print("\n" + "=" * 70 + "\n  DETECTOR STATISTICS\n" + "=" * 70)
        print(f"  {'Distortion':<20}  {'Videos':>7}  {'Valid':>7}  {'Failed':>7}  {'Rate':>6}  "
              f"  {'Det.Fail':>9}  {'Trk.Fail':>9}  {'TooShort':>9}  {'Feat.Fail':>10}  {'Unknown':>8}")
        print("  " + "-" * 115)
        summary = {}
        for dkey, s in dist_stats.items():
            rate = s["successful"] / s["total"] if s["total"] > 0 else 0.0
            print(f"  {dkey:<20}  {s['total']:>7}  {s['successful']:>7}  {s['failed']:>7}  {rate:>6.2f}  "
                  f"  {s['failure_detection']:>9}  {s['failure_tracking']:>9}  {s['failure_track_too_short']:>9}  "
                  f"{s['failure_feature']:>10}  {s['failure_unknown']:>8}")
            summary[dkey] = {
                "total_videos": s["total"], "successful_videos": s["successful"],
                "videos_with_trajectory": s["with_trajectory"],
                "videos_with_usable_trajectory": s["with_usable_trajectory"],
                "failed_videos": s["failed"], "success_rate": rate,
                "failure_detection": s["failure_detection"], "failure_tracking": s["failure_tracking"],
                "failure_track_too_short": s["failure_track_too_short"],
                "failure_feature": s["failure_feature"], "failure_unknown": s["failure_unknown"],
                "n_tracks_total": s["n_tracks_total"], "n_frames_total": s["n_frames_total"],
                "mean_track_length": float(np.mean(s["track_lengths"])) if s["track_lengths"] else 0.0,
            }
        overall = n_success / total * 100 if total > 0 else 0.0
        n_traj = sum(s["with_trajectory"] for s in dist_stats.values())
        traj_pct = n_traj / total * 100 if total > 0 else 0.0
        n_usable = sum(s["with_usable_trajectory"] for s in dist_stats.values())
        print(f"\n  Overall: {n_success}/{total} produced a feature vector ({overall:.1f}%)")
        print(f"  Trajectory yield: {n_traj}/{total} videos kept at least one track ({traj_pct:.1f}%)")
        print(f"  Usable-trajectory yield: {n_usable}/{total} videos had a track of at least "
              f"min_trajectory_len points ({(n_usable / total * 100 if total else 0.0):.1f}%)")
        if self.enable_mf and n_success > n_traj:
            print(f"  NOTE: multi-family mode keeps videos without tracks ({n_success - n_traj} here), "
                  f"so the first number is NOT a trajectory yield. Report the second.\n" + "=" * 70)
        else:
            print("=" * 70)
        if self.run_dir is not None:
            with open(self.run_dir / "detector_stats.json", "w") as f:
                json.dump(summary, f, indent=2)
