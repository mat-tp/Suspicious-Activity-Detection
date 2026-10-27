"""Construct pipeline components from a Config object.

Each ``build_*`` method reads the relevant section of the config and
returns a fully initialised component.  The ``build_extractors`` method
is centralised here so neither the sequential nor the parallel experiment
runner needs to repeat that logic.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from ..features.aggregation import FeatureAggregator
from ..features.optical_flow import OpticalFlowEstimator
from ..interfaces import Detector, FeatureExtractor, Preprocessor, Tracker
from ..registry import build, is_registered
from ..tracking.tfcr import TFCRTracker

# This is the logger of this module.
log = logging.getLogger(__name__)


class DetectionSource(Enum):
    """Choice of detector that ``--detection-source`` can force: plain MOG2 or the ground-truth boxes."""

    MOG2 = "mog2"
    GROUND_TRUTH = "ground_truth"


class PipelineBuilder:
    """Static factory methods that turn the sections of a ``Config`` into pipeline components."""

    @staticmethod
    def build_preprocessor(cfg) -> Preprocessor:
        """Preprocessor from ``cfg.preprocessing``.

        Disabled preprocessing gives ``passthrough``; type ``distortion_aware`` with ``distortion_aware: false`` silently becomes ``standard`` (the flag wins).
        """
        pre = cfg.preprocessing
        if not pre.get("enabled", True):
            return build("preprocessor", "passthrough")

        ptype = pre.type
        # Honour the distortion_aware flag: if the type says distortion_aware
        # but the flag is off, silently fall back to standard.
        if ptype == "distortion_aware" and not pre.get("distortion_aware", True):
            ptype = "standard"

        common = dict(
            frame_resize=tuple(pre.frame_resize),
            blur_kernel=tuple(pre.blur_kernel),
            blur_sigma=pre.blur_sigma,
        )

        if ptype == "standard":
            return build("preprocessor", "standard", **common)

        if ptype == "distortion_aware":
            da = pre.distortion_aware_params
            return build(
                "preprocessor", "distortion_aware",
                **common,
                blur_threshold=da.blur_threshold,
                exposure_low=da.exposure_low,
                exposure_high=da.exposure_high,
                enhancement_for_blur=da.enhancement_for_blur,
                enhancement_for_exposure=da.enhancement_for_exposure,
                unsharp_sigma=da.unsharp_sigma,
                unsharp_strength=da.unsharp_strength,
                clahe_clip_limit=da.clahe_clip_limit,
                clahe_tile_grid=tuple(da.clahe_tile_grid),
                bilateral_d=da.bilateral_d,
                bilateral_sigma_color=da.bilateral_sigma_color,
                bilateral_sigma_space=da.bilateral_sigma_space,
                use_multi_metric=da.get("use_multi_metric", True),
                gamma_correction=da.get("gamma_correction", False),
                gamma_value=da.get("gamma_value", 1.2),
                contrast_stretching=da.get("contrast_stretching", False),
            )

        raise ValueError(f"Unknown preprocessor type: {ptype!r}")

    @staticmethod
    def build_detector(
        cfg, detection_source: Optional[DetectionSource] = None
    ) -> Detector:
        """Detector from ``cfg.detection`` (or from *detection_source* when the command line overrides it)."""
        dtype = detection_source.value if detection_source else cfg.detection.type

        morph = dict(
            morph_close_iter=cfg.detection.morphology.close_iter,
            morph_open_iter=cfg.detection.morphology.open_iter,
            min_contour_area=cfg.detection.min_contour_area,
        )
        mog2_base = dict(
            history=cfg.detection.mog2.history,
            var_threshold=cfg.detection.mog2.var_threshold,
            detect_shadows=cfg.detection.mog2.detect_shadows,
        )

        if dtype == "mog2":
            return build(
                "detector", "mog2",
                **mog2_base, **morph,
                adaptive=cfg.detection.mog2.get("adaptive", False),
            )

        if dtype == "adaptive_mog2":
            ad = cfg.detection.get("adaptive_mog2")
            ad_kwargs = ad.to_dict() if ad is not None else {}
            return build("detector", "adaptive_mog2", **{**mog2_base, **morph, **ad_kwargs})

        if dtype in ("vibe", "pbas", "subsense"):
            own = cfg.detection.get(dtype)
            return build("detector", dtype, **{**morph, **(own.to_dict() if own is not None else {})})

        if dtype == "ground_truth":
            gt = cfg.detection.ground_truth
            return build(
                "detector", "ground_truth",
                gt_dir=cfg.paths.gt_dir,
                frame_resize=tuple(cfg.preprocessing.frame_resize),
                fallback_to_mog2=gt.fallback_to_mog2,
            )

        if dtype == "dense_flow":
            fc = cfg.detection.dense_flow
            # This lets the settings of the detector itself win over the shared morphology settings,
            # so a key that is given in both places does not raise "multiple values for keyword argument".
            return build("detector", "dense_flow", **{**morph, **fc.to_dict()})

        if dtype == "cascaded_fallback":
            cc = cfg.detection.cascaded_fallback
            return build("detector", "cascaded_fallback", **cc.to_dict())

        raise ValueError(f"Unknown detector type: {dtype!r}")

    @staticmethod
    def build_tracker(cfg) -> Tracker:
        """Tracker from ``cfg.tracking``.

        ``tracking.keep_finished_tracks`` (default true) controls whether tracks retired mid-video are still passed to feature extraction; set it to false to reproduce the legacy behaviour.
        """
        ttype = cfg.tracking.type
        common = dict(
            max_disappeared=cfg.tracking.max_disappeared,
            max_trajectory_len=cfg.tracking.max_trajectory_len,
            # True (default): tracks retired mid-video still reach feature extraction.
            # False reproduces the legacy behaviour (only tracks alive at the last frame).
            keep_finished_tracks=cfg.tracking.get("keep_finished_tracks", True),
        )

        if ttype == "centroid":
            return build("tracker", "centroid", **common)

        if ttype == "iou_hungarian":
            return build(
                "tracker", "iou_hungarian",
                **common,
                iou_threshold=cfg.tracking.iou_hungarian.iou_threshold,
            )

        if ttype == "sort":
            sc = cfg.tracking.get("sort", {})
            return build(
                "tracker", "sort",
                **common,
                iou_threshold=sc.get("iou_threshold", 0.3),
                min_hits=sc.get("min_hits", 3),
            )

        if ttype == "tfcr":
            tc = cfg.tracking.get("tfcr", {})
            return build(
                "tracker", "tfcr",
                **common,
                iou_threshold=tc.get("iou_threshold", 0.3),
                eta=tc.get("eta", TFCRTracker.DEFAULT_ETA),
                lam=tc.get("lam", TFCRTracker.DEFAULT_LAM),
                lr=tc.get("lr", TFCRTracker.DEFAULT_LR),
                scale_factors=tc.get("scale_factors", TFCRTracker.DEFAULT_SCALES),
                hog_cell=tc.get("hog_cell", 8),
            )

        raise ValueError(f"Unknown tracker type: {ttype!r}")

    # Modules that accept min_trajectory_len as a constructor argument.
    _MIN_LEN_MODULES = frozenset({"trajectory", "wavelet", "interaction", "dense_trajectories"})

    @classmethod
    def build_extractors(cls, cfg) -> List[FeatureExtractor]:
        """Build the ordered list of FeatureExtractors declared in *cfg*.

        Previously this logic was duplicated in both the sequential runner and
        the parallel worker function.  It lives here now.
        """
        min_len = cfg.tracking.min_trajectory_len
        module_params_section = cfg.features.get("module_params")
        module_params = module_params_section.to_dict() if module_params_section is not None else {}
        extractors: List[FeatureExtractor] = []

        for mod_name in cfg.features.get("modules", []):
            if not is_registered("feature", mod_name):
                log.warning("Feature module %r not registered, skipping it", mod_name)
                continue
            kwargs = {"min_trajectory_len": min_len} if mod_name in cls._MIN_LEN_MODULES else {}
            # Optional per-module constructor arguments from features.module_params,
            # e.g. wavelet: {level: 4} or interaction: {proximity_threshold: 150}.
            kwargs.update(module_params.get(mod_name, {}))
            extractors.append(build("feature", mod_name, **kwargs))

        dt_cfg = cfg.features.get("dense_trajectories", {})
        if dt_cfg and dt_cfg.get("enabled", False):
            extractors.append(build(
                "feature", "dense_trajectories",
                min_trajectory_len=dt_cfg.get("step_size", 5),
                step_size=dt_cfg.get("step_size", 5),
                block_size=dt_cfg.get("block_size", 32),
                feature_types=dt_cfg.get("feature_types", ["trajectory", "hog", "hof", "mbh"]),
            ))

        return extractors

    @staticmethod
    def build_optical_flow(cfg) -> OpticalFlowEstimator:
        return OpticalFlowEstimator.from_config(cfg)

    @classmethod
    def build_all(cls, cfg, detection_source: Optional[DetectionSource] = None):
        """Return (preprocessor, detector, tracker, optical_flow, extractors, aggregator)."""
        preprocessor = cls.build_preprocessor(cfg)
        detector = cls.build_detector(cfg, detection_source)
        tracker = cls.build_tracker(cfg)
        optical_flow = cls.build_optical_flow(cfg)
        extractors = cls.build_extractors(cfg)
        aggregator = FeatureAggregator(
            aggregation_fns=cfg.features.get("aggregation", ["mean", "max", "std"])
        )
        return preprocessor, detector, tracker, optical_flow, extractors, aggregator
