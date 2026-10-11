"""Frame preprocessing: resize → greyscale → blur → (optional) distortion enhancement.

Three preprocessors are registered:
  * ``passthrough``       — greyscale only, no resize or smoothing.
  * ``standard``          — resize + greyscale + Gaussian blur.
  * ``distortion_aware``  — standard pipeline + per-frame distortion
                            detection and adaptive enhancement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import cv2
import numpy as np

from .interfaces import Preprocessor, PreprocessResult
from .registry import register


# ── Distortion detection ─────────────────────────────────────────────────────

@dataclass
class DistortionReport:
    blur_score: float
    brightness: float
    is_blurry: bool
    is_underexposed: bool
    is_overexposed: bool
    distortion_type: str
    extra_metrics: Dict[str, float] = field(default_factory=dict)


def _detect_distortion(
    gray: np.ndarray,
    blur_threshold: float,
    exposure_low: float,
    exposure_high: float,
    use_multi_metric: bool,
) -> DistortionReport:
    """Compute distortion labels from a single grey frame."""
    if use_multi_metric:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        extra = {
            "laplacian_var": float(lap_var),
            "tenengrad": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
            "brenner": float(np.mean(np.abs(np.diff(gray.astype(np.float64), axis=1)))),
            "blur_score": float(1.0 - min(lap_var / 500.0, 1.0)),
        }
        blur_score = float(lap_var)
    else:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        extra = {}

    brightness = float(gray.mean())
    is_blurry = blur_score < blur_threshold
    is_under = brightness < exposure_low
    is_over = brightness > exposure_high

    if is_blurry and (is_under or is_over):
        dtype = "blurry+exposure"
    elif is_blurry:
        dtype = "blurry"
    elif is_under:
        dtype = "underexposed"
    elif is_over:
        dtype = "overexposed"
    else:
        dtype = "pristine"

    return DistortionReport(
        blur_score=blur_score,
        brightness=brightness,
        is_blurry=is_blurry,
        is_underexposed=is_under,
        is_overexposed=is_over,
        distortion_type=dtype,
        extra_metrics=extra,
    )


# ── Enhancement methods ───────────────────────────────────────────────────────

def _unsharp_mask(gray: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Adaptive unsharp mask: strength scales with local variance."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    adaptive_strength = strength * (1.0 + np.tanh(local_var / 100.0))
    return cv2.addWeighted(gray, 1.0 + adaptive_strength, blurred, -adaptive_strength, 0)


def _wiener_filter(gray: np.ndarray, kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Wiener deconvolution approximation using local statistics."""
    f64 = gray.astype(np.float64)
    mean = cv2.blur(f64, kernel)
    mean_sq = cv2.blur(f64 ** 2, kernel)
    variance = mean_sq - mean ** 2
    noise_mask = variance < np.percentile(variance, 20)
    noise_var = np.mean(variance[noise_mask]) if noise_mask.any() else 100.0
    ratio = np.maximum(0.0, variance - noise_var) / np.maximum(variance, noise_var)
    result = mean + ratio * (f64 - mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def _gamma_correct(gray: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    return cv2.LUT(gray, table.astype(np.uint8))


def _contrast_stretch(gray: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(gray, (2, 98))
    return cv2.normalize(gray, None, int(p2), int(p98), cv2.NORM_MINMAX)  # type: ignore[call-overload]


# ── Preprocessors ─────────────────────────────────────────────────────────────

@register("preprocessor", "passthrough")
class PassthroughPreprocessor(Preprocessor):
    """Convert to greyscale only — useful for debugging the raw signal."""

    def process(self, frame: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return PreprocessResult(
            bgr=frame, gray=gray,
            diagnostics={"enabled": False},
            enhanced=False,
        )


@register("preprocessor", "standard")
class StandardPreprocessor(Preprocessor):
    """Resize → greyscale → Gaussian blur.  No adaptive enhancement."""

    def __init__(
        self,
        frame_resize: Tuple[int, int] = (640, 480),
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_sigma: float = 0,
    ) -> None:
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)
        diagnostics = {
            "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(gray.mean()),
        }
        return PreprocessResult(bgr=bgr, gray=gray, diagnostics=diagnostics, enhanced=False)


@register("preprocessor", "distortion_aware")
class DistortionAwarePreprocessor(Preprocessor):
    """Standard pipeline + per-frame distortion detection and adaptive enhancement.

    Decision logic (per frame):
      1. Detect distortion type from the resized-but-not-yet-enhanced frame.
      2. Apply blur enhancement if the frame is blurry.
      3. Apply exposure enhancement if the frame is under- or over-exposed.
      4. Optionally apply gamma correction and/or contrast stretching.
      5. Apply Gaussian blur (same as StandardPreprocessor) last.
    """

    def __init__(
        self,
        frame_resize: Tuple[int, int] = (640, 480),
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_sigma: float = 0,
        blur_threshold: float = 100.0,
        exposure_low: float = 60.0,
        exposure_high: float = 195.0,
        enhancement_for_blur: Literal["unsharp_mask", "wiener"] = "unsharp_mask",
        enhancement_for_exposure: Literal["clahe", "bilateral"] = "clahe",
        unsharp_sigma: float = 2.0,
        unsharp_strength: float = 1.5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        use_multi_metric: bool = True,
        gamma_correction: bool = False,
        gamma_value: float = 1.2,
        contrast_stretching: bool = False,
    ) -> None:
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma
        self.blur_threshold = blur_threshold
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high
        self.enhancement_for_blur = enhancement_for_blur
        self.enhancement_for_exposure = enhancement_for_exposure
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength
        self.use_multi_metric = use_multi_metric
        self.gamma_correction = gamma_correction
        self.gamma_value = gamma_value
        self.contrast_stretching = contrast_stretching
        # Pre-built CLAHE object (cheaper than constructing per-frame).
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=tuple(clahe_tile_grid),
        )
        self._bilateral = {"d": bilateral_d, "sc": bilateral_sigma_color, "ss": bilateral_sigma_space}

    # ── Enhancement helpers (each returns a modified copy of gray) ───────────

    def _enhance_blur(self, gray: np.ndarray) -> np.ndarray:
        if self.enhancement_for_blur == "unsharp_mask":
            return _unsharp_mask(gray, self.unsharp_sigma, self.unsharp_strength)
        if self.enhancement_for_blur == "wiener":
            return _wiener_filter(gray)
        return gray

    def _enhance_exposure(self, gray: np.ndarray) -> np.ndarray:
        if self.enhancement_for_exposure == "clahe":
            return self._clahe.apply(gray)
        if self.enhancement_for_exposure == "bilateral":
            b = self._bilateral
            return cv2.bilateralFilter(gray, b["d"], b["sc"], b["ss"])
        return gray

    # ── Main process method ───────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        report = _detect_distortion(
            gray,
            self.blur_threshold,
            self.exposure_low,
            self.exposure_high,
            self.use_multi_metric,
        )

        enhanced = False

        if self.gamma_correction and (report.is_underexposed or report.is_overexposed):
            gray = _gamma_correct(gray, self.gamma_value)
            enhanced = True

        if report.is_blurry:
            gray = self._enhance_blur(gray)
            enhanced = True

        if report.is_underexposed or report.is_overexposed:
            gray = self._enhance_exposure(gray)
            enhanced = True

        if self.contrast_stretching and (report.is_underexposed or report.is_overexposed):
            gray = _contrast_stretch(gray)
            enhanced = True

        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)

        diagnostics: Dict[str, object] = {
            "blur_score": report.blur_score,
            "brightness": report.brightness,
            "distortion_type": report.distortion_type,
            "enhanced": enhanced,
            "is_blurry": report.is_blurry,
            "is_underexposed": report.is_underexposed,
            "is_overexposed": report.is_overexposed,
            **report.extra_metrics,
        }
        return PreprocessResult(bgr=bgr, gray=gray, enhanced=enhanced, diagnostics=diagnostics)
