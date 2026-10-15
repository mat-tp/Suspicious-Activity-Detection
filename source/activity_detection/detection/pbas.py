"""Pixel-Based Adaptive Segmenter, PBAS (Hofmann, Tiefenbacher and Rigoll, 2012).

Like ViBe, each pixel keeps a history of past values, but two quantities adapt
per pixel from how much that pixel's background fluctuates:

  decision threshold R(x)  grows where the background is dynamic, shrinks where
                           it is static
  learning rate 1/T(x)     the model updates faster where the background is
                           static, and slower around foreground

The distance between the current pixel and a stored sample mixes the intensity
difference with the difference in gradient magnitude (weight alpha / mean
gradient). This is a NumPy implementation on grey frames with the original publication's
default parameters.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import register
from .background_model import BackgroundModelDetector
from .vibe import NEIGHBOURS, shifted


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


@register("detector", "pbas")
class PBASDetector(BackgroundModelDetector):

    def __init__(self, n_samples: int = 20, min_matches: int = 2,
                 alpha: float = 7.0, beta: float = 1.0,
                 r_lower: float = 18.0, r_scale: float = 5.0, r_inc_dec: float = 0.05,
                 t_lower: float = 2.0, t_upper: float = 200.0,
                 t_inc: float = 1.0, t_dec: float = 0.05, seed: int = 0, **post):
        super().__init__(**post)
        self.n_samples = n_samples
        self.min_matches = min_matches
        self.alpha = alpha
        self.beta = beta
        self.r_lower = r_lower
        self.r_scale = r_scale
        self.r_inc_dec = r_inc_dec
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.t_inc = t_inc
        self.t_dec = t_dec
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.values = None           # (N, H, W) intensities
        self.gradients = None        # (N, H, W) gradient magnitudes

    def _initialise(self, gray: np.ndarray) -> None:
        h, w = gray.shape
        grad = gradient_magnitude(gray)
        values, grads = [], []
        for _ in range(self.n_samples):
            dy, dx = NEIGHBOURS[self.rng.integers(len(NEIGHBOURS))]
            values.append(shifted(gray, dy, dx))
            grads.append(shifted(grad, dy, dx))
        self.values = np.stack(values).astype(np.float32)
        self.gradients = np.stack(grads).astype(np.float32)
        self.R = np.full((h, w), self.r_lower, dtype=np.float32)
        self.T = np.full((h, w), 20.0, dtype=np.float32)
        # recent minimal distances per pixel, as a ring buffer; start small but non-zero
        self.min_dist = np.full((self.n_samples, h, w), 1.0, dtype=np.float32)
        self.min_dist_pos = 0
        self.mean_gradient = max(float(grad.mean()), 1.0)

    def segment(self, gray: np.ndarray) -> np.ndarray:
        if self.values is None:
            self._initialise(gray)
            return np.zeros(gray.shape, dtype=bool)

        pixel = gray.astype(np.float32)
        grad = gradient_magnitude(gray)

        distance = (self.alpha / self.mean_gradient * np.abs(grad - self.gradients)
                    + self.beta * np.abs(pixel - self.values))          # (N, H, W)
        matches = (distance < self.R).sum(axis=0)
        foreground = matches < self.min_matches
        background = ~foreground

        # average of the recent minimal distances: how dynamic this pixel's background is
        d_min = distance.min(axis=0)
        self.min_dist[self.min_dist_pos] = np.where(background, d_min, self.min_dist[self.min_dist_pos])
        self.min_dist_pos = (self.min_dist_pos + 1) % self.n_samples
        d_mean = np.maximum(self.min_dist.mean(axis=0), 1e-3)

        self._adapt_threshold(d_mean)
        self._adapt_rate(foreground, d_mean)
        self._update_history(pixel, grad, background)
        self.mean_gradient = max(float(grad.mean()), 1.0)
        return foreground

    def _adapt_threshold(self, d_mean: np.ndarray) -> None:
        """R(x) follows the average minimal distance, scaled by r_scale."""
        too_high = self.R > d_mean * self.r_scale
        self.R = np.where(too_high, self.R * (1 - self.r_inc_dec), self.R * (1 + self.r_inc_dec))
        self.R = np.maximum(self.R, self.r_lower)

    def _adapt_rate(self, foreground: np.ndarray, d_mean: np.ndarray) -> None:
        """T(x) goes up around foreground (slower updates) and down on static background."""
        step = np.where(foreground, self.t_inc / d_mean, -self.t_dec / d_mean)
        self.T = np.clip(self.T + step, self.t_lower, self.t_upper)

    def _update_history(self, pixel, grad, background) -> None:
        h, w = pixel.shape
        rows, cols = np.mgrid[0:h, 0:w]
        chance = 1.0 / self.T

        own = background & (self.rng.random((h, w)) < chance)
        slot = self.rng.integers(self.n_samples, size=(h, w))
        self.values[slot[own], rows[own], cols[own]] = pixel[own]
        self.gradients[slot[own], rows[own], cols[own]] = grad[own]

        spread = background & (self.rng.random((h, w)) < chance)
        choice = self.rng.integers(len(NEIGHBOURS), size=(h, w))
        dy = np.array([n[0] for n in NEIGHBOURS])[choice]
        dx = np.array([n[1] for n in NEIGHBOURS])[choice]
        nr, nc = np.clip(rows + dy, 0, h - 1), np.clip(cols + dx, 0, w - 1)
        slot = self.rng.integers(self.n_samples, size=(h, w))
        self.values[slot[spread], nr[spread], nc[spread]] = pixel[spread]
        self.gradients[slot[spread], nr[spread], nc[spread]] = grad[spread]
