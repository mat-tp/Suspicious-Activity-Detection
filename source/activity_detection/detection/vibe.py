"""ViBe background subtraction (Barnich and Van Droogenbroeck, 2011).

Each pixel keeps ``n_samples`` past values and a new value is background if at least ``min_matches``
of them are within ``radius``. Background pixels overwrite one of their own samples, and one of a random
neighbour, with probability 1/subsampling. This is a NumPy version on grey frames with the default
parameters of the paper (N=20, R=20, #min=2, phi=16).
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .background_model import BackgroundModelDetector

# the 8 neighbours of a pixel
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shifted(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """image moved by (dy, dx), edges repeated."""
    padded = np.pad(image, 1, mode="edge")
    h, w = image.shape
    return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


@register("detector", "vibe")
class ViBeDetector(BackgroundModelDetector):
    """ViBe background model.

    Args:
        n_samples: stored samples per pixel.
        radius: largest grey-level distance for a sample to match.
        min_matches: matches needed for a pixel to be background.
        subsampling: a background pixel updates the model with probability 1/subsampling.
        seed: seed of the random replacement.

    The mask-cleaning options are those of ``BackgroundModelDetector``.
    """

    def __init__(self, n_samples: int = 20, radius: float = 20, min_matches: int = 2,
                 subsampling: int = 16, seed: int = 0, **post):
        super().__init__(**post)
        self.n_samples = n_samples
        self.radius = radius
        self.min_matches = min_matches
        self.subsampling = subsampling
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Re-seed the random generator and drop the stored samples."""
        self.rng = np.random.default_rng(self.seed)
        self.samples = None          # (n_samples, H, W)

    def _initialise(self, gray: np.ndarray) -> None:
        """Fill every pixel's samples from its own value and its neighbours' values."""
        h, w = gray.shape
        layers = []
        for _ in range(self.n_samples):
            dy, dx = NEIGHBOURS[self.rng.integers(len(NEIGHBOURS))]
            layers.append(shifted(gray, dy, dx))
        self.samples = np.stack(layers).astype(np.int16)

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """Boolean foreground mask of *gray*; the first frame sets up the model and gives an empty mask."""
        if self.samples is None:
            self._initialise(gray)
            return np.zeros(gray.shape, dtype=bool)

        pixel = gray.astype(np.int16)
        matches = (np.abs(self.samples - pixel) < self.radius).sum(axis=0)
        foreground = matches < self.min_matches
        self._update(pixel, ~foreground)
        return foreground

    def _update(self, pixel: np.ndarray, background: np.ndarray) -> None:
        h, w = pixel.shape
        rows, cols = np.mgrid[0:h, 0:w]

        # 1. a background pixel replaces one of its own samples
        own = background & (self.rng.integers(self.subsampling, size=(h, w)) == 0)
        slot = self.rng.integers(self.n_samples, size=(h, w))
        self.samples[slot[own], rows[own], cols[own]] = pixel[own]

        # 2. ...and one sample of a random neighbour
        spread = background & (self.rng.integers(self.subsampling, size=(h, w)) == 0)
        choice = self.rng.integers(len(NEIGHBOURS), size=(h, w))
        dy = np.array([n[0] for n in NEIGHBOURS])[choice]
        dx = np.array([n[1] for n in NEIGHBOURS])[choice]
        nr = np.clip(rows + dy, 0, h - 1)
        nc = np.clip(cols + dx, 0, w - 1)
        slot = self.rng.integers(self.n_samples, size=(h, w))
        self.samples[slot[spread], nr[spread], nc[spread]] = pixel[spread]
