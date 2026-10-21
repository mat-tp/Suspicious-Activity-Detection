"""SuBSENSE background subtraction (St-Charles, Bilodeau and Bergevin, 2015).

It combines colour and local binary similarity patterns with per-pixel feedback on its thresholds and
learning rates. This class uses the reference implementation of BGSLibrary through ``pybgs``
(``pip install pybgs``), which compiles C++ against OpenCV and can fail to install. If it is missing,
building the detector raises an error that says so and the experiment is skipped.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .background_model import BackgroundModelDetector

# pybgs is optional and can fail to install, so a missing package is reported when the detector is built.
try:
    import pybgs
except ImportError:
    pybgs = None


@register("detector", "subsense")
class SuBSENSEDetector(BackgroundModelDetector):
    """SuBSENSE through the BGSLibrary implementation (``pybgs``); raises ``ImportError`` if it is missing.

    Args:
        **post: mask-cleaning options, see ``BackgroundModelDetector``.
    """

    def __init__(self, **post):
        super().__init__(**post)
        self._import_pybgs()
        self.reset()

    @staticmethod
    def _import_pybgs():
        if pybgs is None:
            raise ImportError(
                "The SuBSENSE detector needs the pybgs package (pip install pybgs). "
                "It did not import here; see docs/notes/howToRun.md.")
        return pybgs

    def reset(self) -> None:
        """Create a fresh SuBSENSE model for a new video."""
        self.model = self._import_pybgs().SuBSENSE()

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """Boolean foreground mask of *gray* (the model's output thresholded at 127)."""
        return self.model.apply(gray) > 127
