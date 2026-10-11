"""SuBSENSE background subtraction (St-Charles, Bilodeau and Bergevin, 2015).

SuBSENSE combines colour and local binary similarity patterns with per-pixel
feedback on its thresholds and learning rates. It is long to implement, so this
class uses the reference implementation in BGSLibrary through the `pybgs`
package instead of a re-implementation:

    pip install pybgs

pybgs compiles C++ code against OpenCV, so installation can fail on some
systems. If it is missing, building this detector raises an error that says so,
and the experiment is skipped.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .background_model import BackgroundModelDetector


@register("detector", "subsense")
class SuBSENSEDetector(BackgroundModelDetector):

    def __init__(self, **post):
        super().__init__(**post)
        self._import_pybgs()
        self.reset()

    @staticmethod
    def _import_pybgs():
        try:
            import pybgs
        except ImportError as err:
            raise ImportError(
                "The SuBSENSE detector needs the pybgs package (pip install pybgs). "
                "It did not import here; see docs/howToRun.md.") from err
        return pybgs

    def reset(self) -> None:
        self.model = self._import_pybgs().SuBSENSE()

    def segment(self, gray: np.ndarray) -> np.ndarray:
        return self.model.apply(gray) > 127
