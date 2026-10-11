"""Optional third-party libraries.

Each library is imported once here. If it is missing, its name is set to None
and the matching *_AVAILABLE flag is False, so the rest of the code can check
the flag instead of wrapping every import in try/except.
"""
import logging

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb, LIGHTGBM_AVAILABLE = None, False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb, XGBOOST_AVAILABLE = None, False

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    pywt, PYWAVELETS_AVAILABLE = None, False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt, sns, MATPLOTLIB_AVAILABLE = None, None, False

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = tv_models = T = DataLoader = TorchDataset = None
    TORCH_AVAILABLE = False

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except ImportError:
    fo = foz = None
    FIFTYONE_AVAILABLE = False

_MISSING = {
    "LightGBM": LIGHTGBM_AVAILABLE, "XGBoost": XGBOOST_AVAILABLE,
    "PyWavelets": PYWAVELETS_AVAILABLE, "Matplotlib/Seaborn": MATPLOTLIB_AVAILABLE,
    "PyTorch": TORCH_AVAILABLE, "FiftyOne": FIFTYONE_AVAILABLE,
}
for _name, _ok in _MISSING.items():
    if not _ok:
        log.info("%s not installed - the features that need it are disabled.", _name)
