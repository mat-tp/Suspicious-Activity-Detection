from __future__ import annotations

import json
import logging
import pickle
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..config import Config
from ..dataset_info import DatasetInfoHandler
from ..evaluation.cross_validation import check_training_viability, cross_validate_model
from ..evaluation.metrics import confusion, scores_from_confusion
from ..evaluation.report import (
    print_summary,
    write_per_class_csv,
    write_predictions,
    write_summary_csv,
)
from ..experiment.collector import VideoFeatureCollector
from ..experiment.utils import get_git_commit, get_next_run_id
from ..models.imbalance import ImbalanceConfig, build_model
from ..optional import TORCH_AVAILABLE, torch
from ..paths import EXPERIMENTS_DIR, PROJECT_ROOT
from ..pipeline.builder import DetectionSource
from ..pipeline.smoothing import TemporalSmoother
from ..registry import build
from ..viz.fiftyone_viz import FiftyOneVisualizer
from ..viz.plotting import PlotManager

log = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: Config, experiment_name: str = None,
                 detection_source: Optional[DetectionSource] = None,
                 diagnose_only: bool = False,
                 visualise: bool = False,
                 n_visualise: int = 3):
        self.config = config
        self.experiment_name = experiment_name or config.experiment.name
        self.seed = config.experiment.get("seed", 42)
        self.is_comparison = config.get("comparison_only", False)
        self.detection_source = detection_source
        self.diagnose_only = diagnose_only
        self.visualise = visualise
        self.n_visualise = n_visualise

        if config.dataset.get("use_dataset_info", False):
            self.dataset_info = DatasetInfoHandler(config.paths.dataset_excel)
        else:
            self.dataset_info = None

        exp_dir = EXPERIMENTS_DIR / self.experiment_name
        self.run_id = get_next_run_id(exp_dir)
        self.run_dir = exp_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.plots = PlotManager(self.run_dir / "plots")
        self.fo_visualizer = FiftyOneVisualizer(self.run_dir / "fiftyone")

        random.seed(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)

        if self.is_comparison:
            print(f"\n  [NOTE] Group {config.experiment.group} is a comparison baseline.")
        if self.dataset_info and config.dataset.get("justify_parameters", False):
            self._print_parameter_justifications()

    def _print_parameter_justifications(self):
        print("\n" + "=" * 70)
        print("  PARAMETER JUSTIFICATIONS (from datasetInfo.xlsx analysis)")
        print("=" * 70)
        print(self.dataset_info.justify_resize_dimensions(tuple(self.config.preprocessing.frame_resize)))
        print(self.dataset_info.justify_min_trajectory_length(self.config.tracking.min_trajectory_len))
        print("=" * 70)

    def load_dataset(self) -> pd.DataFrame:
        if hasattr(self.config, 'paths') and hasattr(self.config.paths, 'dataset_excel'):
            excel_path = self.config.paths.dataset_excel
        else:
            excel_path = "data/datasetInfo.xlsx"
        excel_path = Path(excel_path)
        if not excel_path.is_absolute():
            excel_path = PROJECT_ROOT / excel_path
        if not excel_path.exists():
            raise FileNotFoundError(f"Dataset Excel not found: {excel_path}")
        df = pd.read_excel(excel_path)
        required = {"Activity", "Distortion", "Name of Video Series"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Excel missing columns: {missing}")
        return df

    def prepare_clip_dataset(self, df: pd.DataFrame, use_sample: bool = False) -> Dict:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")
        target_col = self.config.targets[0] if self.config.targets else "activity"
        col_mapping = {"activity": "Activity", "distortion": "Distortion"}
        actual_col = col_mapping.get(target_col, target_col)
        if actual_col not in df.columns:
            raise KeyError(f"Column '{actual_col}' not found. Available: {df.columns.tolist()}")
        groups = {}
        for target_val, grp in df.groupby(actual_col):
            pool = grp["Name of Video Series"].tolist()
            if use_sample:
                n = self.config.dataset.get("n_random_videos_per_group", 5)
                pool = random.sample(pool, min(n, len(pool)))
            groups[target_val] = pool
        video_paths = []
        labels = []
        for label, vnames in groups.items():
            for vname in vnames:
                vpath = Path(self.config.paths.video_dir) / vname
                if vpath.exists():
                    video_paths.append(str(vpath))
                    labels.append(label)
        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        return {"X": video_paths, "y": y_enc, "label_encoder": le, "classes": le.classes_}

    def train_neural_model(self, clip_data: Dict, target_col: str = "activity") -> Dict:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        X = clip_data["X"]
        y = clip_data["y"]
        classes = clip_data["classes"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.training.test_size,
                                                                stratify=y, random_state=self.seed)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.training.test_size,
                                                                random_state=self.seed)
        results = {}
        clip_len = getattr(self.config.clip, "clip_len", 16)
        frame_size = tuple(getattr(self.config.clip, "frame_size", [112, 112]))
        batch_size = getattr(self.config.clip, "batch_size", 8)

        for model_def in self.config.models.active:
            model_name = model_def.get("name", "")
            params = model_def.get("params", {})
            print(f"\n{'='*60}\n  Neural Model: {model_name}  |  Target: {target_col}")
            try:
                model = build("model", model_name, random_state=self.seed, **params)
                model.clip_len = clip_len
                model.frame_size = frame_size
                model.batch_size = batch_size
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
                self.plots.confusion_matrix_plot(cm, classes.tolist(), f"{model_name} — {target_col}",
                                                 f"cm_{model_name}_{target_col}.png")
                results[model_name] = {"accuracy": acc, "f1_weighted": f1, "precision": prec, "recall": rec,
                                       "confusion_matrix": cm.tolist(), "y_true": y_test, "y_pred": y_pred,
                                       "classes": classes.tolist()}
                if hasattr(model, "_model"):
                    torch.save(model._model.state_dict(), self.run_dir / f"{model_name}_{target_col}.pt")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                results[model_name] = {"error": str(e)}
        with open(self.run_dir / f"results_{target_col}_neural.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    def collect_video_features(self, df: pd.DataFrame, use_sample: bool = False) -> List[Dict]:
        """Run detection -> tracking -> features over every video (see collector.py)."""
        collector = VideoFeatureCollector(
            self.config, self.detection_source, self.dataset_info,
            run_dir=self.run_dir, plots=self.plots, fo_visualizer=self.fo_visualizer,
            visualise=self.visualise, n_visualise=self.n_visualise,
        )
        return collector.collect(df, use_sample)

    def train_and_evaluate(self, video_features: List[Dict], target_col: str = "activity") -> Dict:
        """Cross-validate every configured model on the video-level feature vectors."""
        if not video_features:
            print(f"\nNo video features for target '{target_col}'.")
            return {}

        X, y_enc, le, feature_keys = self._build_matrix(video_features, target_col)
        class_names = le.classes_.tolist()
        print(f"\nFeature matrix: {X.shape[0]} videos x {X.shape[1]} features")
        print(f"Target: {target_col}")
        print(f"Class distribution:\n{pd.Series(le.inverse_transform(y_enc)).value_counts().to_string()}")

        training = self.config.training
        n_folds = training.get("cv_folds", 5)
        viable, msg = check_training_viability(y_enc, n_folds=n_folds,
                                               min_per_class=training.get("min_per_class", 3))
        if not viable:
            log.warning("Training skipped: %s", msg)
            return {"error": msg, "viable": False}

        imbalance = ImbalanceConfig.from_config(self.config.get("imbalance"))
        print(f"Imbalance handling: {imbalance.describe()}")

        results: Dict[str, Dict] = {}
        cv_by_model: Dict[str, Dict] = {}
        oof_predictions: Dict[str, np.ndarray] = {}
        importances: Dict[str, Dict[str, float]] = {}

        for model_def in self.config.models.active:
            name, params = self._model_name_and_params(model_def)
            print(f"\n{'=' * 60}\n  Model: {name}  |  Target: {target_col}"
                  + ("  [COMPARISON]" if self.is_comparison else ""))
            try:
                model = build_model(name, params, imbalance, self.seed)
                cv = cross_validate_model(
                    model, X, y_enc, class_names, n_splits=n_folds,
                    n_repeats=training.get("cv_repeats", 1), seed=self.seed,
                    n_jobs=training.get("n_jobs", 1), n_boot=training.get("n_bootstrap", 1000))
                oof_predictions[name] = cv.pop("_oof_pred")
                print_summary(name, cv)
                cv_by_model[name] = cv

                # Fit on all the data, for feature importances and the saved model.
                model.fit(X, y_enc)
                self._save_importances(model, name, target_col, feature_keys, importances)
                self.plots.confusion_matrix_plot(
                    np.array(cv["pooled"]["confusion_matrix"]), class_names,
                    f"{name} - {target_col} (out-of-fold)", f"cm_{name}_{target_col}.png")
                with open(self.run_dir / f"{name}_{target_col}.pkl", "wb") as f:
                    pickle.dump({"model": model, "label_encoder": le, "feature_keys": feature_keys}, f)

                pooled = cv["pooled"]
                results[name] = {
                    "accuracy": pooled["accuracy"],
                    "balanced_accuracy": pooled["balanced_accuracy"],
                    "macro_f1": pooled["macro_f1"],
                    "cv": cv,
                    "imbalance": imbalance.describe(),
                    "classes": class_names,
                    "feature_count": len(feature_keys),
                    "is_comparison": self.is_comparison,
                }
            except Exception as err:
                print(f"  FAILED: {err}")
                traceback.print_exc()
                results[name] = {"error": str(err)}

        if oof_predictions:
            write_predictions(
                self.run_dir / f"predictions_{target_col}.csv",
                [fv["video_name"] for fv in video_features], le.inverse_transform(y_enc).tolist(),
                {name: le.inverse_transform(pred).tolist() for name, pred in oof_predictions.items()})
        self._add_smoothed_ensemble(results, oof_predictions, y_enc)
        self._save_results(results, cv_by_model, importances, imbalance, target_col,
                           feature_keys, len(video_features))
        return results

    # ---- helpers for train_and_evaluate -----------------------------------

    @staticmethod
    def _build_matrix(video_features: List[Dict], target_col: str):
        meta_keys = {"activity", "distortion", "video_name", "n_tracks", "frames_used",
                     "track_length", "feature_count", "track_confidence"}
        feature_keys = sorted(set().union(*(fv.keys() for fv in video_features)) - meta_keys)
        X = np.array([[fv.get(k, 0.0) for k in feature_keys] for fv in video_features])
        le = LabelEncoder()
        y_enc = le.fit_transform([fv[target_col] for fv in video_features])
        return X, y_enc, le, feature_keys

    @staticmethod
    def _model_name_and_params(model_def):
        if isinstance(model_def, (dict, Config)):
            return model_def.get("name", ""), model_def.get("params", {}) or {}
        return model_def, {}

    def _save_importances(self, model, name, target_col, feature_keys, store):
        values = model.feature_importances_
        if values is None or len(values) != len(feature_keys):
            return
        for i in np.argsort(values)[-10:][::-1]:
            print(f"    {feature_keys[i]:40s}  {values[i]:.4f}")
        store[name] = {k: float(v) for k, v in zip(feature_keys, values)}
        self.plots.feature_importance_plot(
            feature_names=feature_keys, importances=values,
            title=f"{name} - feature importance ({target_col})",
            filename=f"importance_{name}_{target_col}.png")

    def _add_smoothed_ensemble(self, results, oof_predictions, y_enc):
        cfg = self.config.get("smoothing", None)
        if not (cfg and cfg.get("enabled", False) and len(oof_predictions) > 1):
            return
        method = cfg.get("method", "majority_vote")
        smoother = TemporalSmoother(method=method, ema_alpha=cfg.get("ema_alpha", 0.7))
        accuracies = {n: float((p == y_enc).mean()) for n, p in oof_predictions.items()}
        smoothed = smoother.smooth(oof_predictions, accuracies)
        scores = scores_from_confusion(confusion(y_enc, smoothed, int(y_enc.max()) + 1))
        print(f"\n  [{method}] Smoothed ensemble - accuracy {scores['accuracy']:.4f}, "
              f"macro-F1 {scores['macro_f1']:.4f}")
        results["__smoothed__"] = {"method": method, **scores, "y_pred": smoothed.tolist()}

    def _save_results(self, results, cv_by_model, importances, imbalance, target_col,
                      feature_keys, n_videos):
        self.plots.model_comparison_plot(results, "macro_f1", target_col, f"model_macro_f1_{target_col}.png")
        self.plots.model_comparison_plot(results, "accuracy", target_col, f"model_accuracy_{target_col}.png")
        if importances:
            with open(self.run_dir / f"feature_importance_{target_col}.json", "w") as f:
                json.dump(importances, f, indent=2)
        if cv_by_model:
            write_per_class_csv(self.run_dir / f"per_class_{target_col}.csv", cv_by_model)
            write_summary_csv(self.run_dir / f"summary_{target_col}.csv", cv_by_model, imbalance.describe())
        with open(self.run_dir / f"results_{target_col}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(self.run_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        meta = {
            "run_id": self.run_id, "experiment": self.experiment_name,
            "git_commit": get_git_commit(), "config": self.config.experiment.name,
            "target": target_col, "feature_count": len(feature_keys), "n_videos": n_videos,
            "imbalance": imbalance.describe(), "is_comparison": self.is_comparison,
            "timestamp": datetime.now().isoformat(),
            "detection_source": self.detection_source.value if self.detection_source else "config",
        }
        with open(self.run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nResults saved to: {self.run_dir}")
