from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import CONFIGS_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)


class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) if isinstance(v, dict) else v for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list:
        return list(self._data.keys())

    def items(self):
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(v):
            if isinstance(v, Config):
                return {k: unwrap(val) for k, val in v._data.items()}
            if isinstance(v, list):
                return [unwrap(item) for item in v]
            return v
        return unwrap(self)

    def __repr__(self):
        return f"Config({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_paths(data: dict, root: Path) -> dict:
    data = copy.deepcopy(data)
    if "paths" in data:
        for key, rel in data["paths"].items():
            if isinstance(rel, str) and not Path(rel).is_absolute():
                data["paths"][key] = str((root / rel).resolve())
    return data


def _load_single_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_with_extends(config_path: Path, configs_dir: Path, visited: set = None) -> dict:
    if visited is None:
        visited = set()
    config_path = config_path.resolve()
    if config_path in visited:
        log.warning("Circular extends detected: %s", config_path)
        return {}
    visited.add(config_path)
    data = _load_single_yaml(config_path)
    if not data:
        return {}
    extends = data.pop("extends", None)
    if not extends:
        return data
    parent_path = configs_dir / extends
    if not parent_path.suffix:
        parent_path = parent_path.with_suffix(".yaml")
    if not parent_path.exists():
        log.warning("Parent config not found: %s", parent_path)
        return data
    parent_data = _load_with_extends(parent_path, configs_dir, visited)
    return _deep_merge(parent_data, data)


def load_config(group_name: str, overrides: dict = None) -> Config:
    """Load source/configs/<group_name>.yaml, resolving `extends` and relative paths."""
    configs_dir = CONFIGS_DIR
    path = Path(group_name)
    if not path.is_absolute():
        if not path.suffix:
            path = path.with_suffix(".yaml")
        path = configs_dir / path
    if not path.exists():
        available_list = [p.stem for p in configs_dir.glob("*.yaml") if p.stem != "base"]
        raise FileNotFoundError(f"Config '{group_name}' not found at {path}.\nAvailable: {available_list}")
    merged = _load_with_extends(path, configs_dir)
    if not merged:
        raise ValueError(f"Failed to load config: {group_name}")
    merged = _resolve_paths(merged, PROJECT_ROOT)
    if overrides:
        merged = _deep_merge(merged, overrides)
    config = Config(merged)
    if not hasattr(config, "paths"):
        log.warning("No 'paths' section in config. Available keys: %s", list(config.keys()))
    return config
