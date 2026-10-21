"""Name -> class registry that lets YAML configs select detectors, trackers, features and models by name."""

from __future__ import annotations

from typing import Any, Callable, Dict

# This is where the registered classes are kept: one dict of name -> class for each component
# kind.
_REGISTRIES: Dict[str, Dict[str, type]] = {
    "detector": {}, "tracker": {}, "preprocessor": {},
    "feature": {}, "model": {},
}


def register(category: str, name: str) -> Callable:
    """Class decorator that registers the class as (*category*, *name*); it raises if the name is taken."""
    if category not in _REGISTRIES:
        raise KeyError(f"Unknown category '{category}'. Valid: {list(_REGISTRIES.keys())}")
    def decorator(cls: type) -> type:
        if name in _REGISTRIES[category]:
            raise ValueError(f"'{name}' already registered under '{category}'")
        _REGISTRIES[category][name] = cls
        return cls
    return decorator


def build(category: str, name: str, **kwargs) -> Any:
    """Create the class registered as (*category*, *name*) with ``**kwargs``; ``KeyError`` lists the known names.
    """
    if category not in _REGISTRIES or name not in _REGISTRIES[category]:
        available_list = list(_REGISTRIES.get(category, {}).keys())
        raise KeyError(f"'{name}' not registered under '{category}'. Available: {available_list}")
    return _REGISTRIES[category][name](**kwargs)


def available(category: str) -> list:
    return sorted(_REGISTRIES.get(category, {}).keys())


def is_registered(category: str, name: str) -> bool:
    return name in _REGISTRIES.get(category, {})


def get_class(category: str, name: str) -> type:
    """Return the registered class (not an instance), e.g. to inspect its arguments."""
    if not is_registered(category, name):
        raise KeyError(f"'{name}' not registered under '{category}'. Available: {available(category)}")
    return _REGISTRIES[category][name]
