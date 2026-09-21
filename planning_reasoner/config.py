"""Small, strict configuration helpers used by research scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and its optional relative ``base`` recursively."""
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    base_value = config.pop("base", None)
    if base_value is None:
        return config
    base_path = (config_path.parent / base_value).resolve()
    if base_path == config_path:
        raise ValueError("A config cannot inherit from itself")
    return deep_merge(load_experiment_config(base_path), config)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require_keys(config: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys.difference(config))
    if missing:
        raise ValueError(f"Missing {context} keys: {', '.join(missing)}")
