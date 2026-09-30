"""
Shared configuration-loading utilities for OrbitGuard AI.

Centralizing this here means every script (training, evaluation, dashboard)
reads config.yaml / subsystems.yaml the same way, with the same error
messages when something is missing or malformed.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
DEFAULT_SUBSYSTEMS_PATH = os.path.join(PROJECT_ROOT, "config", "subsystems.yaml")


class ConfigError(Exception):
    """Raised when configuration is missing or malformed."""


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    return data


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and lightly validate config.yaml."""
    cfg = load_yaml(path)
    required_top_level = ["dataset", "split", "lstm", "isolation_forest", "fusion", "severity", "trend"]
    missing = [k for k in required_top_level if k not in cfg]
    if missing:
        raise ConfigError(
            f"config.yaml is missing required section(s): {', '.join(missing)}"
        )
    return cfg


def load_subsystems(path: str = DEFAULT_SUBSYSTEMS_PATH) -> dict:
    cfg = load_yaml(path)
    if "mapping" not in cfg:
        raise ConfigError("subsystems.yaml is missing required 'mapping' section")
    return cfg


def resolve_path(relative_or_absolute: str) -> str:
    """Resolve a config-supplied path relative to the project root."""
    if os.path.isabs(relative_or_absolute):
        return relative_or_absolute
    return os.path.join(PROJECT_ROOT, relative_or_absolute)


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested config value using 'a.b.c' notation."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
