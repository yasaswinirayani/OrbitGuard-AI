"""
Maps telemetry feature names to subsystems, using config/subsystems.yaml.
No subsystem is ever guessed for an unmatched feature — it is reported
using the configured 'unmapped_label' (default: "Unknown / Unmapped").
"""

from __future__ import annotations


def map_feature_to_subsystem(feature_name: str, subsystems_cfg: dict) -> str:
    mapping = subsystems_cfg.get("mapping", {})
    unmapped_label = subsystems_cfg.get("unmapped_label", "Unknown / Unmapped")

    feature_lower = feature_name.lower()
    for subsystem, keywords in mapping.items():
        for keyword in keywords:
            if str(keyword).lower() in feature_lower:
                return subsystem
    return unmapped_label


def map_features_to_subsystems(feature_names: list, subsystems_cfg: dict) -> dict:
    """Return {feature_name: subsystem} for a list of feature names."""
    return {f: map_feature_to_subsystem(f, subsystems_cfg) for f in feature_names}


def affected_subsystems(top_features: list, subsystems_cfg: dict) -> list:
    """
    Given a list of top-contributing feature dicts (from attribution.py),
    return the ordered, de-duplicated list of subsystems they map to.
    """
    subsystems = []
    for entry in top_features:
        subsystem = map_feature_to_subsystem(entry["feature"], subsystems_cfg)
        if subsystem not in subsystems:
            subsystems.append(subsystem)
    return subsystems
