from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Tuple

def _require_pyyaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load --policy YAML files. "
            "Install it (pip install pyyaml) or run without --policy."
        ) from e

def load_yaml_policy(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("policy path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"policy file not found: {path}")

    yaml = _require_pyyaml()
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"policy file must parse to a dict, got: {type(obj)}")

    # Minimal required fields
    if "policy_name" not in obj:
        raise RuntimeError("policy missing required key: policy_name")

    return obj

def policy_hash(policy_obj: Dict[str, Any]) -> str:
    """
    Deterministic hash:
      dict -> canonical JSON (sorted keys) -> sha256
    """
    canonical = json.dumps(policy_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def load_policy_and_hash(path: str) -> Tuple[Dict[str, Any], str]:
    obj = load_yaml_policy(path)
    h = policy_hash(obj)
    return obj, h
