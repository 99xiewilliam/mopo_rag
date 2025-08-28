from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}
    return data


def get_data_input_dir(path: Optional[str] = None) -> Optional[str]:
    cfg = load_config(path)
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    input_dir = data.get("input_dir") if isinstance(data, dict) else None
    return input_dir


def get_pipeline_type(path: Optional[str] = None) -> str:
    cfg = load_config(path)
    return (cfg.get("pipeline") or "modular") if isinstance(cfg, dict) else "modular"


def apply_env_from_config(cfg: Dict[str, Any]) -> None:
    env_map = cfg.get("env") if isinstance(cfg, dict) else None
    if not isinstance(env_map, dict):
        return
    for key, value in env_map.items():
        if not isinstance(key, str):
            continue
        if value is None:
            continue
        os.environ[str(key)] = str(value)



