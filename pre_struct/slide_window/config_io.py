"""Local config I/O for slide_window (no imports from pre_struct.ebqa).

Config schema is compatible with EBQA to ease transition, but defaults
to files within this package if present.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> str:
    """Prefer a config colocated with slide_window; fallback to EBQA config if missing."""
    local = Path(__file__).with_name("ebqa_config.json")
    if local.is_file():
        return str(local)
    # fallback to EBQA config path if a local config hasn't been created yet
    alt = Path(__file__).parents[1] / "ebqa" / "ebqa_config.json"
    return str(alt)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or default_config_path()
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    return data


def _require_str(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(f"Missing or invalid '{key}' in config")
    return v.strip()


def resolve_model_dir(cfg: Dict[str, Any]) -> str:
    return _require_str(cfg, "model_dir")


def resolve_tokenizer_name(cfg: Dict[str, Any]) -> str:
    for k in ("tokenizer_name_or_path", "tokenizer_name", "model_name_or_path"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError("Missing tokenizer path: set 'tokenizer_name_or_path' in config")


def resolve_report_struct_path(cfg: Dict[str, Any]) -> str:
    return _require_str(cfg, "report_struct_path")


def lengths_from(cfg: Dict[str, Any]) -> Dict[str, int]:
    max_seq_len = int(cfg["max_seq_len"])
    max_tokens_ctx = int(cfg["max_tokens_ctx"])
    doc_stride = int(cfg.get("doc_stride", max(64, max_tokens_ctx // 4)))
    return {
        "max_seq_len": max_seq_len,
        "max_tokens_ctx": max_tokens_ctx,
        "doc_stride": doc_stride,
        "max_answer_len": int(cfg.get("max_answer_len", 1000)),
    }


def chunk_mode_from(cfg: Dict[str, Any]) -> str:
    cm = cfg.get("chunk_mode")
    if not isinstance(cm, str) or not cm.strip():
        raise KeyError("Missing 'chunk_mode' in config")
    return cm.strip()


def predict_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg.get("predict")
    if not isinstance(p, dict):
        raise KeyError("Missing 'predict' block in config")
    return p


def train_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("train")
    if not isinstance(t, dict):
        raise KeyError("Missing 'train' block in config")
    return t


__all__ = [
    "default_config_path",
    "load_config",
    "resolve_model_dir",
    "resolve_tokenizer_name",
    "resolve_report_struct_path",
    "lengths_from",
    "chunk_mode_from",
    "predict_block",
    "train_block",
]
