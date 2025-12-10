# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> str:
    """Canonical config path for alias/title EBQA."""
    return str(Path(__file__).with_name("merged_config.json"))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or default_config_path()
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Alias-EBQA config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Alias-EBQA config must be a JSON object")
    return data


def _require_str(d: Dict[str, Any], key: str) -> str:
    if key not in d or not isinstance(d[key], str) or not d[key].strip():
        raise KeyError(f"Missing or invalid '{key}' in alias EBQA config")
    return d[key].strip()


def resolve_model_dir(cfg: Dict[str, Any]) -> str:
    return _require_str(cfg, "model_dir")


def resolve_tokenizer_name(cfg: Dict[str, Any]) -> str:
    for k in ("tokenizer_name_or_path", "tokenizer_name", "model_name_or_path"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError(
        "Missing tokenizer path: set 'tokenizer_name_or_path' (or 'tokenizer_name'/'model_name_or_path')"
    )


def resolve_report_struct_path(cfg: Dict[str, Any]) -> str:
    return _require_str(cfg, "report_struct_path")


def lengths_from(cfg: Dict[str, Any]) -> Dict[str, int]:
    try:
        return {
            "max_seq_len": int(cfg["max_seq_len"]),
            "max_tokens_ctx": int(cfg["max_tokens_ctx"]),
            "max_answer_len": int(cfg.get("max_answer_len", 1000)),
        }
    except KeyError as e:
        raise KeyError(f"Missing length config: {e}")


def chunk_mode_from(cfg: Dict[str, Any]) -> str:
    cm = cfg.get("chunk_mode")
    if not isinstance(cm, str) or not cm.strip():
        raise KeyError("Missing 'chunk_mode' in alias EBQA config")
    return cm.strip()


def predict_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("predict")
    if not isinstance(block, dict):
        raise KeyError("Missing 'predict' block in alias EBQA config")
    return block


def train_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("train")
    if not isinstance(block, dict):
        raise KeyError("Missing 'train' block in alias EBQA config")
    return block
