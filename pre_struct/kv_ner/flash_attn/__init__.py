# -*- coding: utf-8 -*-
"""
Lightweight stub of the flash_attn package to satisfy optional imports from
Transformers when flash-attention is not available in the environment.

The real flash_attn package requires CUDA; this stub ensures that importing
`transformers.modeling_flash_attention_utils` does not crash in CPU-only
setups. All functions raise RuntimeError if called.
"""

from __future__ import annotations

from typing import Any


def _raise(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("flash_attn is not available in this environment.")


flash_attn_func = _raise
flash_attn_varlen_func = _raise

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
