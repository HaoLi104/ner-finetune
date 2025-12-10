# -*- coding: utf-8 -*-
"""Stub implementations of flash_attn.bert_padding functions."""

from __future__ import annotations

from typing import Any


def _raise(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("flash_attn is not available in this environment.")


index_first_axis = _raise
pad_input = _raise
unpad_input = _raise

__all__ = ["index_first_axis", "pad_input", "unpad_input"]
