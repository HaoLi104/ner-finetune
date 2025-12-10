# -*- coding: utf-8 -*-
"""Alias recognition (key title detection) data pipeline."""

from .da_core.dataset import EnhancedQADataset, QACollator

# Backward compatibility exported names
AliasQADataset = EnhancedQADataset
AliasCollator = QACollator

__all__ = [
    "EnhancedQADataset",
    "QACollator",
    "AliasQADataset",
    "AliasCollator",
]
