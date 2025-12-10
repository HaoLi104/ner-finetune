# -*- coding: utf-8 -*-
"""Alias/title pipeline utilities delegate to the shared EBQA implementations."""

import os
import sys

# 添加项目根目录到 sys.path
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 导入所有公开的函数和类，包括下划线开头的内部函数
from pre_struct.ebqa.da_core.utils import (  # noqa: F401
    _load_jsonl_or_json,
    _save_jsonl,
    _dedup_keep_order,
    _tighten_span,
    split_train_test_balanced_by_title,
    convert_labelstudio_project_to_clean_records,
    BalancedKVSeparator,
)
