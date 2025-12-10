import os
import sys

# 添加项目根目录到 sys.path (必须在导入前执行)
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .dataset import EnhancedQADataset, QACollator

__all__ = ["EnhancedQADataset", "QACollator"]
