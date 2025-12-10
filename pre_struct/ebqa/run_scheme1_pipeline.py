#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for run_scheme_pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1] if len(_THIS_DIR.parents) > 1 else _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pre_struct.ebqa.run_scheme_pipeline import main


if __name__ == "__main__":
    main()
