# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, os, sys, logging
from pathlib import Path

try:
    from tqdm import tqdm  # type: ignore
except Exception:

    def tqdm(*args, **kwargs):
        class _Dummy:
            def update(self, n=1):
                pass

            def close(self):
                pass

        return _Dummy()


def ensure_sys_path_here():
    this = Path(__file__).resolve().parent
    for p in [str(this), str(this.parent), str(this.parent.parent)]:
        if p not in sys.path:
            sys.path.append(p)


def get_logger(name: str = "data_augmentation", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(h)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def _truncate(obj, n: int = 400) -> str:
    s = (
        obj
        if isinstance(obj, str)
        else json.dumps(obj, ensure_ascii=False, default=str)
    )
    return s if len(s) <= n else (s[:n] + " …(truncated)")


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, obj, indent: int = 2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
