# balancing.py
# -*- coding: utf-8 -*-
from __future__ import annotations


import copy, random
from typing import Dict, List, Tuple
from collections import Counter
import re
def _normalize_title_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("：", ":").replace("（", "(").replace("）", ")")
    # 去掉所有空白（包括中间换行/空格/制表），避免出现 `FISH检\n  `、`21基因化疗\n  益预测` 这种脏标题
    s = re.sub(r"\s+", "", s)
    return s.strip()


def normalize_report_title_inplace(rec: dict) -> None:
    """
    把 'report_titles' 规范到 'report_title'，并对标题做强力清洗（移除所有空白）。
    """
    title = rec.get("report_title", "")
    if not isinstance(title, str) or not title.strip():
        title = rec.get("report_titles", "")
    rec["report_title"] = _normalize_title_text(title)
    rec.pop("report_titles", None)


def normalize_records(records: List[dict]) -> List[dict]:
    out = []
    for r in records:
        rr = copy.deepcopy(r)
        normalize_report_title_inplace(rr)
        out.append(rr)
    return out


def count_titles(records: List[dict]) -> Dict[str, int]:
    c = Counter([str(x.get("report_title", "")).strip() for x in records])
    c.pop("", None)
    return dict(c)


def median_count(counts: Dict[str, int]) -> int:
    vals = sorted([v for v in counts.values() if isinstance(v, int) and v > 0])
    if not vals:
        return 0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 == 1 else vals[mid]  # 取“中间位”，简化为上中位


def upsample_titles_to_median(
    records: List[dict], seed: int = 42
) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """
    将每个 report_title 的样本数上采样到中位数（有放回）。返回 (新列表, 原计数, 新计数)。
    注意：不会修改传入对象内容，但会复制样本。
    """
    rnd = random.Random(seed)
    # 标题规范化
    base = normalize_records(records)
    by_title: Dict[str, List[dict]] = {}
    for r in base:
        t = str(r.get("report_title", "")).strip()
        if t:
            by_title.setdefault(t, []).append(r)
    orig_counts = {k: len(v) for k, v in by_title.items()}
    target = median_count(orig_counts)
    if target <= 0:
        return base, orig_counts, orig_counts

    out: List[dict] = []
    for t, arr in by_title.items():
        n = len(arr)
        if n >= target:
            out.extend(arr)
        else:
            out.extend(arr)
            need = target - n
            # 有放回抽样补齐
            for _ in range(need):
                out.append(copy.deepcopy(rnd.choice(arr)))
    new_counts = count_titles(out)
    return out, orig_counts, new_counts
