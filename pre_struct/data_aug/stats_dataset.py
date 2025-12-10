"""统计 data/clean_ocr_ppt_da_v3_3.json 中每个 report_title 的数量，并将每类裁剪到 1000 条以内。

用法：
    python pre_struct/data_aug/stats_dataset.py

读取固定路径的 JSON/JSONL，提取非空的 report_title，输出各标题计数；
并按照每类最多 1000 条进行裁剪，生成新的平衡文件。
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON 数组
    if text[0] == "[":
        try:
            data = json.loads(text)
        except Exception:
            return []
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    # JSONL
    recs: List[Dict[str, Any]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            recs.append(obj)
    return recs


def main() -> None:
    path = "data/clean_ocr_ppt_da_v3_3.json"
    out_path = "data/clean_ocr_ppt_da_v4.json"
    max_per_title = 800
    recs = load_records(path)

    # 统计每个标题出现次数（裁剪前）
    counts_before: Dict[str, int] = {}
    for r in recs:
        t = r.get("report_title")
        if isinstance(t, str):
            t = t.strip()
        elif t is None:
            t = ""
        else:
            t = str(t).strip()
        if not t:
            continue
        counts_before[t] = counts_before.get(t, 0) + 1

    # 按照原始顺序裁剪到每类最多 max_per_title 条
    kept_per_title: Dict[str, int] = {}
    balanced: List[Dict[str, Any]] = []
    for r in recs:
        t = r.get("report_title")
        if isinstance(t, str):
            t = t.strip()
        elif t is None:
            t = ""
        else:
            t = str(t).strip()

        if not t:
            # 无标题的记录原样保留
            balanced.append(r)
            continue

        c = kept_per_title.get(t, 0)
        if c < max_per_title:
            balanced.append(r)
            kept_per_title[t] = c + 1
        else:
            # 超过上限的记录被裁剪
            pass

    # 写出平衡后的文件
    Path(out_path).write_text(json.dumps(balanced, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计每个标题出现次数（裁剪后）
    counts_after: Dict[str, int] = {}
    for r in balanced:
        t = r.get("report_title")
        if isinstance(t, str):
            t = t.strip()
        elif t is None:
            t = ""
        else:
            t = str(t).strip()
        if not t:
            continue
        counts_after[t] = counts_after.get(t, 0) + 1

    # 输出汇总（含裁剪前/后计数，降序展示）
    def sort_counts(d: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

    result = {
        "in_file": path,
        "out_file": out_path,
        "max_per_title": max_per_title,
        "total_before": len(recs),
        "total_after": len(balanced),
        "counts_before": sort_counts(counts_before),
        "counts_after": sort_counts(counts_after),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
