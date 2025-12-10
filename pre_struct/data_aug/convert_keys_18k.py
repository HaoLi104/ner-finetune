"""
将 pre_struct/data_aug/keys_18000.json 转换为 REPORT_STRUCTURE_MAP 的结构：

- 一级 key：报告类别（category）保持不变
- 二级 key：该类别下的字段名，汇总成列表

输出：keys/keys_merged.json

用法：
    python pre_struct/data_aug/convert_keys_18k.py
或：
    python pre_struct/data_aug/convert_keys_18k.py \
        --in pre_struct/data_aug/keys_18000.json \
        --out keys/keys_merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def convert_18k_to_report_map(in_path: str, out_path: str) -> Dict[str, List[str]]:
    in_p = Path(in_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    raw: Any = json.loads(in_p.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("keys_18000.json should be a dict of {category: {field: [...]}}")

    out: Dict[str, List[str]] = {}
    for category, fields in raw.items():
        # 仅保留字典类别；其它类型跳过
        if not isinstance(fields, dict):
            continue
        # 二级 key 汇总成列表（按出现顺序，去重）
        seen = set()
        arr: List[str] = []
        for field_name in fields.keys():
            if not isinstance(field_name, str):
                field_name = str(field_name)
            name = field_name.strip()
            if not name or name in seen:
                continue
            arr.append(name)
            seen.add(name)
        out[str(category).strip()] = arr

    # 写出到目标文件
    out_p = Path(out_path)
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="in_path",
        default="pre_struct/data_aug/keys_18000.json",
        help="输入 JSON 路径（默认为 pre_struct/data_aug/keys_18000.json）",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="keys/keys_merged.json",
        help="输出 JSON 路径（默认为 keys/keys_merged.json）",
    )
    args = parser.parse_args()
    convert_18k_to_report_map(args.in_path, args.out_path)


if __name__ == "__main__":
    main()

