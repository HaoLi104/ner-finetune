"""
人工规则补充：把旧别名中“未对齐到 18k 类别/字段”的常见同义项，
合并到 18k 版别名映射（pre_struct/ALIAS_MAPPING_18k.json）里对应/最接近的字段，
确保历史别名不丢失，同时不新增 18k 不存在的规范字段。

运行：
    python pre_struct/data_aug/manual_merge_aliases.py
输出覆盖写：
    pre_struct/ALIAS_MAPPING_18k.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _std(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
    )


def _dedup_keep_order(arr: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in arr:
        if not isinstance(x, str):
            x = str(x)
        k = _std(x)
        if not k or k in seen:
            continue
        out.append(x)
        seen.add(k)
    return out


def patch_aliases(data: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    def add(cat: str, field: str, aliases: List[str]):
        if cat not in data:
            return
        d = data[cat]
        if field not in d:
            return
        cur = d.get(field, []) or []
        cur = list(cur) + list(aliases)
        d[field] = _dedup_keep_order(cur)

    # 影像类：CT
    add("CT", "检查所见", [
        "影像学 表现",
        "影像所见",
    ])
    add("CT", "影像诊断", [
        "影像学 诊断",
        "诊断意见",
        "诊断结论",
    ])

    # 影像类：MRI
    add("MRI", "检查所见", [
        "影像表现",
        "影像表现Findings",
        "影像所见",
    ])
    add("MRI", "印象", [
        "影像诊断",
        "诊断结论",
        "影像诊断 Diagnosis",
    ])

    # 影像类：X线
    add("X线", "影像所见", [
        "影像学 表现",
    ])
    add("X线", "诊断结果", [
        "诊断结论",
        "影像学 诊断",
        "诊断意见",
    ])

    # 超声类（兼容其它超声子类别名）
    add("超声", "超声所见", [
        "检查所见",
        "影像所见",
        "超声表现",
    ])
    add("超声", "超声提示", [
        "超声诊断",
        "诊断意见",
        "印象",
    ])

    # 光散射乳腺检测：对齐“报告时间”
    add("光散射乳腺检测", "报告日期", [
        "报告时间",
    ])

    # 基因类：BRCA/21 基因系列
    add("BRCA基因", "报告日期", [
        "报告时间",
    ])
    add("21基因检测", "送检日期", [
        "送检日期",
    ])

    # 病理类：免疫组化归并到活检病理的同名字段
    add("活检病理", "报告时间", [
        "报告审核时间",
        "报告日期",
    ])
    add("活检病理", "病变部位", [])
    add("活检病理", "标本类型", [])
    add("活检病理", "病理诊断", [
        "诊断",
        "病理诊断结果",
        "补充病理诊断",
    ])

    return data


def main() -> None:
    p = Path("pre_struct/ALIAS_MAPPING_18k.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    data = patch_aliases(data)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Patched pre_struct/ALIAS_MAPPING_18k.json with manual merges.")


if __name__ == "__main__":
    main()

