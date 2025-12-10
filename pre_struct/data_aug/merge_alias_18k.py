"""
根据 18k 结构与既有别名表，合并生成新版别名映射：

- 一级分类以 `keys/keys_merged.json` 为准
- 二级字段：为每个字段合并别名，来源包括：
  1) `pre_struct/ALIAS_MAPPING_18k.json` 中相同字段（同类与跨类聚合）
  2) `pre_struct/data_aug/keys_18000.json` 中相同字段（按类与全局聚合）
- 不删除已有别名，去重但保留出现顺序

输出：`pre_struct/ALIAS_MAPPING_18k.json`

用法：
    python pre_struct/data_aug/merge_alias_18k.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


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
        if not k:
            continue
        if k in seen:
            continue
        out.append(x)
        seen.add(k)
    return out


def load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def merge_aliases() -> Dict[str, Dict[str, List[str]]]:
    # 读取新的结构映射
    map18k: Dict[str, List[str]] = load_json("keys/keys_merged.json") or {}
    alias_map_raw: Any = load_json("pre_struct/ALIAS_MAPPING_18k.json") or {}
    keys18k_raw: Any = load_json("pre_struct/data_aug/keys_18000.json") or {}

    # 1) 解析 ALIAS_MAPPING.json
    alias_cat: Dict[str, Dict[str, List[str]]] = {}
    alias_global: Dict[str, List[str]] = {}
    if isinstance(alias_map_raw, dict):
        for k, v in alias_map_raw.items():
            if isinstance(v, list):
                # 全局层：{canon: [aliases...]}
                canon = str(k).strip()
                alias_global[canon] = [str(a).strip() for a in v if isinstance(a, str) and str(a).strip()]
            elif isinstance(v, dict):
                # 分类层：{title: {canon: [aliases...]}}
                title = str(k).strip()
                d: Dict[str, List[str]] = {}
                for canon, al in v.items():
                    if isinstance(al, list):
                        d[str(canon).strip()] = [str(a).strip() for a in al if isinstance(a, str) and str(a).strip()]
                alias_cat[title] = d

    # 聚合为按字段的全局别名（汇总分类层 + 全局层）
    alias_by_key: Dict[str, List[str]] = {}
    for c, arr in alias_global.items():
        alias_by_key.setdefault(c, []).extend(arr)
    for t, d in alias_cat.items():
        for c, arr in d.items():
            alias_by_key.setdefault(c, []).extend(arr)
    for c, arr in list(alias_by_key.items()):
        alias_by_key[c] = _dedup_keep_order(arr)

    # 2) 解析 keys_18000.json（从每条记录的 alias 字段采集）
    k18k_cat: Dict[str, Dict[str, List[str]]] = {}
    if isinstance(keys18k_raw, dict):
        for title, fields in keys18k_raw.items():
            if not isinstance(fields, dict):
                continue
            d: Dict[str, List[str]] = {}
            for canon, items in fields.items():
                if not isinstance(items, list):
                    continue
                out: List[str] = []
                seen = set()
                for it in items:
                    alias = None
                    if isinstance(it, dict):
                        alias = it.get("alias")
                    elif isinstance(it, str):
                        alias = it
                    if not isinstance(alias, str):
                        continue
                    alias = alias.strip()
                    if not alias:
                        continue
                    key = _std(alias)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(alias)
                d[str(canon).strip()] = out
            k18k_cat[str(title).strip()] = d

    # 聚合 keys_18000 为按字段的全局别名
    k18k_by_key: Dict[str, List[str]] = {}
    for t, d in k18k_cat.items():
        for c, arr in d.items():
            k18k_by_key.setdefault(c, []).extend(arr)
    for c, arr in list(k18k_by_key.items()):
        k18k_by_key[c] = _dedup_keep_order(arr)

    # 3) 构建输出：以 18k 的分类与字段为准
    result: Dict[str, Dict[str, List[str]]] = {}
    for title, keys in map18k.items():
        if not isinstance(keys, list):
            continue
        out_fields: Dict[str, List[str]] = {}
        for canon in keys:
            if not isinstance(canon, str):
                canon = str(canon)
            canon_s = canon.strip()
            if not canon_s:
                continue

            merged: List[str] = []

            # 既有（按分类）
            merged.extend(alias_cat.get(title, {}).get(canon_s, []) or [])
            # 既有（跨类聚合）
            merged.extend(alias_by_key.get(canon_s, []) or [])
            # 18k（按分类）
            merged.extend(k18k_cat.get(title, {}).get(canon_s, []) or [])
            # 18k（跨类聚合）
            merged.extend(k18k_by_key.get(canon_s, []) or [])

            # 去重、清洗：去掉与规范名同形的别名
            merged = [a for a in _dedup_keep_order(merged) if _std(a) != _std(canon_s)]

            out_fields[canon_s] = merged
        result[str(title).strip()] = out_fields

    return result


def main() -> None:
    merged = merge_aliases()
    out_p = Path("pre_struct/ALIAS_MAPPING_18k.json")
    out_p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_p} with {len(merged)} categories.")


if __name__ == "__main__":
    main()

