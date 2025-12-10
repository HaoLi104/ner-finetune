# structs.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


def _dedup_keep_order(arr: List[str]) -> List[str]:
    seen, out = set(), []
    for x in arr:
        if isinstance(x, str) and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _std(s: str) -> str:
    s = str(s).strip().replace("：", ":").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


# 简化 get_report_structure_map
def get_report_structure_map(
    path: str = "../../keys/keys_merged.json",
) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        # 兼容绝对路径或项目根目录
        abs_p = Path(__file__).resolve().parent.parent.parent / "keys/keys_merged.json"
        if abs_p.exists():
            p = abs_p
        else:
            raise FileNotFoundError(f"找不到配置文件: {path} (尝试过 {abs_p})")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    
    out: Dict[str, List[str]] = {}
    for title, fields in raw.items():
        # keys_merged.json的结构是 {title: {field_name: {别名: [], Q: ""}}}
        if isinstance(fields, dict):
            # 提取字段名作为列表
            field_names = [k for k in fields.keys() if isinstance(k, str) and k.strip()]
            out[str(title).strip()] = _dedup_keep_order(field_names)
        
    return out


REPORT_STRUCTURE_MAP = get_report_structure_map()

# 兼容：优先使用 keys/keys_merged.json；找不到再尝试 keys/keys.json；均不存在则置空
def _load_keys_mapping() -> dict:
    root = Path(__file__).resolve().parent.parent.parent
    for cand in (root / "keys/keys_merged.json", root / "keys/keys.json"):
        try:
            if cand.exists():
                obj = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return obj
        except Exception:
            continue
    return {}

new_keys_mapping = _load_keys_mapping()


@lru_cache(maxsize=1)
def get_key_alias_maps() -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[str]]],
]:
    """
    返回三层映射（兼容全局 & 分标题）：
    - alias2canon：全局别名到规范名（冲突时保留先见）
    - canon2aliases：全局规范名到所有别名（合并去重）
    - title_canon2aliases：按 report_title 切分的规范名到别名
    """
    alias2canon: Dict[str, str] = {}
    canon2aliases: Dict[str, List[str]] = {}
    title_canon2aliases: Dict[str, Dict[str, List[str]]] = {}

    def _merge_alias_file(path: Path) -> None:
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for k, v in raw.items():
            if isinstance(v, list):
                canon = str(k).strip()
                if not canon:
                    continue
                canon2aliases.setdefault(canon, []).extend(
                    [str(a).strip() for a in v if isinstance(a, str)]
                )
            elif isinstance(v, dict):
                title = str(k).strip()
                if not title:
                    continue
                # 检查报告类型是否在 keys.json 中
                if title in new_keys_mapping:
                    aliases = new_keys_mapping[title].get("别名", [])
                    if isinstance(aliases, list):
                        d = title_canon2aliases.setdefault(title, {})
                        # 修正：通过获取 canon 再获取别名
                        for canon, alias_data in v.items():
                            canon_s = str(canon).strip()
                            if not canon_s:
                                continue
                            field_info = new_keys_mapping[title].get(canon_s, {})

                            aliases = field_info.get("别名", [])
                            if isinstance(aliases, list):
                                clean_aliases = [
                                    str(a).strip()
                                    for a in aliases
                                    if isinstance(a, str)
                                ]
                                if clean_aliases:
                                    d.setdefault(canon_s, []).extend(clean_aliases)
                                    canon2aliases.setdefault(canon_s, []).extend(
                                        clean_aliases
                                    )
                else:
                    # 如果 title 不在 new_keys_mapping 中，继续原来的操作
                    for canon, aliases in v.items():
                        canon_s = str(canon).strip()
                        if not canon_s:
                            continue
                        clean_aliases = [
                            str(a).strip() for a in aliases if isinstance(a, str)
                        ]
                        if clean_aliases:
                            title_canon2aliases.setdefault(title, {}).setdefault(
                                canon_s, []
                            ).extend(clean_aliases)
                            canon2aliases.setdefault(canon_s, []).extend(clean_aliases)

    # 合并新旧别名映射
    candidate_files = [Path("keys/ALIAS_MAP_0919.json")]
    for pf in candidate_files:
        _merge_alias_file(pf)

    # 标题别名去重 & 归一
    for title, cmap in list(title_canon2aliases.items()):
        for canon, als in list(cmap.items()):
            title_canon2aliases[title][canon] = _dedup_keep_order(
                [str(x).strip() for x in als]
            )

    # 全局层去重
    for c, al in list(canon2aliases.items()):
        canon2aliases[c] = _dedup_keep_order([str(x).strip() for x in al])

    # 全局 alias2canon（规范名自身也登记）
    for c, al in canon2aliases.items():
        c_std = _std(c)
        if c_std not in alias2canon:
            alias2canon[c_std] = c
        for a in al:
            a_std = _std(a)
            if a_std not in alias2canon:
                alias2canon[a_std] = c

    return alias2canon, canon2aliases, title_canon2aliases


def normalize_key_name(name: str) -> str:
    alias2canon, _, _ = get_key_alias_maps()
    return alias2canon.get(_std(name), name)
