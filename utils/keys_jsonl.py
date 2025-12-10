# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Iterable, List, Dict


def _norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        x = _norm(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_aliases(raw: Any) -> List[str]:
    """
    解析别名：
      - "切口描述" -> ["切口描述"]
      - ["切口描述", "切口情况"] -> 两个都保留
      - [["切口描述",""], ["切口情况","备注"]] -> 只取每个子列表的第1个元素：["切口描述", "切口情况"]
      - {"别名": [...]} / {"aliases": [...]} / {"alias": "..."} 也支持
    空串会被丢弃；最终结果去重且保持顺序。
    """
    out: List[str] = []
    if raw is None:
        return out

    if isinstance(raw, str):
        out.append(raw)

    elif isinstance(raw, (list, tuple)):
        for it in raw:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, (list, tuple)) and it:
                # 只取第一列作为别名
                if isinstance(it[0], str):
                    out.append(it[0])

    elif isinstance(raw, dict):
        for k in ("别名", "aliases", "alias"):
            if k in raw:
                out.extend(_parse_aliases(raw[k]))
                break

    return _dedup_keep_order(out)


def jsonl_to_struct_and_alias(
    in_jsonl: str, out_struct_json: str, out_alias_json: str
) -> None:
    """
    读取 jsonl，每行形如：
      {"疾病种类":"乳腺癌","报告类型":"其他","报告名":"手术记录",
       "键名":{"标准名":"手术切口","别名":[["切口描述",""]]}}
    生成：
      1) 结构文件：{"报告名":[标准名按出现顺序,...]}
      2) 别名文件：{"报告名":{"标准名":[别名,...]}} —— 没有别名也会写空数组 []
    """
    in_path = Path(in_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(in_jsonl)

    # 用 OrderedDict 保序：report_name -> OrderedDict(标准名 -> None)
    struct_map: Dict[str, OrderedDict] = OrderedDict()
    # 别名暂存：report_name -> 标准名 -> OrderedDict(别名 -> None)
    alias_map: Dict[str, Dict[str, OrderedDict]] = OrderedDict()

    bad_lines = 0
    kept_lines = 0

    with in_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                bad_lines += 1
                # 可选打印：print(f"[WARN] line {ln} JSON 解析失败: {e}")
                continue

            report_name = _norm(
                obj.get("报告名") or obj.get("报告名称") or obj.get("report_name")
            )
            key_block = obj.get("键名") or {}
            if not isinstance(key_block, dict):
                key_block = {}

            std_name = _norm(
                key_block.get("标准名")
                or key_block.get("标准名称")
                or key_block.get("canonical")
            )

            if not report_name or not std_name:
                # 关键字段缺失 -> 跳过该行
                continue

            # 1) 结构表：按出现顺序收集标准名（不丢别名为空的项）
            if report_name not in struct_map:
                struct_map[report_name] = OrderedDict()
            if std_name not in struct_map[report_name]:
                struct_map[report_name][std_name] = None  # 占位

            # 2) 别名表：即使后面没有别名，也会在最终输出时给空数组
            aliases = _parse_aliases(key_block.get("别名"))
            # 去掉与标准名相同的别名
            aliases = [a for a in aliases if a and a != std_name]

            if report_name not in alias_map:
                alias_map[report_name] = {}
            if std_name not in alias_map[report_name]:
                alias_map[report_name][std_name] = OrderedDict()

            for a in aliases:
                alias_map[report_name][std_name][a] = None

            kept_lines += 1

    # === 写出 ===
    # 结构：保持出现顺序
    struct_out = {rp: list(std_od.keys()) for rp, std_od in struct_map.items()}
    # 别名：保证每个标准名都有键（即使没有别名 -> []）
    alias_out = {}
    for rp, std_list in struct_out.items():
        alias_out[rp] = {}
        for std in std_list:
            als_od = alias_map.get(rp, {}).get(std, OrderedDict())
            alias_out[rp][std] = list(als_od.keys())  # 可能为空 []

    Path(out_struct_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_alias_json).parent.mkdir(parents=True, exist_ok=True)

    Path(out_struct_json).write_text(
        json.dumps(struct_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(out_alias_json).write_text(
        json.dumps(alias_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[OK] 读入: {kept_lines} 行，坏行: {bad_lines} 行；"
        f"写出结构: {out_struct_json}；写出别名: {out_alias_json}"
    )


if __name__ == "__main__":
    jsonl_to_struct_and_alias(
        in_jsonl="health_record_data.jsonl",
        out_struct_json="keys/keys_merged.json",
        out_alias_json="keys/ALIAS_MAP_0919.json",
    )
