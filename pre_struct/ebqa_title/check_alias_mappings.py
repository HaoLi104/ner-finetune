#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
本脚本仅保留与本任务直接相关的逻辑：
1) alias 的 key 规范化为 keys_merged.json 中该记录 report_title 的标准字段；
2) alias 的 value 必须来源于原 report 文本中用于指代该字段的标签/小标题（由 LLM 基于值的上下文截取）。

要点：
- 对 alias 中不属于本 title 的 key，结合其 value 让模型映射到本 title 的标准 key；无法映射则丢弃该 alias 键；
- 对缺失别名的标准 key，基于 (canonical, value, report) 在原文片段中抽取标签作为别名；无值可定位则置空；
- 删除与“准确性检查”等无关代码，以聚焦此任务。
"""

import argparse
import json
import logging
import os
import sys
import typing
from typing import Any, Dict, List, Optional, Tuple, Set


# --- 保障可从项目根导入 report_keys_alias.call_model_once ---
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))  # repo_root/pre_struct/ebqa_title -> repo_root/pre_struct
_PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)               # -> repo_root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from report_keys_alias import call_model_once  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError(
        "无法导入 report_keys_alias.call_model_once，请确认在项目根目录下存在该文件。"
    ) from e


logger = logging.getLogger("alias_check")


def _strip_json_fence(x: str) -> str:
    return x.replace("```json", "").replace("```", "").strip()


def suppress_verbose_http_logs(level: int = logging.WARNING) -> None:
    """抑制 OpenAI/httpx 的 HTTP 明细日志（如 HTTP Request: POST ...）。"""
    try:
        for name in ("openai", "openai._base_client", "httpx", "httpcore"):
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = False
            if not lg.handlers:
                lg.addHandler(logging.NullHandler())
    except Exception:
        pass


def _normalize(s: Any) -> str:
    s = str(s or "").strip()
    s = s.replace("\u3000", " ")
    return " ".join(s.split())


def _load_struct_map(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _title_keys(struct_map: Dict[str, Any], title: str) -> List[str]:
    node = struct_map.get(title)
    if isinstance(node, dict):
        if "fields" in node and isinstance(node["fields"], list):
            return [str(x).strip() for x in node["fields"] if str(x).strip()]
        if "keys" in node and isinstance(node["keys"], list):
            return [str(x).strip() for x in node["keys"] if str(x).strip()]
        if all(isinstance(v, list) for v in node.values()):
            return [str(k).strip() for k in node.keys() if str(k).strip()]
        if all(isinstance(v, dict) and "别名" in v for v in node.values()):
            return [str(k).strip() for k in node.keys() if str(k).strip()]
        return []
    if isinstance(node, list):
        return [str(x).strip() for x in node if str(x).strip()]
    return []


def _context_around_value(report: str, value: str, left: int = 80, right: int = 80) -> str:
    report = report or ""
    value = _normalize(value)
    if not report:
        return ""
    if not value:
        return report[: left + right + 160]
    # 简化匹配：不去空格直接查找；找不到再尝试去空格
    pos = report.find(value)
    if pos < 0:
        v2 = value.replace(" ", "")
        if v2:
            # 尝试去除空格匹配：回退为原文截取前1000字符
            return report[: max(200, left + right + 160)]
    start = max(0, pos - left)
    end = min(len(report), pos + len(value) + right)
    return report[start:end]


def _collect_locators(rec: Dict[str, Any]) -> Dict[str, str]:
    """采集用于回溯定位的关键信息字段。只收集存在的字段，统一做适度规范化。"""
    candidates = [
        "报告编号", "病人编号", "检查号", "样本编号", "住院号", "门诊住院号", "门诊号",
        "病理号", "会诊号", "原病理号", "患者ID", "病人ID", "ID号", "医院"
    ]
    out: Dict[str, str] = {}
    for k in candidates:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = _normalize(v)[:120]
    return out


_MISSING_TOKENS = {"", "NA", "N/A", "NULL", "NONE", "—", "-", "无", "未知", "不详"}


def _is_missing_alias(val: Any) -> bool:
    s = _normalize(val).upper()
    return s in _MISSING_TOKENS


# 不参与检查/补全的键
_EXCLUDED_KEYS = {"医院", "无键名", "其他"}


def _is_excluded_key(key: Any) -> bool:
    k = _normalize(key)
    return k in _EXCLUDED_KEYS


# （移除了与“准确性检查”相关的轻量规则与同义词逻辑）


def _sanitize_alias_value(
    canonical: str,
    alias: Any,
    value: Optional[str] = None,
    forbid_values: Optional[List[str]] = None,
) -> str:
    """清洗别名，确保不把值带入，尽量保留标签/小标题。

    策略：
    - 空/占位符 -> 返回 canonical
    - 含冒号：优先取冒号左侧（标签）；若左侧是泛词（如“别名/标签/字段”），取右侧
    - 若含“为/是”并呈现“标签为值”形式，取“为/是”左侧
    - 若提供了 value，且别名包含 value（去空格后匹配），回落为 canonical
    - 限长、去两端标点
    """
    a = _normalize(alias)
    if not a or a.upper() in _MISSING_TOKENS:
        return _normalize(canonical)

    def _trim_punct(s: str) -> str:
        return s.strip().strip("，。,:：;；()（）[]【】<>《》\"'")

    a = _trim_punct(a)

    # 切分冒号场景
    if ":" in a or "：" in a:
        sep = ":" if ":" in a else "："
        left, right = a.split(sep, 1)
        left, right = _trim_punct(left), _trim_punct(right)
        generic = any(x in left for x in ["别名", "标签", "字段", "名称", "项目"])
        # 默认返回左侧（标签），若左侧太泛则用右侧
        a = right if generic else left

    # 标签为/是值 的场景
    for kw in ("为", "是"):
        if kw in a:
            parts = a.split(kw, 1)
            if len(parts) == 2:
                left, right = _trim_punct(parts[0]), _trim_punct(parts[1])
                # 取左侧作为标签
                a = left
                break

    # 若别名包含当前值或其他禁止值（如姓名等），则回落为 canonical
    def _contains(hay: str, needle: str) -> bool:
        if not hay or not needle:
            return False
        return (needle in hay) or (needle.replace(" ", "") in hay.replace(" ", ""))

    if value:
        v = _normalize(value)
        if _contains(a, v):
            return _normalize(canonical)
    if forbid_values:
        for fv in forbid_values:
            fv = _normalize(fv)
            if _contains(a, fv):
                return _normalize(canonical)

    # 长度与简单字符检查
    if len(a) > 50:
        a = a[:50]
    return a


def _default_checker_factory(
    model: str = "qwen3-32b",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
):
    """返回一个函数：checker(prompt:str)->str，内部使用 call_model_once。"""

    def _checker(prompt: str) -> str:
        # 重要：不要把 None 作为 base_url 传入，以免覆盖 report_keys_alias.py 的默认 BASE_URL。
        # 同理，api_key 为空时，让 call_model_once 使用模块内的默认 API_KEY。
        if base_url is None and api_key is None:
            return call_model_once(prompt=prompt, model=model, timeout=timeout)
        if base_url is None:
            return call_model_once(prompt=prompt, model=model, api_key=api_key, timeout=timeout)
        if api_key is None:
            return call_model_once(prompt=prompt, model=model, base_url=base_url, timeout=timeout)
        return call_model_once(
            prompt=prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    return _checker


# （移除了：用于“准确性检查”的提示词构造函数）


def _build_map_to_canonical_prompt(
    data_key: str,
    value: str,
    report_title: str,
    canonical_keys: List[str],
) -> str:
    """构建提示：将数据中的非标准key映射到最相关的标准key"""
    title_line = f"报告类型: {report_title}\n" if report_title else ""
    keys_list = "、".join(canonical_keys[:30])  # 最多显示30个
    if len(canonical_keys) > 30:
        keys_list += f"... (共{len(canonical_keys)}个)"
    return (
        "你是医学报告字段映射助手。\n"
        "现有一个数据中的字段名，需要映射到预定义的标准字段列表中最相关的一个。\n"
        f"数据字段: {data_key}\n"
        f"字段值示例: {value[:50] if value else '(空)'}\n"
        + title_line +
        f"标准字段列表: {keys_list}\n\n"
        "任务：从标准字段列表中选择一个与数据字段语义最相关的标准字段。\n"
        "要求：\n"
        "1) 必须从标准字段列表中选择，不能自创；\n"
        "2) 优先选择语义完全一致的；\n"
        "3) 若无完全一致的，选择最接近/最相关的；\n"
        "4) 若完全无关或无法确定，返回数据字段名本身。\n\n"
        '严格输出JSON: {"canonical_key": "<标准字段名>", "confidence": "high/medium/low"}\n'
        "只输出JSON，不要任何额外文本。"
    )


def _build_find_alias_prompt(
    canonical: str,
    value: str,
    report_title: str,
    context_snippet: str,
) -> str:
    title_line = f"报告类型: {report_title}\n" if report_title else ""
    ctx = context_snippet.strip()
    return (
        "你是医学报告解析助手。现提供一个【标准字段名】、该字段在报告中的【值】、以及原文片段。\n"
        "请在片段中精确定位用于指代该字段的【标签/小标题/表头词】（别名）。\n"
        "重要：候选别名列表可能不完整，仅作参考；别名必须来自 value 的真实上下文。\n"
        "有时报告没有显式标签，标准字段是我们给定的规范名称；若片段中找不到能指代该字段的标签/小标题，alias 必须返回标准字段名本身。\n"
        "提取要求（全部遵守）：\n"
        "1) 只提取能在片段中直接看到、用于指代该字段的【标签/小标题/表头词】，作为 alias；\n"
        "2) 结构匹配（如何取【标签】）：\n"
        "   - 形式【标签:值/标签：值】→ 取标签；\n"
        "   - 形式【值/值连续多个值】→ 用标准字段名作为 alias，这一点需要注意，不得使用值旁边的其他应该是值的部分作为alias，比如张三/53岁，应该用标准字段名作为alias，不得使用张三作为年龄alias\n"
        "3) 严禁返回【值】或包含【值】的字符串；\n"
        "4) 片段中若无可指代该字段的标签，则 alias 必须返回标准字段名本身；\n"
        "5) alias 须为≤15字符的单行文本，不含冒号/换行/引号及多余标点。\n"
        f"标准字段: {canonical}\n"
        f"字段值: {value}\n"
        + title_line +
        (f"报告片段:\n{ctx}\n" if ctx else "") +
        '请严格输出 JSON：{"alias": "<别名或原字段名>"}。\n'
        "重要：alias 只能是用于指代该【标准字段】的标签表述；\n"
        "若无法确定或片段中并非该字段的标签，则 alias 必须返回为标准字段名本身；\n"
        "严禁把值或包含值的字符串作为 alias；不要返回数值/日期/姓名等实际值内容；\n"
        "只返回字段标签/小标题等纯粹的别名文字，且不得输出 JSON 以外的任何文本。"
    )


def map_to_canonical_key(
    data_key: str,
    value: str,
    report_title: str,
    canonical_keys: List[str],
    model_checker: Optional[Any] = None,
) -> Tuple[str, str, str]:
    """将数据中的非标准key映射到最相关的标准key。
    
    返回: (canonical_key, confidence, reason)
    - canonical_key: 映射到的标准字段名
    - confidence: high/medium/low
    - reason: 映射原因或错误信息
    """
    data_key_n = _normalize(data_key)
    value_n = _normalize(value)
    
    # 快速检查：如果数据key本身就在标准列表中，直接返回
    if data_key_n in [_normalize(k) for k in canonical_keys]:
        return data_key_n, "high", "直接匹配"
    
    if model_checker is None:
        model_checker = _default_checker_factory()
    
    prompt = _build_map_to_canonical_prompt(
        data_key=data_key_n,
        value=value_n,
        report_title=_normalize(report_title),
        canonical_keys=canonical_keys,
    )
    
    try:
        raw = model_checker(prompt)
    except Exception as e:
        logger.warning("模型调用失败(map_key): %s", e)
        return data_key_n, "low", "模型调用失败"
    
    try:
        if isinstance(raw, str):
            raw = _strip_json_fence(raw)
            obj = json.loads(raw)
        else:
            obj = raw
        if isinstance(obj, dict) and "canonical_key" in obj:
            mapped_key = _normalize(obj.get("canonical_key", "")) or data_key_n
            confidence = str(obj.get("confidence", "low"))
            # 验证映射的key确实在标准列表中
            canonical_keys_normalized = [_normalize(k) for k in canonical_keys]
            if mapped_key in canonical_keys_normalized:
                return mapped_key, confidence, "模型映射成功"
            else:
                # 模型返回了不在列表中的key，回退
                return data_key_n, "low", f"模型返回的key不在标准列表中: {mapped_key}"
    except Exception as e:
        logger.warning("解析映射结果失败: %s", e)
    
    return data_key_n, "low", "未能解析模型结果"


def find_alias_from_report(
    canonical: str,
    value: str,
    report_title: str,
    report_text: str,
    model_checker: Optional[Any] = None,
) -> Tuple[str, bool, str]:
    """基于 (canonical, value, report 片段) 调用模型寻找别名。

    返回: (alias_text, found, reason)
    - 若模型无法解析或找不到，回落为 (canonical, False, "...")
    """
    canonical_n = _normalize(canonical)
    value_n = _normalize(value)
    snippet = _context_around_value(report_text or "", value_n)
    if model_checker is None:
        model_checker = _default_checker_factory()
    prompt = _build_find_alias_prompt(canonical_n, value_n, _normalize(report_title), snippet)
    try:
        raw = model_checker(prompt)
    except Exception as e:  # pragma: no cover
        logger.warning("模型调用失败(find): %s", e)
        return canonical_n, False, "模型调用失败"

    try:
        if isinstance(raw, str):
            raw = _strip_json_fence(raw)
            obj = json.loads(raw)
        else:
            obj = raw
        if isinstance(obj, dict) and "alias" in obj:
            alias = _normalize(obj.get("alias", "")) or canonical_n
            # 若未提供 found，则根据是否与标准字段名相同来推断
            found = alias != canonical_n and alias != ""
            return alias, found, ""
    except Exception:
        pass
    return canonical_n, False, "未能解析模型结果"


def _atomic_write_json(obj: Any, out_path: str) -> None:
    """原子写文件，避免部分写入造成损坏。"""
    import tempfile
    d = json.dumps(obj, ensure_ascii=False, indent=2)
    dirn = os.path.dirname(out_path) or "."
    os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirn, delete=False) as tf:
        tmp = tf.name
        tf.write(d)
        tf.flush()
    os.replace(tmp, out_path)


# （移除了：检查 alias 准确性的函数）


# （移除了：仅检查模式）


def run_fill_and_check(
    input_path: str,
    out_path: str,
    save_updates: str,
    *,
    struct_path: str = "keys/keys_merged.json",
    fill_out_path: Optional[str] = "data/alias_fill.jsonl",
    limit: Optional[int] = None,
    start: int = 0,
    model: str = "qwen3-32b",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
    num_workers: int = 8,
) -> None:
    """对 alias 进行规范化与缺失别名补充（不做准确性检查）。

    - 读取 struct_map，确定每条记录应补充/校准的标准键集合；
    - 缺失补充：对 alias 缺失的键，基于 (key, value, report) 调模型找别名；无值则置空；
    - 输出：JSONL 过程日志 + JSON 更新后的全量记录。
    """
    # 抑制 HTTP 噪声日志
    suppress_verbose_http_logs()

    # 读数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入必须是列表 JSON：data/merged.converted.json")

    struct_map = _load_struct_map(struct_path)

    # 构建模型调用器
    checker = _default_checker_factory(
        model=model, base_url=base_url, api_key=api_key, timeout=timeout
    )

    # 缓存
    find_cache: Dict[Tuple[str, str], Tuple[str, bool, str]] = {}
    map_cache: Dict[Tuple[str, str], Tuple[str, str, str]] = {}  # (data_key, report_title) -> (canonical_key, confidence, reason)

    # 范围
    total = len(data)
    end = total if limit is None else min(total, start + max(0, int(limit)))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if fill_out_path:
        os.makedirs(os.path.dirname(fill_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(save_updates) or ".", exist_ok=True)

    # --- 并发/串行两种执行路径 ---
    if num_workers is None or num_workers <= 1:
        with open(out_path, "w", encoding="utf-8") as wf, \
             open(fill_out_path, "w", encoding="utf-8") if fill_out_path else open(os.devnull, "w") as wf_fill:
            for i in range(start, end):
                rec = data[i]
                report_title = _normalize(rec.get("report_title", ""))
                report_text = rec.get("report", "") or ""
                alias_map: Dict[str, str] = rec.get("alias") or {}
                if not isinstance(alias_map, dict):
                    alias_map = {}

                # alias 的 key 只能来自 keys_merged.json 中该 report_title 下定义的标准字段
                keys_schema = {
                    k for k in _title_keys(struct_map, report_title) if not _is_excluded_key(k)
                }
                # 规范化映射：规避大小写/空白等差异，便于映射回原始标准名
                keys_schema_norm_map = { _normalize(k): k for k in keys_schema }
                # 在进一步处理前，先把已有 alias 的“非标准 key”映射回当前 report_title 的标准 key
                if isinstance(alias_map, dict) and alias_map:
                    canonical_keys_list = list(keys_schema)
                    normalized_alias_map: Dict[str, str] = {}
                    for akey, aval in alias_map.items():
                        akey_n = _normalize(akey)
                        aval_n = _normalize(aval)
                        # 已是本 title 的标准 key：保留（按原标准名收敛）
                        if akey_n in keys_schema_norm_map:
                            canonical_key_orig = keys_schema_norm_map[akey_n]
                            # 若有冲突，保留已存在的（保持“原样截取”优先）
                            normalized_alias_map.setdefault(canonical_key_orig, aval)
                            continue
                        # 需要映射：优先用 alias 的 value 作为“数据字段名”提示，其次回退用原 key
                        data_key_for_mapping = aval_n or akey_n
                        value_for_mapping = _normalize(rec.get(akey, ""))
                        cache_key = (data_key_for_mapping, report_title)
                        if cache_key in map_cache:
                            mapped_key, confidence, reason = map_cache[cache_key]
                        else:
                            mapped_key, confidence, reason = map_to_canonical_key(
                                data_key=data_key_for_mapping,
                                value=value_for_mapping,
                                report_title=report_title,
                                canonical_keys=canonical_keys_list,
                                model_checker=checker,
                            )
                            map_cache[cache_key] = (mapped_key, confidence, reason)
                        # 将规范化的 mapped_key 映射回原始标准名
                        if _normalize(mapped_key) in keys_schema_norm_map:
                            canonical_key_orig = keys_schema_norm_map[_normalize(mapped_key)]
                            normalized_alias_map.setdefault(canonical_key_orig, aval)
                            # 记录 alias-key 重映射日志
                            wf.write(json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "task": "alias_key_remap",
                                "from_key": akey_n,
                                "alias_value": aval_n,
                                "to_canonical": canonical_key_orig,
                                "confidence": confidence,
                                "reason": reason,
                            }, ensure_ascii=False) + "\n")
                        else:
                            # 放弃无法映射到当前 title 标准集之外的 alias 键
                            wf.write(json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "task": "alias_key_remap_failed",
                                "from_key": akey_n,
                                "alias_value": aval_n,
                                "reason": "未映射到本title标准key，已丢弃该alias键",
                            }, ensure_ascii=False) + "\n")
                    alias_map = normalized_alias_map

                # 记录中实际存在的字段
                record_keys: Set[str] = {
                    k for k, v in rec.items()
                    if isinstance(v, str)
                    and k not in {"report", "report_title", "alias", "added_keys"}
                    and not _is_excluded_key(k)
                }
                
                # 对于不在标准字段中的record_keys，调用模型映射到最相关的标准key
                key_mapping: Dict[str, str] = {}  # 记录key -> 标准key的映射
                non_standard_keys = record_keys - keys_schema
                if non_standard_keys:
                    canonical_keys_list = list(keys_schema)
                    for nsk in non_standard_keys:
                        value = rec.get(nsk, "")
                        cache_key = (_normalize(nsk), report_title)
                        if cache_key in map_cache:
                            mapped_key, confidence, reason = map_cache[cache_key]
                        else:
                            mapped_key, confidence, reason = map_to_canonical_key(
                                data_key=nsk,
                                value=value,
                                report_title=report_title,
                                canonical_keys=canonical_keys_list,
                                model_checker=checker,
                            )
                            map_cache[cache_key] = (mapped_key, confidence, reason)
                        # 使用原始标准名，避免后续与 keys_schema 直接比较失败
                        mapped_key_orig = keys_schema_norm_map.get(_normalize(mapped_key), mapped_key)
                        key_mapping[nsk] = mapped_key_orig
                        # 记录映射日志
                        wf.write(json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "task": "map_to_canonical",
                            "data_key": _normalize(nsk),
                            "canonical_key": mapped_key_orig,
                            "confidence": confidence,
                            "reason": reason,
                            "value": _normalize(value)[:50],
                        }, ensure_ascii=False) + "\n")
                
                # 只处理标准字段
                canonical_keys = [k for k in keys_schema if not _is_excluded_key(k)]
                # mentioned_keys: 标准字段中在数据里直接提到的，或通过映射关联到的
                mentioned_keys = set()
                for rk in record_keys:
                    if rk in keys_schema:
                        mentioned_keys.add(rk)
                    elif rk in key_mapping and key_mapping[rk] in keys_schema:
                        mentioned_keys.add(key_mapping[rk])

                # 1) 缺失补充
                for canonical in canonical_keys:
                    # 检查是否通过映射关联到这个标准key
                    mapped_from_key = None
                    for data_key, mapped_key in key_mapping.items():
                        if _normalize(mapped_key) == _normalize(canonical):
                            mapped_from_key = data_key
                            break
                    
                    # 若通过映射关联，并且别名缺失，优先基于 report+value 由 LLM 截取别名
                    if mapped_from_key and (canonical not in alias_map or _is_missing_alias(alias_map.get(canonical))):
                        # 取值优先使用 canonical 的值；没有则回退到原数据键的值
                        value_for_alias = rec.get(canonical, "") or rec.get(mapped_from_key, "")
                        if _normalize(value_for_alias):
                            cache_key = (_normalize(canonical), _normalize(value_for_alias))
                            if cache_key in find_cache:
                                alias, found, reason = find_cache[cache_key]
                            else:
                                alias, found, reason = find_alias_from_report(
                                    canonical=canonical,
                                    value=value_for_alias,
                                    report_title=report_title,
                                    report_text=report_text,
                                    model_checker=checker,
                                )
                                find_cache[cache_key] = (alias, found, reason)
                            forbid = []
                            nm = rec.get('姓名') or rec.get('病人姓名') or rec.get('患者姓名')
                            if isinstance(nm, str) and nm.strip():
                                forbid.append(nm)
                            alias_map[canonical] = _sanitize_alias_value(canonical, alias, value_for_alias, forbid)
                            wf.write(json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "task": "fill_from_llm",
                                "canonical": _normalize(canonical),
                                "value": _normalize(value_for_alias),
                                "alias": alias_map[canonical],
                                "found": bool(found),
                                "locators": _collect_locators(rec),
                                "context": _context_around_value(report_text, _normalize(value_for_alias))[:240],
                            }, ensure_ascii=False) + "\n")
                            wf_fill.write((json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "canonical": _normalize(canonical),
                                "value": _normalize(value_for_alias),
                                "alias": alias_map[canonical],
                                "found": bool(found),
                                "locators": _collect_locators(rec),
                                "context": _context_around_value(report_text, _normalize(value_for_alias))[:240],
                            }, ensure_ascii=False) + "\n") if fill_out_path else "")
                            continue
                        else:
                            # 没有值可供定位，回退为“无值置空”，遵守“别名来自报告”的约束
                            alias_map[canonical] = ""
                            wf.write(json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "task": "fill_missing",
                                "canonical": _normalize(canonical),
                                "value": "",
                                "alias": "",
                                "found": False,
                                "locators": _collect_locators(rec),
                                "context": "",
                            }, ensure_ascii=False) + "\n")
                            wf_fill.write((json.dumps({
                                "index": i,
                                "report_title": report_title,
                                "canonical": _normalize(canonical),
                                "value": "",
                                "alias": "",
                                "found": False,
                                "locators": _collect_locators(rec),
                                "context": "",
                            }, ensure_ascii=False) + "\n") if fill_out_path else "")
                            continue
                    
                    # 未提到的 key -> 直接置为空串
                    if canonical not in mentioned_keys:
                        alias_map[canonical] = ""
                        wf.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "report_title": report_title,
                                    "task": "fill_missing",
                                    "canonical": _normalize(canonical),
                                    "value": _normalize(rec.get(canonical, "")),
                                    "alias": "",
                                    "found": False,
                                    "locators": _collect_locators(rec),
                                    "context": _context_around_value(report_text, _normalize(rec.get(canonical, "")))[:240],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        wf_fill.write((json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "canonical": _normalize(canonical),
                            "value": _normalize(rec.get(canonical, "")),
                            "alias": "",
                            "found": False,
                            "locators": _collect_locators(rec),
                            "context": _context_around_value(report_text, _normalize(rec.get(canonical, "")))[:240],
                        }, ensure_ascii=False) + "\n") if fill_out_path else "")
                        continue
                    if canonical in alias_map and not _is_missing_alias(alias_map.get(canonical)):
                        continue
                    value = rec.get(canonical, "")
                    # 无值：不调用模型，直接置为空串（不打印日志）
                    if not _normalize(value):
                        alias_map[canonical] = ""
                        wf.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "report_title": report_title,
                                    "task": "fill_missing",
                                    "canonical": _normalize(canonical),
                                    "value": "",
                                    "alias": "",
                                    "found": False,
                                    "locators": _collect_locators(rec),
                                    "context": "",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        wf_fill.write((json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "canonical": _normalize(canonical),
                            "value": "",
                            "alias": "",
                            "found": False,
                            "locators": _collect_locators(rec),
                            "context": "",
                        }, ensure_ascii=False) + "\n") if fill_out_path else "")
                        continue
                    cache_key = (_normalize(canonical), _normalize(value))
                    if cache_key in find_cache:
                        alias, found, reason = find_cache[cache_key]
                    else:
                        alias, found, reason = find_alias_from_report(
                            canonical=canonical,
                            value=value,
                            report_title=report_title,
                            report_text=report_text,
                            model_checker=checker,
                        )
                        find_cache[cache_key] = (alias, found, reason)
                    # 回写：只写入 key 本身或别名，禁止占位符/其他内容（不打印填充日志）
                    forbid = []
                    nm = rec.get('姓名') or rec.get('病人姓名') or rec.get('患者姓名')
                    if isinstance(nm, str) and nm.strip():
                        forbid.append(nm)
                    alias_map[canonical] = _sanitize_alias_value(canonical, alias, value, forbid)
                    wf.write(
                        json.dumps(
                            {
                                "index": i,
                                "report_title": report_title,
                                "task": "fill_missing",
                                "canonical": _normalize(canonical),
                                "value": _normalize(value),
                                "alias": alias_map[canonical],
                                "found": bool(found),
                                "locators": _collect_locators(rec),
                                "context": _context_around_value(report_text, _normalize(value))[:240],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    wf_fill.write((json.dumps({
                        "index": i,
                        "report_title": report_title,
                        "canonical": _normalize(canonical),
                        "value": _normalize(value),
                        "alias": alias_map[canonical],
                        "found": bool(found),
                        "locators": _collect_locators(rec),
                        "context": _context_around_value(report_text, _normalize(value))[:240],
                    }, ensure_ascii=False) + "\n") if fill_out_path else "")

                # （已移除：对现有 alias 的准确性检查）

                # 回写 alias_map：统一做一次清洗，确保值严格为 key 或别名字符串
                clean_alias_map: Dict[str, str] = {}
                for k2, v2 in (alias_map or {}).items():
                    # 未提到的键等缺失场景允许为空串写回
                    clean_alias_map[_normalize(k2)] = "" if _is_missing_alias(v2) else _sanitize_alias_value(k2, v2)
                rec["alias"] = clean_alias_map

                if (i - start + 1) % 20 == 0 or i + 1 == end:
                    print(f"[progress] {i+1}/{end} fill+check -> {out_path}")
    else:
        # --- 并发执行：按记录维度并行 ---
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        cache_lock = threading.Lock()

        def handle_one(i: int):
            rec = data[i]
            report_title = _normalize(rec.get("report_title", ""))
            report_text = rec.get("report", "") or ""
            alias_map: Dict[str, str] = rec.get("alias") or {}
            if not isinstance(alias_map, dict):
                alias_map = {}

            log_lines: List[str] = []
            print_lines: List[str] = []

            # alias 的 key 只能来自 keys_merged.json 中该 report_title 下定义的标准字段
            keys_schema = {
                k for k in _title_keys(struct_map, report_title) if not _is_excluded_key(k)
            }
            keys_schema_norm_map = { _normalize(k): k for k in keys_schema }
            # 在进一步处理前，先把已有 alias 的“非标准 key”映射回当前 report_title 的标准 key
            if isinstance(alias_map, dict) and alias_map:
                canonical_keys_list = list(keys_schema)
                normalized_alias_map: Dict[str, str] = {}
                for akey, aval in alias_map.items():
                    akey_n = _normalize(akey)
                    aval_n = _normalize(aval)
                    if akey_n in keys_schema_norm_map:
                        canonical_key_orig = keys_schema_norm_map[akey_n]
                        normalized_alias_map.setdefault(canonical_key_orig, aval)
                        continue
                    data_key_for_mapping = aval_n or akey_n
                    value_for_mapping = _normalize(rec.get(akey, ""))
                    cache_key = (data_key_for_mapping, report_title)
                    with cache_lock:
                        cached = map_cache.get(cache_key)
                    if cached is not None:
                        mapped_key, confidence, reason = cached
                    else:
                        mapped_key, confidence, reason = map_to_canonical_key(
                            data_key=data_key_for_mapping,
                            value=value_for_mapping,
                            report_title=report_title,
                            canonical_keys=canonical_keys_list,
                            model_checker=checker,
                        )
                        with cache_lock:
                            map_cache[cache_key] = (mapped_key, confidence, reason)
                    if _normalize(mapped_key) in keys_schema_norm_map:
                        canonical_key_orig = keys_schema_norm_map[_normalize(mapped_key)]
                        normalized_alias_map.setdefault(canonical_key_orig, aval)
                        log_lines.append(json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "task": "alias_key_remap",
                            "from_key": akey_n,
                            "alias_value": aval_n,
                            "to_canonical": canonical_key_orig,
                            "confidence": confidence,
                            "reason": reason,
                        }, ensure_ascii=False))
                    else:
                        log_lines.append(json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "task": "alias_key_remap_failed",
                            "from_key": akey_n,
                            "alias_value": aval_n,
                            "reason": "未映射到本title标准key，已丢弃该alias键",
                        }, ensure_ascii=False))
                alias_map = normalized_alias_map

            # 记录中实际存在的字段
            record_keys: Set[str] = {
                k for k, v in rec.items()
                if isinstance(v, str)
                and k not in {"report", "report_title", "alias", "added_keys"}
                and not _is_excluded_key(k)
            }
            
            # 对于不在标准字段中的record_keys，调用模型映射到最相关的标准key
            key_mapping: Dict[str, str] = {}  # 记录key -> 标准key的映射
            non_standard_keys = record_keys - keys_schema
            if non_standard_keys:
                canonical_keys_list = list(keys_schema)
                for nsk in non_standard_keys:
                    value = rec.get(nsk, "")
                    cache_key = (_normalize(nsk), report_title)
                    with cache_lock:
                        cached = map_cache.get(cache_key)
                    if cached is not None:
                        mapped_key, confidence, reason = cached
                    else:
                        mapped_key, confidence, reason = map_to_canonical_key(
                            data_key=nsk,
                            value=value,
                            report_title=report_title,
                            canonical_keys=canonical_keys_list,
                            model_checker=checker,
                        )
                        with cache_lock:
                            map_cache[cache_key] = (mapped_key, confidence, reason)
                
                    mapped_key_orig = keys_schema_norm_map.get(_normalize(mapped_key), mapped_key)
                    key_mapping[nsk] = mapped_key_orig
                    # 记录映射日志
                    log_lines.append(json.dumps({
                        "index": i,
                        "report_title": report_title,
                        "task": "map_to_canonical",
                        "data_key": _normalize(nsk),
                        "canonical_key": mapped_key_orig,
                        "confidence": confidence,
                        "reason": reason,
                        "value": _normalize(value)[:50],
                    }, ensure_ascii=False))
            
            # 只处理标准字段
            canonical_keys = [k for k in keys_schema if not _is_excluded_key(k)]
            # mentioned_keys: 标准字段中在数据里直接提到的，或通过映射关联到的
            mentioned_keys = set()
            for rk in record_keys:
                if rk in keys_schema:
                    mentioned_keys.add(rk)
                elif rk in key_mapping and key_mapping[rk] in keys_schema:
                    mentioned_keys.add(key_mapping[rk])

            # 1) 缺失补充
            for canonical in canonical_keys:
                # 检查是否通过映射关联到这个标准key
                mapped_from_key = None
                for data_key, mapped_key in key_mapping.items():
                    if _normalize(mapped_key) == _normalize(canonical):
                        mapped_from_key = data_key
                        break
                
                # 若通过映射关联，并且别名缺失，优先基于 report+value 由 LLM 截取别名
                if mapped_from_key and (canonical not in alias_map or _is_missing_alias(alias_map.get(canonical))):
                    value_for_alias = rec.get(canonical, "") or rec.get(mapped_from_key, "")
                    if _normalize(value_for_alias):
                        cache_key = (_normalize(canonical), _normalize(value_for_alias))
                        with cache_lock:
                            cached = find_cache.get(cache_key)
                        if cached is not None:
                            alias, found, reason = cached
                        else:
                            alias, found, reason = find_alias_from_report(
                                canonical=canonical,
                                value=value_for_alias,
                                report_title=report_title,
                                report_text=report_text,
                                model_checker=checker,
                            )
                            with cache_lock:
                                find_cache[cache_key] = (alias, found, reason)
                        forbid = []
                        nm = rec.get('姓名') or rec.get('病人姓名') or rec.get('患者姓名')
                        if isinstance(nm, str) and nm.strip():
                            forbid.append(nm)
                        alias_map[canonical] = _sanitize_alias_value(canonical, alias, value_for_alias, forbid)
                        log_lines.append(json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "task": "fill_from_llm",
                            "canonical": _normalize(canonical),
                            "value": _normalize(value_for_alias),
                            "alias": alias_map[canonical],
                            "found": bool(found),
                        }, ensure_ascii=False))
                        continue
                    else:
                        alias_map[canonical] = ""
                        log_lines.append(json.dumps({
                            "index": i,
                            "report_title": report_title,
                            "task": "fill_missing",
                            "canonical": _normalize(canonical),
                            "value": "",
                            "alias": "",
                            "found": False,
                            "reason": "映射键无可用值，置空",
                        }, ensure_ascii=False))
                        continue
                
                # 未提到的 key -> 直接置为空串
                if canonical not in mentioned_keys:
                    alias_map[canonical] = ""
                    log_lines.append(json.dumps({
                        "index": i,
                        "report_title": report_title,
                        "task": "fill_missing",
                        "canonical": _normalize(canonical),
                        "value": _normalize(rec.get(canonical, "")),
                        "alias": "",
                        "found": False,
                        "reason": "未提到的键，置为空",
                    }, ensure_ascii=False))
                    continue
                if canonical in alias_map and not _is_missing_alias(alias_map.get(canonical)):
                    continue
                value = rec.get(canonical, "")
                if not _normalize(value):
                    alias_map[canonical] = ""
                    log_lines.append(json.dumps({
                        "index": i,
                        "report_title": report_title,
                        "task": "fill_missing",
                        "canonical": _normalize(canonical),
                        "value": "",
                        "alias": "",
                        "found": False,
                        "reason": "字段值为空，置为空",
                    }, ensure_ascii=False))
                    continue

                cache_key = (_normalize(canonical), _normalize(value))
                with cache_lock:
                    cached = find_cache.get(cache_key)
                if cached is not None:
                    alias, found, reason = cached
                else:
                    alias, found, reason = find_alias_from_report(
                        canonical=canonical,
                        value=value,
                        report_title=report_title,
                        report_text=report_text,
                        model_checker=checker,
                    )
                    with cache_lock:
                        find_cache[cache_key] = (alias, found, reason)

                prev_val = alias_map.get(canonical, "")
                forbid = []
                nm = rec.get('姓名') or rec.get('病人姓名') or rec.get('患者姓名')
                if isinstance(nm, str) and nm.strip():
                    forbid.append(nm)
                alias_map[canonical] = _sanitize_alias_value(canonical, alias, value, forbid)
                new_val = alias_map[canonical]
                # 减少日志：填充阶段不打印，避免噪声
                log_lines.append(json.dumps({
                    "index": i,
                    "report_title": report_title,
                    "task": "fill_missing",
                    "canonical": _normalize(canonical),
                    "value": _normalize(value),
                    "alias": alias_map[canonical],
                    "found": bool(found),
                    "reason": reason,
                }, ensure_ascii=False))

            # （已移除：对现有 alias 的准确性检查）

            # 清洗 alias_map
            clean_alias_map: Dict[str, str] = {}
            for k2, v2 in (alias_map or {}).items():
                # 未提到的键等缺失场景允许为空串写回
                clean_alias_map[_normalize(k2)] = "" if _is_missing_alias(v2) else _sanitize_alias_value(k2, v2)
            rec["alias"] = clean_alias_map

            return i, rec, log_lines, print_lines

        with open(out_path, "w", encoding="utf-8") as wf, \
             open(fill_out_path, "w", encoding="utf-8") if fill_out_path else open(os.devnull, "w") as wf_fill:
            with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
                futures = [ex.submit(handle_one, i) for i in range(start, end)]
                done = 0
                for fut in as_completed(futures):
                    i2, rec2, lines, prints = fut.result()
                    # 写日志与打印
                    for pl in prints:
                        print(pl)
                    # lines 中既有 fill 也有 check，将 fill 单独写入 fill_out
                    for ln in lines:
                        wf.write(ln + "\n")
                        try:
                            obj = json.loads(ln)
                            if obj.get("task") == "fill_missing" and fill_out_path:
                                wf_fill.write(ln + "\n")
                        except Exception:
                            pass
                    data[i2] = rec2
                    done += 1
                    if done % 20 == 0 or done == len(futures):
                        print(f"[progress] {done}/{len(futures)} fill+check (concurrent) -> {out_path}")

    # 写入更新后的全量 JSON
    _atomic_write_json(data, save_updates)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="使用 LLM 对 alias 进行规范化与缺失补充（不做准确性检查）"
    )
    ap.add_argument("--input", default="data/merged.converted.json", help="输入 JSON 文件路径")
    ap.add_argument("--out", default="data/alias_check_report.jsonl", help="输出 JSONL 结果路径")
    ap.add_argument("--save_updates", default=None, help="更新后的JSON输出路径；未提供则原地写回到 --input")
    ap.add_argument("--struct", default="keys/keys_merged.json", help="结构映射(标准键)文件")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（可选）")
    ap.add_argument("--start", type=int, default=0, help="起始索引（可选）")
    ap.add_argument("--model", default="qwen3-32b", help="模型名称，默认 qwen3-32b")
    ap.add_argument("--base_url", default=None, help="覆盖默认 BASE_URL（可选）")
    ap.add_argument("--api_key", default=None, help="覆盖默认 API_KEY（可选）")
    ap.add_argument("--timeout", type=int, default=30, help="调用超时（秒）")
    return ap.parse_args(argv)


def run_alias_update(
    *,
    input_path: str = "data/merged.converted.json",
    out_path: str = "data/alias_check_report.jsonl",
    save_updates: Optional[str] = None,
    struct_path: str = "keys/keys_merged.json",
    mode: str = "fill",
    limit: Optional[int] = None,
    start: int = 0,
    model: str = "qwen3-32b",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
    num_workers: int = 8,
    fill_out_path: Optional[str] = "data/alias_fill.jsonl",
) -> None:
    """无需命令行，直接以参数方式运行。默认原地写回 input_path。"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    suppress_verbose_http_logs()
    run_fill_and_check(
        input_path=input_path,
        out_path=out_path,
        save_updates=(save_updates or input_path),
        struct_path=struct_path,
        fill_out_path=fill_out_path,
        limit=limit,
        start=start,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        num_workers=num_workers,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """默认不读取命令行，直接以内置默认参数运行（原地写回）。"""
    run_alias_update()


if __name__ == "__main__":
    main()
