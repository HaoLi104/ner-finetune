# -*- coding: utf-8 -*-
from __future__ import annotations

"""
准备数据（两遍调度：强制覆盖全部“可覆盖别名” + 高频优先随机丢 key）
- Pass1：对样本做“丢 key + 结构外过滤”，收集 (title, canonical) 出现位置；为每个组制定别名覆盖计划（尽量让每个别名至少出现一次）。
- Pass2：按计划强制覆盖显示名，组装 report（项间分隔符空格/。/；/，轮换；超长换段 \n\n），并输出两份文件：
  1) *_drop_keys.json         —— 仅体现“丢 key”的结果；
  2) *_key_alias.json         —— meta.alias2canonical={"住院号码":"住院号", ...}。
- 统计：值保真、丢 key 明细；别名覆盖=输入别名覆盖&显示别名覆盖（全局/按标题），并拆分“可覆盖/不可覆盖”。
"""

import json, re, random, math, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set
from collections import defaultdict

# 可选：用于 LLM 排序（保留）
from llm_client import OpenAIFieldWiseLLM  # type: ignore

# 项目根目录
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc

# ===== tokenizer (可选) =====
_TOKENIZER = None
try:
    from tokenizer import HFTokenizer  # type: ignore
except Exception:
    HFTokenizer = None  # type: ignore
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
def set_tokenizer(tok: Any) -> None:
    global _TOKENIZER
    _TOKENIZER = tok


def _token_count(s: str) -> int:
    if _TOKENIZER is None:
        return len(s or "")
    try:
        if hasattr(_TOKENIZER, "encode"):
            return len(_TOKENIZER.encode(s))
        if hasattr(_TOKENIZER, "__call__"):
            out = _TOKENIZER(s)
            if isinstance(out, dict) and "input_ids" in out:
                return len(out["input_ids"])
    except Exception:
        pass
    return len(s or "")


from structs import (  # type: ignore
    get_report_structure_map,
    get_key_alias_maps,
    normalize_key_name,
)

RESERVED = {"report", "report_title", "meta", "added_keys", "report_titles"}


# ---------------- 基础工具 ----------------
def _one_line(s: Any) -> str:
    ss = str(s or "")
    ss = ss.replace("\r", "\n").replace("\t", " ").replace("\n", " ")
    ss = re.sub(r"\s+", " ", ss).strip()
    return ss


def _estimate_len(s: str) -> int:
    return _token_count(s or "")


def _tidy_punct_spaces(s: str) -> str:
    s = re.sub(r"\s+([。；，,、！？:：])", r"\1", s)
    s = re.sub(r"([。！？；，,:：])\1+", r"\1", s)
    return s


# ---------------- LLM 排序（保留） ----------------
def _llm_pick_keys_list(
    llm: Optional[OpenAIFieldWiseLLM],
    title: str,
    present_keys: List[str],
) -> List[str]:
    """
    使用 LLM 对“本条样本实际出现的 key”排序。
    约束：必须**严格保全**字段集合；若集合不一致（丢失/新增/改名），则返回 [] 放弃采用 LLM 顺序。
    - llm 允许为 None（直接返回 []）
    """
    if not present_keys or llm is None:
        return []

    prompt = (
        "请将【待排序字段】按照常见医学报告的标准编排顺序进行排序，"
        "必须保留全部字段，不允许新增、删除或修改字段名称。"
        "仅输出一行 JSON 数组，数组元素为字段原名字符串。"
        "不要输出任何解释、注释或多余文本。\n"
        f"【报告类型】{title}\n"
        f"【待排序字段】{', '.join(present_keys)}\n"
        '【输出格式示例】["姓名", "年龄", "检查日期"]'
    )

    try:
        ret = llm._post_report({"report": prompt})  # type: ignore[attr-defined]
        payload = ret.get("report", ret) if isinstance(ret, dict) else ret
        s = None
        if isinstance(payload, str):
            s = payload
        elif isinstance(payload, dict):
            s = payload.get("llm_ret")
        if isinstance(s, str):
            s = s.replace("```json", "").replace("```", "").strip()
            arr = json.loads(s)
            if isinstance(arr, list):
                out = [str(x) for x in arr if isinstance(x, (str, int, float))]
                # 集合一致性校验（严格）：元素相同且无重复
                if set(out) == set(present_keys) and len(out) == len(set(out)):
                    return out
    except Exception:
        pass
    return []


def _order_by_structure(
    title: str, fields: List[Tuple[str, str]], struct_map: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    keys_map = struct_map.get(title)
    if not keys_map:
        return fields
    existed = {k for k, _ in fields}
    ordered = [k for k in keys_map if k in existed]
    rest = [k for k, _ in fields if k not in set(ordered)]
    d = dict(fields)
    return [(k, d[k]) for k in (ordered + rest)]


def _load_title_alias_mapping_fallback(
    path: str = "keys/ALIAS_MAP_0919.json",
) -> Dict[str, Dict[str, List[str]]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ---------------- 别名均衡器（含“项内/项间分隔符”） ----------------
class GlobalBalancer:
    def __init__(
        self,
        canon2aliases: Dict[str, List[str]],
        title_canon2aliases: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> None:
        self.canon2aliases = canon2aliases or {}
        self.title_canon2aliases = title_canon2aliases or {}
        # 统计
        self.kv_sep_usage: Dict[str, int] = {"colon": 0, "space": 0, "blank": 0}
        self.item_sep_usage: Dict[str, int] = {
            "space": 0,  # " "
            "period": 0,  # "。"
            "semicolon": 0,  # "；"
            "comma": 0,  # "，"
        }

    # —— 项内分隔符（key 与 value 之间） ——
    def choose_kv_separator_for_shortline(self) -> str:
        try:
            use_blank = random.random() < 0.10
        except Exception:
            use_blank = False
        if use_blank:
            self.kv_sep_usage["blank"] = self.kv_sep_usage.get("blank", 0) + 1
            return " "
        c = self.kv_sep_usage.get("colon", 0)
        s = self.kv_sep_usage.get("space", 0)
        if c <= s:
            self.kv_sep_usage["colon"] = c + 1
            return "："
        else:
            self.kv_sep_usage["space"] = s + 1
            return ":"

    # —— 项间分隔符（组与组之间） ——
    def choose_item_group_separator(self) -> str:
        usage = self.item_sep_usage
        candidates = [
            (usage.get("space", 0), " "),
            (usage.get("period", 0), "。"),
            (usage.get("semicolon", 0), "；"),
            (usage.get("comma", 0), "，"),
        ]
        minc = min(c for c, _ in candidates)
        pool = [sep for c, sep in candidates if c == minc]
        sep = random.choice(pool)
        key = {" ": "space", "。": "period", "；": "semicolon", "，": "comma"}[sep]
        usage[key] = usage.get(key, 0) + 1
        return sep


# ---------------- 归一到 canonical（返回值合并 + 源别名） ----------------
def _canonicalize_fields_for_title(
    title: str,
    fields: List[Tuple[str, str]],
    title_canon2aliases: Dict[str, Dict[str, List[str]]],
    alias2canon_global: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]], Dict[str, str]]:
    rev_title: Dict[str, str] = {}
    tmap = title_canon2aliases.get(title, {}) or {}
    for canon, aliases in tmap.items():
        rev_title[normalize_key_name(canon)] = canon
        for a in aliases or []:
            rev_title[normalize_key_name(a)] = canon

    def to_canon(alias: str) -> str:
        kn = normalize_key_name(alias)
        if kn in rev_title:
            return rev_title[kn]
        g = alias2canon_global.get(kn)
        return g if g else alias

    order: List[str] = []
    bag: Dict[str, List[str]] = {}
    seen_val: Dict[str, set[str]] = {}
    alias_sources: Dict[str, List[str]] = {}
    alias2canon_used: Dict[str, str] = {}

    for k, v in fields:
        canon = to_canon(k)
        if canon not in bag:
            bag[canon] = []
            seen_val[canon] = set()
            order.append(canon)
            alias_sources[canon] = []
        vv = _one_line(v)
        if vv and vv not in seen_val[canon]:
            bag[canon].append(vv)
            seen_val[canon].add(vv)
        if k not in alias_sources[canon]:
            alias_sources[canon].append(k)
        if k != canon:
            alias2canon_used[k] = canon

    out: List[Tuple[str, str]] = []
    for canon in order:
        vals = bag.get(canon, [])
        if not vals:
            continue
        out.append((canon, "；".join(vals)))
    return out, alias_sources, alias2canon_used


# ---------------- 频次统计（供“高频优先丢”使用） ----------------
def _build_freq_tables(
    data: List[Dict[str, Any]],
    title_canon2aliases: Dict[str, Dict[str, List[str]]],
    alias2canon_global: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    freq_by_title: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    freq_global: Dict[str, int] = defaultdict(int)

    for rec in data:
        title = str(rec.get("report_title", "")).strip()
        if not title:
            continue
        fields = [
            (k, _one_line(v))
            for k, v in rec.items()
            if k not in RESERVED and isinstance(k, str) and k.strip() and _one_line(v)
        ]
        fields_canon, _, _ = _canonicalize_fields_for_title(
            title, fields, title_canon2aliases, alias2canon_global
        )
        for canon, _ in fields_canon:
            freq_by_title[title][canon] += 1
            freq_global[canon] += 1
    return freq_by_title, freq_global


def _quantile_threshold(counts: List[int], q: float) -> int:
    if not counts:
        return math.inf
    s = sorted(counts)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s)) - 1)))
    return s[idx]


# ---------------- 渲染底层 ----------------
def _render_items(items: List[Tuple[str, str, str]]) -> List[str]:
    out: List[str] = []
    for _canon, sep, val in items:
        if sep == "" or sep is None:
            out.append(_one_line(val))
        else:
            out.append(_one_line(f"{sep[:-1]}{sep[-1]}{val}"))
    return out


def _pack_paragraphs_with_seps(
    items_text: List[str], length_threshold: int, *, choose_sep: callable
) -> List[str]:
    paragraphs: List[str] = []
    cur = ""
    cur_len = 0
    for t in items_text:
        t = _one_line(t)
        if not t:
            continue
        if not cur:
            cur = t
            cur_len = _estimate_len(t)
            continue
        sep = choose_sep()
        need = _estimate_len(sep) + _estimate_len(t)
        if cur_len + need > length_threshold:
            paragraphs.append(cur)
            cur = t
            cur_len = _estimate_len(t)
        else:
            cur = f"{cur}{sep}{t}"
            cur_len += need
    if cur:
        paragraphs.append(cur)
    return [_tidy_punct_spaces(p) for p in paragraphs]


# ---------------- 计划驱动的 KV 三元组生成（支持强制显示名） ----------------
def _compose_kv_triplets_with_plan(
    rec: Dict[str, Any],
    struct_map: Dict[str, List[str]],
    balancer: GlobalBalancer,
    *,
    preordered_fields: Optional[List[Tuple[str, str]]] = None,
    order_by_struct: bool = True,
    drop_unknown_keys: bool,
    title_canon2aliases: Dict[str, Dict[str, List[str]]],
    alias2canon_global: Dict[str, str],
    planned_display_for_record: Optional[Dict[str, str]] = None,  # canonical -> display
) -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str]],
    Dict[str, str],  # display_map
    Dict[str, List[str]],
    Dict[str, str],  # source aliases -> canonical
    Dict[str, str],  # display aliases actually used -> canonical
]:
    # Update Pass2 filtering logic in _compose_kv_triplets_with_plan
    title = str(rec.get("report_title", "")).strip()

    std = struct_map.get(title, []) or []
    std_norms = {normalize_key_name(k) for k in std}
    allowed_norm = set(std_norms)
    
    # 对于不在结构映射中的类型，保留所有字段（不过滤）
    title_in_struct = bool(std)

    # Title-specific aliases
    tmap = title_canon2aliases.get(title, {}) or {}
    for canon in std:
        for a in (tmap.get(canon) or []):
            allowed_norm.add(normalize_key_name(a))

    # Global aliases pointing to this title's standard keys
    for a, canon in alias2canon_global.items():
        if normalize_key_name(canon) in std_norms:
            allowed_norm.add(normalize_key_name(a))

    raw_fields: List[Tuple[str, str]] = []
    for k, v in rec.items():
        if k in RESERVED:
            continue
        if not isinstance(k, str) or not k.strip():
            continue
        # 修改过滤逻辑：对于不在结构中的类型，保留所有字段
        if drop_unknown_keys and title_in_struct and normalize_key_name(k) not in allowed_norm:
            continue
        vs = _one_line(v)
        if vs:
            raw_fields.append((k, vs))

    fields = list(preordered_fields) if preordered_fields is not None else raw_fields
    fields, alias_sources, alias2canon_used = _canonicalize_fields_for_title(
        title, fields, title_canon2aliases, alias2canon_global
    )

    if order_by_struct:
        fields = _order_by_structure(title, fields, struct_map)

    display_map: Dict[str, str] = {}
    items: List[Tuple[str, str, str]] = []
    alias_display_map: Dict[str, str] = {}
    for canon, vs in fields:
        if "检验数据" in str(canon):
            items.append((canon, "", vs))
            display_map[canon] = canon
            # ★ 修改：也记录检验数据的映射
            alias_display_map[canon] = canon
            continue
        # —— 特殊处理医院名和报告类型：只显示value，不显示key ——
        if canon in ["医院名", "报告类型"]:
            items.append((canon, "", vs))  # 使用空分隔符，只显示value
            display_map[canon] = canon
            # ★ 修改：也记录特殊字段的映射
            alias_display_map[canon] = canon
            continue
        # —— 若有计划则强制使用（保障覆盖），否则走均衡选择 ——
        if planned_display_for_record and canon in planned_display_for_record:
            disp = planned_display_for_record[canon]
        else:
            # 覆盖已在"计划阶段"保障，这里仅做兜底
            disp = canon
        sep_ch = balancer.choose_kv_separator_for_shortline()
        items.append((canon, f"{disp}{sep_ch}", vs))
        display_map[canon] = disp
        # ★ 修改：无论是否使用别名，都记录显示名到canonical的映射
        if disp:
            alias_display_map[disp] = canon

    return (
        items,
        raw_fields,
        display_map,
        alias_sources,
        alias2canon_used,
        alias_display_map,
    )


# ---------------- 主流程（两遍调度） ----------------
def compose_reports_dual_outputs(
    in_path: str,
    out_path_drop_keys: str,
    out_path_key_alias: str,
    *,
    length_threshold: int = 500,
    stats_path: Optional[str] = None,
    preview_dirty_limit: int = 50,
    drop_unknown_keys: bool = False,
    ai_pick_keys: bool = False,  # 计划模式下通常不再调用 LLM 排序，这里留兜底
    llm: Optional[OpenAIFieldWiseLLM] = None,
    value_loss_debug_path: Optional[str] = None,
    # 高频优先丢 key
    drop_keys_enable: bool = False,
    drop_keys_ratio: float = 0.0,  # 0~1
    drop_keys_quantile: float = 0.7,  # [0,1]
    drop_keys_scope: str = "per_type",  # "per_type" or "global"
    # 报告类型过滤（数量限制已在上游处理）
    target_report_types: Optional[List[str]] = None,  # 目标报告类型列表，None表示处理所有类型
    # max_samples_per_type 参数已删除，数量控制在上游 run_da.py 中完成
) -> None:
    # 读入数据
    data: List[Dict[str, Any]] = json.loads(Path(in_path).read_text(encoding="utf-8"))
    
    # 按报告类型过滤（数量控制已在上游完成）
    if target_report_types is not None:
        filtered_data = []
        type_counts = {}
        
        for rec in data:
            title = str(rec.get("report_title", "")).strip()
            
            # 检查报告类型
            if target_report_types is not None and title not in target_report_types:
                continue
            
            # 统计类型数量（仅用于显示）
            type_counts[title] = type_counts.get(title, 0) + 1
            filtered_data.append(rec)
        
        print(f"[filter] original data: {len(data)} records")
        print(f"[filter] after filtering by report types {target_report_types}: {len(filtered_data)} records")
        for title, count in type_counts.items():
            print(f"  {title}: {count} samples")
        
        data = filtered_data

    # 确保 REPORT_STRUCTURE_MAP 使用最新的结构映射
    from structs import REPORT_STRUCTURE_MAP

    # 准备别名映射（用于把“允许的键”扩展到别名）
    try:
        alias2canon, canon2aliases, title_canon2aliases = get_key_alias_maps()
    except Exception:
        # 兼容旧返回
        ret = get_key_alias_maps()
        if isinstance(ret, tuple) and len(ret) == 3:
            alias2canon, canon2aliases, title_canon2aliases = ret  # type: ignore
        else:
            alias2canon, canon2aliases = ret  # type: ignore
            title_canon2aliases = {}

    # 确保 alias2canon_global 在所有上下文中正确传递
    alias2canon_global = alias2canon  # 确保全局别名映射一致

    # Refactored _allowed_norms_for_title to ensure consistency and alias normalization
    def _allowed_norms_for_title(title: str) -> Set[str]:
        # Use REPORT_STRUCTURE_MAP for consistency
        std_keys = REPORT_STRUCTURE_MAP.get(title, []) or []
        std_norms = {normalize_key_name(k) for k in std_keys}
        allow = set(std_norms)

        # Title-specific aliases
        tmap = title_canon2aliases.get(title, {}) or {}
        for canon in std_keys:
            for a in (tmap.get(canon) or []):
                allow.add(normalize_key_name(a))

        # Global aliases pointing to this title's standard keys
        for a, canon in alias2canon_global.items():
            if normalize_key_name(canon) in std_norms:
                allow.add(normalize_key_name(a))

        return allow

    # —— 真正的“过滤到结构内” —— 保留保留键 + 标准名/别名命中到的键
    filtered_data: List[Dict[str, Any]] = []
    for record in data:
        title = str(record.get("report_title", "")).strip()
        allow_norm = _allowed_norms_for_title(title)
        keep: Dict[str, Any] = {}
        for k, v in record.items():
            if k in RESERVED:
                keep[k] = v                      # 永久保留
            elif normalize_key_name(k) in allow_norm:
                keep[k] = v                      # 标准名或别名命中
            # 其余键丢弃
        # 确保存在 report_title
        if "report_title" not in keep:
            keep["report_title"] = title
        filtered_data.append(keep)

    # 关键：后续统计/组装一律基于“已过滤”的数据
    data = filtered_data

    # （不在这里写盘；最终仍由函数末尾一次性写出 out_path_drop_keys/out_path_key_alias）

    # 确保 Pass2 使用与前面一致的结构映射
    STRUCT_PATH = "keys/keys_merged.json"
    struct_map = get_report_structure_map(STRUCT_PATH)

    balancer = GlobalBalancer(canon2aliases, title_canon2aliases)

    # 频次统计（供“高频优先丢”）
    freq_by_title, freq_global = _build_freq_tables(
        data, title_canon2aliases, alias2canon
    )
    if drop_keys_scope == "per_type":
        hf_thresh_by_title = {
            t: _quantile_threshold(list(mp.values()), drop_keys_quantile)
            for t, mp in freq_by_title.items()
        }
        hf_thresh_global = None
    else:
        hf_thresh_by_title = {}
        hf_thresh_global = _quantile_threshold(
            list(freq_global.values()), drop_keys_quantile
        )

    rng = random.Random(2025)

    # ========== Pass1：预处理 + 收集覆盖计划所需信息 ==========
    # 保存“丢 key + 过滤”之后的样本，以便 Pass2 直接使用
    prepped_records: List[Dict[str, Any]] = [None] * len(data)  # type: ignore

    # (title, canon) -> [record_idx, ...]
    occ_by_title_canon: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    # 统计输入侧别名覆盖（样本字段名里的别名）
    alias_universe_global: Set[str] = set()
    alias_universe_by_title: Dict[str, Set[str]] = defaultdict(set)
    alias_seen_global_input: Set[str] = set()
    alias_seen_by_title_input: Dict[str, Set[str]] = defaultdict(set)

    # 统计“可覆盖/不可覆盖”所需：出现过的 canonical 集
    canon_seen_global: Set[str] = set()
    canon_seen_by_title: Dict[str, Set[str]] = defaultdict(set)

    # 初始化别名全集（标题内 + 全局）
    for t, cmap in (title_canon2aliases or {}).items():
        for _c, als in (cmap or {}).items():
            for a in als or []:
                alias_universe_by_title[t].add(normalize_key_name(a))
                alias_universe_global.add(normalize_key_name(a))
    for _c_g, als_g in (canon2aliases or {}).items():
        for a in als_g or []:
            alias_universe_global.add(normalize_key_name(a))

    drop_no_alias_total = 0
    drop_has_alias_total = 0
    removed_unknown_counter = defaultdict(int)
    dirty_preview: List[Dict[str, Any]] = []

    def to_canon(title: str, alias: str) -> str:
        kn = normalize_key_name(alias)
        tmap = title_canon2aliases.get(title, {}) or {}
        for c, als in tmap.items():
            if normalize_key_name(c) == kn:
                return c
            if any(normalize_key_name(a) == kn for a in (als or [])):
                return c
        g = alias2canon.get(kn)
        return g if g else alias

    for i, rec in enumerate(data):
        base = dict(rec)
        title = str(base.get("report_title", "")).strip()
        rec_drop = dict(base)

        # (A) 高频优先丢 key
        if drop_keys_enable:
            present_keys = [
                k
                for k, v in rec_drop.items()
                if k not in RESERVED
                and isinstance(k, str)
                and k.strip()
                and _one_line(v)
            ]
            bucket: Dict[str, List[str]] = {}
            for k in present_keys:
                c = to_canon(title, k)
                bucket.setdefault(c, []).append(k)

            if drop_keys_scope == "per_type":
                th = hf_thresh_by_title.get(title, math.inf)

                def _freq(c: str) -> int:
                    return freq_by_title.get(title, {}).get(c, 0)

            else:
                th = hf_thresh_global if hf_thresh_global is not None else math.inf

                def _freq(c: str) -> int:
                    return freq_global.get(c, 0)

            candidates = [c for c in bucket if _freq(c) >= th and _freq(c) > 0]

            def _has_alias(canon: str) -> bool:
                if title in title_canon2aliases and canon in (
                    title_canon2aliases[title] or {}
                ):
                    if title_canon2aliases[title].get(canon) or []:
                        return True
                return bool((canon2aliases.get(canon) or []))

            cand_no_alias = [c for c in candidates if not _has_alias(c)]
            cand_has_alias = [c for c in candidates if _has_alias(c)]
            rng.shuffle(cand_no_alias)
            rng.shuffle(cand_has_alias)

            drop_n = (
                math.ceil(len(candidates) * float(drop_keys_ratio)) if candidates else 0
            )
            picked: List[str] = []
            if drop_n > 0:
                take = min(drop_n, len(cand_no_alias))
                picked.extend(cand_no_alias[:take])
                if len(picked) < drop_n and cand_has_alias:
                    need_more = drop_n - len(picked)
                    picked.extend(cand_has_alias[: min(need_more, len(cand_has_alias))])

            drop_no_alias_total += len(set(picked) & set(cand_no_alias))
            drop_has_alias_total += len(set(picked) & set(cand_has_alias))
            for c in picked:
                for alias_k in bucket.get(c, []):
                    rec_drop.pop(alias_k, None)

        # (B) 结构外字段过滤（可选）
        if drop_unknown_keys:
            allowed_norm = {
                normalize_key_name(x) for x in (struct_map.get(title, []) or [])
            }
            for k in list(rec_drop.keys()):
                if k in RESERVED or not isinstance(k, str) or not k.strip():
                    continue
                if normalize_key_name(k) not in allowed_norm:
                    rec_drop.pop(k, None)
                    removed_unknown_counter[title] += 1
                    if len(dirty_preview) < int(preview_dirty_limit):
                        dirty_preview.append(
                            {
                                "index": i,
                                "title": title,
                                "key": k,
                                "reason": "not in structure",
                            }
                        )

        # 记录“输入别名覆盖”
        for k in list(rec_drop.keys()):
            if k in RESERVED or not isinstance(k, str) or not k.strip():
                continue
            canon = to_canon(title, k)
            if k != canon:
                aa = normalize_key_name(k)
                alias_seen_global_input.add(aa)
                alias_seen_by_title_input[title].add(aa)

        # 归一到 canonical，统计出现位置（供计划）
        fields = [
            (k, _one_line(v))
            for k, v in rec_drop.items()
            if k not in RESERVED and isinstance(k, str) and k.strip() and _one_line(v)
        ]
        fields_canon, _, _ = _canonicalize_fields_for_title(
            title, fields, title_canon2aliases, alias2canon
        )
        if fields_canon:
            for canon, _ in fields_canon:
                occ_by_title_canon[(title, canon)].append(i)
                canon_seen_global.add(normalize_key_name(canon))
                canon_seen_by_title[title].add(normalize_key_name(canon))

        prepped_records[i] = rec_drop  # 保存预处理样本

    # ========== 构建“强制显示名计划” ==========
    # plan[(idx, canon)] = display_name（优先让每个别名至少出现一次；若出现次数不足，则部分别名“不可覆盖”）
    planned_display: Dict[Tuple[int, str], str] = {}
    used_alias_global_display: Set[str] = set()
    used_alias_by_title_display: Dict[str, Set[str]] = defaultdict(set)

    # 可覆盖别名全集（仅统计：其 canonical 至少出现过一次的别名）
    coverable_alias_global: Set[str] = set()
    coverable_alias_by_title: Dict[str, Set[str]] = defaultdict(set)

    # 收集每组（title, canon）的别名列表：标题内优先 + 全局补充（去重）
    def alias_list_for(title: str, canon: str) -> List[str]:
        seen = set()
        out: List[str] = []
        tmap = title_canon2aliases.get(title, {}) or {}
        for a in list(tmap.get(canon, []) or []):
            if a not in seen and a != canon:
                out.append(a)
                seen.add(a)
        for a in list((canon2aliases.get(canon) or []) or []):
            if a not in seen and a != canon:
                out.append(a)
                seen.add(a)
        return out

    for (title, canon), idx_list in occ_by_title_canon.items():
        aliases = alias_list_for(title, canon)
        # 这组 canonical 在数据中出现过，因此其所有别名均“可覆盖”
        for a in aliases:
            aa = normalize_key_name(a)
            coverable_alias_global.add(aa)
            coverable_alias_by_title[title].add(aa)

        if not idx_list:
            continue
        idx_list_sorted = sorted(idx_list)
        if not aliases:
            # 无别名，全部用 canonical
            for j, ridx in enumerate(idx_list_sorted):
                planned_display[(ridx, canon)] = canon
            continue

        # 先保证每个别名至少出现一次
        for j, a in enumerate(aliases):
            if j < len(idx_list_sorted):
                ridx = idx_list_sorted[j]
                planned_display[(ridx, canon)] = a
                used_alias_global_display.add(normalize_key_name(a))
                used_alias_by_title_display[title].add(normalize_key_name(a))
        # 多余的出现，轮换别名（如需可改为穿插 canonical）
        if len(idx_list_sorted) > len(aliases):
            k = 0
            for ridx in idx_list_sorted[len(aliases) :]:
                a = aliases[k % len(aliases)]
                planned_display[(ridx, canon)] = a
                used_alias_global_display.add(normalize_key_name(a))
                used_alias_by_title_display[title].add(normalize_key_name(a))
                k += 1

    # ========== Pass2：按计划组装 & 统计 ==========
    out_drop: List[Dict[str, Any]] = []
    out_alias: List[Dict[str, Any]] = []

    total_values = 0
    missing_values = 0
    missing_by_title: Dict[str, int] = defaultdict(int)
    if value_loss_debug_path:
        Path(value_loss_debug_path).write_text("", encoding="utf-8")

    for i, rec_drop in enumerate(prepped_records):
        title = str(rec_drop.get("report_title", "")).strip()

        # 可选 LLM 排序（计划模式下一般不用）
        pre_fields: Optional[List[Tuple[str, str]]] = None
        if ai_pick_keys and llm is not None:
            present_keys2 = [
                k
                for k in rec_drop.keys()
                if isinstance(k, str) and k not in RESERVED and k.strip()
            ]
            picked_order = _llm_pick_keys_list(llm, title, present_keys2)
            if picked_order:
                pre_fields = [(k, _one_line(rec_drop.get(k, ""))) for k in picked_order]

        # 根据计划强制显示名
        plan_for_this: Dict[str, str] = {}
        # 需要把“输入键名 -> canonical”再跑一遍，才能知道本条有哪些 canonical
        fields_tmp = [
            (k, _one_line(v))
            for k, v in rec_drop.items()
            if k not in RESERVED and isinstance(k, str) and k.strip() and _one_line(v)
        ]
        fields_canon_tmp, _, _ = _canonicalize_fields_for_title(
            title, fields_tmp, title_canon2aliases, alias2canon
        )
        for canon, _ in fields_canon_tmp:
            disp = planned_display.get((i, canon))
            if disp:
                plan_for_this[canon] = disp

        # 组装
        (
            items,
            raw_fields,
            display_map,
            alias_sources,
            alias2canon_used,
            alias_display_used,
        ) = (
            _compose_kv_triplets_with_plan(
                rec_drop,
                struct_map,
                balancer,
                preordered_fields=pre_fields,
                order_by_struct=(pre_fields is None),
                drop_unknown_keys=drop_unknown_keys,
                title_canon2aliases=title_canon2aliases,
                alias2canon_global=alias2canon,
                planned_display_for_record=plan_for_this,
            )
        )
        paragraphs = _pack_paragraphs_with_seps(
            _render_items(items),
            length_threshold,
            choose_sep=balancer.choose_item_group_separator,
        )
        composed = _tidy_punct_spaces(
            "\n\n".join(paragraphs)
        )

        # —— 输入侧别名覆盖（已在 Pass1 统计，这里不重复） ——

        # —— 显示侧别名覆盖：取 display_map 中 alias != canonical 的显示名 ——
        for canon, disp in display_map.items():
            if disp and disp != canon:
                used_alias_global_display.add(normalize_key_name(disp))
                used_alias_by_title_display[title].add(normalize_key_name(disp))

        # 值保真
        report_norm = _tidy_punct_spaces(_one_line(composed))
        for _k, _v in raw_fields:
            vv = _tidy_punct_spaces(_one_line(_v))
            total_values += 1
            if vv and (vv not in report_norm):
                missing_values += 1
                missing_by_title[title] += 1
                if value_loss_debug_path:
                    with Path(value_loss_debug_path).open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "title": title,
                                    "lost": {"key": _k, "value": vv},
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        # 输出两份
        rec_out_drop = dict(rec_drop)
        rec_out_drop["report"] = composed
        out_drop.append(rec_out_drop)

        rec_out_alias = dict(rec_drop)
        rec_out_alias["report"] = composed
        meta = dict(rec_out_alias.get("meta") or {})
        if alias_display_used:
            meta["alias2canonical"] = dict(alias_display_used)
        else:
            meta.pop("alias2canonical", None)
        rec_out_alias["meta"] = meta
        out_alias.append(rec_out_alias)

    # —— “可覆盖/不可覆盖”统计（全局/按标题） ——
    # 可覆盖：其 canonical 至少出现过一次
    # 不可覆盖：其 canonical 从未出现（或被完全丢弃）——无处安放
    # 全局别名全集 alias_universe_global，按标题 alias_universe_by_title
    coverable_global = coverable_alias_global
    uncovered_coverable_global = coverable_global - used_alias_global_display

    # 对于“按标题”，先补上“本轮出现过的 canonical 的全局别名”
    def _global_aliases_for_canon(canon: str) -> List[str]:
        for c2, als in (canon2aliases or {}).items():
            if normalize_key_name(c2) == normalize_key_name(canon):
                return als or []
        return []

    for t, canons in canon_seen_by_title.items():
        for c in canons:
            for a in _global_aliases_for_canon(c):
                alias_universe_by_title[t].add(normalize_key_name(a))
                coverable_alias_by_title[t].add(normalize_key_name(a))

    # 计算“不可覆盖全集”= 全局别名全集 - 可覆盖全集
    not_coverable_global = alias_universe_global - coverable_global

    # ---- 抽样检测：别名是否真实出现在报告文本中 ----
    alias_check_total = 0
    alias_mismatch_total = 0
    alias_mismatch_samples: List[Dict[str, Any]] = []
    for idx, rec_alias in enumerate(out_alias):
        alias_map = (rec_alias.get("meta") or {}).get("alias2canonical") or {}
        if not alias_map:
            continue
        report_text = rec_alias.get("report", "") or ""
        for alias, canon in alias_map.items():
            alias_check_total += 1
            if alias and alias not in report_text:
                alias_mismatch_total += 1
                if len(alias_mismatch_samples) < 20:
                    alias_mismatch_samples.append(
                        {
                            "index": idx,
                            "title": rec_alias.get("report_title"),
                            "alias": alias,
                            "canonical": canon,
                        }
                    )

    if alias_mismatch_total:
        print(
            f"[WARN] alias/report mismatch: {alias_mismatch_total}/{alias_check_total} alias strings not found in composed text"
        )
        for sample in alias_mismatch_samples:
            print(
                f"    idx={sample['index']} title={sample['title']} alias='{sample['alias']}' -> canonical='{sample['canonical']}'"
            )
    else:
        print(
            f"[CHECK] alias/report consistency OK ({alias_check_total} alias strings verified)"
        )

    # ---- 写文件 ----
    # 确保写入不同的路径或变量，避免覆盖
    Path(out_path_drop_keys).write_text(
        json.dumps(out_drop, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 如果需要写入其他内容，使用不同的路径或变量
    Path(out_path_drop_keys.replace(".json", "_filtered.json")).write_text(
        json.dumps(filtered_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(out_path_key_alias).write_text(
        json.dumps(out_alias, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 汇总统计（中文键） ----
    present_values = total_values - missing_values
    miss_rate = (missing_values / total_values) if total_values else 0.0

    def _build_cov(total_set: Set[str], seen_set: Set[str]) -> Dict[str, Any]:
        tot = len(total_set)
        seen = len(seen_set)
        cov = (seen / tot) if tot > 0 else 1.0
        return {
            "别名总数": int(tot),
            "命中别名数": int(seen),
            "覆盖比例": round(cov, 6),
            "是否全覆盖": bool(seen >= tot),
        }

    # 全局（输入 & 显示）
    cov_global_input = _build_cov(alias_universe_global, alias_seen_global_input)
    cov_global_display = _build_cov(alias_universe_global, used_alias_global_display)

    # “可覆盖全局”的显示覆盖（真正的达标口径）
    cov_global_display_coverable = _build_cov(
        coverable_global, used_alias_global_display
    )

    # 按标题
    coverage_by_title_input: Dict[str, Dict[str, Any]] = {}
    coverage_by_title_display: Dict[str, Dict[str, Any]] = {}
    coverage_by_title_display_coverable: Dict[str, Dict[str, Any]] = {}
    all_titles = (
        set(alias_universe_by_title.keys())
        | set(used_alias_by_title_display.keys())
        | set(alias_seen_by_title_input.keys())
    )
    for t in sorted(all_titles):
        total_t = alias_universe_by_title.get(t, set())
        seen_in_t_input = alias_seen_by_title_input.get(t, set())
        seen_in_t_display = used_alias_by_title_display.get(t, set())
        coverable_t = coverable_alias_by_title.get(t, set())
        coverage_by_title_input[t] = _build_cov(total_t, seen_in_t_input)
        coverage_by_title_display[t] = _build_cov(total_t, seen_in_t_display)
        coverage_by_title_display_coverable[t] = _build_cov(
            coverable_t, seen_in_t_display
        )

    stats = {
        "total_records": len(data),
        "value_coverage": {
            "total_values": int(total_values),
            "present_values": int(present_values),
            "missing_values": int(missing_values),
            "missing_rate": round(miss_rate, 6),
            "missing_by_title": dict(missing_by_title),
        },
        "drop_keys": {
            "enabled": drop_keys_enable,
            "ratio": drop_keys_ratio,
            "quantile": drop_keys_quantile,
            "scope": drop_keys_scope,
            "dropped_no_alias": int(drop_no_alias_total),
            "dropped_has_alias": int(drop_has_alias_total),
        },
        "alias_coverage": {
            "全局_输入别名": cov_global_input,
            "全局_显示别名": cov_global_display,
            "全局_显示别名_可覆盖口径": cov_global_display_coverable,
            "不可覆盖_全局别名数": int(len(not_coverable_global)),
            "不可覆盖_全局别名样例": sorted(list(not_coverable_global))[:50],
            "按标题_输入别名": coverage_by_title_input,
            "按标题_显示别名": coverage_by_title_display,
            "按标题_显示别名_可覆盖口径": coverage_by_title_display_coverable,
        },
        "item_group_separators_usage": balancer.item_sep_usage,
        "kv_separators_usage": balancer.kv_sep_usage,
        "alias_report_consistency": {
            "total_checked": int(alias_check_total),
            "mismatched": int(alias_mismatch_total),
            "samples": alias_mismatch_samples,
        },
    }
    if stats_path:
        Path(stats_path).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"[ok] wrote: {out_path_drop_keys}")
    print(f"[ok] wrote: {out_path_key_alias}")


# ============================== main ==============================
if __name__ == "__main__":
    try:
        from conf import API_KEY
    except Exception:
        API_KEY = None

    # 配置参数
    TARGET_REPORT_TYPES = None  # 指定要处理的报告类型，None表示处理所有类型
    # MAX_SAMPLES_PER_TYPE 不需要在这里指定，数量控制在上游 run_da.py 中已完成

    # 可选 tokenizer
    try:
        tokenizer_name = DEFAULT_TOKENIZER_PATH
        tok = None
        if HFTokenizer is not None:
            tok = HFTokenizer.from_pretrained(tokenizer_name)  # type: ignore
        elif AutoTokenizer is not None:
            tok = AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore
        if tok is not None:
            set_tokenizer(tok)
            print("[tokenizer] loaded for token-based length estimation.")
        else:
            print("[tokenizer] not available, fallback to char length.")
    except Exception as e:
        print(f"[tokenizer] load failed, fallback to char length. err={e}")

    import os
    from datetime import datetime

    date_tag = os.environ.get("DA_DATE_TAG") or datetime.now().strftime("%Y%m%d")
    version_tag = os.environ.get("DA_OUTPUT_VERSION") or "v5_0"
    base_dir = Path("data") / date_tag
    base_dir.mkdir(parents=True, exist_ok=True)

    # ====== 路径 / 模型配置 - 第三阶段：报告组装 ======
    env_in = os.environ.get("DA_COMPOSE_IN")
    IN_PATH = env_in or str(base_dir / "clean_ocr_ppt_da_v5_0_field_cleaned.json")  # 来自 data_augmentation_recheck.py 的输出
    OUT_DROP = str(base_dir / f"clean_ocr_ppt_da_{version_tag}_report_drop_keys.json")  # 报告组装输出（丢key版）
    OUT_ALIAS = str(base_dir / f"clean_ocr_ppt_da_{version_tag}_report_key_alias.json")  # 报告组装输出（别名版）
    STATS_PATH = str(base_dir / f"clean_ocr_ppt_da_{version_tag}_report.stats.json")  # 统计信息

    USE_LLM_SORT = True
    llm = None

    LENGTH_THRESHOLD = 500

    compose_reports_dual_outputs(
        in_path=IN_PATH,
        out_path_drop_keys=OUT_DROP,
        out_path_key_alias=OUT_ALIAS,
        length_threshold=LENGTH_THRESHOLD,
        stats_path=STATS_PATH,
        preview_dirty_limit=80,
        drop_unknown_keys=False,
        ai_pick_keys=USE_LLM_SORT,
        llm=llm,
        value_loss_debug_path=None,
        drop_keys_enable=True,
        drop_keys_ratio=0.10,
        drop_keys_quantile=0.80,
        drop_keys_scope="per_type",
        target_report_types=TARGET_REPORT_TYPES,
        # max_samples_per_type 在上游已处理，这里不需要
    )
    print("[main] done.")
