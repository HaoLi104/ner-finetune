#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER 模型评估脚本

参考 pre_struct/evaluate.py，为 KV-NER 模型提供键值对级别的评估：
1. 读取评估数据（JSONL 格式，包含 spans）
2. 使用模型预测 NER 实体并组装成键值对
3. 使用 evaluation 库计算键值对匹配的 F1 指标
4. 生成评估报告和错误样本分析
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import unicodedata
import re
import string

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录到路径
if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import build_bio_label_list
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
    from pre_struct.kv_ner.chunking import predict_with_chunking
else:
    from . import config_io
    from .data_utils import build_bio_label_list
    from .modeling import BertCrfTokenClassifier
    from .chunking import predict_with_chunking

# 评测库
sys.path.append(str(_PACKAGE_ROOT if '_PACKAGE_ROOT' in locals() else Path.cwd()))
from evaluation.src.easy_eval import evaluate_entities  # type: ignore

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _expand_truncated_keys(
    entities: List[Dict[str, Any]], 
    text: str
) -> List[Dict[str, Any]]:
    """
    扩展被截断的 KEY 实体
    
    例如：如果模型只识别出"病理诊断"，但原文是"术后病理诊断"，
    则向前扩展以包含完整的 KEY。
    
    Args:
        entities: 实体列表
        text: 原文文本
    
    Returns:
        扩展后的实体列表
    """
    # 定义需要扩展的 KEY 模式：{被截断的: [可能的前缀]}
    key_patterns = {
        "病理诊断": ["术后", "术前", "穿刺", "冰冻"],
        "病理": ["术后", "术前"],
        "诊断": ["病理", "入院", "出院", "初步", "最终"],
        "检查": ["体格", "辅助", "实验室"],
        "治疗": ["术后", "术前", "放射", "化学"],
    }
    
    expanded = []
    for ent in entities:
        if ent["type"] != "KEY":
            expanded.append(ent)
            continue
        
        key_text = ent.get("text", "")
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        
        # 检查是否需要扩展
        if key_text not in key_patterns:
            expanded.append(ent)
            continue
        
        # 尝试向前扩展
        prefixes = key_patterns[key_text]
        extended = False
        
        for prefix in prefixes:
            # 检查前面是否有这个前缀
            prefix_start = start - len(prefix)
            if prefix_start >= 0:
                candidate = text[prefix_start:start]
                if candidate == prefix:
                    # 找到了前缀，扩展实体
                    new_text = text[prefix_start:end]
                    expanded.append({
                        "type": "KEY",
                        "start": prefix_start,
                        "end": end,
                        "text": new_text,
                    })
                    extended = True
                    break
        
        if not extended:
            expanded.append(ent)
    
    return expanded


def _merge_optional_dicts(
    *candidates: Tuple[str, Any]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for label, candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, dict):
            raise ValueError(f"{label} must be a dict, got {type(candidate).__name__}")
        merged.update(candidate)
    return merged


def _collect_tokenizer_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = _merge_optional_dicts(
        ("config.tokenizer_kwargs", cfg.get("tokenizer_kwargs")),
        ("config.tokenizer_load_kwargs", cfg.get("tokenizer_load_kwargs")),
    )
    train_block = cfg.get("train")
    if isinstance(train_block, dict):
        kwargs.update(
            _merge_optional_dicts(
                ("train.tokenizer_kwargs", train_block.get("tokenizer_kwargs")),
            )
        )
    return kwargs


def _build_tokenizer(
    name_or_path: str,
    extra_kwargs: Dict[str, Any],
) -> AutoTokenizer:
    load_kwargs = dict(extra_kwargs or {})
    use_fast = bool(load_kwargs.pop("use_fast", True))

    def _instantiate(use_fast_flag: bool) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=use_fast_flag,
            **load_kwargs,
        )

    try:
        tokenizer = _instantiate(use_fast)
    except ValueError as exc:
        if use_fast:
            logger.warning(
                "加载 fast tokenizer 失败，将尝试 use_fast=False：%s",
                exc,
            )
            tokenizer = _instantiate(False)
        else:
            raise

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer '{name_or_path}' 不是 fast 版本，无法提供 offset_mapping，"
            "KV-NER 训练/评估依赖 fast tokenizer，请更换模型或提供 fast 版本。"
        )
    return tokenizer


# =========================
# 工具函数
# =========================
def set_seed(seed: Optional[int]) -> None:
    """设置随机种子"""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def strip_trailing_punctuation(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """去除开头和末尾的标点符号，并调整 start/end 位置"""
    # 同时去除首尾空白与标点（不改动中间内容），以减少因OCR空格/标点导致的边界误差
    trailing_puncts = set(' \t\u3000\r\n。，、；：？！,.;:?![]【】()（）「」『』""\'\'…—')
    
    original_text = text
    
    # 从开头去除标点
    while text and text[0] in trailing_puncts:
        text = text[1:]
        start += 1
    
    # 从末尾去除标点
    while text and text[-1] in trailing_puncts:
        text = text[:-1]
        end -= 1
    
    # 如果全部被去除了，保留原始值
    if not text:
        return original_text, start, end
    
    return text, start, end


def _normalize_text_for_eval(s: str) -> str:
    """Normalize text to reduce spurious exact/overlap gaps.

    Operations:
    - Unicode NFKC to fold full-width -> half-width forms (：）（，；％ → :)(,;% …)
    - Unify various dash characters (—, –) to '-'
    - Collapse all whitespace (including ideographic space) to nothing
      (gold/pred often differ only by spaces; removing stabilizes exact match)
    - Trim leading/trailing punctuation again after normalization
    """
    if not s:
        return ""
    # Unicode normalization (full-width, compatibility forms)
    s = unicodedata.normalize("NFKC", s)
    # Unify dashes
    s = s.replace("—", "-").replace("–", "-")
    # Only strip leading/trailing whitespace (keep interior spaces)
    s = s.replace("\u3000", " ")  # ideographic space -> normal space
    s = re.sub(r"^\s+|\s+$", "", s)
    # Final trim of common edge punctuation
    # Note: we only trim at edges here; interior punctuation is kept
    edge_punct = "。，、；:;,:()[]{}<>"  # both Chinese and ASCII common marks after NFKC
    # left trim
    i = 0
    while i < len(s) and s[i] in edge_punct:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in edge_punct:
        j -= 1
    return s[i:j]


EDGE_STRIP_CHARS = set(string.whitespace) | set("。，、；：？！,.!?;:()（）[]【】{}<>「」『』""''\"'…—-_/\\") | {"\u3000"}

# 常见边界噪声字符：空格、冒号、数字、括号等
BOUNDARY_NOISE_CHARS = set(string.whitespace) | set("：:") | set(string.digits) | set("()（）[]【】") | {"\u3000"}

# 可容忍的结尾字符：标点符号等
TOLERANT_TRAILING_CHARS = set("。，、；：？！,.!?;:") | set(string.whitespace) | {"\u3000"}


def _strip_edge_chars_limited(text: str, max_remove: int = 3) -> Tuple[str, int]:
    if not text or max_remove <= 0:
        return text, 0
    left = 0
    right = len(text)
    removed = 0
    while left < right and removed < max_remove and text[left] in EDGE_STRIP_CHARS:
        left += 1
        removed += 1
    while left < right and removed < max_remove and text[right - 1] in EDGE_STRIP_CHARS:
        right -= 1
        removed += 1
    return text[left:right], removed


def _strip_boundary_noise(text: str) -> str:
    """完全去除边界的噪声字符（空格、冒号、数字、括号等）"""
    if not text:
        return ""
    left = 0
    right = len(text)
    while left < right and text[left] in BOUNDARY_NOISE_CHARS:
        left += 1
    while left < right and text[right - 1] in BOUNDARY_NOISE_CHARS:
        right -= 1
    return text[left:right]


def _strip_boundary_noise_with_offset(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """去除边界噪声字符，同时返回调整后的 start 和 end 位置"""
    if not text:
        return text, start, end
    
    original_text = text
    left = 0
    right = len(text)
    
    # 从左边去除噪声字符
    while left < right and text[left] in BOUNDARY_NOISE_CHARS:
        left += 1
    
    # 从右边去除噪声字符
    while left < right and text[right - 1] in BOUNDARY_NOISE_CHARS:
        right -= 1
    
    # 调整 start 和 end
    new_start = start + left
    new_end = start + right
    new_text = text[left:right]
    
    return new_text, new_start, new_end


def _normalize_key_name(s: str) -> str:
    """Normalize key names for alignment between gold/pred.

    - Unicode NFKC
    - Remove all whitespace
    - Strip common trailing punctuation like ':' '：' '、' '。' '；'
    - Do not change interior text beyond NFKC/whitespace removal
    """
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"\s+", "", s)
    # drop trailing separators repeatedly
    while s and s[-1] in ":：;；、。，,.;":
        s = s[:-1]
    return s


def _accumulate_char_freq(freq: Dict[str, int], s: str) -> None:
    for ch in s:
        if not ch:
            continue
        freq[ch] = freq.get(ch, 0) + 1


def _top_k_items(d: Dict[Any, int], k: int = 20) -> Dict[str, int]:
    return {str(k_): int(v) for k_, v in sorted(d.items(), key=lambda x: -x[1])[:k]}


def _char_to_token_index(offsets: List[Tuple[int, int]], pos: int, right: bool = False) -> int:
    """Map character position to token index using offset mapping.
    If right=True, map to the token whose end equals pos (or nearest to the left)."""
    if pos <= 0:
        return 0
    n = len(offsets)
    for i, (s, e) in enumerate(offsets):
        if s <= pos < e:
            return i
    # if not inside any token span, find nearest token boundary
    if right:
        for i in range(n - 1, -1, -1):
            if offsets[i][1] <= pos:
                return i
        return 0
    else:
        for i in range(n):
            if offsets[i][0] >= pos:
                return i
        return n - 1


def _is_valid_span(v: dict) -> bool:
    """有效 span：start>=0, end>start, text 非空。"""
    try:
        s = int(v.get("start", -1))
        e = int(v.get("end", -1))
        t = str(v.get("text", "")).strip()
        return (s >= 0) and (e > s) and bool(t)
    except Exception:
        return False


def _to_eval_pack(spans_or_preds: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    将 {key: {start,end,text}} 打包为:
        {"entities": [{"start": s, "end": e, "text": t}, ...]}
    仅收集"有效 span"。在计算position时会去除末尾的标点符号。
    
    Args:
        spans_or_preds: 键值对字典
        exclude_keys: 要排除的key列表（如["无键名"]）
    """
    exclude_keys = exclude_keys or []
    ents: List[Dict[str, Any]] = []
    
    for k, v in (spans_or_preds or {}).items():
        # 跳过排除的key
        if k in exclude_keys:
            continue
        
        if not isinstance(v, dict):
            continue
        if not _is_valid_span(v):
            continue
        s = int(v["start"])
        e = int(v["end"])
        t = str(v.get("text", "")).strip()
        
        # 去除末尾标点符号，并调整位置
        t_clean, s_clean, e_clean = strip_trailing_punctuation(t, s, e)
        # 对文本做统一的评估规范化，缓解标点/空格/全角差异
        t_norm = _normalize_text_for_eval(t_clean)
        ents.append({"start": s_clean, "end": e_clean, "text": t_norm})
    return {"entities": ents}


# =========================
# 预测和转换
# =========================
def predict_entities(
    model: BertCrfTokenClassifier,
    tokenizer: AutoTokenizer,
    text: str,
    id2label: Dict[int, str],
    device: torch.device,
    max_seq_length: int = 512,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
    merge_adjacent_gap: int = 2,
    adjust_boundaries: bool = False,
    adjust_max_shift: int = 1,
    adjust_chars: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    对单个样本进行预测，返回 NER 实体列表（支持分块处理）
    
    Returns:
        [{"type": "KEY", "start": 0, "end": 5, "text": "姓名"}, ...]
    """
    ents = predict_with_chunking(
        text=text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_adjacent_gap=merge_adjacent_gap,
    )
    if adjust_boundaries and ents:
        chars = set(adjust_chars or "")
        L = len(text)
        adjusted: List[Dict[str, Any]] = []
        for ent in ents:
            # 仅对 VALUE（和可选其他非KEY）做边界微调，避免键名被扩展进冒号/空格导致键对齐失败
            if str(ent.get("type")) == "KEY":
                adjusted.append(ent)
                continue
            s = int(ent.get("start", -1))
            e = int(ent.get("end", -1))
            if 0 <= s < e <= L:
                # expand left within max_shift if previous char is in chars
                for _ in range(int(adjust_max_shift)):
                    if s > 0 and text[s-1] in chars:
                        s -= 1
                    else:
                        break
                # expand right within max_shift if next char is in chars
                for _ in range(int(adjust_max_shift)):
                    if e < L and text[e] in chars:
                        e += 1
                    else:
                        break
                ent = dict(ent)
                ent["start"] = s
                ent["end"] = e
                ent["text"] = text[s:e].strip()
            adjusted.append(ent)
        return adjusted
    return ents


def assemble_key_value_pairs(
    entities: List[Dict[str, Any]],
    text: str,
    *,
    value_attach_window: int = 50,
    value_same_line_only: bool = True,
    value_crossline_fallback_len: int = 0,
    # 允许 KEY 与 VALUE 之间存在少量噪声字符
    bridge_noise_max_chars: int = 0,
    bridge_noise_chars: Optional[str] = None,
    # 保留没有 KEY 的 VALUE
    keep_orphan_values: bool = False,
    orphan_key_name: str = "无键名",
) -> Dict[str, Dict[str, Any]]:
    """
    将 NER 实体组装成键值对（参考 predict.py 的逻辑）
    
    Args:
        entities: NER 实体列表（包含 KEY, VALUE, HOSPITAL）
        text: 原文本
    
    Returns:
        {key_text: {start, end, text}} 格式的字典
        注意：
        - 即使没有VALUE，KEY也会输出（text为空）
        - HOSPITAL实体会以"医院名称"为key输出
    """
    # 分离 KEY、VALUE 和 HOSPITAL
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    seq.sort(key=lambda x: (x["start"], x["end"]))
    
    # 单独处理HOSPITAL实体
    hospitals = [e for e in entities if e["type"] == "HOSPITAL"]
    
    pairs: List[Tuple[Dict, List[Dict]]] = []  # [(key, [values])]
    pending: Optional[Tuple[Dict, List[Dict]]] = None
    # 默认允许的噪声字符集合（空白/标点/数字/括号/破折号等）
    default_bridge_chars = " \t\u3000:：,，.;；。()（）[]【】{}<>%％-—–/_\\0123456789"
    bridge_set = set(str(bridge_noise_chars) if bridge_noise_chars is not None else default_bridge_chars)
    bridge_limit = max(0, int(bridge_noise_max_chars))
    
    for idx, ent in enumerate(seq):
        if ent["type"] == "KEY":
            if pending:
                pairs.append(pending)
            pending = (ent, [])
        elif ent["type"] == "VALUE":
            if pending:
                pending[1].append(ent)
            else:
                # heuristic attach to nearest previous KEY under window and line constraint
                attached = False
                for j in range(idx - 1, -1, -1):
                    prev = seq[j]
                    if prev["type"] != "KEY":
                        continue
                    gap = ent["start"] - prev["end"]
                    if gap <= value_attach_window:
                        middle = text[prev["end"]:ent["start"]]
                        if value_same_line_only and "\n" in middle and len(ent.get("text", "")) > int(value_crossline_fallback_len):
                            # 特例：允许少量噪声跨行
                            if not (bridge_limit > 0 and len(middle) <= bridge_limit and set(middle).issubset(bridge_set)):
                                continue
                        pairs.append((prev, [ent]))
                        attached = True
                        break
                if not attached:
                    if keep_orphan_values:
                        # 保留无键名 VALUE，作为一个键值对输出
                        key_stub = {"type": "KEY", "start": ent.get("start", 0), "end": ent.get("start", 0), "text": orphan_key_name}
                        pairs.append((key_stub, [ent]))
                    else:
                        # drop unattached VALUE for KV evaluation; they are still in entities view
                        pass
            
    if pending:
        pairs.append(pending)
    
    # 转换为 {key_text: {start, end, text}} 格式
    result: Dict[str, Dict[str, Any]] = {}
    # tolerant keys (from env; used for sentence expansion)
    tol_keys_env = os.environ.get("KVNER_TOL_KEYS")
    tol_set = {"现病史", "体格检查", "病理诊断", "治疗计划", "处理", "注意事项", "既往史", "个人史", "婚育史", "辅助检查"}
    if tol_keys_env:
        for k in tol_keys_env.split(","):
            k = k.strip()
            if k:
                tol_set.add(k)
    tolerant_keys_norm = {_normalize_key_name(k) for k in tol_set}
    
    # 收集所有键值对（先不添加到result，用于排序）
    all_kvs = []  # [(key_text, start, end, text, is_hospital)]
    
    # 处理KEY-VALUE对
    for key_ent, value_ents in pairs:
        key_text = key_ent["text"]
        if not key_text:
            continue
        
        # 如果有VALUE，使用全文的连续切片
        if value_ents:
            first_val = value_ents[0]
            last_val = value_ents[-1]
            span_start = int(first_val["start"]) if isinstance(first_val.get("start"), int) else first_val["start"]
            span_end = int(last_val["end"]) if isinstance(last_val.get("end"), int) else last_val["end"]
            span_start = max(0, min(len(text), span_start))
            span_end = max(span_start, min(len(text), span_end))
            # 针对长文本键（如现病史等）尝试将末端扩展到句末，最多扩展 N 个字符
            key_norm = _normalize_key_name(key_text)
            expand_to_sentence = key_norm in tolerant_keys_norm and bool(int(os.environ.get("KVNER_EXPAND_SENTENCE", "0")))
            if expand_to_sentence:
                stopset = set("。；;.!？！?\n")
                limit = int(os.environ.get("KVNER_EXPAND_MAX", "120"))
                i = span_end
                L = len(text)
                steps = 0
                while i < L and steps < limit:
                    ch = text[i]
                    i += 1
                    steps += 1
                    if ch in stopset:
                        span_end = i
                        break
            # 用原文切片构造值文本，更贴近标注连续片段
            slice_text = text[span_start:span_end]
            clean_text, clean_start, clean_end = strip_trailing_punctuation(
                slice_text, span_start, span_end
            )
            all_kvs.append((
                key_text,
                clean_start,
                clean_end,
                clean_text,
                False,  # 不是hospital
                key_ent["start"],  # 用于排序
            ))
        else:
            # 没有VALUE的KEY也输出（VALUE为空）
            all_kvs.append((
                key_text,
                key_ent["end"],
                key_ent["end"],
                "",
                False,
                key_ent["start"],  # 用于排序
            ))
    
    # 处理HOSPITAL实体
    for hospital_ent in hospitals:
        hospital_text = hospital_ent.get("text", "").strip()
        if hospital_text:
            all_kvs.append((
                "医院名称",
                hospital_ent["start"],
                hospital_ent["end"],
                hospital_text,
                True,  # 是hospital
                hospital_ent["start"],  # 用于排序
            ))
    
    # 排序：医院名称在最前，其他按start位置排序
    all_kvs.sort(key=lambda x: (0 if x[4] else 1, x[5]))  # (is_hospital的反向, start)
    
    # 按顺序构建result（Python 3.7+字典保持插入顺序）
    result = {}
    orphan_count = 0
    for key_text, start, end, text_content, _, _ in all_kvs:
        # 对无键名条目编号，避免覆盖
        if key_text == orphan_key_name and key_text in result:
            orphan_count += 1
            key_text = f"{orphan_key_name}_{orphan_count}"
        result[key_text] = {
            "start": start,
            "end": end,
            "text": text_content,
        }
    
    return result


# =========================
# 评估主流程
# =========================
def evaluate_dataset(
    model: BertCrfTokenClassifier,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, Any]],
    id2label: Dict[int, str],
    device: torch.device,
    max_seq_length: int = 512,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
    merge_adjacent_gap: int = 2,
    error_dump_path: Optional[str] = None,
    error_threshold: float = 0.99,
    align_mode: str = "gold",
    exclude_keys: Optional[List[str]] = None,
    report_title_filter: Optional[List[str]] = None,
    value_attach_window: int = 50,
    value_same_line_only: bool = True,
    adjust_boundaries: bool = False,
    adjust_max_shift: int = 1,
    adjust_chars: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    评估整个数据集（键值对级别）
    
    Args:
        test_data: JSONL 格式的测试数据，每条包含 report, spans
        align_mode: 'gold' | 'pred' | 'union'
        exclude_keys: 排除的键列表
        report_title_filter: 报告类型过滤列表（如 ["入院记录", "门诊病历"]）
    
    Returns:
        {
            "position": {...},
            "text_exact": {...},
            "text_overlap": {...},
            "num_samples": n,
        }
    """
    totals_txt_exact: Dict[str, float] = {}
    totals_txt_overlap: Dict[str, float] = {}
    totals_txt_tol: Dict[str, float] = {}
    totals_txt_tolerant: Dict[str, float] = {}
    totals_txt_edge3: Dict[str, float] = {}
    counted_reports = 0
    error_samples = []
    
    if report_title_filter:
        logger.info(f"报告类型过滤: {report_title_filter}")
    else:
        logger.info("报告类型: 所有类型")
    
    logger.info(f"开始评估 {len(test_data)} 个样本（键值对级别）...")
    # 统计边界偏差与差异字符（帮助定位“text exact vs overlap 差距”来源）
    start_delta_hist: Dict[int, int] = {}
    end_delta_hist: Dict[int, int] = {}
    leading_extra_chars: Dict[str, int] = {}
    trailing_extra_chars: Dict[str, int] = {}
    # Key-level mismatch diagnostics
    key_total: Dict[str, int] = defaultdict(int)
    key_miss: Dict[str, int] = defaultdict(int)
    key_substr: Dict[str, int] = defaultdict(int)
    key_edge1: Dict[str, int] = defaultdict(int)
    key_other: Dict[str, int] = defaultdict(int)

    # Limit number of samples if requested (useful for quick experiments)
    iter_data = test_data if not max_samples else test_data[: int(max_samples)]
    # tolerant key settings
    default_tolerant_keys = {
        "现病史", "体格检查", "病理诊断", "治疗计划", "处理", "注意事项",
        "既往史", "个人史", "婚育史", "辅助检查",
    }
    env_tol = os.environ.get("KVNER_TOL_KEYS")
    if env_tol:
        for k in env_tol.split(","):
            k = k.strip()
            if k:
                default_tolerant_keys.add(k)
    tolerant_keys_norm = {_normalize_key_name(k) for k in default_tolerant_keys}
    tol_cov = float(os.environ.get("KVNER_TEXT_COVERAGE", "0.85"))

    def _covered_equal(a: str, b: str) -> bool:
        # normalize and compare with coverage tolerance
        na, nb = _normalize_text_for_eval(a), _normalize_text_for_eval(b)
        if na == nb:
            return True
        la, lb = len(na), len(nb)
        if la == 0 or lb == 0:
            return False
        # substring coverage
        if na in nb or nb in na:
            cov = min(la, lb) / max(la, lb)
            return cov >= tol_cov
        return False

    for item in tqdm(iter_data, desc="评估进度"):
        text = str(item.get("report", "") or "")
        if not text.strip():
            continue
        
        report_title = str(item.get("report_title", "") or "")
        
        # 报告类型过滤
        if report_title_filter and report_title not in report_title_filter:
            continue
        gold_raw = item.get("spans", {}) or {}
        
        # 预测 NER 实体（使用分块处理）
        entities = predict_entities(
            model, tokenizer, text, id2label, device, max_seq_length,
            chunk_size, chunk_overlap,
            merge_adjacent_gap=merge_adjacent_gap,
            adjust_boundaries=adjust_boundaries,
            adjust_max_shift=adjust_max_shift,
            adjust_chars=adjust_chars,
        )
        
        # 组装成键值对
        pred_pairs = assemble_key_value_pairs(
            entities, text,
            value_attach_window=value_attach_window,
            value_same_line_only=value_same_line_only,
            value_crossline_fallback_len=int(os.environ.get('KVNER_CROSSL_FALLBACK_LEN', '2')),
            bridge_noise_max_chars=int(os.environ.get('KVNER_BRIDGE_MAX', str(int(os.environ.get('KVNER_CROSSL_FALLBACK_LEN', '2'))))),
            bridge_noise_chars=os.environ.get('KVNER_BRIDGE_CHARS'),
            keep_orphan_values=bool(int(os.environ.get('KVNER_KEEP_ORPHAN_VALUES', '1'))),
            orphan_key_name=os.environ.get('KVNER_ORPHAN_KEY_NAME', '无键名'),
        )
        
        # 只保留有效 span
        gold_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": str(v.get("text", "")).strip(),
            }
            for k, v in gold_raw.items()
            if isinstance(v, dict) and _is_valid_span(v)
        }
        pred_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": str(v.get("text", "")).strip(),
            }
            for k, v in pred_pairs.items()
            if isinstance(v, dict) and _is_valid_span(v)
        }
        
        # 对齐键（对 key 做轻量归一，避免“姓名/姓名：/ 姓名 ：”导致对齐失败）
        am = str(align_mode or "gold").lower()
        gold_norm: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for k, v in gold_valid.items():
            nk = _normalize_key_name(k)
            gold_norm.setdefault(nk, (k, v))
        pred_norm: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for k, v in pred_valid.items():
            nk = _normalize_key_name(k)
            pred_norm.setdefault(nk, (k, v))

        if am == "gold":
            ref_keys = set(gold_norm.keys())
        elif am == "pred":
            ref_keys = set(pred_norm.keys())
        else:
            ref_keys = set(gold_norm.keys()) | set(pred_norm.keys())

        # 回到原始 key 以保留位置信息/文本，仅使用规范化键作对齐桥梁
        true_map = {nk: gold_norm[nk][1] for nk in ref_keys if nk in gold_norm}
        pred_map = {nk: pred_norm[nk][1] for nk in ref_keys if nk in pred_norm}
        
        data_true = _to_eval_pack(true_map, exclude_keys=exclude_keys)
        data_pred = _to_eval_pack(pred_map, exclude_keys=exclude_keys)
        
        # 两边都没有实体 -> 跳过该报告
        if not data_true["entities"] and not data_pred["entities"]:
            continue
        
        # 评测 1：文本严格匹配（exact）
        res_txt_exact = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="exact",
        )
        
        # 评测 2：文本重叠（overlap）
        res_txt_overlap = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="overlap",
        )

        # 评测 3：Text Exact within K tokens（默认 K=3）
        # 使用同一 tokenizer 的 offsets 将 char 位置映射到 token 位置
        try:
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoding.get('offset_mapping')
            if offsets:
                tp = fp = fn = 0
                for nk in ref_keys:
                    pred_has = nk in pred_map
                    gold_has = nk in true_map
                    if pred_has and gold_has:
                        gs, ge = true_map[nk]['start'], true_map[nk]['end']
                        ps, pe = pred_map[nk]['start'], pred_map[nk]['end']
                        gs_tok = _char_to_token_index(offsets, gs)
                        ge_tok = _char_to_token_index(offsets, ge, right=True)
                        ps_tok = _char_to_token_index(offsets, ps)
                        pe_tok = _char_to_token_index(offsets, pe, right=True)
                        delta = max(abs(ps_tok - gs_tok), abs(pe_tok - ge_tok))
                        if delta <= int(os.environ.get('KVNER_TOL_TOKENS', '3')):
                            tp += 1
                        else:
                            fp += 1
                            fn += 1
                    elif pred_has and not gold_has:
                        fp += 1
                    elif (not pred_has) and gold_has:
                        fn += 1
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                for k,v in {"precision":prec,"recall":rec,"f1_score":f1}.items():
                    totals_txt_tol[k] = totals_txt_tol.get(k, 0.0) + float(v)
        except Exception:
            pass

        # 评测 4：Text Exact (tolerant by key coverage)
        try:
            tp = fp = fn = 0
            for nk in ref_keys:
                pred_has = nk in pred_map
                gold_has = nk in true_map
                if pred_has and gold_has:
                    gtxt = str(true_map[nk].get("text", ""))
                    ptxt = str(pred_map[nk].get("text", ""))
                    if nk in tolerant_keys_norm:
                        ok = _covered_equal(gtxt, ptxt)
                    else:
                        ok = _normalize_text_for_eval(gtxt) == _normalize_text_for_eval(ptxt)
                    if ok:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_has and not gold_has:
                    fp += 1
                elif (not pred_has) and gold_has:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            for k, v in {"precision": prec, "recall": rec, "f1_score": f1}.items():
                totals_txt_tolerant[k] = totals_txt_tolerant.get(k, 0.0) + float(v)
        except Exception:
            pass

        # 评测 5：Text Exact with ≤3 edge chars difference
        try:
            tp = fp = fn = 0
            for nk in ref_keys:
                pred_has = nk in pred_map
                gold_has = nk in true_map
                if pred_has and gold_has:
                    gtxt = str(true_map[nk].get("text", ""))
                    ptxt = str(pred_map[nk].get("text", ""))
                    g_trim, _ = _strip_edge_chars_limited(gtxt, 3)
                    p_trim, _ = _strip_edge_chars_limited(ptxt, 3)
                    if _normalize_text_for_eval(g_trim) == _normalize_text_for_eval(p_trim):
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_has and not gold_has:
                    fp += 1
                elif (not pred_has) and gold_has:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            for k, v in {"precision": prec, "recall": rec, "f1_score": f1}.items():
                totals_txt_edge3[k] = totals_txt_edge3.get(k, 0.0) + float(v)
        except Exception:
            pass

        # 边界偏差统计（仅对齐到共同出现的键，忽略噪声字符和可容忍的差异）
        inter_keys = set(true_map.keys()) & set(pred_map.keys())
        for k in inter_keys:
            gs, ge = int(true_map[k]["start"]), int(true_map[k]["end"])
            ps, pe = int(pred_map[k]["start"]), int(pred_map[k]["end"])
            
            # 清理边界噪声后再计算差异
            try:
                gtxt = str(true_map[k].get("text", ""))
                ptxt = str(pred_map[k].get("text", ""))
                
                # 清理后再比较位置
                gtxt_clean, gs_clean, ge_clean = _strip_boundary_noise_with_offset(gtxt, gs, ge)
                ptxt_clean, ps_clean, pe_clean = _strip_boundary_noise_with_offset(ptxt, ps, pe)
                
                # 只有清理后仍有差异才统计
                if gtxt_clean and ptxt_clean:
                    sd = ps_clean - gs_clean
                    ed = pe_clean - ge_clean
                    
                    # 检查是否是可容忍的差异（±1字符，且该字符是标点符号）
                    is_tolerant_start = False
                    is_tolerant_end = False
                    
                    # 检查开头差异是否可容忍
                    if abs(sd) == 1:
                        try:
                            if sd > 0 and gs_clean < len(text):
                                # pred延后1个字符开始（跳过了1个字符）
                                # 检查被跳过的字符是否是可容忍的标点
                                if text[gs_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_start = True
                            elif sd < 0 and ps_clean < len(text):
                                # pred提前1个字符开始（多包含了1个字符）
                                # 检查多包含的字符是否是可容忍的标点
                                if text[ps_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_start = True
                        except Exception:
                            pass
                    
                    # 检查结尾差异是否可容忍
                    if abs(ed) == 1:
                        try:
                            if ed < 0 and pe_clean < len(text):
                                # pred提前1个字符结束（截掉了1个字符）
                                # 检查被截掉的字符是否是可容忍的标点
                                if text[pe_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_end = True
                            elif ed > 0 and ge_clean < len(text):
                                # pred延后1个字符结束（多包含了1个字符）
                                # 检查多包含的字符是否是可容忍的标点
                                if text[ge_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_end = True
                        except Exception:
                            pass
                    
                    # 如果不是可容忍的差异，才统计
                    should_count_start = (sd != 0 and not is_tolerant_start)
                    should_count_end = (ed != 0 and not is_tolerant_end)
                    
                    if should_count_start or should_count_end:
                        # 统计位置差异（使用原始差异值）
                        if should_count_start:
                            start_delta_hist[sd] = start_delta_hist.get(sd, 0) + 1
                        if should_count_end:
                            end_delta_hist[ed] = end_delta_hist.get(ed, 0) + 1
                        
                        # 统计清理后仍存在的边界差异字符
                        if should_count_start:
                            if sd < 0:
                                diff_text = text[ps_clean:gs_clean] if ps_clean < gs_clean else ""
                                _accumulate_char_freq(leading_extra_chars, diff_text)
                            elif sd > 0:
                                diff_text = text[gs_clean:ps_clean] if gs_clean < ps_clean else ""
                                _accumulate_char_freq(leading_extra_chars, diff_text)
                        
                        if should_count_end:
                            if ed > 0:
                                diff_text = text[ge_clean:pe_clean] if ge_clean < pe_clean else ""
                                _accumulate_char_freq(trailing_extra_chars, diff_text)
                            elif ed < 0:
                                diff_text = text[pe_clean:ge_clean] if pe_clean < ge_clean else ""
                                _accumulate_char_freq(trailing_extra_chars, diff_text)
            except Exception:
                pass

        # 键级别不一致统计（帮助定位 long-text 子串导致的差距）
        all_keys = set(true_map.keys()) | set(pred_map.keys())
        for nk in all_keys:
            key_total[nk] += 1
            gt = _normalize_text_for_eval(str(true_map.get(nk, {}).get("text", ""))) if nk in true_map else ""
            pt = _normalize_text_for_eval(str(pred_map.get(nk, {}).get("text", ""))) if nk in pred_map else ""
            if not gt or not pt:
                key_miss[nk] += 1
                continue
            if gt == pt:
                continue
            if gt in pt or pt in gt:
                key_substr[nk] += 1
            elif (gt[:-1] == pt or pt[:-1] == gt or (len(gt) > 1 and gt[1:] == pt) or (len(pt) > 1 and pt[1:] == gt)):
                key_edge1[nk] += 1
            else:
                key_other[nk] += 1
        
        # 错误样本记录（添加 key-value 详细信息）
        if error_dump_path and res_txt_exact.get("f1_score", 1.0) < error_threshold:
            true_entities = data_true.get("entities", [])
            pred_entities = data_pred.get("entities", [])
            
            true_full_map = {(e["start"], e["end"], e["text"]): e for e in true_entities}
            pred_full_map = {(e["start"], e["end"], e["text"]): e for e in pred_entities}
            
            error_true_ents = []
            error_pred_ents = []
            matched_count = 0
            
            all_full_keys = set(true_full_map.keys()) | set(pred_full_map.keys())
            
            for key in all_full_keys:
                true_ent = true_full_map.get(key)
                pred_ent = pred_full_map.get(key)
                
                if true_ent is not None and pred_ent is not None:
                    matched_count += 1
                    continue
                
                if true_ent is not None:
                    error_true_ents.append(true_ent)
                if pred_ent is not None:
                    error_pred_ents.append(pred_ent)
            
            if error_true_ents or error_pred_ents:
                # 构建 key-value 映射（用于错误分析）
                true_kv_map = {}
                for nk in ref_keys:
                    if nk in true_map:
                        v = true_map[nk]
                        true_kv_map[nk] = {
                            "start": v["start"],
                            "end": v["end"],
                            "text": v["text"]
                        }
                
                pred_kv_map = {}
                for nk in ref_keys:
                    if nk in pred_map:
                        v = pred_map[nk]
                        pred_kv_map[nk] = {
                            "start": v["start"],
                            "end": v["end"],
                            "text": v["text"]
                        }
                
                error_samples.append({
                    "metrics": res_txt_exact,
                    "ground_truth": {
                        "entities": error_true_ents,
                        "key_value_pairs": true_kv_map  # 添加键值对信息
                    },
                    "predict": {
                        "entities": error_pred_ents,
                        "key_value_pairs": pred_kv_map  # 添加键值对信息
                    },
                    "report_title": report_title,
                    "report": text,
                    "total_true": len(true_entities),
                    "total_pred": len(pred_entities),
                    "matched": matched_count,
                    "error_count": len(error_true_ents) + len(error_pred_ents),
                    "all_keys": list(ref_keys),  # 添加所有涉及的键
                })
        
        # 累加
        for k, v in res_txt_exact.items():
            if isinstance(v, (int, float)):
                totals_txt_exact[k] = totals_txt_exact.get(k, 0.0) + float(v)
        for k, v in res_txt_overlap.items():
            if isinstance(v, (int, float)):
                totals_txt_overlap[k] = totals_txt_overlap.get(k, 0.0) + float(v)
        
        counted_reports += 1
    
    # 求平均
    def _avg_inplace(d: Dict[str, float], n: int):
        if n <= 0:
            return
        for m in ("precision", "recall", "f1_score"):
            if m in d:
                d[m] = d[m] / n
    
    _avg_inplace(totals_txt_exact, counted_reports)
    _avg_inplace(totals_txt_overlap, counted_reports)
    _avg_inplace(totals_txt_tol, counted_reports)
    _avg_inplace(totals_txt_tolerant, counted_reports)
    _avg_inplace(totals_txt_edge3, counted_reports)
    
    # 保存错误样本
    if error_dump_path and error_samples:
        error_path = Path(error_dump_path)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", encoding="utf-8") as f:
            for err in error_samples:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
        logger.info(f"错误样本已保存到: {error_dump_path} ({len(error_samples)} 条)")
    
    logger.info(
        f"[INFO] 计入平均的样本数 = {counted_reports}\n"
        f" - text (exact)          : {totals_txt_exact}\n"
        f" - text (overlap)        : {totals_txt_overlap}\n"
        f" - text exact in ≤K tok.: {totals_txt_tol}\n"
        f" - text exact edge≤3     : {totals_txt_edge3}\n"
        f"注意：boundary_stats 已过滤：\n"
        f"  1. 边界噪声字符（空格、冒号、数字、括号）\n"
        f"  2. 可容忍的±1字符差异（标点符号）"
    )
    
    # 边界偏差与差异字符 Top 统计
    boundary_stats = {
        "start_delta_top": _top_k_items({int(k): int(v) for k, v in start_delta_hist.items()}),
        "end_delta_top": _top_k_items({int(k): int(v) for k, v in end_delta_hist.items()}),
        "leading_diff_top_chars": _top_k_items(leading_extra_chars),
        "trailing_diff_top_chars": _top_k_items(trailing_extra_chars),
    }
    logger.info("\nBoundary delta (start->pred-start, end->pred-end):")
    logger.info(f"  start_delta_top: {boundary_stats['start_delta_top']}")
    logger.info(f"  end_delta_top  : {boundary_stats['end_delta_top']}")
    logger.info("  leading_diff_top_chars: %s", boundary_stats['leading_diff_top_chars'])
    logger.info("  trailing_diff_top_chars: %s", boundary_stats['trailing_diff_top_chars'])

    # 输出 key 级诊断 Top10（按不一致率排序，仅日志展示）
    try:
        items = []
        for nk in key_total:
            t = key_total[nk]
            m = key_miss[nk] + key_substr[nk] + key_edge1[nk] + key_other[nk]
            if t >= 50:  # 只展示出现次数较多的键
                rate = m / t if t else 0.0
                items.append((rate, nk, t, key_substr[nk], key_edge1[nk], key_other[nk], key_miss[nk]))
        items.sort(reverse=True)
        logger.info("Top key mismatches (rate,total,substr,±1,other,missing):")
        for rate, nk, t, ss, e1, ot, ms in items[:10]:
            logger.info("  %s  rate=%.2f%% total=%d substr=%d ±1=%d other=%d miss=%d", nk, rate*100, t, ss, e1, ot, ms)
    except Exception:
        pass
    
    return {
        "text_exact": totals_txt_exact,
        "text_overlap": totals_txt_overlap,
        "text_exact_in_k": totals_txt_tol,
        "text_exact_tolerant": totals_txt_tolerant,
        "text_exact_edge3": totals_txt_edge3,
        "num_samples": counted_reports,
        "num_errors": len(error_samples),
        "boundary_stats": boundary_stats,
        "key_mismatch_stats": {
            "total": key_total,
            "substr": key_substr,
            "edge1": key_edge1,
            "other": key_other,
            "missing": key_miss,
        },
    }


# =========================
# 主评估流程
# =========================
def run_evaluation(
    config_path: str,
    model_dir: Optional[str] = None,
    test_data_path: Optional[str] = None,
    output_dir: str = "data/kv_ner_eval",
    seed: Optional[int] = 42,
    error_threshold: float = 0.99,
    align_mode: str = "gold",
    exclude_keys: Optional[List[str]] = None,
    report_title_filter: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    override_chunk_size: Optional[int] = None,
    override_chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    运行完整的评估流程
    
    Args:
        config_path: 配置文件路径
        model_dir: 模型目录（默认从配置读取）
        test_data_path: 测试数据路径（JSONL 格式，默认从配置读取）
        output_dir: 输出目录
        seed: 随机种子
        error_threshold: 错误样本 F1 阈值
        align_mode: 键对齐模式 'gold' | 'pred' | 'union'
        exclude_keys: 排除的键列表
    
    Returns:
        评估结果字典
    """
    set_seed(seed)
    
    # 读取配置
    cfg = config_io.load_config(config_path)
    train_block = cfg.get("train", {})
    
    # 确定模型目录
    if model_dir is None:
        model_dir = str(Path(train_block.get("output_dir", "runs/kv_ner")) / "best")
    
    # 确定测试数据路径（JSONL 格式）
    if test_data_path is None:
        raise ValueError("test_data_path is required")
    
    logger.info(f"配置文件: {config_path}")
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"测试数据: {test_data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"对齐模式: {align_mode}")
    
    # 加载测试数据
    logger.info("加载测试数据（JSONL 格式）...")
    test_data = _read_jsonl(Path(test_data_path))
    logger.info(f"测试样本数: {len(test_data)}")
    
    # 构建标签映射
    label_map = config_io.label_map_from(cfg)
    # 构建 BIOE 标签
    labels = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"标签: {labels}")
    
    # 加载模型
    logger.info("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(model_dir).to(device)
    
    # 加载 tokenizer
    tokenizer_path = Path(model_dir) / "tokenizer"
    tokenizer: Optional[AutoTokenizer] = None
    if tokenizer_path.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        except Exception as exc:  # fallback if saved tokenizer files are corrupted/incomplete
            logger.warning(
                "加载保存的 tokenizer 失败（%s），将回退到配置中的 tokenizer: %s",
                exc,
                tokenizer_path,
            )
    if tokenizer is None:
        tokenizer_kwargs = _collect_tokenizer_kwargs(cfg)
        tokenizer = _build_tokenizer(
            config_io.tokenizer_name_from(cfg),
            tokenizer_kwargs,
        )
        if tokenizer_kwargs:
            logger.info(
                "Tokenizer 额外参数键: %s",
                ", ".join(sorted(tokenizer_kwargs.keys())),
            )
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    
    # 评估
    max_seq_length = config_io.max_seq_length(cfg)
    chunk_size = int(cfg.get("chunk_size", 450))
    chunk_overlap = int(cfg.get("chunk_overlap", 50))
    if isinstance(override_chunk_size, int):
        chunk_size = int(override_chunk_size)
    if isinstance(override_chunk_overlap, int):
        chunk_overlap = int(override_chunk_overlap)
    error_dump_path = str(Path(output_dir) / "error_samples.jsonl")
    
    # Default boundary micro-adjust to reduce off-by-one around punctuation
    default_adjust_chars = " \t\u3000:：,，.;；。()（）[]【】{}<>%％-—–/\\"  # common separators
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        id2label=id2label,
        device=device,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_adjacent_gap=int(cfg.get("merge_adjacent_gap", 2)),
        error_dump_path=error_dump_path,
        error_threshold=error_threshold,
        align_mode=align_mode,
        exclude_keys=exclude_keys,
        report_title_filter=report_title_filter,
        value_attach_window=int(cfg.get("value_attach_window", 50)),
        value_same_line_only=bool(cfg.get("value_same_line_only", True)),
        adjust_boundaries=bool(cfg.get("adjust_boundaries", True)),
        adjust_max_shift=int(cfg.get("adjust_max_shift", 1)),
        adjust_chars=str(cfg.get("adjust_chars", default_adjust_chars)),
        max_samples=max_samples,
    )
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "config_path": config_path,
        "model_dir": model_dir,
        "test_data_path": test_data_path,
        "num_samples": results["num_samples"],
        "num_errors": results["num_errors"],
        "max_seq_length": max_seq_length,
        "seed": seed,
        "error_threshold": error_threshold,
        "align_mode": align_mode,
        "exclude_keys": exclude_keys,
        "boundary_stats_note": "已过滤：1)边界噪声字符（空格、冒号、数字、括号） 2)可容忍的±1字符差异（标点符号）",
        "text_exact_metrics": results["text_exact"],
        "text_overlap_metrics": results["text_overlap"],
        "text_exact_in_k_metrics": results.get("text_exact_in_k", {}),
        "text_exact_tolerant_metrics": results.get("text_exact_tolerant", {}),
        "text_exact_edge3_metrics": results.get("text_exact_edge3", {}),
        "boundary_stats": results.get("boundary_stats", {}),
    }
    
    summary_path = output_path / "eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("评估结果（键值对级别）")
    logger.info(f"{'='*80}")
    logger.info(f"样本数: {results['num_samples']}")
    logger.info(f"错误样本数: {results['num_errors']}")
    logger.info(f"\nText Exact Matching:")
    logger.info(f"  Precision: {results['text_exact'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Overlap Matching:")
    logger.info(f"  Precision: {results['text_overlap'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_overlap'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_overlap'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (≤K tokens) Matching:")
    logger.info(f"  Precision: {results['text_exact_in_k'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_in_k'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_in_k'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (tolerant) Matching:")
    logger.info(f"  Precision: {results['text_exact_tolerant'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_tolerant'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_tolerant'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (edge≤3) Matching:")
    logger.info(f"  Precision: {results['text_exact_edge3'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_edge3'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_edge3'].get('f1_score', 0):.4f}")
    logger.info(f"\n注意：boundary_stats 已过滤边界噪声和可容忍差异，只统计实质性问题")
    logger.info(f"\n结果已保存到: {summary_path}")
    logger.info(f"{'='*80}\n")
    
    return summary


# =========================
# 命令行入口
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 KV-NER 模型（键值对级别）")
    parser.add_argument(
        "--config",
        type=str,
        default="pre_struct/kv_ner/kv_ner_config.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="模型目录（默认从配置读取）",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="测试数据路径（JSONL 格式，默认 data/ground_truth.jsonl）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kv_ner_eval",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--error_threshold",
        type=float,
        default=0.99,
        help="错误样本 F1 阈值",
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="gold",
        choices=["gold", "pred", "union"],
        help="键对齐模式",
    )
    parser.add_argument(
        "--exclude_keys",
        type=str,
        nargs="*",
        default=None,
        help="排除的键列表",
    )
    parser.add_argument(
        "--report_titles",
        type=str,
        nargs="*",
        default=None,
        help="评估的报告类型过滤（不指定=评估所有类型，指定=只评估指定类型）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多评估多少条样本（用于快速调试）",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="覆盖配置的 chunk_size（推理/评估）",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=None,
        help="覆盖配置的 chunk_overlap（推理/评估）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        model_dir=args.model_dir,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        seed=args.seed,
        error_threshold=args.error_threshold,
        align_mode=args.align_mode,
        exclude_keys=args.exclude_keys,
        report_title_filter=args.report_titles,
        max_samples=args.max_samples,
        override_chunk_size=args.chunk_size,
        override_chunk_overlap=args.chunk_overlap,
    )
