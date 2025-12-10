# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os
import sys
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到 sys.path
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pre_struct.ebqa.da_core.chunking import SemanticChunker
from pre_struct.ebqa.da_core.dataset import (
    EnhancedQADataset as _BaseDataset,
    QACollator,
)


def _clean_alias_map(data: Any) -> Dict[str, str]:
    if not isinstance(data, dict):
        return {}
    cleaned: Dict[str, str] = {}
    for raw_key, raw_val in data.items():
        key = str(raw_key).strip()
        val = str(raw_val or "").strip()
        if not key or not val:
            continue
        cleaned[key] = val
    return cleaned


class EnhancedQADataset(_BaseDataset):
    """Alias/title QA dataset that reuses the EBQA core implementation with a different question template."""

    # 线程本地存储：每个线程有独立的tokenizer
    _tokenizer_local = threading.local()
    _tokenizer_name_or_path: Optional[str] = None

    def __init__(
        self,
        data_path: str,
        data_paths: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_name: Optional[str] = None,
        max_seq_len: int = 512,
        max_tokens_ctx: int = 480,
        max_answer_len: int = 384,
        use_question_templates: bool = True,
        keep_debug_fields: bool = False,
        negative_downsample: float = 1.0,
        seed: int = 42,
        autobuild: bool = True,
        show_progress: bool = True,
        chunk_mode: str = "budget",
        report_struct_path: Optional[str] = None,
        only_title_keys: bool = False,
        use_concurrent_build: bool = False,
        max_workers: Optional[int] = None,
        inference_mode: bool = False,
        dynamic_answer_length: bool = True,
        *,
        alias_field: str = "alias",
        question_template: str = "报告中用什么词表示'{key}'？",
    ) -> None:
        self.alias_field = alias_field
        self.question_template = question_template
        requested_autobuild = bool(autobuild)
        
        # 保存tokenizer路径用于线程本地创建
        EnhancedQADataset._tokenizer_name_or_path = tokenizer_name
        self._tokenizer_name_or_path = tokenizer_name

        # 始终以 autobuild=False 调用基类，稍后在别名投影后再构建样本
        super().__init__(
            data_path=data_path,
            data_paths=data_paths,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            max_tokens_ctx=max_tokens_ctx,
            max_answer_len=max_answer_len,
            use_question_templates=use_question_templates,
            keep_debug_fields=keep_debug_fields,
            negative_downsample=negative_downsample,
            seed=seed,
            autobuild=False,
            show_progress=show_progress,
            chunk_mode=chunk_mode,
            report_struct_path=report_struct_path,
            only_title_keys=only_title_keys,
            use_concurrent_build=use_concurrent_build,
            max_workers=max_workers,
            inference_mode=inference_mode,
            dynamic_answer_length=dynamic_answer_length,
        )

        # 将别名字段投影到记录表面，方便基类逻辑复用
        self._project_alias_records()

        if self.records and requested_autobuild:
            if self.show_progress:
                print(f"[INFO] Loaded {len(self.records)} alias reports from {data_path}")
                print("[INFO] Collecting alias length priors ...")
            self._collect_length_priors()
            if self.show_progress:
                print("[INFO] Building alias QA samples ...")
            self.samples = self._build_samples(
                use_concurrent=self.use_concurrent_build,
                max_workers=self.max_workers,
            )

    # 仅对别名数据使用的辅助字段
    _ALIAS_KEYS_FIELD = "_alias_keys"
    _ALIAS_VALUE_MAP = "_alias_value_map"

    @classmethod
    def _get_thread_tokenizer(cls):
        """获取线程本地的tokenizer实例。每个线程独立加载一个tokenizer。"""
        if not hasattr(cls._tokenizer_local, 'tokenizer'):
            if not cls._tokenizer_name_or_path:
                raise RuntimeError("Tokenizer name or path not set")
            from transformers import AutoTokenizer
            cls._tokenizer_local.tokenizer = AutoTokenizer.from_pretrained(
                cls._tokenizer_name_or_path,
                trust_remote_code=True
            )
        return cls._tokenizer_local.tokenizer
    
    def _get_tok(self):
        """覆盖基类方法，使用线程本地tokenizer避免多线程冲突。"""
        # 如果启用了并发，使用线程本地tokenizer
        if self.use_concurrent_build:
            return self._get_thread_tokenizer()
        # 否则使用基类方法（单线程模式）
        return super()._get_tok()

    def _project_alias_records(self) -> None:
        """将别名字段并入记录，保留原始字段值用于在报告中定位答案。
        
        关键逻辑：
        - 标准键（如"姓名"）保留原始值（如"郑红英"），用于在报告中查找答案位置
        - alias 值（如"病人姓名"）保存在 _ALIAS_VALUE_MAP 中，用于生成问题
        """
        projected: List[Dict[str, Any]] = []
        for rec in self.records:
            rec_copy = copy.deepcopy(rec)
            alias_map = _clean_alias_map(rec_copy.get(self.alias_field))
            rec_copy[self._ALIAS_KEYS_FIELD] = list(alias_map.keys())
            rec_copy[self._ALIAS_VALUE_MAP] = alias_map
            # ⚠️ 不覆盖原始字段值！保留原始值用于在报告中定位答案
            # for key, value in alias_map.items():
            #     rec_copy[key] = value  # ❌ 这会覆盖答案值
            projected.append(rec_copy)
        self.records = projected

    def _alias_keys_for(self, rec: Dict[str, Any]) -> List[str]:
        alias_keys = rec.get(self._ALIAS_KEYS_FIELD)
        if isinstance(alias_keys, list):
            return [str(k).strip() for k in alias_keys if str(k).strip()]
        return []

    def _alias_value(self, rec: Dict[str, Any], key: str) -> str:
        """获取用于生成问题的 alias 值。
        
        返回：
        - 如果存在 alias 映射，返回 alias 值（如"病人姓名"）
        - 如果不存在，返回空字符串（与推理保持一致）
        """
        alias_map = rec.get(self._ALIAS_VALUE_MAP)
        if isinstance(alias_map, dict):
            val = alias_map.get(key)
            if isinstance(val, str) and val:
                return val
        # 没有 alias 时，返回空字符串而不是 key
        return ""

    def _question_keys_for(self, rec: Dict) -> List[str]:  # type: ignore[override]
        """获取要生成问题的字段列表。
        
        策略：把 keys_merged.json 中的所有字段分散到同类型的多条记录中
        - 每条记录只处理部分字段
        - 所有字段都会被覆盖（分散到不同记录中）
        - 有 alias 的字段总是包含
        - 其他字段按 report_index 哈希分配
        
        这样既覆盖所有字段，又避免每条记录处理太多字段导致负样本过多。
        """
        # 首先获取该记录的 alias 字段（必须包含）
        alias_keys = self._alias_keys_for(rec)
        alias_keys_set = set(alias_keys)
        
        # 然后从 struct_map 获取该report类型的所有标准字段
        report_title = rec.get('report_title', '')
        report_index = rec.get('report_index', 0)  # 记录索引用于哈希分配
        
        if report_title and self.struct_map:
            title_fields = self.struct_map.get(report_title)
            if isinstance(title_fields, dict):
                # 所有字段
                all_struct_keys = [str(k).strip() for k in title_fields.keys() if str(k).strip()]
                
                # 分离有alias和无alias的字段
                keys_with_alias = [k for k in all_struct_keys if k in alias_keys_set]
                keys_without_alias = [k for k in all_struct_keys if k not in alias_keys_set]
                
                # 对于无alias的字段，动态计算选中率以控制负样本比例在40%左右
                selected_keys_without_alias = []
                if keys_without_alias:
                    import hashlib
                    # 动态计算选中率，控制负样本比例在30-40%左右
                    # 简化后每个字段只生成1个候选，所以倍增因子接近1.0
                    target_neg_ratio = 0.40  # 目标负样本40%
                    neg_candidate_multiplier = 1.0  # 每个字段只生成1个候选
                    
                    if len(keys_with_alias) > 0:
                        # 计算需要选中多少无alias字段
                        # 公式：选中字段数 × 候选倍数 = 目标负样本数
                        target_neg_samples = len(keys_with_alias) * target_neg_ratio / (1 - target_neg_ratio)
                        target_neg_fields = int(target_neg_samples / neg_candidate_multiplier)
                        # 计算选中率（百分比）
                        selection_rate = min(100, max(5, int(target_neg_fields * 100 / len(keys_without_alias))))
                    else:
                        # 如果没有有alias的字段，使用最小覆盖率
                        selection_rate = 10
                    
                    for k in keys_without_alias:
                        # 使用 (report_title, report_index, key) 的哈希决定是否包含
                        hash_input = f"{report_title}_{report_index}_{k}".encode('utf-8')
                        hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
                        if hash_val % 100 < selection_rate:
                            selected_keys_without_alias.append(k)
                
                # 合并：有alias的字段（全部）+ 无alias的字段（部分）
                final_keys = keys_with_alias + selected_keys_without_alias
                if final_keys:
                    return final_keys
        
        # 如果 struct_map 中没有，使用 alias 字段中的键
        if alias_keys:
            return alias_keys
        
        # 如果都没有，调用基类方法
        return super()._question_keys_for(rec)

    def _format_question(self, key: str, rec: Optional[Dict[str, Any]] = None) -> str:
        """生成问题，使用 alias 的键。
        
        正确逻辑：
        - question_template: "报告中'{key}'的别名是什么？"
        - key 是 alias 的键（如"姓名"）
        - 问题变成："报告中'姓名'的别名是什么？"
        - 答案是 alias 的值（如"病人姓名"）
        """
        # 直接使用 key（alias字段的键），不需要替换
        return self.question_template.format(key=key)

    def _build_one_report(self, ridx: int, rec: Dict) -> List[Dict]:  # type: ignore[override]
        out: List[Dict] = []
        report = self._get_report_text(rec)
        if not report:
            return out

        tok = self._get_tok()
        chunker = SemanticChunker(
            tok, max_tokens_ctx=self.max_tokens_ctx, chunk_mode=self.chunk_mode
        )

        keys = self._question_keys_for(rec)
        if not keys:
            return []

        if self.inference_mode:
            expected_map: Dict[str, str] = {}
        else:
            # 别名识别任务：答案是别名值本身
            # 在报告中查找别名值的位置
            # 例如：key="年龄", alias_map={"年龄": "患者年龄"}
            # 则在报告中查找"患者年龄"的位置
            alias_map = rec.get(self.alias_field, {}) or {}
            expected_map = {}
            for k in keys:
                # 只有当 key 在 alias_map 中有对应的别名时，才加入 expected_map
                if k in alias_map and alias_map[k]:
                    expected_map[k] = str(alias_map[k]).strip()
                # 如果没有别名，则不加入 expected_map（这会导致无法找到 span，生成负样本）
        spans = self._extract_spans_from_report(report, keys, expected_map=expected_map)

        rng = random.Random(self._base_seed + int(ridx))

        questions = []
        question_tokens = {}
        if self.use_question_templates:
            questions_batch = [self._format_question(key, rec) for key in keys]
        else:
            raise ValueError("use_question_templates must be True")

        for key, question in zip(keys, questions_batch):
            questions.append((key, question))
            question_tokens[key] = self._cached_tokenize_len(question)

        for key, question in questions:
            q_tok = question_tokens[key]
            ctx_budget = max(1, min(self.max_tokens_ctx, self.max_seq_len - q_tok - 3))

            s_abs, e_abs = spans.get(key, (-1, -1))

            if s_abs >= 0 and e_abs >= 0 and s_abs < e_abs:
                lines = chunker.line_spans(report)
                if lines:
                    L = next(
                        (
                            i
                            for i, sp in enumerate(lines)
                            if sp["start"] <= s_abs < sp["end"]
                        ),
                        0,
                    )
                    R = next(
                        (
                            i
                            for i, sp in enumerate(lines)
                            if sp["start"] < e_abs <= sp["end"]
                        ),
                        len(lines) - 1,
                    )
                    R2 = min(R + 1, len(lines) - 1)
                    s0, e0 = lines[L]["start"], lines[R2]["end"]
                    candidate = [
                        {"text": report[s0:e0], "char_start": s0, "char_end": e0}
                    ]
                else:
                    candidate = [
                        {"text": report, "char_start": 0, "char_end": len(report)}
                    ]
            else:
                # 负样本：简化生成，只生成1个候选chunk（避免大量overlapping chunks）
                # 每个负样本字段生成1个随机chunk
                # 只生成1个随机chunk作为负样本候选
                lines = chunker.line_spans(report)
                if lines and len(lines) > 1:
                    # 随机选择一个起始位置
                    start_idx = rng.randint(0, len(lines) - 1)
                    end_idx = min(start_idx + rng.randint(1, 3), len(lines))
                    s0, e0 = lines[start_idx]["start"], lines[end_idx - 1]["end"]
                    candidate = [{"text": report[s0:e0], "char_start": s0, "char_end": e0}]
                else:
                    # 只有1个段落：直接用整个报告
                    candidate = [{"text": report, "char_start": 0, "char_end": len(report)}]

            chunks: List[Dict] = []
            for cand in candidate:
                if self._cached_tokenize_len(cand["text"]) > ctx_budget:
                    kb = chunker.split_with_keys(cand["text"], keys=[key] + keys)
                    chosen = []
                    for ch in (kb or []):
                        if self._cached_tokenize_len(ch["text"]) <= ctx_budget:
                            ch["char_start"] += cand["char_start"]
                            ch["char_end"] += cand["char_start"]
                            chosen.append(ch)
                    if not chosen:
                        sub = chunker.split(cand["text"], budget_tokens=ctx_budget)
                        for ch in sub:
                            ch["char_start"] += cand["char_start"]
                            ch["char_end"] += cand["char_start"]
                        chunks.extend(sub)
                    else:
                        chunks.extend(chosen)
                else:
                    chunks.append(cand)

            question_chunk_pairs = [(question, ch["text"]) for ch in chunks]

            features = []
            fixed_chunks = []
            for ch in chunks:
                try:
                    feat = self._encode_pair(question, ch["text"])
                    features.append(feat)
                    fixed_chunks.append(ch)
                except RuntimeError as e:
                    if "truncated" in str(e).lower():
                        sub_chunks = chunker.split(
                            ch["text"], budget_tokens=int(ctx_budget * 0.8)
                        )
                        for sub in sub_chunks:
                            sub["char_start"] += ch["char_start"]
                            sub["char_end"] += ch["char_start"]
                            try:
                                feat = self._encode_pair(question, sub["text"])
                                features.append(feat)
                                fixed_chunks.append(sub)
                            except RuntimeError:
                                continue
                    else:
                        raise
            chunks = fixed_chunks

            for ci, (ch, feat) in enumerate(zip(chunks, features)):
                start_pos, end_pos = (0, 0)
                length_reasonableness = 1.0

                field_exists_in_record = (
                    key in rec
                    and str(rec.get(key, "")).strip()
                    and str(rec.get(key, "")).strip() != ""
                )

                if s_abs >= 0 and e_abs >= 0:
                    while s_abs < e_abs and report[s_abs].isspace():
                        s_abs += 1
                    while e_abs > s_abs and report[e_abs - 1].isspace():
                        e_abs -= 1
                    while e_abs > s_abs and report[e_abs - 1] in "，。,；;、:：)]）】>》":
                        e_abs -= 1

                    s_use = max(ch["char_start"], s_abs)
                    e_use = min(ch["char_end"], e_abs)

                    if s_abs < ch["char_start"] or e_abs > ch["char_end"]:
                        continue

                    if e_use > s_use:
                        s_rel = s_use - ch["char_start"]
                        e_rel = e_use - ch["char_start"]
                        sp, ep = self._char_span_to_token_span(
                            feat["offset_mapping"],
                            feat["sequence_ids"],
                            s_rel,
                            e_rel,
                        )
                        if sp is not None and ep is not None:
                            actual_len = ep - sp + 1
                            start_pos, end_pos = (sp, ep)

                            try:
                                tokenizer = self._get_tok()
                            except Exception:
                                tokenizer = None
                            is_short_field = _BaseDataset._is_short_field(
                                str(rec.get(key, "")), tokenizer
                            )

                            if self.dynamic_answer_length:
                                field_value = str(rec.get(key, "")).strip()
                                expected_len = (
                                    self._cached_tokenize_len(field_value)
                                    if field_value
                                    else actual_len
                                )
                                length_reasonableness = self._length_reasonableness_score(
                                    key=key,
                                    actual_len=actual_len,
                                    expected_len=expected_len,
                                    is_short=is_short_field,
                                )
                            else:
                                length_reasonableness = 1.0

                if start_pos == 0 and end_pos == 0:
                    if self.inference_mode:
                        pass
                    else:
                        # 负样本下采样策略
                        # - 字段不存在：这是"真负样本"，保留所有以避免模型过度预测
                        # - 字段存在但未找到：可能是标注问题，按常规下采样
                        if not field_exists_in_record:
                            # 字段不存在的负样本全部保留（系数 1.0）
                            keep_prob = self.negative_downsample * 1.0
                            if rng.random() > keep_prob:
                                continue
                        elif self.negative_downsample < 1.0:
                            if rng.random() > self.negative_downsample:
                                continue

                sample = {
                    "input_ids": feat["input_ids"],
                    "token_type_ids": feat["token_type_ids"],
                    "attention_mask": feat["attention_mask"],
                    "start_positions": start_pos,
                    "end_positions": end_pos,
                    "question_key": key,
                    "chunk_index": ci,
                    "report_index": ridx,
                }
                try:
                    tokenizer = self._get_tok()
                except Exception:
                    tokenizer = None
                sample["is_short_field"] = _BaseDataset._is_short_field(
                    str(rec.get(key, "")), tokenizer
                )
                sample["length_reasonableness"] = length_reasonableness

                if self.keep_debug_fields:
                    sample.update(
                        {
                            "offset_mapping": feat["offset_mapping"],
                            "sequence_ids": feat["sequence_ids"],
                            "chunk_char_start": ch["char_start"],
                            "chunk_char_end": ch["char_end"],
                            "chunk_text": ch["text"],
                        }
                    )
                out.append(sample)
        return out


__all__ = ["EnhancedQADataset", "QACollator"]
