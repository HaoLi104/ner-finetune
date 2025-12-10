# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class QACollator:
    def __init__(
        self,
        pad_id: int = 0,
        pad_token_type_id: int = 0,
        pad_attention_mask: int = 0,
        keep_debug_fields: bool = True,
    ) -> None:
        self.pad_id = int(pad_id or 0)
        self.pad_tt = int(pad_token_type_id or 0)
        self.pad_am = int(pad_attention_mask or 0)
        self.keep_debug_fields = bool(keep_debug_fields)

    def _pad_2d(self, seqs, pad_val: int):
        max_len = max(len(s) for s in seqs) if seqs else 0
        out = []
        for s in seqs:
            if len(s) < max_len:
                s = list(s) + [pad_val] * (max_len - len(s))
            out.append(s)
        return torch.tensor(out, dtype=torch.long)

    def __call__(self, batch: List[Dict[str, Any]]):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        token_type_ids = [b.get("token_type_ids", []) for b in batch]

        out = {
            "input_ids": self._pad_2d(input_ids, self.pad_id),
            "attention_mask": self._pad_2d(attention_mask, self.pad_am),
        }
        if any(len(tt) > 0 for tt in token_type_ids):
            out["token_type_ids"] = self._pad_2d(token_type_ids, self.pad_tt)

        # training labels if present (处理混合批次：有些样本有标签，有些没有)
        has_start = all("start_positions" in b for b in batch)
        has_end = all("end_positions" in b for b in batch)
        if has_start and has_end:
            out["start_positions"] = torch.tensor([int(b["start_positions"]) for b in batch], dtype=torch.long)
            out["end_positions"] = torch.tensor([int(b["end_positions"]) for b in batch], dtype=torch.long)

        # Debug fields for decoding
        if self.keep_debug_fields and batch:
            dbg_keys = (
                "question_key",
                "report_index",
                "offset_mapping",
                "sequence_ids",
                "chunk_char_start",
                "chunk_char_end",
                "chunk_text",
                "is_short_field",
            )
            for k in dbg_keys:
                vals = [b.get(k) for b in batch]
                if any(v is not None for v in vals):
                    out[k] = vals
        return out


class PrecomputedQADataset(Dataset):
    """预计算数据集加载器
    
    直接从预计算的 jsonl 文件加载样本，无需重新构建。
    """
    
    def __init__(
        self,
        precomputed_path: str,
        keep_debug_fields: bool = True,
        require_labels: bool = True,
    ) -> None:
        """
        Args:
            precomputed_path: 预计算数据文件路径 (.jsonl)
            keep_debug_fields: 是否保留调试字段
            require_labels: 是否要求样本必须有训练标签（start_positions, end_positions）
        """
        self.precomputed_path = precomputed_path
        self.keep_debug_fields = bool(keep_debug_fields)
        self.require_labels = bool(require_labels)
        self.samples: List[Dict[str, Any]] = []
        
        # 加载预计算的样本
        print(f"加载预计算数据: {precomputed_path}")
        
        # 先读取所有行
        with open(precomputed_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # 解析JSON（带进度条），并过滤无标签样本
        skipped = 0
        if HAS_TQDM:
            for line in tqdm(lines, desc="加载样本", unit="样本"):
                sample = json.loads(line)
                # 训练时过滤掉没有标签的样本
                if self.require_labels and ("start_positions" not in sample or "end_positions" not in sample):
                    skipped += 1
                    continue
                self.samples.append(sample)
        else:
            for i, line in enumerate(lines):
                sample = json.loads(line)
                # 训练时过滤掉没有标签的样本
                if self.require_labels and ("start_positions" not in sample or "end_positions" not in sample):
                    skipped += 1
                    continue
                self.samples.append(sample)
                if (i + 1) % max(1, len(lines) // 10) == 0:
                    print(f"  加载进度: {(i + 1) / len(lines) * 100:.0f}% ({i + 1}/{len(lines)})")
        
        print(f"✓ 成功加载 {len(self.samples)} 个预计算样本")
        if skipped > 0:
            print(f"⚠️  跳过 {skipped} 个无标签样本 ({skipped / len(lines) * 100:.1f}%)")
        
        # 加载元数据（如果存在）
        meta_path = Path(precomputed_path).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"元数据: {self.metadata.get('n_records', 'N/A')} 条原始数据 → {self.metadata.get('n_samples', 'N/A')} 个样本")
        else:
            self.metadata = {}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        out = {
            "input_ids": item["input_ids"],
            "token_type_ids": item.get("token_type_ids", []),
            "attention_mask": item["attention_mask"],
        }
        
        # 训练标签
        if "start_positions" in item and "end_positions" in item:
            out["start_positions"] = item["start_positions"]
            out["end_positions"] = item["end_positions"]
        
        # 调试字段
        if self.keep_debug_fields:
            for k in (
                "question_key",
                "offset_mapping",
                "sequence_ids",
                "chunk_char_start",
                "chunk_char_end",
                "report_index",
                "chunk_text",
                "is_short_field",
            ):
                if k in item:
                    out[k] = item[k]
        return out


class EnhancedQADataset(Dataset):
    """Minimal EBQA-style dataset for inference using tokenizer overflow windows.

    - Builds (question, context) pairs for each key under the given report title
    - Uses HF tokenizer overflow with stride to generate sliding windows
    - Emits debug fields required by EBQAModel.predict
    """

    def __init__(
        self,
        data_path: str,
        report_struct_path: str,
        tokenizer_name: Optional[str] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
        max_seq_len: int = 512,
        max_tokens_ctx: int = 480,
        doc_stride: Optional[int] = None,
        use_question_templates: bool = True,
        only_title_keys: bool = True,
        keep_debug_fields: bool = True,
        inference_mode: bool = True,
        autobuild: bool = True,
        show_progress: bool = False,
    ) -> None:
        self.max_seq_len = int(max_seq_len)
        self.max_tokens_ctx = int(max_tokens_ctx)
        self.doc_stride = int(doc_stride if doc_stride is not None else max(64, max_tokens_ctx // 4))
        self.use_question_templates = bool(use_question_templates)
        self.only_title_keys = bool(only_title_keys)
        self.keep_debug_fields = bool(keep_debug_fields)
        self.inference_mode = bool(inference_mode)
        self.show_progress = bool(show_progress)

        # tokenizer
        if tokenizer is not None:
            self.tok = tokenizer
        else:
            base_name = tokenizer_name
            if not base_name:
                # fallback to project defaults
                try:
                    import model_path_conf as _mpc  # type: ignore

                    base_name = getattr(
                        _mpc, "DEFAULT_TOKENIZER_PATH", getattr(_mpc, "DEFAULT_MODEL_PATH", None)
                    )
                except Exception:
                    base_name = None
            if not base_name:
                raise RuntimeError(
                    "Tokenizer path not set. Configure tokenizer_name or model_path_conf.DEFAULT_TOKENIZER_PATH."
                )
            self.tok = BertTokenizerFast.from_pretrained(base_name)
            try:
                self.tok.model_max_length = int(1e6)
            except Exception:
                pass

        # struct map
        self.struct_map: Dict[str, List[str]] = {}
        try:
            self.struct_map = json.loads(Path(report_struct_path).read_text(encoding="utf-8"))
        except Exception:
            self.struct_map = {}

        # records
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.records: List[Dict[str, Any]] = data if isinstance(data, list) else []
        self.samples: List[Dict[str, Any]] = []

        if autobuild and self.records:
            self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        out = {
            "input_ids": item["input_ids"],
            "token_type_ids": item.get("token_type_ids", []),
            "attention_mask": item["attention_mask"],
        }
        if self.keep_debug_fields:
            for k in (
                "question_key",
                "offset_mapping",
                "sequence_ids",
                "chunk_char_start",
                "chunk_char_end",
                "report_index",
                "chunk_text",
                "is_short_field",
            ):
                if k in item:
                    out[k] = item[k]
        return out

    # ----- helpers -----
    @staticmethod
    def _get_report_text(rec: Dict[str, Any]) -> str:
        # support both 'report' and legacy fields
        for key in ("report", "text", "content"):
            v = rec.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _question_keys_for(self, rec: Dict[str, Any]) -> List[str]:
        title = str(rec.get("report_title", "") or "").strip()
        keys: List[str] = []
        if self.struct_map and title and title in self.struct_map:
            ks = self.struct_map.get(title, {})
            # struct_map 可能是字典格式: {"字段名": {"别名": [], "Q": "..."}}
            if isinstance(ks, dict):
                keys.extend(list(ks.keys()))
            elif isinstance(ks, list):
                keys.extend([str(k) for k in ks if isinstance(k, str)])
        else:
            # fallback: all non-reserved keys present in record
            reserved = {"report", "report_title", "report_index"}
            for k, v in rec.items():
                if k not in reserved and isinstance(v, (str, int, float)):
                    keys.append(str(k))
        # ensure order and dedup
        seen = set()
        out = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _convert_key_to_question(self, title: str, key: str) -> str:
        try:
            from pre_struct.map_key2question import convert_key_to_question  # project helper

            return convert_key_to_question(title, key)
        except Exception:
            return f"{title} 的 {key} 是什么？"

    def _normalize_offsets_to_context(self, encoding_obj):
        # For pair inputs, set non-context offsets to (None, None)
        seq_ids = encoding_obj.sequence_ids
        if callable(seq_ids):
            seq_ids = seq_ids()
        offsets = encoding_obj.offsets
        cleaned = []
        for sid, off in zip(seq_ids, offsets):
            if sid != 1:
                cleaned.append((None, None))
            else:
                s, e = off
                cleaned.append((int(s), int(e)))
        return seq_ids, cleaned

    def _build_samples(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        processed_records = 0
        skipped_nonempty_fields = 0
        total_nonempty_fields = 0
        total_windows = 0  # 总窗口数
        skipped_window_no_answer = 0  # 跳过的窗口（答案不在此窗口）
        skipped_window_mapping_failed = 0  # 跳过的窗口（token映射失败）
        
        # 准备迭代器（带进度条）
        if self.show_progress and HAS_TQDM:
            iterator = tqdm(
                enumerate(self.records), 
                total=len(self.records),
                desc="构建样本",
                unit="条"
            )
        else:
            iterator = enumerate(self.records)
        
        for ridx, rec in iterator:
            report = self._get_report_text(rec)
            if not report:
                continue
            processed_records += 1
            title = str(rec.get("report_title", "") or "")
            keys = self._question_keys_for(rec)
            
            if not keys:
                continue
            for key in keys:
                # 训练模式下，跳过空值字段
                if not self.inference_mode:
                    value = str(rec.get(key, "") or "").strip()
                    if not value:
                        continue  # 跳过空值字段
                    total_nonempty_fields += 1
                
                q = self._convert_key_to_question(title, key)
                enc = self.tok(
                    q,
                    report,
                    max_length=self.max_seq_len,
                    truncation="only_second",
                    return_offsets_mapping=True,
                    padding=False,
                    return_overflowing_tokens=True,
                    stride=int(self.doc_stride),
                )

                n = len(enc.encodings)
                total_windows += n  # 统计总窗口数
                
                # optional gold answer for training
                gold_text = None if self.inference_mode else str(rec.get(key, "") or "").strip()
                gold_start = gold_end = -1
                if not self.inference_mode and gold_text:
                    pos = report.find(gold_text)
                    if pos >= 0:
                        gold_start, gold_end = pos, pos + len(gold_text)

                for i in range(n):
                    seq_ids, offm_global = self._normalize_offsets_to_context(enc.encodings[i])

                    ctx_offsets = [off for sid, off in zip(seq_ids, offm_global) if sid == 1 and off[0] is not None]
                    if ctx_offsets:
                        chunk_start = min(off[0] for off in ctx_offsets)
                        chunk_end = max(off[1] for off in ctx_offsets)
                    else:
                        chunk_start = 0
                        chunk_end = 0

                    offset_local = []
                    for sid, off in zip(seq_ids, offm_global):
                        if sid == 1 and off[0] is not None and off[1] is not None:
                            offset_local.append((int(off[0] - chunk_start), int(off[1] - chunk_start)))
                        else:
                            offset_local.append((None, None))

                    chunk_text = report[chunk_start:chunk_end] if (chunk_end > chunk_start and self.keep_debug_fields) else ""
                    sample = {
                        "input_ids": enc["input_ids"][i],
                        "token_type_ids": enc.get("token_type_ids", [[]])[i] if "token_type_ids" in enc else [],
                        "attention_mask": enc["attention_mask"][i],
                        "offset_mapping": offset_local,
                        "sequence_ids": seq_ids,
                        "chunk_char_start": int(chunk_start),
                        "chunk_char_end": int(chunk_end if chunk_end >= chunk_start else chunk_start),
                        "chunk_text": chunk_text,
                        "report_index": int(rec.get("report_index", ridx)),
                        "question_key": str(key),
                        "is_short_field": bool(gold_text and len(gold_text) <= 8),
                    }

                    # training: map gold char span to token span (improved from ebqa)
                    if not self.inference_mode:
                        # 训练模式：采用ebqa的智能标注策略
                        ctx_idx = [j for j, sid in enumerate(seq_ids) if sid == 1 and offm_global[j][0] is not None]
                        
                        if not ctx_idx:
                            # 没有上下文token，跳过
                            continue
                        
                        # 情况1：有有效的答案 → 尝试精确映射
                        if gold_start >= 0 and gold_end > gold_start:
                            # 检查答案是否与当前窗口有重叠
                            has_overlap = (gold_end > chunk_start and gold_start < chunk_end)
                            
                            if has_overlap:
                                # 采用ebqa的映射策略：优先选"包含"的token，再选"邻近"的token
                                t_s = None
                                # 1. 优先：第一个包含 gold_start 的token (offset_start <= gold_start < offset_end)
                                for j in ctx_idx:
                                    s, e = offm_global[j]
                                    if s is not None and e is not None and s <= gold_start < e:
                                        t_s = j
                                        break
                                # 2. 备用：第一个 offset_start >= gold_start
                                if t_s is None:
                                    for j in ctx_idx:
                                        s, e = offm_global[j]
                                        if s is not None and e is not None and s >= gold_start:
                                            t_s = j
                                            break
                                
                                if t_s is not None:
                                    # 找 end token
                                    t_e = None
                                    # 1. 优先：最后一个 offset_end <= gold_end 的token
                                    for j in reversed(ctx_idx):
                                        if j < t_s:
                                            break
                                        s, e = offm_global[j]
                                        if s is not None and e is not None and e <= gold_end:
                                            t_e = j
                                            break
                                    # 2. 备用：最后一个覆盖 gold_end-1 的token
                                    if t_e is None:
                                        for j in reversed(ctx_idx):
                                            if j < t_s:
                                                break
                                            s, e = offm_global[j]
                                            if s is not None and e is not None and s < gold_end and e >= gold_end:
                                                t_e = j
                                                break
                                    
                                    if t_e is not None and t_e >= t_s:
                                        sample["start_positions"] = int(t_s)
                                        sample["end_positions"] = int(t_e)
                                        out.append(sample)
                                        continue
                                    else:
                                        skipped_window_mapping_failed += 1
                                else:
                                    skipped_window_mapping_failed += 1
                            else:
                                # 答案不在当前窗口
                                skipped_window_no_answer += 1
                                continue
                        
                        # 情况2：无答案样本 → 添加为负样本（用于训练"无答案"能力）
                        # 标注为CLS token ([0,0])
                        sample["start_positions"] = 0
                        sample["end_positions"] = 0
                        out.append(sample)
                    else:
                        # 推理模式：保留所有窗口（不需要标签）
                        out.append(sample)
        
        if self.show_progress:
            print(f"\n数据集构建统计:")
            print(f"  处理的记录数: {processed_records}")
            print(f"  非空字段总数: {total_nonempty_fields}")
            print(f"  总滑动窗口数: {total_windows}")
            print(f"  生成的样本数: {len(out)}")
            if not self.inference_mode:
                print(f"\n滑动窗口过滤统计:")
                print(f"  答案不在窗口内: {skipped_window_no_answer} ({100*skipped_window_no_answer/max(1,total_windows):.1f}%)")
                print(f"  Token映射失败: {skipped_window_mapping_failed} ({100*skipped_window_mapping_failed/max(1,total_windows):.1f}%)")
                print(f"  成功保留: {len(out)} ({100*len(out)/max(1,total_windows):.1f}%)")
            if total_nonempty_fields > 0 and len(out) == 0:
                print(f"  ⚠️ 警告: {total_nonempty_fields} 个非空字段，但生成了 0 个样本！")
                print(f"     可能原因: 所有字段值都没能在文本中找到，或 token 映射失败")
        
        return out
