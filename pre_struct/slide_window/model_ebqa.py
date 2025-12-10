# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
)

try:
    import model_path_conf as _mpc  # project-level default paths
except Exception:
    _mpc = None  # optional


def _has_tokenizer_files(path: Optional[str]) -> bool:
    if not path:
        return False
    if not os.path.isdir(path):
        return False
    for f in [
        "tokenizer.json",
        "vocab.txt",
        "spiece.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        if os.path.isfile(os.path.join(path, f)):
            return True
    return False


def _resolve_tokenizer_path(model_dir_or_id: str, tokenizer_name_or_path: Optional[str]) -> str:
    # explicit
    if tokenizer_name_or_path and str(tokenizer_name_or_path).strip():
        return str(tokenizer_name_or_path).strip()
    # inside model dir
    if _has_tokenizer_files(model_dir_or_id):
        return model_dir_or_id
    # saved train_config.json
    try:
        tc = os.path.join(model_dir_or_id, "train_config.json")
        if os.path.isfile(tc):
            meta = json.load(open(tc, "r", encoding="utf-8"))
            cand = (
                meta.get("tokenizer_name")
                or meta.get("tokenizer_name_or_path")
                or meta.get("model_name_or_path")
            )
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
    except Exception:
        pass
    # project defaults
    if _mpc is not None:
        cand = getattr(_mpc, "DEFAULT_TOKENIZER_PATH", None) or getattr(_mpc, "DEFAULT_MODEL_PATH", None)
        if cand:
            return cand
    raise RuntimeError("Tokenizer path not provided; set tokenizer_name_or_path or model_path_conf.DEFAULT_TOKENIZER_PATH")


class EBQADecoder:
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        max_answer_len: int = 128,
        top_k: int = 20,
        short_field_boost: float = 0.2,
        dyn_alpha: float = 0.90,
        short_cap: int = 6,
        short_caps_by_key: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tok = tokenizer
        self.max_answer_len = int(max_answer_len)
        self.top_k = int(top_k)
        self.short_field_boost = float(short_field_boost)
        self.dyn_alpha = float(dyn_alpha)
        self.short_cap = int(short_cap)
        self.short_caps_by_key = short_caps_by_key or {}

    @staticmethod
    def _ctx_indices(
        sequence_ids: List[Optional[int]],
        offset_mapping: List[Tuple[Optional[int], Optional[int]]],
    ) -> List[int]:
        idx = []
        for i, sid in enumerate(sequence_ids):
            if sid == 1:
                off = offset_mapping[i]
                if off is not None and off[0] is not None and off[1] is not None:
                    idx.append(i)
        return idx

    def _get_cap_for_key(self, question_key: Optional[str]) -> int:
        if question_key and question_key in self.short_caps_by_key:
            return min(self.short_cap, int(self.short_caps_by_key[question_key]))
        return self.short_cap

    def _dynamic_cap_for_start(
        self,
        end_logits: np.ndarray,
        ctx_sel: List[int],
        s_abs: int,
        is_short: bool,
        question_key: Optional[str] = None,
    ) -> int:
        try:
            s_idx = ctx_sel.index(s_abs)
        except ValueError:
            return self._get_cap_for_key(question_key) if is_short else self.max_answer_len
        tail_indices = ctx_sel[s_idx:]
        if len(tail_indices) <= 1:
            return 1
        tail_logits = end_logits[tail_indices]
        exp_logits = np.exp(tail_logits - np.max(tail_logits))
        probs = exp_logits / np.sum(exp_logits)
        cumsum = np.cumsum(probs)
        cap_idx = np.searchsorted(cumsum, self.dyn_alpha)
        cap_idx = min(cap_idx + 1, len(tail_indices))
        if is_short:
            cap_idx = min(cap_idx, self._get_cap_for_key(question_key))
        cap_idx = min(cap_idx, self.max_answer_len)
        return max(1, cap_idx)

    def best_span_in_chunk(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        offset_mapping: List[Tuple[Optional[int], Optional[int]]],
        sequence_ids: List[Optional[int]],
        chunk_text: str,
        chunk_char_start: int,
        is_short_field: bool = False,
        question_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        ctx_sel = self._ctx_indices(sequence_ids, offset_mapping)
        if not ctx_sel:
            return {
                "text": "",
                "score": -1e9,
                "start_char": -1,
                "end_char": -1,
                "token_start": -1,
                "token_end": -1,
                "start_logit": float("-inf"),
                "end_logit": float("-inf"),
                "null_score": float(start_logits[0] + end_logits[0]) if len(start_logits) > 0 else -1e9,
            }

        s_logits = start_logits[ctx_sel]
        e_logits = end_logits[ctx_sel]
        k = min(self.top_k, len(ctx_sel))
        top_s_rel = np.argpartition(-s_logits, kth=k - 1)[:k]
        top_e_rel = np.argpartition(-e_logits, kth=k - 1)[:k]
        top_s = [ctx_sel[int(i)] for i in top_s_rel]
        top_e = [ctx_sel[int(i)] for i in top_e_rel]

        best = {
            "text": "",
            "score": -1e9,
            "start_char": -1,
            "end_char": -1,
            "token_start": -1,
            "token_end": -1,
            "start_logit": float("-inf"),
            "end_logit": float("-inf"),
            "null_score": float(start_logits[0] + end_logits[0]) if len(start_logits) > 0 else -1e9,
        }

        for s_abs in top_s:
            dyn_cap = self._dynamic_cap_for_start(end_logits, ctx_sel, s_abs, is_short_field, question_key)
            for e_abs in top_e:
                if e_abs < s_abs:
                    continue
                span_len = e_abs - s_abs + 1
                if span_len > dyn_cap:
                    continue
                score = float(start_logits[s_abs] + end_logits[e_abs])
                if is_short_field and self.short_field_boost > 0:
                    span_length = e_abs - s_abs + 1
                    if span_length <= 8:
                        score += self.short_field_boost * (8 - span_length) / 8
                s_char, e_char = offset_mapping[s_abs][0], offset_mapping[e_abs][1]
                if s_char is None or e_char is None:
                    continue
                if score > best["score"]:
                    s_loc, e_loc = int(s_char), int(e_char)
                    # light tail trim on punctuation/whitespace
                    _TRIM_TAIL = set(" \t\r\n，。,；;、:：)]）】>》")
                    while e_loc > s_loc and chunk_text and chunk_text[e_loc - 1] in _TRIM_TAIL:
                        e_loc -= 1
                    # re-align token end to e_loc
                    t_s, t_e = s_abs, e_abs
                    # start: first token with offset_end > s_loc
                    for j in ctx_sel:
                        ss, ee = offset_mapping[j]
                        if ss is None or ee is None:
                            continue
                        if ee > s_loc:
                            t_s = j
                            break
                    # end: prefer token covering e_loc
                    t_e_new = None
                    for j in reversed(ctx_sel):
                        ss, ee = offset_mapping[j]
                        if ss is None or ee is None:
                            continue
                        if ss < e_loc and ee >= e_loc:
                            t_e_new = j
                            break
                    if t_e_new is None:
                        for j in reversed(ctx_sel):
                            ss, ee = offset_mapping[j]
                            if ss is None or ee is None:
                                continue
                            if ee <= e_loc:
                                t_e_new = j
                                break
                    t_e = max(t_e_new if t_e_new is not None else t_e, t_s)
                    best.update(
                        {
                            "score": score,
                            "token_start": int(t_s),
                            "token_end": int(t_e),
                            "start_logit": float(start_logits[t_s]),
                            "end_logit": float(end_logits[t_e]),
                            "start_char": int(chunk_char_start + s_loc),
                            "end_char": int(chunk_char_start + e_loc),
                            "text": (chunk_text[s_loc:e_loc] if chunk_text else ""),
                        }
                    )
        if best["score"] <= -1e8 and len(ctx_sel) > 0:
            s_abs = ctx_sel[int(np.argmax(start_logits[ctx_sel]))]
            dyn_cap = self._dynamic_cap_for_start(end_logits, ctx_sel, s_abs, is_short_field, question_key)
            s_idx = ctx_sel.index(s_abs)
            valid_ends = ctx_sel[s_idx : s_idx + dyn_cap]
            e_abs = valid_ends[int(np.argmax(end_logits[valid_ends]))] if valid_ends else s_abs
            if e_abs < s_abs:
                e_abs = s_abs
            s_char, e_char = offset_mapping[s_abs][0], offset_mapping[e_abs][1]
            if s_char is not None and e_char is not None:
                score = float(start_logits[s_abs] + end_logits[e_abs])
                if is_short_field and self.short_field_boost > 0:
                    span_length = e_abs - s_abs + 1
                    if span_length <= 8:
                        score += self.short_field_boost * (8 - span_length) / 8
                s_loc, e_loc = int(s_char), int(e_char)
                _TRIM_TAIL = set(" \t\r\n，。,；;、:：)]）】>》")
                while e_loc > s_loc and chunk_text and chunk_text[e_loc - 1] in _TRIM_TAIL:
                    e_loc -= 1
                t_s, t_e = s_abs, e_abs
                for j in [*ctx_sel]:
                    ss, ee = offset_mapping[j]
                    if ss is None or ee is None:
                        continue
                    if ee > s_loc:
                        t_s = j
                        break
                t_e_new = None
                for j in reversed(ctx_sel):
                    ss, ee = offset_mapping[j]
                    if ss is None or ee is None:
                        continue
                    if ss < e_loc and ee >= e_loc:
                        t_e_new = j
                        break
                if t_e_new is None:
                    for j in reversed(ctx_sel):
                        ss, ee = offset_mapping[j]
                        if ss is None or ee is None:
                            continue
                        if ee <= e_loc:
                            t_e_new = j
                            break
                t_e = max(t_e_new if t_e_new is not None else t_e, t_s)
                best.update(
                    {
                        "score": score,
                        "token_start": int(t_s),
                        "token_end": int(t_e),
                        "start_logit": float(start_logits[t_s]),
                        "end_logit": float(end_logits[t_e]),
                        "start_char": int(chunk_char_start + s_loc),
                        "end_char": int(chunk_char_start + e_loc),
                        "text": (chunk_text[s_loc:e_loc] if chunk_text else ""),
                    }
                )
        return best


class EBQAModel:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        per_device_eval_batch_size: int = 16,
        fp16: bool = True,
        max_answer_len: int = 128,
        tokenizer: Optional[BertTokenizerFast] = None,
    ) -> None:
        tok_path = _resolve_tokenizer_path(model_name_or_path, tokenizer_name_or_path)
        self.tokenizer = tokenizer or BertTokenizerFast.from_pretrained(tok_path)
        
        # 处理本地路径和HuggingFace Hub路径
        from pathlib import Path
        model_path = Path(model_name_or_path)
        if model_path.exists() and model_path.is_dir():
            # 本地路径存在，直接加载
            self.model = BertForQuestionAnswering.from_pretrained(str(model_path))
        else:
            # 尝试从HuggingFace Hub加载
            try:
                self.model = BertForQuestionAnswering.from_pretrained(model_name_or_path)
            except Exception as e:
                # 如果Hub加载失败，检查是否是本地路径问题
                if Path(model_name_or_path).exists():
                    self.model = BertForQuestionAnswering.from_pretrained(str(model_name_or_path))
                else:
                    raise ValueError(f"无法加载模型: {model_name_or_path}. 错误: {str(e)}")
        
        self.batch_size = int(per_device_eval_batch_size)
        self.fp16 = bool(fp16 and torch.cuda.is_available())
        self.decoder = EBQADecoder(self.tokenizer, max_answer_len=max_answer_len, short_cap=6)

    def _get_device(self) -> torch.device:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _get_chunk_text(self, batch, i: int) -> str:
        try:
            return str(batch.get("chunk_text")[i])
        except Exception:
            return ""

    @torch.no_grad()
    def predict(
        self,
        dataset,
        data_collator,
        batch_size: Optional[int] = None,
        enable_no_answer: bool = True,
        null_threshold: float = 0.0,
        null_agg: str = "mean",
        debug_print: bool = False,
    ) -> List[Dict[str, Any]]:
        device = self._get_device()
        self.model.eval().to(device)
        bs = int(batch_size or self.batch_size)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=data_collator)

        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            offset_mapping = batch.get("offset_mapping", [])
            sequence_ids = batch.get("sequence_ids", [])
            chunk_texts = batch.get("chunk_text", None)
            chunk_starts = batch.get("chunk_char_start", [0] * len(offset_mapping))
            report_indices = batch.get("report_index", [0] * len(offset_mapping))
            question_keys = batch.get("question_key", [""] * len(offset_mapping))
            is_short_fields = batch.get("is_short_field", None)

            for i in range(len(offset_mapping)):
                key = (int(report_indices[i]), str(question_keys[i]))
                s_log = start_logits[i]
                e_log = end_logits[i]
                ch_txt = str(chunk_texts[i]) if chunk_texts is not None else ""
                ch_s = int(chunk_starts[i]) if chunk_starts is not None else 0
                seq_ids = list(sequence_ids[i])

                is_short = False
                if is_short_fields is not None and i < len(is_short_fields):
                    try:
                        is_short = bool(is_short_fields[i])
                    except Exception:
                        is_short = False

                cand = self.decoder.best_span_in_chunk(
                    start_logits=s_log,
                    end_logits=e_log,
                    offset_mapping=offset_mapping[i],
                    sequence_ids=seq_ids,
                    chunk_text=ch_txt,
                    chunk_char_start=ch_s,
                    is_short_field=is_short,
                    question_key=str(question_keys[i]),
                )

                slot = buckets.get(key)
                if slot is None:
                    buckets[key] = {
                        "report_index": key[0],
                        "answer": cand,
                        "best_span": float(cand["score"]),
                        "best_null": float(cand.get("null_score", -1e9)),
                        "_null_list": [float(cand.get("null_score", -1e9))],
                    }
                else:
                    if float(cand["score"]) > slot["best_span"]:
                        slot["best_span"] = float(cand["score"])
                        slot["answer"] = cand
                    slot["_null_list"].append(float(cand.get("null_score", -1e9)))
                    if null_agg == "max":
                        slot["best_null"] = max(slot["best_null"], float(cand.get("null_score", -1e9)))

        results: List[Dict[str, Any]] = []
        for (r_idx, q_key), slot in buckets.items():
            best_span = float(slot["best_span"])
            best_null = float(slot["best_null"]) if null_agg == "max" else float(np.mean(slot["_null_list"]))
            ans = slot["answer"]
            if enable_no_answer and (best_null - best_span) > float(null_threshold):
                final = {"text": "", "score": best_null, "start_char": -1, "end_char": -1}
            else:
                final = {
                    "text": str(ans.get("text", "")),
                    "score": float(ans.get("score", -1e9)),
                    "start_char": int(ans.get("start_char", -1)),
                    "end_char": int(ans.get("end_char", -1)),
                }
            results.append(
                {
                    "report_index": int(r_idx),
                    "question_key": str(q_key),
                    "text": str(final.get("text", "")),
                    "score": float(final.get("score", -1e9)),
                    "start_char": int(final.get("start_char", -1)),
                    "end_char": int(final.get("end_char", -1)),
                    "best_null": float(best_null),
                    "best_span": float(best_span),
                }
            )
        return results

