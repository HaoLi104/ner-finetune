#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Alias-title QA inference (test.py)
- 与 dataset.py/train.py 一致的问法与切分
- 只用结构文件(标准键列表)，不需要 alias_map
- Public API: predict_one(cfg, title, report_text)  不变
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
try:
    from torch.amp import autocast  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import autocast  # PyTorch 1.x

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复用 EBQA 组件
from pre_struct.ebqa.da_core.chunking import SemanticChunker


# ---------------- config ----------------
@dataclass
class PredictConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    report_struct_path: str
    batch_size: int
    max_seq_len: int
    max_tokens_ctx: int
    max_answer_len: int
    chunk_mode: str
    use_question_templates: bool = True
    enable_no_answer: bool = False
    null_threshold: float = 0.0
    null_agg: str = "max"
    question_template: str = "报告中用什么词表示'{key}'？"


def load_cfg(path: str) -> PredictConfig:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    pred = cfg.get("predict", {}) or {}
    train = cfg.get("train", {}) or {}
    return PredictConfig(
        model_name_or_path=str(cfg.get("model_dir", cfg.get("output_dir"))),
        tokenizer_name_or_path=str(
            cfg.get("tokenizer_name_or_path", cfg.get("model_name_or_path"))
        ),
        report_struct_path=str(cfg.get("report_struct_path", "keys/keys_merged.json")),
        batch_size=int(pred.get("batch_size", 8)),
        max_seq_len=int(cfg.get("max_seq_len", 512)),
        max_tokens_ctx=int(cfg.get("max_tokens_ctx", 500)),
        max_answer_len=int(cfg.get("max_answer_len", 200)),
        chunk_mode=str(cfg.get("chunk_mode", "budget")),
        use_question_templates=bool(pred.get("use_question_templates", True)),
        enable_no_answer=bool(pred.get("enable_no_answer", False)),
        null_threshold=float(pred.get("null_threshold", 0.0)),
        null_agg=str(pred.get("null_agg", "max")),
        question_template=str(train.get("question_template", "报告中用什么词表示'{key}'？")),
    )


# ---------------- utils ----------------
def _get_logger():
    import logging

    lg = logging.getLogger("alias_test")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        )
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


def _load_title_keys(struct_path: str, title: str) -> List[str]:
    """从 STRUCTURE_MAP_0919.json 读取某标题对应的标准键列表。"""
    p = Path(struct_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8")) or {}
    node = data.get(title)
    if isinstance(node, dict):
        # 允许 { "fields":[...]} 或 {"keys":[...]} 或 {canon:[aliases]} 这几种形态
        if "fields" in node and isinstance(node["fields"], list):
            return [str(x).strip() for x in node["fields"] if str(x).strip()]
        if "keys" in node and isinstance(node["keys"], list):
            return [str(x).strip() for x in node["keys"] if str(x).strip()]
        # 若是 {canon:[aliases]}，取 canon 列表
        if all(isinstance(v, list) for v in node.values()):
            return [str(k).strip() for k in node.keys() if str(k).strip()]
        # ★ 新增：支持 {canon: {"别名": [], "Q": ""}} 格式
        if all(isinstance(v, dict) and "别名" in v for v in node.values()):
            return [str(k).strip() for k in node.keys() if str(k).strip()]
        return []
    if isinstance(node, list):
        return [str(x).strip() for x in node if str(x).strip()]
    return []


def _build_questions(keys: List[str], question_template: str, use_tpl: bool = True) -> Dict[str, str]:
    """使用与训练时相同的问题模板生成问题。
    
    Args:
        keys: 字段名列表
        question_template: 问题模板，如 "报告中用什么词表示'{key}'？"
        use_tpl: 是否使用模板
    """
    if use_tpl:
        return {k: question_template.format(key=k) for k in keys}
    return {k: k for k in keys}


def _best_span_from_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    seq_ids: List[Optional[int]],
    offsets: List[Tuple[Optional[int], Optional[int]]],
    max_answer_len: int,
) -> Tuple[int, int, float]:
    """
    只在 context(seq_id==1) 中找最优 (s,e)，返回 token 级别索引 (s,e, score)。
    score = start_logits[s] + end_logits[e]
    """
    # 允许的位置
    ctx_idx = [i for i, sid in enumerate(seq_ids) if sid == 1]
    if not ctx_idx:
        return -1, -1, -1e9

    best_s, best_e, best_score = -1, -1, -1e9
    # 简洁起见 O(n^2)；如需更快可转为前缀最大
    for i in ctx_idx:
        # 跳过无效 offset
        si = offsets[i]
        if si is None or si[0] is None or si[1] is None:
            continue
        for j in ctx_idx:
            if j < i:
                continue
            if (j - i + 1) > max_answer_len:
                break
            ej = offsets[j]
            if ej is None or ej[0] is None or ej[1] is None:
                continue
            score = float(start_logits[i]) + float(end_logits[j])
            if score > best_score:
                best_s, best_e, best_score = i, j, score
    return best_s, best_e, best_score


# ---------------- model management ----------------
class TitleQAModel:
    """预定义的Title问答模型，避免重复加载"""
    
    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        # CUDA优化：启用FP16混合精度
        if self.device.type == "cuda":
            self.model.half()
        logger = _get_logger()
        logger.info(f"✓ 模型已加载到设备: {self.device} | CUDA可用: {torch.cuda.is_available()}")
        self.chunker = SemanticChunker(
            self.tokenizer, max_tokens_ctx=cfg.max_tokens_ctx, chunk_mode=cfg.chunk_mode
        )
    
    @torch.no_grad()
    def predict_batch(self, questions: List[str], contexts: List[str]) -> torch.Tensor:
        """批量预测，返回start_logits和end_logits"""
        enc = self.tokenizer(
            questions,
            contexts,
            max_length=self.cfg.max_seq_len,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        # CUDA优化：使用混合精度推理
        if self.device.type == "cuda":
            with autocast('cuda'):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )
        
        return {
            'start_logits': out.start_logits.detach().cpu().float(),
            'end_logits': out.end_logits.detach().cpu().float(),
            'encodings': enc.encodings
        }


# ---------------- model cache ----------------
_model_cache = {}

def load_title_model(cfg: PredictConfig) -> TitleQAModel:
    """加载并返回预定义的模型实例"""
    return TitleQAModel(cfg)

def _get_cached_model(cfg: PredictConfig) -> TitleQAModel:
    """获取缓存的模型，避免重复加载"""
    cache_key = (cfg.model_name_or_path, cfg.tokenizer_name_or_path)
    
    if cache_key not in _model_cache:
        logger = _get_logger()
        logger.info(f"首次加载模型: {cfg.model_name_or_path}")
        _model_cache[cache_key] = TitleQAModel(cfg)
        logger.info(f"模型加载完成，缓存键: {cache_key}")
    
    return _model_cache[cache_key]

def clear_model_cache():
    """清理模型缓存（释放内存）"""
    global _model_cache
    logger = _get_logger()
    logger.info(f"清理模型缓存，释放 {len(_model_cache)} 个缓存模型")
    _model_cache.clear()

def get_cache_info() -> Dict[str, Any]:
    """获取缓存信息"""
    return {
        "cached_models": len(_model_cache),
        "cache_keys": list(_model_cache.keys())
    }


# ---------------- core: predict_one ----------------
@torch.no_grad()
def predict_one_with_model(
    model: TitleQAModel, title: str, report_text: str
) -> Dict[str, Dict[str, Any]]:
    """
    使用预加载模型进行预测
    给定 (title, report_text) -> {标准键: {start, end, text}}
    start/end 是在原始全文中的字符偏移；未命中则返回 -1 和空串。
    """
    logger = _get_logger()
    keys = _load_title_keys(model.cfg.report_struct_path, title)
    if not keys:
        logger.warning(
            f"[schema] '{title}' 没有标准键（或文件不存在）：{model.cfg.report_struct_path}"
        )
        return {}

    # 组 (key, chunk) jobs（按问题长度计算 budget，确保与训练一致的长度约束）
    qmap = _build_questions(keys, model.cfg.question_template, use_tpl=model.cfg.use_question_templates)
    jobs: List[Tuple[str, Dict[str, Any]]] = []

    # 为每个键基于问题长度计算 ctx 预算，并进行 budget-aware 切分
    for k in keys:
        q = qmap[k]
        try:
            q_tok = len(model.tokenizer.tokenize(q))
        except Exception:
            q_tok = 0
        ctx_budget = max(1, min(model.cfg.max_tokens_ctx, model.cfg.max_seq_len - q_tok - 3))

        # 先按预算切分，确保每段不被编码阶段截断
        chunks = model.chunker.split(report_text or "", budget_tokens=ctx_budget)
        if not chunks:
            chunks = [
                {
                    "text": report_text or "",
                    "char_start": 0,
                    "char_end": len(report_text or ""),
                }
            ]
        for ch in chunks:
            jobs.append((k, ch))

    # 为每个键聚合最佳 span 与 null 分数（与 EBQA 聚合逻辑一致）
    outputs: Dict[str, Dict[str, Any]] = {k: {"start": -1, "end": -1, "text": ""} for k in keys}
    best_scores: Dict[str, float] = {k: -1e9 for k in keys}
    best_null: Dict[str, float] = {k: -1e9 for k in keys}
    null_list: Dict[str, List[float]] = {k: [] for k in keys}

    # 批处理
    bs = max(1, int(model.cfg.batch_size))
    for i in range(0, len(jobs), bs):
        batch = jobs[i : i + bs]
        questions = [qmap[k] for (k, _) in batch]
        contexts = [ch["text"] for (_, ch) in batch]

        # 使用模型的批量预测方法
        pred_result = model.predict_batch(questions, contexts)
        start_logits = pred_result['start_logits']
        end_logits = pred_result['end_logits']
        encodings = pred_result['encodings']

        for b, (key, ch) in enumerate(batch):
            enc0 = encodings[b]
            # sequence_ids 兼容调用
            seq_attr = getattr(enc0, "sequence_ids", None)
            seq_ids = list(seq_attr()) if callable(seq_attr) else list(seq_attr)
            # offsets：对非-context置 None
            offs = []
            for sid, (s, e) in zip(seq_ids, enc0.offsets):
                if sid == 1 and s is not None and e is not None:
                    offs.append((int(s), int(e)))
                else:
                    offs.append((None, None))

            s_tok, e_tok, score = _best_span_from_logits(
                start_logits[b], end_logits[b], seq_ids, offs, model.cfg.max_answer_len
            )
            if s_tok < 0 or e_tok < s_tok:
                continue

            a, b_ = offs[s_tok][0], offs[e_tok][1]
            if a is None or b_ is None or b_ <= a:
                continue

            # 转回到全文字符偏移
            chunk_start = int(ch.get("char_start", 0))
            abs_s = chunk_start + a
            abs_e = chunk_start + b_
            text = (ch["text"] or "")[a:b_]

            # 记录 null 分数
            null_score = float(start_logits[b][0] + end_logits[b][0])
            null_list[key].append(null_score)
            if model.cfg.null_agg == "max":
                best_null[key] = max(best_null[key], null_score)

            # 按分数聚合最优 span（不在此处做 no-answer 判定）
            if score > best_scores[key] and text.strip():
                best_scores[key] = score
                outputs[key] = {"start": abs_s, "end": abs_e, "text": text}

    # 统一 no-answer 判定（与 EBQA 逻辑对齐）
    for k in keys:
        if model.cfg.null_agg == "mean" and null_list[k]:
            best_null[k] = float(sum(null_list[k]) / max(1, len(null_list[k])))

        if model.cfg.enable_no_answer:
            span_score = float(best_scores[k])
            null_sc = float(best_null[k])
            if (null_sc - span_score) > float(model.cfg.null_threshold):
                outputs[k] = {"start": -1, "end": -1, "text": ""}

    return outputs


# ---------------- main API ----------------
@torch.no_grad()
def predict_one(
    cfg: PredictConfig, title: str, report_text: str
) -> Dict[str, Dict[str, Any]]:
    """
    主要预测函数，使用缓存模型（只在首次调用时加载）
    给定 (title, report_text) -> {标准键: {start, end, text}}
    start/end 是在原始全文中的字符偏移；未命中则返回 -1 和空串。
    
    调用接口保持不变，但内部使用模型缓存机制，确保高性能
    """
    model = _get_cached_model(cfg)
    return predict_one_with_model(model, title, report_text)


# ---------------- CLI ----------------
if __name__ == "__main__":

    cfg = load_cfg("pre_struct/ebqa_title/merged_config.json")
    demo_title = "入院记录"
    demo_text = """武溪大学人民医院 湖北省人民醫院 RENMIN HOSPITAL OF WUIAN UNIVERSITY HUBEI GENERAL HOSPITAL 病理补充报告 登记号：0009xx8135 病理号：25-44132 林红女/51岁/2592685  科别：乳腺甲状腺中心Ⅱ科 乳腺甲状腺中心Ⅱ科病 病区： 床号：J41 临床诊断：乳房肿块 送检日期：2025-07-16 送检材料：右乳房肿块*1 病理补充诊断： 穿刺标本： 1.(右乳房肿块)乳腺浸润性癌，非特殊类型，3级(3+3+2)。 2.免疫组化：25-44132-3：浸润性癌：ER(约1%弱+)，PR(约5%弱+)，Her-2(1+),Ki67(约 75%+),GATA-3(部分+)，TRPS1(+),E-Cadherin(胞膜+)，CK5/6(-,示肌上皮缺失)。 汉大学人民医院 审核医生：李杜娟 李杜娟 病理科 此报告供临床医师参考，报告无病理科专用章无效。 报告日期：2025-07-1719：24"""

    # 展示模型缓存效果
    import time
    
    print("=== 演示模型缓存效果 ===")
    
    # 第一次调用（会加载模型）
    print("第一次调用（首次加载模型）...")
    start_time = time.time()
    res1 = predict_one(cfg, demo_title, demo_text)
    first_call_time = time.time() - start_time
    print(f"第一次调用耗时: {first_call_time:.2f}秒")
    print(json.dumps(res1, ensure_ascii=False, indent=2))
    
    # 第二次调用（使用缓存模型）
    print(f"\n第二次调用（使用缓存模型）...")
    start_time = time.time()
    res2 = predict_one(cfg, demo_title, demo_text)
    second_call_time = time.time() - start_time
    print(f"第二次调用耗时: {second_call_time:.2f}秒")
    
    # 性能对比
    if first_call_time > 0:
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        print(f"\n性能提升: {speedup:.1f}x 倍速")
    
    # 验证结果一致性
    print(f"两次调用结果一致: {res1 == res2}")
