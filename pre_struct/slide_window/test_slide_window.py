# -*- coding: utf-8 -*-
from __future__ import annotations

"""EBQA-style predict entry for slide_window (self-contained).

Provides similar API/naming as pre_struct/ebqa/test_ebqa.py but uses local
implementations. This keeps structure aligned without cross-importing ebqa.
"""

import os
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

import torch

# 支持两种运行方式：作为包或直接运行
try:
    from .config_io import (
        load_config,
        resolve_model_dir,
        resolve_tokenizer_name,
        resolve_report_struct_path,
        lengths_from,
        chunk_mode_from,
        predict_block,
    )
    from .model_ebqa import EBQAModel
    from .dataset import EnhancedQADataset, QACollator
except ImportError:
    # 直接运行时的导入方式
    sys.path.insert(0, str(Path(__file__).parent))
    from config_io import (
        load_config,
        resolve_model_dir,
        resolve_tokenizer_name,
        resolve_report_struct_path,
        lengths_from,
        chunk_mode_from,
        predict_block,
    )
    from model_ebqa import EBQAModel
    from dataset import EnhancedQADataset, QACollator


@dataclass
class PredictConfig:
    report_struct_path: str
    tokenizer_name: str
    model_dir: str
    max_seq_len: int
    max_tokens_ctx: int
    doc_stride: int
    max_answer_len: int
    batch_size: int
    chunk_mode: str
    enable_no_answer: bool = True
    null_threshold: float = 0.0
    null_agg: str = "mean"


def load_ebqa() -> tuple[EBQAModel, QACollator, str, PredictConfig]:
    cfg_path = os.environ.get("EBQA_CONFIG_PATH")
    cfgd = load_config(cfg_path) if cfg_path else load_config()
    lens = lengths_from(cfgd)
    pred_blk = predict_block(cfgd)
    
    # 尝试使用训练后的模型，如果不存在则使用原始预训练模型
    model_dir_use = resolve_model_dir(cfgd)
    if not Path(model_dir_use).exists():
        print(f"⚠️  训练后的模型不存在: {model_dir_use}")
        print(f"   使用原始预训练模型: {cfgd['model_name_or_path']}")
        model_dir_use = cfgd.get("model_name_or_path", model_dir_use)
    
    cfg = PredictConfig(
        report_struct_path=resolve_report_struct_path(cfgd),
        tokenizer_name=resolve_tokenizer_name(cfgd),
        model_dir=model_dir_use,
        max_seq_len=lens["max_seq_len"],
        max_tokens_ctx=lens["max_tokens_ctx"],
        doc_stride=lens["doc_stride"],
        max_answer_len=lens["max_answer_len"],
        batch_size=int(pred_blk["batch_size"]),
        chunk_mode=chunk_mode_from(cfgd),
        enable_no_answer=bool(pred_blk.get("enable_no_answer", True)),
        null_threshold=float(pred_blk.get("null_threshold", 0.0)),
        null_agg=str(pred_blk.get("null_agg", "mean")),
    )

    model = EBQAModel(
        model_name_or_path=cfg.model_dir,
        tokenizer_name_or_path=cfg.tokenizer_name,
        per_device_eval_batch_size=cfg.batch_size,
        fp16=torch.cuda.is_available(),
        max_answer_len=cfg.max_answer_len,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collate = QACollator(pad_id=getattr(model.tokenizer, "pad_token_id", 0) or 0)
    return model, collate, device, cfg


def predict_one(cfg: PredictConfig, model: EBQAModel, collate: QACollator, report_title: str, report_text: str) -> Dict[str, Any]:
    # build single-record dataset
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmpf:
        json.dump([{"report_title": report_title, "report": report_text}], tmpf, ensure_ascii=False)
        data_path = tmpf.name

    ds = EnhancedQADataset(
        data_path=data_path,
        report_struct_path=cfg.report_struct_path,
        tokenizer_name=cfg.tokenizer_name,
        max_seq_len=cfg.max_seq_len,
        max_tokens_ctx=cfg.max_tokens_ctx,
        doc_stride=cfg.doc_stride,
        inference_mode=True,
        keep_debug_fields=True,
    )

    preds = model.predict(
        dataset=ds,
        data_collator=collate,
        batch_size=cfg.batch_size,
        enable_no_answer=cfg.enable_no_answer,
        null_threshold=cfg.null_threshold,
        null_agg=cfg.null_agg,
    )
    try:
        os.unlink(data_path)
    except Exception:
        pass

    out: Dict[str, Dict[str, Any]] = {}
    full = report_text or ""
    for it in preds:
        k = str(it.get("question_key") or "")
        s = int(it.get("start_char", -1))
        e = int(it.get("end_char", -1))
        t = str(it.get("text", "") or "")
        if (not t) and s >= 0 and e > s and full:
            try:
                t = full[s:e]
            except Exception:
                t = ""
        out[k] = {"text": t, "start": s, "end": e}
    return out


if __name__ == "__main__":
    model, collate, device, cfg = load_ebqa()
    demo_title = "出院记录"
    demo_text = "姓名：李四 性别：女 年龄：30岁\n报告日期：2024-05-01\n出院诊断：胃炎。"
    res = predict_one(cfg, model, collate, demo_title, demo_text)
    print(json.dumps(res, ensure_ascii=False, indent=2))
