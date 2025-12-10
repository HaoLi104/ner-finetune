# -*- coding: utf-8 -*-
"""使用预计算数据进行训练

这个脚本可以使用预计算的数据集进行训练，也可以动态构建数据集。
通过配置文件中的 'precomputed' 字段控制。

使用方法：
    python -m pre_struct.slide_window.train_with_precomputed
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)

from .config_io import load_config, train_block
from .dataset import EnhancedQADataset, PrecomputedQADataset, QACollator


@dataclass
class TrainConfig:
    data_path: str
    report_struct_path: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    output_dir: str
    use_precomputed: bool = False
    eval_data_path: str = ""
    max_seq_len: int = 512
    max_tokens_ctx: int = 480
    doc_stride: int = 128
    max_answer_len: int = 128
    per_device_batch_size: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_ratio: float = 0.1


def build_datasets_from_precomputed(
    train_path: str,
    eval_path: str,
    keep_debug_fields: bool = False,
):
    """从预计算文件加载数据集"""
    print("\n使用预计算数据集:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {eval_path}")
    
    ds_train = PrecomputedQADataset(
        precomputed_path=train_path,
        keep_debug_fields=keep_debug_fields,
        require_labels=True,  # 训练时必须有标签
    )
    
    ds_eval = PrecomputedQADataset(
        precomputed_path=eval_path,
        keep_debug_fields=keep_debug_fields,
        require_labels=True,  # 训练时必须有标签
    )
    
    return ds_train, ds_eval


def build_datasets_dynamically(cfg: TrainConfig):
    """动态构建数据集（传统方式）"""
    print("\n动态构建数据集:")
    print(f"  数据路径: {cfg.data_path}")
    
    ds = EnhancedQADataset(
        data_path=cfg.data_path,
        report_struct_path=cfg.report_struct_path,
        tokenizer_name=cfg.tokenizer_name_or_path,
        max_seq_len=cfg.max_seq_len,
        max_tokens_ctx=cfg.max_tokens_ctx,
        doc_stride=cfg.doc_stride,
        inference_mode=False,  # 训练模式，构建标签
        keep_debug_fields=False,
        autobuild=True,
    )
    
    # 简单分割
    n = len(ds)
    n_eval = max(1, int(cfg.eval_ratio * n))
    
    ds_train = type(ds).__new__(type(ds))
    ds_eval = type(ds).__new__(type(ds))
    for d in (ds_train, ds_eval):
        d.__dict__ = dict(ds.__dict__)
    
    ds_train.samples = ds.samples[:-n_eval]
    ds_eval.samples = ds.samples[-n_eval:]
    
    print(f"  训练样本: {len(ds_train)}")
    print(f"  验证样本: {len(ds_eval)}")
    
    return ds_train, ds_eval


def main() -> None:
    print("=" * 70)
    print("EBQA 训练脚本（支持预计算数据）")
    print("=" * 70)
    
    # 读取配置
    cfg_path = os.environ.get("EBQA_CONFIG_PATH")
    print(f"\n加载配置: {cfg_path or '(默认)'}")
    cfgd: Dict[str, Any] = load_config(cfg_path) if cfg_path else load_config()
    tb = train_block(cfgd)
    
    # 构建训练配置
    tcfg = TrainConfig(
        data_path=str(tb.get("data_path") or tb.get("train_path") or "data/train.json"),
        report_struct_path=str(cfgd["report_struct_path"]),
        model_name_or_path=str(cfgd["model_name_or_path"]),
        tokenizer_name_or_path=str(cfgd.get("tokenizer_name_or_path", cfgd["model_name_or_path"])),
        output_dir=str(cfgd.get("output_dir", "runs/slide_window")),
        use_precomputed=bool(tb.get("precomputed", False)),
        eval_data_path=str(tb.get("eval_data_path", "")),
        max_seq_len=int(cfgd.get("max_seq_len", 512)),
        max_tokens_ctx=int(cfgd.get("max_tokens_ctx", 480)),
        doc_stride=int(cfgd.get("doc_stride", max(64, int(cfgd.get("max_tokens_ctx", 480)) // 4))),
        max_answer_len=int(cfgd.get("max_answer_len", 128)),
        per_device_batch_size=int(tb.get("per_device_batch_size", 8)),
        num_train_epochs=int(tb.get("num_train_epochs", 2)),
        learning_rate=float(tb.get("learning_rate", 2e-5)),
        weight_decay=float(tb.get("weight_decay", 0.0)),
        warmup_ratio=float(tb.get("warmup_ratio", 0.1)),
        logging_steps=int(tb.get("logging_steps", 50)),
        eval_ratio=float(tb.get("eval_ratio", 0.1)),
    )
    
    print("\n训练配置:")
    print(f"  模型: {tcfg.model_name_or_path}")
    print(f"  输出目录: {tcfg.output_dir}")
    print(f"  使用预计算数据: {tcfg.use_precomputed}")
    print(f"  批次大小: {tcfg.per_device_batch_size}")
    print(f"  训练轮数: {tcfg.num_train_epochs}")
    print(f"  学习率: {tcfg.learning_rate}")
    
    os.makedirs(tcfg.output_dir, exist_ok=True)
    
    # 加载 tokenizer 和模型
    print("\n加载模型和 tokenizer...")
    tok = BertTokenizerFast.from_pretrained(tcfg.tokenizer_name_or_path)
    model = BertForQuestionAnswering.from_pretrained(tcfg.model_name_or_path)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 构建数据集
    if tcfg.use_precomputed:
        # 使用预计算数据
        if not tcfg.eval_data_path:
            # 自动推断验证集路径
            train_precomp_path = Path(tcfg.data_path)
            if "train" in train_precomp_path.name:
                eval_precomp_path = train_precomp_path.parent / train_precomp_path.name.replace("train", "eval")
            else:
                raise ValueError(
                    "使用预计算数据时，请在配置中指定 'eval_data_path'，"
                    "或者训练数据路径包含 'train' 字样以便自动推断"
                )
        else:
            eval_precomp_path = Path(tcfg.eval_data_path)
        
        ds_train, ds_eval = build_datasets_from_precomputed(
            train_path=tcfg.data_path,
            eval_path=str(eval_precomp_path),
            keep_debug_fields=False,
        )
    else:
        # 动态构建数据集
        ds_train, ds_eval = build_datasets_dynamically(tcfg)
    
    # Collator
    collate = QACollator(
        pad_id=getattr(tok, "pad_token_id", 0) or 0,
        keep_debug_fields=False,
    )
    
    # 训练参数
    args = TrainingArguments(
        output_dir=tcfg.output_dir,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        num_train_epochs=tcfg.num_train_epochs,
        per_device_train_batch_size=tcfg.per_device_batch_size,
        per_device_eval_batch_size=tcfg.per_device_batch_size,
        eval_strategy="epoch",  # 使用 eval_strategy 而不是 evaluation_strategy
        logging_steps=tcfg.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        warmup_ratio=tcfg.warmup_ratio,
        report_to=[],
    )
    
    def compute_metrics(eval_pred):
        # 简单的指标计算
        return {}
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model(tcfg.output_dir)
    
    # 保存训练配置
    config_save_path = Path(tcfg.output_dir) / "train_config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name_or_path": tcfg.tokenizer_name_or_path,
                "model_name_or_path": tcfg.model_name_or_path,
                "max_answer_len": tcfg.max_answer_len,
                "chunk_mode": cfgd.get("chunk_mode", "budget"),
                "max_seq_len": tcfg.max_seq_len,
                "max_tokens_ctx": tcfg.max_tokens_ctx,
                "doc_stride": tcfg.doc_stride,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    print(f"\n训练完成！模型已保存至: {tcfg.output_dir}")
    print(f"训练配置已保存至: {config_save_path}")


if __name__ == "__main__":
    main()

