#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
try:
    # Preferred in newer PyTorch
    from torch.amp import autocast as amp_autocast  # type: ignore
except Exception:  # pragma: no cover
    amp_autocast = None  # type: ignore
try:
    # Backward-compatible fallback
    from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler  # type: ignore
except Exception:  # pragma: no cover
    cuda_autocast = None  # type: ignore
    CudaGradScaler = None  # type: ignore
try:
    # New GradScaler API
    from torch.amp import GradScaler as AmpGradScaler  # type: ignore
except Exception:  # pragma: no cover
    AmpGradScaler = None  # type: ignore
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

if __package__ in (None, ""):
    # Allow running as a script: python pre_struct/kv_ner/train.py
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import (
        build_bio_label_list,
        load_labelstudio_export,
        split_samples,
    )
    from pre_struct.kv_ner.dataset import TokenClassificationDataset, collate_batch
    from pre_struct.kv_ner.metrics import compute_ner_metrics
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
else:
    from . import config_io
    from .data_utils import (
        build_bio_label_list,
        load_labelstudio_export,
        split_samples,
    )
    from .dataset import TokenClassificationDataset, collate_batch
    from .metrics import compute_ner_metrics
    from .modeling import BertCrfTokenClassifier

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _merge_optional_dicts(
    *candidates: Tuple[str, Any]
) -> Dict[str, Any]:
    """Merge optional dicts while validating their types."""
    merged: Dict[str, Any] = {}
    for label, candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, dict):
            raise ValueError(f"{label} must be a dict, got {type(candidate).__name__}")
        merged.update(candidate)
    return merged


def _build_tokenizer(
    name_or_path: str,
    extra_kwargs: Dict[str, Any],
) -> AutoTokenizer:
    """Load a fast tokenizer with helpful fallbacks/errors."""
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_dataloaders(
    cfg: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    train_samples,
    val_samples,
    test_samples,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    max_len = config_io.max_seq_length(cfg)
    label_all_tokens = config_io.label_all_tokens(cfg)
    chunk_size = int(cfg.get("chunk_size", max_len))
    if chunk_size > max_len:
        chunk_size = max_len
    chunk_overlap = int(cfg.get("chunk_overlap", 0))

    train_dataset = TokenClassificationDataset(
        train_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
    )
    val_dataset = TokenClassificationDataset(
        val_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
    )
    test_dataset = TokenClassificationDataset(
        test_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
    )
    train_block = config_io.ensure_block(cfg, "train")
    batch_size = int(train_block.get("train_batch_size", 16))
    eval_batch_size = int(train_block.get("eval_batch_size", batch_size))
    num_workers = int(train_block.get("num_workers", 0))
    pin_memory = bool(train_block.get("pin_memory", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    return train_loader, eval_loader, test_loader


def _evaluate_model(
    model: BertCrfTokenClassifier,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    model.eval()
    predictions: List[List[int]] = []
    references: List[List[int]] = []
    masks: List[List[bool]] = []
    offsets: List[List[Tuple[int, int]]] = []

    o_id = next((idx for idx, name in id2label.items() if name == "O"), 0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_type_ids = batch.token_type_ids.to(device)
            labels = batch.labels.to(device)

            decoded = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            seq_len = labels.size(1)
            for i, seq in enumerate(decoded):
                seq_list = list(seq)
                if len(seq_list) < seq_len:
                    seq_list = seq_list + [o_id] * (seq_len - len(seq_list))
                elif len(seq_list) > seq_len:
                    seq_list = seq_list[:seq_len]
                predictions.append(seq_list)
            references.extend(labels.cpu().tolist())
            masks.extend(attention_mask.cpu().tolist())
            offsets.extend(batch.offset_mapping.cpu().tolist())

    return compute_ner_metrics(predictions, references, masks, id2label, offsets=offsets)


def train(args: argparse.Namespace) -> None:
    cfg = config_io.load_config(args.config)
    train_block = config_io.ensure_block(cfg, "train")

    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    data_path = Path(train_block["data_path"])
    set_seed(int(train_block.get("seed", 42)))

    tokenizer_kwargs = _merge_optional_dicts(
        ("config.tokenizer_kwargs", cfg.get("tokenizer_kwargs")),
        ("config.tokenizer_load_kwargs", cfg.get("tokenizer_load_kwargs")),
        ("train.tokenizer_kwargs", train_block.get("tokenizer_kwargs")),
    )
    tokenizer_name = config_io.tokenizer_name_from(cfg)
    tokenizer = _build_tokenizer(tokenizer_name, tokenizer_kwargs)
    if tokenizer_kwargs:
        logger.info(
            "Tokenizer 额外参数键: %s",
            ", ".join(sorted(tokenizer_kwargs.keys())),
        )
    logger.info(
        "Loaded tokenizer: %s (fast=%s)",
        tokenizer.name_or_path,
        getattr(tokenizer, "is_fast", False),
    )

    # 读取训练集
    train_samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
    logger.info(f"训练集: {data_path} ({len(train_samples)} 条)")
    
    # 检查是否有单独的验证集/测试集文件（固定划分优先）
    val_path = train_block.get("val_data_path")
    test_path = train_block.get("test_data_path")

    if val_path and test_path:
        # 固定验证集与测试集（若文件不存在则回退为动态划分）
        val_p = Path(val_path)
        test_p = Path(test_path)
        if val_p.is_file() and test_p.is_file():
            val_samples = load_labelstudio_export(val_p, label_map, include_unlabeled=False)
            test_samples = load_labelstudio_export(test_p, label_map, include_unlabeled=False)
            logger.info(
                "使用固定验证/测试集合: val=%d, test=%d", len(val_samples), len(test_samples)
            )
        else:
            logger.warning(
                "指定了固定验证/测试路径，但文件不存在，将回退为动态划分: val=%s(exists=%s), test=%s(exists=%s)",
                val_p, val_p.is_file(), test_p, test_p.is_file()
            )
            # 回退为动态划分（保留可用的 val_path），立即按 val.json 动态划分
            val_pool = load_labelstudio_export(val_p, label_map, include_unlabeled=False)
            logger.info(f"验证集池: {val_path} ({len(val_pool)} 条)")
            test_split_ratio = float(train_block.get("test_split_ratio", 0.5))
            val_pool_labeled = [s for s in val_pool if s.has_labels]
            val_samples, test_samples = train_test_split(
                val_pool_labeled,
                test_size=test_split_ratio,
                random_state=int(train_block.get("seed", 42)),
                shuffle=True,
            )
            logger.info(
                "从验证集池动态划分: val=%d (%.1f%%), test=%d (%.1f%%)",
                len(val_samples),
                (1 - test_split_ratio) * 100,
                len(test_samples),
                test_split_ratio * 100,
            )
    elif val_path:
        # 读取验证集，并从中划分出测试集（向后兼容）
        val_pool = load_labelstudio_export(Path(val_path), label_map, include_unlabeled=False)
        logger.info(f"验证集池: {val_path} ({len(val_pool)} 条)")

        # 从验证集池中划分验证集和测试集（默认 50:50）
        test_split_ratio = float(train_block.get("test_split_ratio", 0.5))

        # 过滤有标注的样本
        val_pool_labeled = [s for s in val_pool if s.has_labels]

        val_samples, test_samples = train_test_split(
            val_pool_labeled,
            test_size=test_split_ratio,
            random_state=int(train_block.get("seed", 42)),
            shuffle=True,
        )

        logger.info(
            "从验证集池动态划分: val=%d (%.1f%%), test=%d (%.1f%%)",
            len(val_samples),
            (1 - test_split_ratio) * 100,
            len(test_samples),
            test_split_ratio * 100,
        )
    else:
        # 向后兼容：自动划分（不推荐）
        logger.warning("未配置 val_data_path，将从训练数据自动划分（不推荐）")
        logger.warning("建议先运行 prepare_data.py 生成已划分的数据")
        
        samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
        train_ratio_compat = float(train_block.get("train_ratio", 0.8))
        val_ratio_compat = float(train_block.get("val_ratio", 0.1))
        train_samples, val_samples, test_samples = split_samples(
            samples,
            train_ratio=train_ratio_compat,
            val_ratio=val_ratio_compat,
            seed=int(train_block.get("seed", 42)),
        )
        logger.info(
            "Dataset split: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, eval_loader, test_loader = _prepare_dataloaders(
        cfg,
        tokenizer,
        label2id,
        train_samples,
        val_samples,
        test_samples,
    )

    backbone_kwargs = _merge_optional_dicts(
        ("config.model_kwargs", cfg.get("model_kwargs")),
        ("config.model_load_kwargs", cfg.get("model_load_kwargs")),
        ("config.hf_model_kwargs", cfg.get("hf_model_kwargs")),
        ("train.model_kwargs", train_block.get("model_kwargs")),
    )
    if backbone_kwargs:
        logger.info(
            "Backbone 额外加载参数键: %s",
            ", ".join(sorted(backbone_kwargs.keys())),
        )

    model = BertCrfTokenClassifier(
        model_name_or_path=config_io.model_name_from(cfg),
        label2id=label2id,
        id2label=id2label,
        backbone_kwargs=backbone_kwargs,
        dropout=float(train_block.get("dropout", 0.1)),
        freeze_encoder=bool(train_block.get("freeze_encoder", False)),
        unfreeze_last_n_layers=train_block.get("unfreeze_last_n_layers"),
        use_bilstm=bool(train_block.get("use_bilstm", False)),
        lstm_hidden_size=train_block.get("lstm_hidden_size"),
        lstm_num_layers=int(train_block.get("lstm_num_layers", 1)),
        lstm_dropout=float(train_block.get("lstm_dropout", 0.0)),
        lstm_bidirectional=bool(train_block.get("lstm_bidirectional", True)),
        # conv residual removed
        boundary_loss_weight=float(train_block.get("boundary_loss_weight", 0.0)),
        boundary_positive_weight=float(train_block.get("boundary_positive_weight", 1.0)),
        include_hospital_boundary=bool(train_block.get("include_hospital_boundary", True)),
        token_ce_loss_weight=float(train_block.get("token_ce_loss_weight", 0.0)),
        token_ce_label_smoothing=float(train_block.get("token_ce_label_smoothing", 0.0)),
        boundary_ce_label_smoothing=float(train_block.get("boundary_ce_label_smoothing", 0.0)),
        token_ce_value_class_weight=float(train_block.get("token_ce_value_class_weight", 3.0)),
        end_boundary_loss_weight=float(train_block.get("end_boundary_loss_weight", 0.0)),
        end_boundary_positive_weight=float(train_block.get("end_boundary_positive_weight", 1.0)),
    ).to(device)

    # Optional gradient checkpointing for encoder
    if bool(train_block.get("grad_checkpointing", train_block.get("gradient_checkpointing", False))):
        try:
            if hasattr(model, 'bert') and hasattr(model.bert, 'gradient_checkpointing_enable'):
                model.bert.gradient_checkpointing_enable()
            if hasattr(model, 'bert') and hasattr(model.bert, 'config'):
                if hasattr(model.bert.config, 'gradient_checkpointing'):
                    model.bert.config.gradient_checkpointing = True
                if hasattr(model.bert.config, 'use_cache'):
                    try:
                        model.bert.config.use_cache = False
                    except Exception:
                        pass
            logger.info("Enabled gradient checkpointing for encoder")
        except Exception:
            logger.warning("Failed to enable gradient checkpointing; continue without it")

    # Optimizer & scheduler
    lr = float(train_block.get("learning_rate", 3e-5))
    weight_decay = float(train_block.get("weight_decay", 0.01))
    # Optimizer with optional discriminative learning rates
    # If 'head_learning_rate' or 'encoder_learning_rate' are provided in config,
    # use separate LRs for encoder (BERT) and newly initialized heads (BiLSTM/CRF/classifiers).
    enc_lr = float(train_block.get("encoder_learning_rate", lr))
    head_lr = float(train_block.get("head_learning_rate", enc_lr * 5.0))

    if ("encoder_learning_rate" in train_block) or ("head_learning_rate" in train_block):
        bert_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(p)
            else:
                head_params.append(p)
        if not bert_params and head_params:
            # Fallback to single group if classification-only (unlikely here)
            optimizer = AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
        elif bert_params and not head_params:
            optimizer = AdamW(bert_params, lr=enc_lr, weight_decay=weight_decay)
        else:
            optimizer = AdamW(
                [
                    {"params": bert_params, "lr": enc_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                lr=enc_lr,
                weight_decay=weight_decay,
            )
        logger.info(
            "Optimizer param groups: encoder_lr=%.2e, head_lr=%.2e (weight_decay=%.3g)",
            enc_lr, head_lr, weight_decay,
        )
    else:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    num_epochs = int(train_block.get("num_train_epochs", 5))
    grad_accum = int(train_block.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_block.get("max_grad_norm", 1.0))
    total_steps = num_epochs * len(train_loader) // max(1, grad_accum)
    warmup_ratio = float(train_block.get("warmup_ratio", 0.1))
    warmup_steps_cfg = train_block.get("warmup_steps")
    warmup_steps = int(warmup_steps_cfg) if isinstance(warmup_steps_cfg, int) else int(total_steps * warmup_ratio)
    # 学习率调度器选择
    scheduler_name = str(train_block.get("lr_scheduler", train_block.get("scheduler", "linear"))).lower()
    if scheduler_name in {"cos", "cosine"}:
        num_cycles = float(train_block.get("num_cycles", 0.5))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )
        logger.info("LR scheduler: cosine (cycles=%.3g, warmup=%d/%d)", num_cycles, warmup_steps, total_steps)
    elif scheduler_name in {"cosine_restarts", "cosine_with_restarts", "cos_restarts"}:
        num_cycles = int(train_block.get("num_cycles", 2))
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )
        logger.info("LR scheduler: cosine_with_restarts (cycles=%d, warmup=%d/%d)", num_cycles, warmup_steps, total_steps)
    elif scheduler_name in {"constant_with_warmup", "constant"}:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
        logger.info("LR scheduler: constant_with_warmup (warmup=%d/%d)", warmup_steps, total_steps)
    elif scheduler_name in {"poly", "polynomial"}:
        power = float(train_block.get("poly_power", 1.0))
        lr_end = float(train_block.get("lr_end", 0.0))
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            lr_end=lr_end,
            power=power,
        )
        logger.info("LR scheduler: polynomial (power=%.3g, lr_end=%.2e, warmup=%d/%d)", power, lr_end, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info("LR scheduler: linear (warmup=%d/%d)", warmup_steps, total_steps)

    output_dir = Path(train_block.get("output_dir", "runs/kv_ner"))
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    global_step = 0
    history: List[Dict[str, float]] = []

    # AMP settings
    use_amp = bool(train_block.get("use_amp", train_block.get("amp", False))) and torch.cuda.is_available() and ((amp_autocast is not None) or (cuda_autocast is not None))
    amp_dtype_str = str(train_block.get("amp_dtype", "bf16")).lower()
    bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)()) if torch.cuda.is_available() else False
    if amp_dtype_str == 'bf16' and bf16_supported:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    if use_amp:
        if AmpGradScaler is not None:
            scaler = AmpGradScaler("cuda", enabled=True)
        elif CudaGradScaler is not None:
            scaler = CudaGradScaler(enabled=True)
        else:
            scaler = None
    else:
        scaler = None

    # 创建外层 epoch 进度条
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        running_tokens = 0  # 有效 token 总数（attention_mask==1）
        optimizer.zero_grad(set_to_none=True)
        
        # 创建 batch 进度条
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        
        for step, batch in enumerate(batch_pbar, start=1):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_type_ids = batch.token_type_ids.to(device)
            labels = batch.labels.to(device)

            # Forward with optional AMP
            if use_amp and (amp_autocast is not None or cuda_autocast is not None):
                amp_ctx = (amp_autocast("cuda", dtype=amp_dtype) if amp_autocast is not None else cuda_autocast(dtype=amp_dtype))
            else:
                amp_ctx = nullcontext()
            with amp_ctx:
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
            loss = loss / grad_accum
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()
            # 统计有效 token 数（用于 loss/tok）
            if bool(train_block.get("log_per_token_loss", True)):
                try:
                    running_tokens += int(attention_mask.int().sum().item())
                except Exception:
                    pass

            if step % grad_accum == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            # 更新 batch 进度条显示当前损失和学习率
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
            avg_batch_loss = running_loss * grad_accum / max(1, step)
            postfix = {
                'loss': f'{avg_batch_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            }
            if bool(train_block.get("log_per_token_loss", True)):
                try:
                    # 本 step 的近似 loss/tok：用当前平均损失按 token 归一
                    # 注意：训练实际优化仍基于 batch-mean；这里仅用于显示
                    step_tokens = int(attention_mask.int().sum().item())
                    if step_tokens > 0:
                        step_loss_tok = (loss.item() * grad_accum) / step_tokens
                        postfix['loss_tok'] = f'{step_loss_tok:.6f}'
                except Exception:
                    pass
            batch_pbar.set_postfix(postfix)

        avg_loss = running_loss * grad_accum / max(1, len(train_loader))
        if bool(train_block.get("log_per_token_loss", True)) and running_tokens > 0:
            avg_loss_tok = (running_loss * grad_accum) / running_tokens
            logger.info("Epoch %d/%d - train loss: %.4f, loss/tok: %.6f (tokens=%d)",
                        epoch, num_epochs, avg_loss, avg_loss_tok, running_tokens)
        else:
            logger.info("Epoch %d/%d - train loss: %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate_model(model, eval_loader, device, id2label)
        overall_f1 = metrics["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_f1": overall_f1})
        # 额外打印类别级 F1，便于观察 VALUE 边界收敛
        logger.info("Validation F1: %.4f (KEY=%.4f, VALUE=%.4f, HOSPITAL=%.4f)",
                    overall_f1,
                    metrics.get('KEY', {}).get('f1', 0.0),
                    metrics.get('VALUE', {}).get('f1', 0.0),
                    metrics.get('HOSPITAL', {}).get('f1', 0.0))

        if overall_f1 > best_f1:
            best_f1 = overall_f1
            model.save_pretrained(best_dir)
            (best_dir / "tokenizer").mkdir(exist_ok=True)
            tokenizer.save_pretrained(best_dir / "tokenizer")
            logger.info("Saved new best model to %s (F1=%.4f)", best_dir, best_f1)
        
        # 更新 epoch 进度条显示最佳 F1
        postfix_epoch = {
            'best_f1': f'{best_f1:.4f}',
            'val_f1': f'{overall_f1:.4f}',
            'loss': f'{avg_loss:.4f}'
        }
        if bool(train_block.get("log_per_token_loss", True)) and running_tokens > 0:
            postfix_epoch['loss_tok'] = f'{((running_loss * grad_accum) / running_tokens):.6f}'
        epoch_pbar.set_postfix(postfix_epoch)

    # Final evaluation on the test split
    best_model = BertCrfTokenClassifier.from_pretrained(best_dir).to(device)
    test_metrics = _evaluate_model(best_model, test_loader, device, id2label)

    # Persist the effective configuration for reference
    (output_dir / "used_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "best_val_f1": best_f1,
        "history": history,
        "test_metrics": test_metrics,
        "config_path": args.config,
        "model_dir": str(best_dir),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_task_ids": [s.task_id for s in train_samples],
        "val_task_ids": [s.task_id for s in val_samples],
        "test_task_ids": [s.task_id for s in test_samples],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Training finished. Test F1: %.4f", test_metrics["overall"]["f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KV-NER BERT+CRF model.")
    parser.add_argument(
        "--config",
        type=str,
        default=config_io.default_config_path(),
        help="Path to kv_ner_config.json",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Override chunk_size in config (<= max_seq_length).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=None,
        help="Override chunk_overlap in config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow overriding chunk params without editing JSON
    if args.chunk_size is not None or args.chunk_overlap is not None:
        try:
            cfg_path = args.config
            cfg = config_io.load_config(cfg_path)
            if args.chunk_size is not None:
                cfg["chunk_size"] = int(args.chunk_size)
            if args.chunk_overlap is not None:
                cfg["chunk_overlap"] = int(args.chunk_overlap)
            # Write a temporary in-memory override by passing a modified args object
            # We won't persist to disk; the train() will reload from args.config, so we
            # inject a small shim by temporarily dumping to a sidecar path.
            from pathlib import Path
            tmp_cfg_path = Path(cfg_path).with_suffix(".override.tmp.json")
            tmp_cfg_path.write_text(__import__("json").dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            args.config = str(tmp_cfg_path)
        except Exception:
            pass
    train(args)
