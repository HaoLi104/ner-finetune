#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长文本分块处理模块

处理超过 max_seq_length 的文本：
1. 将文本分成多个重叠的 chunk
2. 每个 chunk 独立预测
3. 合并预测结果
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 450,
    overlap: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    按 token 数量分块文本
    
    Args:
        text: 原始文本
        tokenizer: 分词器
        chunk_size: 每个 chunk 的 token 数
        overlap: chunk 之间的重叠 token 数
    
    Returns:
        [(chunk_text, char_start, char_end), ...]
    """
    # 先 tokenize 整个文本，获取 offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    
    if len(tokens) <= chunk_size:
        # 文本足够短，不需要分块
        return [(text, 0, len(text))]
    
    chunks = []
    stride = chunk_size - overlap
    
    i = 0
    while i < len(tokens):
        # 确定当前 chunk 的 token 范围
        chunk_start_token = i
        chunk_end_token = min(i + chunk_size, len(tokens))
        
        # 获取字符级别的起始和结束位置
        char_start = offsets[chunk_start_token][0]
        char_end = offsets[chunk_end_token - 1][1] if chunk_end_token > 0 else len(text)
        
        # 提取 chunk 文本
        chunk_text = text[char_start:char_end]
        chunks.append((chunk_text, char_start, char_end))
        
        # 如果已经到末尾，退出
        if chunk_end_token >= len(tokens):
            break
        
        i += stride
    
    logger.info(f"文本分块: {len(text)} 字符 → {len(chunks)} 个 chunk")
    return chunks


def merge_entities(
    chunk_entities: List[List[Dict[str, Any]]],
    chunk_offsets: List[Tuple[int, int]],
    overlap: int = 50,
    merge_gap: int = 2,
) -> List[Dict[str, Any]]:
    """
    合并多个 chunk 的实体预测结果
    
    Args:
        chunk_entities: 每个 chunk 的实体列表
        chunk_offsets: 每个 chunk 的起始位置 [(char_start, char_end), ...]
        overlap: 重叠区域大小（用于去重）
    
    Returns:
        合并后的实体列表
    """
    if len(chunk_entities) == 1:
        return chunk_entities[0]
    
    all_entities = []
    seen_spans = set()  # 用于去重：(type, start, end)
    
    for chunk_idx, entities in enumerate(chunk_entities):
        chunk_start, chunk_end = chunk_offsets[chunk_idx]
        
        for entity in entities:
            # 调整 offset（chunk 内的相对位置 → 全文的绝对位置）
            global_start = chunk_start + entity["start"]
            global_end = chunk_start + entity["end"]
            
            # 去重：如果在重叠区域，只保留第一次出现的
            entity_key = (entity["type"], global_start, global_end)
            if entity_key in seen_spans:
                continue
            
            seen_spans.add(entity_key)
            all_entities.append({
                "type": entity["type"],
                "start": global_start,
                "end": global_end,
                "text": entity["text"],
            })
    
    # 按位置排序
    all_entities.sort(key=lambda x: (x["start"], x["end"]))

    # 进一步对同类实体做重叠/相邻合并，避免跨chunk断裂
    merged = []
    for ent in all_entities:
        if not merged:
            merged.append(ent)
            continue
        last = merged[-1]
        if ent["type"] == last["type"] and ent["start"] <= last["end"] + int(merge_gap):
            # overlap or small gap - 合并边界
            new_end = max(last["end"], ent["end"])
            merged[-1] = {
                "type": last["type"],
                "start": last["start"],
                "end": new_end,
                "text": "",  # 将在后续从原文重新提取，避免使用片段文本
            }
        else:
            merged.append(ent)
    return merged


def predict_with_chunking(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device,
    max_seq_length: int = 512,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
    merge_adjacent_gap: int = 2,
) -> List[Dict[str, Any]]:
    """
    对长文本进行分块预测
    
    Args:
        text: 原始文本
        model: 模型
        tokenizer: 分词器
        id2label: ID到标签的映射
        device: 设备
        max_seq_length: 最大序列长度
        chunk_size: chunk 大小（tokens）
        chunk_overlap: chunk 重叠（tokens）
    
    Returns:
        合并后的实体列表
    """
    from pre_struct.kv_ner.metrics import char_spans
    
    # 分块
    chunks = chunk_text_by_tokens(text, tokenizer, chunk_size, chunk_overlap)
    
    if len(chunks) == 1:
        # 单个 chunk，直接预测
        chunk_text, _, _ = chunks[0]
        return _predict_single_chunk(
            chunk_text, model, tokenizer, id2label, device, max_seq_length
        )
    
    # 多个 chunk，逐个预测后合并
    chunk_entities = []
    chunk_offsets = []
    
    for chunk_text, char_start, char_end in chunks:
        entities = _predict_single_chunk(
            chunk_text, model, tokenizer, id2label, device, max_seq_length
        )
        chunk_entities.append(entities)
        chunk_offsets.append((char_start, char_end))
    
    # 合并结果
    merged_entities = merge_entities(chunk_entities, chunk_offsets, chunk_overlap, merge_adjacent_gap)

    # 使用全文根据最终边界重切文本，避免保留chunk片段的text造成边界/文本不一致
    for ent in merged_entities:
        s, e = int(ent.get("start", -1)), int(ent.get("end", -1))
        if 0 <= s < e <= len(text):
            # 不使用 strip()，保留原始边界的文本
            ent["text"] = text[s:e]
    
    return merged_entities


def _predict_single_chunk(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device,
    max_seq_length: int,
) -> List[Dict[str, Any]]:
    """预测单个 chunk"""
    import torch
    from pre_struct.kv_ner.metrics import char_spans
    
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    # Predict
    with torch.no_grad():
        predictions = model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    
    # Decode
    pred_labels = predictions[0] if isinstance(predictions[0], list) else predictions[0].tolist()
    mask = attention_mask[0].bool().tolist()
    
    # Extract spans
    spans = char_spans(pred_labels, mask, offset_mapping, id2label)
    
    # Convert to entity dicts
    entities = []
    for ent_type, start, end in spans:
        if start < 0 or end > len(text) or start >= end:
            continue
        entities.append({
            "type": ent_type,
            "start": start,
            "end": end,
            # 不使用 strip()，保留原始边界的文本
            "text": text[start:end],
        })
    
    return entities
