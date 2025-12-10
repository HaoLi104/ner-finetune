#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练诊断脚本 - 检查数据和模型配置
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import torch

# 添加项目根目录
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from pre_struct.kv_ner import config_io
from pre_struct.kv_ner.data_utils import load_labelstudio_export, build_bio_label_list
from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
from transformers import AutoTokenizer


def diagnose():
    """诊断训练配置和数据"""
    
    print("="*80)
    print("KV-NER 训练诊断")
    print("="*80)
    
    # 1. 检查配置
    print("\n1. 配置文件检查")
    print("-"*80)
    cfg = config_io.load_config("pre_struct/kv_ner/kv_ner_config.json")
    
    label_map = config_io.label_map_from(cfg)
    print(f"标签映射: {label_map}")
    
    labels = build_bio_label_list(label_map)
    print(f"BIO 标签: {labels}")
    print(f"标签数量: {len(labels)}")
    
    train_block = cfg.get("train", {})
    print(f"\n训练配置:")
    print(f"  batch_size: {train_block.get('train_batch_size')}")
    print(f"  learning_rate: {train_block.get('learning_rate')}")
    print(f"  num_epochs: {train_block.get('num_train_epochs')}")
    print(f"  dropout: {train_block.get('dropout')}")
    print(f"  freeze_encoder: {train_block.get('freeze_encoder')}")
    print(f"  unfreeze_last_n_layers: {train_block.get('unfreeze_last_n_layers')}")
    
    # 2. 检查数据
    print("\n2. 数据检查")
    print("-"*80)
    
    train_path = train_block.get("data_path")
    val_path = train_block.get("val_data_path")
    
    if not train_path or not Path(train_path).exists():
        print(f"❌ 训练数据不存在: {train_path}")
        print("请先运行: python pre_struct/kv_ner/prepare_data.py")
        return
    
    print(f"训练数据: {train_path}")
    train_samples = load_labelstudio_export(Path(train_path), label_map, include_unlabeled=False)
    print(f"  样本数: {len(train_samples)}")
    print(f"  有标注: {sum(1 for s in train_samples if s.has_labels)}")
    
    if val_path and Path(val_path).exists():
        val_samples = load_labelstudio_export(Path(val_path), label_map, include_unlabeled=False)
        print(f"\n验证数据: {val_path}")
        print(f"  样本数: {len(val_samples)}")
        print(f"  有标注: {sum(1 for s in val_samples if s.has_labels)}")
    
    # 3. 检查标注分布
    print("\n3. 标注分布统计")
    print("-"*80)
    
    all_labels = []
    for sample in train_samples[:1000]:  # 取前1000个样本
        for entity in sample.entities:
            all_labels.append(entity.label)
    
    label_counts = Counter(all_labels)
    print("实体类型分布（前1000个样本）:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    # 4. 检查样本长度
    print("\n4. 文本长度统计")
    print("-"*80)
    
    tokenizer = AutoTokenizer.from_pretrained(config_io.tokenizer_name_from(cfg))
    lengths = []
    for sample in train_samples[:500]:
        try:
            tokens = tokenizer.tokenize(sample.text)
            lengths.append(len(tokens))
        except:
            pass
    
    if lengths:
        print(f"样本长度（前500个）:")
        print(f"  平均: {sum(lengths)/len(lengths):.1f} tokens")
        print(f"  最小: {min(lengths)}")
        print(f"  最大: {max(lengths)}")
        print(f"  中位数: {sorted(lengths)[len(lengths)//2]}")
        
        max_seq = cfg.get("max_seq_length", 512)
        truncated = sum(1 for l in lengths if l > max_seq)
        print(f"  超过 max_seq_length({max_seq}): {truncated}/{len(lengths)} ({truncated/len(lengths)*100:.1f}%)")
    
    # 5. 检查模型
    print("\n5. 模型检查")
    print("-"*80)
    
    model_path = Path(train_block.get("output_dir", "runs/kv_ner")) / "best"
    if model_path.exists():
        print(f"模型目录: {model_path}")
        
        try:
            model = BertCrfTokenClassifier.from_pretrained(str(model_path))
            print(f"✅ 模型加载成功")
            print(f"  标签数: {model.num_labels}")
            print(f"  BERT 隐藏层大小: {model.config.hidden_size}")
            
            # 统计可训练参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    else:
        print(f"模型目录不存在: {model_path}")
        print("（首次训练前这是正常的）")
    
    # 6. 潜在问题检查
    print("\n6. 潜在问题检查")
    print("-"*80)
    
    issues = []
    
    if label_counts.get("VALUE", 0) < label_counts.get("KEY", 0) * 0.8:
        issues.append("⚠️  VALUE 实体数量明显少于 KEY，可能导致很多键无值")
    
    if len(train_samples) < 1000:
        issues.append("⚠️  训练样本少于 1000，建议增加数据")
    
    if truncated and truncated / len(lengths) > 0.3:
        issues.append(f"⚠️  超过 30% 的样本被截断，建议增大 max_seq_length")
    
    batch_size = train_block.get('train_batch_size', 16)
    if batch_size < 4:
        issues.append(f"⚠️  batch_size={batch_size} 太小，可能训练不稳定")
    
    lr = train_block.get('learning_rate', 3e-5)
    if lr > 1e-4:
        issues.append(f"⚠️  learning_rate={lr} 可能太大，建议 1e-5 到 5e-5")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ 没有发现明显问题")
    
    # 7. 建议
    print("\n7. 训练建议")
    print("-"*80)
    
    print("数据量优化:")
    if len(train_samples) < 2000:
        print("  - 增加更多标注数据（当前训练集较小）")
    print("  - 确保 KEY-VALUE 关系标注完整")
    
    print("\n超参数建议:")
    print("  - batch_size: 8-16（根据GPU内存）")
    print("  - learning_rate: 2e-5 到 5e-5")
    print("  - num_epochs: 5-10")
    print("  - dropout: 0.1-0.3")
    
    print("\n微调策略:")
    print("  - 数据少（<2000）: unfreeze_last_n_layers=2-4")
    print("  - 数据中（2000-5000）: unfreeze_last_n_layers=null（全量训练）")
    print("  - 数据多（>5000）: unfreeze_last_n_layers=null")
    
    print("\n当前数据量: {} 条".format(len(train_samples)))
    if len(train_samples) < 2000:
        print("  建议: unfreeze_last_n_layers=3")
    elif len(train_samples) < 5000:
        print("  建议: unfreeze_last_n_layers=null（全量训练）")
    else:
        print("  建议: unfreeze_last_n_layers=null（全量训练）")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    diagnose()

