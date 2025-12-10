# -*- coding: utf-8 -*-
"""数据分割和预计算脚本

功能：
1. 读取原始数据并按比例分割为训练集和验证集
2. 使用 EnhancedQADataset 预计算所有样本
3. 保存预计算结果，加速后续训练

使用方法：
    python -m pre_struct.slide_window.precompute_dataset
    或
    EBQA_CONFIG_PATH=path/to/config.json python -m pre_struct.slide_struct.slide_window.precompute_dataset
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)")

from .config_io import load_config, train_block
from .dataset import EnhancedQADataset


def split_data(
    data: List[Dict[str, Any]],
    eval_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """将数据分割为训练集和验证集
    
    Args:
        data: 原始数据列表
        eval_ratio: 验证集比例
        seed: 随机种子
        shuffle: 是否打乱数据
        
    Returns:
        (train_data, eval_data) 元组
    """
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)
    
    n_total = len(data)
    n_eval = max(1, int(n_total * eval_ratio))
    n_train = n_total - n_eval
    
    train_data = data[:n_train]
    eval_data = data[n_train:]
    
    print(f"数据分割完成:")
    print(f"  总样本数: {n_total}")
    print(f"  训练集: {n_train} ({100 * n_train / n_total:.1f}%)")
    print(f"  验证集: {n_eval} ({100 * n_eval / n_total:.1f}%)")
    
    return train_data, eval_data


def save_split_data(
    train_data: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    output_dir: str,
    train_filename: str = "train.json",
    eval_filename: str = "eval.json",
) -> tuple[str, str]:
    """保存分割后的数据
    
    Args:
        train_data: 训练数据
        eval_data: 验证数据
        output_dir: 输出目录
        train_filename: 训练集文件名
        eval_filename: 验证集文件名
        
    Returns:
        (train_path, eval_path) 文件路径元组
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / train_filename
    eval_path = output_dir / eval_filename
    
    # 保存训练集
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存: {train_path}")
    
    # 保存验证集
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"验证集已保存: {eval_path}")
    
    return str(train_path), str(eval_path)


def precompute_and_save_dataset(
    data_path: str,
    report_struct_path: str,
    tokenizer_name: str,
    output_file: str,
    max_seq_len: int = 512,
    max_tokens_ctx: int = 480,
    doc_stride: int = 128,
    inference_mode: bool = False,
    show_progress: bool = True,
) -> int:
    """预计算数据集并保存
    
    Args:
        data_path: 数据文件路径
        report_struct_path: 报告结构配置路径
        tokenizer_name: tokenizer 路径
        output_file: 输出文件路径
        max_seq_len: 最大序列长度
        max_tokens_ctx: 最大上下文token数
        doc_stride: 滑动窗口步长
        inference_mode: 是否为推理模式（训练模式会构建标签）
        show_progress: 是否显示进度
        
    Returns:
        预计算的样本数量
    """
    print(f"\n开始预计算数据集: {data_path}")
    print(f"  推理模式: {inference_mode}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  max_tokens_ctx: {max_tokens_ctx}")
    print(f"  doc_stride: {doc_stride}")
    
    # 构建数据集
    dataset = EnhancedQADataset(
        data_path=data_path,
        report_struct_path=report_struct_path,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        max_tokens_ctx=max_tokens_ctx,
        doc_stride=doc_stride,
        inference_mode=inference_mode,
        keep_debug_fields=True,
        autobuild=True,
        show_progress=True,  # 始终显示调试信息
    )
    
    n_samples = len(dataset.samples)
    print(f"预计算完成: {n_samples} 个样本")
    
    # 保存预计算的样本
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为jsonl格式（每行一个样本，便于大规模数据处理）
    print(f"正在保存 {n_samples} 个样本...")
    with open(output_path, "w", encoding="utf-8") as f:
        if HAS_TQDM:
            # 使用进度条
            for sample in tqdm(dataset.samples, desc="保存样本", unit="样本"):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            # 无进度条，显示每10%的进度
            for i, sample in enumerate(dataset.samples):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if (i + 1) % max(1, n_samples // 10) == 0:
                    progress = (i + 1) / n_samples * 100
                    print(f"  保存进度: {progress:.0f}% ({i + 1}/{n_samples})")
    
    print(f"预计算数据已保存: {output_path}")
    
    # 保存元数据
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "n_samples": n_samples,
        "n_records": len(dataset.records),
        "max_seq_len": max_seq_len,
        "max_tokens_ctx": max_tokens_ctx,
        "doc_stride": doc_stride,
        "inference_mode": inference_mode,
        "tokenizer_name": tokenizer_name,
        "data_path": data_path,
        "report_struct_path": report_struct_path,
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {meta_path}")
    
    return n_samples


def main():
    """主函数"""
    print("=" * 60)
    print("数据分割和预计算脚本")
    print("=" * 60)
    
    # 读取配置
    cfg_path = os.environ.get("EBQA_CONFIG_PATH")
    print(f"\n加载配置文件: {cfg_path or '(默认)'}")
    cfg = load_config(cfg_path)
    tb = train_block(cfg)
    
    # 提取配置参数
    # 预计算脚本应该读取原始数据，不是训练配置中的data_path
    data_path = str(cfg.get("precompute_input_path", "data/merged.converted.json"))
    report_struct_path = str(cfg["report_struct_path"])
    tokenizer_name = str(cfg.get("tokenizer_name_or_path", cfg["model_name_or_path"]))
    output_dir = str(cfg.get("output_dir", "runs/slide_window"))
    
    max_seq_len = int(cfg.get("max_seq_len", 512))
    max_tokens_ctx = int(cfg.get("max_tokens_ctx", 480))
    doc_stride = int(cfg.get("doc_stride", max(64, max_tokens_ctx // 4)))
    
    eval_ratio = float(tb.get("eval_ratio", 0.1))
    seed = int(tb.get("seed", 42))
    
    # 创建预计算输出目录
    precompute_dir = Path(output_dir) / "precomputed"
    precompute_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  原始数据路径: {data_path}")
    print(f"  报告结构配置: {report_struct_path}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  输出目录: {output_dir}")
    print(f"  预计算目录: {precompute_dir}")
    print(f"  验证集比例: {eval_ratio}")
    print(f"  随机种子: {seed}")
    
    # 步骤1: 加载原始数据
    print("\n" + "=" * 60)
    print("步骤 1/3: 加载原始数据")
    print("=" * 60)
    
    print(f"读取文件: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        # 支持 json 和 jsonl 格式
        content = f.read().strip()
        if content.startswith("["):
            # JSON 数组格式
            data = json.loads(content)
        else:
            # JSONL 格式
            lines = content.split("\n")
            if HAS_TQDM:
                data = [json.loads(line) for line in tqdm(lines, desc="解析JSONL") if line.strip()]
            else:
                data = [json.loads(line) for line in lines if line.strip()]
    
    if not isinstance(data, list):
        data = [data]
    
    print(f"✓ 成功加载 {len(data)} 条原始数据")
    
    # 步骤2: 分割数据
    print("\n" + "=" * 60)
    print("步骤 2/3: 分割数据集")
    print("=" * 60)
    
    train_data, eval_data = split_data(data, eval_ratio=eval_ratio, seed=seed, shuffle=True)
    
    # 保存分割后的数据
    split_dir = precompute_dir / "split"
    train_split_path, eval_split_path = save_split_data(
        train_data,
        eval_data,
        str(split_dir),
        train_filename="train.json",
        eval_filename="eval.json",
    )
    
    # 步骤3: 预计算数据集
    print("\n" + "=" * 60)
    print("步骤 3/3: 预计算数据集")
    print("=" * 60)
    
    # 预计算训练集（训练模式，构建标签）
    n_train_samples = precompute_and_save_dataset(
        data_path=train_split_path,
        report_struct_path=report_struct_path,
        tokenizer_name=tokenizer_name,
        output_file=str(precompute_dir / "train_precomputed.jsonl"),
        max_seq_len=max_seq_len,
        max_tokens_ctx=max_tokens_ctx,
        doc_stride=doc_stride,
        inference_mode=False,  # 训练模式
        show_progress=True,
    )
    
    # 预计算验证集（训练模式，构建标签）
    n_eval_samples = precompute_and_save_dataset(
        data_path=eval_split_path,
        report_struct_path=report_struct_path,
        tokenizer_name=tokenizer_name,
        output_file=str(precompute_dir / "eval_precomputed.jsonl"),
        max_seq_len=max_seq_len,
        max_tokens_ctx=max_tokens_ctx,
        doc_stride=doc_stride,
        inference_mode=False,  # 训练模式
        show_progress=True,
    )
    
    # 完成
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n数据分割结果:")
    print(f"  训练集原始数据: {len(train_data)} 条 → {train_split_path}")
    print(f"  验证集原始数据: {len(eval_data)} 条 → {eval_split_path}")
    print(f"\n预计算结果:")
    print(f"  训练集样本: {n_train_samples} 个 → {precompute_dir / 'train_precomputed.jsonl'}")
    print(f"  验证集样本: {n_eval_samples} 个 → {precompute_dir / 'eval_precomputed.jsonl'}")
    print(f"\n后续使用:")
    print(f"  1. 在配置文件中设置 'precomputed': true")
    print(f"  2. 更新 'data_path' 指向预计算文件")
    print(f"  3. 运行训练脚本即可使用预计算数据")
    

if __name__ == "__main__":
    main()

