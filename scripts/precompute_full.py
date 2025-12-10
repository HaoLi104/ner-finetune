#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对完整的project-1.converted.json进行预计算，支持多文件合并和报告类型替换
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pre_struct.ebqa.da_core.dataset import EnhancedQADataset



def _save_json(data, path):
    """保存为JSON格式"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_jsonl(samples, path):
    """保存为JSONL格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def merge_and_replace_report_types(
    base_file: str,
    replacement_files: Dict[str, str],
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """合并多个数据文件，并按报告类型替换
    
    Args:
        base_file: 基础文件路径（主训练文件）
        replacement_files: {报告类型: 文件路径} 的映射
            例如: {"入院记录": "data/ruyuanjilu/ruyuan-2025-10-16.converted.json"}
        output_file: 合并后的输出文件路径（可选）
    
    Returns:
        合并后的记录列表
    
    Example:
        merged = merge_and_replace_report_types(
            base_file="data/project-1.converted.json",
            replacement_files={
                "入院记录": "data/ruyuanjilu/ruyuan-2025-10-16.converted.json",
                "出院记录": "data/other/chuyuan.converted.json",
            },
            output_file="data/merged.converted.json"
        )
    """
    # 读取基础文件
    print(f"读取基础文件: {base_file}")
    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    base_by_type = {}
    for rec in base_data:
        report_type = rec.get("report_title", "")
        if report_type not in base_by_type:
            base_by_type[report_type] = []
        base_by_type[report_type].append(rec)
    
    print(f"  基础文件包含 {len(base_data)} 条记录")
    print(f"  报告类型分布:")
    for rt, recs in sorted(base_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {rt}: {len(recs)} 条")
    
    # 读取替换文件
    replacement_data = {}
    for report_type, file_path in replacement_files.items():
        print(f"\n读取替换文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 只提取指定报告类型的记录
        filtered = [rec for rec in data if rec.get("report_title", "") == report_type]
        replacement_data[report_type] = filtered
        
        print(f"  总记录: {len(data)}")
        print(f"  {report_type}: {len(filtered)} 条")
    
    # 合并：用replacement替换base中的相应类型
    print(f"\n合并策略:")
    merged_by_type = {}
    
    for report_type, recs in base_by_type.items():
        if report_type in replacement_data:
            # 替换
            merged_by_type[report_type] = replacement_data[report_type]
            print(f"  {report_type}: 使用替换文件 ({len(base_by_type[report_type])} -> {len(replacement_data[report_type])})")
        else:
            # 保留base中的
            merged_by_type[report_type] = recs
            print(f"  {report_type}: 保留基础文件 ({len(recs)})")
    
    # 添加replacement中有但base中没有的类型
    for report_type, recs in replacement_data.items():
        if report_type not in merged_by_type:
            merged_by_type[report_type] = recs
            print(f"  {report_type}: 新增类型 ({len(recs)})")
    
    # 合并为列表
    merged_records = []
    for report_type in sorted(merged_by_type.keys()):
        merged_records.extend(merged_by_type[report_type])
    
    print(f"\n合并结果:")
    print(f"  总记录数: {len(merged_records)}")
    print(f"  报告类型分布:")
    for rt in sorted(merged_by_type.keys(), key=lambda x: len(merged_by_type[x]), reverse=True):
        print(f"    {rt}: {len(merged_by_type[rt])} 条")
    
    # 保存到文件（如果指定）
    if output_file:
        print(f"\n保存合并结果到: {output_file}")
        _save_json(merged_records, output_file)
    
    return merged_records

def main():
    # ===== 配置区域 =====
    # 是否启用合并和替换功能
    USE_MERGE = True
    
    if USE_MERGE:
        # 合并模式：用A文件中的报告类型替换B文件中的报告类型
        BASE_FILE = "data/project-1.converted.json"
        REPLACEMENT_FILES = {
            "入院记录": "data/ruyuanjilu/ruyuan-2025-10-16.converted.json",
            # 可以继续添加其他报告类型的替换
            # "出院记录": "data/other/chuyuan.converted.json",
        }
        MERGED_FILE = "data/merged.converted.json"
        INPUT_JSON = MERGED_FILE
    else:
        # 直接模式：不合并，直接预计算
        INPUT_JSON = "data/project-1.converted.json"
    
    OUTPUT_JSONL = INPUT_JSON.replace(".json", ".jsonl")
    
    print("=" * 80)
    print("🚀 预计算完整数据集")
    print("=" * 80)
    
    # 如果启用合并，先执行合并
    if USE_MERGE:
        print("\n📁 步骤1: 合并和替换报告类型")
        print("=" * 80)
        merged_data = merge_and_replace_report_types(
            base_file=BASE_FILE,
            replacement_files=REPLACEMENT_FILES,
            output_file=MERGED_FILE,
        )
        print(f"\n✓ 合并完成，保存到: {MERGED_FILE}")
        print("=" * 80)
    
    print(f"\n📊 步骤2: 预计算样本")
    print("=" * 80)
    print(f"输入: {INPUT_JSON}")
    print(f"输出: {OUTPUT_JSONL}")
    print()
    
    # 从配置获取tokenizer
    with open("pre_struct/ebqa/ebqa_config.json", 'r') as f:
        cfg = json.load(f)
        tokenizer_path = cfg.get("tokenizer_name_or_path")
    
    print(f"✅ Tokenizer: {tokenizer_path}")
    print()
    
    # 构建数据集
    print("⏳ 正在构建数据集（串行模式）...")
    ds = EnhancedQADataset(
        data_path=INPUT_JSON,
        tokenizer_name=tokenizer_path,
        max_seq_len=512,
        max_tokens_ctx=500,
        max_answer_len=512,
        use_question_templates=True,
        keep_debug_fields=True,
        report_struct_path="keys/keys_merged.json",
        only_title_keys=True,
        inference_mode=False,
        dynamic_answer_length=True,
        negative_downsample=0.2,  # 使用最优配置
        chunk_mode="budget",
        seed=42,
        autobuild=True,
        show_progress=True,
        use_concurrent_build=False,  # 串行，避免卡死
        max_workers=None,
    )
    
    print()
    print("⏳ 保存预计算样本...")
    _save_jsonl(ds.samples, OUTPUT_JSONL)
    
    print()
    print("=" * 70)
    print("✅ 预计算完成!")
    print("=" * 70)
    print(f"输入记录: {len(ds.records)}")
    print(f"输出样本: {len(ds.samples)}")
    
    # 统计
    pos_count = sum(1 for s in ds.samples if s.get('start_positions', 0) != 0)
    neg_count = len(ds.samples) - pos_count
    
    print(f"正样本: {pos_count} ({pos_count/len(ds.samples)*100:.1f}%)")
    print(f"负样本: {neg_count} ({neg_count/len(ds.samples)*100:.1f}%)")
    print()

if __name__ == "__main__":
    main()

