#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算 alias/title QA 样本（使用 merged.converted.json 的 alias 字段）
"""

import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pre_struct.ebqa_title.da_core.dataset import EnhancedQADataset, QACollator

# ==================== 配置 ====================
ENABLE_QUICK_TEST = False  # 改为False做完整预计算
QUICK_TEST_SIZE = 500


def _save_jsonl(samples, path):
    """保存为JSONL格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    # ===== 配置区域 =====
    INPUT_JSON = "data/merged.converted.json"
    OUTPUT_JSONL = INPUT_JSON.replace(".json", ".alias_title.jsonl")
    
    # 快速测试模式：只处理前N条记录
    if ENABLE_QUICK_TEST:
        print(f"⚡ 快速测试模式：只处理前{QUICK_TEST_SIZE}条记录")
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        test_data = all_data[:QUICK_TEST_SIZE]
        INPUT_JSON = "data/.tmp_quick_test.json"
        with open(INPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
        OUTPUT_JSONL = INPUT_JSON.replace(".json", ".alias_title.jsonl")
        print(f"✓ 临时文件: {INPUT_JSON}")
    
    print("=" * 80)
    print("🚀 预计算 Alias/Title QA 数据集")
    print("=" * 80)
    print(f"输入: {INPUT_JSON}")
    print(f"输出: {OUTPUT_JSONL}")
    print()
    
    # 只从 ebqa_title 的 merged_config.json 读取参数（不再使用任何默认值）
    config_path = "pre_struct/ebqa_title/merged_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件必须是 JSON 对象")

    tokenizer_path = cfg.get("tokenizer_name_or_path")
    qtpl = ((cfg.get("train", {}) or {}).get("question_template") or "").strip()
    if not tokenizer_path:
        raise KeyError("tokenizer_name_or_path 缺失，请在 merged_config.json 中配置")
    if not qtpl:
        raise KeyError("train.question_template 缺失，请在 merged_config.json 中配置")

    print(f"✅ Tokenizer: {tokenizer_path}")
    print(f"✅ 问题模板: {qtpl}")
    print()
    
    # 构建数据集
    print("⏳ 正在构建数据集...")
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
        negative_downsample=1.0,  # 保留所有负样本（已通过字段分配控制数量）
        chunk_mode="budget",
        seed=42,
        autobuild=True,
        show_progress=True,
        # Alias/Title 特有参数
        alias_field="alias",
        question_template=qtpl,
        use_concurrent_build=False,   # 改回串行模式
        max_workers=None              # 单线程
    )
    
    print()
    print("⏳ 保存预计算样本...")
    _save_jsonl(ds.samples, OUTPUT_JSONL)
    
    print()
    print("=" * 80)
    print("✅ 预计算完成!")
    print("=" * 80)
    print(f"输入记录: {len(ds.records)}")
    print(f"输出样本: {len(ds.samples)}")
    
    # 统计
    pos_count = sum(1 for s in ds.samples if s.get('start_positions', 0) != 0)
    neg_count = len(ds.samples) - pos_count
    
    print(f"正样本: {pos_count} ({pos_count/len(ds.samples)*100:.1f}%)")
    print(f"负样本: {neg_count} ({neg_count/len(ds.samples)*100:.1f}%)")
    
    # 显示几个样本的问题示例
    print()
    print("📝 问题示例（前5个不同的问题）：")
    seen_questions = set()
    count = 0
    for sample in ds.samples:
        if 'chunk_text' in sample:
            # 从 chunk_text 中提取问题（在 [CLS] 和 [SEP] 之间）
            key = sample.get('question_key', '')
            if key and key not in seen_questions:
                seen_questions.add(key)
                # 通过 alias 映射获取问题
                rec_idx = sample.get('report_index', 0)
                if rec_idx < len(ds.records):
                    rec = ds.records[rec_idx]
                    question = ds._format_question(key, rec)
                    print(f"   {count+1}. {question}")
                    count += 1
                    if count >= 5:
                        break
    print()


if __name__ == "__main__":
    main()
