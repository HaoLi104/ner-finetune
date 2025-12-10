#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Label Studio 项目导出转换为 clean_ocr 格式，并智能分段长文本
"""
import sys
sys.path.append(".")

from pre_struct.ebqa.da_core.utils import convert_labelstudio_project_to_clean_records

# 配置路径
IN_PATH = "data/project-1-at-2025-10-13-09-18-782e09b9.json"
OUT_PATH = "data/project-1.converted.json"

# 转换参数
MAX_REPORT_TOKENS = 512  # report 超过此长度时自动分段
TOKENIZER_NAME = None    # 可选：指定 tokenizer 路径以精确计算 token

# 尝试从配置获取 tokenizer
try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH
    TOKENIZER_NAME = DEFAULT_TOKENIZER_PATH
    print(f"✓ 使用 tokenizer: {TOKENIZER_NAME}")
except Exception:
    print("⚠ 未找到 tokenizer，使用字符数估算 token")

print(f"\n=== Label Studio 项目转换（含智能分段） ===")
print(f"输入: {IN_PATH}")
print(f"输出: {OUT_PATH}")
print(f"分段阈值: {MAX_REPORT_TOKENS} tokens")
print()

# 执行转换
records = convert_labelstudio_project_to_clean_records(
    in_path=IN_PATH,
    out_path=OUT_PATH,
    max_report_tokens=MAX_REPORT_TOKENS,
    tokenizer_name=TOKENIZER_NAME,
)

print(f"✅ 转换完成")
print(f"   记录数: {len(records)}")
print(f"   输出: {OUT_PATH}")

# 统计分段情况
segmented_count = sum(1 for r in records if "\n\n" in r.get("report", ""))
print(f"   已分段: {segmented_count} / {len(records)} ({segmented_count/len(records)*100:.1f}%)")

# 显示示例
if records:
    print(f"\n📄 示例记录:")
    sample = records[0]
    print(f"   标题: {sample.get('report_title', 'N/A')}")
    print(f"   字段数: {len([k for k in sample.keys() if k not in ('report', 'report_title', 'added_keys')])}")
    report = sample.get("report", "")
    print(f"   report 长度: {len(report)} 字符")
    if "\n\n" in report:
        segments = report.split("\n\n")
        print(f"   已分段: {len(segments)} 段")
    else:
        print(f"   未分段（长度适中）")

