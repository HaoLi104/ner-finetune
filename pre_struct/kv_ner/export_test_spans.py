#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Label Studio 格式导出评估用的 ground_truth.jsonl

将 Label Studio 的 NER 标注（KEY/VALUE + 关系）转换为键值对格式：
{
    "report": "文本",
    "report_title": "标题",
    "spans": {
        "姓名": {"start": 0, "end": 2, "text": "张三"},
        "年龄": {"start": 10, "end": 12, "text": "25"}
    }
}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# 添加项目根目录到路径
if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import load_labelstudio_export
else:
    from . import config_io
    from .data_utils import load_labelstudio_export

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def extract_key_value_spans(sample) -> Dict[str, Dict[str, Any]]:
    """
    从 Sample 中提取键值对 spans
    
    Args:
        sample: data_utils.Sample 对象
    
    Returns:
        {key_text: {start, end, text}} 格式的字典
    """
    # 收集所有 KEY 和 VALUE 实体
    keys = [e for e in sample.entities if e.label == "KEY"]
    values = [e for e in sample.entities if e.label == "VALUE"]
    
    # 构建关系映射：from_id -> to_id
    relations_map = {}
    for rel in sample.relations:
        relations_map[rel.from_id] = rel.to_id
    
    # 收集 HOSPITAL 实体
    hospitals = [e for e in sample.entities if e.label == "HOSPITAL"]
    
    # 收集所有键值对用于排序
    all_kvs = []  # [(key_text, start, end, text, is_hospital, sort_pos)]
    
    for key_ent in keys:
        key_text = key_ent.text or sample.text[key_ent.start:key_ent.end]
        if not key_text.strip():
            continue
        
        # 查找与这个 KEY 关联的 VALUE
        if key_ent.result_id and key_ent.result_id in relations_map:
            value_id = relations_map[key_ent.result_id]
            # 找到对应的 VALUE 实体
            value_ent = next((v for v in values if v.result_id == value_id), None)
            
            if value_ent:
                value_text = value_ent.text or sample.text[value_ent.start:value_ent.end]
                all_kvs.append((
                    key_text.strip(),
                    value_ent.start,
                    value_ent.end,
                    value_text.strip(),
                    False,
                    key_ent.start,
                ))
        else:
            # 没有关联VALUE的KEY也输出（VALUE为空）
            all_kvs.append((
                key_text.strip(),
                key_ent.end,
                key_ent.end,
                "",
                False,
                key_ent.start,
            ))
    
    # 添加HOSPITAL实体（用"医院名称"作为key）
    for hospital_ent in hospitals:
        hospital_text = hospital_ent.text or sample.text[hospital_ent.start:hospital_ent.end]
        if hospital_text.strip():
            all_kvs.append((
                "医院名称",
                hospital_ent.start,
                hospital_ent.end,
                hospital_text.strip(),
                True,
                hospital_ent.start,
            ))
    
    # 排序：医院名称在最前，其他按start位置排序
    all_kvs.sort(key=lambda x: (0 if x[4] else 1, x[5]))
    
    # 按顺序构建result（Python 3.7+字典保持插入顺序）
    result: Dict[str, Dict[str, Any]] = {}
    for key_text, start, end, text_content, _, _ in all_kvs:
        result[key_text] = {
            "start": start,
            "end": end,
            "text": text_content,
        }
    
    return result


def export_test_spans(
    input_paths: List[str],
    output_path: str,
    config_path: Optional[str] = None,
    show_progress: bool = True,
) -> None:
    """
    从 Label Studio 导出评估用的 ground_truth.jsonl
    
    Args:
        input_paths: Label Studio JSON 文件路径列表（可以是多个）
        output_path: 输出 JSONL 文件路径
        config_path: 配置文件路径（用于读取 label_map）
        show_progress: 是否显示进度条
    """
    # 读取配置
    if config_path:
        cfg = config_io.load_config(config_path)
        label_map = config_io.label_map_from(cfg)
    else:
        # 默认标签映射
        label_map = {
            "键名": "KEY",
            "值": "VALUE",
            "医院名称": "HOSPITAL",
        }
    
    logger.info(f"加载 Label Studio 数据: {len(input_paths)} 个文件")
    all_samples = []
    for input_path in input_paths:
        logger.info(f"  - {input_path}")
        samples = load_labelstudio_export(input_path, label_map)
        all_samples.extend(samples)
    logger.info(f"总样本数: {len(all_samples)}")
    
    results: List[Dict[str, Any]] = []
    
    iterator = tqdm(all_samples, desc="导出 spans") if show_progress else all_samples
    
    for sample in iterator:
        if not sample.text.strip():
            continue
        
        # 提取键值对
        spans = extract_key_value_spans(sample)
        
        # 跳过没有键值对的样本
        if not spans:
            continue
        
        results.append({
            "report_index": sample.task_id,
            "report_title": sample.title,
            "report": sample.text,
            "spans": spans,
        })
    
    # 保存 JSONL
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path_obj.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"[OK] 导出 {len(results)} 条记录 -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 KV-NER 评估数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个文件
  python export_test_spans.py --input data/file1.json
  
  # 多个文件
  python export_test_spans.py --input data/file1.json data/file2.json
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="输入 Label Studio JSON 文件（可以指定多个）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ground_truth.jsonl",
        help="输出 JSONL 文件",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（用于读取 label_map）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_test_spans(
        input_paths=args.input,
        output_path=args.output,
        config_path=args.config,
        show_progress=True,
    )

