#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER 数据准备脚本

从 Label Studio 导出的原始数据准备训练和评估数据：
1. 读取 Label Studio 导出的 JSON（包含 NER 标注和关系）
2. 不做任何键的过滤，保留所有有标注的数据
3. 按比例划分训练集/验证集/测试集
4. 保存训练数据（Label Studio 格式）
5. 导出评估数据（JSONL 键值对格式）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# 添加项目根目录到路径
if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
else:
    from . import config_io

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_key_aliases(keys_file: str = "keys/keys_merged.json") -> Dict[str, Dict[str, List[str]]]:
    """加载键名别名映射
    
    Args:
        keys_file: keys映射文件路径
    
    Returns:
        {report_type: {key_name: [aliases]}}
    """
    keys_path = Path(keys_file)
    if not keys_path.exists():
        logger.warning(f"键名映射文件不存在: {keys_file}")
        return {}
    
    try:
        # 尝试用更宽松的方式读取
        with keys_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取别名映射
        alias_map = {}
        total_aliases = 0
        for report_type, keys_info in data.items():
            alias_map[report_type] = {}
            for key_name, key_info in keys_info.items():
                aliases = key_info.get('别名', [])
                if aliases:
                    alias_map[report_type][key_name] = aliases
                    total_aliases += len(aliases)
        
        logger.info(f"✅ 加载键名别名映射: {keys_file}")
        logger.info(f"   报告类型数: {len(alias_map)}")
        logger.info(f"   总别名数: {total_aliases}")
        return alias_map
    except Exception as e:
        logger.warning(f"⚠️  加载键名映射失败: {e}")
        logger.warning(f"   将使用空映射（不影响Label Studio数据）")
        return {}


def read_extracted_kv_data(
    input_path: Path,
    exclude_titles: Optional[List[str]] = None,
    keys_alias_map: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> List[Dict[str, Any]]:
    """读取已提取的键值对格式数据（如 clean_ocr_ppt_da_v4_7_recheck.json）
    
    Args:
        input_path: 已提取的JSON文件路径
        exclude_titles: 要排除的报告类型列表
        keys_alias_map: 键名别名映射（用于提高定位准确性，KEY和VALUE都支持别名）
    
    Returns:
        转换为Label Studio格式的任务列表（只保留KEY和VALUE都能完整定位的）
    """
    if not input_path.exists():
        logger.warning(f"文件不存在: {input_path}")
        return []
    
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.warning(f"格式错误: {input_path} (必须是 JSON 数组)")
            return []
        
        logger.info(f"读取已提取数据: {input_path.name} ({len(data)} 条)")
        
        # 筛选：排除指定的report_title
        if exclude_titles:
            filtered_data = [
                d for d in data 
                if d.get('report_title', '') not in exclude_titles
            ]
            logger.info(f"  排除 {exclude_titles} 后: {len(filtered_data)} 条")
        else:
            filtered_data = data
        
        # 不再抽样，保留所有能定位的数据
        # (抽样逻辑已移除，保留所有filtered_data)
        
        # 转换为Label Studio格式
        tasks = []
        located_count = 0
        total_kvs = 0
        
        # 获取该报告类型的别名映射（如果有）
        report_type_aliases = {}
        
        for idx, item in enumerate(filtered_data):
            report = item.get('report', '')
            report_title = item.get('report_title', '')
            
            if not report.strip():
                continue
            
            # 获取该报告类型的键别名映射
            if keys_alias_map and report_title in keys_alias_map:
                report_type_aliases = keys_alias_map[report_title]
            
            # 提取所有键值对字段
            meta_fields = {'__idx', 'report_title', 'report', 'added_keys'}
            
            # 第一阶段：检查所有KV能否定位，收集需要从report中删除的内容
            all_kvs_info = []  # 收集能成功定位的KV信息
            segments_to_remove = []  # 需要从report中删除的片段 [(start, end, reason)]
            all_kvs_count = 0
            
            # 获取该样本的别名映射（如果有meta.alias2canonical）
            alias_to_canonical = {}
            canonical_to_alias = {}
            if 'meta' in item and isinstance(item['meta'], dict):
                alias_to_canonical = item['meta'].get('alias2canonical', {})
                # 反向映射：标准名 → 别名
                canonical_to_alias = {v: k for k, v in alias_to_canonical.items()}
            
            for k, v in item.items():
                if k not in meta_fields and v and k != 'meta':
                    # 字段名映射：统一命名
                    original_key = k
                    if k == '医院名':
                        k = '医院名称'  # 统一为"医院名称"
                    
                    # 【重要】医院名称是特殊处理，不参与key替换
                    # 医院名称应该作为HOSPITAL标签，保持原值
                    is_hospital = (k == '医院名称')
                    
                    # 如果JSON中是标准名，但report中可能用别名，优先用别名查找
                    key_text_to_search = k
                    if k in canonical_to_alias and not is_hospital:
                        # 这个标准key在report中有对应的别名（医院名除外）
                        key_text_to_search = canonical_to_alias[k]
                    
                    key_text = k  # JSON中的标准名
                    value_str = str(v)
                    total_kvs += 1
                    
                    # 1. 查找KEY在report中的位置
                    # 优先使用meta中的别名（如果有），否则用标准名
                    key_pos = -1
                    key_end = -1
                    found_key_text = key_text_to_search  # 使用别名或标准名
                    
                    # 收集所有候选KEY（包括别名），按长度降序排列（优先匹配更长的）
                    candidate_keys = [key_text_to_search]
                    if original_key in report_type_aliases:
                        candidate_keys.extend(report_type_aliases[original_key])
                    # 去重并按长度降序排列（长的优先，避免"病理诊断"误匹配"术后病理诊断"）
                    candidate_keys = sorted(set(candidate_keys), key=lambda x: -len(x))
                    
                    # 尝试找"KEY："或"KEY:"的模式
                    for candidate in candidate_keys:
                        if key_pos >= 0:
                            break
                        for separator in ['：', ':']:
                            pattern = candidate + separator
                            pos = report.find(pattern)
                            if pos >= 0:
                                key_pos = pos
                                key_end = pos + len(candidate)
                                found_key_text = candidate
                                break
                    
                    # 如果没找到，尝试直接查找（仍按长度优先）
                    if key_pos < 0:
                        for candidate in candidate_keys:
                            pos = report.find(candidate)
                            if pos >= 0:
                                key_pos = pos
                                key_end = pos + len(candidate)
                                found_key_text = candidate
                                break
                    
                    # 2. 查找VALUE在report中的位置
                    value_pos = report.find(value_str)
                    value_end = -1
                    found_value_text = value_str
                    
                    # 如果找到了，从report中提取实际文本
                    if value_pos >= 0:
                        value_end = value_pos + len(value_str)
                        found_value_text = report[value_pos:value_end]
                    else:
                        # 如果找不到，尝试用别名查找
                        if original_key in report_type_aliases:
                            aliases = report_type_aliases[original_key]
                            for alias in aliases:
                                alias_pos = report.find(alias)
                                if alias_pos >= 0:
                                    # 用别名找到了，从report中提取实际文本
                                    value_pos = alias_pos
                                    value_end = alias_pos + len(alias)
                                    found_value_text = report[value_pos:value_end]
                                    break
                    
                    all_kvs_count += 1
                    
                    # 检查KEY和VALUE是否都能定位
                    if key_pos >= 0 and value_pos >= 0:
                        # 记录能定位的KV信息
                        # 关键：用found_key_text和found_value_text（report中实际的文本）
                        # 这样就实现了"用report中的实际文本替换JSON中的值"
                        # 【例外】医院名称不替换，保持原值
                        final_key_text = key_text if is_hospital else found_key_text
                        all_kvs_info.append({
                            'key_text': final_key_text,  # ← 医院名称保持原值，其他key用report中的文本
                            'original_key': original_key,
                            'canonical_key': key_text,  # 保存原始的标准名
                            'value': v,
                            'key_pos': key_pos,
                            'key_end': key_end,
                            'found_key_text': found_key_text,
                            'value_pos': value_pos,
                            'value_end': value_end,
                            'found_value_text': found_value_text,
                        })
                    else:
                        # KV无法完整定位
                        # 如果KEY能找到，标记需要删除这部分内容
                        if key_pos >= 0:
                            # KEY找到了但VALUE没找到，需要删除KEY及其后面的内容
                            # 删除到下一个可能的分隔符（换行、句号、分号等）
                            segments_to_remove.append({
                                'key_text': key_text,
                                'key_pos': key_pos,
                                'reason': 'value_not_found'
                            })
            
            # 策略判断：
            # 1. 如果没有任何KV能定位 → 删除整个样本
            # 2. 如果有无法定位的KV → 删除整个样本（保证数据一致性）
            # 3. 只保留100%KV都能定位的样本
            if not all_kvs_info or len(all_kvs_info) < all_kvs_count:
                # 有KV无法定位，删除整个样本
                continue
            
            # 生成NER标注（只为能定位的KV生成）
            kv_fields = {}
            ner_annotations = []
            
            for kv_info in all_kvs_info:
                key_text = kv_info['key_text']
                kv_fields[key_text] = kv_info['value']
                located_count += 1
                
                # 判断是否是HOSPITAL类型（医院名称）
                if key_text == '医院名称':
                    # HOSPITAL实体：只标注VALUE部分
                    ner_annotations.append({
                        "type": "labels",
                        "id": f"{idx}_{len(ner_annotations)}",
                        "value": {
                            "start": kv_info['value_pos'],
                            "end": kv_info['value_end'],
                            "text": kv_info['found_value_text'],
                            "labels": ["医院名称"]  # HOSPITAL标注
                        }
                    })
                else:
                    # 普通KEY-VALUE对：标注KEY和VALUE
                    # KEY标注（使用在report中实际找到的文本）
                    ner_annotations.append({
                        "type": "labels",
                        "id": f"{idx}_{len(ner_annotations)}",
                        "value": {
                            "start": kv_info['key_pos'],
                            "end": kv_info['key_end'],
                            "text": kv_info['found_key_text'],  # 使用实际找到的文本（可能是别名）
                            "labels": ["键名"]  # KEY标注
                        }
                    })
                    
                    # VALUE标注
                    ner_annotations.append({
                        "type": "labels",
                        "id": f"{idx}_{len(ner_annotations)}",
                        "value": {
                            "start": kv_info['value_pos'],
                            "end": kv_info['value_end'],
                            "text": kv_info['found_value_text'],
                            "labels": ["值"]  # VALUE标注
                        }
                    })
                    
                    # 生成KEY→VALUE的关系
                    ner_annotations.append({
                        "type": "relation",
                        "from_id": f"{idx}_{len(ner_annotations)-2}",  # KEY的id
                        "to_id": f"{idx}_{len(ner_annotations)-1}",    # VALUE的id
                        "direction": "right"
                    })
            
            # 构造Label Studio格式的annotations
            task = {
                "id": f"extracted_{idx}",
                "data": {
                    "text": report,
                    "ocr_text": report,
                    "report_title": report_title,
                },
                "annotations": [{
                    "id": idx,
                    "was_cancelled": False,
                    "result": ner_annotations,  # 生成的NER标注！
                }],
                "_extracted_kvs": kv_fields,  # 保存提取好的KV对
            }
            tasks.append(task)
        
        logger.info(f"  转换为任务格式: {len(tasks)} 条")
        if total_kvs > 0:
            logger.info(f"  键值对定位成功率: {located_count}/{total_kvs} ({located_count/total_kvs*100:.1f}%)")
        return tasks
        
    except Exception as e:
        logger.warning(f"读取失败: {input_path} - {e}")
        return []


def read_labelstudio_export(input_paths: List[Path]) -> List[Dict[str, Any]]:
    """读取一个或多个 Label Studio 导出的 JSON 文件
    
    Args:
        input_paths: Label Studio JSON 文件路径列表
    
    Returns:
        合并后的任务列表
    """
    all_tasks = []
    
    for input_path in input_paths:
        if not input_path.exists():
            logger.warning(f"文件不存在，跳过: {input_path}")
            continue
        
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning(f"格式错误，跳过: {input_path} (必须是 JSON 数组)")
                continue
            
            logger.info(f"读取 Label Studio 数据: {input_path.name} ({len(data)} 条任务)")
            all_tasks.extend(data)
        except Exception as e:
            logger.warning(f"读取失败，跳过: {input_path} - {e}")
            continue
    
    logger.info(f"总计: {len(all_tasks)} 条任务（来自 {len(input_paths)} 个文件）")
    return all_tasks


def has_valid_annotations(task: Dict[str, Any]) -> bool:
    """检查任务是否有有效的标注
    
    Args:
        task: Label Studio 任务 或 已提取的KV数据
    
    Returns:
        是否有有效标注（KEY/VALUE/HOSPITAL 实体 或 已提取的KV对）
    """
    # 如果是已提取的数据，检查是否有KV字段
    if "_extracted_kvs" in task:
        return len(task.get("_extracted_kvs", {})) > 0
    
    # Label Studio格式数据
    annotations = task.get("annotations", [])
    if not annotations:
        return False
    
    # 获取最新的非取消标注
    valid_annos = [a for a in annotations if not a.get("was_cancelled")]
    if not valid_annos:
        return False
    
    # 检查是否有 KEY/VALUE/HOSPITAL 标注
    latest_anno = valid_annos[-1]  # 最新标注
    results = latest_anno.get("result", [])
    
    for result in results:
        if result.get("type") == "labels":
            labels = result.get("value", {}).get("labels", [])
            if any(label in ["键名", "值", "医院名称"] for label in labels):
                return True
    
    return False


def extract_key_value_pairs(task: Dict[str, Any], label_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """从任务中提取键值对
    
    Args:
        task: Label Studio 任务 或 已提取的KV数据
        label_map: 标签映射 {"键名": "KEY", "值": "VALUE", ...}
    
    Returns:
        {key_text: {start, end, text}} 格式的字典
    """
    # 检查是否是已提取的KV数据
    if "_extracted_kvs" in task:
        # 已经提取好的键值对，从annotations中提取VALUE位置
        annotations = task.get("annotations", [])
        if annotations and annotations[0].get("result"):
            # 从NER标注中提取VALUE的位置
            result = {}
            for annotation in annotations[0]["result"]:
                if annotation.get("type") == "labels":
                    labels = annotation.get("value", {}).get("labels", [])
                    # 只提取VALUE和HOSPITAL的位置（用于评估的spans）
                    if "值" in labels or "医院名称" in labels:
                        text = annotation["value"].get("text", "")
                        # 需要找到对应的key名
                        # 这里简化处理：直接使用_extracted_kvs
                        # 在实际评估时会重新组装
                        pass
            
            # 实际上对于有NER标注的，直接返回_extracted_kvs
            # 位置信息会从annotations中提取
            return task["_extracted_kvs"]
        
        # 向后兼容：如果没有annotations
        return task["_extracted_kvs"]
    
    # Label Studio格式数据
    annotations = task.get("annotations", [])
    if not annotations:
        return {}
    
    # 获取最新的非取消标注
    valid_annos = [a for a in annotations if not a.get("was_cancelled")]
    if not valid_annos:
        return {}
    
    latest_anno = valid_annos[-1]
    results = latest_anno.get("result", [])
    
    # 提取文本
    data = task.get("data", {})
    text = str(data.get("ocr_text") or data.get("text") or "")
    
    # 收集所有实体和关系
    entities_by_id = {}  # {result_id: entity_info}
    relations = {}  # {from_id: to_id}
    
    for result in results:
        result_type = result.get("type")
        result_id = result.get("id")
        value = result.get("value", {})
        
        if result_type == "labels":
            # NER 实体
            labels = value.get("labels", [])
            if not labels:
                continue
            
            label = labels[0]  # 取第一个标签
            normalized_label = label_map.get(label, label)
            
            start = value.get("start")
            end = value.get("end")
            entity_text = value.get("text", "")
            
            if start is not None and end is not None:
                entities_by_id[result_id] = {
                    "label": normalized_label,
                    "start": start,
                    "end": end,
                    "text": entity_text,
                    "id": result_id,
                }
        
        elif result_type == "relation":
            # 关系（from_id 和 to_id 直接在 result 下，不在 value 中）
            from_id = result.get("from_id")
            to_id = result.get("to_id")
            if from_id and to_id:
                relations[from_id] = to_id
    
    # 收集所有键值对用于排序
    all_kvs = []  # [(key_text, start, end, text, is_hospital)]
    
    for entity_id, entity in entities_by_id.items():
        if entity["label"] == "KEY":
            key_text = entity["text"].strip()
            if not key_text:
                continue
            
            # 查找关联的 VALUE
            if entity_id in relations:
                value_id = relations[entity_id]
                value_entity = entities_by_id.get(value_id)
                
                if value_entity and value_entity["label"] == "VALUE":
                    all_kvs.append((
                        key_text,
                        value_entity["start"],
                        value_entity["end"],
                        value_entity["text"].strip(),
                        False,
                        entity["start"],  # KEY的start用于排序
                    ))
            else:
                # 没有关联VALUE的KEY也输出（VALUE为空）
                all_kvs.append((
                    key_text,
                    entity["end"],
                    entity["end"],
                    "",
                    False,
                    entity["start"],
                ))
    
    # 添加HOSPITAL实体（用"医院名称"作为key）
    for entity_id, entity in entities_by_id.items():
        if entity["label"] == "HOSPITAL":
            hospital_text = entity["text"].strip()
            if hospital_text:
                all_kvs.append((
                    "医院名称",
                    entity["start"],
                    entity["end"],
                    hospital_text,
                    True,
                    entity["start"],
                ))
    
    # 排序：医院名称在最前，其他按start位置排序
    all_kvs.sort(key=lambda x: (0 if x[4] else 1, x[5]))
    
    # 按顺序构建key_value_pairs（Python 3.7+字典保持插入顺序）
    key_value_pairs = {}
    for key_text, start, end, text_content, _, _ in all_kvs:
        key_value_pairs[key_text] = {
            "start": start,
            "end": end,
            "text": text_content,
        }
    
    return key_value_pairs


def split_data(
    tasks: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按比例划分数据集（只划分训练集和验证集）
    
    Args:
        tasks: 任务列表
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        (train_tasks, val_tasks)
    """
    rng = random.Random(seed)
    tasks_copy = list(tasks)
    rng.shuffle(tasks_copy)
    
    total = len(tasks_copy)
    train_size = int(total * train_ratio)
    
    train_tasks = tasks_copy[:train_size]
    val_tasks = tasks_copy[train_size:]
    
    logger.info(f"数据划分: 训练={len(train_tasks)} ({train_ratio*100:.1f}%), 验证={len(val_tasks)} ({(1-train_ratio)*100:.1f}%)")
    
    return train_tasks, val_tasks


def save_labelstudio_format(tasks: List[Dict[str, Any]], output_path: Path) -> None:
    """保存为 Label Studio 格式（用于训练）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"保存 Label Studio 格式: {output_path} ({len(tasks)} 条)")


def save_evaluation_format(
    tasks: List[Dict[str, Any]],
    output_path: Path,
    label_map: Dict[str, str],
) -> None:
    """保存为评估格式（JSONL，用于评估）
    
    Args:
        tasks: 任务列表
        output_path: 输出路径
        label_map: 标签映射
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for task in tqdm(tasks, desc="导出评估数据"):
        data = task.get("data", {})
        # TODO ocr_text分块
        text = str(data.get("ocr_text") or data.get("text") or "")
        title = str(data.get("category") or data.get("report_title") or "")
        
        if not text.strip():
            continue
        
        # 提取键值对
        spans = extract_key_value_pairs(task, label_map)
        
        if not spans:
            continue
        
        records.append({
            "report_index": str(task.get("id", "")),
            "report_title": title,
            "report": text,
            "spans": spans,
        })
    
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"保存评估格式: {output_path} ({len(records)} 条)")


def prepare_data(
    input_paths: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    train_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    config_path: Optional[str] = None,
    test_only: bool = False,
    outputs: Optional[List[str]] = None,
    minimal_outputs: bool = False,
) -> Dict[str, Any]:
    """
    准备 KV-NER 训练和评估数据
    
    Args:
        input_paths: Label Studio 导出的 JSON 文件路径列表（优先级高于配置文件）
        output_dir: 输出目录（优先级高于配置文件）
        train_ratio: 训练集比例，验证集 = 1 - train_ratio（优先级高于配置文件）
        seed: 随机种子（优先级高于配置文件）
        config_path: 配置文件路径
        test_only: 如果为 True，将所有数据作为测试集，不划分训练集
    
    Returns:
        统计信息
    """
    # 读取配置
    cfg = None
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
    
    # 从配置或参数获取设置（命令行参数优先）
    data_block = cfg.get("data", {}) if cfg else {}
    
    # 输入文件（命令行 > 配置文件）
    if input_paths is None:
        input_paths = data_block.get("input_files", [])
        if isinstance(input_paths, str):
            input_paths = [input_paths]
    
    if not input_paths:
        raise ValueError("必须指定输入文件（通过 --input 或配置文件的 data.input_files）")
    
    # 输出目录（命令行 > 配置文件 > 默认值）
    if output_dir is None:
        output_dir = data_block.get("output_dir", "data/kv_ner_prepared")
    
    # 训练比例（命令行 > 配置文件 > 默认值）
    if train_ratio is None:
        train_ratio = data_block.get("train_ratio", 0.8)
    
    # 随机种子（命令行 > 配置文件 > 默认值）
    if seed is None:
        seed = data_block.get("seed", 42)
    
    logger.info(f"标签映射: {label_map}")
    logger.info(f"训练比例: {train_ratio}, 验证比例: {1-train_ratio}, 种子: {seed}")
    
    # 读取原始数据（分离训练数据和评估数据）
    input_files = [Path(p) for p in input_paths]
    
    # 分离：Label Studio文件（用于训练）和 DA文件（只用于评估）
    labelstudio_files = []
    da_files = []
    
    for input_file in input_files:
        file_name = input_file.name
        if "clean_ocr_ppt_da" in file_name or file_name.endswith("_recheck.json"):
            da_files.append(input_file)
        else:
            labelstudio_files.append(input_file)
    
    logger.info(f"数据源分类:")
    logger.info(f"  Label Studio文件（用于训练+评估）: {len(labelstudio_files)} 个")
    logger.info(f"  DA文件（只用于评估）: {len(da_files)} 个")
    logger.info("")
    
    # 1. 读取Label Studio文件（用于训练）
    logger.info("【读取训练数据】")
    train_eval_tasks = read_labelstudio_export(labelstudio_files)
    
    # 过滤：只保留有标注的任务
    valid_train_tasks = [task for task in train_eval_tasks if has_valid_annotations(task)]
    logger.info(f"有效训练任务数: {len(valid_train_tasks)} / {len(train_eval_tasks)}")
    logger.info("")
    
    # 2. 读取DA文件（只用于评估）
    logger.info("【读取评估补充数据】")
    da_eval_tasks = []
    exclude_titles_from_da = data_block.get(
        "exclude_titles_from_da",
        ["入院记录", "术后病历", "门诊病历", "活检病历"]
    )
    
    # 加载键名别名映射（用于提高定位准确性）
    keys_alias_map = load_key_aliases("keys/keys_merged.json")
    
    for da_file in da_files:
        logger.info(f"读取DA文件: {da_file.name}")
        logger.info(f"  排除类型: {exclude_titles_from_da}")
        logger.info(f"  策略: 保留所有能完整定位的数据（KEY和VALUE都能找到）")
        tasks = read_extracted_kv_data(
            da_file,
            exclude_titles=exclude_titles_from_da,
            keys_alias_map=keys_alias_map
        )
        da_eval_tasks.extend(tasks)
    
    if da_eval_tasks:
        logger.info(f"DA评估数据: {len(da_eval_tasks)} 条")
    logger.info("")
    
    # 合并用于评估的所有数据（Label Studio + DA）
    all_eval_tasks = valid_train_tasks + da_eval_tasks
    logger.info(f"总评估数据: {len(all_eval_tasks)} 条")
    
    # 合并用于训练的数据（Label Studio + DA都参与）
    # DA数据现在有生成的NER标注，可以参与训练
    all_train_tasks = valid_train_tasks + da_eval_tasks
    logger.info(f"总训练数据: {len(all_train_tasks)} 条（Label Studio + DA）")
    
    # 用于训练的数据（包含DA）
    valid_tasks = all_train_tasks
    
    # 解析需要生成哪些输出文件
    # 可选键：'train_json','val_json','val_eval','da_eval','test_eval','test_json','summary'
    # 注：按最新实验约束，移除 'full_eval' 输出，评估统一使用 val_eval.jsonl 或 test_eval.jsonl。
    DEFAULT_ALL = {"train_json", "val_json", "val_eval", "summary"}
    DEFAULT_ALL_WITH_DA = set(DEFAULT_ALL) | {"da_eval"}
    default_minimal = {"test_eval", "summary"} if test_only else {"train_json", "val_json", "val_eval", "summary"}

    cfg_outputs = None
    if cfg and isinstance(cfg.get("data", {}), dict):
        o = cfg["data"].get("outputs")
        if isinstance(o, list) and all(isinstance(x, str) for x in o):
            cfg_outputs = {x for x in o}

    if outputs is not None:
        outputs_set = {str(x).strip() for x in outputs if str(x).strip()}
    elif minimal_outputs:
        outputs_set = set(default_minimal)
    elif cfg_outputs:
        outputs_set = cfg_outputs
    else:
        outputs_set = set(DEFAULT_ALL_WITH_DA)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_files: List[str] = []
    if test_only:
        # 测试集模式：全部数据作为测试集，不划分
        logger.info("测试集模式：全部数据转换为评估格式")
        if "test_eval" in outputs_set:
            save_evaluation_format(valid_tasks, output_path / "test_eval.jsonl", label_map)
            created_files.append("test_eval.jsonl")
        if "test_json" in outputs_set:
            save_labelstudio_format(valid_tasks, output_path / "test.json")
            created_files.append("test.json")
        train_tasks = []
        val_tasks = valid_tasks
    else:
        # 训练模式：划分训练集和验证集（只用Label Studio数据）
        train_tasks, val_tasks = split_data(
            valid_tasks,
            train_ratio=train_ratio,
            seed=seed,
        )
        
        # 保存训练数据（Label Studio 格式 - 只有Label Studio数据）
        if "train_json" in outputs_set:
            save_labelstudio_format(train_tasks, output_path / "train.json")
            created_files.append("train.json")
        if "val_json" in outputs_set:
            save_labelstudio_format(val_tasks, output_path / "val.json")
            created_files.append("val.json")
        
        # 保存评估数据（JSONL 格式）
        # 1. 验证集评估数据（来自Label Studio）
        if "val_eval" in outputs_set:
            save_evaluation_format(val_tasks, output_path / "val_eval.jsonl", label_map)
            created_files.append("val_eval.jsonl")
        
        # 2. DA文件评估数据（单独保存，包含更多报告类型）
        if da_eval_tasks and ("da_eval" in outputs_set):
            logger.info(f"保存DA评估数据...")
            save_evaluation_format(da_eval_tasks, output_path / "da_eval.jsonl", label_map)
            created_files.append("da_eval.jsonl")
        
        # 3. 完整评估数据输出已移除（请使用 val_eval.jsonl 或 da_eval.jsonl 按需评估）
    
    # 统计信息
    summary = {
        "input_files": input_paths,
        "output_dir": str(output_dir),
        "labelstudio_files": [str(f) for f in labelstudio_files],
        "da_files": [str(f) for f in da_files],
        "total_train_tasks": len(train_eval_tasks),
        "valid_train_tasks": len(valid_train_tasks),
        "da_eval_tasks": len(da_eval_tasks),
        "total_eval_tasks": len(all_eval_tasks),
        "mode": "test_only" if test_only else "train_val_split",
        "label_map": label_map,
    }
    
    if test_only:
        summary["test_size"] = len(valid_tasks)
        summary["note"] = "全部数据作为测试集"
    else:
        summary["train_size"] = len(train_tasks)
        summary["val_size"] = len(val_tasks)
        summary["train_ratio"] = float(train_ratio)
        summary["val_ratio"] = float(1 - train_ratio)
        summary["seed"] = int(seed)
        summary["note"] = "训练用Label Studio数据，评估用Label Studio+DA数据"
    
    # 保存统计信息
    summary_path = output_path / "data_summary.json"
    if "summary" in outputs_set:
        summary["generated_files"] = created_files
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    
    logger.info(f"\n{'='*80}")
    logger.info("数据准备完成")
    logger.info(f"{'='*80}")
    logger.info(f"输入文件: {len(input_paths)} 个")
    logger.info(f"  Label Studio文件: {len(labelstudio_files)} 个")
    for i, p in enumerate(labelstudio_files, 1):
        logger.info(f"    {i}. {p.name}")
    if da_files:
        logger.info(f"  DA文件: {len(da_files)} 个")
        for i, p in enumerate(da_files, 1):
            logger.info(f"    {i}. {p.name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"\n数据统计:")
    logger.info(f"  Label Studio任务: {len(train_eval_tasks)} → 有效 {len(valid_train_tasks)}")
    if da_eval_tasks:
        logger.info(f"  DA评估任务: {len(da_eval_tasks)}")
    logger.info(f"  总评估数据: {len(all_eval_tasks)}")
    
    if test_only:
        logger.info(f"\n模式: 测试集模式")
        logger.info(f"测试集: {len(valid_tasks)} 条")
        logger.info(f"\n生成文件:")
        if "test_eval.jsonl" in created_files:
            logger.info(f"  - test_eval.jsonl (评估数据)")
        if "test.json" in created_files:
            logger.info(f"  - test.json (测试集，Label Studio 格式)")
        if "summary" in outputs_set:
            logger.info(f"  - data_summary.json (统计信息)")
    else:
        logger.info(f"\n模式: 训练模式")
        logger.info(f"训练集: {len(train_tasks)} ({train_ratio*100:.1f}%)")
        logger.info(f"  - Label Studio: ~{len(valid_train_tasks)} 样本（有精确NER标注）")
        logger.info(f"  - DA文件: ~{len(da_eval_tasks)} 样本（生成的NER标注）")
        logger.info(f"验证集: {len(val_tasks)} ({(1-train_ratio)*100:.1f}%)")
        logger.info(f"注意: 测试集将在训练时从验证集动态划分")
        logger.info(f"\n生成文件:")
        if "train.json" in created_files:
            logger.info(f"  - train.json (训练数据，Label Studio格式)")
        if "val.json" in created_files:
            logger.info(f"  - val.json (验证+测试数据池，Label Studio格式)")
        if "val_eval.jsonl" in created_files:
            logger.info(f"  - val_eval.jsonl (验证集评估数据)")
        if "da_eval.jsonl" in created_files:
            logger.info(f"  - da_eval.jsonl (DA文件评估数据，{len(da_eval_tasks)}条)")
        # full_eval.jsonl 已移除
        if "summary" in outputs_set:
            logger.info(f"  - data_summary.json (统计信息)")
    
    logger.info(f"{'='*80}\n")
    
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="准备 KV-NER 训练和评估数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从配置文件读取所有参数
  python prepare_data.py --config pre_struct/kv_ner/kv_ner_config.json
  
  # 使用配置文件，但覆盖输入文件
  python prepare_data.py --config pre_struct/kv_ner/kv_ner_config.json --input data/file1.json
  
  # 多个文件（覆盖配置）
  python prepare_data.py --config pre_struct/kv_ner/kv_ner_config.json --input data/*.json
  
  # 不使用配置文件（纯命令行）
  python prepare_data.py --input data/file1.json --output_dir data/output
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pre_struct/kv_ner/kv_ner_config.json",
        help="配置文件路径（默认: pre_struct/kv_ner/kv_ner_config.json）",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="Label Studio 导出的 JSON 文件路径（可选，默认从配置文件读取）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（可选，默认从配置文件读取）",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="训练集比例，验证集 = 1 - train_ratio（可选，默认从配置文件读取）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，默认从配置文件读取）",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="测试集模式：全部数据转换为测试集，不划分训练集",
    )
    parser.add_argument(
        "--minimal-outputs",
        action="store_true",
        help=(
            "仅生成必要输出：训练模式(train.json, val.json, val_eval.jsonl, data_summary.json)；"
            "测试模式(test_eval.jsonl, data_summary.json)"
        ),
    )
    parser.add_argument(
        "--outputs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "精确指定要生成的文件集合（覆盖默认）。可选："
            "train_json val_json val_eval da_eval test_eval test_json summary"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(
        config_path=args.config,
        input_paths=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        test_only=args.test_only,
        outputs=args.outputs,
        minimal_outputs=args.minimal_outputs,
    )
