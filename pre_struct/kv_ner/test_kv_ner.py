#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER 单条样本测试脚本

在 main 函数中直接输入报告文本，输出结构化 JSON 结果
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

# 添加项目根目录到路径
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from pre_struct.kv_ner import config_io
from pre_struct.kv_ner.data_utils import build_bio_label_list
from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
from pre_struct.kv_ner.chunking import predict_with_chunking


def predict_kv(
    text: str,
    title: str = "",
    config_path: str = "pre_struct/kv_ner/kv_ner_config.json",
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    预测单条报告的键值对
    
    Args:
        text: 报告文本
        title: 报告标题
        config_path: 配置文件路径
        model_dir: 模型目录（可选，默认从配置读取）
    
    Returns:
        结构化 JSON 结果
    """
    # 读取配置
    cfg = config_io.load_config(config_path)
    
    # 模型目录
    if model_dir is None:
        train_block = cfg.get("train", {})
        model_dir = str(Path(train_block.get("output_dir", "runs/kv_ner")) / "best")
    
    # 构建标签映射
    label_map = config_io.label_map_from(cfg)
    labels = build_bio_label_list(label_map)
    id2label = {idx: label for idx, label in enumerate(labels)}
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(model_dir).to(device)
    model.eval()
    
    # 加载 tokenizer
    tokenizer_path = Path(model_dir) / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_io.model_name_from(cfg))
    
    # 使用分块预测（处理超长文本）
    max_seq_length = config_io.max_seq_length(cfg)
    chunk_size = cfg.get("chunk_size", 450)
    chunk_overlap = cfg.get("chunk_overlap", 50)
    
    entities = predict_with_chunking(
        text=text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_adjacent_gap=int(cfg.get("merge_adjacent_gap", 2)),
    )
    
    # 组装键值对（与 predict.py 逻辑一致）
    hospital = [e for e in entities if e["type"] == "HOSPITAL"]
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    seq.sort(key=lambda x: (x["start"], x["end"]))
    
    pairs: List[Dict[str, Any]] = []
    key_without_value: List[Dict[str, Any]] = []
    value_without_key: List[Dict[str, Any]] = []
    
    pending: Optional[Dict[str, Any]] = None
    for ent in seq:
        if ent["type"] == "KEY":
            if pending:
                if pending["values"]:
                    texts = "；".join(v["text"] for v in pending["values"])
                    pairs.append({
                        "key": pending["key"],
                        "values": pending["values"],
                        "value_text": texts,
                    })
                else:
                    key_without_value.append(pending["key"])
            pending = {"key": ent, "values": []}
        elif ent["type"] == "VALUE":
            if pending:
                pending["values"].append(ent)
            else:
                value_without_key.append(ent)
    
    if pending:
        if pending["values"]:
            texts = "；".join(v["text"] for v in pending["values"])
            pairs.append({
                "key": pending["key"],
                "values": pending["values"],
                "value_text": texts,
            })
        else:
            key_without_value.append(pending["key"])
    
    # 收集所有键值对用于排序（保持原文顺序）
    all_kvs = []  # [(key_text, value_text, start_pos, is_hospital)]
    
    for entry in pairs:
        key_text = entry["key"]["text"]
        value_text = entry["value_text"]
        if not key_text:
            continue
        key_start = entry["key"].get("start", 0)
        all_kvs.append((key_text, value_text, key_start, False))
    
    # 添加没有VALUE的KEY
    for key_ent in key_without_value:
        key_text = key_ent.get("text", "").strip()
        if key_text:
            key_start = key_ent.get("start", 0)
            all_kvs.append((key_text, "", key_start, False))
    
    # 添加没有KEY的VALUE（key使用空字符串）
    for value_ent in value_without_key:
        value_text = value_ent.get("text", "").strip()
        if value_text:
            value_start = value_ent.get("start", 0)
            all_kvs.append(("", value_text, value_start, False))
    
    # 添加HOSPITAL实体
    for hospital_ent in hospital:
        hospital_text = hospital_ent.get("text", "").strip()
        if hospital_text:
            hospital_start = hospital_ent.get("start", 0)
            all_kvs.append(("医院名称", hospital_text, hospital_start, True))
    
    # 排序：医院名称在最前，其他按start位置排序
    all_kvs.sort(key=lambda x: (0 if x[3] else 1, x[2]))
    
    # 按顺序构建structured（Python 3.7+字典保持插入顺序）
    structured: Dict[str, Any] = {}
    for key_text, value_text, _, _ in all_kvs:
        if key_text == "":
            # 对于空字符串key，合并多个value（用分号连接）
            if "" in structured:
                structured[""] = structured[""] + "；" + value_text
            else:
                structured[""] = value_text
        else:
            structured[key_text] = value_text
    
    return {
        "report_title": title,
        "report": text,
        "entities": entities,
        "pairs": pairs,
        "structured": structured,
        "hospital": hospital,
        "key_without_value": key_without_value,
        "value_without_key": value_without_key,
    }


if __name__ == "__main__":
    # ========================================
    # 在这里输入要测试的报告
    # ========================================

    # 报告标题
    REPORT_TITLE = "入院记录"

    # 报告文本
    REPORT_TEXT = """
第2页 上海市第一妇婴保健院 24小时内入出院记录 0 姓名：徐松蕾 科别：乳腺科 病区：1A病区 床位：1A12 住院号：D00263180 区 区 病区 瘤7病 第1页 信扫一扫 A病区 上海市第一妇婴保健院 门诊号：0P31186444 1A病 肿瘤 24小时内入出院记录 住院号：D00263180 肿 科别：乳腺科 病室：1A病区床位：1A12 姓名：徐松蕾性别：女年龄：36岁职业：其他 入院时间： 2025.09.2308 出院时间： 2025.09.2315 Y 科 主诉：右乳癌新辅助化疗 入院情况：手术时间：2025.08.18 手术方式：右侧乳房穿刺活检 术后病理结果： 1、(右乳外侧肿块穿刺标本)少量浸润性癌组织。 2、(右乳内上肿块穿刺标本)乳腺浸润性癌Ⅲ级，非特殊型。 3、(右腋下淋巴结穿刺标本)见癌累及。 免疫组化IHC:2#(右乳内上肿块穿刺标本)：ER(-),PR(-),CerbB-2(1+),Ki-67(70%+) 化疗方案：wPCb*18 化疗适应症：浸润性癌，三阴性，腋窝淋巴结多发转移。 化疗禁忌症：暂无。 入院诊断：术前新辅助化疗(C1D15);右乳癌（浸润性癌II级，pT4N3Mx,IIIC期） 入院时主要症状和体征：血常规、肝肾功能无殊 EBRA 诊疗经过：化疗前评估：无明显化疗禁忌 化疗方案：白蛋白紫杉醇 200mg,d1;卡铂（波贝） 200mg,d1。 此次化疗不良反应：无 出院情况：暂无明显不适 出院诊断：术前新辅助化疗(C2D1);右乳癌（浸润性癌II级，pT4N3Mx,IIIC期） 出院医嘱：1.下次化疗时间距本次化疗1周。 2.出院后每周门诊随访血常规至化疗后3~4周，若白细胞低于3.5×10^9/L,注射升白细胞药
    """.strip()

    # 配置文件路径
    CONFIG_PATH = "pre_struct/kv_ner/kv_ner_config.json"

    # ========================================
    # 执行预测
    # ========================================

    print("正在加载模型...")
    result = predict_kv(
        text=REPORT_TEXT,
        title=REPORT_TITLE,
        config_path=CONFIG_PATH,
    )

    # 输出结构化 JSON
    print("\n" + "="*80)
    print("结构化 JSON 输出")
    print("="*80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("="*80)
