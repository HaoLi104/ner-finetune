#!/bin/bash
# -*- coding: utf-8 -*-
#
# KV-NER 一键运行脚本
# 完成从数据准备到评估的完整流程
#

set -e  # 遇到错误立即退出

CONFIG="pre_struct/kv_ner/kv_ner_config.json"

echo "=========================================="
echo "KV-NER 完整训练和评估流程"
echo "=========================================="
echo ""

# 步骤 1：数据准备
echo "步骤 1/4: 数据准备"
echo "------------------------------------------"
python pre_struct/kv_ner/prepare_data.py --config ${CONFIG}
echo ""

# 步骤 2：模型训练
echo "步骤 2/4: 模型训练"
echo "------------------------------------------"
python pre_struct/kv_ner/train.py --config ${CONFIG}
echo ""

# 步骤 3：模型评估
echo "步骤 3/4: 模型评估（键值对级别）"
echo "------------------------------------------"
python pre_struct/kv_ner/evaluate.py \
  --config ${CONFIG} \
  --test_data data/kv_ner_prepared/val_eval.jsonl
echo ""

# 步骤 4：推理预测
echo "步骤 4/4: 推理预测"
echo "------------------------------------------"
python pre_struct/kv_ner/predict.py --config ${CONFIG}
echo ""

echo "=========================================="
echo "✅ 完整流程执行完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  数据准备: data/kv_ner_prepared/"
echo "  训练模型: runs/kv_ner_ruyuan/best/"
echo "  训练报告: runs/kv_ner_ruyuan/training_summary.json"
echo "  评估报告: data/kv_ner_eval/eval_summary.json"
echo "  预测结果: runs/kv_ner_ruyuan/predictions.json"
echo ""

