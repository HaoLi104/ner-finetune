# KV-NER 模型评估指南

本指南详细说明如何使用 `evaluate.py` 对训练好的 KV-NER 模型进行评估。

## 快速开始

### 基本用法

评估默认配置的模型：

```bash
python pre_struct/kv_ner/evaluate.py --config pre_struct/kv_ner/kv_ner_config.json
```

### 自定义评估

```bash
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --model_dir runs/kv_ner_ruyuan/best \
  --data_path data/ruyuanjilu/ruyuan-2025-10-16.json \
  --output_dir data/kv_ner_eval \
  --max_samples 500 \
  --seed 42 \
  --error_threshold 0.9
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--config` | str | `pre_struct/kv_ner/kv_ner_config.json` | 配置文件路径 |
| `--model_dir` | str | 从配置读取 | 模型目录（通常是 `runs/kv_ner_ruyuan/best`） |
| `--data_path` | str | 从配置读取 | 测试数据路径（Label Studio 格式） |
| `--output_dir` | str | `data/kv_ner_eval` | 评估结果输出目录 |
| `--max_samples` | int | None | 最大评估样本数（用于快速测试） |
| `--seed` | int | 42 | 随机种子（用于采样） |
| `--error_threshold` | float | 0.99 | 错误样本 F1 阈值（低于此值的样本会被记录） |

## 输出文件

评估完成后，会在输出目录生成以下文件：

### 1. `eval_summary.json`

完整的评估指标和配置信息。

**结构示例：**

```json
{
  "config_path": "pre_struct/kv_ner/kv_ner_config.json",
  "model_dir": "runs/kv_ner_ruyuan/best",
  "data_path": "data/ruyuanjilu/ruyuan-2025-10-16.json",
  "num_samples": 1782,
  "num_errors": 145,
  "max_seq_length": 512,
  "seed": 42,
  "error_threshold": 0.99,
  "overall_metrics": {
    "precision": 0.9245,
    "recall": 0.9102,
    "f1_score": 0.9173,
    "tp": 1623,
    "fp": 134,
    "fn": 160
  },
  "per_type_metrics": {
    "KEY": {
      "precision": 0.9456,
      "recall": 0.9312,
      "f1_score": 0.9383,
      "tp": 845,
      "fp": 49,
      "fn": 63
    },
    "VALUE": {
      "precision": 0.9124,
      "recall": 0.8987,
      "f1_score": 0.9055,
      "tp": 689,
      "fp": 66,
      "fn": 78
    },
    "HOSPITAL": {
      "precision": 0.8956,
      "recall": 0.8845,
      "f1_score": 0.8900,
      "tp": 89,
      "fp": 19,
      "fn": 19
    }
  }
}
```

### 2. `error_samples.jsonl`

F1 分数低于阈值的样本，每行一个 JSON 对象。

**每条记录包含：**

```json
{
  "task_id": 12345,
  "text": "原始报告文本...",
  "metrics": {
    "precision": 0.75,
    "recall": 0.80,
    "f1_score": 0.77
  },
  "ground_truth": {
    "entities": [
      {"type": "KEY", "start": 0, "end": 4, "text": "姓名"},
      {"type": "VALUE", "start": 5, "end": 8, "text": "张三"}
    ]
  },
  "predictions": {
    "entities": [
      {"type": "KEY", "start": 0, "end": 4, "text": "姓名"},
      {"type": "VALUE", "start": 5, "end": 9, "text": "张三丰"}
    ]
  },
  "total_true": 2,
  "total_pred": 2,
  "matched": 1
}
```

## 评估指标说明

### 整体指标 (Overall Metrics)

- **Precision（精准度）**: TP / (TP + FP)
  - 预测的实体中，有多少是正确的
  - 衡量模型的**准确性**

- **Recall（召回率）**: TP / (TP + FN)
  - 实际的实体中，有多少被正确预测
  - 衡量模型的**完整性**

- **F1 Score（F1分数）**: 2 × (Precision × Recall) / (Precision + Recall)
  - 精准度和召回率的调和平均
  - 综合评价指标

- **TP（True Positives）**: 正确预测的实体数
- **FP（False Positives）**: 错误预测的实体数
- **FN（False Negatives）**: 漏预测的实体数

### 按类型指标 (Per-Type Metrics)

分别计算以下三类实体的指标：
- **KEY**: 键实体（如"姓名"、"年龄"等）
- **VALUE**: 值实体（如"张三"、"25岁"等）
- **HOSPITAL**: 医院名称实体

### 匹配规则

评估采用**严格的字符级别位置匹配**：
- **start** 和 **end** 必须完全一致
- **text** 自动去除首尾标点符号后比较
- 使用与 EBQA 模型相同的 evaluation 库

## 常见使用场景

### 场景 1: 快速验证模型性能

```bash
# 只评估 100 个样本，快速查看指标
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --max_samples 100
```

### 场景 2: 完整评估并分析错误

```bash
# 评估所有样本，记录 F1 < 0.9 的样本
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --error_threshold 0.9 \
  --output_dir data/kv_ner_full_eval
```

### 场景 3: 评估特定检查点

```bash
# 评估某个特定的模型检查点
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --model_dir runs/kv_ner_ruyuan/checkpoint-epoch-2
```

### 场景 4: 在不同数据集上评估

```bash
# 在新的测试集上评估模型
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --data_path data/new_test_set.json \
  --output_dir data/new_test_eval
```

## 与训练评估的对比

| 特性 | 训练时评估 | 独立评估脚本 |
|------|-----------|-------------|
| 时机 | 训练过程中自动执行 | 手动执行 |
| 数据集 | 训练划分的测试集 | 任意指定的数据集 |
| 评估库 | 内置 metrics.py | evaluation 库（与EBQA一致） |
| 错误分析 | 无 | 详细的错误样本记录 |
| 灵活性 | 固定 | 高度可配置 |
| 适用场景 | 模型开发期间 | 模型验证、对比、部署前 |

## 评估结果解读

### 高质量模型指标参考

- **F1 Score > 0.90**: 优秀
- **F1 Score 0.85-0.90**: 良好
- **F1 Score 0.80-0.85**: 一般
- **F1 Score < 0.80**: 需要改进

### 常见问题诊断

**问题 1: Precision 高，Recall 低**
- **原因**: 模型过于保守，只预测非常确定的实体
- **解决**: 调整模型阈值，增加训练数据多样性

**问题 2: Recall 高，Precision 低**
- **原因**: 模型过于激进，预测了很多错误实体
- **解决**: 增加负样本训练，调整 CRF 参数

**问题 3: 某类实体 F1 特别低**
- **原因**: 该类实体标注数据不足或不一致
- **解决**: 增加该类实体的标注样本，检查标注质量

**问题 4: 整体 F1 低**
- **原因**: 数据质量问题、模型欠拟合、超参数不合适
- **解决**: 检查数据、增加训练轮数、调整学习率

## 进阶技巧

### 1. 批量评估多个检查点

```bash
for epoch in 1 2 3; do
  python pre_struct/kv_ner/evaluate.py \
    --config pre_struct/kv_ner/kv_ner_config.json \
    --model_dir runs/kv_ner_ruyuan/checkpoint-epoch-${epoch} \
    --output_dir data/eval_epoch_${epoch}
done
```

### 2. 提取错误样本进行人工审核

```bash
# 评估后查看错误样本
python -c "
import json
with open('data/kv_ner_eval/error_samples.jsonl') as f:
    errors = [json.loads(line) for line in f]
    print(f'错误样本数: {len(errors)}')
    for i, err in enumerate(errors[:5]):  # 显示前5个
        print(f'\n样本 {i+1}:')
        print(f'F1: {err[\"metrics\"][\"f1_score\"]:.3f}')
        print(f'文本: {err[\"text\"][:100]}...')
"
```

### 3. 对比两个模型

```bash
# 模型 A
python pre_struct/kv_ner/evaluate.py \
  --model_dir runs/model_a/best \
  --output_dir data/eval_model_a

# 模型 B
python pre_struct/kv_ner/evaluate.py \
  --model_dir runs/model_b/best \
  --output_dir data/eval_model_b

# 比较结果
python -c "
import json
from pathlib import Path

a = json.loads(Path('data/eval_model_a/eval_summary.json').read_text())
b = json.loads(Path('data/eval_model_b/eval_summary.json').read_text())

print('模型对比:')
print(f'模型 A F1: {a[\"overall_metrics\"][\"f1_score\"]:.4f}')
print(f'模型 B F1: {b[\"overall_metrics\"][\"f1_score\"]:.4f}')
print(f'差异: {b[\"overall_metrics\"][\"f1_score\"] - a[\"overall_metrics\"][\"f1_score\"]:+.4f}')
"
```

## 最佳实践

1. **定期评估**: 每次模型更新后都进行完整评估
2. **保留历史**: 保存每次评估的 `eval_summary.json` 以追踪性能变化
3. **分析错误**: 定期查看 `error_samples.jsonl`，发现数据质量问题
4. **交叉验证**: 在多个数据集上评估，确保模型泛化性
5. **对比基线**: 建立基线模型，与新模型对比改进效果

## 故障排除

### 问题: 找不到模型文件

```
错误: OSError: Unable to load weights from pytorch model file
```

**解决**: 
- 检查 `--model_dir` 路径是否正确
- 确保模型已训练完成并保存

### 问题: 数据格式错误

```
错误: KeyError: 'annotations'
```

**解决**:
- 确保数据是 Label Studio 导出格式
- 检查数据文件是否完整

### 问题: 内存不足

```
错误: RuntimeError: CUDA out of memory
```

**解决**:
- 使用 `--max_samples` 限制样本数
- 降低 `max_seq_length`
- 使用 CPU 评估（自动切换）

## 总结

`evaluate.py` 提供了完整、灵活、标准化的模型评估流程：
- ✅ 使用与 EBQA 相同的 evaluation 库
- ✅ 支持整体和按类型的详细指标
- ✅ 自动记录和分析错误样本
- ✅ 高度可配置，适用多种场景
- ✅ 输出清晰，便于对比和追踪

建议在模型训练完成后、部署前，使用此脚本进行完整的模型验证。

