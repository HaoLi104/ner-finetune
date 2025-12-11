# Quick Start Guide

## 快速开始

这个指南将帮助你快速使用实体评估功能，只需要几行代码就可以评估你的NER或半结构化实体提取结果。

## 基本使用

### 1. 导入函数

```python
from src.easy_eval import evaluate_entities, quick_eval
```

### 2. 准备数据

#### NER数据格式
```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"},
    {"start": 60, "end": 70, "type": "手术名称", "text": "额外手术"}  # False positive
]}]
```

#### 半结构化数据格式

```python
y_true = [{"entities": [
    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
    {"text": "张三", "start": 10, "end": 12},
    {"text": "45岁", "start": 21, "end": 24}
]}]

y_pred = [{"entities": [
    {"text": "右侧乳腺发现无痛性肿物约6天，大小无显著变化，无皮肤改变", "start": 52, "end": 60},
    {"text": "张三", "start": 10, "end": 12},
    {"text": "45岁", "start": 21, "end": 24},
    {"text": "额外内容", "start": 100, "end": 105}  # False positive
]}]
```

### 3. 评估结果

#### 方法1：完整函数（推荐）
```python
# NER评估
results = evaluate_entities(y_true, y_pred, mode="ner")
print(f"F1 Score: {results['f1_score']:.3f}")

# 半结构化评估
results = evaluate_entities(y_true, y_pred, mode="semi_structured")
print(f"F1 Score: {results['f1_score']:.3f}")
```

#### 方法2：快速评估
```python
# 使用默认设置快速评估
results = quick_eval(y_true, y_pred, mode="ner")
print(f"F1 Score: {results['f1_score']:.3f}")
```

## 完整示例

```python
from src.easy_eval import evaluate_entities

# 你的数据
y_true = [{"entities": [
    {"start": 0, "end": 10, "type": "患者姓名", "text": "张三"},
    {"start": 15, "end": 25, "type": "年龄", "text": "45岁"},
    {"start": 30, "end": 50, "type": "诊断", "text": "高血压"}
]}]

y_pred = [{"entities": [
    {"start": 0, "end": 10, "type": "患者姓名", "text": "张三"},
    {"start": 15, "end": 25, "type": "年龄", "text": "45岁"}
    # 缺少诊断信息
]}]

# 评估
results = evaluate_entities(y_true, y_pred, mode="ner")

# 输出结果
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1 Score: {results['f1_score']:.3f}")
print(f"True Positives: {results['true_positives']}")
print(f"False Positives: {results['false_positives']}")
print(f"False Negatives: {results['false_negatives']}")
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `y_true` | List[Dict] | 必需 | 真实标签数据 |
| `y_pred` | List[Dict] | 必需 | 预测结果数据 |
| `mode` | str | "ner" | "ner" 或 "semi_structured" |
| `strict_match` | bool | True | 严格匹配或重叠匹配 |
| `enable_validation` | bool | False | 是否启用验证 |

## 返回结果

函数返回一个字典，包含以下指标：

- `precision`: 精确率
- `recall`: 召回率
- `f1_score`: F1分数
- `true_positives`: 真正例数量
- `false_positives`: 假正例数量
- `false_negatives`: 假负例数量
- `accuracy`: 准确率（NER中无意义，始终为0）
- `true_negatives`: 真负例数量（NER中无意义，始终为0）

## 常见用法

### 1. 基本NER评估
```python
results = evaluate_entities(y_true, y_pred, mode="ner")
```

### 2. 半结构化评估
```python
results = evaluate_entities(y_true, y_pred, mode="semi_structured")
```

### 3. 重叠匹配
```python
results = evaluate_entities(y_true, y_pred, mode="ner", strict_match=False)
```

### 4. 启用验证
```python
results = evaluate_entities(y_true, y_pred, mode="ner", enable_validation=True)
```

### 5. 快速评估
```python
results = quick_eval(y_true, y_pred, mode="ner")
```

## 注意事项

1. **数据格式**：确保 `y_true` 和 `y_pred` 长度相同
2. **位置索引**：`start` 和 `end` 是字符位置索引
3. **字段要求**：
   - NER模式：需要 `start`, `end`, `type`, `text` 字段
   - 半结构化模式：需要 `start`, `end`, `text` 字段
4. **验证功能**：建议在开发阶段启用验证，生产环境根据需要选择

现在你可以轻松评估你的实体识别和提取结果了！ 