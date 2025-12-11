# 实体一致性验证指南

## 概述

评估模块现在包含了一个可配置的实体一致性验证功能，支持两种模式：

1. **NER模式**: 确保输入的true和pred数据的start、end、type字段能够正确对应
2. **半结构化模式**: 确保输入的true和pred数据的start、end、content字段能够正确对应

验证功能默认关闭，可以通过参数启用。

**重要说明**: 在实体评估中，无法计算True Negatives (TN)和Accuracy，因为我们只关注实体，不跟踪"非实体"区域。

## 验证规则

### NER模式验证规则

#### 允许的情况

1. **完全匹配**: true和pred中的实体具有相同的start、end、type
2. **False Positive**: pred中有额外的实体（true中没有对应实体）
3. **False Negative**: pred中缺少某些实体（true中有但pred中没有）
4. **文本差异**: 相同位置和类型的实体可以有不同的text内容（包括空文本）

#### 不允许的情况

1. **位置冲突**: 相同start位置但不同的end位置或type
2. **类型冲突**: 相同start和end位置但不同的type
3. **边界冲突**: 相同end位置但不同的start位置或type

### 半结构化模式验证规则

#### 允许的情况

1. **完全匹配**: true和pred中的实体具有相同的start、end、content
2. **False Positive**: pred中有额外的实体（true中没有对应实体）
3. **False Negative**: pred中缺少某些实体（true中有但pred中没有）

#### 不允许的情况

1. **位置冲突**: 相同start位置但不同的end位置
2. **内容冲突**: 相同start和end位置但不同的content
3. **边界冲突**: 相同end位置但不同的start位置

## 使用示例

### NER模式基本用法

#### 默认行为（验证关闭）

```python
from src.ner_metrics import NERMetrics

# 创建评估器（验证默认关闭）
metrics = NERMetrics(strict_match=True, enable_validation=False)

# 准备数据
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"},
    {"start": 30, "end": 50, "type": "手术名称", "text": "单乳切除术"}
]}]

# 计算指标（不会进行验证）
results = metrics.calculate_metrics(y_true, y_pred)
print(f"F1 Score: {results['f1_score']:.3f}")
```

### 半结构化模式基本用法

#### 默认行为（验证关闭）

```python
from src.ner_metrics import SemiStructuredMetrics

# 创建半结构化评估器（验证默认关闭）
metrics = SemiStructuredMetrics(strict_match=True, enable_validation=False)

# 准备数据（使用content字段而不是type）
y_true = [{"entities": [
    {"start": 10, "end": 20, "content": "姓名"},
    {"start": 30, "end": 50, "content": "年龄"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "content": "姓名"},
    {"start": 30, "end": 50, "content": "年龄"}
]}]

# 计算指标（不会进行验证）
results = metrics.calculate_metrics(y_true, y_pred)
print(f"F1 Score: {results['f1_score']:.3f}")
```

#### 启用验证

```python
# 创建评估器（启用验证）
metrics = NERMetrics(strict_match=True, enable_validation=True)

# 计算指标（会自动进行验证）
try:
    results = metrics.calculate_metrics(y_true, y_pred)
    print(f"F1 Score: {results['f1_score']:.3f}")
except ValueError as e:
    print(f"验证失败: {e}")
```



#### 半结构化模式启用验证

```python
# 创建半结构化评估器（启用验证）
metrics = SemiStructuredMetrics(strict_match=True, enable_validation=True)

# 计算指标（会自动进行验证）
try:
    results = metrics.calculate_metrics(y_true, y_pred)
    print(f"F1 Score: {results['f1_score']:.3f}")
except ValueError as e:
    print(f"验证失败: {e}")
```



### NER模式验证失败的情况

#### 1. 不同的start位置

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
]}]

y_pred = [{"entities": [
    {"start": 11, "end": 20, "type": "手术日期", "text": "2024-09-10"}  # 不同的start
]}]

# 这会抛出ValueError: Position conflict at end 20: true (10, 20, 手术日期) vs pred (11, 20, 手术日期)
```

#### 2. 不同的end位置

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 21, "type": "手术日期", "text": "2024-09-10"}  # 不同的end
]}]

# 这会抛出ValueError: Position conflict at start 10: true (10, 20, 手术日期) vs pred (10, 21, 手术日期)
```

#### 3. 不同的type

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "type": "手术日期", "text": "2024-09-10"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "type": "手术名称", "text": "2024-09-10"}  # 不同的type
]}]

# 这会抛出ValueError: Type mismatch at position (10, 20): true types {'手术日期'} vs pred types {'手术名称'}
```



### 半结构化模式验证失败的情况

#### 1. 不同的start位置

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "content": "姓名"}
]}]

y_pred = [{"entities": [
    {"start": 11, "end": 20, "content": "姓名"}  # 不同的start
]}]

# 这会抛出ValueError: Position conflict at end 20: true (10, 20) vs pred (11, 20)
```

#### 2. 不同的end位置

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "content": "姓名"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 21, "content": "姓名"}  # 不同的end
]}]

# 这会抛出ValueError: Position conflict at start 10: true (10, 20) vs pred (10, 21)
```

#### 3. 不同的content

```python
y_true = [{"entities": [
    {"start": 10, "end": 20, "content": "姓名"}
]}]

y_pred = [{"entities": [
    {"start": 10, "end": 20, "content": "年龄"}  # 不同的content
]}]

# 这会抛出ValueError: Position conflict at start 10: true (10, 20) vs pred (10, 20)
```



## 测试验证功能

运行以下测试来验证功能：

```bash
# 基本验证测试
python test_validation.py

# 全面验证测试
python test_comprehensive_validation.py

# 半结构化模式测试
python -m pytest tests/test_semi_structured_metrics.py -v
```

## 错误消息说明

验证失败时会抛出`ValueError`，错误消息包含以下信息：

1. **位置冲突**: 显示冲突的位置和具体的start/end/type差异
2. **类型冲突**: 显示冲突位置和不同的类型信息

## 注意事项

1. 验证功能默认关闭（`enable_validation=False`），需要手动启用
2. 验证只检查start、end、type的一致性，不检查text内容
3. 允许false positive和false negative，这是NER评估的正常情况
4. 如果验证失败，整个评估过程会停止并抛出异常
5. 建议在开发和调试阶段启用验证，在生产环境中根据需求选择
6. **True Negatives (TN)和Accuracy在NER评估中没有意义**，因为无法确定哪些"非实体"区域被正确识别

## 最佳实践

1. 在开发和调试阶段启用验证（`enable_validation=True`）
2. 在生产环境中根据数据质量要求选择是否启用验证
3. 在评估前确保数据格式正确
4. 检查实体边界是否一致
5. 确保相同位置的实体具有相同的类型
6. 使用测试脚本验证数据格式

## 配置选项

### NER模式配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strict_match` | bool | True | 是否使用严格匹配（精确匹配） |
| `enable_validation` | bool | False | 是否启用实体一致性验证 |

### 半结构化模式配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strict_match` | bool | True | 是否使用严格匹配（精确匹配） |
| `enable_validation` | bool | False | 是否启用实体一致性验证 |

**注意**: 两种模式的配置选项相同，但验证的内容不同：
- NER模式验证：start、end、type字段
- 半结构化模式验证：start、end、content字段 