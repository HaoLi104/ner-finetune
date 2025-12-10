# Alias 字段使用说明

## 📝 概述

本文档说明了 `ebqa_title` 模块如何使用 `data/merged.converted.json` 中的 `alias` 字段来构建训练数据。

## 🎯 核心逻辑

### 数据结构

在 `merged.converted.json` 中，每条记录包含：

```json
{
  "report": "病人编号 01941583 检查技术 病人姓名 郑红英 病人性别 女 ...",
  "姓名": "郑红英",           // 标准键 -> 答案值（需要在报告中定位）
  "年龄": "54岁",
  "alias": {
    "姓名": "病人姓名",       // 标准键 -> 报告中的表达（用于生成问题）
    "年龄": "病人性别"
  }
}
```

### 关键点

- **标准键**（如 `"姓名"`）：用于标识字段
- **答案值**（如 `"郑红英"`）：需要在报告中定位的实际内容
- **Alias 值**（如 `"病人姓名"`）：报告中用于表述该字段的方式，用于生成问题

## 🔧 实现细节

### 1. 数据投影（`_project_alias_records`）

修改前的错误逻辑：
```python
for key, value in alias_map.items():
    rec_copy[key] = value  # ❌ 会把 "病人姓名" 覆盖 "郑红英"
```

修改后的正确逻辑：
```python
# ✅ 不覆盖原始字段值，保留答案用于定位
rec_copy[self._ALIAS_KEYS_FIELD] = list(alias_map.keys())
rec_copy[self._ALIAS_VALUE_MAP] = alias_map
# 原始字段值保持不变：rec["姓名"] = "郑红英"
```

### 2. 问题生成（`_format_question`）

```python
def _format_question(self, key: str, rec: Optional[Dict[str, Any]] = None) -> str:
    """生成问题，优先使用 alias 值"""
    display_key = key
    if rec:
        alias_value = self._alias_value(rec, key)  # 获取 "病人姓名"
        if alias_value:
            display_key = alias_value
    return self.question_template.format(key=display_key)
```

生成的问题示例：
```
找到文中和病人姓名类似的表达：
```

### 3. 答案定位（`_extract_spans_from_report`）

使用原始字段值（如 `"郑红英"`）在报告中查找答案位置：
```python
expected_map = {k: str(rec.get(k, "") or "").strip() for k in keys}
# expected_map["姓名"] = "郑红英"
spans = self._extract_spans_from_report(report, keys, expected_map=expected_map)
```

## ✅ 验证结果

运行 `test_alias_logic.py` 的结果：

```
3️⃣  验证原始字段值是否保留：
   - 姓名:
      原始值（答案）: '郑红英'
      Alias值（问题）: '病人姓名'
      ✓ 原始值已正确保留

4️⃣  测试问题生成：
   - 姓名: 找到文中和病人姓名类似的表达：

5️⃣  构建样本测试...
   - 定位到的答案: '郑红英'
   - 期望的答案: '郑红英'
   ✅ 答案定位正确！
```

## 🚀 使用方式

### 训练预计算样本

确保你的数据包含 `alias` 字段，然后运行：

```bash
python scripts/precompute_full.py
```

### 训练模型

使用预计算的 `.jsonl` 文件：

```bash
python pre_struct/ebqa_title/train_title.py
```

### 测试 alias 逻辑

```bash
python pre_struct/ebqa_title/test_alias_logic.py
```

## 📊 完整流程示例

对于这条记录：
```json
{
  "report": "病人姓名 郑红英 病人性别 女",
  "姓名": "郑红英",
  "alias": {"姓名": "病人姓名"}
}
```

处理流程：
1. **保留原始值**：`rec["姓名"]` = `"郑红英"`（不被覆盖）
2. **生成问题**：`"找到文中和病人姓名类似的表达："`
3. **在报告中定位答案**：查找 `"郑红英"` 的位置
4. **生成训练样本**：`(问题, 上下文) -> (start_pos, end_pos)` 指向 `"郑红英"`

## 🔍 关键修改文件

- `pre_struct/ebqa_title/da_core/dataset.py`
  - `_project_alias_records()`: 不覆盖原始字段值
  - `_format_question()`: 使用 alias 值生成问题
  - `_build_one_report()`: 传递 `rec` 参数给 `_format_question()`

## 📝 注意事项

1. **原始字段值不能丢失**：必须保留标准键的原始值，用于在报告中定位答案
2. **Alias 值用于问题**：alias 值仅用于生成更自然的问题，不参与答案定位
3. **兼容性**：如果某个字段没有 alias，会自动回退到使用标准键生成问题

## 🎉 总结

通过这次修改，`ebqa_title` 现在能够：
- ✅ 正确保留原始字段值用于答案定位
- ✅ 使用 alias 值生成更符合报告表述的问题
- ✅ 构建高质量的训练样本，提升模型泛化能力

