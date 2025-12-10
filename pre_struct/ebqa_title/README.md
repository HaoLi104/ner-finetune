# 字段别名识别模块

## 🎯 功能

识别标准字段名在医疗报告中的实际表达方式。

**示例**:
```
标准字段: "姓名"
报告中的表达: "患者姓名"、"病人姓名"、"病员姓名"等

输出: {"姓名": ["患者姓名", "病人姓名", "病员姓名"]}
```

---

## 📁 文件说明

### 数据准备脚本

1. **prepare_field_alias_data.py** - 完整版
   - 处理所有字段×所有报告
   - LLM调用: 20万+次
   - ⚠️ 成本高，耗时长

2. **prepare_field_alias_data_optimized.py** - 优化版 ✅推荐
   - 只处理Top 50常见字段
   - 每字段抽样3个报告
   - LLM调用: 150次
   - ✅ 成本低，快速

### 已有代码

- `train_title.py` - 训练脚本
- `test_title.py` - 测试脚本
- `evaluate_title.py` - 评估脚本
- `dataset.py` - 数据处理

---

## 🚀 使用方法

### 1. 准备数据（使用LLM）

```bash
# 配置LLM API
export OPENAI_API_KEY="your-api-key"

# 运行优化版（推荐）
python pre_struct/ebqa_title/prepare_field_alias_data_optimized.py \
  --llm-url "https://api.openai.com/v1/" \
  --llm-model "gpt-4"

# 或使用本地LLM
python pre_struct/ebqa_title/prepare_field_alias_data_optimized.py \
  --llm-url "http://localhost:8000/v1/" \
  --llm-model "qwen3-32b"
```

### 2. 查看生成的数据

```bash
cat data/field_alias_dataset.json
```

格式:
```json
{
  "姓名": ["患者姓名", "病人姓名", "姓名"],
  "年龄": ["年龄", "患者年龄", "入院年龄"],
  "性别": ["性别", "患者性别"],
  ...
}
```

### 3. 使用数据

**方式1**: 补充到keys_merged.json
- 将识别出的别名添加到对应字段的"别名"字段

**方式2**: 训练字段名识别模型
- 使用已有的train_title.py训练
- 输入: 报告文本
- 输出: 标准字段名

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | data/project-1.converted.json | 输入报告 |
| `--output` | data/field_alias_dataset.json | 输出数据集 |
| `--llm-url` | None | LLM API地址 |
| `--llm-model` | qwen3-32b | LLM模型名称 |
| `--report-limit` | None | 限制报告数（调试用） |

---

## 📊 优化策略说明

### 为什么只处理Top 50字段？

从`data/project-1.converted.json`分析:
- 常见字段（如姓名、年龄）出现在大部分报告中
- 罕见字段可能只在1-2个报告中
- Top 50字段覆盖了80-90%的实际需求

### 为什么每字段只抽样3个报告？

- 同一字段的别名通常不会太多（2-5种）
- 3个报告足够发现主要别名
- 更多报告收益递减

### LLM调用成本

```
优化版:
  Top 50字段 × 3个报告 = 150次调用
  预计成本: $0.5-1（使用GPT-4）
  预计时间: 5-10分钟

完整版:
  200+字段 × 1090个报告 = 20万+次调用
  预计成本: $500+（不推荐）
```

---

## 💡 使用建议

1. **先小规模测试**
   ```bash
   python prepare_field_alias_data_optimized.py --report-limit 10
   ```

2. **验证LLM返回格式**
   - 确保返回的是JSON
   - 格式: `{"字段名": "别名"}`

3. **完整运行**
   ```bash
   python prepare_field_alias_data_optimized.py
   ```

4. **检查结果**
   - 查看生成的`field_alias_dataset.json`
   - 验证别名是否合理

---

## 🔑 关键点

- ✅ LLM返回JSON格式: `{"姓名": "患者姓名"}`
- ✅ 不包含冒号，只有字段名
- ✅ 优化版成本可控（150次调用）
- ⚠️ 需要配置LLM API

---

**配置好LLM API后即可运行！** 🚀


