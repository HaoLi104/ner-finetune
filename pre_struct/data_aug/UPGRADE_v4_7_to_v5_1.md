# 数据增强升级指南：v4_7 → v5_1

## 📌 概述

基于更新后的 `keys/keys_merged.json`，从 `data/clean_ocr_ppt_da_v4_7_recheck.json` 生成 v5_1 版本数据。

**初始输入：** `data/clean_ocr_ppt_da_v4_7_recheck.json`  
**最终输出：** `data/clean_ocr_ppt_da_v5_1_report_drop_keys.json`

---

## 🔧 需要修改的文件

### 1️⃣ `pre_struct/data_aug/run_da.py`

**位置：** 第 43-45 行

**修改前：**
```python
pipe = DataAugmentPipeline(
    in_path="data/clean_ocr_ppt_da_v4_6_report_drop_keys.json",
    out_path="data/clean_ocr_ppt_da_v4_7_origin.json",
```

**修改后：**
```python
pipe = DataAugmentPipeline(
    in_path="data/clean_ocr_ppt_da_v4_7_recheck.json",
    out_path="data/clean_ocr_ppt_da_v5_0_origin.json",
```

---

### 2️⃣ `pre_struct/data_aug/data_augmentation_recheck.py`

**位置：** 文件末尾 `if __name__ == "__main__":` 部分

**修改前：**
```python
llm_clean_fields_only(
    in_path="data/clean_ocr_ppt_da_v4_7_origin.json",
    out_path="data/clean_ocr_ppt_da_v4_7_origin_recheck.json",
```

**修改后：**
```python
llm_clean_fields_only(
    in_path="data/clean_ocr_ppt_da_v5_0_origin.json",
    out_path="data/clean_ocr_ppt_da_v5_0_origin_recheck.json",
```

---

### 3️⃣ `pre_struct/data_aug/compose_and_noise.py`

**位置：** 文件末尾主函数调用部分

**修改前：**
```python
compose_main(
    in_path="data/clean_ocr_ppt_da_v4_7_origin_recheck.json",
    out_drop_keys="data/clean_ocr_ppt_da_v4_7_report_drop_keys.json",
    out_alias="data/clean_ocr_ppt_da_v4_7_report_key_alias.json",
```

**修改后：**
```python
compose_main(
    in_path="data/clean_ocr_ppt_da_v5_0_origin_recheck.json",
    out_drop_keys="data/clean_ocr_ppt_da_v5_1_report_drop_keys.json",
    out_alias="data/clean_ocr_ppt_da_v5_1_report_key_alias.json",
```

---

## 🚀 执行方式

### 方式 1: 一键执行（推荐）

```bash
./pre_struct/data_aug/da_workflow.sh
```

### 方式 2: 逐步执行

```bash
# 步骤 1: 字段补充（约 10-30 分钟）
python pre_struct/data_aug/run_da.py

# 步骤 2: LLM 清洗（约 30-60 分钟）
python pre_struct/data_aug/data_augmentation_recheck.py

# 步骤 3: 报告组装（约 5-10 分钟）
python pre_struct/data_aug/compose_and_noise.py
```

---

## 📊 数据流图

```
data/clean_ocr_ppt_da_v4_7_recheck.json
         ↓
   [run_da.py]
   - 补充新字段（基于 keys/keys_merged.json）
   - 删除旧字段
   - 记录 added_keys
         ↓
data/clean_ocr_ppt_da_v5_0_origin.json
         ↓
   [data_augmentation_recheck.py]
   - LLM 清洗新增字段
   - 并发处理
         ↓
data/clean_ocr_ppt_da_v5_0_origin_recheck.json
         ↓
   [compose_and_noise.py]
   - 随机丢键
   - 别名覆盖
   - 组装 report
         ↓
data/clean_ocr_ppt_da_v5_1_report_drop_keys.json  ✅ 最终训练数据
data/clean_ocr_ppt_da_v5_1_report_key_alias.json  ✅ 别名版本
```

---

## ⚙️ 关键参数说明

### run_da.py
- `inc_synthesize_new_keys=True` - 补充新字段
- `reports_workers=8` - 样本级并发数
- `fields_workers=4` - 字段级并发数
- `inc_max_keys_per_record=4` - 每条记录最多补充 4 个字段

### data_augmentation_recheck.py
- `record_workers=6` - 记录级并发
- `inner_workers=12` - 单记录内字段并发

### compose_and_noise.py
- `drop_key_probs` - 各字段丢弃概率
- `alias_coverage_mode` - 别名覆盖策略

---

## 📝 检查清单

执行前确认：
- [ ] `keys/keys_merged.json` 已更新
- [ ] `data/clean_ocr_ppt_da_v4_7_recheck.json` 存在
- [ ] API_KEY 已配置（在 `conf.py` 或环境变量）
- [ ] LLM 服务可访问

执行后验证：
- [ ] `data/clean_ocr_ppt_da_v5_0_origin.json` 生成
- [ ] `data/clean_ocr_ppt_da_v5_0_origin_recheck.json` 生成
- [ ] `data/clean_ocr_ppt_da_v5_1_report_drop_keys.json` 生成
- [ ] 查看日志确认无报错

---

## 🔍 常见问题

**Q: 如何加速处理？**
A: 增加 `reports_workers` 和 `fields_workers` 参数（需确保 LLM 服务能承受并发）

**Q: 如何只处理特定报告类型？**
A: 修改 `run_da.py` 中的 `TARGET_REPORT_TYPES` 列表

**Q: 字段清洗失败怎么办？**
A: 检查 `.changes.jsonl` 文件查看详细错误信息

**Q: 如何跳过某个阶段？**
A: 可以单独执行某个脚本，但需确保输入文件路径正确

---

## 📞 支持

遇到问题请检查：
1. 日志输出（每个脚本都有详细进度）
2. 中间文件是否生成
3. LLM API 是否正常响应
