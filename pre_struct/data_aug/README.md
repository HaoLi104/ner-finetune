# 数据增强流水线

## 快速开始

```bash
# 一键运行完整流水线（默认写入 data/$(date +%Y%m%d)）
./da_workflow.sh
```

## 流程说明

### 三阶段处理
1. **run_da.py** - 数据增强：过滤指定报告类型，智能增强到目标数量
2. **data_augmentation_recheck.py** - 字段清洗：LLM并发清洗新增字段  
3. **compose_and_noise.py** - 报告组装：生成最终格式化报告

### 数据流
```
DATE_TAG=$(date +%Y%m%d)
BASE_DIR=data/${DATE_TAG}

input: ${BASE_DIR}/clean_ocr_ppt_da_v4_7_recheck.json
  ↓ (过滤+增强)
${BASE_DIR}/clean_ocr_ppt_da_v4_7_recheck_balanced.json
  ↓ (LLM清洗)
${BASE_DIR}/clean_ocr_ppt_da_v5_0_field_cleaned.json
  ↓ (报告组装)
${BASE_DIR}/clean_ocr_ppt_da_v5_0_report_*.json
```

## 配置调整

### 报告类型和数量
```python
# run_da.py
TARGET_REPORT_TYPES = ["入院记录"]  # 指定类型
MAX_SAMPLES_PER_TYPE = 1000        # 目标数量
```

### 并发参数
```python
# run_da.py
reports_workers=8,    # 报告级并发
fields_workers=2,     # 字段级并发

# data_augmentation_recheck.py  
RECORD_WORKERS = 4    # 记录级并发
INNER_WORKERS = 5     # 字段级并发
```

## 单独运行

```bash
# 建议先固定日期
export DA_DATE_TAG=$(date +%Y%m%d)

# 步骤1: 数据增强
python pre_struct/data_aug/run_da.py

# 步骤2: 字段清洗
python pre_struct/data_aug/data_augmentation_recheck.py

# 步骤3: 报告组装  
python pre_struct/data_aug/compose_and_noise.py
```
