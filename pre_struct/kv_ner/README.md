# KV-NER 管道

基于 **BERT + BiLSTM + CRF** 的序列标注模型，用于医疗报告的键值对提取。支持长文本分块处理、多文件训练、键值对级别评估。

## 结构图

```
           +------------------+     +-------------------+     +------------------+
           |  数据准备阶段    | --> |   模型训练阶段    | --> |   模型评估阶段   |
           +------------------+     +-------------------+     +------------------+
                    |                         |                       |
         +----------+----------+    +---------+----------+   +--------+---------+
         |  prepare_data.py    |    |   train.py         |   |  evaluate.py     |
         |                     |    |                    |   |                  |
         |  多文件数据合并      |    |  BERT+BiLSTM+CRF   |   |  键值对级别评估   |
         |  训练/验证集划分     |    |  分块训练支持       |   |  多种匹配方式     |
         |  格式转换          |    |  混合精度训练       |   |  错误样本分析     |
         +---------------------+    |  梯度检查点        |   +------------------+
                                    |  差分学习率        |
                                    +---------+----------+
                                              |
                                   +----------+-----------+
                                   |   模型推理阶段        |
                                   +----------+-----------+
                                              |
                             +----------------+-----------------+
                             |  predict.py  |  test_kv_ner.py  |
                             |              |                  |
                             | 批量推理      | 单条样本测试      |
                             +----------------------------------+
```

## 微调框架详解

KV-NER 微调框架是一个完整的序列标注训练和推理系统，专为医疗报告中的键值对提取任务设计。该框架基于 BERT + BiLSTM + CRF 架构，支持长文本处理、多文件训练和键值对级别评估。

### 框架特点

1. **完整的训练流水线**：
   - 数据准备、模型训练、模型评估、模型推理四个阶段
   - 每个阶段都有独立的脚本，支持分步执行或一键运行
   - 支持配置文件驱动，便于实验管理和参数调整

2. **灵活的模型架构**：
   - 基于 BERT 的编码器-解码器架构
   - 可选 BiLSTM 层增强序列特征提取能力
   - CRF 层保证标签序列的有效性
   - 支持差分学习率、混合精度训练、梯度检查点等优化技术

3. **长文本处理能力**：
   - 自动分块处理超长文本（>512 tokens）
   - 块间重叠确保实体边界不被截断
   - 预测结果智能合并去重

4. **多文件训练支持**：
   - 支持同时加载多个 Label Studio 标注文件
   - 自动合并不同来源的数据
   - 保持数据分布的完整性

5. **键值对级别评估**：
   - 与 EBQA 使用相同的 evaluation 库
   - 支持 Position/Text Exact/Text Overlap 三种匹配方式
   - 提供详细的错误样本分析

### 核心特性

- **BIO 标注方案** - KEY（键）、VALUE（值）、HOSPITAL（医院名称）
- **BERT + BiLSTM + CRF** - BERT 编码 + BiLSTM 特征增强 + CRF 解码，强制有效标签转移
- **长文本分块处理** - 自动分块、预测、合并，支持任意长度文本
- **多文件训练** - 在配置文件中指定多个 Label Studio 导出文件
- **键值对级别评估** - 与 EBQA 使用相同的 evaluation 库
- **完整工具链** - 数据准备、训练、评估、测试、诊断

## 文件说明

### 配置文件

| 文件 | 说明 |
|------|------|
| [kv_ner_config.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config.json) | **主配置文件**。包含：<br>• 模型路径和分词器路径<br>• 标签映射（键名→KEY, 值→VALUE, 医院名称→HOSPITAL）<br>• 序列长度和分块参数（max_seq_length=512, chunk_size=500, chunk_overlap=50）<br>• 数据准备配置（input_files 支持多文件，train_ratio, seed）<br>• 训练超参数（batch_size, learning_rate, epochs, dropout 等）<br>• 推理配置（input_path, output_path） |
| [kv_ner_config_with_bilstm.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config_with_bilstm.json) | **带BiLSTM增强的配置文件**，启用BiLSTM层和多种辅助损失函数 |
| [kv_ner_config_without_bilstm.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config_without_bilstm.json) | **不带BiLSTM的配置文件**，仅使用BERT+CRF基础架构 |
| [kv_ner_config.example.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config.example.json) | **配置示例文件**，带详细注释，方便理解各参数含义 |

### 核心脚本

| 文件 | 说明 |
|------|------|
| **[prepare_data.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/prepare_data.py)** | **数据准备脚本**。功能：<br>• 读取配置文件中的多个 Label Studio 导出文件<br>• 自动合并所有数据<br>• **不做任何键的过滤**（只要有标注就保留）<br>• 自动提取 KEY-VALUE 关系<br>• 划分训练集（90%）和验证集（10%）<br>• 生成 train.json, val.json, val_eval.jsonl<br>• 支持命令行参数覆盖配置 |
| **[train.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/train.py)** | **模型训练脚本**。功能：<br>• 读取 train.json 作为训练集<br>• 从 val.json 动态划分验证集和测试集（避免重复划分）<br>• BERT + BiLSTM + CRF 端到端训练<br>• **三层进度条**：Epoch 级别显示 best_f1/val_f1/loss，Batch 级别显示 loss/lr<br>• 自动保存最佳模型（基于验证 F1）<br>• 支持 **safetensors** 格式保存<br>• 训练结束后在测试集最终评估<br>• 生成 training_summary.json |
| **[evaluate.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/evaluate.py)** | **键值对级别评估脚本**。功能：<br>• 与 EBQA 使用**完全相同的 evaluation 库**<br>• 读取 val_eval.jsonl（JSONL 键值对格式）<br>• 三种匹配方式：Position / Text Exact / Text Overlap<br>• **报告类型过滤**（默认：入院记录、门诊病历、术后病理）<br>• 支持对齐模式（gold/pred/union）和排除键<br>• **分块预测**支持长文本<br>• 错误样本包含 key_value_pairs 详细信息<br>• 生成 eval_summary.json 和 error_samples.jsonl |
| [predict.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/predict.py) | **批量推理脚本**。读取配置中的 input_path，对所有样本进行预测，输出到 predictions.json |
| **[test_kv_ner.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/test_kv_ner.py)** | **单条样本测试工具**。在 `main` 函数中直接修改 `REPORT_TEXT`，运行后输出结构化 JSON。支持分块处理长文本 |
| **[diagnose_training.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/diagnose_training.py)** | **训练诊断工具**。功能：<br>• 检查配置参数是否合理<br>• 统计数据分布（样本数、标注分布）<br>• 计算样本长度分布和**截断率**<br>• 检查模型参数量<br>• 给出针对性优化建议<br>• 建议合适的 unfreeze_last_n_layers 值 |
| [export_test_spans.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/export_test_spans.py) | **手动导出评估数据工具**。从 Label Studio 格式导出 JSONL 键值对格式，通常不需要单独使用（prepare_data.py 已自动生成） |
| [run_all.sh](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/run_all.sh) | **一键运行脚本**。自动执行：数据准备 → 训练 → 评估 → 推理 |

### 核心模块

| 文件 | 说明 |
|------|------|
| **[chunking.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/chunking.py)** | **长文本分块处理模块**。功能：<br>• 智能分块（chunk_size=500, overlap=50）<br>• 自动预测每个 chunk<br>• 合并结果并去重（避免重叠区域重复）<br>• 支持任意长度文本 |
| [modeling.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/modeling.py) | **BERT + BiLSTM + CRF 模型**。包含：<br>• BertCrfTokenClassifier 类<br>• 支持 freeze_encoder 和 unfreeze_last_n_layers<br>• 支持 BiLSTM 特征增强层<br>• 支持 Conv1D 残差连接<br>• 支持多种辅助损失函数（边界损失、token级CE损失等）<br>• save/load 支持 **safetensors** 格式<br>• 自动保存 label2id.json 和 model_config.json |
| [dataset.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/dataset.py) | **PyTorch 数据集**。TokenClassificationDataset + collate_batch，处理 tokenization 和标签对齐 |
| [data_utils.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/data_utils.py) | **数据工具函数**。load_labelstudio_export（读取 LS 格式）、build_bio_label_list、split_samples 等 |
| [metrics.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/metrics.py) | **评估指标计算**。char_spans（字符级 span 提取）、compute_ner_metrics（精准度/召回率/F1） |
| [config_io.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/config_io.py) | **配置读取工具**。load_config、model_name_from、label_map_from 等辅助函数 |

### 辅助文件

| 文件 | 说明 |
|------|------|
| [README.md](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/README.md) | 本文档 |
| [evaluation_guide.md](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/evaluation_guide.md) | 详细的评估指南，包含命令行参数、输出格式、故障排除等 |
| `flash_attn/` | Flash Attention 兼容性 stub（CPU 环境支持） |

## 架构详解

### 模型架构

KV-NER采用三层架构设计：

1. **BERT编码层**：使用预训练BERT模型作为基础特征提取器
2. **BiLSTM增强层**：通过双向LSTM进一步提取序列特征（可选但默认启用）
3. **CRF解码层**：使用条件随机场保证标签转移的有效性

模型特点：
- 支持BiLSTM层（默认启用），3层，每层384个隐藏单元
- 支持多种辅助损失函数，包括边界检测损失和token级交叉熵损失
- 支持差分学习率训练（BERT编码器和头部层可设置不同学习率）
- 支持混合精度训练（BF16/FP16）
- 支持梯度检查点以节省显存

### 数据处理

- **多源数据融合**：支持多个来源的 Label Studio 标注数据同时训练
- **自动合并**：不同来源的数据自动合并，形成统一的训练语料库
- **键名保留**：不对键名进行过滤，保留所有标注过的键值对

### 训练策略

- **分块处理**：将超过最大序列长度(512)的文本切分为多个重叠块
- **重叠处理**：相邻块之间保留50个token的重叠，防止实体被截断
- **结果合并**：预测完成后合并所有块的结果，并去除重复实体
- **差分学习率**：支持为BERT编码器和头部层设置不同学习率
- **混合精度训练**：支持BF16/FP16混合精度训练以提高训练效率

### 评估方法

- **位置匹配(Position)**：要求预测和真实标签的位置完全一致（最严格）
- **文本精确匹配(Text Exact)**：要求预测和真实标签的文本完全一致
- **文本重叠匹配(Text Overlap)**：要求预测和真实标签有一定重叠（最宽松）

## 快速开始

### 方式 1：一键运行

```bash
# 1. 编辑配置文件，设置 data.input_files
vi pre_struct/kv_ner/kv_ner_config.json

# 2. 运行完整流程
bash pre_struct/kv_ner/run_all.sh
```

### 方式 2：分步执行

```bash
# 1. 数据准备
python pre_struct/kv_ner/prepare_data.py --config pre_struct/kv_ner/kv_ner_config.json

# 2. 训练
python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json

# 3. 评估（使用 val_eval.jsonl）
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --test_data data/kv_ner_prepared/val_eval.jsonl

# 4. 单条测试（编辑 test_kv_ner.py 的 main 函数）
python pre_struct/kv_ner/test_kv_ner.py
```

### 方式 3：诊断优先

```bash
# 先诊断配置和数据
python pre_struct/kv_ner/diagnose_training.py

# 根据建议调整配置，然后训练
python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json
```

## 详细使用指南

### 数据准备

#### 配置多文件输入

编辑 [kv_ner_config.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config.json)：

```json
{
  "data": {
    "input_files": [
      "data/ruyuanjilu/ruyuan.json",
      "data/menzhenbingli/menzhen.json",
      "data/shuhoubingli/shuhou.json"
    ],
    "train_ratio": 0.9
  }
}
```

#### 运行数据准备

```bash
python pre_struct/kv_ner/prepare_data.py --config pre_struct/kv_ner/kv_ner_config.json
```

**生成文件：**
- [train.json](file:///mnt/windows/wy/ner-bert-crf/data/kv_ner_prepared/train.json) - 训练集（90%）
- [val.json](file:///mnt/windows/wy/ner-bert-crf/data/kv_ner_prepared/val.json) - 验证+测试池（10%）
- [val_eval.jsonl](file:///mnt/windows/wy/ner-bert-crf/data/kv_ner_prepared/val_eval.jsonl) - 评估数据（键值对格式）
- [data_summary.json](file:///mnt/windows/wy/ner-bert-crf/data/kv_ner_prepared/data_summary.json) - 数据统计

**特点：**
- ✅ 自动合并多个文件
- ✅ 不做键过滤（只要有标注就保留）
- ✅ 自动提取 KEY-VALUE 关系

### 模型训练

```bash
python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json
```

**数据划分：**
```
train.json (90%) → 训练集
val.json (10%)   → 动态划分为：
  ├─ 验证集 (5%) - 每个 epoch 后评估，选最佳模型
  └─ 测试集 (5%) - 训练结束后最终评估
```

**进度条显示：**
```
Training:  60%|██████    | 3/5 [15:32<10:21, best_f1=0.8456, val_f1=0.8234, loss=45.23]
Epoch 3/5: 45%|████▌     | 442/982 [02:15<02:45, loss=45.23, lr=2.5e-05]
```

**输出：**
- `runs/kv_ner4_bioe/best/` - 最佳模型（safetensors 格式）
- [runs/kv_ner4_bioe/training_summary.json](file:///mnt/windows/wy/ner-bert-crf/runs/kv_ner4_bioe/training_summary.json) - 训练历史和指标

### 模型评估

#### 评估已有验证集

```bash
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --test_data data/kv_ner_prepared/val_eval.jsonl
```

#### 评估新数据集

**步骤 1：将新数据集转换为评估格式**

```bash
# 方式 1：使用 prepare_data.py（推荐）
python pre_struct/kv_ner/prepare_data.py \
  --input data/new_dataset/new_data.json \
  --output_dir data/new_dataset \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --test-only

# 方式 2：使用 export_test_spans.py
python pre_struct/kv_ner/export_test_spans.py \
  --input data/new_dataset/new_data.json \
  --output data/new_dataset/val_eval.jsonl \
  --config pre_struct/kv_ner/kv_ner_config.json
```

**步骤 2：运行评估**

```bash
python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --test_data data/new_dataset/val_eval.jsonl
```

**评估方式：**
- Position Matching - 位置完全匹配（最严格）
- Text Exact Matching - 文本精确匹配
- Text Overlap Matching - 文本重叠匹配（最宽松）

**报告类型过滤：** 默认评估所有类型（可通过 `--report_titles` 参数指定）

**输出：**
- `data/kv_ner_eval_bioe/eval_summary.json` - 评估指标
- `data/kv_ner_eval_bioe/error_samples.jsonl` - 错误样本（含 key_value_pairs）

### 模型推理

**批量推理：**
```bash
python pre_struct/kv_ner/predict.py --config pre_struct/kv_ner/kv_ner_config.json
```

**单条测试：**
编辑 [test_kv_ner.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/test_kv_ner.py) 的 `REPORT_TEXT`，然后运行：
```bash
python pre_struct/kv_ner/test_kv_ner.py
```

### 对比实验记录（填写中）

为便于直接复现，每一行给出“训练命令 + 评估命令 + 关键配置改动”。F1 默认为 KV 级评估的 Text Exact F1（在 `eval_summary.json` 中查看）。

<!-- EXPERIMENT_TABLE_START -->
| 方法/设置 | 训练命令 | 评估命令 | 关键配置改动 | F1(Exact) | 备注 |
|---|---|---|---|---:|---|
| Bert+CRF（基线） | `python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_without_bilstm.json` | `python pre_struct/kv_ner/evaluate.py --config pre_struct/kv_ner/kv_ner_config_without_bilstm.json --model_dir runs/kv_ner4_bioe_no_bilstm/best --test_data data/kv_ner_prepared/val_eval.jsonl --output_dir runs/kv_ner4_bioe_no_bilstm/eval` | `use_bilstm=false` |  | 当前稳定基线 |
| BiLSTM（2 层, 384） | `python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_with_bilstm.json` | `python pre_struct/kv_ner/evaluate.py --config pre_struct/kv_ner/kv_ner_config_with_bilstm.json --model_dir runs/kv_ner4_bioe_bilstm/best --test_data data/kv_ner_prepared/val_eval.jsonl --output_dir runs/kv_ner4_bioe_bilstm/eval` | `use_bilstm=true, lstm_num_layers=2, lstm_hidden_size=384` |  | 主对比：是否引入 BiLSTM |
| BiLSTM（1 层） | 同上 | 同上（输出目录改名） | 仅改 `lstm_num_layers=1` |  | 轻量 BiLSTM |
| BiLSTM（3 层） | 同上 | 同上（输出目录改名） | 仅改 `lstm_num_layers=3` |  | 更深 BiLSTM（视显存） |
| KEY 边界权重 ↑ | 基线命令 | 基线评估命令 | `boundary_loss_weight=0.2, boundary_positive_weight=3.0` |  | 强化 B-KEY 召回/对齐 |
<!-- EXPERIMENT_TABLE_END -->

辅助：一键跑四个常用变体（base/conv/bilstm/bilstm_conv）

```bash
bash scripts/kv_ner_ablation.sh
```

运行完后，可用下面的一行命令快速收集每个变体的 F1（Exact/Overlap）：

```bash
python - <<'PY'
import json,glob,os
base='runs/kv_ner_ablate'
for p in sorted(glob.glob(os.path.join(base,'*','eval','eval_summary.json'))):
    name=p.split(os.sep)[2]
    d=json.load(open(p,'r',encoding='utf-8'))
    ex=d.get('text_exact',{}).get('f1_score',0)
    ov=d.get('text_overlap',{}).get('f1_score',0)
    print(f"{name:14s}  Exact={ex:.4f}  Overlap={ov:.4f}  -> {os.path.dirname(p)}")
PY
```

提示：若你希望将 Overlap F1 也纳入表格，可把上述输出粘贴到“备注”列。

单实验文件夹（便于留存记录）

每个实验在 `scripts/experiments/<name>/` 下包含一份独立的 `config.json` 与 `run.sh`：

- bilstm_1：BiLSTM 1 层（384）
  - 运行：`bash scripts/experiments/bilstm_1/run.sh`
- bilstm_2：BiLSTM 2 层（384）
  - 运行：`bash scripts/experiments/bilstm_2/run.sh`
- bilstm_3：BiLSTM 3 层（384）
  - 运行：`bash scripts/experiments/bilstm_3/run.sh`
- key_boundary_up：提高 KEY 边界权重（boundary_loss_weight/positive_weight）
  - 运行：`bash scripts/experiments/key_boundary_up/run.sh`

以上脚本会先训练再用 val_eval.jsonl 做评估，并把结果写到 `runs/experiments/<name>/` 目录下，便于长期留存与对比。也可一键运行全部实验并回写表格：

```bash
bash scripts/experiments/run_all.sh
```

## 配置参数详解

### 分块处理参数

```json
{
  "max_seq_length": 512,      // BERT 最大长度（不要改）
  "chunk_size": 500,          // 分块大小（tokens）
  "chunk_overlap": 50         // 重叠大小（避免边界实体被切断）
}
```

**工作原理：**
```
长文本 (1000 tokens)
  → chunk1 [0-500]
  → chunk2 [450-950]  (重叠 50)
  → chunk3 [900-1000] (重叠 50)
  → 合并去重 → 完整实体
```

### 训练超参数

```json
{
  "train": {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,    // 有效 batch=8
    "learning_rate": 2e-5,               // BERT标准学习率
    "encoder_learning_rate": 2e-5,       // BERT编码器学习率
    "head_learning_rate": 3e-4,          // 头部层学习率
    "num_train_epochs": 4,               // 充分训练 CRF
    "warmup_ratio": 0.1,                 // 平滑预热
    "dropout": 0.2,                      // 防止过拟合
    "weight_decay": 0.03,                // 权重衰减
    "use_amp": true,                     // 启用混合精度训练
    "amp_dtype": "bf16",                 // 混合精度类型
    "grad_checkpointing": true,          // 启用梯度检查点
    "unfreeze_last_n_layers": null,      // 全量训练
    "use_bilstm": true,                  // 使用BiLSTM增强
    "lstm_hidden_size": 384,             // LSTM隐藏单元数
    "lstm_num_layers": 3,                // LSTM层数
    "lstm_dropout": 0.2,                 // LSTM Dropout
    "use_conv": true,                    // 使用Conv1D残差连接
    "conv_kernel_sizes": [3, 5],         // 卷积核大小
    "conv_dropout": 0.1,                 // Conv Dropout
    "token_ce_loss_weight": 0.3,         // Token CE损失权重
    "token_ce_value_class_weight": 5.0,  // VALUE类别权重
    "token_ce_label_smoothing": 0.1,     // 标签平滑因子
    "boundary_loss_weight": 0.1,         // 边界损失权重
    "boundary_positive_weight": 2.0,     // 边界正样本权重
    "end_boundary_loss_weight": 0.05,    // 结束边界损失权重
    "end_boundary_positive_weight": 2.0  // 结束边界正样本权重
  }
}
```

### 数据划分参数

```json
{
  "data": {
    "train_ratio": 0.9              // 训练集 90%，验证池 10%
  },
  "train": {
    "test_split_ratio": 0.5         // 从验证池划分 50% 作测试集
  }
}
```

## 工作流程

```
1. 配置 kv_ner_config.json
   ├─ data.input_files: 多个 Label Studio 文件
   └─ 训练超参数

2. prepare_data.py
   ├─ 合并多文件
   ├─ 提取 KEY-VALUE 关系
   ├─ 划分 train + val
   └─ 生成 val_eval.jsonl

3. train.py
   ├─ 读取 train.json
   ├─ 从 val.json 动态划分 val + test
   ├─ 训练（支持分块）
   └─ 保存最佳模型

4. evaluate.py
   ├─ 使用 val_eval.jsonl
   ├─ 键值对级别评估
   ├─ 三种匹配方式
   └─ 生成评估报告

5. test_kv_ner.py
   └─ 单条样本快速测试
```

## 常见问题

**Q: 为什么 valid_tasks 比 total_tasks 少？**

A: 自动过滤无效数据：无标注、标注取消、没有 KEY/VALUE/HOSPITAL 标签的任务。这是正常的。

**Q: 用于评估的测试文件是哪个？**

A: **[data/kv_ner_prepared/val_eval.jsonl](file:///mnt/windows/wy/ner-bert-crf/data/kv_ner_prepared/val_eval.jsonl)** - 用于与 EBQA 对比的标准评估数据。

**Q: Text Exact 和 Text Overlap 差距大怎么办？**

A: 说明边界不精确。已通过以下方式优化：
- 增加边界检测辅助损失
- 引入BiLSTM特征增强层
- 使用VALUE类别权重平衡

**Q: 如何处理长文本？**

A: 自动分块处理，无需手动操作。配置 `chunk_size=500, chunk_overlap=50`，支持任意长度文本。

**Q: 如何添加多个数据文件？**

A: 在配置文件的 `data.input_files` 数组中添加路径。

**Q: unfreeze_last_n_layers 设置多少？**

A: 
- 数据 < 1000 条：`2-4`
- 数据 1000-5000 条：`null`（全量，推荐）
- 数据 > 5000 条：`null`

**Q: GPU 内存不足怎么办？**

A:
```json
{
  "train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "unfreeze_last_n_layers": 4,
  "use_amp": true,
  "grad_checkpointing": true
}
```

**Q: 训练效果不理想？**

A: 
1. 运行 [diagnose_training.py](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/diagnose_training.py) 检查问题
2. 检查样本截断率（应 < 30%）
3. 查看 error_samples.jsonl 分析错误模式
4. 使用 test_kv_ner.py 测试单条样本

## 输出格式

**结构化 JSON：**
```json
{
  "structured": {
    "姓名": "张三",
    "年龄": "45岁",
    "主诉": "发现左侧乳房肿块3月余"
  },
  "pairs": [
    {
      "key": {"text": "姓名", "start": 15, "end": 17},
      "values": [{"text": "张三", "start": 18, "end": 20}],
      "value_text": "张三"
    }
  ],
  "entities": [...],
  "hospital": [...]
}
```

## 诊断和调试

```bash
# 诊断配置和数据
python pre_struct/kv_ner/diagnose_training.py

# 测试单条样本（编辑 main 函数中的文本）
python pre_struct/kv_ner/test_kv_ner.py

# 查看数据统计
cat data/kv_ner_prepared/data_summary.json

# 查看错误样本
head -1 data/kv_ner_eval_bioe/error_samples.jsonl | python -m json.tool
```

## 详细文档

- [evaluation_guide.md](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/evaluation_guide.md) - 评估详细指南（命令行参数、输出格式、故障排除）
- [kv_ner_config.example.json](file:///mnt/windows/wy/ner-bert-crf/pre_struct/kv_ner/kv_ner_config.example.json) - 配置文件注释版

## 与 EBQA 的区别

| 特性 | EBQA | KV-NER |
|------|------|--------|
| 方法 | 问答模型 | 序列标注（BERT+BiLSTM+CRF） |
| 提取方式 | 逐字段提取 | 一次提取所有字段 |
| 长文本处理 | 滑动窗口 | 分块+合并 |
| 评估标准 | evaluation 库 | **相同的 evaluation 库** ✅ |
| 评估数据格式 | JSONL 键值对 | **相同格式** ✅ |

**可以直接对比 Position F1 分数！**
