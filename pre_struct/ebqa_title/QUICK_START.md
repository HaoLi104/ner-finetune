# 快速开始 - 训练 Alias/Title QA 模型

## ⚡ 一键训练

```bash
cd /mnt/windows/wy/ner-bert-crf

# 方式1: 直接运行
python pre_struct/ebqa_title/train_title.py

# 方式2: 后台运行（推荐）
nohup python pre_struct/ebqa_title/train_title.py > train.log 2>&1 &

# 查看训练进度
tail -f train.log
```

---

## 📊 训练数据

- **预计算文件**: `data/merged.converted.alias_title.jsonl`
- **总样本数**: ~18,952+
- **配置文件**: `pre_struct/ebqa_title/merged_config.json`

---

## 🎯 优化的训练参数

### 核心参数
| 参数 | 值 | 说明 |
|------|---|------|
| Epochs | 8 | 训练轮数 |
| Batch Size | 8 | 单卡批次大小 |
| Accumulation | 4 | 有效batch=32 |
| Learning Rate | 2e-5 | 学习率 |
| Warmup | 15% | 预热比例 |

### 样本平衡
| 参数 | 值 | 说明 |
|------|---|------|
| Weighted Sampler | ✅ | 加权采样 |
| Negative Keep | 100% | 保留全部负样本 |
| Short Field Weight | 2.5 | 短字段权重 |

### 正则化
| 参数 | 值 | 说明 |
|------|---|------|
| Label Smoothing | 0.1 | 标签平滑 |
| Weight Decay | 0.01 | 权重衰减 |
| Null Margin | 0.15 | 负样本边界 |
| Null Weight | 0.05 | 负样本权重 |

---

## ⏱️ 预期时长

- **单epoch**: 5-10分钟
- **总时长**: 40-80分钟（8轮）
- **可能提前停止**: 5-6轮后

---

## 📈 预期效果

| 指标 | 目标值 |
|------|--------|
| Token F1 | > 85% |
| Exact Match | > 75% |
| Train Loss | < 0.5 |
| Eval Loss | < 0.6 |

---

## 📁 输出文件

```
runs/ebqa_title_merged/
├── best/                    # 最佳模型 ⭐
├── checkpoint-epoch-*/      # 每轮checkpoint
├── training_curves.png      # 训练曲线
├── metrics_history.json     # 训练指标
└── train.log               # 训练日志
```

---

## 🔧 显存不足？

如果遇到 OOM 错误，修改 `merged_config.json`:

```json
"per_device_batch_size": 4,    // 8 → 4
"grad_accum_steps": 8,         // 4 → 8
```

---

## ✅ 训练完成后

### 1. 查看训练曲线
```bash
# 图片位置
runs/ebqa_title_merged/training_curves.png
```

### 2. 评估模型
```bash
python pre_struct/ebqa_title/evaluate_title.py
```

### 3. 测试推理
```bash
python pre_struct/ebqa_title/test_title.py
```

---

## 📚 详细文档

完整参数说明见: [TRAIN_GUIDE.md](TRAIN_GUIDE.md)

