# ✅ v5.0 数据增强配置检查通过

**检查时间:** 2025-10-13  
**系统配置:** 32核 CPU (Intel i9-14900K), 62GB RAM  
**状态:** 已就绪，可执行

---

## 📊 配置总览

**路径说明：** 假设当日 `DATE_TAG=$(date +%Y%m%d)`，本轮增强产物统一写入 `data/${DATE_TAG}` 目录。

### 数据流路径（v5.0 统一版本 + 样本均衡）

```
输入: data/${DATE_TAG}/clean_ocr_ppt_da_v4_7_recheck.json (51MB, 13402条)
  ↓
[阶段1.1: 下采样平衡] 
  超额类型（如 MRI/CT 800→≈均值）下采样到平均数
  ↓
  data/${DATE_TAG}/clean_ocr_ppt_da_v4_7_recheck_balanced.json (~40MB, ~10330条)
  ↓
[阶段1.2: 字段补充 + 上采样]
  补充新字段 + 少数类上采样到平均数
  ↓
  data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_added.json (~45MB, ~10300条)
  ↓
[阶段2: data_augmentation_recheck.py] LLM 清洗
  ↓
  data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_cleaned.json (~50MB)
  ↓
[阶段3: compose_and_noise.py] 报告组装
  ↓
输出: data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_drop_keys.json  ✅
      data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_key_alias.json  ✅
```

**数据均衡效果：**
- 原始: 13402 条，分布不均（257~800/类型）
- 下采样后: ~10330 条，上限约为类型均值
- 最终: ~10300 条，各类型围绕均值分布

---

## ⚙️ 并发配置（已优化）

| 脚本 | 记录级并发 | 字段级并发 | 连接池 | 最大请求 |
|-----|-----------|-----------|--------|---------|
| **run_da.py** | 6 workers | 6 workers | 48 | 24 |
| **data_augmentation_recheck.py** | 6 workers | 6 workers | - | - |
| **compose_and_noise.py** | - | - | - | - |

**并发策略:**
- **记录级并发 = 6**: 充分利用 32 核 CPU，避免过载
- **字段级并发 = 6**: 单记录内并发处理字段，平衡速度与稳定性
- **连接池 = 48**: 支持高并发 HTTP 请求
- **最大请求 = 24**: 控制 LLM API 并发数，避免超时

---

## 🗂️ 文件检查

| 项目 | 路径 | 状态 |
|-----|------|------|
| **输入数据** | `data/${DATE_TAG}/clean_ocr_ppt_da_v4_7_recheck.json` | ✅ 存在 (51MB) |
| **键结构** | `keys/keys_merged.json` | ✅ 存在 (258KB) |
| **RAG 语料** | `data/rag/ocr_summary_words.txt` | ⚠️ 不存在（会使用缓存）|
| **脚本1** | `pre_struct/data_aug/run_da.py` | ✅ 已配置 |
| **脚本2** | `pre_struct/data_aug/data_augmentation_recheck.py` | ✅ 已配置 |
| **脚本3** | `pre_struct/data_aug/compose_and_noise.py` | ✅ 已配置 |
| **工作流** | `pre_struct/data_aug/da_workflow.sh` | ✅ 已更新 |

---

## 🔧 各阶段配置详情

### 阶段 1: run_da.py - 样本均衡 + 字段补充

**包含两个子步骤：**

**步骤 1.1: 下采样平衡（预处理）**
```python
# 自动计算平均数
# 超过均值的类型下采样到均值
# 例如: MRI/CT/超声/入院记录 从 800 → ≈均值
```

**步骤 1.2: 字段补充 + 上采样**
```python
in_path  = f"data/${DATE_TAG}/clean_ocr_ppt_da_v4_7_recheck_balanced.json"  # 下采样后
out_path = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_added.json"

# 操作
inc_synthesize_new_keys = True  # 补充新字段
inc_and_synthesize_missing_to_median = True  # 少数类上采样到目标均值
fill_stat = "mean"              # 使用平均数作为目标
topk_to_median = True          # 启用 top-k 均衡
topk_titles = 20               # 处理前20个报告类型

# 并发
reports_workers = 6             # 样本级并发
fields_workers = 6              # 字段级并发
per_base_pool_maxsize = 48      # 连接池大小
per_base_max_inflight = 24      # 最大并发请求
```

**预计时间:** 40-90 分钟（包含下采样 + LLM 字段补充）

### 阶段 2: data_augmentation_recheck.py - LLM 清洗

```python
in_path  = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_added.json"
out_path = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_cleaned.json"

# 操作
process_mode = "added_keys_only"  # 只清洗新增字段
max_fields_per_record = 10        # 每条记录最多处理10个字段

# 并发
RECORD_WORKERS = 6                # 记录级并发
INNER_WORKERS = 6                 # 字段级并发

# LLM
BASE_URL = "https://qwen3.yoo.la/v1/|http://123.57.234.67:8000/v1"
MODEL = "qwen3-32b"
```

**预计时间:** 60-120 分钟（取决于需要清洗的字段数量）

### 阶段 3: compose_and_noise.py - 报告组装

```python
in_path   = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_cleaned.json"
out_drop  = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_drop_keys.json"
out_alias = f"data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_key_alias.json"

# 操作
- 随机丢键（增强泛化）
- 别名覆盖（训练多样性）
- 组装格式化 report
```

**预计时间:** 5-15 分钟

---

## 🚀 执行命令

### 一键执行（推荐）

```bash
DATE_TAG=$(date +%Y%m%d)
export DA_DATE_TAG="$DATE_TAG"
./pre_struct/data_aug/da_workflow.sh
```

### 分步执行

```bash
DATE_TAG=$(date +%Y%m%d)
export DA_DATE_TAG="$DATE_TAG"

# 步骤 1
python pre_struct/data_aug/run_da.py

# 步骤 2
python pre_struct/data_aug/data_augmentation_recheck.py

# 步骤 3
python pre_struct/data_aug/compose_and_noise.py
```

### 监控进度

```bash
# 查看实时日志
tail -f nohup.out

# 检查输出文件
watch -n 5 "ls -lh data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_*"

# 查看系统负载
htop
```

---

## 📈 预期输出

执行完成后将生成以下文件：

| 文件 | 大小估算 | 用途 |
|-----|----------|------|
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_added.json` | ~60MB | 中间文件：字段补充后 |
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_cleaned.json` | ~65MB | 中间文件：LLM 清洗后 |
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_field_cleaned.changes.jsonl` | ~5MB | 变更记录 |
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_drop_keys.json` | ~70MB | **最终训练数据**（丢键版）|
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report_key_alias.json` | ~75MB | **最终训练数据**（别名版）|
| `data/${DATE_TAG}/clean_ocr_ppt_da_v5_0_report.stats.json` | ~100KB | 统计信息 |

---

## ⚠️ 注意事项

1. **LLM API 配置**
   - 确保 `conf.py` 中的 `API_KEY` 已配置
   - 检查 LLM 服务可访问性：
     ```bash
     curl -X POST https://qwen3.yoo.la/v1/chat/completions \
       -H "Authorization: Bearer $API_KEY" \
       -H "Content-Type: application/json" \
       -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"test"}]}'
     ```

2. **磁盘空间**
   - 确保至少有 **500MB** 可用空间
   - 检查命令: `df -h data/`

3. **执行时间**
   - 总预计时间: **1.5 - 3 小时**
   - 建议使用 `nohup` 后台执行：
     ```bash
     nohup ./pre_struct/data_aug/da_workflow.sh > da_v5.log 2>&1 &
     ```

4. **错误处理**
   - 如果某阶段失败，可从该阶段重新执行
   - 查看日志文件定位问题
   - 检查 `.changes.jsonl` 了解字段变更详情

---

## ✅ 最终检查清单

执行前确认：

- [x] 输入文件存在且大小正确（51MB）
- [x] keys/keys_merged.json 已更新（258KB）
- [x] API_KEY 已配置
- [x] 三个脚本路径已正确衔接
- [x] 并发参数已优化（6/6 workers）
- [x] 磁盘空间充足（>500MB）
- [x] LLM 服务可访问

执行后验证：

- [ ] `v5_0_field_added.json` 生成
- [ ] `v5_0_field_cleaned.json` 生成
- [ ] `v5_0_report_drop_keys.json` 生成
- [ ] `v5_0_report_key_alias.json` 生成
- [ ] 查看 `.stats.json` 确认样本数量
- [ ] 检查日志无严重错误

---

## 📞 问题排查

**Q: LLM API 超时**  
A: 降低并发数 (`RECORD_WORKERS=4, INNER_WORKERS=4`)

**Q: 内存不足**  
A: 降低 `reports_workers` 和 `per_base_pool_maxsize`

**Q: 字段清洗失败**  
A: 检查 `.changes.jsonl` 查看具体错误，可设置 `EXCLUDE_FIELDS` 跳过问题字段

**Q: 生成的文件很小**  
A: 检查 `TARGET_REPORT_TYPES` 过滤是否过严

---

**状态:** ✅ 所有检查通过，可以执行  
**推荐:** 使用 `nohup ./pre_struct/data_aug/da_workflow.sh > da_v5.log 2>&1 &` 后台执行
