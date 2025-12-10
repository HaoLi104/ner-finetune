#!/usr/bin/env bash
set -euo pipefail

# 安全训练脚本 - 防止内存/缓存占满导致系统卡死
# 适用于 Debian/Linux 系统

cd "$(dirname "$0")/.."

echo "========================================"
echo "🛡️ 安全训练启动脚本 (防内存占满)"
echo "========================================"
echo ""

# ========== 步骤1: 清理系统缓存 ==========
echo "[1/5] 清理系统缓存..."
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "⚠️ 需要sudo权限清理缓存，跳过"
echo "  当前内存状态:"
free -h | grep -E "Mem:|Swap:"
echo ""

# ========== 步骤2: 设置环境变量 ==========
echo "[2/5] 设置环境变量..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# 限制PyTorch内存分配
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
echo "  ✓ CUDA_VISIBLE_DEVICES=0"
echo "  ✓ OMP_NUM_THREADS=4"
echo "  ✓ PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
echo ""

# ========== 步骤3: 检查GPU可用性 ==========
echo "[3/5] 检查GPU状态..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 不可用，GPU可能未正确安装"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ========== 步骤4: 验证训练数据 ==========
echo "[4/5] 验证训练数据..."
DATA_PATH="data/20251013/ebqa_v5_0_plus_project1.samples.jsonl"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "❌ 训练数据不存在: $DATA_PATH"
    exit 1
fi

DATA_SIZE=$(du -sh "$DATA_PATH" | cut -f1)
LINE_COUNT=$(wc -l < "$DATA_PATH")
echo "  ✓ 数据文件: $DATA_PATH"
echo "  ✓ 文件大小: $DATA_SIZE"
echo "  ✓ 样本数: $LINE_COUNT"
echo ""

# ========== 步骤5: 启动训练（带内存监控） ==========
echo "[5/5] 启动训练..."
LOG_FILE="train_v5_$(date +%Y%m%d_%H%M%S).log"

echo "  训练日志: $LOG_FILE"
echo "  后台运行中..."
echo ""

# 启动训练
nohup python -u pre_struct/ebqa/train_ebqa.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "  ✓ 训练进程PID: $TRAIN_PID"
echo ""
echo "========================================"
echo "🎯 训练已启动"
echo "========================================"
echo ""
echo "监控命令:"
echo "  # 查看日志"
echo "  tail -f $LOG_FILE"
echo ""
echo "  # 监控GPU"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "  # 监控内存"
echo "  watch -n 2 'free -h'"
echo ""
echo "  # 检查进程"
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "  # 如需停止"
echo "  kill $TRAIN_PID"
echo ""
echo "========================================"

# 等待几秒确认启动
sleep 3
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✅ 训练进程运行正常"
    echo ""
    echo "开始监控日志（按 Ctrl+C 退出监控）..."
    tail -f "$LOG_FILE"
else
    echo "❌ 训练进程启动失败，请检查日志:"
    echo "   tail -20 $LOG_FILE"
    exit 1
fi

