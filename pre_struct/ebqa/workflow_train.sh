#!/usr/bin/env bash
# Three-step EBQA training workflow (directly run python scripts)
# 1) pre_struct/data_aug/compose_and_noise.py  -> writes its own _report.json
# 2) pre_struct/ebqa/da_core/dataset.py        -> builds precomputed jsonl
# 3) pre_struct/ebqa/train_ebqa.py             -> trains with ebqa_config.json

set -euo pipefail

# Resolve repo root (this script lives in pre_struct/ebqa/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONIOENCODING=UTF-8
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[Step 1/3] Compose reports (python pre_struct/data_aug/compose_and_noise.py)"
python pre_struct/data_aug/compose_and_noise.py

echo "[Step 2/3] Precompute dataset (python pre_struct/ebqa/da_core/dataset.py)"
python pre_struct/ebqa/da_core/dataset.py

echo "[Step 3/3] Train EBQA (python pre_struct/ebqa/train_ebqa.py --config pre_struct/ebqa/ebqa_config.json)"
python pre_struct/ebqa/train_ebqa.py --config pre_struct/ebqa/ebqa_config.json

echo "[DONE] Workflow completed."
