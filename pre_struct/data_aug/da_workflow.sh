#!/bin/bash
set -euo pipefail

# Project root (this script lives in pre_struct/data_aug)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DA_DATE_TAG:-$(date +%Y%m%d)}"
export DA_DATE_TAG="$DATE_TAG"
BASE_DIR="data/${DATE_TAG}"
mkdir -p "$BASE_DIR"
echo "[info] data directory: ${BASE_DIR}"

echo "[1/3] Run DA pipeline -> ${DATE_TAG}"
python3 pre_struct/data_aug/run_da.py

echo "[2/3] Recheck fields -> ${DATE_TAG}"
python3 pre_struct/data_aug/data_augmentation_recheck.py

echo "[3/3] Compose reports -> ${DATE_TAG}"
python3 pre_struct/data_aug/compose_and_noise.py

echo "✅ Done. Output files ( ${BASE_DIR} ):"
ls -la ${BASE_DIR}/clean_ocr_ppt_da_v5_0_* 2>/dev/null | tail -6
