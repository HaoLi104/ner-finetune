#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
CFG="$DIR/config.json"
echo "[bert_crf] Training..."
python "$ROOT/pre_struct/kv_ner/train.py" --config "$CFG"
echo "[bert_crf] Evaluating (KV-level)..."
python "$ROOT/pre_struct/kv_ner/evaluate.py" \
  --config "$CFG" \
  --model_dir "$ROOT/runs/experiments/bert_crf/best" \
  --test_data "$ROOT/data/kv_ner_prepared/val_eval.jsonl" \
  --output_dir "$ROOT/runs/experiments/bert_crf/eval"

