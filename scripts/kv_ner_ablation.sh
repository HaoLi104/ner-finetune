#!/usr/bin/env bash
set -euo pipefail

# KV-NER ablation runner
# Compares four variants: base, conv, bilstm, bilstm_conv
# Prereqs: data prepared under data/kv_ner_prepared, GPU available (optional)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_CFG="${ROOT_DIR}/pre_struct/kv_ner/kv_ner_config_with_bilstm.json"
OUT_ROOT="${ROOT_DIR}/runs/kv_ner_ablate"
EVAL_DATA="${ROOT_DIR}/data/kv_ner_prepared/full_eval.jsonl"

mkdir -p "${OUT_ROOT}/configs"

if [[ ! -f "${EVAL_DATA}" ]]; then
  echo "[ERROR] Missing eval data at ${EVAL_DATA}. Run prepare_data first." >&2
  exit 1
fi

mk_cfg() {
  local name="$1"; shift
  local use_bilstm="$1"; shift
  local use_conv="$1"; shift
  local cfg_out="${OUT_ROOT}/configs/${name}.json"
  python - "$BASE_CFG" "$cfg_out" "$name" "$use_bilstm" "$use_conv" <<'PY'
import json,sys,os
src, dst, name, use_bilstm, use_conv = sys.argv[1:6]
use_bilstm = (use_bilstm.lower()=='true')
use_conv = (use_conv.lower()=='true')
with open(src,'r',encoding='utf-8') as f:
    cfg=json.load(f)

# Tweak output dirs per variant
out_dir = f"runs/kv_ner_ablate/{name}"
cfg.setdefault('train',{})
cfg['train']['output_dir']=out_dir
cfg['train']['use_bilstm']=use_bilstm
cfg['train']['use_conv']=use_conv

# Point predict/evaluate to the variant model dir
cfg.setdefault('predict',{})
cfg['predict']['model_dir']=out_dir+"/best"
cfg.setdefault('evaluate',{})
cfg['evaluate']['model_dir']=out_dir+"/best"

# Ensure relation head is off (fully removed in code, but keep safe)
for k in ('use_relation_head','relation_loss_weight','relation_ramp_epochs'):
    if k in cfg.get('train',{}):
        cfg['train'].pop(k,None)

with open(dst,'w',encoding='utf-8') as f:
    json.dump(cfg,f,ensure_ascii=False,indent=2)
print(dst)
PY
  echo "$cfg_out"
}

run_one() {
  local name="$1"; shift
  local cfg="$1"; shift
  echo "\n===== Training: ${name} ====="
  python "${ROOT_DIR}/pre_struct/kv_ner/train.py" --config "$cfg"
  echo "\n===== Evaluating (KV-level): ${name} ====="
  python "${ROOT_DIR}/pre_struct/kv_ner/evaluate.py" \
    --config "$cfg" \
    --model_dir "${OUT_ROOT}/${name}/best" \
    --test_data "${EVAL_DATA}" \
    --output_dir "${OUT_ROOT}/${name}/eval"
  # Print quick F1s
  python - <<'PY'
import json,sys,os
base=sys.argv[1]
summ=os.path.join(base,'eval','eval_summary.json')
if os.path.isfile(summ):
    d=json.load(open(summ,'r',encoding='utf-8'))
    print(f"Exact F1={d.get('text_exact',{}).get('f1_score',0):.4f}  Overlap F1={d.get('text_overlap',{}).get('f1_score',0):.4f}")
else:
    print("No eval_summary.json found",summ)
PY "${OUT_ROOT}/${name}"
}

# Matrix
declare -a NAMES=(
  base
  conv
  bilstm
  bilstm_conv
)
declare -a BILSTM=( false false true  true )
declare -a CONV=(   false true  false true  )

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; b="${BILSTM[$i]}"; c="${CONV[$i]}"
  cfg_path="$(mk_cfg "$name" "$b" "$c")"
  run_one "$name" "$cfg_path"
done

echo "\nAll ablations finished. Outputs under ${OUT_ROOT}"

