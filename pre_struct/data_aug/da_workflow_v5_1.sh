#!/bin/bash
set -euo pipefail

# -----------------------
# Helpers & arguments
# -----------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

log() {
  local level="$1"; shift
  printf '[%s] [%s] %s\n' "$(date '+%H:%M:%S')" "$level" "$*"
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [steps]

Steps:
  1   Run DA pipeline (v5.0 → field_added)
  2   Recheck fields (v5.0 → field_cleaned)
  3   Compose reports (emit v5.1 outputs)

Examples:
  $(basename "$0")           # run all (1,2,3)
  $(basename "$0") 2         # only run step 2
  $(basename "$0") 1 3       # run steps 1 and 3
  $(basename "$0") 1,2       # comma-separated also supported
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage; exit 0
fi

# Collect steps from args (default all)
declare -a REQ_STEPS=()
if [[ $# -eq 0 ]]; then
  REQ_STEPS=(1 2 3)
else
  for a in "$@"; do
    IFS=',' read -r -a parts <<< "$a"
    for p in "${parts[@]}"; do
      [[ -z "$p" ]] && continue
      if [[ "$p" =~ ^[123]$ ]]; then
        REQ_STEPS+=("$p")
      else
        log ERROR "Invalid step: $p"; usage; exit 2
      fi
    done
  done
fi

# Date-tagged output dir & version tag
DATE_TAG="${DA_DATE_TAG:-$(date +%Y%m%d)}"
export DA_DATE_TAG="$DATE_TAG"
export DA_OUTPUT_VERSION="${DA_OUTPUT_VERSION:-v5_1}"
BASE_DIR="data/${DATE_TAG}"
mkdir -p "$BASE_DIR"
log INFO "data dir: ${BASE_DIR} (version=${DA_OUTPUT_VERSION})"

# -----------------------
# Step functions
# -----------------------
step1() {
  log INFO "[1/3] Enhance from v5.0 (→ ${DA_OUTPUT_VERSION} field_added)"
  DA_BASE_INPUT="${BASE_DIR}/clean_ocr_ppt_da_v5_0_field_cleaned.json" \
  DA_OUTPUT_VERSION="${DA_OUTPUT_VERSION}" \
  python3 pre_struct/data_aug/run_da.py
}

step2() {
  log INFO "[2/3] Recheck fields (${DA_OUTPUT_VERSION} → field_cleaned)"
  DA_CLEAN_IN="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_added.json" \
  DA_CLEAN_OUT="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_cleaned.json" \
  DA_CHANGES_OUT="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_cleaned.changes.jsonl" \
  python3 pre_struct/data_aug/data_augmentation_recheck.py
}

step3() {
  log INFO "[3/3] Compose reports (emit ${DA_OUTPUT_VERSION} outputs)"
  DA_COMPOSE_IN="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_cleaned.json" \
  python3 pre_struct/data_aug/compose_and_noise.py
}

# -----------------------
# Preconditions (best-effort hints)
# -----------------------
hint_inputs() {
  # Helper: check if a step is requested
  _has_step() { for x in "${REQ_STEPS[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

  # Step1: run_da.py 当前在脚本内手动指定输入路径，跳过严格预检以避免误报
  if _has_step 1; then
    log INFO "step1: skip strict precheck (input path is defined in run_da.py)"
  fi

  # Step2: 期望 step1 产物 ${DA_OUTPUT_VERSION}_field_added.json
  if _has_step 2; then
    local in2="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_added.json"
    if [[ -f "$in2" ]]; then
      log INFO "step2 input ready: $in2"
    else
      if _has_step 1; then
        log INFO "step2 input will be produced by step1: $in2"
      else
        log WARN "missing step2 input (and step1 not requested): $in2"
      fi
    fi
  fi

  # Step3: 期望 step2 产物 ${DA_OUTPUT_VERSION}_field_cleaned.json
  if _has_step 3; then
    local in3="${BASE_DIR}/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_field_cleaned.json"
    if [[ -f "$in3" ]]; then
      log INFO "step3 input ready: $in3"
    else
      if _has_step 2; then
        log INFO "step3 input will be produced by step2: $in3"
      else
        log WARN "missing step3 input (and step2 not requested): $in3"
      fi
    fi
  fi
}

hint_inputs

# -----------------------
# Dispatch
# -----------------------
for s in "${REQ_STEPS[@]}"; do
  case "$s" in
    1) step1;;
    2) step2;;
    3) step3;;
    *) log ERROR "unknown step: $s"; exit 2;;
  esac
done

log INFO "Done. Outputs (if step3 ran):"
ls -la "${BASE_DIR}"/clean_ocr_ppt_da_${DA_OUTPUT_VERSION}_report_* 2>/dev/null || true
