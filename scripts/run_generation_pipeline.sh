#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${1:-${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/generations}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"
TOXICCHAT_MAX_SAMPLES="${TOXICCHAT_MAX_SAMPLES:-500}"

DATASETS=(advbench toxicchat xstest)
if [[ -f "${ROOT_DIR}/security_data.csv" ]]; then
  DATASETS+=(security)
fi

echo "Running controlled decoding for model=${MODEL_ID}"
echo "Datasets: ${DATASETS[*]}"
echo "Output dir: ${OUTPUT_DIR}"

for DATASET in "${DATASETS[@]}"; do
  echo
  echo "=== Dataset: ${DATASET} ==="

  CMD=(
    "${PYTHON_BIN}" gradient_controlled_decoding/controlled_decode.py
    --model-id "${MODEL_ID}"
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
  )

  if [[ "${DATASET}" == "toxicchat" ]]; then
    CMD+=(--max-samples "${TOXICCHAT_MAX_SAMPLES}")
  fi

  if [[ "${OVERWRITE}" == "1" ]]; then
    CMD+=(--overwrite)
  fi

  "${CMD[@]}"
done

echo
echo "Done."
echo "Generations saved in: ${OUTPUT_DIR}"
