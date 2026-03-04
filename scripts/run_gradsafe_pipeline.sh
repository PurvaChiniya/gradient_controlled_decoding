#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${1:-${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}}"
TOXICCHAT_MAX_SAMPLES="${TOXICCHAT_MAX_SAMPLES:-500}"
DETECTION_OUTPUT_DIR="${DETECTION_OUTPUT_DIR:-baselines/gradSafe/outputs/detection}"
THRESHOLD_OUTPUT_DIR="${THRESHOLD_OUTPUT_DIR:-baselines/gradSafe/outputs/threshold}"

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
}

MODEL_SLUG="$(slugify "${MODEL_ID}")"

DATASETS=(advbench toxicchat xstest)
if [[ -f "${ROOT_DIR}/security_data.csv" ]]; then
  DATASETS+=(security)
fi

echo "Running gradSafe pipeline for model=${MODEL_ID}"
echo "Datasets: ${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  echo
  echo "=== Dataset: ${DATASET} ==="

  DETECTION_ARGS=(
    "${PYTHON_BIN}" baselines/gradSafe/detection.py
    --dataset "${DATASET}"
    --model-id "${MODEL_ID}"
    --output-dir "${DETECTION_OUTPUT_DIR}"
  )

  if [[ "${DATASET}" == "toxicchat" ]]; then
    DETECTION_ARGS+=(--max-samples "${TOXICCHAT_MAX_SAMPLES}")
  fi

  SCORES_CSV="${DETECTION_OUTPUT_DIR}/${DATASET}__${MODEL_SLUG}__scores.csv"
  THRESHOLD_JSON="${THRESHOLD_OUTPUT_DIR}/${DATASET}__${MODEL_SLUG}__scores__thresholds.json"

  echo "[1/2] Detection"
  "${DETECTION_ARGS[@]}"

  echo "[2/2] Threshold analysis"
  "${PYTHON_BIN}" baselines/gradSafe/threshold.py \
    --dataset "${DATASET}" \
    --scores-csv "${SCORES_CSV}" \
    --output-dir "${THRESHOLD_OUTPUT_DIR}"
done

echo
echo "Done."
echo "Detection outputs: ${DETECTION_OUTPUT_DIR}"
echo "Threshold outputs: ${THRESHOLD_OUTPUT_DIR}"
