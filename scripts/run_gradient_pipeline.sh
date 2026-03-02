#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${1:-${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}}"
TOXICCHAT_MAX_SAMPLES="${TOXICCHAT_MAX_SAMPLES:-500}"
DETECTION_OUTPUT_DIR="${DETECTION_OUTPUT_DIR:-outputs/detection}"
THRESHOLD_OUTPUT_DIR="${THRESHOLD_OUTPUT_DIR:-outputs/threshold}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/eval}"

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
}

build_output_stem() {
  local dataset="$1"
  local scores_csv="$2"
  local stem
  stem="$(basename "${scores_csv}" .csv)"
  stem="$(slugify "${stem}")"
  if [[ "${stem}" == "${dataset}"__* ]]; then
    printf '%s' "${stem}"
  else
    printf '%s__%s' "${dataset}" "${stem}"
  fi
}

MODEL_SLUG="$(slugify "${MODEL_ID}")"

DATASETS=(advbench toxicchat xstest)
if [[ -f "${ROOT_DIR}/security_data.csv" ]]; then
  DATASETS+=(security)
fi

echo "Running gradient pipeline for model=${MODEL_ID}"
echo "Datasets: ${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  echo
  echo "=== Dataset: ${DATASET} ==="

  DETECTION_ARGS=(
    "${PYTHON_BIN}" gradient_controlled_decoding/detection.py
    --dataset "${DATASET}"
    --model-id "${MODEL_ID}"
    --output-dir "${DETECTION_OUTPUT_DIR}"
  )

  if [[ "${DATASET}" == "toxicchat" ]]; then
    DETECTION_ARGS+=(--max-samples "${TOXICCHAT_MAX_SAMPLES}")
  fi

  SCORES_CSV="${DETECTION_OUTPUT_DIR}/${DATASET}__${MODEL_SLUG}__scores.csv"
  DETECTION_METRICS_JSON="${DETECTION_OUTPUT_DIR}/${DATASET}__${MODEL_SLUG}__metrics.json"
  OUTPUT_STEM="$(build_output_stem "${DATASET}" "${SCORES_CSV}")"
  THRESHOLD_JSON="${THRESHOLD_OUTPUT_DIR}/${OUTPUT_STEM}__thresholds.json"
  EVAL_JSON="${EVAL_OUTPUT_DIR}/${OUTPUT_STEM}__eval_metrics.json"

  if [[ -f "${SCORES_CSV}" && -f "${DETECTION_METRICS_JSON}" ]]; then
    echo "[1/3] Detection skipped, outputs already exist"
  else
    echo "[1/3] Detection"
    "${DETECTION_ARGS[@]}"
  fi

  if [[ -f "${THRESHOLD_JSON}" ]]; then
    echo "[2/3] Threshold analysis skipped, output already exists"
  else
    echo "[2/3] Threshold analysis"
    "${PYTHON_BIN}" gradient_controlled_decoding/threshold.py \
      --dataset "${DATASET}" \
      --scores-csv "${SCORES_CSV}" \
      --output-dir "${THRESHOLD_OUTPUT_DIR}"
  fi

  if [[ -f "${EVAL_JSON}" ]]; then
    echo "[3/3] Evaluation skipped, output already exists"
  else
    echo "[3/3] Evaluation"
    "${PYTHON_BIN}" gradient_controlled_decoding/eval.py \
      --dataset "${DATASET}" \
      --scores-csv "${SCORES_CSV}" \
      --output-dir "${EVAL_OUTPUT_DIR}"
  fi
done

echo
echo "Done."
echo "Detection outputs: ${DETECTION_OUTPUT_DIR}"
echo "Threshold outputs: ${THRESHOLD_OUTPUT_DIR}"
echo "Eval outputs: ${EVAL_OUTPUT_DIR}"
