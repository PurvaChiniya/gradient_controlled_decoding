#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
DATASET_NAME="${DATASET_NAME:-xstest}"
OUTPUT_CSV="${OUTPUT_CSV:-${DATASET_NAME}.csv}"

echo "[1/3] Generating baseline outputs for dataset=${DATASET_NAME}"
"${PYTHON_BIN}" baselines/safe_decoding/run.py \
  --data_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --output_csv "${OUTPUT_CSV}"

echo "[2/3] Evaluating outputs"
"${PYTHON_BIN}" baselines/safe_decoding/eval.py \
  --data_name "${DATASET_NAME}" \
  --save-metrics-json

echo "[3/3] Done"
echo "Generated files:"
echo "  - ${DATASET_NAME}_output_with_predictions.csv"
echo "  - ${DATASET_NAME}_evaluation_metrics.json"
