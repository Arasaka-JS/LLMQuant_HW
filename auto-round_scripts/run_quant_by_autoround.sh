#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/auto-round_scripts/auto_round_results}"

# Pass a local .json/.jsonl calibration file as the first argument, or set CALIB_DATASET.
# AutoRound local calibration input must be a JSON/JSONL file, not the downloaded parquet directory.
DEFAULT_CALIB_DATASET="${PROJECT_ROOT}/datasets/pile-10k/pile_10k_text.jsonl"
CALIB_DATASET="${1:-${CALIB_DATASET:-${DEFAULT_CALIB_DATASET}}}"

if [[ ! -f "${CALIB_DATASET%%:*}" ]]; then
  printf 'Calibration dataset not found: %s\n' "${CALIB_DATASET%%:*}" >&2
  exit 1
fi

cmd=(
  auto-round
  --model "${MODEL_PATH}"
  --scheme W4A16
  --format auto_round
  --output_dir "${OUTPUT_DIR}"
  --dataset "${CALIB_DATASET}"
)

CUDA_VISIBLE_DEVICES=1 "${cmd[@]}"
