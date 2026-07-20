#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW}"

# The quantization script writes W4A16 results below OUTPUT_DIR/<model>-w4g128.
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/results/Qwen3-4B-w4g128}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${PROJECT_ROOT}/auto-round_scripts/eval_results}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets}"

# Classic language-model evaluations:
# - hellaswag/piqa/winogrande: commonsense reasoning accuracy
# - arc_easy/arc_challenge: science question-answering accuracy
# - mmlu: multi-domain knowledge and reasoning accuracy
TASKS="${TASKS:-hellaswag}"

# CUDA_DEVICE is a physical GPU ID. AutoRound sees it as logical device 0 after
# CUDA_VISIBLE_DEVICES is set below.
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_BS="${EVAL_BS:-}"
LIMIT="${LIMIT:-}"

mkdir -p "${EVAL_RESULTS_DIR}" "${DATA_DIR}/.lm_eval_cache"

# Keep Hugging Face/lm-eval artifacts inside this workspace. Network access is
# intentionally not disabled: the AutoRound CLI does not install this repo's
# custom ModelScope dataset loader, so uncached datasets may need downloading.
export HF_HOME="${HF_HOME:-${DATA_DIR}/.lm_eval_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
model_name="$(basename "${MODEL_PATH%/}")"
log_file="${EVAL_RESULTS_DIR}/${model_name}_autoround_cli_${timestamp}.log"

cmd=(
  auto-round eval
  "${MODEL_PATH}"
  --tasks "${TASKS}"
  --device 0
  --eval_backend hf
  --eval_model_dtype "${EVAL_DTYPE}"
  --eval_task_by_task
)

# Leaving EVAL_BS empty uses AutoRound's lm-eval default, auto:8. The current
# AutoRound CLI accepts only an integer when --eval_bs is specified.
if [[ -n "${EVAL_BS}" ]]; then
  cmd+=(--eval_bs "${EVAL_BS}")
fi

# LIMIT is intended for smoke tests, e.g. LIMIT=10. Omit it for full evaluation.
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ "${DISABLE_TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  cmd+=(--disable_trust_remote_code)
fi

# Llama
if [[ "${ADD_BOS_TOKEN:-0}" == "1" ]]; then
  cmd+=(--add_bos_token)
fi

printf 'AutoRound quantized-model evaluation\n'
printf '  Model:          %s\n' "${MODEL_PATH}"
printf '  Tasks:          %s\n' "${TASKS}"
printf '  Metrics:        task accuracy/normalized accuracy\n'
printf '  Physical GPU:   %s (logical device 0)\n' "${CUDA_DEVICE}"
printf '  Eval dtype:     %s\n' "${EVAL_DTYPE}"
printf '  Batch size:     %s\n' "${EVAL_BS:-auto:8}"
printf '  Example limit:  %s\n' "${LIMIT:-none (full evaluation)}"
printf '  Log:            %s\n\n' "${log_file}"

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
