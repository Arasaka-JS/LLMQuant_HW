#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW}"

# The quantization script writes W4A16 results below OUTPUT_DIR/<model>-w4g128.
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${PROJECT_ROOT}/lm_eval/eval_results}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets}"

# Classic language-model evaluations:
# - wikitext: language-model perplexity (lower is better)
# - hellaswag/piqa/winogrande: commonsense reasoning accuracy
# - arc_easy/arc_challenge: science question-answering accuracy
# - mmlu: multi-domain knowledge and reasoning accuracy
TASKS="${TASKS:-hellaswag}"

# CUDA_DEVICE is a physical GPU ID. eval_autoround.py sees it as logical cuda:0
# after CUDA_VISIBLE_DEVICES is set below.
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_BS="${EVAL_BS:-auto:8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
LIMIT="${LIMIT:-}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100000}"

mkdir -p "${EVAL_RESULTS_DIR}" "${DATA_DIR}/.lm_eval_cache"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
model_name="$(basename "${MODEL_PATH%/}")"
log_file="${EVAL_RESULTS_DIR}/${model_name}_lmeval_${timestamp}.log"

cmd=(
  python "${PROJECT_ROOT}/auto-round_scripts/eval_autoround.py"
  --model "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${EVAL_RESULTS_DIR}"
  --tasks "${TASKS}"
  --device cuda:0
  --dtype "${EVAL_DTYPE}"
  --batch-size "${EVAL_BS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
)

# LIMIT is intended for smoke tests, e.g. LIMIT=10. Omit it for full evaluation.
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ "${DISABLE_TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  cmd+=(--no-trust-remote-code)
fi

# Llama
if [[ "${ADD_BOS_TOKEN:-0}" == "1" ]]; then
  cmd+=(--add-bos-token)
fi

printf 'lm-eval quantized-model evaluation\n'
printf '  Model:          %s\n' "${MODEL_PATH}"
printf '  Tasks:          %s\n' "${TASKS}"
printf '  Metrics:        task-defined lm-eval metrics\n'
printf '  Physical GPU:   %s (logical cuda:0)\n' "${CUDA_DEVICE}"
printf '  Eval dtype:     %s\n' "${EVAL_DTYPE}"
printf '  Batch size:     %s\n' "${EVAL_BS}"
printf '  Max batch size: %s\n' "${MAX_BATCH_SIZE}"
printf '  Example limit:  %s\n' "${LIMIT:-none (full evaluation)}"
printf '  Log:            %s\n\n' "${log_file}"

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
