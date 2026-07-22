#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

EVAL_BACKEND="${EVAL_BACKEND:-autoround}"
if [[ "${EVAL_BACKEND}" != "autoround" && "${EVAL_BACKEND}" != "liftquant" ]]; then
  printf 'Unsupported EVAL_BACKEND: %s (expected autoround or liftquant)\n' "${EVAL_BACKEND}" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/results/Qwen3-4B-w4g128}"
FP_MODEL_PATH="${FP_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-${PROJECT_ROOT}/LiftQuant/qmodels/Qwen3-4B/Qwen3-4B+24to8-packed.pth}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${PROJECT_ROOT}/lm_eval/eval_results}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets}"

# TASKS controls which lm-eval tasks are evaluated:
#   single task: TASKS=hellaswag
#   multi tasks: TASKS=hellaswag,piqa,winogrande
#   all tasks:   TASKS=all
# Supported tasks:
#   mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc1,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge
TASKS="${TASKS:-all}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_BS="${EVAL_BS:-auto:8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
LIMIT="${LIMIT:-}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-100000}"
LIFTQUANT_WBITS="${LIFTQUANT_WBITS:-${WBITS:-2}}"
LIFTQUANT_EXPC="${LIFTQUANT_EXPC:-${EXPC:-24to8}}"
LIFTQUANT_W_TERNARY="${LIFTQUANT_W_TERNARY:-}"

mkdir -p "${EVAL_RESULTS_DIR}" "${DATA_DIR}/.lm_eval_cache"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${EVAL_BACKEND}" == "liftquant" ]]; then
  model_name="$(basename "${QUANT_MODEL_PATH%/}")"
else
  model_name="$(basename "${MODEL_PATH%/}")"
fi
log_file="${EVAL_RESULTS_DIR}/${model_name}_${EVAL_BACKEND}_lmeval_${timestamp}.log"

cmd=(
  python "${PROJECT_ROOT}/allq/eval/eval_quant_lmeval.py"
  --backend "${EVAL_BACKEND}"
  --data-dir "${DATA_DIR}"
  --output-dir "${EVAL_RESULTS_DIR}"
  --tasks "${TASKS}"
  --device cuda:0
  --dtype "${EVAL_DTYPE}"
  --batch-size "${EVAL_BS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
)

if [[ "${EVAL_BACKEND}" == "liftquant" ]]; then
  cmd+=(
    --fp-model-path "${FP_MODEL_PATH}"
    --quant-model-path "${QUANT_MODEL_PATH}"
    --wbits "${LIFTQUANT_WBITS}"
    --expc "${LIFTQUANT_EXPC}"
  )
else
  cmd+=(--model "${MODEL_PATH}")
fi

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ "${EVAL_BACKEND}" != "liftquant" && "${DISABLE_TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  cmd+=(--no-trust-remote-code)
fi

if [[ "${EVAL_BACKEND}" == "liftquant" && "${LOAD_PER_LAYER:-0}" == "1" ]]; then
  cmd+=(--load-per-layer)
fi

if [[ "${EVAL_BACKEND}" == "liftquant" && "${AUTO_MIX_PRECISION:-0}" == "1" ]]; then
  cmd+=(--auto-mix-precision)
fi

if [[ "${EVAL_BACKEND}" == "liftquant" && -n "${LIFTQUANT_W_TERNARY}" ]]; then
  cmd+=(--w-ternary "${LIFTQUANT_W_TERNARY}")
fi

if [[ "${ADD_BOS_TOKEN:-0}" == "1" ]]; then
  cmd+=(--add-bos-token)
fi

printf 'lm-eval quantized-model evaluation\n'
printf '  Backend:        %s\n' "${EVAL_BACKEND}"
if [[ "${EVAL_BACKEND}" == "liftquant" ]]; then
  printf '  FP model:       %s\n' "${FP_MODEL_PATH}"
  printf '  Quant model:    %s\n' "${QUANT_MODEL_PATH}"
  printf '  LiftQuant cfg:  wbits=%s expc=%s\n' "${LIFTQUANT_WBITS}" "${LIFTQUANT_EXPC}"
else
  printf '  Model:          %s\n' "${MODEL_PATH}"
fi
printf '  Tasks:          %s\n' "${TASKS}"
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
