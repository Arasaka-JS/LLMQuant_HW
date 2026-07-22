#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}/LiftQuant"

export REDPAJAMA_CACHE_DIR="${REDPAJAMA_CACHE_DIR:-${PROJECT_ROOT}/datasets/redpajama_cache}"
export WIKITEXT2_CACHE_DIR="${WIKITEXT2_CACHE_DIR:-${PROJECT_ROOT}/datasets/wikitext2_cache}"
export C4_CACHE_DIR="${C4_CACHE_DIR:-${PROJECT_ROOT}/datasets/c4_cache}"
export GSM8K_CACHE_DIR="${GSM8K_CACHE_DIR:-${PROJECT_ROOT}/datasets/gsm8k_cache}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export EVAL_GSM8K="${EVAL_GSM8K:-0}"
export FULL_FINTUNE="${FULL_FINTUNE:-0}"
export GSM8K_REASONING_CAPABILITY="${GSM8K_REASONING_CAPABILITY:-no}"

CUDA_DEVICE="${CUDA_DEVICE:-2}"

ARGS=(
    --model "${MODEL_PATH}"
    --save_dir ./qmodels
    --eval_ppl
    --wbits 2
    --expc 24to8
    --w_sym
    --abits 16
    --kbits 16
    --vbits 16
    --true-sequential
    --act-order
    --use_fpinps
    --Rres_init Hadamard
    --nsamples1 4096
    --nsamples2 4096
    --epochs1 2
    --epochs2 2
    --batch_size 2
    --calib_dataset redpajama
    --usefullfp
    --training_trans
    --align 1
    --lscale_lr 5e-3
    --lexw_lr 2e-2
    --lw_lr 2e-5
    --la_lr 2e-3
    --lt_lr 2e-4
    --load_dir ""
)

if [[ "${EVAL_GSM8K}" == "1" ]]; then
    ARGS+=(
        --eval_gsm8k
        --gsm8k_reasoning_capability "${GSM8K_REASONING_CAPABILITY}"
    )
fi

if [[ "${FULL_FINTUNE}" == "1" ]]; then
    ARGS+=(
        --finetuning_weights
    )
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python main.py "${ARGS[@]}" "$@"
