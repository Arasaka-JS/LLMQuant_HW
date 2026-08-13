#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT/LiftQuant"

export REDPAJAMA_CACHE_DIR="${PROJECT_ROOT}/datasets/redpajama_cache"
export WIKITEXT2_CACHE_DIR="${PROJECT_ROOT}/datasets/wikitext2_cache"
export C4_CACHE_DIR="${PROJECT_ROOT}/datasets/c4_cache"
export GSM8K_CACHE_DIR="${PROJECT_ROOT}/datasets/gsm8k_cache"
export MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export EVAL_GSM8K=${EVAL_GSM8K:-0}
export FULL_FINTUNE=${FULL_FINTUNE:-1}
export GSM8K_REASONING_CAPABILITY=${GSM8K_REASONING_CAPABILITY:-no}

ARGS=(
    --model "$MODEL_PATH"
    --save_dir ./qmodels
    --eval_ppl
    --wbits 2
    --expc 16to8
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
    --batch_size 4
    --calib_dataset redpajama
    --usefullfp
    --training_trans
    --align 1
    --lscale_lr 5e-3
    --lexw_lr 2e-2
    --lw_lr 2e-5
    --la_lr 2e-3
    --lt_lr 2e-4
    # --only_eval
    --load_dir ""
    --fast_nearest
)

if [[ "$EVAL_GSM8K" == "1" ]]; then
    ARGS+=(
        --eval_gsm8k
        --gsm8k_reasoning_capability "$GSM8K_REASONING_CAPABILITY"
    )
fi

if [[ "$FULL_FINTUNE" == "1" ]]; then
    ARGS+=(
        --finetuning_weights
    )
fi

# Demo 1: single GPU. Do not pass --quant_training_ddp so Stage1/Stage2 use the single-process path.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main.py "${ARGS[@]}" "$@"

# Demo 2: multi GPU. Uncomment this line and comment out Demo 1 to enable Stage1/Stage2 DDP.
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main.py "${ARGS[@]}" --quant_training_ddp "$@"
