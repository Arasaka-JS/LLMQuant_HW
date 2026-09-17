#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT/LiftQuant"

export REDPAJAMA_CACHE_DIR="${PROJECT_ROOT}/datasets/redpajama_cache"
export WIKITEXT2_CACHE_DIR="${PROJECT_ROOT}/datasets/wikitext2_cache"
export C4_CACHE_DIR="${PROJECT_ROOT}/datasets/c4_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507"

ARGS=(
    --model "$MODEL_PATH"
    --save_dir ./qmodels
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
    --nsamples1 32
    --nsamples2 32
    --epochs1 1
    --epochs2 1
    --batch_size 1
    --calib_dataset redpajama
    --usefullfp
    --training_trans
    --align 1
    --quant_layers 0
    --save_per_layer
    --fast_nearest
    --lscale_lr 5e-3
    --lexw_lr 2e-2
    --lw_lr 2e-5
    --la_lr 2e-3
    --lt_lr 2e-4
    --load_dir ""
)

# GPU 2, single process, no DDP (no --quant_training_ddp)
CUDA_VISIBLE_DEVICES=2 python main.py "${ARGS[@]}" "$@"

