#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT/LiftQuant"

export REDPAJAMA_CACHE_DIR="${PROJECT_ROOT}/datasets/redpajama_cache"
export WIKITEXT2_CACHE_DIR="${PROJECT_ROOT}/datasets/wikitext2_cache"
export C4_CACHE_DIR="${PROJECT_ROOT}/datasets/c4_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507"

ARGS=(
    --model "$MODEL_PATH"
    --save_dir ./qmodels_smoke
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
    --batch_size 3
    --calib_dataset redpajama
    --usefullfp
    --training_trans
    --align 1
    --fast_nearest
    --quant_layers 0
    --save_per_layer
    --lscale_lr 5e-3
    --lexw_lr 2e-2
    --lw_lr 2e-5
    --la_lr 2e-3
    --lt_lr 2e-4
    --load_dir ""
    --moe_num_groups 4
    --finetuning_weights
)

CUDA_VISIBLE_DEVICES=1,2,3 /home/kris/miniconda3/envs/skw_quant_env/bin/torchrun --standalone --nproc_per_node=3 main.py "${ARGS[@]}" --quant_training_ddp "$@"
