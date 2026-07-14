#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT/LiftQuant"

export REDPAJAMA_CACHE_DIR=/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/redpajama_cache
export WIKITEXT2_CACHE_DIR=/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/wikitext2_cache
export C4_CACHE_DIR=/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/c4_cache
export GSM8K_CACHE_DIR=/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/gsm8k_cache
export MODEL_PATH=/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/checkpoints/meta-llama/Llama-2-7b-hf
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export EVAL_GSM8K=${EVAL_GSM8K:-0}
export GSM8K_REASONING_CAPABILITY=${GSM8K_REASONING_CAPABILITY:-no}

ARGS=(
    --model "$MODEL_PATH"
    --save_dir ./qmodels
    --eval_ppl
    --wbits 2
    --expc 20to8
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
    --finetuning_weights
    --align 1
    --lscale_lr 5e-3
    --lexw_lr 2e-2
    --lw_lr 2e-5
    --la_lr 2e-3
    --lt_lr 2e-4
    # --only_eval
    --load_dir ""
)

if [[ "$EVAL_GSM8K" == "1" ]]; then
    ARGS+=(
        --eval_gsm8k
        --gsm8k_reasoning_capability "$GSM8K_REASONING_CAPABILITY"
    )
fi

python main.py "${ARGS[@]}" "$@"
