#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EVAL_SCRIPT="${PROJECT_ROOT}/allq/scripts/eval_by_lmeval.sh"

# # Demo 1: evaluate an AutoRound quantized model on multiple tasks.
# CUDA_DEVICE=4 \
# EVAL_BACKEND=autoround \
# MODEL_PATH="${PROJECT_ROOT}/checkpoints/results/Qwen3-4B-w3g128" \
# TASKS=hellaswag,piqa,winogrande \
# EVAL_DTYPE=bfloat16 \
# EVAL_BS=16 \
# "${EVAL_SCRIPT}"

# Demo 2: evaluate a LiftQuant quantized model on multiple tasks.
CUDA_DEVICE=2 \
EVAL_BACKEND=liftquant \
FP_MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B" \
QUANT_MODEL_PATH="${PROJECT_ROOT}/LiftQuant/qmodels/Qwen3-4B/Qwen3-4B+24to8-packed.pth" \
TASKS=hellaswag,piqa,winogrande \
EVAL_DTYPE=bfloat16 \
EVAL_BS=16 \
LIFTQUANT_WBITS=2 \
LIFTQUANT_EXPC=24to8 \
"${EVAL_SCRIPT}"
