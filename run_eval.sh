#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EVAL_SCRIPT="${PROJECT_ROOT}/allq/scripts/eval_by_lmeval.sh"

# =============================================================================
# Qwen3-30B-A3B-Instruct-2507 (MoE) 评估 demos
#
#   Demo A (fp):        加载 Hugging Face 浮点模型做基线评估。
#   Demo B (liftquant): 加载 FP 权重 + 量化 layer0 权重，评估前把量化层替换进 FP 模型。
#
# 说明：
#   - QUANT_MODEL_PATH 指向「前缀」（不含 -layer0.pth），配合 LOAD_PER_LAYER=1 使用。
#   - 当前默认执行 Demo B（量化评估）；Demo A（FP 基线）默认注释，需要跑基线时取消注释。
# =============================================================================

# Demo A: MoE 浮点(FP)基线评估（默认注释，需要跑基线时取消注释）
# CUDA_DEVICE=2 \
# EVAL_BACKEND=fp \
# MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507" \
# TASKS=hellaswag,piqa,winogrande \
# EVAL_DTYPE=bfloat16 \
# EVAL_BS=8 \
# "${EVAL_SCRIPT}"

# Demo B: MoE 量化模型评估（仅量化 layer0，Stage2 packed 权重）
CUDA_DEVICE=2 \
EVAL_BACKEND=liftquant \
FP_MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507" \
QUANT_MODEL_PATH="${PROJECT_ROOT}/LiftQuant/qmodels/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507+24to8" \
TASKS=hellaswag,piqa,winogrande \
EVAL_DTYPE=bfloat16 \
EVAL_BS=8 \
LIFTQUANT_WBITS=2 \
LIFTQUANT_EXPC=24to8 \
LOAD_PER_LAYER=1 \
"${EVAL_SCRIPT}"

# =============================================================================
# 以下为非 MoE 旧 demos（全部注释，仅作参考）
# =============================================================================

# Demo 1: evaluate an AutoRound quantized model on multiple tasks.
# CUDA_DEVICE=4 \
# EVAL_BACKEND=autoround \
# MODEL_PATH="${PROJECT_ROOT}/checkpoints/results/Qwen3-4B-w3g128" \
# TASKS=hellaswag,piqa,winogrande \
# EVAL_DTYPE=bfloat16 \
# EVAL_BS=16 \
# "${EVAL_SCRIPT}"

# Demo 2: evaluate an original floating-point Hugging Face model on multiple tasks.
# CUDA_DEVICE=6 \
# EVAL_BACKEND=fp \
# MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B" \
# TASKS=all \
# EVAL_DTYPE=bfloat16 \
# EVAL_BS=16 \
# "${EVAL_SCRIPT}"

# Demo 3: evaluate a LiftQuant model with selected quantized layers.
# CUDA_DEVICE=2 \
# EVAL_BACKEND=liftquant \
# FP_MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen/Qwen3-4B" \
# QUANT_MODEL_PATH="${PROJECT_ROOT}/LiftQuant/qmodels/Qwen3-4B/Qwen3-4B+16to8-layers2.pth" \
# TASKS=hellaswag,piqa,winogrande \
# EVAL_DTYPE=bfloat16 \
# EVAL_BS=16 \
# LIFTQUANT_WBITS=2 \
# LIFTQUANT_EXPC=16to8 \
# "${EVAL_SCRIPT}"
