#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/LiftQuant"

# Override these values from the command line, for example:
# MODEL=/path/to/Llama-2-7B NET=Llama-2-7b bash run_quant.sh
MODEL=${MODEL:-/path/to/your/Llama-2-7B}
NET=${NET:-$(basename "$MODEL")}
SAVE_DIR=${SAVE_DIR:-./qmodels}
CACHE_DIR=${CACHE_DIR:-./cache}
OUTPUT_DIR=${OUTPUT_DIR:-../log}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

WBITS=${WBITS:-2}
ABITS=${ABITS:-16}
KBITS=${KBITS:-16}
VBITS=${VBITS:-16}
EXPC=${EXPC:-20to8}
CALIB_DATASET=${CALIB_DATASET:-redpajama}
NSAMPLES1=${NSAMPLES1:-4096}
NSAMPLES2=${NSAMPLES2:-4096}
EPOCHS1=${EPOCHS1:-2}
EPOCHS2=${EPOCHS2:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
SEED=${SEED:-42}
DTYPE=${DTYPE:-float16}

export CUDA_VISIBLE_DEVICES

python main.py \
  --model "$MODEL" \
  --net "$NET" \
  --save_dir "$SAVE_DIR" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --eval_ppl \
  --wbits "$WBITS" \
  --abits "$ABITS" \
  --kbits "$KBITS" \
  --vbits "$VBITS" \
  --expc "$EXPC" \
  --w_sym \
  --true-sequential \
  --act-order \
  --use_fpinps \
  --Rres_init Hadamard \
  --nsamples1 "$NSAMPLES1" \
  --nsamples2 "$NSAMPLES2" \
  --epochs1 "$EPOCHS1" \
  --epochs2 "$EPOCHS2" \
  --batch_size "$BATCH_SIZE" \
  --calib_dataset "$CALIB_DATASET" \
  --usefullfp \
  --training_trans \
  --finetuning_weights \
  --align 1 \
  --lscale_lr 5e-3 \
  --lexw_lr 2e-2 \
  --lw_lr 2e-5 \
  --la_lr 2e-3 \
  --lt_lr 2e-4 \
  --dtype "$DTYPE" \
  "$@"
