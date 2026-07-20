# Repository Instructions

## Layout
- Treat `LiftQuant/` as the active project root; the workspace root only has this instruction file, `opencode.json`, and `docs/`.
- `LiftQuant/main.py` is the quantization/evaluation entrypoint, `LiftQuant/e2efinetune.py` is optional end-to-end fine-tuning, and `LiftQuant/chat/chat_bitblas_compile.py` is the accelerated chat path.
- Core implementation lives in `LiftQuant/quantize/`, `LiftQuant/models/`, `LiftQuant/gptq/`, and dataset helpers `LiftQuant/datautils*.py`.

## Setup And Commands
- Run Python commands from `LiftQuant/` so relative paths like `./cache`, `./lattice`, and `../log` resolve as intended.
- Environment documented by the repo: `conda create -n liftquant_env python=3.12 -y`, then `pip install -r requirements.txt`.
- Main quantization command shape is `python main.py --model /path/to/hf_model --save_dir ./qmodels ...`; use README examples as the source for long argument sets.
- Generate projection matrices only when needed with `python lattice_generator2.py`; pretrained matrices are already in `LiftQuant/lattice/`.
- Accelerated chat requires CUDA-compatible `torch==2.6.0`, `bitblas==0.1.0.post1`, and model paths: `python chat/chat_bitblas_compile.py --fp_model_path ... --quant_model_path ...`.

## Verification Gotchas
- There is no checked-in test suite, CI workflow, or lint/format config; avoid inventing validation commands beyond targeted import/CLI smoke checks.
- Most real runs require CUDA, Hugging Face model checkpoints, and downloadable datasets, so full quantization/eval is expensive and environment-dependent.
- `e2efinetune.py` imports `datautils_block`, but that file is not present in this checkout; verify or add the missing module before running e2e fine-tuning.
- Dataset helpers contain hard-coded cache locations such as `/mnt/bn/.../redpajama_cache` and `/data/shared_data/datasets`; update these deliberately if running outside that environment.
- `main.py` writes logs to `../log/` by default and dataset/model caches under `./cache`; keep generated artifacts out of source changes unless explicitly requested.

## Notice 
如果你要更改auto-round源码，必须征得同意
