# AutoRound Environment Setup

This note records the environment setup used to install the local `auto-round` project in the current workspace.

The requested workspace path resolves to the same project root used during installation:

```bash
readlink -f /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW
# /data/kris/sunkaiwei/LLMQuant_HW
```

## 1. Environment Used

Current Conda environment:

```bash
conda activate skw_liftquant_env
```

Observed environment after activation:

```bash
python -V
# Python 3.12.13

python -c "import sys; print(sys.executable)"
# /home/kris/miniconda3/envs/skw_liftquant_env/bin/python

python -m pip --version
# pip 26.1.2 from /home/kris/miniconda3/envs/skw_liftquant_env/lib/python3.12/site-packages/pip (python 3.12)
```

PyTorch check:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
# 2.6.0+cu124
# cuda True
```

## 2. Install AutoRound From Local Source

Run the install from the `auto-round` source directory:

```bash
cd /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/auto-round
python -m pip install --no-build-isolation -e .
```

`--no-build-isolation` is important because the project build expects the already-installed PyTorch package in the active environment.

During installation, the following additional packages were installed because they were not already present:

```text
annotated-types==0.7.0
auto-round==0.15.0.dev11+g4d24c405
py-cpuinfo==9.0.0
pydantic==2.13.4
pydantic-core==2.46.4
typing-inspection==0.4.2
```

## 3. Verify Installation

Verify the Python import:

```bash
python -c "import auto_round; print(auto_round.__version__)"
# 0.15.0
```

Verify package metadata:

```bash
python -m pip show auto-round
```

Expected important fields:

```text
Name: auto-round
Version: 0.15.0.dev11+g4d24c405
Editable project location: /data/kris/sunkaiwei/LLMQuant_HW/auto-round
Requires: accelerate, datasets, numpy, py-cpuinfo, pydantic, torch, tqdm, transformers
```

Verify CLI entry points:

```bash
auto-round --help
auto-round-best --help
auto-round-light --help
auto-round-rtn --help
```

All four commands should print the AutoRound quantization CLI help.

## 4. Optional Download Configuration

If Hugging Face access is slow or unavailable, AutoRound supports ModelScope by setting:

```bash
export AR_USE_MODELSCOPE=1
```

Use this only when model or dataset downloads should go through ModelScope.

## 5. Quick Quantization Command

The downloaded pile-10k dataset lives here:

```text
/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/pile-10k
```

That directory contains a parquet shard. AutoRound's local dataset loader expects a `.json` or `.jsonl` file, so the parquet `text` column was converted to:

```text
/home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/pile-10k/pile_10k_text.jsonl
```

Regenerate it with:

```bash
python /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/auto-round_scripts/prepare_pile10k_jsonl.py \
  /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/pile-10k/data/train-00000-of-00001-4746b8785c874cc7.parquet \
  /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/pile-10k/pile_10k_text.jsonl
```

The existing script in this directory runs W4A16 quantization for the local Qwen checkpoint and uses the local JSONL calibration file by default:

```bash
cd /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/auto-round_scripts
bash run_quant_by_autoround.sh
```

Equivalent explicit command:

```bash
auto-round \
  --model /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/checkpoints/Qwen/Qwen3-4B \
  --scheme W4A16 \
  --format auto_round \
  --dataset /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/datasets/pile-10k/pile_10k_text.jsonl \
  --output_dir /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/auto-round_scripts/auto_round_results
```

## 6. Useful Troubleshooting

If import or CLI resolution fails, confirm the active Python and PATH are from the intended Conda environment:

```bash
which python
which auto-round
python -m pip show auto-round
```

If the build fails while importing PyTorch during installation, rerun the install with:

```bash
python -m pip install --no-build-isolation -e /home/kris/workspace/sunkaiwei/Quant/LLMQuant_HW/auto-round
```

If CUDA is expected but unavailable, verify PyTorch first:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
