# 后端抽象层 `device_utils`（为昇腾准备，CUDA 行为不变）

> 新增：`LiftQuant/device_utils.py`
> 自检：`allq/tools/device_utils_selfcheck.py`（stub NPU，**18/18 通过**，无需真机）
> 环境：`skw_quant_env`（torch 2.8.0+cu128），本机无 `torch_npu` / 无 `npu-smi`

## 本次规划

- 提供一个**唯一**的后端判定点，让评测路径具备"随时可上昇腾"的能力；
- 硬约束：**在没有 torch_npu 的环境里行为完全不变**（不能因为引入抽象层而影响现有 CUDA 评测）；
- 记录两条在实现/自检中真实踩到的坑。

## 设计原则

1. **懒加载**：`torch_npu` 只在 `npu_importable()` 里、需要时才 import；模块顶部不 import。
2. **自动探测**：`npu_available()` 要求"能 import 且 `torch.npu.is_available()` 为真"，避免半装环境误判。
3. **返回字符串**：`resolve()` 返回设备**字符串**（`cuda:0` / `npu:0` / `cpu`），可直接交给 transformers / lm-eval / accelerate；`device()` 才返回 `torch.device`。
4. **不静默降级**：显式指定了与当前后端不符的设备（如在 NPU 机器上写 `cuda:0`）直接报错，而不是稍后在库里以奇怪方式失败。

## API

| 函数 | 作用 |
|---|---|
| `npu_importable()` | 能否 import `torch_npu` |
| `npu_available()` | `torch_npu` 可用且 `torch.npu.is_available()` |
| `device_type()` | `"npu"` / `"cuda"` / `"cpu"`（优先级即此顺序） |
| `device_count()` | 当前后端的设备数（0 表示 CPU） |
| `device(index=0)` | `torch.device`；后端未注册时抛**可读**错误 |
| `resolve(spec)` | `None/""/"auto"` → 当前后端默认设备；显式 `cuda[:i]/npu[:i]/cpu` 校验后返回字符串 |
| `synchronize(index=None)` | 对应后端的 `synchronize` |
| `empty_cache()` | 对应后端的 `empty_cache`（CPU 下 no-op） |
| `autocast(dtype=None, enabled=True)` | `torch.amp.autocast(device_type=当前后端)` |

## 踩坑（自检抓到的真实问题）

1. **`torch.device("npu:0")` 在 CUDA-only 构建里会崩**：报
   `RuntimeError: Expected one of cpu, cuda, ... device type at start of device string: npu`。
   真实 `torch_npu` 会把 privateuse1 后端 rename 成 `npu` 所以能用，但在没有它的环境（以及 stub 场景）不可用。
   处置：`device()` 捕获并重抛带指引的错误；`resolve()` **完全不依赖 `torch.device` 解析**，因此 stub 下也能正确返回 `"npu:0"`。
2. **`torch.amp.autocast(device_type="npu")` 同理**会报 device type 错误 → 同样包成可读错误。
   自检对这两项采用"可用则断言可用、不可用则断言报错可读"的双分支写法。

## 接线（CUDA 行为零改变）

| 位置 | 改动 |
|---|---|
| `LiftQuant/e2e_utils.py:9-10` | `from device_utils import device_count, empty_cache` |
| `LiftQuant/e2e_utils.py:387` | `num_gpus = device_count() or 1`（原 `torch.cuda.device_count()`，并顺手防 0） |
| `LiftQuant/e2e_utils.py:290/399` | `torch.cuda.empty_cache()` → `empty_cache()` |
| `allq/eval/eval_quant_lmeval.py` | `--device` 默认 `auto`，在 `main()` 里 `resolve_device(args.device)` |
| `allq/scripts/eval_by_lmeval.sh` | 新增 `EVAL_DEVICE`（默认 `cuda:0`，保持原行为）；启动时按 `EVAL_DEVICE` 前缀决定用 `ASCEND_RT_VISIBLE_DEVICES` 还是 `CUDA_VISIBLE_DEVICES` |

## 验证

### 1) 无硬件的分支覆盖（stub）

```
python allq/tools/device_utils_selfcheck.py      # 18/18 PASS
```

覆盖：不 eager import torch_npu、基准后端判定、`device_count`、`resolve('auto'/'cpu'/'npu:3')`、NPU 激活时拒绝 `cuda:0`、`synchronize` 派发到 `torch.npu`、不可用时回退 cuda、以及上面两条可读报错。

> 边界：stub **只能验证选择逻辑**，不能验证真实昇腾算子（那要用 `ascend_op_probe.py` 到真机上跑）。

### 2) CUDA 行为不变的证据

同一条 lm-eval 冒烟（Qwen3-4B 全量化，hellaswag LIMIT=8）在接线前后：

```
acc = 0.250   acc_norm = 0.625      # 两次完全一致
```

## 局限与后续

- `device()` 返回 `torch.device` 的路径**只有真机**能最终验证（需要 torch_npu 注册 `npu` 设备类型）。
- 本阶段**没有**动量化训练路径（`main.py` / `liftq.py` / `stage_training.py` 里的 `device_type='cuda'`、`.cuda()` 等），那些属于整库移植（见 `06_ascend_porting_plan.md` 的 Phase E）。
