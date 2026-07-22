#!/usr/bin/env python3
"""Convert a LiftQuant TmpLinear checkpoint to packed FWTLinear format."""

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


QUANTIZED_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
    "out_proj",
    "in_proj_qkv",
    "in_proj_z",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "LiftQuant" / "qmodels" / "Qwen3-4B" / "Qwen3-4B+24to8.pth"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp-model-path",
        type=Path,
        default=project_root / "checkpoints" / "Qwen" / "Qwen3-4B",
        help="Original floating-point Hugging Face model directory.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=default_input,
        help="LiftQuant TmpLinear .pth checkpoint to convert.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Packed FWTLinear output path. Defaults to INPUT stem + '-packed.pth'.",
    )
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--expc", default="24to8")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--check-load-only",
        action="store_true",
        help="Only rebuild the TmpLinear structure and load the checkpoint, then exit before conversion.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def should_replace_linear(name: str) -> bool:
    return any(target in name for target in QUANTIZED_LINEAR_NAMES) and "orilinear" not in name


def set_child(module: torch.nn.Module, name: str, child: torch.nn.Module) -> None:
    setattr(module, name, child)


def replace_linears_with_tmplinear(module: torch.nn.Module, args: SimpleNamespace, TmpLinear) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and should_replace_linear(child_name):
            replacement = TmpLinear(
                child,
                args.wbits,
                expc=args.expc,
                training_trans=True,
                groupsize=-1,
                fast_nearest=True,
            )
            set_child(module, child_name, replacement)
        else:
            replace_linears_with_tmplinear(child, args, TmpLinear)


def convert_tmplinear_modules(module: torch.nn.Module, args: argparse.Namespace, TmpLinear, FWTLinear, prefix: str = "") -> int:
    converted = 0
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, TmpLinear):
            print(f"Converting {child_prefix}", flush=True)
            child = child.to(args.device).float()
            with torch.no_grad():
                fwtlinear = FWTLinear()
                fwtlinear.convert_form_tmplinear(
                    child,
                    bits=args.wbits,
                    expc=args.expc,
                    training_trans=True,
                    groupsize=-1,
                    fast_nearest=True,
                )
                fwtlinear.bit_channel_convert()
                fwtlinear = fwtlinear.to(torch_dtype(args.dtype))
                fwtlinear.pack_to_int8()
                fwtlinear = fwtlinear.to("cpu")
            set_child(module, child_name, fwtlinear)
            del child, fwtlinear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            converted += 1
        else:
            converted += convert_tmplinear_modules(child, args, TmpLinear, FWTLinear, child_prefix)
    return converted


def validate_tmplinear_checkpoint(state_dict: dict[str, torch.Tensor]) -> None:
    has_tmplinear = any(key.endswith(".orilinear.weight") for key in state_dict)
    if not has_tmplinear:
        raise RuntimeError("Input checkpoint does not look like a TmpLinear checkpoint: missing *.orilinear.weight keys")
    if any(key.endswith(".packed_weight") for key in state_dict):
        raise RuntimeError("Input checkpoint already contains packed_weight keys; use it directly for evaluation")


def prepare_tmplinear_for_checkpoint(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], TmpLinear) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, TmpLinear):
            continue

        prefix = f"{name}."

        a1 = state_dict.get(prefix + "a1")
        if a1 is not None and tuple(module.a1.shape) != tuple(a1.shape):
            module.a1 = torch.nn.Parameter(torch.empty_like(a1, device="meta"))

        scale = state_dict.get(prefix + "quantizer.scale")
        if scale is not None and tuple(module.quantizer.scale.shape) != tuple(scale.shape):
            module.quantizer._buffers["scale"] = torch.empty_like(scale, device="meta")

        zero = state_dict.get(prefix + "quantizer.zero")
        if zero is not None and tuple(module.quantizer.zero.shape) != tuple(zero.shape):
            module.quantizer._buffers["zero"] = torch.empty_like(zero, device="meta")

        alpha = state_dict.get(prefix + "quantizer.alpha")
        if alpha is not None and not hasattr(module.quantizer, "alpha"):
            module.quantizer.register_parameter(
                "alpha",
                torch.nn.Parameter(torch.empty_like(alpha, device="meta")),
            )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    liftquant_root = project_root / "LiftQuant"
    fp_model_path = args.fp_model_path.expanduser().resolve()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path or input_path.with_name(f"{input_path.stem}-packed{input_path.suffix}")
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if not fp_model_path.is_dir():
        raise FileNotFoundError(f"FP model directory not found: {fp_model_path}")

    sys.path.insert(0, str(liftquant_root))
    previous_cwd = Path.cwd()
    os.chdir(liftquant_root)
    try:
        from quantize.tmplinear import FWTLinear, TmpLinear

        print(f"Loading TmpLinear checkpoint: {input_path}", flush=True)
        state_dict = torch.load(input_path, map_location="cpu")
        validate_tmplinear_checkpoint(state_dict)

        print(f"Creating model structure from: {fp_model_path}", flush=True)
        config = AutoConfig.from_pretrained(fp_model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype(args.dtype),
                trust_remote_code=True,
            )

        replace_args = SimpleNamespace(wbits=args.wbits, expc=args.expc)
        replace_linears_with_tmplinear(model, replace_args, TmpLinear)
        model.tie_weights()
        prepare_tmplinear_for_checkpoint(model, state_dict, TmpLinear)

        print("Loading checkpoint into TmpLinear structure", flush=True)
        load_result = model.load_state_dict(state_dict, assign=True, strict=False)
        if load_result.unexpected_keys:
            print(f"Unexpected keys: {len(load_result.unexpected_keys)}", flush=True)
        if load_result.missing_keys:
            print(f"Missing keys: {len(load_result.missing_keys)}", flush=True)
        if args.check_load_only:
            print("Checkpoint loaded successfully; stopping before conversion", flush=True)
            return
        del state_dict
        gc.collect()

        print("Converting TmpLinear modules to packed FWTLinear", flush=True)
        converted = convert_tmplinear_modules(model, args, TmpLinear, FWTLinear)
        if converted == 0:
            raise RuntimeError("No TmpLinear modules were converted")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving packed checkpoint: {output_path}", flush=True)
        torch.save(model.state_dict(), output_path)
        print(f"Converted {converted} modules", flush=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
