#!/usr/bin/env python3
"""Evaluate AutoRound or LiftQuant quantized models with lm-eval using local datasets only."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "mmlu",
    "lambada_openai",
    "hellaswag",
    "winogrande",
    "piqa",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "openbookqa",
    "boolq",
    "rte",
    "arc_easy",
    "arc_challenge",
)

SUMMARY_METRICS = {
    "mmlu": ("acc",),
    "lambada_openai": ("acc", "perplexity"),
    "hellaswag": ("acc_norm",),
    "winogrande": ("acc",),
    "piqa": ("acc_norm",),
    "truthfulqa_mc1": ("acc",),
    "truthfulqa_mc2": ("acc",),
    "openbookqa": ("acc_norm",),
    "boolq": ("acc",),
    "rte": ("acc",),
    "arc_easy": ("acc_norm",),
    "arc_challenge": ("acc_norm",),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_tasks(value: str) -> list[str]:
    if value == "all":
        return list(DEFAULT_TASKS)

    tasks = [task.strip() for task in value.split(",") if task.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")

    unknown = [task for task in tasks if task not in SUMMARY_METRICS]
    if unknown:
        supported = ", ".join(SUMMARY_METRICS)
        raise argparse.ArgumentTypeError(
            f"unsupported task(s): {', '.join(unknown)}. Supported tasks: {supported}"
        )
    return tasks


def number(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("autoround", "liftquant"), default="autoround")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "checkpoints" / "results" / "Qwen3-4B-w4g128",
        help="AutoRound model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--fp-model-path",
        type=Path,
        default=root / "checkpoints" / "Qwen" / "Qwen3-4B",
        help="Original floating-point Hugging Face model directory for LiftQuant.",
    )
    parser.add_argument(
        "--quant-model-path",
        type=Path,
        default=root / "LiftQuant" / "qmodels" / "Qwen3-4B" / "Qwen3-4B+24to8-packed.pth",
        help="LiftQuant .pth checkpoint, or prefix when --load-per-layer is set.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "datasets",
        help="Directory containing local task datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "lm_eval" / "eval_results",
        help="Directory for JSON results (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=list(DEFAULT_TASKS),
        help="Comma-separated lm-eval tasks, or 'all' (default: all supported tasks)",
    )
    parser.add_argument("--device", default="cuda:0", help="lm-eval device (default: %(default)s)")
    parser.add_argument("--batch-size", default="auto:8")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--limit", type=number, default=None, help="Only for smoke tests; evaluate N examples per task")
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--add-bos-token", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--task-by-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the task list one task at a time, matching AutoRound CLI behavior (default: true)",
    )
    parser.add_argument("--wbits", type=int, default=2, help="LiftQuant weight bits")
    parser.add_argument("--expc", default="24to8", help="LiftQuant expc setting")
    parser.add_argument("--w-ternary", default=None)
    parser.add_argument("--load-per-layer", action="store_true")
    parser.add_argument("--auto-mix-precision", action="store_true")
    return parser.parse_args()


def configure_huggingface_cache(data_dir: Path) -> None:
    cache_dir = data_dir / ".lm_eval_cache"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_ASSETS_CACHE"] = str(cache_dir / "assets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir / "modelscope")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def validate_common(data_dir: Path) -> dict[str, Any] | None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Local dataset directory not found: {data_dir}")

    manifest_path = data_dir / "lm_eval_dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_hf_model_dir(model_dir: Path, label: str) -> None:
    required_model_files = ("config.json", "tokenizer_config.json")
    missing_model_files = [name for name in required_model_files if not (model_dir / name).is_file()]
    if missing_model_files:
        raise FileNotFoundError(f"{label} is missing: {', '.join(missing_model_files)}")


def validate_liftquant_checkpoint(quant_model_path: Path, load_per_layer: bool) -> None:
    if load_per_layer:
        non_layer_path = Path(f"{quant_model_path}-non_layer.pth")
        if not non_layer_path.is_file():
            raise FileNotFoundError(f"LiftQuant non-layer checkpoint not found: {non_layer_path}")
    elif not quant_model_path.is_file():
        raise FileNotFoundError(f"LiftQuant checkpoint not found: {quant_model_path}")


def find_metric(metrics: dict[str, Any], metric_name: str) -> Any | None:
    for key, value in metrics.items():
        if key.split(",", maxsplit=1)[0] == metric_name:
            return value
    return None


def build_summary(results: dict[str, Any], tasks: list[str]) -> list[dict[str, Any]]:
    task_results = results.get("results", {})
    group_results = results.get("groups", {})
    summary = []
    for task_name in tasks:
        metric_names = SUMMARY_METRICS[task_name]
        metrics = group_results.get(task_name) or task_results.get(task_name) or {}
        row = {"task": task_name}
        for metric_name in metric_names:
            row[metric_name] = find_metric(metrics, metric_name)
        summary.append(row)
    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nCore metric summary")
    print("=" * 72)
    for row in summary:
        values = []
        for name, value in row.items():
            if name == "task":
                continue
            rendered = "n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
            values.append(f"{name}={rendered}")
        print(f"{row['task']:<22} {'  '.join(values)}")


def build_autoround_lm(args: argparse.Namespace):
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=str(args.model),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        trust_remote_code=args.trust_remote_code,
        add_bos_token=args.add_bos_token,
    )


def build_liftquant_lm(args: argparse.Namespace):
    from lm_eval.models.huggingface import HFLM

    root = project_root()
    liftquant_root = root / "LiftQuant"
    sys.path.insert(0, str(liftquant_root))
    previous_cwd = Path.cwd()
    os.chdir(liftquant_root)
    try:
        from e2e_utils import load_quantized_model

        model, tokenizer = load_quantized_model(
            fp_model_path=str(args.fp_model_path),
            quant_model_path=str(args.quant_model_path),
            wbits=args.wbits,
            expc=args.expc,
            w_ternary=args.w_ternary,
            load_per_layer=args.load_per_layer,
            auto_mix_precision=args.auto_mix_precision,
            eval_dtype=args.dtype,
        )
    finally:
        os.chdir(previous_cwd)

    model.eval()
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dtype=args.dtype,
        add_bos_token=args.add_bos_token,
    )


def evaluate_lm(args: argparse.Namespace, lm_eval_model, tasks: list[str]) -> dict[str, Any]:
    import lm_eval
    from lm_eval.tasks import TaskManager

    return lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
        task_manager=TaskManager(),
    )


def merge_lm_eval_results(results_by_task: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for results in results_by_task:
        for key, value in results.items():
            if isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            elif key not in merged:
                merged[key] = value
    return merged


def run_evaluation(args: argparse.Namespace, tasks: list[str]) -> dict[str, Any]:
    from lm_eval.utils import make_table

    if args.backend == "autoround":
        lm_eval_model = build_autoround_lm(args)
    else:
        lm_eval_model = build_liftquant_lm(args)

    if not args.task_by_task:
        results = evaluate_lm(args, lm_eval_model, tasks)
        if results is None:
            raise RuntimeError("lm-eval returned no results on the main process")
        print(make_table(results))
        return results

    results_by_task = []
    for index, task_name in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] Evaluating {task_name}", flush=True)
        task_results = evaluate_lm(args, lm_eval_model, [task_name])
        if task_results is None:
            raise RuntimeError(f"lm-eval returned no results for task: {task_name}")
        print(make_table(task_results))
        results_by_task.append(task_results)
    return merge_lm_eval_results(results_by_task)


def result_stem(args: argparse.Namespace) -> str:
    if args.backend == "liftquant":
        return args.quant_model_path.stem
    return args.model.name


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.fp_model_path = args.fp_model_path.expanduser().resolve()
    args.quant_model_path = args.quant_model_path.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tasks = args.tasks

    manifest = validate_common(data_dir)
    if args.backend == "autoround":
        validate_hf_model_dir(args.model, "AutoRound model directory")
    else:
        validate_hf_model_dir(args.fp_model_path, "FP model directory")
        validate_liftquant_checkpoint(args.quant_model_path, args.load_per_layer)

    configure_huggingface_cache(data_dir)

    print(f"Backend: {args.backend}")
    print(f"Using local datasets from: {data_dir}")
    if args.backend == "autoround":
        print(f"AutoRound model: {args.model}")
    else:
        print(f"FP model: {args.fp_model_path}")
        print(f"LiftQuant checkpoint: {args.quant_model_path}")
        print(f"LiftQuant cfg: wbits={args.wbits} expc={args.expc}")
    if manifest is None:
        print("Dataset manifest: not found (optional; continuing with the local files)")
    else:
        print(f"Dataset manifest: {data_dir / 'lm_eval_dataset_manifest.json'}")

    from lm_eval.utils import handle_non_serializable
    from local_eval_datasets import install_local_dataset_loader

    install_local_dataset_loader(data_dir)

    results = run_evaluation(args, tasks)

    summary = build_summary(results, tasks)
    print_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{result_stem(args)}_{args.backend}_{timestamp}.json"
    payload = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model": str(args.model) if args.backend == "autoround" else None,
            "fp_model": str(args.fp_model_path) if args.backend == "liftquant" else None,
            "quant_model": str(args.quant_model_path) if args.backend == "liftquant" else None,
            "data_dir": str(data_dir),
            "offline": True,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "tasks": tasks,
            "wbits": args.wbits if args.backend == "liftquant" else None,
            "expc": args.expc if args.backend == "liftquant" else None,
            "load_per_layer": args.load_per_layer if args.backend == "liftquant" else None,
            "auto_mix_precision": args.auto_mix_precision if args.backend == "liftquant" else None,
            "dataset_manifest": manifest,
        },
        "summary": summary,
        "lm_eval": results,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, default=handle_non_serializable) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
