#!/usr/bin/env python3
"""Evaluate an AutoRound model with lm-eval using local datasets only."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = (
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


def parse_tasks(value: str) -> list[str]:
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
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "checkpoints" / "results" / "Qwen3-4B-w4g128",
        help="AutoRound model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "datasets",
        help="Directory containing the local task datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_results",
        help="Directory for JSON results (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=list(TASKS),
        help="Comma-separated lm-eval tasks to run (default: all supported tasks)",
    )
    parser.add_argument("--device", default="cuda:0", help="lm-eval device (default: %(default)s)")
    parser.add_argument(
        "--batch-size",
        default="auto:8",
        help="Integer or lm-eval auto batch size such as auto:8 (default: %(default)s)",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--limit", type=number, default=None, help="Only for smoke tests; evaluate N examples per task")
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--add-bos-token", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
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


def validate_inputs(model_dir: Path, data_dir: Path) -> dict[str, Any] | None:
    required_model_files = ("config.json", "tokenizer_config.json")
    missing_model_files = [name for name in required_model_files if not (model_dir / name).is_file()]
    if missing_model_files:
        raise FileNotFoundError(f"Model directory is missing: {', '.join(missing_model_files)}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Local dataset directory not found: {data_dir}")

    # The manifest records how prepare_lm_eval_datasets.py created the snapshots,
    # but it is not needed to load them. Keep it as optional run metadata so an
    # existing local dataset tree can be evaluated directly.
    manifest_path = data_dir / "lm_eval_dataset_manifest.json"
    if not manifest_path.is_file():
        return None

    return json.loads(manifest_path.read_text(encoding="utf-8"))


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


def main() -> None:
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tasks = args.tasks
    manifest = validate_inputs(model_dir, data_dir)
    configure_huggingface_cache(data_dir)

    print(f"Using local datasets from: {data_dir}")
    if (model_dir / "quantization_config.json").is_file():
        print("Model type: quantized (quantization_config.json found)")
    else:
        print("Model type: floating-point (quantization_config.json not found)")
    if manifest is None:
        print("Dataset manifest: not found (optional; continuing with the local files)")
    else:
        print(f"Dataset manifest: {data_dir / 'lm_eval_dataset_manifest.json'}")

    # Import after configuring offline mode and cache paths.
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, make_table
    from local_eval_datasets import install_local_dataset_loader

    install_local_dataset_loader(data_dir)

    model_args = {
        "pretrained": str(model_dir),
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "add_bos_token": args.add_bos_token,
    }
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
        task_manager=TaskManager(),
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results on the main process")

    summary = build_summary(results, tasks)
    print(make_table(results))
    print_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{model_dir.name}_{timestamp}.json"
    payload = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": str(model_dir),
            "data_dir": str(data_dir),
            "offline": True,
            "device": args.device,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "tasks": tasks,
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
