#!/usr/bin/env python3
"""Download and prepare the lm-eval datasets used by AutoRound evaluation."""

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


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

MODELSCOPE_DATASET_REPOSITORIES = (
    "cais/mmlu",
    "EleutherAI/lambada_openai",
    "allenai/hellaswag",
    "allenai/winogrande",
    "vikhyatk/piqa",
    "evalscope/truthful_qa",
    "allenai/openbookqa",
    "google/boolq",
    "nyu-mll/glue",
    "allenai/ai2_arc",
)


def configure_huggingface_cache(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / ".lm_eval_cache"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_ASSETS_CACHE"] = str(cache_dir / "assets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir / "modelscope")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download all datasets required by the standard AutoRound lm-eval task set."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "datasets",
        help="Directory for ModelScope snapshots and processed datasets (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    configure_huggingface_cache(data_dir)

    # Import after configuring the cache; datasets reads these paths at import time.
    from lm_eval.tasks import TaskManager, get_task_dict
    from local_eval_datasets import download_modelscope_snapshots, install_local_dataset_loader

    snapshots = download_modelscope_snapshots(data_dir)
    install_local_dataset_loader(data_dir)

    task_manager = TaskManager()
    for index, task_name in enumerate(TASKS, start=1):
        print(f"[{index}/{len(TASKS)}] Preparing {task_name}", flush=True)
        task_dict = get_task_dict([task_name], task_manager=task_manager)
        if not task_dict:
            raise RuntimeError(f"lm-eval did not resolve task: {task_name}")
        del task_dict
        gc.collect()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "tasks": list(TASKS),
        "dataset_source": "ModelScope",
        "modelscope_dataset_repositories": list(MODELSCOPE_DATASET_REPOSITORIES),
        "modelscope_snapshots": snapshots,
        "versions": {
            "lm-eval": version("lm-eval"),
            "datasets": version("datasets"),
            "modelscope": version("modelscope"),
        },
    }
    manifest_path = data_dir / "lm_eval_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared all datasets in {data_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
