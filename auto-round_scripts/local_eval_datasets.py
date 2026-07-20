"""Local dataset mirrors used by the local lm-eval scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetMirror:
    modelscope_id: str
    local_name: str
    dataset_name: str | None = None
    config_subdir: bool = True


# Keys match dataset_path values in lm-eval 0.4.11 task YAML files.
DATASET_MIRRORS = {
    "cais/mmlu": DatasetMirror("cais/mmlu", "mmlu"),
    "EleutherAI/lambada_openai": DatasetMirror("EleutherAI/lambada_openai", "lambada_openai"),
    "Rowan/hellaswag": DatasetMirror("allenai/hellaswag", "hellaswag", config_subdir=False),
    "allenai/winogrande": DatasetMirror("allenai/winogrande", "winogrande"),
    "baber/piqa": DatasetMirror("vikhyatk/piqa", "piqa", config_subdir=False),
    "truthfulqa/truthful_qa": DatasetMirror("evalscope/truthful_qa", "truthful_qa"),
    "allenai/openbookqa": DatasetMirror("allenai/openbookqa", "openbookqa"),
    "aps/super_glue": DatasetMirror("google/boolq", "boolq", dataset_name=None, config_subdir=False),
    "nyu-mll/glue": DatasetMirror("nyu-mll/glue", "glue"),
    "allenai/ai2_arc": DatasetMirror("allenai/ai2_arc", "ai2_arc"),
}


def snapshot_path(data_dir: Path, mirror: DatasetMirror) -> Path:
    return data_dir / mirror.local_name


def download_modelscope_snapshots(data_dir: Path) -> dict[str, str]:
    from modelscope import dataset_snapshot_download

    snapshots = {}
    unique_mirrors = {mirror.modelscope_id: mirror for mirror in DATASET_MIRRORS.values()}
    for index, (modelscope_id, mirror) in enumerate(unique_mirrors.items(), start=1):
        local_dir = snapshot_path(data_dir, mirror)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(unique_mirrors)}] Downloading modelscope://{modelscope_id}", flush=True)
        downloaded_path = dataset_snapshot_download(
            modelscope_id,
            local_dir=str(local_dir),
            ignore_patterns=[".cache/**"],
        )
        snapshots[modelscope_id] = str(Path(downloaded_path).resolve())
    return snapshots


def install_local_dataset_loader(data_dir: Path) -> None:
    """Redirect lm-eval dataset IDs to previously downloaded local snapshots."""
    import datasets

    original_load_dataset = datasets.load_dataset

    def parquet_files(local_path: Path, dataset_name: str | None, mirror: DatasetMirror) -> dict[str, list[str]]:
        if mirror.config_subdir:
            if not dataset_name:
                raise RuntimeError(f"A dataset configuration is required for {mirror.modelscope_id}")
            search_root = local_path / dataset_name
        else:
            search_root = local_path / "data"

        if not search_root.is_dir():
            raise FileNotFoundError(f"Local dataset configuration not found: {search_root}")

        split_files: dict[str, list[str]] = {}
        known_splits = ("train", "validation", "test", "dev", "auxiliary_train")
        for parquet_path in sorted(search_root.rglob("*.parquet")):
            split = parquet_path.parent.name if parquet_path.parent.name in known_splits else None
            if split is None:
                split = next(
                    (candidate for candidate in known_splits if parquet_path.name.startswith(f"{candidate}-")),
                    None,
                )
            if split is None:
                raise RuntimeError(f"Cannot determine split for local parquet file: {parquet_path}")
            split_files.setdefault(split, []).append(str(parquet_path))

        if not split_files:
            raise FileNotFoundError(f"No parquet files found under {search_root}")
        return split_files

    def load_local_dataset(path: str, name: str | None = None, **kwargs: Any):
        mirror = DATASET_MIRRORS.get(path)
        if mirror is None:
            raise RuntimeError(f"No local mirror configured for lm-eval dataset: {path}")

        local_path = snapshot_path(data_dir, mirror)
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"Local dataset snapshot not found: {local_path}. "
                "Run prepare_lm_eval_datasets.py first."
            )

        dataset_name = mirror.dataset_name if path == "aps/super_glue" else name
        data_files = parquet_files(local_path, dataset_name, mirror)
        dataset = original_load_dataset("parquet", data_files=data_files, **kwargs)

        # google/boolq stores the SuperGLUE label as a boolean `answer` column.
        if path == "aps/super_glue":
            for split in dataset:
                if "answer" in dataset[split].column_names:
                    dataset[split] = dataset[split].rename_column("answer", "label")
                    dataset[split] = dataset[split].cast_column("label", datasets.Value("int64"))
        return dataset

    datasets.load_dataset = load_local_dataset
