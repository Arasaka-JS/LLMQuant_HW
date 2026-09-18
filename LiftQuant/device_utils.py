"""Backend-agnostic device helpers (CUDA today, Ascend NPU when available).

LiftQuant currently runs on CUDA.  This module is the single place that knows
which accelerator is active, so the evaluation path can be prepared for Ascend
NPU (``torch_npu``) without touching the CUDA behaviour:

  * ``torch_npu`` is imported lazily and never at module import time, so a plain
    CUDA/CPU environment is completely unaffected by this module;
  * every helper falls back to the CUDA/CPU path when no NPU is present;
  * the NPU branch is exercised without hardware by
    ``allq/tools/device_utils_selfcheck.py`` (it stubs ``torch.npu``).

Nothing here changes numerics; it only resolves device strings/counts.
"""

import torch

__all__ = [
    "npu_importable",
    "npu_available",
    "device_type",
    "device_count",
    "device",
    "resolve",
    "synchronize",
    "empty_cache",
    "autocast",
]


def npu_importable() -> bool:
    """True when the ``torch_npu`` extension can be imported (cheap to call)."""
    try:
        import torch_npu  # noqa: F401
    except Exception:  # noqa: BLE001 - torch_npu may be missing or half-installed
        return False
    return True


def npu_available() -> bool:
    """True when an Ascend NPU is both installed and usable right now."""
    if not npu_importable():
        return False
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:  # noqa: BLE001
        return False


def device_type() -> str:
    """``"npu"`` when an Ascend NPU is usable, else ``"cuda"``, else ``"cpu"``."""
    if npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_count() -> int:
    kind = device_type()
    if kind == "npu":
        return torch.npu.device_count()
    if kind == "cuda":
        return torch.cuda.device_count()
    return 0


def device(index: int = 0) -> torch.device:
    """``torch.device`` for the active backend.

    Raises a readable error when the backend name is not registered in this torch
    build (e.g. ``"npu"`` without torch_npu installed) instead of leaking a bare
    ``RuntimeError: Expected one of cpu, cuda, ...``.
    """
    kind = device_type()
    if kind == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(f"{kind}:{index}")
    except RuntimeError as exc:  # backend not registered in this torch build
        raise RuntimeError(
            f"backend '{kind}' is not registered in this torch build "
            f"(torch {torch.__version__}); install the matching torch_npu / CUDA build. "
            f"Original error: {exc}"
        ) from exc


def resolve(spec):
    """Resolve a user supplied device spec to a device *string*.

    ``None``/``""``/``"auto"`` -> the active backend's default device (e.g.
    ``npu:0`` on Ascend, ``cuda:0`` on CUDA).  A device string is returned
    deliberately (not a ``torch.device``) so this works on any torch build and
    can be handed straight to transformers/lm-eval/accelerate.  An explicit
    backend that does not match the active one raises, instead of silently
    failing later inside those libraries.
    """
    if spec in (None, "", "auto"):
        kind = device_type()
        return "cpu" if kind == "cpu" else f"{kind}:0"
    spec = str(spec)
    kind = spec.split(":")[0]
    if kind not in ("cuda", "npu", "cpu"):
        raise ValueError(f"unsupported device '{spec}' (expected cuda[:i] | npu[:i] | cpu | auto)")
    active = device_type()
    if kind != "cpu" and kind != active:
        raise ValueError(
            f"device '{spec}' requested but the active backend is '{active}'. "
            "Use --device auto to follow the current accelerator."
        )
    return spec


def synchronize(index=None):
    kind = device_type()
    if kind == "npu":
        torch.npu.synchronize() if index is None else torch.npu.synchronize(index)
    elif kind == "cuda":
        torch.cuda.synchronize() if index is None else torch.cuda.synchronize(index)


def empty_cache():
    """Release the active backend's cached device memory (no-op on CPU)."""
    kind = device_type()
    if kind == "npu":
        torch.npu.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


def autocast(dtype=None, enabled=True):
    """``torch.amp.autocast`` bound to the active backend.

    On a torch build whose backend is not registered yet (e.g. ``"npu"`` without
    torch_npu) the error is re-raised with guidance instead of the raw
    "Expected one of cpu, cuda, ... device type" message.
    """
    kind = device_type()
    try:
        return torch.amp.autocast(device_type=kind, dtype=dtype, enabled=enabled)
    except RuntimeError as exc:
        raise RuntimeError(
            f"torch.amp.autocast does not support device_type='{kind}' in this torch build "
            f"({torch.__version__}); install the matching torch_npu / CUDA build. "
            f"Original error: {exc}"
        ) from exc
