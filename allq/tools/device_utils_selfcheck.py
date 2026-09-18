#!/usr/bin/env python3
"""Verify LiftQuant/device_utils.py without any NPU hardware.

The NPU branch of the device abstraction is exercised by installing a *stub*
``torch_npu`` module and a stub ``torch.npu`` object, so the selection logic can
be validated on a CUDA/CPU-only machine.  This cannot validate real Ascend
operators (that is what ``allq/tools/ascend_op_probe.py`` is for) -- it only
proves that the abstraction picks the right backend, resolves device strings
correctly and never imports torch_npu eagerly.

Usage:  python allq/tools/device_utils_selfcheck.py
Exit code 0 = all checks passed.
"""

import sys
import types
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "LiftQuant"))

import device_utils as du  # noqa: E402


RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((bool(condition), f"{name:<46}{detail}"))


class NpuStub:
    """Minimal stand-in for the torch.npu namespace."""

    def __init__(self, available=True, count=8):
        self._available = available
        self._count = count
        self.calls = []

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def synchronize(self, *args):
        self.calls.append(("synchronize", args))

    def current_device(self):
        return 0


def install_stub(available=True, count=8):
    stub = NpuStub(available=available, count=count)
    module = types.ModuleType("torch_npu")
    module.__version__ = "stub"
    sys.modules["torch_npu"] = module
    torch.npu = stub
    return stub


def remove_stub():
    sys.modules.pop("torch_npu", None)
    if hasattr(torch, "npu"):
        delattr(torch, "npu")


def main():
    # --- baseline: no torch_npu importable ---------------------------------
    check("import device_utils does not import torch_npu",
          "torch_npu" not in sys.modules)
    baseline_kind = du.device_type()
    check("baseline backend is cuda on this machine", baseline_kind == "cuda",
          f"device_type={baseline_kind}")
    check("npu_available() is False without torch_npu", du.npu_available() is False)
    check("device_count() > 0 on CUDA", du.device_count() > 0,
          f"count={du.device_count()}")
    check("resolve('auto') follows the active backend",
          du.resolve("auto") == f"{baseline_kind}:0", f"-> {du.resolve('auto')}")
    check("resolve('cpu') is always allowed", du.resolve("cpu") == "cpu")
    check("autocast() binds to the active backend",
          du.autocast().device == baseline_kind)

    # --- stub NPU available -------------------------------------------------
    stub = install_stub(available=True, count=8)
    try:
        check("npu_available() True with stub NPU", du.npu_available() is True)
        check("device_type() becomes 'npu'", du.device_type() == "npu",
              f"-> {du.device_type()}")
        check("device_count() uses torch.npu", du.device_count() == 8,
              f"-> {du.device_count()}")
        # A stub cannot register the "npu" device *type* in a CUDA-only torch
        # build, so torch.device('npu:0') is unavailable here.  What we can check
        # is that the error is readable instead of a bare device-string error
        # (with real torch_npu the rename registers "npu" and this succeeds).
        try:
            du.device(1)
            check("device(1) works (real npu backend registered)", True, "-> npu:1")
        except RuntimeError as exc:
            check("device(1) raises a readable backend error under the stub",
                  "npu" in str(exc) and "torch_npu" in str(exc), f"-> {str(exc)[:60]}...")
        check("resolve('auto') -> 'npu:0'", du.resolve("auto") == "npu:0",
              f"-> {du.resolve('auto')}")
        check("resolve(None) -> 'npu:0'", du.resolve(None) == "npu:0")
        check("resolve('npu:3') accepted", du.resolve("npu:3") == "npu:3")
        try:
            du.resolve("cuda:0")
            check("resolve('cuda:0') rejected while NPU is active", False)
        except ValueError:
            check("resolve('cuda:0') rejected while NPU is active", True)
        du.synchronize()
        check("synchronize() dispatches to torch.npu", stub.calls == [("synchronize", ())],
              f"calls={stub.calls}")
        # autocast also needs the backend registered in the torch build; accept
        # either a working autocast or the guided error (real torch_npu -> works).
        try:
            check("autocast() binds to npu", du.autocast().device == "npu")
        except RuntimeError as exc:
            check("autocast() raises a readable backend error under the stub",
                  "npu" in str(exc) and "torch_npu" in str(exc), f"-> {str(exc)[:60]}...")
    finally:
        remove_stub()

    # --- stub present but unavailable --------------------------------------
    install_stub(available=False, count=0)
    try:
        check("unavailable stub NPU falls back to cuda",
              du.device_type() == baseline_kind, f"-> {du.device_type()}")
    finally:
        remove_stub()

    print("device_utils self-check (stubbed NPU; no Ascend hardware needed)")
    print("=" * 78)
    for ok, line in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {line}")
    failed = sum(1 for ok, _ in RESULTS if not ok)
    print("=" * 78)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
