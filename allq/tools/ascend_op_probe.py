#!/usr/bin/env python3
"""Probe the operators that LiftQuant's fast MoE evaluation path needs.

The vectorized MoE forward in ``LiftQuant/models/moe_fast_eval.py`` uses a small
set of "glue" operators on top of ``torch.bmm``.  On CUDA they are all trivially
available; on Ascend NPU a few historically fall back to CPU (which turns into a
host sync and destroys performance) or are unsupported outright.  This script
answers, on the machine it is run on:

  1. is each op available, and how fast is it,
  2. is ``torch.bmm`` usable with contiguous / transposed (non-contiguous)
     operands at the batch sizes the MoE path produces, and is bf16 supported,
  3. does this torch_npu build expose native fused MoE operators
     (``npu_moe_*`` / grouped matmul), which would be the preferred Ascend path.

Usage
-----
    python allq/tools/ascend_op_probe.py                  # auto-detect device
    python allq/tools/ascend_op_probe.py --device npu     # force Ascend NPU
    python allq/tools/ascend_op_probe.py --device cuda --json /tmp/probe.json

Everything is read-only; nothing is written except the optional JSON report.
"""

import argparse
import json
import statistics
import time


MOE_SHAPES = {
    # name: (num_experts, tokens_per_expert, hidden, intermediate)
    "prefill_2048tok_top8": (128, 128, 2048, 768),
    "decode_1tok_top8": (8, 1, 2048, 768),
}


def pick_device(requested, torch):
    if requested == "auto":
        if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def make_sync(device, torch):
    if device == "npu":
        return torch.npu.synchronize
    if device == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def timeit(fn, sync, repeat=20, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def probe(fn, sync, repeat=20):
    """Return {'ok': True, 'ms': ...} or {'ok': False, 'error': ...}."""
    try:
        fn()
        sync()
    except Exception as exc:  # noqa: BLE001 - the whole point is the report
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:
        return {"ok": True, "ms": round(timeit(fn, sync, repeat=repeat), 4)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": True, "ms": None, "error": f"{type(exc).__name__}: {exc}"[:300]}


def op_probes(device, torch):
    """Small-tensor probes for every glue op the MoE fast path relies on."""
    dev = torch.device(device)
    fp16 = torch.float16
    E, H = 128, 2048
    idx = torch.randint(0, E, (2048 * 8,), device=dev)
    tokens = torch.arange(2048, device=dev).repeat_interleave(8)
    f32 = torch.randn(4096, device=dev)
    u8 = torch.randint(0, 256, (4096,), device=dev, dtype=torch.uint8)
    x3 = torch.randn(E, 8, H, device=dev, dtype=fp16)
    w_kn = torch.randn(E, 768, H, device=dev, dtype=fp16)   # (E, N, K) = F.linear layout
    w_kn_contig = w_kn.transpose(1, 2).contiguous()          # pre-transposed (E, K, N)
    flat = x3.reshape(-1, H)
    dest = torch.zeros(E, 512, H, device=dev, dtype=fp16)
    src = torch.randn(1024, H, device=dev, dtype=fp16)

    return {
        # --- core GEMM (the MoE fast path is built on this) ---
        "bmm_contig_B": lambda: torch.bmm(x3, w_kn_contig),
        "bmm_strided_B": lambda: torch.bmm(x3, w_kn.transpose(1, 2)),
        "matmul_2d": lambda: flat @ w_kn[0].transpose(0, 1),
        # --- routing / bucketing glue ---
        "argsort_stable": lambda: torch.argsort(idx, stable=True),
        "argsort_plain": lambda: torch.argsort(idx),
        "bincount": lambda: torch.bincount(idx, minlength=E),
        "cumsum": lambda: torch.cumsum(torch.bincount(idx, minlength=E), 0),
        "repeat_interleave": lambda: tokens[:1024].repeat_interleave(2),
        "index_select": lambda: flat.index_select(0, tokens[:1024]),
        "advanced_index_assign": lambda: dest.__setitem__(
            (idx[:512] % E, torch.arange(512, device=dev) % 256), src[:512]
        ),
        "gather_2d": lambda: flat.gather(0, (tokens[:1024] % E).unsqueeze(1).expand(-1, H)),
        "unique": lambda: torch.unique(idx[:64]),
        "topk": lambda: torch.topk(f32, 8, dim=-1),
        "index_add": lambda: dest.reshape(-1, H).index_add_(0, tokens[:1024] % E, src[:1024]),
        "silu": lambda: torch.nn.functional.silu(x3),
        # --- quantized-weight unpacking (needed by the packed checkpoint format) ---
        "uint8_bitwise_and": lambda: u8.unsqueeze(-1) & torch.arange(8, device=dev, dtype=torch.uint8),
        "uint8_to_float": lambda: u8.to(torch.float32) - 0.5,
        # --- host sync (what the fast path tries to minimise) ---
        "item_sync": lambda: float(f32.sum().item()),
    }


def bmm_benchmarks(device, torch):
    """bmm at the shapes the MoE fast path actually produces."""
    dev = torch.device(device)
    sync = make_sync(device, torch)
    out = {}
    for name, (num_experts, tokens_per_expert, hidden, inter) in MOE_SHAPES.items():
        x = torch.randn(num_experts, tokens_per_expert, hidden, device=dev, dtype=torch.float16)
        # native HF layout is (E, 2I, H) / (E, H, I); bmm wants (E, K, N), so the
        # weight is used either as a pre-transposed contiguous tensor (candidate
        # optimisation) or as a strided view (what moe_fast_eval does today).
        gate_up_nk = torch.randn(num_experts, 2 * inter, hidden, device=dev, dtype=torch.float16)
        gate_up_kn = gate_up_nk.transpose(1, 2).contiguous()
        down_nk = torch.randn(num_experts, hidden, inter, device=dev, dtype=torch.float16)
        down_kn = down_nk.transpose(1, 2).contiguous()
        z = torch.randn(num_experts, tokens_per_expert, inter, device=dev, dtype=torch.float16)
        entry = {
            "num_experts": num_experts,
            "tokens_per_expert": tokens_per_expert,
            "gate_up_bmm_contig_B": probe(lambda x=x, w=gate_up_kn: torch.bmm(x, w), sync),
            "gate_up_bmm_strided_B": probe(lambda x=x, w=gate_up_nk: torch.bmm(x, w.transpose(1, 2)), sync),
            "down_bmm_contig_B": probe(lambda z=z, w=down_kn: torch.bmm(z, w), sync),
            "down_bmm_strided_B": probe(lambda z=z, w=down_nk: torch.bmm(z, w.transpose(1, 2)), sync),
        }
        try:
            torch.bmm(x.to(torch.bfloat16), gate_up_kn.to(torch.bfloat16))
            entry["bf16_supported"] = True
        except Exception as exc:  # noqa: BLE001
            entry["bf16_supported"] = f"no: {type(exc).__name__}: {exc}"[:160]
        out[name] = entry
    return out


def npu_fused_ops():
    """Enumerate torch_npu fused operators relevant to MoE ({} on a CUDA build)."""
    try:
        import torch
        names = list(dir(torch.ops.npu))
    except Exception:  # noqa: BLE001
        return {"available": False}
    keywords = ("moe", "grouped", "matmul", "topk", "routing", "expert", "bmm")
    return {
        "available": True,
        "total_ops": len(names),
        "matching": sorted({n for n in names if any(k in n.lower() for k in keywords)}),
    }


SLOW_MS = 1.0  # a small-tensor op above this almost certainly left the accelerator


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "npu", "cpu"))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", default=None, help="optional path for the raw JSON report")
    args = parser.parse_args(argv)

    import torch

    try:  # registers the NPU backend when torch_npu is installed
        import torch_npu  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    device = pick_device(args.device, torch)
    sync = make_sync(device, torch)
    report = {"torch": torch.__version__, "device": device}

    if device == "cuda":
        report["device_name"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))
    elif device == "npu":
        try:
            report["device_name"] = torch.npu.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            report["device_name"] = f"unknown ({exc})"
        try:
            report["torch_npu"] = torch_npu.__version__
        except Exception:  # noqa: BLE001
            pass

    print(f"torch={report['torch']}  device={device}  name={report.get('device_name')}")

    ops = {}
    for name, fn in op_probes(device, torch).items():
        ops[name] = probe(fn, sync, repeat=args.repeat)
    report["operators"] = ops
    print("\n== operator probes (median ms on small tensors) ==")
    print(f"{'op':<26}{'status':<10}{'ms':>10}")
    for name, res in ops.items():
        status = "ok" if res.get("ok") else "FAILED"
        value = f"{res['ms']:.4f}" if res.get("ms") is not None else "-"
        print(f"{name:<26}{status:<10}{value:>10}")
        if not res.get("ok"):
            print(f"{'':<26}{res['error']}")

    report["bmm"] = bmm_benchmarks(device, torch)
    print("\n== bmm at MoE shapes ==")
    for shape, entry in report["bmm"].items():
        print(f"{shape}: E={entry['num_experts']} T={entry['tokens_per_expert']} "
              f"bf16={entry['bf16_supported']}")
        for key in ("gate_up_bmm_contig_B", "gate_up_bmm_strided_B",
                    "down_bmm_contig_B", "down_bmm_strided_B"):
            res = entry[key]
            value = f"{res['ms']:.4f} ms" if res.get("ms") is not None else res.get("error", "?")
            print(f"    {key:<26}{'ok' if res.get('ok') else 'FAILED':<8}{value}")
        for proj in ("gate_up", "down"):
            contig = entry[f"{proj}_bmm_contig_B"].get("ms")
            strided = entry[f"{proj}_bmm_strided_B"].get("ms")
            if contig and strided:
                print(f"    -> {proj}: strided/contiguous B = {strided / contig:.2f}x "
                      "(store the bmm-ready layout if this is >> 1)")

    report["npu_fused_ops"] = npu_fused_ops()
    print("\n== accelerator fused-op candidates ==")
    if report["npu_fused_ops"].get("available"):
        print(f"torch.ops.npu total={report['npu_fused_ops']['total_ops']}")
        for name in report["npu_fused_ops"]["matching"]:
            print(f"    {name}")
    else:
        print("not a torch_npu build (no torch.ops.npu)")

    failed = [n for n, r in ops.items() if not r.get("ok")]
    # The GEMM entries are large by design; the bmm section reports them properly,
    # so exclude them from the small-tensor "suspect CPU fallback" heuristic.
    gemm_like = ("bmm", "matmul")
    slow = [
        n for n, r in ops.items()
        if r.get("ok") and (r.get("ms") or 0) > SLOW_MS and not n.startswith(gemm_like)
    ]
    print("\n== verdict ==")
    print(f"bmm: {'USABLE' if ops['bmm_contig_B'].get('ok') else 'UNUSABLE'}"
          f"{'' if ops.get('bmm_strided_B', {}).get('ok') else ' (strided-B variant failed)'}")
    print(f"missing ops: {failed or 'none'}")
    print(f"slow ops (>{SLOW_MS} ms on small tensors -> suspect CPU fallback): {slow or 'none'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
