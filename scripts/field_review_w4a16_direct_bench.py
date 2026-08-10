#!/usr/bin/env python3
"""Matched native-ModelOpt W4A16 direct-path benchmark.

Run this file from outside the SparkInfer checkout with the exact candidate
checkout first on ``PYTHONPATH``.  It records immutable source/hardware
provenance, direct-path dispatch, CUDA-graph replay latency, allocation, and a
small output fingerprint for the aligned GLM-like shapes used by the field
review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

import torch

import sparkinfer
import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16_kernel
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    MoEMicroKernelW4A16SmallMDirect,
)
from tests._reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
)
from tests.moe.test_w4a16_e2e import _quantize_dense_moe_weight_storage


def _run_text(*args: str) -> str:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_revision() -> tuple[str, str, bool]:
    checkout = pathlib.Path(sparkinfer.__file__).resolve().parents[1]
    revision = _run_text("git", "-C", str(checkout), "rev-parse", "HEAD")
    dirty = bool(_run_text("git", "-C", str(checkout), "status", "--porcelain"))
    return str(checkout), revision, dirty


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _benchmark_shape(
    *,
    hidden_size: int,
    intermediate_size: int,
    m: int,
    warmup: int,
    replays: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed + hidden_size + m)
    device = torch.device("cuda")
    # Reset the peak allocator counter per shape: empty_cache() between records
    # does not clear it, so without this reset max_memory_allocated_bytes would
    # report the running max across all prior (larger) shapes, not this one.
    torch.cuda.reset_peak_memory_stats()
    experts = 1
    topk = 1
    rows = 2 * intermediate_size

    w13_dense = (
        torch.randn(experts, rows, hidden_size, device=device, dtype=torch.float32)
        * 0.12
    )
    w2_dense = (
        torch.randn(
            experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        * 0.12
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device=device)
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device=device)
    w13, w13_blockscale = _quantize_dense_moe_weight_storage(
        w13_dense,
        w13_global_scale,
    )
    w2, w2_blockscale = _quantize_dense_moe_weight_storage(
        w2_dense,
        w2_global_scale,
    )
    del w13_dense, w2_dense

    x = (torch.randn(m, hidden_size, device=device) * 0.5).to(torch.bfloat16)
    topk_ids = torch.zeros((m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.ones((m, topk), device=device, dtype=torch.float32)
    a_global_scale = torch.ones(experts, dtype=torch.float32, device=device)
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_global_scale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_global_scale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation="silu",
        quant_mode="w4a16",
    )
    output = torch.empty((m, hidden_size), dtype=x.dtype, device=device)
    binding = make_tp_moe_fp4_binding(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    )

    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat
    direct_launches = 0

    def spy_direct_launch(*args, **kwargs) -> None:
        nonlocal direct_launches
        direct_launches += 1
        real_direct_launch(*args, **kwargs)

    w4a16_kernel._w4a16_small_m_direct_launch_flat = spy_direct_launch
    try:
        for _ in range(warmup):
            binding.run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = binding.run()
        graph.replay()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        for index in range(replays):
            starts[index].record()
            graph.replay()
            ends[index].record()
        torch.cuda.synchronize()
        latencies = [
            start.elapsed_time(end)
            for start, end in zip(starts, ends, strict=True)
        ]
    finally:
        w4a16_kernel._w4a16_small_m_direct_launch_flat = real_direct_launch

    if direct_launches != warmup + 1:
        raise RuntimeError(
            "direct launch count mismatch: "
            f"expected {warmup + 1}, observed {direct_launches}"
        )
    if not bool(torch.isfinite(graph_output).all().item()):
        raise RuntimeError("non-finite direct-path output")
    output_bytes = graph_output.detach().cpu().contiguous().view(torch.uint8).numpy()
    output_sha256 = hashlib.sha256(output_bytes.tobytes()).hexdigest()
    geometry = MoEMicroKernelW4A16SmallMDirect(
        activation="silu",
        fast_math=True,
        share_input_across_experts=True,
        share_expert_scales=True,
        single_token=m == 1,
    )
    geometry.configure(
        m=m,
        k=hidden_size,
        n=intermediate_size,
        num_topk=topk,
        weight_E=experts,
        device=device,
    )
    if geometry._cfg is None:
        raise RuntimeError("direct-path geometry was not configured")
    cfg = geometry._cfg
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "m": m,
        "fc1_chunks": cfg.fc1_chunks,
        "i_chunk": cfg.i_chunk,
        "direct_python_launches": direct_launches,
        "warmup": warmup,
        "replays": replays,
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "memory_allocated_bytes": torch.cuda.memory_allocated(),
        "memory_reserved_bytes": torch.cuda.memory_reserved(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "output_sha256": output_sha256,
        "output_sum": float(graph_output.float().sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-sizes", default="2688,6144")
    parser.add_argument("--intermediate-size", type=int, default=1856)
    parser.add_argument("--m-values", default="1,3")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    torch.cuda.reset_peak_memory_stats()
    checkout, revision, dirty = _source_revision()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    records: list[dict[str, object]] = []
    for hidden_size in (int(value) for value in args.hidden_sizes.split(",")):
        for m in (int(value) for value in args.m_values.split(",")):
            records.append(
                _benchmark_shape(
                    hidden_size=hidden_size,
                    intermediate_size=args.intermediate_size,
                    m=m,
                    warmup=args.warmup,
                    replays=args.replays,
                    seed=args.seed,
                )
            )
            torch.cuda.empty_cache()

    result = {
        "timestamp_unix": time.time(),
        "source_checkout": checkout,
        "source_revision": revision,
        "source_dirty": dirty,
        "sparkinfer_module": str(pathlib.Path(sparkinfer.__file__).resolve()),
        "gpu_name": props.name,
        "gpu_sm_count": props.multi_processor_count,
        "gpu_total_memory_bytes": props.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_inventory": _run_text(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,power.limit",
            "--format=csv,noheader",
        ),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "records": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
