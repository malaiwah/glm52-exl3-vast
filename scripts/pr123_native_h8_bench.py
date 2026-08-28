#!/usr/bin/env python3
"""Matched graph-replay benchmark for PR #123's production GLM native-H8 path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import time

import torch

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)
from sparkinfer.attention._shared.mla.kernel import run_unified_decode
from sparkinfer.attention._shared.mla.traits import ScaleFormat
from sparkinfer.attention.sparse_mla._scratch import (
    SPARKINFERSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)
from tests.attention.test_attention_mla_kv_cache import (
    _HEAD_DIM,
    _PAGE_SIZE,
    _V_HEAD_DIM,
    _assert_reader_matches_dequantized_records,
    _make_written_reader_case,
    _reference_attention_from_records,
)


def summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "p10_us": ordered[max(0, round(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, round(0.90 * (len(ordered) - 1)))],
    }


def git_value(root: pathlib.Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        key = "QUAL_GIT_TREE" if "HEAD^{tree}" in args else "QUAL_GIT_COMMIT"
        if "status" in args:
            return os.environ.get("QUAL_GIT_STATUS", "unavailable")
        return os.environ.get(key, "unavailable")


@torch.inference_mode()
def bench_shape(
    *,
    rows: int,
    topk: int,
    valid_topk: int | None,
    warmup: int,
    replays: int,
    seed: int,
    expect_native: bool,
) -> dict[str, object]:
    import sparkinfer.attention._shared.mla.kernel as launch

    device = torch.device("cuda")
    heads = 8
    q, cache, indices = _make_written_reader_case(
        rows=rows,
        topk=topk,
        seed=seed,
        device=device,
        per_token_scale=True,
        heads=heads,
    )
    live_topk = topk if valid_topk is None else valid_topk
    if not 0 <= live_topk <= topk:
        raise ValueError(f"valid_topk must be in [0, {topk}], got {live_topk}")
    lengths = torch.full((rows,), live_topk, dtype=torch.int32, device=device)
    if live_topk < topk:
        indices[:, live_topk:].fill_(-1)
    sm_scale = _HEAD_DIM**-0.5
    max_chunks = max(8, (topk + 63) // 64)
    caps = SPARKINFERSparseMLAScratchCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=heads,
        max_q_rows=rows,
        max_batch=rows,
        max_width=topk,
        max_kv_rows=topk,
        head_dim=_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        max_chunks_per_row=max_chunks,
        page_size=_PAGE_SIZE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.zeros(
        scratch_spec.shape, dtype=scratch_spec.dtype, device=scratch_spec.device
    )
    binding = plan.bind(
        scratch=scratch,
        q=q,
        selected_indices=indices,
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )
    output = torch.empty(
        (rows, heads, _V_HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    def run() -> None:
        run_unified_decode(
            q_all=q,
            swa_k_cache=cache,
            swa_indices=indices,
            swa_topk_lengths=lengths,
            workspace=binding.scratch,
            sm_scale=sm_scale,
            swa_page_size=_PAGE_SIZE,
            scale_format_override=ScaleFormat.NVFP4_E4M3,
            fp8_rope_override=True,
            latent_scale_per_token=True,
            out=output,
        )

    torch.cuda.reset_peak_memory_stats(device)
    run()
    torch.cuda.synchronize(device)
    expected = _reference_attention_from_records(
        q=q,
        cache=cache,
        indices=indices,
        lengths=lengths,
        sm_scale=sm_scale,
        per_token_scale=True,
    )
    _assert_reader_matches_dequantized_records(output, expected)
    eager_plan = dict(launch.LAST_DECODE_PLAN)
    if bool(eager_plan.get("native_glm_h8")) is not expect_native:
        raise AssertionError(
            f"native GLM H8 route mismatch: expected={expect_native}, "
            f"plan={eager_plan}"
        )

    graph = capture_cuda_graph(run, warmup=warmup)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_reader_matches_dequantized_records(output, expected)

    hot = bench_cuda_graph(graph, replays=replays)["replay_us"]
    flush = make_l2_flush_fn(True)
    cold = bench_cuda_graph(graph, replays=replays, l2_flush=flush)["replay_us"]
    result = {
        "rows": rows,
        "heads": heads,
        "topk": topk,
        "seed": seed,
        "plan": eager_plan,
        "correctness": {
            "graph_replay": True,
            "max_abs": float((output.float() - expected).abs().max().item()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    output.float().reshape(-1), expected.reshape(-1), dim=0
                ).item()
            ),
        },
        "hot": {"summary": summary(hot), "raw_us": hot},
        "cold_l2": {"summary": summary(cold), "raw_us": cold},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del graph
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rows", default="1,4,8")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--valid-topk", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123123)
    parser.add_argument("--expect-native", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.source_root).resolve()
    started = time.time()
    document: dict[str, object] = {
        "schema": 1,
        "label": args.label,
        "source_root": str(root),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "git_status": git_value(root, "status", "--short"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(),
        "settings": {
            "rows": [int(value) for value in args.rows.split(",")],
            "topk": args.topk,
            "warmup": args.warmup,
            "replays": args.replays,
        },
        "results": [],
    }
    for index, rows in enumerate(document["settings"]["rows"]):
        document["results"].append(
            bench_shape(
                rows=rows,
                topk=args.topk,
                valid_topk=args.valid_topk,
                warmup=args.warmup,
                replays=args.replays,
                seed=args.seed + index,
                expect_native=args.expect_native,
            )
        )
    document["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
    document["duration_seconds"] = time.time() - started
    document["ok"] = True
    pathlib.Path(args.out).write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
