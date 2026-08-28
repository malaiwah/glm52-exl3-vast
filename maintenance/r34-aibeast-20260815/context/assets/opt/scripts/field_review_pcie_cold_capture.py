#!/usr/bin/env python3
"""Focused first-use CUDA-graph gate for SparkInfer PCIe one-shot sources.

Run this with the candidate source on PYTHONPATH and a candidate-specific
TORCH_EXTENSIONS_DIR.  It deliberately performs no eager all-reduce before
capture, so host-side launch planning and occupancy queries execute for the
first time while CUDA stream capture is active.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sparkinfer.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=1 << 20,
    )
    try:
        stream = torch.cuda.Stream(device=device)
        inp = torch.empty(4096, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()

        # Do not call pool.all_reduce before this capture. The candidate's
        # residency cache must be cold at the first captured launch.
        with pool.capture(stream) as channel, torch.cuda.graph(graph, stream=stream):
            channel.all_reduce(inp, out=out)
        stream.synchronize()

        rank_sum = world_size * (world_size - 1) // 2
        for step in range(8):
            with torch.cuda.stream(stream):
                inp.fill_(float(step + rank))
                graph.replay()
            stream.synchronize()
            expected = torch.full_like(out, float(world_size * step + rank_sum))
            torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    finally:
        pool.close()
        dist.destroy_process_group()


def main() -> None:
    world_size = int(os.getenv("FIELD_REVIEW_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 6, 8, 10):
        raise SystemExit("FIELD_REVIEW_WORLD_SIZE must be one of 2,4,6,8,10")
    if torch.cuda.device_count() < world_size:
        raise SystemExit(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
    print(f"first-use CUDA capture: PASS ({world_size} ranks)")


if __name__ == "__main__":
    main()
