"""Model-free two-GPU PCIe graph/rollback/close check, adapted from PR257."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x
from b12x.comm.pcie import pcie_oneshot


def check_source(manifest_path: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    target = "b12x/comm/pcie/pcie_oneshot.py"
    entry = next(item for item in manifest["files"] if item["target"] == target)
    actual = hashlib.sha256(Path(pcie_oneshot.__file__).read_bytes()).hexdigest()
    if actual != entry["candidate_sha256"]:
        raise RuntimeError(f"installed {target} does not match candidate manifest: {actual}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def check_reduction(output: torch.Tensor, expected: float) -> None:
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)


def worker(rank: int, port: int, manifest_path: str) -> None:
    check_source(manifest_path)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    other_device = 1 - rank
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    group = dist.group.WORLD
    pool = None
    graph = None
    try:
        for cycle in range(2):
            torch.cuda.set_device(rank)
            pool = pcie_oneshot.PCIeOneshotAllReducePool.from_process_group(
                process_group=group, device=device, max_input_bytes=1 << 16
            )
            pool.prepare_channels(("eager:retained",))
            inp = torch.full((4, 1024), rank + 1, device=device, dtype=torch.bfloat16)
            output = torch.empty_like(inp)
            retained = pool.for_stream(channel_id="eager:retained")
            pool.all_reduce(inp, out=output, channel_id="eager:retained")
            torch.cuda.synchronize(device)
            check_reduction(output, 3)
            checkpoint = pool.checkpoint_channels()
            pool.prepare_channels(("graph:transient",))
            transient = pool._logical_channels["graph:transient"]
            assert transient._owned_buffers and retained._owned_buffers

            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), pool.capture(
                stream=stream, channel_id="graph:transient"
            ):
                for _ in range(2):
                    pool.all_reduce(inp, out=output)
                stream.synchronize()
                dist.barrier(group=group, device_ids=[rank])
                b12x.freeze_kernel_resolution("isolated PCIe teardown preparation")
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        pool.all_reduce(inp, out=output)
                finally:
                    b12x.unfreeze_kernel_resolution()
            torch.cuda.synchronize(device)
            for step in range(3):
                inp.fill_(rank + 1 + step)
                output.fill_(float("nan"))
                dist.barrier(group=group, device_ids=[rank])
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                allocated_bytes = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
                assert torch.cuda.memory_allocated(device) == allocated_bytes
                check_reduction(output, 3 + 2 * step)
            graph.reset()
            graph = None
            torch.cuda.synchronize(device)

            barriers = []
            real_barrier = dist.barrier

            def checked_barrier(*args, **kwargs):
                # Forward the real collective, but fail before an unbound barrier
                # can choose the wrong GPU and hang an unpatched candidate.
                if kwargs.get("device_ids") != [rank]:
                    raise AssertionError(f"teardown barrier did not bind owner {rank}: {kwargs}")
                barriers.append(kwargs["device_ids"])
                return real_barrier(*args, **kwargs)

            torch.cuda.set_device(other_device)
            with patch.object(dist, "barrier", checked_barrier):
                pool.rollback_channels(checkpoint)
            assert torch.cuda.current_device() == other_device
            assert barriers == [[rank]] * 3
            assert transient._ipc_imports_closed and transient._ipc_exports_freed
            assert not transient._owned_buffers
            assert not retained._ipc_imports_closed and retained._owned_buffers
            assert pool.checkpoint_channels() == checkpoint

            torch.cuda.set_device(rank)
            inp.fill_(rank + 1)
            pool.all_reduce(inp, out=output, channel_id="eager:retained")
            torch.cuda.synchronize(device)
            check_reduction(output, 3)
            barriers.clear()
            torch.cuda.set_device(other_device)
            with patch.object(dist, "barrier", checked_barrier):
                pool.close()
                pool.close()
            assert torch.cuda.current_device() == other_device
            assert barriers == [[rank]] * 3
            assert retained._ipc_imports_closed and retained._ipc_exports_freed
            assert not retained._owned_buffers and not pool._all_channels
            assert pool._closed
            print(json.dumps({"rank": rank, "cycle": cycle, "pool_device": str(device),
                              "wrong_current_device": other_device,
                              "graph_replays": 3, "rollback_retained_reuse_close": "passed"}),
                  flush=True)
            pool = None
            del retained, transient, inp, output, stream
    finally:
        torch.cuda.set_device(rank)
        if graph is not None:
            graph.reset()
            torch.cuda.synchronize(device)
        if pool is not None and not pool._closed:
            pool.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "manifest.json")
    args = parser.parse_args()
    check_source(str(args.manifest))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("assign exactly two CUDA devices with CUDA_VISIBLE_DEVICES")
    context = mp.spawn(worker, args=(free_port(), str(args.manifest.resolve())),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("PCIe teardown workers exceeded the 240-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    print(json.dumps({"status": "passed", "weights_loaded": False}), flush=True)


if __name__ == "__main__":
    main()
