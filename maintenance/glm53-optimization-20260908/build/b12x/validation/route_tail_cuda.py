"""Bounded, model-free CUDA route-tail qualification; run only in the candidate image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from b12x.moe._shared.kernels.w4a16 import route_pack
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity


def check_source(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    target = "b12x/moe/_shared/kernels/w4a16/route_pack.py"
    entry = next(item for item in manifest["files"] if item["target"] == target)
    actual = hashlib.sha256(Path(route_pack.__file__).read_bytes()).hexdigest()
    if actual != entry["candidate_sha256"]:
        raise RuntimeError(f"installed {target} does not match candidate manifest: {actual}")


def make_ids(rows: int, topk: int, experts: int, mapped: bool):
    generator = torch.Generator().manual_seed(739 + rows)
    ids = torch.randint(0, experts, (rows, topk), generator=generator, dtype=torch.int64)
    flat = ids.view(-1)
    flat[::11] = 0
    flat[::37] = -1
    flat[::41] = experts + 3
    expert_map = None
    if mapped:
        expert_map = (torch.arange(experts, dtype=torch.int32) * 3 + 1) % experts
        expert_map[::7] = -1
        expert_map[::17] = experts + 2
    return ids, expert_map


def check_output(ids, expert_map, result, block_size, experts) -> dict:
    routes, blocks, count = (tensor.cpu().to(torch.int64) for tensor in result)
    flat = ids.view(-1)
    live = flat.numel()
    valid = (flat >= 0) & (flat < experts)
    resolved = flat.clone()
    if expert_map is not None:
        resolved[valid] = expert_map[flat[valid]].to(torch.int64)
        valid &= (resolved >= 0) & (resolved < experts)
    counts = torch.bincount(resolved[valid], minlength=experts)
    padded = ((counts + block_size - 1) // block_size) * block_size
    expected_count = int(padded.sum())
    expected_blocks = torch.repeat_interleave(torch.arange(experts), padded // block_size)
    assert int(count[0]) == expected_count
    assert torch.equal(blocks[:expected_blocks.numel()], expected_blocks)
    assert bool((blocks[expected_blocks.numel():] == -1).all())
    assert bool((routes[expected_count:] == live).all())
    assert bool(((routes >= 0) & (routes <= live)).all())
    payload = routes[routes < live]
    assert torch.equal(payload.sort().values, torch.nonzero(valid).view(-1))
    # Atomics intentionally do not promise a stable ordering within each expert.
    # Compare each expert's route multiset, not incidental launch-order output.
    start = 0
    for expert, width in enumerate(padded.tolist()):
        segment = routes[start:start + width]
        actual = segment[segment < live].sort().values
        expected = torch.nonzero(valid & (resolved == expert)).view(-1)
        assert torch.equal(actual, expected), (expert, actual, expected)
        start += width
    return {"live_routes": live, "packed_count": expected_count,
            "returned_routes": routes.numel(), "returned_blocks": blocks.numel()}


def run_family(device, max_rows, rows_to_check, topk, experts, block_size, mapped,
               graph=False, provide_counts=True, fixed_arena=True):
    _, route_slots, block_slots = route_pack_capacity(
        max_rows * topk, block_size, experts, topk=topk, bucket_tokens=False
    )
    # Guard suffixes catch writes beyond caller-owned slices, including nonaligned bounds.
    guard = 32
    poison = -912345
    routes_storage = torch.full((route_slots + guard,), poison, dtype=torch.int32, device=device)
    blocks_storage = torch.full((block_slots + guard,), poison, dtype=torch.int32, device=device)
    workspace = {
        "packed_route_indices": routes_storage[:route_slots],
        "block_expert_ids": blocks_storage[:block_slots],
        "packed_route_count": torch.empty(1, dtype=torch.int32, device=device),
        "expert_offsets": torch.empty(experts + 1, dtype=torch.int32, device=device),
    }
    if provide_counts:
        workspace["expert_counts"] = torch.empty(experts, dtype=torch.int32, device=device)
    for rows in rows_to_check:
        ids_cpu, map_cpu = make_ids(rows, topk, experts, mapped)
        ids = ids_cpu.to(device)
        expert_map = None if map_cpu is None else map_cpu.to(device)
        routes_storage.fill_(poison)
        blocks_storage.fill_(poison)
        result = route_pack.pack_topk_routes_by_expert(
            ids, block_size, experts, expert_map=expert_map, **workspace
        )
        torch.cuda.synchronize(device)
        if graph:
            if not provide_counts:
                # Eager routing uses atomic counts, so explicitly warm the old
                # graph-only prefix kernel before entering capture. This family
                # uses a full fixed arena (1536 rows, K16, E896, block 8).
                route_pack._pack_topk_routes_prefix_kernel[(1,)](
                    ids, expert_map if expert_map is not None else ids,
                    workspace["packed_route_count"], workspace["expert_offsets"],
                    ids.numel(), NUMEL_CAPACITY=max_rows * topk,
                    BLOCK_SIZE=block_size, NUM_EXPERTS=experts,
                    HAS_EXPERT_MAP=expert_map is not None,
                    BLOCK_E=1 << (experts - 1).bit_length(),
                    BLOCK_T=256, num_warps=8,
                )
                torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                # Without counts this executes the old captured-prefix fallback.
                result = route_pack.pack_topk_routes_by_expert(
                    ids, block_size, experts, expert_map=expert_map, **workspace
                )
            captured.replay()
            torch.cuda.synchronize(device)
            captured.reset()
        assert result[0].data_ptr() == workspace["packed_route_indices"].data_ptr()
        assert result[1].data_ptr() == workspace["block_expert_ids"].data_ptr()
        assert result[2].data_ptr() == workspace["packed_route_count"].data_ptr()
        if fixed_arena:
            assert result[0].numel() == route_slots
            assert result[1].numel() == block_slots
        assert bool((routes_storage[route_slots:] == poison).all().item())
        assert bool((blocks_storage[block_slots:] == poison).all().item())
        report = check_output(ids_cpu, map_cpu, result, block_size, experts)
        print(json.dumps({"case": "route_tail", "max_rows": max_rows, "rows": rows,
                          "topk": topk, "experts": experts, "mapped": mapped,
                          "graph": graph, "provide_counts": provide_counts, **report}), flush=True)


def check_undersized(device):
    rows, topk, experts, block_size = 17, 8, 32, 8
    _, slots, blocks = route_pack_capacity(rows * topk, block_size, experts,
                                           topk=topk, bucket_tokens=False)
    workspace = {
        "packed_route_indices": torch.empty(slots - 1, dtype=torch.int32, device=device),
        "block_expert_ids": torch.empty(blocks, dtype=torch.int32, device=device),
        "packed_route_count": torch.empty(1, dtype=torch.int32, device=device),
        "expert_offsets": torch.empty(experts + 1, dtype=torch.int32, device=device),
    }
    ids = torch.zeros((rows, topk), dtype=torch.int64, device=device)
    try:
        route_pack.pack_topk_routes_by_expert(ids, block_size, experts, **workspace)
    except ValueError:
        print(json.dumps({"case": "undersized_arena", "rejected": True}), flush=True)
    else:
        raise AssertionError("undersized workspace was accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "manifest.json")
    args = parser.parse_args()
    check_source(args.manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("an explicitly assigned CUDA device is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    # PR226's small-prefix and production-sized tail boundaries.
    for mapped in (False, True):
        run_family(device, 24, (17, 23, 24), 1, 32, 8, mapped, graph=True)
        run_family(device, 1536, (1177, 1184, 1536), 16, 896, 8, mapped)
        # GLM production geometry; these tails all select routed block size 64.
        run_family(device, 3072, (2049, 2303, 3071, 3072), 8, 256, 64, mapped)
    # PR256: another adequate fixed arena with different runtime/alignment bounds.
    run_family(device, 1535, (1177, 1184), 16, 896, 8, True)
    # Both allocation-based eager counting and preallocated captured fallback.
    run_family(device, 1536, (1177,), 16, 896, 8, True,
               graph=True, provide_counts=False)
    # Below-bucket and empty calls retain existing bucket selection/empty semantics.
    run_family(device, 24, (0, 1, 16), 1, 32, 8, False, fixed_arena=False)
    check_undersized(device)
    torch.cuda.synchronize(device)
    print(json.dumps({"status": "passed", "weights_loaded": False}), flush=True)


if __name__ == "__main__":
    main()
