# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm GLM5 K-pool prefill kernels before serving requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


@torch.inference_mode()
def warmup_glm5_kpool_prefill(worker: "Worker") -> int:
    """Compile K-pool cache-write and tail-seed variants for finalized caches."""
    if not current_platform.is_cuda():
        return 0
    config = worker.model_config.hf_text_config
    pool_size = int(getattr(config, "index_kpool", 1))
    if pool_size <= 1 or not bool(getattr(config, "index_kpool_compress", False)):
        return 0

    from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
        SparseAttnIndexerKpool,
    )
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_compress_and_write_cache,
        kpool_seed_tail_cache,
    )


    max_tokens = max(int(worker.scheduler_config.max_num_batched_tokens), pool_size)
    warmed_variants = 0
    seen_signatures: set[tuple[object, ...]] = set()
    for module in worker.get_model().modules():
        if not isinstance(module, SparseAttnIndexerKpool):
            continue

        kv_cache = module.k_cache.kv_cache
        tail_cache = module.tail_cache
        tail_kv_cache = None if tail_cache is None else tail_cache.kv_cache
        if kv_cache.numel() == 0 or tail_kv_cache is None or tail_kv_cache.numel() == 0:
            continue

        head_dim = int(module.head_dim)
        page_size = int(kv_cache.shape[1])
        round_scale = bool(module.scale_fmt)
        cache_width = int(kv_cache.shape[2])
        signature = (
            kv_cache.device,
            page_size,
            cache_width,
            pool_size,
            head_dim,
            round_scale,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        device = kv_cache.device
        dummy_cache = torch.empty(
            (1, page_size, cache_width),
            dtype=torch.uint8,
            device=device,
        )
        slot_k = torch.zeros(
            (1, pool_size, head_dim), dtype=torch.bfloat16, device=device
        )
        slot_score = torch.zeros_like(slot_k)
        ape = torch.zeros((pool_size, head_dim), dtype=torch.float32, device=device)
        loc_storage = torch.zeros((2,), dtype=torch.int64, device=device)
        write_mask = torch.ones((1,), dtype=torch.bool, device=device)
        for loc in (loc_storage[:1], loc_storage[1:]):
            kpool_compress_and_write_cache(
                dummy_cache,
                slot_k,
                slot_score,
                ape,
                loc,
                pool_size,
                head_dim=head_dim,
                write_mask=write_mask,
                round_scale=round_scale,
                return_compressed=False,
                write_cache=True,
            )
            warmed_variants += 1

        num_tail_blocks = (max_tokens + pool_size - 1) // pool_size
        dummy_tail = torch.zeros(
            (num_tail_blocks, 2, pool_size, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        key = torch.zeros((max_tokens, head_dim), dtype=torch.bfloat16, device=device)
        gate_score = torch.zeros_like(key)
        aligned_tail_slots = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        unaligned_tail_storage = torch.empty(
            (max_tokens + 1,), dtype=torch.int64, device=device
        )
        unaligned_tail_slots = unaligned_tail_storage[1:]
        unaligned_tail_slots.copy_(aligned_tail_slots)
        for tail_slots in (aligned_tail_slots, unaligned_tail_slots):
            kpool_seed_tail_cache(
                dummy_tail,
                key,
                gate_score,
                tail_slots,
                pool_size,
                head_dim=head_dim,
            )
            warmed_variants += 1

    if warmed_variants:
        torch.accelerator.synchronize()
    return warmed_variants
