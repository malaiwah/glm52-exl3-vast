## What changed

- Add a backend-owned `warmup_mixed_trellis_route_pack()` entry point.
- Enumerate both Triton scalar alignment classes reachable inside every power-of-two route-capacity bucket: the bucket maximum and its first live member.
- Reuse the launch's exact expert geometry, route block, route-ID dtype, expert map, and caller-owned buffers.
- Prewarm the rank-sliced full-rotation W4A16 plans used by native MTP drafts, for both `int32` and `int64` route IDs and mapped/unmapped routes.
- Include the route-ID dtype in the prewarm signature so incompatible Triton specializations cannot be treated as equivalent.
- Synchronize before recording a signature as warm, so a failed warmup remains retryable.

## Why

Mixed EXL3 Trellis calls `pack_topk_routes_by_expert()` outside the ordinary `B12xExperts` warmup path. A maximum-batch profile therefore does not materialize smaller decode, speculative, or final-prefill-chunk Triton specializations. Triton also specializes the runtime `live_numel` scalar by alignment/divisibility class, so warming only a bucket maximum does not cover the first non-aligned live size in that bucket.

On a tightly packed 4x RTX PRO 6000 production service, one such specialization first loaded under live traffic and failed in `_pack_topk_routes_post_prefix_kernel` with a CUDA module-load OOM despite low active KV usage.

This API lets the serving integration load the finite set of reachable modules before final memory profiling and KV-cache sizing. Rank-sliced draft plans call the same helper during their existing plan warmup, closing the equivalent native-MTP allocation gap without requiring vLLM to traverse a separate draft model.

## Scope and duplicate check

This is not a duplicate of b12x issue #98: that issue covers first-use CuTe compilation during graph capture in the ordinary W4A16 fused-MoE path. This change covers Triton route-pack module residency for runtime-dynamic mixed-Trellis targets and rank-sliced native-MTP draft plans.

The companion [local-inference-lab/vLLM PR #250](https://github.com/local-inference-lab/vllm/pull/250) calls this API from the existing pre-KV kernel-warmup phase.

## Validation

- `ruff check` on all changed files; focused format checks on the helper, planner, and test files
- `py_compile` on changed source/tests
- Focused CPU/static tests in the exact r28 runtime image: **6 passed**
- AIBoss RTX 5090 GPU qualification using GGv20r28 plus only b12x #126 and vLLM #250:
  - 32 mixed-Trellis variants materialized before KV sizing for the 2K profile; measured residency **2.0 MiB**
  - logical KV changed **2,926,208 -> 2,925,952** (-256 tokens, -0.009%)
  - route-pack inference JIT changed from `small_prefix + sort` on control to **none** over a 1/2/5/8/9-token battery
  - identical generated output and 35.1 vs 36.0 tok/s CC1 (noise-level parity)
  - reverse-order warm-cache prefill A/B stayed within **-1.22% to +1.44%** at 256/1K/2K/4K/6K rows
  - 32K long-context gate passed with no route-pack inference JIT and successful planted-name retrieval
  - native MTP1 draft: post-start route-pack JIT changed from `small_prefix + sort` to **none**
  - MTP draft residency was exactly **2.0 MiB**: logical FP8-KV capacity changed **1,865,920 -> 1,865,728** (-192 tokens, -0.010%)
  - native-MTP output and acceptance were unchanged: **451/571** accepted (79.0%), MAL **1.790**, `FRUIT-MTP-OK`
  - the exact final head restarted from a populated compilation cache in **6.80 s** with no route-pack inference JIT
  - read-only AIBeast production audit corroborated the incident: an MTP3 TP4/DCP4 r28 worker failed in Triton's `load_binary()` for `_pack_topk_routes_post_prefix_kernel` with `CUDA: out of memory` at only **1.18% active KV usage**; the restarted engine later lazily loaded `post_prefix` and `sort` again

Exact hardware, image/model revisions, and measurements are posted in the GPU evidence comment below.

Production AIBeast remained online and untouched. Full TP4/DCP4 and CUDA-graph qualification remains pending, so this PR stays draft.

AI assistance was used. The submitter reviewed the complete diff, reproduced both the original and scalar-alignment misses, and ran the checks above.
