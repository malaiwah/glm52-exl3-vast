## What changed

- Add a backend-owned `warmup_mixed_trellis_route_pack()` entry point.
- Enumerate one live row count for every power-of-two route-capacity specialization reachable by the compiled mixed-Trellis launch.
- Reuse the launch's exact expert geometry, route block, route-ID dtype, expert map, and caller-owned buffers.
- Synchronize before recording a signature as warm, so a failed warmup remains retryable.

## Why

Mixed EXL3 Trellis calls `pack_topk_routes_by_expert()` outside the ordinary `B12xExperts` warmup path. A maximum-batch profile therefore does not materialize smaller decode, speculative, or final-prefill-chunk Triton specializations. On a tightly packed 4x RTX PRO 6000 production service, one such specialization first loaded under live traffic and failed in `_pack_topk_routes_post_prefix_kernel` with a CUDA module-load OOM despite low active KV usage.

This API lets the serving integration load the finite set of reachable modules before final memory profiling and KV-cache sizing.

## Scope and duplicate check

This is not a duplicate of b12x issue #98: that issue covers first-use CuTe compilation during graph capture in the ordinary W4A16 fused-MoE path. This change covers Triton route-pack module residency for the separate runtime-dynamic mixed-Trellis path.

The companion [local-inference-lab/vLLM PR #250](https://github.com/local-inference-lab/vllm/pull/250) calls this API from the existing pre-KV kernel-warmup phase.

## Validation

- `ruff check` and `ruff format --check` on all changed files
- `py_compile` on changed source/tests
- Focused CPU/static test in the exact r28 runtime image: **5 passed**
- Production AIBeast service remained online and healthy; no GPU tests were run there

Full multi-rank GPU startup, memory-delta, first-request, concurrency, and long-prefill qualification is intentionally pending on a separate test host. This PR is draft until that evidence is attached.

AI assistance was used. The submitter reviewed the complete diff, reproduced the failure path, and ran the checks above.
