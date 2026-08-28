## What changed

- Retain the immutable mixed-Trellis runtime created by the first eager profile pass.
- Delegate route-pack specialization enumeration and materialization to the matching b12x backend.
- Invoke that warmup from vLLM's existing kernel-warmup phase, before the second profile pass and KV-cache sizing.
- Fail closed when a selected mixed-EXL3 layer was not planned or the matching backend API is absent.
- Log the warmed specialization count and observed device free-memory delta.

## Why

The existing `warmup_b12x_moe_dynamic()` recognizes ordinary `B12xExperts`, but EXL3 mixed Trellis uses its own quant method and reaches b12x route packing directly. Profiling the maximum batch creates the runtime but leaves smaller route-capacity and scalar-alignment Triton specializations lazy.

An r28 GLM-5.2 EXL3 production process later loaded `_pack_topk_routes_post_prefix_kernel` during live traffic and failed inside Triton's CUDA binary loader with OOM. Active KV usage was low; the KV pool had already consumed the remaining device headroom. Running the backend warmup between the first and second profile passes makes persistent module residency visible before KV blocks are assigned and turns insufficient headroom into a startup failure rather than a serving crash.

## Dependency and base

- Stacked on local-inference-lab/vLLM PR #228, which owns the EXL3 mixed-Trellis integration.
- Depends on [local-inference-lab/b12x PR #126](https://github.com/local-inference-lab/b12x/pull/126).

## Scope and duplicate check

This does not duplicate local-inference-lab/vLLM PR #248, which prewarms the CuTe PCIe one-shot collective, or vllm-project/vllm PR #41481, which warms speculative-decoding helper kernels. It is the missing EXL3 mixed-Trellis route-pack provider hook. It is compatible with the broader vllm-project/vllm JIT warmup RFC #47456.

The companion b12x PR also prewarms rank-sliced native-MTP draft plans from their backend-owned plan warmup. That draft runtime is not reachable from this vLLM target-model hook, so the two changes intentionally close the target and draft sides at their respective owners.

## Validation

- `ruff check` and `ruff format --check` on all changed files
- `py_compile` on changed source/tests
- Exact r28 runtime image: **17 vLLM tests passed** plus **6 focused b12x tests passed**
- AIBoss RTX 5090 GPU qualification using GGv20r28 plus only vLLM #250 and b12x #126:
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
