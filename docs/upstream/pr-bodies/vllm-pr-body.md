## What changed

- Retain the immutable mixed-Trellis runtime created by the first eager profile pass.
- Delegate route-pack specialization enumeration and materialization to the matching b12x backend.
- Invoke that warmup from vLLM's existing kernel-warmup phase, before the second profile pass and KV-cache sizing.
- Fail closed when a selected mixed-EXL3 layer was not planned or the matching backend API is absent.
- Log the warmed specialization count and observed device free-memory delta.

## Why

The existing `warmup_b12x_moe_dynamic()` recognizes ordinary `B12xExperts`, but EXL3 mixed Trellis uses its own quant method and reaches b12x route packing directly. Profiling the maximum batch creates the runtime but leaves smaller power-of-two route-capacity Triton modules lazy.

An r28 GLM-5.2 EXL3 production process later loaded `_pack_topk_routes_post_prefix_kernel` during live traffic and failed inside Triton's CUDA binary loader with OOM. Active KV usage was low; the KV pool had already consumed the remaining device headroom. Running the backend warmup between the first and second profile passes makes persistent module residency visible before KV blocks are assigned and turns insufficient headroom into a startup failure rather than a serving crash.

## Dependency and base

- Stacked on local-inference-lab/vLLM PR #228, which owns the EXL3 mixed-Trellis integration.
- Depends on [local-inference-lab/b12x PR #126](https://github.com/local-inference-lab/b12x/pull/126).

## Scope and duplicate check

This does not duplicate local-inference-lab/vLLM PR #248, which prewarms the CuTe PCIe one-shot collective, or vllm-project/vllm PR #41481, which warms speculative-decoding helper kernels. It is the missing EXL3 mixed-Trellis route-pack provider hook. It is compatible with the broader vllm-project/vllm JIT warmup RFC #47456.

## Validation

- `ruff check` and `ruff format --check` on all changed files
- `py_compile` on changed source/tests
- Exact r28 runtime image, CPU/static only: **17 passed** across the new tests and existing `test_exl3_prefill_plan.py`
- Production AIBeast service remained online and healthy; no GPU tests were run there

Full TP4/DCP4 GPU startup, measured module-memory/KV delta, first-request, MTP/concurrency, and long-prefill qualification is intentionally pending on a separate test host. This PR is draft until that evidence is attached.

AI assistance was used. The submitter reviewed the complete diff, reproduced the failure path, and ran the checks above.
