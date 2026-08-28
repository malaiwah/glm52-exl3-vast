## Summary

The GLM-5.2 fused norm/RoPE kernel always launches a four-plane
`(4, num_tokens)` grid. On the 57 target layers that reuse a prior sparse index,
plane 0 and plane 3 immediately return because there is no indexer. This
schedules 350,208 guaranteed no-op Triton programs per 3,072-token target pass:

```text
57 layers * 2 unused planes * 3,072 rows = 350,208 programs
```

At C12 it is 1,368 no-op programs per target decode step, before small MTP reuse
overhead. This issue proposes a measured two-plane no-indexer specialization,
plus a separately gated removal of a redundant B12X top-K preclear.

Audited source: local vLLM
[`e2666d9a65f41fc376607531453cbd57c4c71016`](https://github.com/local-inference-lab/vllm/commit/e2666d9a65f41fc376607531453cbd57c4c71016).

- four-plane launch:
  [`kernels.py`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v32/nvidia/kernels.py#L474-L532)
- indexer plane early return:
  [`kernels.py`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v32/nvidia/kernels.py#L153-L167)
- top-K-clear plane early return:
  [`kernels.py`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v32/nvidia/kernels.py#L270-L273)

## Candidate A: compact no-indexer grid

Compile a `HAS_INDEXER=False` specialization with grid `(2, N)`, remapping its
two tasks to the current Q-normalization/RoPE and KV-normalization/RoPE planes.
Keep the four-plane route for full-indexer layers.

## Candidate B: backend-aware top-K preclear

On indexer layers, plane 3 clears the complete `N x 2048` top-K buffer. The
B12X tiled selector writes every output position, including invalid `-1`
sentinels, and the vLLM producer contract already describes B12X as filling the
buffer. Make the preclear conditional on the selected producer's overwrite
contract; retain it for all producers without a proven full-write guarantee.

Expected combined range is 0.5-3% PP and 0.2-2% TG. These are source-derived
estimates, not results; the promotion bar should be a repeatable 0.5% E2E gain
with exact parity and no additional memory.

## Qualification

1. Nsight Systems kernel/grid counts for one 3,072-token chunk and C1/C12.
2. Microbench `has_indexer=true/false`, N=`1,12,3072`:
   current four-plane grid, compact two-plane grid, and compact grid plus
   B12X no-preclear.
3. Require exact Q output, MLA/indexer-cache bytes, top-K indices and sentinel
   contents.
4. Cover zero valid KV, fewer than 2,048 candidates, padded rows, profiling
   mode, CUDA graph capture/replay, MTP3, and non-B12X producers.
5. Full-server 8K/64K/128K PP, C1/C8/C12 TG/MAL, mixed workload and long-context
   retrieval.

## Related but distinct work

- #207 tracks repeated MLA/DCP metadata and mapping work.
- #272 tracks discarded unfinished-prefill logits/proposals.
- #273 tracks mixed prefill/decode attention specialization.
- #274 tracks GLM projection fusion/selective overlap.

## AI assistance disclosure

The source audit, launch-count calculation, and test plan were prepared with
Codex assistance. The submitter remains responsible for the claims and any
resulting implementation.
