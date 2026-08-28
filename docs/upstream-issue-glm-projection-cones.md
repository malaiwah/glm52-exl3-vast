# [perf][research][GLM-5.2] Fuse or selectively overlap independent MLA Q/indexer projection cones

## Summary

The r31 GLM-5.2 / deepseek_v32 attention path has independent main-Q and
sparse-indexer projection cones, but currently executes them serially.

`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` does not affect GLM-5.2. Its sole
r31 consumer is DeepSeek-V4:

- knob: https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/envs.py#L2190-L2200
- DeepSeek-V4 consumer: https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/models/deepseek_v4/attention.py#L587-L645

This issue proposes a measurement-first GLM optimization. It does not claim a
speedup yet, and any implementation must remain opt-in until GPU-qualified.

## Exact GLM dependency graph

Current serial path:

https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/layers/mla.py#L268-L338

After `fused_qkv_a_proj(hidden_states)` and `q_a_layernorm(q_c)`, these branches
are independent:

1. Main attention:
   - `q_b_proj(q_c)`
   - main Q reshape/RoPE
   - normalized compressed KV / k_pe
2. Sparse indexer:
   - `indexer.wq_b(q_c)`
   - `indexer.wk_weights_proj(hidden_states)`
   - indexer Q/K RoPE and quantization
   - sparse top-k

The branches join only before sparse MLA consumes main Q plus selected indices.

Indexer source:

https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v2.py#L759-L966

`Indexer.forward()` already accepts precomputed `kw` and `q_raw`.

## Prior art

- vLLM DeepSeek-V4 pre-attention overlap:
  https://github.com/vllm-project/vllm/pull/41061
- large-batch regression and gating:
  https://github.com/vllm-project/vllm/pull/41443
- threshold sweep:
  https://github.com/vllm-project/vllm/pull/41526
- TensorRT-LLM outer indexer stream plus independent inner stream:
  https://github.com/NVIDIA/TensorRT-LLM/pull/14142

The DeepSeek-V4 gains must not be assumed for GLM. Upstream measurements
explicitly show that multi-stream GEMMs can regress at large M once the main
GEMM saturates the GPU.

## Candidate implementations, in order

### A. Fuse the skinny BF16 indexer WK/weights projection into fused-QKV-A

On full-indexer layers, extend `fused_qkv_a_proj(hidden_states)` from:

```text
[q_lora_rank, kv_lora_rank + rope_dim]
```

to:

```text
[q_lora_rank, kv_lora_rank + rope_dim, indexer_head_dim + index_n_heads]
```

Pass the third slice as `kw=` to `Indexer.forward()`.

Both projections are replicated BF16 linears reading the same hidden states:

- fused-A class:
  https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v2.py#L1123-L1167
- indexer WK/weights:
  https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v2.py#L791-L800

Expected advantages:

- remove one skinny BF16 GEMM launch on each full-indexer layer;
- avoid a second read of `hidden_states`;
- no duplicated persistent weight payload;
- approximately unchanged activation footprint because the separate `kw`
  output already exists.

Initial expected PP gain: 0.5-2%; this is an estimate, not a result.

### B. Opt-in whole-indexer overlap

After `q_c` normalization:

- default stream: main `q_b_proj`, KV normalization and main RoPE;
- dedicated auxiliary stream: whole indexer;
- join before MLA attention.

Do not silently make the existing threshold active for GLM. Use an explicit
default-off model-specific gate during qualification.

Initial expected M=3072 result: -5% to +2%. Large-M online-K6 projection grids
may already saturate all SMs.

### C. Do not initially fuse the two online-K6 Q weights

Each EXL3 logical matrix owns independent Hadamard vectors/codebook metadata:

https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/layers/quantization/exl3.py#L4-L13

Jointly encoding the main and indexer Q weights would change the quantization
contract and requires independent KLD/retrieval qualification. A grouped
two-weight kernel may be explored later, but it should not block A/B.

## Memory/lifetime bound

At production `M=3072`, each online-K6 `2048x4096` projection explicitly
allocates about:

- 12 MiB FP16 input
- 24 MiB output
- 24 MiB GEMM output
- 48 MiB FP32 temporary
- 12 MiB rotated input

Source:

https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/layers/quantization/exl3.py#L604-L690

Whole-branch overlap is expected to add roughly 96 MiB/rank of simultaneous
live storage; budget 96-128 MiB/rank after allocator/graph effects.

Required safeguards:

- dedicated non-aliased auxiliary stream;
- graph-private CUDA event generations;
- correct `record_stream` ownership;
- no concurrent sharing of K6/B12X scratch;
- include the overlapped shape in activation/KV profiling;
- verify no persistent graph-pool growth across capture shapes.

## Benchmark plan

Exact production-like target:

- 4x RTX PRO 6000 Blackwell, 280 W/GPU
- TP4/DCP4
- GLM-5.2 EXL3 3.42 bpw
- online K6
- NVFP4 MLA KV
- MTP3
- max batched tokens 3072
- max sequences 12
- CUDA graphs through the production capture envelope

### Projection microbench

Sweep M:

```text
1, 4, 12, 32, 128, 256, 512, 1024, 1536, 2048, 3072
```

Compare:

1. serial baseline;
2. fused-A + indexer-WK candidate;
3. whole-indexer overlap;
4. fused-A + selective overlap.

Record per-branch and join critical-path CUDA time, SM active/eligible warps,
tensor-core and DRAM utilization, achieved occupancy, power-cap behavior,
max allocated/reserved VRAM and graph-pool delta.

### End-to-end

- uncached PP: 8K, 64K, 128K, repeated warm samples;
- decode: C1/C4/C8/C12 at 0 and 128K context;
- mixed workload: one long prefill while C1/C4 decode is active;
- MTP mean acceptance length and accepted throughput;
- physical VRAM and logical KV capacity.

Promotion gate:

- at least 2% repeatable 64K/128K PP gain;
- no more than 1% decode/TG regression;
- no MAL regression;
- no material KV-capacity loss or less than 512 MiB physical safety margin;
- no graph/eager/compile-cache instability.

Correctness requires serial/candidate tensor parity, logits/KLD, cold retrieval
through maximum context, degeneration/tool/structured-output smoke, and a C12
soak with no OOM, illegal access, event or workspace race.

## Duplicate search

No existing local-inference-lab/vllm issue tracks this GLM projection-cone
fusion/overlap.

Nearby but distinct:

- #207: DCP/indexer metadata hot path
- #203: EXL3 memory lifecycle
- #157: absorbed MLA projections

## AI assistance disclosure

This issue's source audit, dependency graph, estimates, and draft benchmark
plan were prepared with Codex assistance. The submitter remains responsible
for the technical claims and any resulting implementation.
