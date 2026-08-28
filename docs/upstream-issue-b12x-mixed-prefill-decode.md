# [perf][B12X MLA] Preserve specialized decode and prefill attention in mixed batches

## Summary

The B12X sparse-MLA metadata builder already identifies the contiguous decode
and prefill regions in a reordered vLLM batch, but the attention forward path
chooses one kernel for the entire batch. If any long prefill row is present,
all decode rows are routed through the unified extend/prefill kernel.

This leaves a concrete mixed-workload opportunity: retain hybrid batching for
the shared linear/MoE work, but split only sparse attention into its specialized
decode and prefill paths. Establish a memory-neutral sequential split first;
then test concurrent, priority-aware execution only if profiling shows
complementary resource use and a safe scratch layout.

Reviewed source:
`local-inference-lab/vllm@e2666d9a65f41fc376607531453cbd57c4c71016`

Motivating production shape:

- GLM-5.2 EXL3-TR3, online K6
- TP4/DCP4 over PCIe
- dynamic NVFP4 MLA KV + FP8 RoPE
- MTP3 probabilistic
- chunked prefill budget 3,072
- up to 12 requests
- long-running agent traffic that mixes large cold prefills with active decodes

This is a static source finding. No end-to-end gain is claimed before a matched
GPU A/B.

## Source evidence

`vllm/v1/attention/backends/mla/b12x_mla_sparse.py:973-994` computes:

```python
num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
    cm.batch_topology.split_decodes_and_prefills(...)
)
```

vLLM has already reordered the batch as:

```text
decode -> short_extend -> long_extend -> prefill
```

so the two token regions are contiguous.

However, `b12x_mla_sparse.py:2766-2775` selects the decode kernel only when the
whole batch is decode/spec-decode shaped:

```python
use_decode_kernel = attn_metadata.max_query_len <= 1 or (...)
```

Otherwise `b12x_mla_sparse.py:2834-2886` sends `q_all`, all selected indices
and the entire output through one `_sparse_mla_extend_forward` call. A single
long prefill row therefore changes the attention implementation used by every
concurrent decode row.

The same mixed-batch condition also prevents the transient full-CKV prefill
gather: `b12x_mla_sparse.py:1026-1028` and `2141-2153` require
`num_decode_tokens == 0`. A split design could evaluate gather eligibility for
the prefill slice independently while decode continues on its sharded path.

## Why this may help

Prefill and decode have complementary shapes. Decode uses small query batches
and a split/merge kernel; prefill uses the large-M single-pass extend kernel.
Routing both through extend preserves correctness, but gives up the specialized
decode route and couples decode ITL to each prefill chunk.

POD-Attention reports up to 59% and a 28% mean improvement in attention-kernel
time from explicit prefill/decode overlap on other architectures. That is
useful prior art, not a transferable GLM/vLLM number. Whole-server benefit is
bounded by the measured sparse-attention fraction and by power/SM contention.
An honest initial expectation is 0-5% aggregate throughput on mixed loads, with
potentially larger p95/p99 decode-ITL benefit. Pure-prefill PP and pure-decode
TG should remain unchanged.

## Proposed stages

### 1. Sequential split baseline

- Slice the already-contiguous decode and prefill token regions.
- Bind `_decode_plan` to the decode views and `_extend_plan` to the prefill
  views.
- Execute decode first, then prefill, and write directly into corresponding
  views of the existing output tensor.
- Preserve the current single-path fast cases for pure decode and pure prefill.

This establishes whether specialization alone helps and should require no new
persistent GPU allocation.

### 2. Independently qualify CKV gather for the prefill slice

Build sliced request/token metadata so the prefill region can use the existing
full-CKV gather policy while decode remains sharded. Charge every additional
metadata/workspace byte explicitly; do not duplicate the KV cache.

### 3. Optional concurrent split

Only after the sequential path wins:

- launch decode on a high-priority non-default stream and prefill on a normal
  stream;
- join with one event before the dependent output projection;
- constrain the prefill grid if both persistent kernels otherwise occupy all
  SMs;
- prove that decode and extend scratch/output regions do not alias.

CUDA stream priority is a launch-order hint, not preemption. A long running
prefill CTA cannot be interrupted, so kernel granularity and grid sizing matter.
If concurrency requires another full scratch arena, it is unlikely to fit this
memory-tight 520K profile and should be rejected in favor of the sequential
split.

Do not split the model-wide GEMM/MoE batch: those operations benefit from
processing prefill and decode rows together and sharing weight traffic. This
proposal is attention-only.

## Instrumentation and test plan

Add opt-in NVTX ranges for:

- `b12x_mla.decode`
- `b12x_mla.extend`
- `b12x_mla.ckv_gather`
- `b12x_mla.join`

Use Nsight Systems to report per-range GPU time, queue time, overlap, and kernel
launch count. Use DCGM/Nsight Compute for Tensor Active, DRAM Active, achieved
occupancy, barrier stalls and minimum-rank clocks.

Matched A/B/A cells:

1. 1/4/8 active decoders plus one cold 64K prefill.
2. The same at 128K and near 500K.
3. Pure decode C1/C4/C8/C12 and pure prefill 8K/64K/128K controls.
4. Sequential split; then concurrent split at several prefill grid caps.

Report aggregate PP/TG, per-request TG, TTFT, ITL p50/p95/p99, MAL,
preemptions, attention time, PCIe bytes, peak activation and minimum physical
free VRAM.

Correctness gates:

- exact output parity for the sequential split;
- MTP3 acceptance and final-prefill transition;
- DCP rank parity and no collective-order divergence;
- structured outputs and tool calls;
- KLD, degeneration and two-seed five-depth maximum-context needles;
- no late JIT, OOM, Xid, restart or cache-connector failure.

## Related work

- POD-Attention: https://arxiv.org/abs/2410.18038
- vLLM's generic batch reorder/split contract:
  `vllm/v1/attention/backends/utils.py:566-720`
