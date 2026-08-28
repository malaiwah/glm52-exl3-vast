# GLM-5.2 prefill optimization research — 2026-08-09

## Outcome

The current r31 GLM-5.2 profile is already close to the **configuration**
Pareto frontier for four 96 GiB RTX PRO 6000 Blackwell cards. The large
improvements that were available from choosing the right DCP policy, query
split, indexer sharding, CKV gather, EXL3 prefill layout, lossless PCIe
collectives, dynamic NVFP4 KV, and online K6 have already been integrated.

There is no credible untried environment-variable combination that promises a
safe 20–40% cold-prefill gain while retaining all of:

- a 520,192-token request envelope;
- MTP3 decode at C1 through C12;
- exact/lossless PCIe transport;
- long-context needle and degeneration quality;
- less than 96 GiB per rank.

The next gains split into three classes:

1. **Immediate controlled A/Bs:** restore the previously proven 6 MiB B12X
   PCIe-DMA crossover, and tune memory clocks against the unusually low 280 W
   card power cap.
2. **Small, low-risk code improvements:** skip output projection/sampling for
   unfinished prefill chunks, hoist repeated MLA metadata work, and test
   targeted overlap of independent GLM projection pairs.
3. **Real kernel work:** incremental cooperative-grid tuning of the native
   mixed K3/K4 Trellis path that is already integrated, removing the >140K
   projection compaction copy safely, eliminating no-op fused norm/RoPE work,
   and—only after profiling—an exact SM120 adaptation of LiteTopK-style
   candidate suppression.

A realistic near-term target is **5–10% more cold-prefill throughput** from a
successful code/configuration combination. A result approaching 15% is a
stretch case that additionally requires the DMA and fixed-power clock A/Bs to
land near their optimistic ends. Neither range is additive: several candidates
attack overlapping work. Effective TTFT for repeated Hermes-agent prefixes can
improve much more through prefix reuse.

## Exact production baseline reviewed

The source and runtime audit used the service actually listening on AIBeast
port 8000, not just checked-in profile defaults.

| Area | Current production posture |
|---|---|
| Hardware | 4x RTX PRO 6000 Blackwell, 96 GiB, PCIe Gen5 x16, all pairs `NODE`, one NUMA node |
| Power | 280 W/card, versus a 600 W nominal card limit |
| GPU order | `CUDA_VISIBLE_DEVICES=2,1,0,3`; rank 0 is already on the coldest physical card |
| Model | GLM-5.2 EXL3-TR3-3.42bpw, online dense K6 |
| Runtime | Gilded Gnosis v20 r31-derived image; vLLM MRv2 plus B12X/SparkInfer |
| Parallelism | TP4, DCP4, native MTP3 |
| Scheduling | 3,072 max batched tokens, 12 max sequences |
| Request limit | 520,192 tokens |
| KV | dynamic `nvfp4_ds_mla`, FP8 RoPE, exact byte-pinned 520,192 serving envelope |
| Cache offload | LMCache configured with 125/512 `GB` L1/L2 values; record effective byte capacities before the next A/B |
| DCP policy | query split, 2 indexer shards, owner merge off, full CKV gather through 140K, prefetch depth 0 |
| Collectives | lossless B12X PCIe one-shot DMA, `NCCL_BUFFSIZE=1 MiB` |
| Capture | graph/Trellis maximum 48 for C12 x MTP3 |
| Memory | about 725 MiB/rank free in an ordinary live sample; only 31 MiB was observed during a deliberate 517K stress prefill |

Retained matched measurements include approximately 2,324 tok/s at 64K and
2,200 tok/s at 128K cold prefill, and aggregate decode of 108.7 / 226.5 /
266.8 tok/s at C1 / C4 / C12. Two seeded five-depth 517K tests passed 5/5, but
an alternate needle arrangement once scored 1/5; quality gates therefore
remain mandatory even for mathematically exact-looking changes.

### What the hardware telemetry says

During mixed inference, all four cards were at 97–100% GPU utilization and
roughly 260–280 W. Memory-controller utilization was only about 20–43%, while
SM clocks varied from 2.08 to 2.63 GHz. The memory clock was pinned at 16,365
MHz versus the 14,001 MHz nominal point. GPU 3 was the slowest and hottest
rank, around 70–71 C and 2.08–2.35 GHz.

No thermal-slowdown or external power-brake reason was active; the software
power cap was. This is evidence for a **power-allocation experiment**, not yet
proof that the memory overclock is harmful. At TP4, the slowest rank can gate
the step, so minimum-rank clock and time matter more than the four-card mean.

## Ranked opportunities

### P0 — move JIT compilation to CUDA 13.2 Update 2

The live image reports NVCC 13.2.78 (CUDA 13.2 Update 1). NVIDIA's
[CUDA 13.2 Update 2 release notes](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html)
document a critical compiler correctness fix for nested thread divergence that
could otherwise leave stale or corrupted registers. Update 2 contains NVCC
13.2.86. B12X and EXL3 runtime/JIT kernels make this relevant even though no
AIBeast corruption has been attributed to the compiler.

This is a correctness prerequisite, not a performance claim. The first Update
2 image must use a fresh compile-cache namespace and repeat KLD, Aider,
degeneration, structured output, and seeded maximum-context needle gates.

### P1 — re-test 6 MiB versus the current 24 MiB PCIe-DMA crossover

The r31 worker currently exports:

```text
VLLM_PCIE_DMA_MIN_BYTES=25165824
```

That 24 MiB point came from startup calibration. A matched r17 sustained GLM
A/B found the same auto-selected crossover **3–6% slower for prefill** than a
fixed 6 MiB threshold, with no memory gain. The retained result was:

```text
VLLM_PCIE_DMA_MIN_BYTES=6291456
```

The current runtime and kernels have changed since r17, so this is an A/B, not
an automatic profile edit. It is nevertheless the strongest already-evidenced
immediate candidate. Measure PP, C1/C4/C12 TG, MAL, and exposed NCCL/B12X
traffic. Promote only if the sustained r31 result repeats.

### P2 — find the memory-clock/power Pareto point

At a 280 W cap, a 16.9% memory overclock may consume board power that would be
more valuable to SM clocks during compute-heavy prefill. Test 14,001 MHz, one
intermediate supported clock, and 16,365 MHz at the same 280 W limit.

Capture per rank:

- minimum and median SM clock;
- memory clock, power, temperature;
- DCGM Tensor Active and DRAM Active;
- 8K/64K/128K/256K prefill;
- C1/C4/C12 decode and MAL.

Hypothesis only: nominal or intermediate memory clocks may improve PP by
returning watts to the SMs, while TG may fall if decode is more bandwidth
sensitive. A card-order-only gain is unlikely; the current order already places
rank 0 on the coldest card. If GPU 3 remains the slowest independent of logical
rank, cooling/airflow is a better intervention than rank rotation.

### P3 — skip logits and sampling for unfinished prefill chunks

Upstream vLLM [PR #49171](https://github.com/vllm-project/vllm/pull/49171)
shows that MRv2 still computes a vocabulary projection and runs sampling for
unfinished chunked-prefill requests, then discards the sampled tokens. Its
Qwen3.5-4B concurrency test reports +9.87% output throughput and -9.43% mean
TTFT, but those numbers must not be projected directly onto GLM.

The audited r31 MRv2 path still has the waste. On an isolated C1 GLM prefill it
is only one logit row per chunk, so the gain may be modest. It becomes more
interesting for multiple simultaneous long prompts. The change should reduce
transient work and preserve output exactly.

The exact MTP3 audit is now tracked in local vLLM
[issue #272](https://github.com/local-inference-lab/vllm/issues/272). The
upstream target-only patch does help an all-unfinished batch even when an MTP
speculator is configured, but it misses mixed decode/prefill batches and all
draft-side work. r31 hydrates the required MTP KV, then computes and samples
three draft proposals per unfinished request and the scheduler discards them.

Do not skip MTP hydration. The safe first extension is a hydrate-only fast path
for an all-unfinished batch: retain the draft forward/KV writes, but skip the
draft LM heads, sampling and two extra MTP decode steps. At C12/MTP3 the current
path can discard 12 target and 36 draft vocabulary rows per chunk, plus 24
one-token draft forwards. A credible GLM expectation is 1-4% PP under
concurrent long-prefill load, not the upstream 9.87% headline.

Backport requirements:

- keep prompt-logprobs projection intact when requested;
- retain speculative/MTP semantics;
- compare sampled-token identity at temperature zero;
- test pure unfinished-prefill and mixed prefill/decode batches;
- re-profile peak memory rather than assuming a saving.

### P4 — incremental mixed K3/K4 Trellis tuning

The original B12X
[issue #107](https://github.com/local-inference-lab/b12x/issues/107) is fixed
and has been closed. B12X #112 shipped the native one-grid large-M mixed path
with paired-M8 FC2 and block 32; #117 made both real tier layouts runtime
dynamic. Exact r31 and the live service use one-grid block 8 for decode and
one-grid block 32 for the 206/50 and 148/108 prefill partitions.

Retained measured value is already substantial: -6.6% mixed-kernel time,
-47.1 MiB/GPU persistent scratch, +1.9% to +3.5% full-server PP and decode
parity. The earlier 3-10% estimate is obsolete.

Remaining ideas are incremental and do not yet justify another issue:

- right-size the route-packed cooperative grid rather than always launching
  the full cap (estimated 0-3% E2E PP);
- jointly tune route block/tile around the qualified block 32 point;
- test four-way instead of paired-M8 FC2 reuse only if registers remain
  spill-free.

Open a successor only after a microkernel win and at least 1% repeatable
full-server PP gain.

### P5 — remove the >140K DCP projection compaction copy safely

The audited r31 source still performs a pitched-to-compact attention-output
copy before a BF16 BMM after CKV gather ceases at 140K. The source and impact
analysis are in local-inference-lab/vLLM
[issue #207](https://github.com/local-inference-lab/vllm/issues/207).

At a 3,072-token chunk, the copy moves 288 MiB/layer; including read and write,
that is about 43.9 GiB across 78 layers. A bandwidth-only estimate is 29–34 ms
per chunk, or roughly 2–3% of long-prefill time.

The copy is a guard against a real cuBLAS tail out-of-bounds-read/Xid hazard.
The safe design is a padded-tail or ping-pong workspace that lets BMM consume
the pitched view while backing the possible over-read. Blindly deleting
`copy_` is unacceptable. Qualification requires Compute Sanitizer, Xid
monitoring, exact output, KLD, and five-depth 510K+ needles.

### P6 — hoist repeated MLA metadata and avoid eager construction overhead

Two source-audited low-risk bundles remain:

- B12X #96: probe the fused-kernel cache before constructing the complete
  `W4A16FusedMoeKernel`, and hoist environment parsing;
- vLLM #207: derive rank totals host-side, vectorize C12 lens construction,
  and compute batch-constant CKV causal/index metadata once rather than 78
  times per prefill chunk.

The current depth-0 route repeats roughly 780 tiny launches per 3,072-token
chunk. The honest combined estimate is only 0.3–0.7% PP, but it is
memory-neutral and can also reduce host jitter. Require launch-count and
timeline evidence; do not add persistent pinned slabs unless allocation churn
is actually measured.

The follow-up duplicate-work audit found two larger instances already covered
by #207 and worth implementing before the lower-value host cleanup:

- GLM computes a new sparse-indexer top-K on only 21 of 78 target layers, but
  the 57 shared-index layers repeat global-to-local/page-table remapping,
  masking and minima on the same top-K. Cache graph-stable mapped views between
  indexer layers; do not reuse this shortcut for MTP layers, whose live rows
  are compacted between draft steps. Estimated value: 1-5% PP and 0.5-3% TG.
- `_append_current_chunk_to_gathered` rebuilds the same token-to-gathered-slot
  geometry on every layer. Precompute a max-batched int64 slot map in metadata
  once (about 24 KiB at a 3,072-token chunk). Estimated value: 1-4% PP.

Those ranges overlap and must not be added. Both need exact page-table/index
identity tests plus C1/C8/C12 and long-context retrieval gates.

Two newly identified candidates are filed as measurement-first research
issues; profiling is still required before claiming or implementing a win:

- the B12X two-level paged-indexer fold may allocate transient value/index
  slabs in every indexer invocation rather than borrowing profiled caller
  scratch. At production top-k/chunk shapes a slice is roughly 48 MiB and the
  policy can use up to its 256 MiB workspace budget. Reserve bounded views in
  `B12XIndexerPagedScratchPlan`, or select streaming carry when no reserved
  slab exists. This is first a memory-accounting/reliability problem; expected
  PP from eliminating allocator churn is only 0-2%.
- the fused norm/RoPE launch uses a four-plane grid even on the 57 no-indexer
  layers, where two planes immediately return. A two-plane no-indexer
  specialization and backend-aware removal of a redundant top-K preclear may
  be worth 0.5-3% PP and 0.2-2% TG, subject to Nsight launch evidence and exact
  sentinel/index parity.

Both are now documented upstream with qualification-first scope:

- [B12X #134](https://github.com/local-inference-lab/b12x/issues/134) for
  caller-owned/profiled two-level fold scratch;
- [local vLLM #275](https://github.com/local-inference-lab/vllm/issues/275) for
  the compact no-indexer fused norm/RoPE grid and conditional B12X preclear.

### P7 — target the two independent GLM projection pairs with multi-stream overlap

The GLM/V32 attention path has two independent pairs:

1. `fused_qkv_a_proj(hidden_states)` and
   `indexer.wk_weights_proj(hidden_states)`;
2. `q_b_proj(q_c)` and `indexer.wq_b(q_c)`.

The same r31 tree overlaps analogous work for DeepSeek-V4 using
`execute_in_parallel`. TensorRT-LLM also implements a
[multi-stream attention transform](https://nvidia.github.io/TensorRT-LLM/latest/_modules/tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_attn.html).

Important: the flagship currently exports
`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024`, but the source audit confirms
that GLM-5.2's `deepseek_v32` path does not read it. It is inert for this model.
Do not describe it as an active optimization until the GLM path is wired.

The source audit and measurement contract are now tracked in local vLLM
[issue #274](https://github.com/local-inference-lab/vllm/issues/274).

The first candidate should be fusion, not streams: fold the skinny replicated
BF16 `indexer.wk_weights_proj(hidden_states)` into the existing replicated BF16
`fused_qkv_a_proj(hidden_states)` on the 21 full-indexer layers. This removes a
launch and a second hidden-state read while preserving the existing `kw=`
injection point. Estimated PP value is 0.5-2%.

Whole-cone overlap remains a second experiment. At M=3,072 both online-K6
2048x4096 GEMMs can fill the 188 SMs, and the machine is power-limited. It may
range from -5% to +2% PP and adds about 96-128 MiB/rank of simultaneous live
storage. Keep decode serial unless profiling replaces the current tuned SM
budgets.

### P8 — test the real 3,072-token mHC selector

B12X consumes:

```text
B12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS
```

Its default is 4,096, so a 3,072-token prefill uses the base h7168 geometry.
Test 4,096 versus 3,072. Any `SPARKINFER_...` spelling of this setting is a
no-op. The 4,096 crossover was intentional, so a win must be measured; a
reasonable prior is -2% to +2% whole-prefill effect, not a guaranteed gain.
Prewarm both compiled kernels before memory profiling.

### P9 — exact LiteTopK-style candidate suppression for SM120

[LiteTopK](https://github.com/Heisenberg-Yin/LiteTopK) is the most interesting
outside effort because it directly attacks the GLM-5.2 DSA indexer/top-k path.
Its public implementation targets B200/SM100 and FP8, not RTX SM120,
dynamic-NVFP4 B12X. Published repository evidence reports roughly 1.25x GLM
prefill and a 524K indexer reduction from about 25.9 to 20.5 ms.

This is not production-ready for our exactness contract. The public result
describes approximately 99.998% candidate-set recall rather than literal 100%
identity. The safest transplant is:

1. port score-range/binning and candidate-write suppression to the existing
   B12X SM120 indexer;
2. retain exact final top-k selection;
3. fall back to the dense/exact route whenever a boundary is ambiguous;
4. compare top-k index identity, not merely generated text.

Only start this port if Nsight shows indexer/top-k is still a material share of
128K–520K prefill.

The first trace needs no code change: aggregate kernel time for
`attention.indexer.contiguous_logits`, `attention.indexer.tiled_topk`,
`attention.indexer.row_topk`, paged gather and DCP merge. Then add optional,
default-off NVTX ranges around WK/WQ projection, gather, score, top-k, DCP
merge, sparse attention and EXL3 MoE so `nvtx_gpu_proj_sum` attributes the
complete stack.

Measure the LiteTopK-addressable fraction `f` as score + top-k + candidate
write time divided by total prefill GPU time. If a port matches LiteTopK's
1.265x kernel speedup, Amdahl's law caps whole-prefill speedup at:

```text
1 / ((1 - f) + f / 1.265)
```

Examples: `f=10% -> 2.1%`, `20% -> 4.4%`, `30% -> 6.7%`, and
`40% -> 9.1%`. Invest only if the conservative modeled gain is at least 3%
and the exact top-k fallback remains feasible. GLM uses the full indexer on 21
of 78 target layers, so dominance must be measured rather than assumed.

### P10 — dense-MHA crossover for short and moderate contexts

SGLang [PR #20062](https://github.com/sgl-project/sglang/pull/20062)
adds a threshold for dense versus sparse attention on GLM/V32. Its reported
2K/16-request workload reduced TTFT substantially while keeping TPOT flat.
ExLlamaV3 independently added an MHA-form MLA prefill path.

B12X deliberately reports `supports_mha_prefill=False`, so this needs code,
not an environment tweak. A conservative first implementation uses dense MHA
only where the KV length is no greater than sparse top-k, making the selected
set equivalent. Sweep 0/2K/4K/8K/16K. This improves early chunks, not the
steady-state 128K–520K path.

### P11 — optimize the dynamic-NVFP4 KV writer only if it is visible in the trace

SGLang [PR #25311](https://github.com/sgl-project/sglang/pull/25311)
replaced an elementwise MLA KV scatter with a one-row-per-CTA/TMA bulk-store
design and measured its SM103 writer microkernel at 21.6 to 1.76 us for 4,096
rows. That 12x number is neither an end-to-end result nor portable directly to
this stack.

B12X already fuses dynamic NVFP4 quantization, FP8 RoPE and the KV write, with
one CTA per token. Its remaining suspicious detail is that the RoPE part has
less intra-warp parallelism than the grouped non-RoPE record. The transferable
ideas are cooperative row staging, a bulk store, or more parallel RoPE
quantization—not replacing the B12X writer with SGLang's BF16/FP8-record
kernel. Attempt this only if Nsight attributes material layer time to the KV
writer; otherwise a spectacular microkernel ratio will disappear in the
753B-model denominator.

## Prior-art disposition

| Source | Useful principle | Disposition for AIBeast |
|---|---|---|
| Main vLLM | Skip unfinished-prefill sampling; per-request chunk scheduling; fused GLM indexer/RoPE work | Sampling skip is a near-term backport. Much of the GLM fusion work is already present. |
| local-inference-lab vLLM/B12X | Exact TP4/DCP4 query split, CKV gather, mixed EXL3, custom PCIe collectives | This is the production foundation and already carries the largest applicable gains. |
| SGLang | Dense/sparse crossover, TMA KV store, context-parallel DSA, dynamic chunking | Crossover and writer techniques are transferable; CP/disaggregation is a different memory/topology design. |
| TensorRT-LLM | Targeted multi-stream attention transforms and dynamic chunk budgeting | Apply to the two independent GLM projection pairs and mixed workload policy, not wholesale runtime replacement. |
| ExLlamaV3 | MHA-form MLA prefill and quant-cache staging | Dense crossover is relevant. Full dequantized staging can require gigabytes and is not compatible with present headroom. |
| FlashInfer / FlashAttention-4 / FlashMLA | Persistent/load-balanced sparse kernels, async pipelines, larger tiles | Kernel-design references; published implementations target different architectures/records. |
| llama.cpp | Tensor splitting and CPU/GPU graph-placement lessons | No superior GLM-5.2 sparse-MLA implementation was found; no direct path to transplant. |
| Sarathi / DeepSpeed-FastGen | Chunked/SplitFuse scheduling | The broad mechanism is already in vLLM V1; only dynamic budgeting remains interesting. |
| Hydragen / ChunkAttention | Compute sharing for common prefixes | Valuable longer-term for concurrent agents; current prefix caching shares storage, not attention computation. |
| LiteTopK/LiteDSA | Suppress candidate writes before exact top-k | Highest-upside external research lead, but requires an exact SM120 adaptation. |

Reddit and community searches were useful for finding configurations and
cross-checking expectations, but produced no independently verified raw-PP
path that beats the current four-card B12X work. In particular, unverified
ForceP2P/registry changes, generic NCCL channel recipes and FP8-KV anecdotes do
not override the measured topology, lossless-quality and 520K requirements.
Community claims should continue to enter the matrix as hypotheses and leave
it only with immutable-source A/B evidence.

## Workload-level gains that can exceed kernel tuning

### Prefix reuse

For repeated Hermes-agent histories, avoiding prefill is better than making
prefill 10% faster. Preserve canonical system prompts, deterministic tool
ordering and stable message serialization so both GPU prefix caching and
LMCache can hit. The observed live interval had meaningful but far from
complete prefix reuse, leaving room for cache-aware scheduling and prompt
normalization.

Tree-aware shared-prefix attention, as explored by
[Hydragen](https://arxiv.org/abs/2402.05099) and
[ChunkAttention](https://arxiv.org/abs/2402.15220), is a longer-term route to
share attention computation across concurrent agents, not merely KV storage.

Approximate non-prefix cache blending must remain opt-in: it can affect the
exact long-context retrieval contract, and the current LMCache V1 integration
still has open edge cases.

### Dynamic chunk budgeting

Sarathi-style chunked prefill is already present in vLLM V1, but a fixed 3,072
budget is a compromise. A scheduler could retain 3,072 when no decode is
running and temporarily reduce to 2,560 or 2,048 during active decode to
protect ITL. This changes mixed-workload latency and goodput, not isolated cold
PP, and must not be confused with a raw kernel win.

### Mixed prefill/decode row splitting

B12X chooses its decode route only when the batch-wide maximum query length is
at most one. One long prefill can therefore send accompanying decoder rows
through the unified extend route. A POD-Attention-inspired split can preserve
the optimized decoder kernel while sparse extend handles the prefill rows.
Start sequentially with explicit scratch ownership; concurrency comes later.
The value is protecting TG/ITL under a 128K–520K prefill, not accelerating an
isolated prompt. This is now tracked in local vLLM
[issue #273](https://github.com/local-inference-lab/vllm/issues/273).

### CPU run-ahead and DDR5 as an active cache tier

AIBeast has a 32-core/64-thread Threadripper 9970X, one NUMA node and four
populated DDR5-5600 RDIMM channels. AMD documents four memory channels; the
installed configuration has a theoretical 179.2 GB/s payload ceiling. All four
GPU links negotiate PCIe 5.0 x16.

The live process sample shows why moving tensor work to CPU is the wrong goal:
the host is about 98% idle overall, but the EngineCore consumes one complete
CPU core. Model/attention projections on CPU or spilling 100+ MiB activation
cones through DDR/PCIe would serialize the GPU path. Useful CPU work is instead
run-ahead: schedule the next batch, build/page-table metadata, tokenize, and
prefetch cache records while the current GPU step runs.

The exact r31 tree supports async scheduling with MTP and its LMCache adapter
deduplicates async completion IDs, but the turnkey profile explicitly uses
`--no-async-scheduling` due to earlier speculative/CKV lifetime concerns. This
is now a high-value controlled A/B: it can hide the pegged EngineCore work but
must pass MTP, structured output, LMCache, preemption and maximum-memory gates.
It does not make two model forwards run concurrently.

DDR5 is already productive through LMCache. The live r31 snapshot showed:

- L1 DRAM: 83.3 GB used of 125 GB (62%);
- L2 NVMe: 260.4 GB used of 512 GB;
- L0->L1 store average: about 9.6-10.2 GB/s per GPU;
- L1->L0 restore average: about 1.72-1.75 GB/s per GPU;
- L2->L1 load average: about 8.85 GB/s;
- L1 reads 173,728 versus 18,712 writes since startup.

The low restore rate relative to DDR/PCIe makes predictive and more concurrent
prefetch worth testing for repeated Hermes histories. Compare CPU/GPU worker
counts, prefetch concurrency and canonical-prompt prewarming using LMCache's
throughput/inflight/failure metrics. Do not increase workers blindly: DDR and
PCIe traffic competes with DCP4 collectives, and this improves cache-hit TTFT,
not cold-prompt PP.

## Already integrated or intentionally rejected

Do not spend a maintenance window rediscovering these:

- **Current DCP policy:** query split + two indexer shards + owner merge off is
  correct for TP4/DCP4's single query partition. The r26 policy change already
  produced double-digit PP gains over the old owner-exchange route.
- **CKV gather 140K:** raising directly to 520K costs about 333.5 MiB/rank,
  unsafe against a 31 MiB deliberate maximum-context transient margin. A 192K
  re-test costs about 45.8 MiB/rank and belongs only after the compaction-copy
  change alters the crossover.
- **CKV prefetch:** depth 1 historically gained a few percent at long context
  but cost hundreds of MiB and logical KV. It is not compatible with current
  worst-case headroom.
- **Owner merge:** beneficial at larger TP, pure overhead for TP4/DCP4's one
  query partition. Keep it off.
- **One-shard full replication:** weak PP result versus the wrong control and
  consumes replication memory. Current two shards are better grounded.
- **Remote-push all-reduce:** current B12X work targets TG and costs roughly
  112 MiB/rank/channel. It is contrary to the present PP-first tight-memory
  objective.
- **FP8 KV:** faster and higher quality in Aider/KLD tests, but measured usable
  context was only about 295K. It fails the 512K–520K requirement.
- **Lossy FP8 PCIe transport:** modest historical PP benefit, but deep-context
  quality concerns make it unsuitable without a complete KLD/Aider/needle
  campaign.
- **DCP1/DCP2:** cannot retain the present model, speculation and 520K envelope
  on four 96 GiB ranks.
- **4,096 fixed prefill batch:** already produced unsafe memory/OOM behavior at
  exact 520K. Revisit only after a measured memory recovery.
- **More CUDA graph variants:** consumes scarce persistent memory and does not
  solve large eager-prefill kernels.
- **Blind NCCL tuning:** B12X custom collectives carry the dominant traffic.
  Instrument fallbacks first. If channels are tested, use current
  `NCCL_MAX_CTAS`, not deprecated `NCCL_MAX_NCHANNELS`.
- **FlashAttention-4/FlashMLA direct adoption:** useful design references, but
  published kernels target SM100/B200 or different KV records, not this
  SM120 dynamic-NVFP4 sparse-MLA ABI.
- **Disaggregated prefill:** official vLLM documentation states it controls
  TTFT/ITL rather than improving throughput, and one four-GPU copy cannot host
  separate 753B prefill and decode replicas.
- **Approximate token dropping/sparse prefill:** incompatible with the exact
  maximum-context needle contract unless a separate quality tier is created.

## Expected frontier after this work

The most defensible sequence is:

1. CUDA 13.2 Update 2 correctness rebuild;
2. 6 MiB DMA and power/clock A/B;
3. one representative 128K Nsight/DCGM trace;
4. unfinished-prefill sampling skip and MLA metadata hoist;
5. targeted GLM multi-stream overlap;
6. >140K compaction-copy removal;
7. incremental mixed-Trellis cooperative-grid tuning;
8. exact LiteTopK-inspired work only if the profile justifies it.

If the first six stages yield a combined 5–10% without a TG or memory
regression, that is an excellent outcome. The native mixed-Trellis path is
already integrated; its remaining grid experiments have a 0-3% E2E ceiling
until profiling proves otherwise. Prefix reuse can deliver much larger
user-visible TTFT reductions for recurring agents and should be optimized in
parallel.

## Measurement contract

All candidates use A/B/A or A/B/B/A order with identical immutable sources,
model revision, byte-pinned KV, warmed compile artifacts and cold unique prompt
prefixes.

- Prefill: 3K, 8K, 32K, 64K, 128K, 180K, 256K, and 500K+.
- Decode: C1, C4, C8, C12; report aggregate and per-request TG plus MAL.
- Mixed: one cold 64K/128K/500K prefill while 1/4/8 decoders are active.
- Hardware: minimum-rank SM clock, power, temperature, Tensor Active, DRAM
  Active, PCIe TX/RX.
- Runtime: TTFT, ITL, PP, TG, MAL, preemption, prefix/offload hits, peak
  activation, minimum physical free VRAM, logical KV.
- Correctness: deterministic output parity, KLD, Aider, structured output,
  degeneration, five-depth maximum-context needles with at least two seeds.
- Reliability: no OOM, restart, Xid, late JIT, new fallback, or cache failure.

Promotion requires a repeatable PP improvement of at least 2% for a standalone
knob (0.5% is acceptable for a proven low-risk launch-overhead cleanup), TG
geometric-mean regression no worse than 2%, no normalized MAL regression, and
at least 512 MiB ordinary physical headroom. Deliberate maximum-context tests
may go below that only if every runtime allocation has already been profiled
and the test is explicitly classified as a stress gate.

## Primary references

- [NVIDIA CUDA 13.2 Update 2 release notes](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html)
- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [NVIDIA DCGM profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [vLLM PR #49171: skip logits and sampling for unfinished prefills](https://github.com/vllm-project/vllm/pull/49171)
- [local vLLM issue #272: MTP hydrate-only unfinished-prefill path](https://github.com/local-inference-lab/vllm/issues/272)
- [local vLLM issue #273: mixed B12X prefill/decode attention](https://github.com/local-inference-lab/vllm/issues/273)
- [local vLLM issue #274: GLM projection fusion/selective overlap](https://github.com/local-inference-lab/vllm/issues/274)
- [local vLLM issue #207: B12X MLA/DCP hot-path audit](https://github.com/local-inference-lab/vllm/issues/207)
- [B12X issue #134: profiled two-level fold scratch](https://github.com/local-inference-lab/b12x/issues/134)
- [local vLLM issue #275: eliminate fused norm/RoPE no-op planes](https://github.com/local-inference-lab/vllm/issues/275)
- [B12X issue #107: mixed K3/K4 large-M Trellis](https://github.com/local-inference-lab/b12x/issues/107)
- [LiteTopK repository](https://github.com/Heisenberg-Yin/LiteTopK)
- [NVIDIA Nsight Systems analysis reports](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [NVIDIA CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [AMD Threadripper 9970X specification](https://www.amd.com/en/products/processors/ryzen-threadripper/9000-series/amd-ryzen-threadripper-9970x.html)
- [LMCache MP observability metrics](https://docs.lmcache.ai/mp/observability/metrics.html)
- [SGLang PR #20062: dense/sparse GLM threshold](https://github.com/sgl-project/sglang/pull/20062)
- [SGLang context-parallelism roadmap](https://github.com/sgl-project/sglang/issues/21788)
- [vLLM disaggregated-prefill documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [POD-Attention](https://arxiv.org/abs/2410.18038)
- [Hydragen](https://arxiv.org/abs/2402.05099)
- [ChunkAttention](https://arxiv.org/abs/2402.15220)
