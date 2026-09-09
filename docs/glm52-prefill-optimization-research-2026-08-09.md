# GLM-5.2 prefill optimization research — 2026-08-09

> Historical research: the August 9 baseline, hypotheses and forecasts below
> are preserved as a dated record, not current tuning instructions. The
> [September 8 appendix](#2026-09-08-glm-53-deployment-and-constrained-frontier-review)
> supersedes its present-tense frontier assessment and power/clock/DMA
> recommendations, and corrects the later mHC causal attribution.

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

## 2026-09-08: GLM-5.3 deployment and constrained-frontier review

### Verdict and evidence standard

**[INFERENCE] The deployed full GLM-5.3 stack is plausibly near the practical
constrained frontier, not a proven Pareto optimum. No other investigated stack
demonstrably dominates it under the same contract.** That contract is four
96 GB RTX PRO 6000 Blackwell Workstation cards, 520,192 total tokens, C12,
full 3.42bpw weights, TP4/DCP4, exact/lossless transport, the current quality
policy and production reliability. A faster alternative with fewer tokens,
lower precision, different hardware or weaker correctness is a different
tradeoff, not domination. No distance from optimal or local speed forecast
can be justified from the available evidence.

Evidence grades used here:

- **A — independent evaluator:** its own dated measured results, with the
  evaluator's harness and population limits.
- **B — vendor:** first-party model-card results, even when using public suites.
- **C — public runtime report:** firsthand qualification, testimonial or local
  receipt; not automatically independently reproduced or paired.
- **P — proxy:** distributional fidelity, not task-level correctness.
- **Local operational evidence:** immutable deployment receipts and retained
  log scans; these establish exercised behavior, not a model-quality ranking.

All research and follow-up proposals below are investigations/recommendations,
**not executed tuning experiments**. Completed deployment, restoration, soak
and cleanup actions are separately identified by their receipts.

### What actually ran, and how narrow the memory margin is

The [active identity](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/active.json)
is image `8006d209b8f1d1bbf815983514e430fb77bbf01bd66075578483473d9310416a`
(source `499d34e`), container `glm53-turnkey-r34-parity-20260908t015846z`.
The [complete weight audit](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/weight-integrity.json)
verified all 81 files of
[`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@99c6f951333d2b38f1efefa533c7afadf0d376e3`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/tree/99c6f951333d2b38f1efefa533c7afadf0d376e3).
Canonical weights are unchanged; a
[local metadata derivative](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/metadata-reconciliation.json)
reconciles the obsolete native-MTP projection ignore entry. NFS/cachefilesd
priming and a host manual-NVIDIA-hook fix in `run-local-podman.sh` were
deployment work, not a hot model-runtime patch.

The old resource shape survived: scheduler/EXL3 prefill arena 3072/3072,
12 sequences, graphs/Trellis 48, GMU 0.95, fixed KV 4,518,907,904 bytes/GPU,
TP4/DCP4, native probabilistic MTP3, dynamic NVFP4 KV + FP8 RoPE, online K6,
LMCache 125 GiB initial RAM plus 384 GiB disk. Byte-pinned KV supersedes GMU
autosizing; C12 is an admission ceiling, not twelve concurrent 520K requests.
The [hardware receipt](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/hardware.json)
shows 97,887 MiB/card, 375 W caps, driver 595.71.05, one NUMA node and NODE
PCIe topology—not NVSwitch-equivalent bandwidth.

Initial startup ran 02:13:47–02:29:15 UTC. The
[first parity needle](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/needle-initial-parity.json)
retrieved 3/3 facts from 501,098 haystack tokens in 336.364 seconds.
An explicit 02:58:58 restart restored the old B12X policy; the
[final parity needle](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/needle-final-parity.json)
retrieved 3/3 at 501,099 tokens in 275.874 seconds, followed by a healthy
512-token temperature-1 sampler in 7.701 seconds. These are body-token counts
excluding template/question overhead. Different cache warmth and live load
make the two durations **observations, not a causal fold-cap speed A/B**.

The [03:36:34 UTC accepted soak](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/soak-accepted.json)
records 64 healthy/0 failed samples and 31.55 minutes since the first external
client, with Chat and Responses traffic. There were 87 external POST 200
headers and 96 engine completions **including probes**, with zero engine-error
requests. HTTP 200 streaming headers are not independently proof of completed
response bodies. No OOM, engine crash or unexpected restart was observed.
Free VRAM minima at 30-second cadence were **215/245/217/215 MiB**:
memory-thin operational acceptance, not continuous worst-case allocator,
sampler or workspace headroom. It does not satisfy every historical August
stress gate or substitute for the user's 24-day old-model stability record.

At 03:37:31, [cleanup](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/old-cache-cleanup.json)
removed only the old runtime LMCache and compile-cache directories, reclaiming
370,534,412,288 bytes (345.09 GiB); health afterward was 200. Old
image/container/state/logs/weights, new caches and cachefilesd were preserved.
Port 8000, `AUTH=none`, `LANDING=0`, and aliases `GLM-5.2`/`local-primary`
remain compatible, with `GLM-5.3` added. No global `latest`/`main` promotion or
boot-policy change occurred; restart policy remains `no`.

### Effective B12X parity, and the mHC attribution correction

The live image still had legacy `SPARK_` fold settings; the installed B12X
reader does not translate them. The absent-override policy was `auto`/256 MiB,
not the intended 64 MiB. The
[six restored controls](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/b12x-parity-restoration.json)
were supplied through `TUNE_B12X_*` launch overrides. The
[installed pure-policy receipt](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/b12x-effective-policy.json)
now reports `auto`/67,108,864 bytes and an over-budget candidate choosing carry.
This proves the policy reader under the observed launch environment, not an
observed peak allocation or every live kernel route. Source now uses the
correct B12X names for future builds; that source edit did not mutate the
running image.

Only fold 64 versus 256 MiB is demonstrated to change applicable policy.
Fold mode `auto`, remote-push `0`, small-M split-K `0` and T12 SMEM `1`
otherwise match defaults; the latter is additionally T12-specific, not this
MCG codebook. Fold is an exact memory/performance choice, not an approximate
quality knob. Actual query rows and workspace geometry matter: scheduler 3072
alone does not prove the speculative 96–240 MiB slab sizes were allocated.
See pinned [indexer policy](https://github.com/local-inference-lab/b12x/blob/7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b/b12x/attention/nsa_indexer/paged.py),
[environment reader](https://github.com/local-inference-lab/b12x/blob/7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b/b12x/_lib/env.py)
and open [scratch-budget issue #134](https://github.com/local-inference-lab/b12x/issues/134)
(August 9; status checked September 8).

**Correction to the historical mHC explanation:** the audited
[mHC kernels](https://github.com/local-inference-lab/b12x/blob/7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b/b12x/norm/mhc/_kernels.py)
support hidden sizes 4096/7168; this pinned full GLM has hidden size 6144.
The [GLM model mapping](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/model_executor/models/deepseek_v2.py)
and [optimized implementation](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/models/deepseek_v32/nvidia/model.py)
do not establish an mHC path; the latter uses RMSNorm/fused all-reduce RMSNorm.
Thus threshold 3072 is **likely inert for full GLM**. The
[August maintenance record](aibeast-r33-maintenance-results-2026-08-09.md#mhc-prefill-geometry-crossover)
still records PP control/candidate medians 2542/2672 at 3K, 2678/3061 at 8K,
2590/2880 at 64K, 2398/2736 at 128K and 2175/2352 at 262K tok/s.
Keep these observations, including their live-traffic caveat, but retract the
causal claim that changing mHC geometry produced those GLM gains. Restoring
3072 preserves explicit configuration, not a proven speed optimization.

### Is the original model smarter? Evidence with dates and versions

There is real evidence of gains in specific measured capabilities, **not a
universal percentage smarter**, and not proof that the exact local quantized
`high`-effort deployment retains all of them.

| Evaluation | GLM-5.2 → GLM-5.3 | Date/version and interpretation |
|---|---|---|
| AA Intelligence Index, current | 39 → 45 (+6 rounded points) | **A**, retrieved September 8; both [5.2](https://artificialanalysis.ai/models/glm-5-2) and [5.3](https://artificialanalysis.ai/models/glm-5-3) pages label **v4.3**. The 5.2 chart has an “Estimate (independent evaluation forthcoming)” legend whose per-datapoint scope was not recoverable; not every new component can be called independently rerun. |
| AA Intelligence Index, launch | 53 → 60 (+7 points) | **A**, [August 18 launch](https://x.com/ArtificialAnlys/status/2089830890709135426), [original chart](https://pbs.twimg.com/media/HQCRfGPakAArNUF.jpg?name=orig), max/max, **v4.1-family**. Chart says v4.1; current method history calls the August–September revision v4.1.1. |
| GDPval-AA v2 | 1524 → 1770 (+246 Elo) | **A**, same August 18 launch, not a claim about today's leaderboard or a task-solved percentage. |
| FrontierSWE v1 | Mean@5 average rank 6.21 → 4.50 (lower better); chart dominance 67% → 78% | **A**, [Proximal August 14](https://x.com/ProximalHQ/status/2088336618495377636), [chart](https://pbs.twimg.com/media/HPtAamBaUAA87bz.jpg?name=orig), [v1 method](https://frontierswe.com/v1). Vendor card's decimal precision is 67.5 → 78.1 (**B**), +10.6 dominance points. Dominance is pairwise win probability, **not 78.1% of tasks solved**. |
| AA-Omniscience | score 4 → 14; accuracy 24% → 34%; attempt rate 46% → 55% | **A**, August 18 launch. More correct answers support genuine gains; this is not a general intelligence percentage. |
| AA-Omniscience hallucination rate | 26% → 30% (+4 percentage points worse) | **A**, August 18 launch. Keep the evaluator's denominator; do not infer universally improved reliability. |
| Output use per AA task | about 15,700 → 18,700 tokens | **A**, August 18 v4.1-family: about +19%, rounded to **20%** in the post. Added capability has an output/latency cost, not a free speedup. |
| Terminal Bench 3.0 | 4.6 → 28.3 | **B**, [official card](https://huggingface.co/zai-org/GLM-5.3), retrieved September 8. Vendor-run public suite, not independent replication. |
| DeepSWE v1.1 | 46.2 → 66.9 | **B**, same card; +20.7 points under its harness, not a local coding A/B. |

The [AA version history](https://artificialanalysis.ai/methodology/intelligence-benchmarking#version-history)
explains why **60 and 45 must not be compared as a regression**. v4.2 added
Briefcase/GDP.pdf, removed GPQA, revised LCR and SciCode grading and reweighted
categories; v4.3 replaced tau with AutomationBench-AA and Terminal Bench 2.1
with 4.0/mini-SWE-agent. Current weights are agents/coding/scientific/general
30/20/20/30, versus v4.1's 34/24/24/18. The earlier 44-versus-42 observation
has no retrieved, dated index-version evidence; do not relabel it v4.2 or
repeat it as current.

[GDPval-AA methodology](https://artificialanalysis.ai/evaluations/gdpval-aa)
uses 220 real deliverable tasks across 44 occupations and nine industries,
agent tools and blind pairwise judging with Bradley–Terry Elo anchored to
human deliverables at 1000. It is independent agentic-work evidence, not a
coding unit-test score. No GLM-specific confidence interval was retrieved,
and current readable leaderboard rows did not expose this pair; 1524/1770
is the **dated launch pair**. The vendor's 1508/1769 is a different snapshot,
not interchangeable endpoints. FrontierSWE v1 has 17 tasks; partial test
progress ranks implementation tasks with no complete successful solution.
Mean@5 is not pass@1, best@5 or percent solved.

Vendor settings also limit transfer. Terminal Bench 3 uses Claude Code
2.1.207, max effort, 400K context, 128K output, avg@3, up to 600 turns/10 hours
and a separate verifier. DeepSWE uses mini-swe-agent, 400K context and six
hours. FrontierSWE, ALE and several other card protocols use 1M context,
beyond this local envelope. HLE-with-tools uses a 163840 output cap, versus
local 131072 (32768 tokens/20% less), plus its own tools/context management.
Near-500K input cannot simultaneously reserve 131072 output within 520192
total. The vendor's “50% coding improvement” refers to private Z.ai Code
Bench, not 50% smarter or an independent suite aggregate. Disclosed verifier,
anti-cheat and environment changes require pinned harness comparisons; they
are neither proof of cheating nor reproducible by changing only model ID.
The February [GLM-5 paper](https://arxiv.org/abs/2602.15763) (v2 February 24)
is architecture/RL background, not documentation of August GLM-5.3 posttraining.
No defensible same-date full-model human-arena pair was retrieved; Flash
AutoEval results and ranks across changing pools are not a substitute.

### Exact quant quality: preserve the contrary evidence

The user's [August 30 GPQA report](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/discussions/2)
is an actual 3.42bpw evaluation: **169/198 = 85.35%**, Wilson 95% interval
79.76–89.60%, zero API errors; temperature 0.6, max effort, C8, TP4/DCP4,
NVFP4 KV + FP8 RoPE, 131072 output cap, llm-inference-bench 0.4.29 at
`42c38fdd`. Eight responses hit the cap; two lacked final answers and six
already contained one. Even awarding both unanswered cases yields 171/198
(86.36%). The August 31 comment says it scored below same-bpw GLM-5.2,
without supplying the numeric 5.2 ledger there. **C: prior self-report, not
independent corroboration, not a paired proof of regression**, and not this
current image. The cited AA 91.7% reference is 6.35 points higher, but prompts,
ordering, repeat counts, output limits and the entire serving stack differ:
this is not isolated quantization loss. Do not reuse the report's inconsistent
aggregate-throughput label as a hardware benchmark.

The [3.42 card](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw)
and [3.25 card](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
report August 29 teacher-forced full-vocabulary KL(teacher||student), four
held-out windows of 2047 positions, with a separate CN3 reproduction:

| Weight / KV | Original KL | CN3 KL |
|---|---:|---:|
| 3.42 / FP8 | 0.024105 | 0.023966 |
| 3.25 / FP8 | 0.026103 | 0.026776 |
| 3.42 / NVFP4 + FP8 RoPE | 0.039518 | 0.037695 |
| 3.25 / NVFP4 + FP8 RoPE | unmeasured | 0.039396 |

**P:** this measures short-window distributional fidelity, not real coding
success, 500K synthesis or retained 5.3-over-5.2 gains. KV format moves this
proxy more than 0.17 weight bits; it does not establish global monotonicity
or that 3.25 dominates. The separate [September 5 discussion correction](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/discussions/3)
labels its reconstructed-BF16/Transformers KL lane **advisory**, not strict
serving-kernel fidelity; do not pool unlike panels/teachers.

3.42 uses 148 K3/108 K4 experts per layer versus 3.25's 192/64; it upgrades
44 experts/layer with the same selection machinery. Correct mixed-tier
loading is essential: the card warns a uniform-K loader can produce fluent
garbage. Nominal 3.25 checkpoint metadata is 339,256,027,136 bytes versus
3.42 tensor payload 355,034,998,784: about 3.674 GiB/equal quarter saved,
**not measured VRAM recovery**. Full 3.42's tensor payload exceeds 5.2's by
3,640,983,552 bytes total (about 0.848 GiB/equal quarter); loaded model,
KV, graphs and workspaces are distinct allocations.

Preserve current 3.42/NVFP4. A **separate, explicit 3.25 + FP8-KV experiment**
is worth considering only after actual artifact/layout, per-rank loading and
peak fit demonstrate the same 520K/C12 envelope, and matched task-quality
measurements justify the exchange. It is not an automatic fallback or a
claim that the checkpoint-size saving necessarily pays for FP8 KV.
Higher-bit full 5.3 fit at unchanged context/concurrency is likewise unproven.

### Reasoning budget and history are part of the deployment contract

The [pinned-template rendering receipt](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/thinking-template-probe.json)
shows:

| Template input | Pinned GLM-5.2 | Pinned GLM-5.3 |
|---|---|---|
| Defaults | Clears old reasoning; opens `<think>` | Preserves old reasoning; opens `<think>` |
| `enable_thinking: false` | Clears old reasoning; emits `<think></think>` | Still preserves old reasoning and opens `<think>`; flag does not disable it |
| `clear_thinking: true` | Clears old reasoning | Clears old reasoning, still opens new `<think>` |

This is a rendered-template proof, not a measured speed or answer-quality
benefit. No global template edit was made. Use the
[official GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) `low`/`high`/`max`
efforts rather than promising thought disable; official default and reported
benchmarks use max, whereas this appliance defaults high. Clearing history
may save tokens and alter prefix identity, but also removes reasoning context:
measure real agent tasks before changing the client policy.

### Public community evidence, including negative reports

These are accessible public sources, not private Discord testimony:

| Report and date | What it supports | Grade and limit |
|---|---|---|
| [3.42 bring-up discussion](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/discussions/1), August 29–30 | Positive historical 393216/C8 qualification: 72/72 temperature-1 decode requests and 15/15 long-context facts; author corroborates first-sampler/graph OOM in the older 520K attempt | **C**. Qualification is the user's campaign, not an independent positive vote; author corroborates the OOM mechanism. C8 227.55 tok/s is aggregate, not C1 or AIBeast parity speed. The historical rejected arm does not disprove today's successful 520192 profile. |
| [Independent agent PR #22](https://github.com/KrishnaAnnavaram/semantic-mcp-data-access-gateway/pull/22), August 30 | Reverted default to 5.2: 5.3 truncated 1/4 cases at both 1200 and 4000 ceilings, 5.2 0/4; one turn spent 9999/10000 tokens reasoning | **C**, firsthand Z.ai API workflow, small sample, unknown quant/hardware. Reported 476/557-second agent latencies versus earlier 110–370 seconds are multistage observations, not controlled decode/TTFT. Low effort helped one probe; not evidence no larger budget could work. |
| [DCP2 issue #236](https://github.com/local-inference-lab/vllm/issues/236) and [PR #54](https://github.com/local-inference-lab/rtx6kpro/pull/54), August 4–10 | GLM-5.2 DCP2 long-context corruption could look fast; reporter later confirms r33 fix with salted probes | **C**, same reporter/campaign, not two replications. DCP4 control was clean; current DCP4 is not implicated. Broken 14.97-second readout versus 50.21 seconds is not a speed win. |
| [Persistent-L2 contamination #55](https://github.com/local-inference-lab/rtx6kpro/issues/55), August 4 | Bad KV from broken images replayed through healthy ones and produced false bisections; salts exposed cache contamination | **C**, detailed firsthand mechanism, not independent reproduction or evidence the stable old AIBeast service was corrupt. Preserve checkpoint/engine/KV-layout cache namespaces. |
| [Prefill tuning PR #57](https://github.com/local-inference-lab/rtx6kpro/pull/57), August 4 | Same-boot 64K prefill: CKV gather cap 16384 yielded 1478 tok/s versus 2895 with 140K default | **C**, counterexample to treating smaller workspace as free performance; incomplete power/batch/harness controls, not 5.3 decode or AIBeast gain. |
| [Malformed-history #684](https://github.com/local-inference-lab/vllm/issues/684) and [candidate fixes #701](https://github.com/local-inference-lab/vllm/pull/701), September 6–7 | Incomplete tool JSON can cause pre-generation 400; Flash/Jovian report also identifies late callback races | **C**, concrete frontend lead and CPU tests, open PR, not current full-GLM GG GPU qualification. Establish exact path applicability before porting or inventing JSON repairs. |
| [Lifecycle issue #60](https://github.com/local-inference-lab/rtx6kpro/issues/60) / [fix #252](https://github.com/local-inference-lab/vllm/pull/252), August 7 | Positive reporter/maintainer reproducer, fix and 1792/1792-request live qualification | **C**, strong reliability evidence for that native-offload DS4 TP2 profile, not current full-GLM LMCache or model-quality evidence. |

GitHub #54/#57 refer to Discord tuning posts; the actual sources retrieved were
GitHub. No authenticated/private Discord content or direct message archive
was accessed, and no accessible controlled independent positive full-5.3
coding testimonial was found. This does not prove none exists. Positive
qualification and author corroboration are retained without manufacturing a
community consensus; the user's own HF reports are not counted twice.

### Complete-log review: stable engine, rare request failures

The retained old stdout scan covered **134,038,560 bytes / 767,123 records**,
August 15 01:57:47 to September 8 02:10:09 UTC (24 days, 12 minutes).
It contains one startup/engine initialization/ready event and one retained
PID per role, with no matched engine crash or restart: this respects the
user's stable-uptime observation. The initial new snapshot covered
700,903 bytes / 3291 records through 02:40:19; it had only 11 post-ready
minutes and is not the later accepted soak. Main additionally scanned the
old internal LMCache log (about 348 MB / 2,078,720 records).

The [final scan ledger](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/log-review.json)
also covers the complete first-parity attempt and final container through
04:10:11 UTC: the latter has 4,946 stdout records and 5,127 cache-service
records, with no ERROR-level entries. Its token-weighted draft acceptance is
73.17%; the earlier 70.89% figure below describes the short initial snapshot.
Neither is a matched speed or quality comparison. The
[extended soak](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/final-soak-summary.json)
has 129 successful health samples and no failed samples over roughly 64 minutes.

- Five logical KV retrieve failures on August 21 produced 20 rank errors,
  five scheduler request failures, and continued healthy serving—not five
  crashes. Internal logs add 4736 prefetched-object read errors in the same
  five bursts, with “exists but cannot read”/non-readlocked warnings consistent
  with a lease gap. This is not proof of every root cause. The adapter's
  “recompute” wording did not mean recovery: the deployed scheduler used
  failure policy `fail`. Do not blindly switch to recompute, especially where
  DMA ownership after timeout is unresolved.
- All 22 old HTTP 500s were legacy `/v1/completions` logprob failures on
  August 26: `IndexError` at `top_logprobs[i]` in
  `_create_completion_logprobs`, while health and generation continued.
  [Upstream PR #53722](https://github.com/vllm-project/vllm/pull/53722)
  guards a **KeyError**, not this array-length mismatch. No new legacy calls
  in the initial snapshot means zero new failures is not evidence of a fix.
- Both boots contain **1644 INFO-level `fallback proxy error` measurements**:
  numerical quantization-error metrics, not network/proxy/runtime errors.
  SymmMem native peer-atomic warnings do not mean all peer access is disabled.
- The old log's 45 preemption-warning records, nine late/duplicate callback
  warnings and queue/engine-stat gaps are not unique-victim or downtime
  counts. HTTP 200 during a gap establishes API response headers, not
  uninterrupted streaming progress. Absence of cache-eviction text is not
  absence of eviction.
- Old token-weighted MTP acceptance was 76.17% versus 70.89% in the short
  initial new window; this measures draft agreement, not intelligence or
  controlled speed. Old hot caches and 24-day mixed workloads cannot be
  compared to new JIT/cold-cache/long-prefill probes using aggregate interval
  throughput means.

Cache observability is incomplete: standalone 9090 metrics are disabled and
the current 8089 admin API exposes six informational GET routes, not
`/metrics`. The shared `metrics_api` module also defines mutating
`POST /metrics/reset`; future exposure must select **read-only GET** rather
than whitelist that module. Source supports session retention, but the
wrapper does not forward `LMCACHE_SESSION_TTL_SECONDS`; **5400 seconds is not
an active claim**. Its 600/300-second L1 write/read TTLs are different controls.

### Why a newer stack is not yet a dominating replacement

The [published stack inventory](https://github.com/local-inference-lab/blackwell-llm-docker/blob/master/README.md)
and [r34 integration lock](https://github.com/local-inference-lab/blackwell-llm-docker/blob/master/patches/releases/gilded-gnosis-v20-r34/b12x/integration.lock.json)
were reviewed September 8. GG's public 3.5bpw qualification is GLM-5.2
R7 K3/K4/K5, TP4/DCP1, C8, 65K—not full 5.3 at 520K. Infernal Invocation
r18 (August 18; CUDA 13.3/Torch 2.13) GPU-qualifies DS4 Flash and only
source-qualifies the listed GLM-5.2 profiles. Jovian's DS4 TP2 and Kimi TP16
qualification do not transfer to full GLM TP4/DCP4. LIL SGLang's documented
GLM NVFP4 quickstart uses eight GPUs; no matching full 5.3 mixed-EXL3
520K/C12 proof was found. Stock vLLM/SGLang/TRT-LLM/vanilla ExLlama are not
drop-in loaders for this exact mixed-tier artifact.

[B12X #133](https://github.com/local-inference-lab/b12x/pull/133) merged August 11
and is already in r34. Its TP8 two-island 12–20% claims are not TP4 gains;
reported TP4 gains are 1.67/2.27% at C4/C8 with additional staging memory.
Keep remote-push off rather than advertise a free 20%. September 6–7 open
[paired-push #339](https://github.com/local-inference-lab/b12x/pull/339),
[T12 #328](https://github.com/local-inference-lab/b12x/pull/328) and
[two-bit planner #327](https://github.com/local-inference-lab/b12x/pull/327)
are topology/model/layout-specific, not already merged GLM improvements.
SM100/SM103 kernel headlines and TP9 Kimi microseconds are not SM120 TP4
end-to-end evidence. A generic LMCache or NCCL update likewise does not
automatically preserve this appliance's adapter, leases and device-safety
contracts.

CPU offload is a different capacity/latency tradeoff, not demonstrated
domination. About 251 GiB host RAM minus the 125 GiB cache initial allocation
leaves about 126 GiB **before** OS, workers, pinned buffers, page cache and
loaders; this is not a safe available offload budget. Full checkpoint payload
already exceeds host RAM. The [generic offload contract](https://github.com/vllm-project/vllm/blob/main/vllm/config/offload.py)
does not qualify mixed EXL3 slabs in GG; UVA transfers weights each forward
and prefetch spends extra GPU memory. The
[official card bandwidth](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
1792 GB/s is local GDDR7 peak: summing four links into 7.168 TB/s does not
make a shared memory fabric or predict MoE decode rate. One NUMA node offers
no intersocket-placement windfall; idle host cores do not accelerate GPU
tensor kernels without a proven offload path.

### Retained decisions: do not recycle rejected knobs as free gains

The [August maintenance measurements](aibeast-r33-maintenance-results-2026-08-09.md)
supersede this document's early 280 W/clock/DMA hypotheses:

- **375 W** retained **98.6%** of 400 W's 262K prefill: 2154 versus 2184
  tok/s; 280 W delivered 1788. The 400 W hottest GPU briefly reached 92°C
  without an observed thermal-throttle indication. The retained cap is a
  measured energy/performance choice, not arbitrary underpowering.
- **16365 MHz memory clock** at 375 W delivered 2442 tok/s at 128K; 15165
  delivered 2341 (−4.2%), 13965 delivered 2305 (−5.6%). Downclocking has
  already been tested; do not resell it as an untried gain.
- **24 MiB lossless DMA crossover** remains selected. 6/24 MiB medians were
  2721/2678 at 8K, 2677/2641 at 32K, 2502/2590 at 64K, 2304/2398 at 128K
  and 2217/2175 at 262K; C4 TG 193.3/194.2 was parity. Live-traffic
  confounders remain; 6 MiB was not consistently better. The actual
  communicator uses 25,165,824 bytes even when the old wrapper banner says
  6 MB.
- **Async scheduling and MTP3** are not newly discovered free upgrades.
  Old async pressure qualification completed twelve 65,725-token inputs in
  393.23 seconds without error, preemption, OOM or restart; decode comparisons
  were traffic/acceptance-sensitive. MTP5 was previously slower than MTP3
  on an older 3.42 profile, not proof of the current 5.3 optimum.

### Ranked post-soak follow-ups (not yet executed)

1. **Reasoning-budget/history contract.** Replay the eight capped GPQA cases
   plus ordinary real agent controls with high/max and, where safely possible,
   131072/163840 output limits. Compare current history preservation against
   explicit `clear_thinking: true`, keeping system/tools and task facts fixed.
   Record correctness, loops, truncations, valid tools, token count and
   time-to-correct-result. Test low for bounded routing work, not as a claim
   of disabled reasoning. Do not change the global template speculatively.
2. **Matched task quality and quant/KV attribution.** Pair pinned 5.2/5.3
   on actual coding/agent tasks with hidden verifiers, identical harness,
   effort, budgets and repeated seeds. Add reference/exact-3.42 and FP8/NVFP4
   arms only with fit and safety evidence. Treat 3.25 + FP8 as a separately
   justified experiment, not fallback. Include 128K/300K/~500K multi-file
   synthesis and verifiable edits; needles alone do not measure that quality.
3. **Cache reliability/observability and exact-prefix reuse.** Correlate the
   five old failure bursts with leases, occupancy, retrieval completion and
   session lifecycle. Expose GET-only metrics safely, not the reset route;
   verify actual retention forwarding before promising a TTL. Distinguish
   cold token-zero nonce, GPU hit, pressure/DRAM restore and L2/restart arms,
   with hit tokens, tier bytes, TTFT and correct outputs. Keep namespaces
   separate; do not infer disk/restart qualification from DRAM-hit speed.
4. **Legacy logprob array alignment.** Reproduce the old completions
   `top_logprobs[i]` mismatch with the exact speculative/sampling contract,
   then investigate output-array alignment. The KeyError guard is not its
   fix; do not suppress an exception or claim the new model solved it.
5. **Isolated CUDA 13.2 Update 2 correctness rebuild.** The
   [runtime compiler audit](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/runtime-compiler-audit.json)
   found system NVCC/ptxas **13.2.78**, Triton's bundled ptxas **12.8.93**,
   NVRTC **13.2.78**, package cuBLAS **13.4.0.1**. NVIDIA's
   [Update 2 release notes](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html)
   fix nested-divergence register corruption and the Update 1 NVFP4 cuBLAS
   tensor-scale omission; fixed NVCC is 13.2.86 and cuBLAS 13.4.1.3.
   The later [live library map](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/loaded-compute-libraries.json)
   and [wheel-record hash check](../maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/cublas-record-check.json)
   identify the loaded cuBLAS as the unmodified **13.4.0.1** package. The
   NVFP4 scaling regression was **introduced in Update 1**; an older version
   number alone does not establish exposure to that regression. Keep this
   distinct from the compiler issue documented as present since CUDA 12.8.
   Inventory predating fixes is not proof an affected kernel executed or
   caused an incident. Audit **every actual compiler/assembler/JIT/library
   path**, not just system `nvcc`; isolate image and JIT namespaces, retain
   GG sources and all resource/quality settings, then requalify long context
   and soak. This is correctness work, not a predicted speedup or blind
   CUDA 13.3/Torch upgrade.
6. **Prefill cadence 1 versus 2.** [vLLM #546](https://github.com/local-inference-lab/vllm/pull/546)
   merged September 1 and is already backported; current 1 is unthrottled.
   Its public DCP1/DFlash 1-versus-8 experiment traded decode responsiveness
   against prompt throughput, not proof of native-MTP3 interval-2 gains.
   Use A/B/B/A with a position-aligned long prefill during matched decode;
   measure p50/p95/p99 ITL, prefill TTFT, goodput, queue progress and acceptance.
   A better decode tail at worse prefill TTFT is another frontier point.
7. **MTP depth and unused draft work.** Measure per-position acceptance,
   draft/verify duration and total latency at C1/C4/C12 before a bounded
   depth comparison; mean accepted length is not throughput or intelligence.
   [Issue #272](https://github.com/local-inference-lab/vllm/issues/272)
   (August 9, still open at review) proposes hydrating required draft KV but
   skipping unused heads/sampling/additional draft steps for unfinished
   prefills. Profile actual calls before implementing; final chunks, mixed
   row compaction, grammar, prompt logprobs and draft KV are correctness
   boundaries. No quoted 1–4% or upstream Qwen headline is a measured local
   gain. DFlash/DSpark need their own weight/graph/runtime qualification.

Only after those measurements justify it should exact metadata reuse
([#207](https://github.com/local-inference-lab/vllm/issues/207)) or caller-owned
fold scratch ([#134](https://github.com/local-inference-lab/b12x/issues/134))
be reconsidered. The >140K compaction copy protects a real cuBLAS tail/OOB
hazard; removing it without padded backing and correctness evidence is not
an optimization.

For any candidate, pin image, checkpoint, template/parser, sampler, KV format,
compiler and cache namespaces. Screen intermediate workspace boundaries before
near-limit shapes; separate cold/warm/tier-restored prefixes and mixed
prefill/decode workloads. Record output correctness, accepted tokens and
step durations, tail latency, queue/preemption behavior, worst-rank
power/clocks/temperature and allocated/reserved/physical VRAM. Promote only
after matched quality and reliability hold at the retained context/C12
contract. Until then, stronger original-model scores and a successful
deployment support the upgrade's rationale—not proven local quality
dominance, a free speed gain, or a measured Pareto optimum.
