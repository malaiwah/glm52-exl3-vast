# AIBeast next prefill maintenance plan — 2026-08-09

This plan turns the research in
[`glm52-prefill-optimization-research-2026-08-09.md`](glm52-prefill-optimization-research-2026-08-09.md)
into bounded experiments. It does not authorize a maintenance window by
itself.

## Priorities

1. Preserve correctness and the 520,192-token request envelope.
2. Improve cold long-prompt PP.
3. Preserve C1 through C12 TG and MTP acceptance.
4. Keep all runtime memory accounted and avoid late-allocation OOMs.
5. Improve mixed Hermes-agent goodput/latency after isolated PP is understood.

## Research issue ledger

The following items are banked upstream and should be updated with measured
evidence rather than duplicated:

- [local vLLM #203](https://github.com/local-inference-lab/vllm/issues/203):
  EXL3 runtime-memory lifecycle foundation used by the r31 baseline;
- [local vLLM #207](https://github.com/local-inference-lab/vllm/issues/207):
  repeated MLA/DCP mapping, slot geometry and host/launch cleanup;
- [local vLLM #272](https://github.com/local-inference-lab/vllm/issues/272):
  hydrate MTP KV while skipping discarded unfinished-prefill work;
- [local vLLM #273](https://github.com/local-inference-lab/vllm/issues/273):
  preserve specialized decode and prefill attention in mixed batches;
- [local vLLM #274](https://github.com/local-inference-lab/vllm/issues/274):
  fuse or selectively overlap GLM Q/indexer projection cones;
- [local vLLM #275](https://github.com/local-inference-lab/vllm/issues/275):
  eliminate no-op fused norm/RoPE planes and conditional B12X top-K preclear;
- [B12X #96](https://github.com/local-inference-lab/b12x/issues/96): early
  fused-kernel cache lookup and environment/host overhead;
- [B12X #134](https://github.com/local-inference-lab/b12x/issues/134): reserve
  and profile two-level paged-fold candidate scratch;
- [upstream vLLM #49171](https://github.com/vllm-project/vllm/pull/49171):
  target-only unfinished-prefill precedent.

[B12X #107](https://github.com/local-inference-lab/b12x/issues/107) is closed:
the native large-M mixed K3/K4 path shipped in #112/#117. Remaining grid work
is incremental and gets a successor issue only after a repeatable microkernel
gain and at least 1% server PP gain.

The exact r31 production baseline also composes and has GPU-qualified these
still-open heads:

- [B12X #130](https://github.com/local-inference-lab/b12x/pull/130)
  `9ead9eaa188c` on the #126/#228 route-prewarm foundation;
- [local vLLM #270](https://github.com/local-inference-lab/vllm/pull/270)
  `244d85a6fe99`, together with #258 `63b77c803149` and #271
  `310c3ac9718f`.

The qualification is composed-stack evidence at the exact heads, not isolated
performance attribution to either PR. Update their review state separately
from this plan; do not silently drop them when refreshing the base image.

CUDA 13.2 Update 2, DMA crossover, memory clocks, the mHC threshold and
LiteTopK are intentionally plan-only hypotheses. Open a new issue only after
the corresponding prerequisite or measurement gate demonstrates actionable
project work.

## Release gates

The final candidate must satisfy all of the following:

- health, `/v1/models`, alias and structured-output smoke tests pass;
- no restart, OOM, Xid, traceback, late compile or unexpected fallback;
- exact 520,192 request limit and at least 512K usable context;
- two-seed five-depth maximum-context needles and degeneration pass;
- KLD and Aider do not show a meaningful quality regression;
- C1/C4/C8/C12 TG geometric mean is no worse than 2% below control;
- normalized MTP MAL is not worse than control beyond run variance;
- at least 512 MiB ordinary free physical VRAM/rank after all lazy paths have
  been exercised;
- maximum-context stress is observed separately and its minimum physical
  margin is recorded;
- byte-pinned KV and identical graph/capture contracts are used between arms.

Before accepting results, freeze the quality controls:

- KLD: online-K6 + dynamic-NVFP4/FP8-RoPE baseline `0.089888`, 2,047
  teacher-forced positions, BF16 reference SHA-256
  `87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`.
  Exact-math candidates should stay within `0.001` absolute and report the
  exact delta; investigate any larger movement before promotion.
- Aider: commit `5dc9490`, whole edit, no repo map, temperature disabled,
  `max_tokens=32768`, `reasoning_effort=max`, two tries, four threads, all 225
  exercises. Retained healed NVFP4 result: PR1 48.4%, PR2 87.6%, zero
  malformed. The final combined candidate may not lose more than 2 percentage
  points on either rate without an explicit acceptance decision.
- SoftEval: copy the retained prompt set, evaluator revision, scoring command
  and raw baseline into the evidence root before testing; otherwise mark the
  gate unavailable rather than inventing a comparison.

If a candidate is close to a release gate, ask before rejecting it rather than
silently restoring the old profile.

## Evidence layout

Create one immutable root:

```text
/mnt/fast/build/<date>-glm52-prefill/
  manifest/
  compile-cache/
  baseline/
  dma/
  clocks/
  mhc/
  patches/
  mixed/
  quality/
  summary/
```

Each cell records image digest, vLLM/B12X commits and diff, checkpoint revision,
complete environment, launch argv, CUDA/NVCC/cuBLAS/driver versions, topology,
clocks/power, KV/graph profile, raw benchmark JSON, logs and a SHA-256 manifest.

Use counterbalanced A/B/B/A or A/B/A order for every isolated cell, with at
least five warmed repetitions and the median plus spread reported. Do not call
five separate measurements “medians.” Run on an alternate endpoint with agent
traffic excluded when possible; otherwise record request counts/timestamps and
label the cell traffic-contaminated rather than treating it as matched.

## Phase 0 — prerequisites and clean baseline

1. Prefer an image built with CUDA 13.2 Update 2 / NVCC 13.2.86. If that image
   is unavailable, qualify it as a separate correctness arm before combining
   performance changes.
2. Use a fresh deterministic compile-cache root for the Update 2 build; warm
   every kernel/graph route before final memory profiling.
3. Capture an executable rollback bundle before stopping production: immutable
   image ID/digest, checkpoint revision, `podman inspect`, complete environment
   and argv, mount/cache roots, resurrection command, and successful health +
   model-alias checks. Keep the previous image and compile cache until final
   promotion and soak complete.
4. Confirm exact profile: 3.42bpw + online K6, TP4/DCP4/MTP3, dynamic NVFP4 MLA
   + FP8 RoPE, 3,072 tokens, C12, 520,192 request limit, byte-pinned KV,
   graph/Trellis 48. LMCache uses configured 125/512 `GB` L1/L2 values; record
   the CLI-reported effective bytes rather than silently treating them as GiB.
5. Verify no stale `num_gpu_blocks_override` bypasses safe memory profiling.
6. Resolve and record the two-level-fold budget spelling. GG r31 logs the
   `SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB` alias, but the installed B12X
   source consumes `B12X_INDEXER_TWO_LEVEL_FOLD*`; verify the worker receives
   the latter and profile its reserved scratch before trusting a 64 MiB cap.
7. Record baseline at nominal current clocks/power and the current 24 MiB DMA
   crossover.
8. Run five warmed repetitions at 3K, 8K, 32K, 64K, 128K, 180K, 256K; one 500K+
   stress prefill; C1/C4/C8/C12 decode; and a 128K + C4 mixed test.
9. Capture one low-overhead DCGM trace. Reserve one 128K Nsight Systems trace
   for the best early arm rather than perturbing every benchmark.
10. Locate/install a version-compatible Nsight Compute CLI before Phase 2, or
   mark its counter-level cells unavailable and retain Nsight Systems + DCGM.
   The current host has Nsight Systems 2026.1.3 but no `ncu` binary.
11. Under sustained PP and sustained TG, remeasure every physical card's
    temperature, power and minimum/median SM clock. Re-derive the cold-to-hot
    order before retaining `CUDA_VISIBLE_DEVICES=2,1,0,3`; ordering is for
    determinism and rank-0 thermal placement, not a presumed throughput gain.
12. Preserve the retained r31 control evidence currently held only on local
    AIBeast flash (`/mnt/fast/build/r31-memory-stack-20260808/evidence/`, about
    2.9 MiB / 22 files): scan for credentials, copy to durable storage, and
    generate a SHA-256 manifest before any `/mnt/fast` cleanup.

## Phase 1 — safe configuration/hardware A/Bs

### Cell 1: PCIe DMA crossover

```text
A: VLLM_PCIE_DMA_MIN_BYTES=25165824   # current auto result, 24 MiB
B: VLLM_PCIE_DMA_MIN_BYTES=6291456    # retained r17 result, 6 MiB
```

Use A/B/B/A. Measure all prefill depths through 256K plus C1/C4/C12 and MAL.
Trace which transfers use B12X DMA versus fallback collectives. Promote B only
if the earlier 3–6% PP result repeats without a decode penalty.

### Cell 2: memory clock at fixed 280 W

With the winning DMA arm fixed, test:

```text
A: 14001 MHz
B: one supported intermediate memory clock
C: 16365 MHz (current)
```

Keep GPU power limit, fan policy, GPU order and ambient conditions constant.
Record minimum-rank SM clock and latency, not just averages. Run 8K/64K/128K/
256K PP and C1/C4/C12 TG. If nominal memory improves PP but hurts TG, select
the intermediate Pareto point. Restore the original memory clock immediately
after the matrix if no arm passes.

### Cell 3: real mHC threshold

```text
A: B12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS=4096
B: B12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS=3072
```

Precompile both. Run 3K/8K/64K/128K/256K PP, then TG/KV sanity. Reject if the
result is within noise or negative. Do not use a `SPARKINFER_...` spelling.

### Cell 4: query-split short-context crossover

Only if the first trace shows query-split overhead at short context, test
8K versus 16K for `VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS`. This is not a
long-context headline candidate. Keep two indexer shards and owner merge off.

### Explicitly excluded from Phase 1

- 4,096 max batched tokens;
- CKV prefetch depth 1;
- CKV gather 520K;
- owner merge;
- one indexer shard/full replication;
- lossy F8 transport;
- FP8 KV;
- DCP1 or DCP2;
- extra CUDA graph sizes;
- arbitrary NCCL channel counts.

They either fail the memory/context contract, were already rejected by matched
tests, or do not address the dominant B12X path.

## Phase 2 — profile the winning early arm

Capture uncached 128K and, if feasible, 256K Nsight Systems/Compute traces.
Attribute time and bytes to:

- dense and indexer A projections;
- q_b and indexer wq_b projections;
- fused indexer score/top-k;
- CKV gather/query split and PCIe collectives;
- sparse MLA attention;
- DCP project-before-merge compaction;
- EXL3 mixed K3/K4 MoE;
- mHC and dense residual paths;
- vocabulary projection/sampling on unfinished chunks;
- host/D2H synchronization and repeated tiny metadata kernels.

This attribution decides the order of Phase 3. Do not begin LiteTopK or a new
attention kernel because another machine reported a microbenchmark win.

## Phase 3 — code candidates

Each candidate gets a source-only unit/static test pass before GPU work, an
isolated A/B, and a peer review. Do not combine patches until each effect is
known.

### Patch A: unfinished-prefill logits/sampling skip

Backport/adapt upstream vLLM #49171 to the exact r31 MRv2 tree. Test:

- pure unfinished prefill;
- mixed unfinished prefill + decoder rows;
- prompt logprobs;
- MTP/speculative decoding;
- structured output;
- deterministic output identity.

Use local vLLM #272 as the contract. The upstream patch handles an
all-unfinished target batch even with MTP configured, but mixed rows still get
target bonus logits and the MTP drafter still creates proposals that the
scheduler discards. Preserve the MTP prefill forward/KV hydration, then test an
all-unfinished hydrate-only path that skips draft logits, sampling and the two
extra MTP3 draft forwards. Prioritize concurrent long prompts; expected GLM
value is 1-4%, not the upstream 9.87% headline.

### Patch B: MLA host/launch cleanup

Combine only the low-risk parts of B12X #96 and vLLM #207:

- cache probe before fused-kernel construction;
- hoisted environment parsing;
- host-derived DCP rank totals without `.tolist()` synchronization;
- vectorized C12 per-token lengths;
- batch-constant CKV causal/slot metadata computed once, then reused by 78
  layers.

Use Nsight to prove launch/sync removal. A repeatable 0.5% PP gain is enough for
promotion because the change is exact and memory-neutral.

Test the two higher-value duplicate-work removals first:

- reuse graph-stable global/local/page-table mappings across the 57 shared
  top-K target layers, while keeping MTP layers on the existing live-row path;
- precompute token-to-gathered-slot geometry once per batch instead of once per
  layer.

Instrument the two-level paged-indexer fold before changing it. Record selected
fold route, transient slab bytes, peak allocated/reserved memory and allocator
events. If the auto policy uses transient slabs on the production shape, add a
bounded caller-owned scratch arm and a streaming-carry control. Also count the
no-op fused norm/RoPE planes on no-indexer layers before implementing a
two-plane specialization. Use B12X #134 and local vLLM #275 as the respective
contracts.

### Patch C: targeted GLM multi-stream GEMM overlap

The current `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` is inert on GLM-5.2; use
local vLLM #274 as the contract. Implement four arms:

1. fuse skinny BF16 indexer-WK into fused-QKV-A;
2. overlap outer A projections only;
3. overlap inner q/indexer projections only;
4. combine fusion with the winning selective overlap.

Profile activation peak after every arm. Whole-cone overlap may add
96-128 MiB/rank and regress at M=3,072 under the 280 W cap. Reject a speed win
that consumes the maximum-context safety margin or degrades TG.

### Patch D: remove the >140K projection compaction safely

Use a padded-tail or ping-pong workspace; preserve the cuBLAS over-read guard.
Focus benchmarks at 128K, 180K, 256K and 500K+. Run Compute Sanitizer, monitor
Xids, and repeat all long-context quality gates. Promotion bar: at least 1.5%
repeatable gain above the 140K boundary.

Only after Patch D wins, optionally compare CKV gather 140K and 192K. Charge
the latter's approximately 45.8 MiB/rank cost explicitly.

### Patch E: incremental mixed-Trellis grid tuning

The native large-M specialization already shipped in B12X #112/#117 and r31;
B12X #107 is closed. Retain its one-grid block-8 decode and block-32 prefill.
Only test cooperative-grid right-sizing at 1/2, 3/4 and full cap after earlier
cells. Require a repeatable microkernel win and at least 1% full-server PP;
otherwise do not open another upstream issue.

### Patch F: exact LiteTopK-inspired indexer

This is conditional on profiling. First aggregate the existing scorer/top-k
kernel names and add default-off NVTX ranges for projections, gather, score,
top-k, DCP merge, attention and EXL3 MoE. Compute the LiteTopK-addressable
fraction `f`; model the ceiling as `1 / ((1-f) + f/1.265)`. Do not port unless
the conservative whole-prefill estimate is at least 3%. If it qualifies, retain
exact final selection and an ambiguity fallback, and require exact top-k index
identity on adversarial and recorded GLM traces.

### Patch G/H: trace-gated prior-art candidates

Do not implement these unless Phase 2 attributes material time to their exact
target:

- dense-MHA crossover: if short/moderate sparse attention dominates, prototype
  exact-equivalent thresholds 0/2K/4K/8K/16K where KV length does not exceed
  sparse top-k; preserve the sparse long-context path;
- dynamic-NVFP4 writer: only if the fused quantize/RoPE/write kernel is visible
  in whole-prefill time, test cooperative row staging, bulk stores, or more
  parallel RoPE quantization. Do not transplant an SM103 BF16/FP8-record kernel
  into the SM120 368-byte ABI.

### Staged combination

Combine isolated winners one at a time. After each addition, rerun activation
profiling, late-allocation routes, minimum physical VRAM, short correctness and
the affected PP/TG cells. Finish with A/B/A of the complete stack against the
pristine rollback baseline; never report the sum of isolated percentages as a
combined result.

## Phase 4 — mixed Hermes workload

For the best isolated candidate, run:

- 64K, 128K and 500K cold prefill with 1/4/8 active decoders;
- fixed 3,072 chunk budget;
- optional dynamic 3,072 idle / 2,560 or 2,048 decode-active policy;
- optional sequential decode-row/prefill-row split prototype under
  [local vLLM #273](https://github.com/local-inference-lab/vllm/issues/273):
  attention-only, decode first, non-aliased scratch/output, independent
  prefill-slice CKV-gather eligibility, and pure PP/TG parity before any stream
  overlap;
- EngineCore affinity off/on using one isolated physical core (not its SMT
  sibling). Expected value is jitter reduction and under 1%; reject if neutral;
- async scheduler off/on with MTP3 + LMCache and exact memory profiling. Cover
  cancellation, preemption, mixed prefill/decode, prompt-logprobs, structured
  outputs, target/proposal identity, MAL, LMCache completion-ID deduplication,
  and graph/eager memory peaks;
- LMCache CPU/GPU worker and prefetch-concurrency A/B using a reproducible
  repeated-prefix fixture plus a cold-prefix control. Record effective L1/L2
  bytes, host available RAM, swap-in/out, CPU/bandwidth use, PCIe/DCP
  contention, hit/restore latency, inflight operations and failures.

Report aggregate PP/TG, per-request TG, TTFT, ITL p50/p95/p99, MAL,
preemptions, power and prefix/offload hit rates. The objective is to keep decode
healthy while a Hermes agent ingests a large history.

## Phase 5 — quality and soak

On the final combined candidate:

1. deterministic short/long output parity;
2. KLD with the same windows/seeds as the retained 3.42bpw baseline;
3. SoftEval with the retained prompts, evaluator revision and scoring contract;
4. Aider smoke and retained representative trajectories;
5. structured output + tool calls + interleaved/preserved thinking;
6. two-seed five-depth 500K–520K needle suite;
7. degeneration/repetition checks;
8. deliberate C12 + near-maximum-context memory stress;
9. at least one hour of representative traffic, then production monitoring.

The production performance profile has vision disabled to retain the 520K
text envelope. This plan therefore does not qualify vision; record that scope
explicitly rather than implying full multimodal coverage.

If all gates pass, promote the combined winner. Otherwise restore the exact
documented r31 production posture—except a CUDA Update 2 rebuild may still be
promoted independently if it passes correctness and performance parity.

## Expected value by candidate

| Candidate | Honest expected range | Memory | TG risk | Confidence |
|---|---:|---:|---:|---|
| DMA 6 MiB | 0–6% PP; prior r17 evidence favors it | neutral | low/medium | high enough to test first |
| Memory-clock Pareto | -5% to +15% PP hypothesis | neutral | medium | medium-low until DCGM A/B |
| mHC 3,072 selector | about -2% to +2% whole PP | near-neutral | none | low |
| Skip unfinished target/MTP work | about 1–4% under concurrent long prefills | likely lower transient | low | medium/high |
| MLA host/launch cleanup | 0.3–0.7% PP | neutral | low | high |
| Profiled two-level fold scratch | 0–2% PP; primary value is deterministic memory | up to ~240 MiB made explicit/reused | none | high reliability value |
| Compact fused norm/RoPE grid | 0.5–3% PP; 0.2–2% TG | neutral | low with producer gate | medium |
| GLM GEMM overlap | unknown, plausibly low single digits | activation peak may rise | medium | medium |
| Remove >140K compaction | about 2–3% PP above cutoff | neutral/small tail | none | medium |
| Shared-top-K/slot-map reuse | about 1–5% PP (overlapping estimates) | ~24 KiB plus reused views | low with MTP exclusion | medium |
| Incremental mixed-Trellis grid tuning | about 0–3% E2E PP | neutral/small | low if dispatch retained | medium-low |
| Exact LiteTopK adaptation | potentially material if indexer-bound | implementation-dependent | none | research |
| Mixed attention split (#273) | 0–5% aggregate mixed-load; primarily ITL protection | neutral sequentially | low sequentially | medium |
| Async scheduling / EngineCore affinity | host-gap and jitter reduction; no cold-kernel claim | neutral after profiling | MTP/cache lifecycle | medium |
| LMCache prefetch tuning | repeated-prefix TTFT only; cold PP control should be flat | host RAM/PCIe contention | indirect | medium |
| Dense-MHA crossover / NVFP4 writer | trace-gated only | implementation-dependent | low if exact path retained | research |

These ranges overlap and must not be summed. The final combined gain is the
measured A/B/A result, not the sum of individual best cases.
