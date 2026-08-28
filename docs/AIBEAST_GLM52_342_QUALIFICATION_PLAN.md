# AIBeast GLM-5.2 EXL3-TR3-3.42bpw qualification plan

Prepared 2026-08-06 for the next authorized AIBeast maintenance window.

## Execution record — 2026-08-06

The maintenance window began after the 3.42 snapshot and all 83 SHA256-addressed
LFS objects were verified. AIBeast uses driver `595.71.05`, CUDA 13.2, four
280 W RTX PRO 6000 Blackwell GPUs, and deterministic physical order
`CUDA_VISIBLE_DEVICES=2,1,0,3` for these cells.

### Completed controls

- r26 + 3.36 + online K6 + dynamic NVFP4/FP8-RoPE repeated the established
  KLD exactly: `0.08250703559901781` over 2,047 positions. The zero standard
  deviation is structural because this repeat still uses one fixed 2,048-token
  window.
- The isolated r26/3.36 control exposed exactly 524,288 KV tokens and completed
  the matched benchmark with no errors or preemptions: PP 2,457 / 2,348 / 2,189
  tok/s at 3K / 32K / 128K; aggregate decode 71.1 / 140.2 / 181.2 tok/s at
  C1 / C4 / C8.
- Pristine r28 with the same 3.36 serving shape also exposed 524,288 KV tokens.
  It explicitly included 233.25 MiB/GPU of B12X sparse-DCP transient memory
  and CUDA-graph peaks in its accounting. PP was 2,447 / 2,380 / 2,227 tok/s
  (-0.4% / +1.4% / +1.7% versus r26). Aggregate decode was 63.1 / 127.9 /
  194.9 tok/s; this stochastic probabilistic-MTP sample needs a final repeat
  before attributing its lower C1/C4 or higher C8 to the release.
- r28 control power averaged 1,018 W aggregate (254.5 W/GPU), p95 1,122 W,
  and briefly reported 1,142 W aggregate at the one-second sample boundary.

### First 3.42 capacity result

The first r28/3.42 boot used natural KV profiling, GMU 0.95, MTP3,
batch/prefill 3,072, C8, online K6, dynamic NVFP4/FP8-RoPE, and fresh compile,
online-quant, state, and LMCache namespaces. The immutable NFS checkpoint was
read-only. Cold loading and K6 encoding took 3,815 seconds; mixed-Trellis
hydration added about 25 minutes. The profile exposed 524,800 logical KV
tokens and left about 2.3 GiB/GPU immediately after startup.

That natural pool did **not** pass the workload gate. A cold 131,070-token
prefill with 3,072-token chunks failed on the first chunk when Trellis requested
a 48 MiB temporary and only 32.81 MiB was physically free. The EngineCore died
with a CUDA OOM. The failed container, complete logs, scheduler dump, and
partial benchmark evidence are retained as the negative result. Its successful
3K and 32K probes measured 2,457 and 2,356 tok/s respectively.

The next cell pins 2,032 blocks and aligns `MAX_MODEL_LEN=520192`, preserving
the top of the requested 512K-520K range while returning eighteen KV blocks of
execution headroom. It reuses the same deterministic K6/JIT caches and uses a
90-minute health start period; the 45-minute default falsely marked the valid
cold load unhealthy. If this cell passes, an exact 524,288-token envelope will
still be challenged at a 2,048-token prefill capacity before promotion.

### Qualified 520,192-token control

The corrected cell booted healthy with exactly 2,032 blocks / 520,192 logical
KV tokens. Warm model loading reused every online-K6 artifact and completed in
699.7 seconds (324.9 seconds for the 81 safetensor shards, followed by mixed
Trellis hydration), versus roughly 89 minutes for the cold encode plus
hydration. No cache corruption, low-throughput signature, stale lock, or JIT
rebuild was observed.

The exact 131,071-token workload that killed the natural pool then passed at
2,214.6 tok/s. The fresh-salt matched matrix completed without errors or
preemptions:

| Workload | Result |
|---|---:|
| PP 3K / 32K / 128K | 2,367 / 2,263 / 2,137 tok/s |
| Aggregate decode C1 / C4 / C8 | 60.5 / 149.4 / 173.5 tok/s |
| MTP3 MAL C1 / C4 / C8 | 1.95 / 2.48 / 2.43 |
| Draft-token acceptance C1 / C4 / C8 | 31.5% / 49.3% / 47.6% |
| Power average / p95 / max | 1,059 / 1,122 / 1,136 W aggregate |

All release-gating API features passed: non-thinking and thinking chat,
reasoning visibility, streaming usage, preserved-thinking multi-turn,
structured JSON with and without thinking, tool calls and tool-result round
trip, tokenization, and both `GLM-5.2` and `local-primary` aliases. Vision is
disabled for this text checkpoint.

The salted needle matrix recovered all five facts at depths 1%, 25%, 50%, 75%,
and 99% for every prompt size: 8K, 32K, 96K, 128K, 162K, 256K, 384K, 507,904,
and 516,096 tokens. The last case plus its 4,096-token generation reserve
proves the complete 520,192-token serving envelope. This is 45/45 successful
retrievals with lossless PCIe transport, zero OOMs, and zero preemptions.

Matched single-window KLD measurements use the same 2,048-token BF16-logit
reference (`87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`)
and all 2,047 teacher-forced positions:

| 3.42 serving form | Mean KLD | Model VRAM/GPU (offline MTP0/eager) |
|---|---:|---:|
| Native checkpoint + dynamic NVFP4/FP8-RoPE KV | 0.082039 | 83.73 GiB |
| Online K6 + dynamic NVFP4/FP8-RoPE KV | 0.089888 | 79.13 GiB |
| Online K6 + FP8/BF16-RoPE KV | 0.077949 | 79.13 GiB |

Online K6 therefore recovers 4.60 GiB/GPU in this isolated model-load shape,
at a measured +0.007849 absolute KLD cost. For context, the matched retained
3.25 native result was 0.091256 and the repeated 3.36+K6 control was 0.082507.
The reference bundle contains one window, so no non-zero sampling standard
deviation can be reported; this is a limitation of the reference, not evidence
that the model has zero run-to-run variance.

Holding online K6 fixed, FP8 KV improved mean KLD by 0.011939 versus dynamic
NVFP4. This agrees directionally with the healed Aider A/B, but the 656-byte
FP8 record reduced the practical context envelope to about 295K versus 520K
for the 368-byte dynamic-NVFP4 record. NVFP4 remains the production choice for
Hermes Agents; FP8 is the higher-fidelity shorter-context option.

The owner explicitly accepted promotion if the 3.42 candidate misses the
aspirational PP/TG gates, provided the correctness, 520,192-token envelope,
stability, and quality gates remain satisfied. PP/TG are therefore optimization
targets for the remaining low-risk A/Bs rather than release blockers.

## Decision target

Decide whether `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` should replace the
current 3.36bpw checkpoint as the flagship production profile while preserving
all of these non-negotiable requirements:

- four RTX PRO 6000 Blackwell GPUs, TP4/DCP4;
- a 524,288-token model envelope, demonstrated as at least a 520,192-token
  prompt plus 4,096 generated tokens;
- dynamic-token NVFP4 MLA KV with FP8 RoPE (368-byte records);
- at least eight small concurrent requests;
- approximately 2,400 tok/s or better long-prefill throughput;
- approximately 100 tok/s or better C1 decode;
- no OOM under the C8 and maximum-context gates;
- no output degeneration and successful salted needle retrieval through the
  maximum supported prompt;
- LMCache with 125 GB DRAM and a bounded 512 GB NVMe tier;
- both `GLM-5.2` and `local-primary` served aliases.

If a candidate is close to a gate, especially 512K versus 520K usable input or
roughly 95 versus 100 tok/s C1, stop and ask before rejecting it or rolling
production back.

## Pinned artifacts

### Candidate checkpoint

- Repository: `willfalco/GLM-5.2-EXL3-TR3-3.42bpw`
- Downloaded revision: `a350292cb2038f2c31732569a711a89e5d72fd46`
- Upstream r28 validation revision:
  `ae68c65947efa90bea37308e15421872f124c46d`
- The only later commit is a README update. The complete safetensor
  path/size/LFS-OID manifests at the two revisions hash identically to
  `f48e32cd765a67525076227c36fd57ce1de7c00af6f73627893e96438fe3de67`.
- Current Hugging Face stored size: 351,564,534,963 bytes (~327.42 GiB).
- Layout: `shared_h_v1`, mixed K3/K4; layer 3 is 206 K3 + 50 K4,
  layers 4-77 are 148 K3 + 108 K4, and calibrated layer 78 is uniform K3.

### Runtime

- Current latest published v20 image as of 2026-08-06:
  `voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28`
- Registry digest:
  `sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0`
- r28 is the minimum image for the combined `shared_h_v1` and mixed-tier
  contract. r26 can load those features separately but cannot load this model.
- Recheck the upstream release page, build repository, and registry immediately
  before the window. If a newer release exists, inspect its complete source and
  configuration delta before substituting it; retain r28 as the known control.

### Controls

- Checkpoint control:
  `willfalco/GLM-5.2-EXL3-TR3-3.36bpw@8d9aa923a17502675ca23737349b67f2e66bb69d`
- Runtime control: preserve the exact current r26 3.36/NVFP4 production
  container and its inspect output so it can be resurrected unchanged.
- The temporary FP8-KV Aider experiment is not the production baseline. Finish
  and archive it before the maintenance window, then restore/capture the proven
  3.36 dynamic-NVFP4 state.

## Pre-window preparation (no serving interruption)

1. Finish and archive both Aider FP8-KV passes and their paired per-exercise
   outcomes.
2. Complete the pinned 3.42 download into
   `/mnt/vault/llm/huggingface/models--willfalco--GLM-5.2-EXL3-TR3-3.42bpw`.
3. Verify the snapshot and all safetensors against the repository manifest.
4. Pull r28 by immutable tag and verify its registry digest.
5. Allocate separate, deterministic paths for:
   - r28/3.36 and r28/3.42 online-K6 caches;
   - compile/JIT caches keyed by image source fingerprint, checkpoint revision,
     TP/DCP, K6 bits, and graph shape;
   - fresh LMCache L2 namespaces for every checkpoint/runtime combination.
6. Confirm at least 500 GB free on `/mnt/fast`, NFS responsiveness, healthy
   `cachefilesd`, and no dirty/writeback backlog likely to distort cold load.
7. Save container inspect, image digest, environment, launch command,
   `nvidia-smi -q`, driver/CUDA versions, clock offsets, power limits,
   `nvidia-smi topo -m`, PCIe link status, kernel messages, and current API
   health.

## Maintenance execution

### 1. Establish controls

1. Drain new work and let active requests finish.
2. Preserve the stopped production container rather than deleting it.
3. Run the standard short benchmark and repeat the matched KLD baseline on the
   current 3.36/NVFP4/K6 profile.
4. Boot 3.36 on pristine r28 with the same serving shape. This separates an
   r26-to-r28 runtime delta from the 3.36-to-3.42 checkpoint delta.
5. Record cold and warm startup, model VRAM, scratch/graph reservations, KV
   capacity, free VRAM, PP/TG/MAL, and correctness for both controls.

### 2. First 3.42 boot: safe capacity baseline

Start from the upstream-tested posture, adjusted only for our production
requirements:

```text
TP=4
DCP=4
MTP=3
ONLINE_QUANT=exl3-b6
KV_CACHE_DTYPE=nvfp4_ds_mla
KV_FP8_ROPE=1
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
VLLM_NVFP4_MLA_SCALES_FILE=
MAX_NUM_SEQS=8
GRAPH=32
MAX_BATCHED_TOKENS=3072
MAX_MODEL_LEN=524288
GPU_MEMORY_UTILIZATION=0.95
LMCACHE_L1_GB=125
LMCACHE_L2_GB=512
F8_DMA=0
```

Do not initially carry over the 3.36 `GPU_BLOCKS_OVERRIDE=2048`; allow r28 to
profile the new model honestly. Record the natural block count and use an
explicit override only after the measured safe capacity is known. Require a
fresh LMCache namespace. The first K6 encode is a cold-start qualification;
the second boot must reuse the deterministic on-disk K6 artifacts and must not
show corruption, abnormally low PP/TG, or a rebuild.

Required loader evidence:

- exact checkpoint revision and all shard hashes;
- `shared_h_v1` selected without expanding rotations;
- 206/50 for layer 3, 148/108 for layers 4-77, uniform K3 MTP-78;
- about 681 MiB/GPU shared-H saving with MTP loaded;
- no eager-parity `m=3` path and no MTP/Trellis capture warning;
- full and piecewise CUDA graphs captured;
- lossless B12X/SparkInfer transport actually selected;
- no unknown environment-variable warnings, stale lock waits, rank exits,
  Xid events, allocator retries, or hidden fallback.

### 3. Resolve policy and tune one variable per boot

The r28 helper policy and Josh's final 3.42 profile disagree in a few places.
Capture the dry-run expansion first, then A/B only the live variables below.
Use A/B/A for any result large enough to influence promotion.

| Candidate | A/B | Why it is in the matrix | Promotion rule |
|---|---|---|---|
| `SPARKINFER_MLA_SM120_NUM_SPLITS` | auto/4 vs 8 | Shared-H mixed layers reportedly under-split; 8 gave +2.7 C1 | Keep only if repeatable and no PP/KV loss |
| `SPARKINFER_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS` | 4096 vs 3072 | Lets the 3072-token batch use the fast TF32/TMA geometry | Keep if PP improves without correctness loss |
| `SPARKINFER_PCIE_DCP_THREADS` / `SPARKINFER_PCIE_DCP_BLOCK_LIMIT` | auto vs 512/16 | Reported small C1 gain on DCP4 | Keep only if repeatable; all ranks must agree |
| `VLLM_EXL3_TRELLIS_MAX_M` | 32 vs 48 | Reported ~3 C1 gain on mixed checkpoints | Must fit C8 graphs and 524,288 context |
| `VLLM_DCP_INDEXER_SHARDS` | helper auto vs 0, then 2 only if needed | Notes report shards=2 gives +6.5% PP but -2.7% C1 and -8% KV | Prefer 0 if it meets PP; shards=2 cannot be promoted if usable context drops below 512K |
| MTP depth | 3 control; 5 spot-check only | 3.42 notes find MTP3 optimal and MTP5 ~8.5% slower | Promote MTP3 unless AIBeast real-workload MAL disproves it |
| `MAX_BATCHED_TOKENS` | 3072 control; 4096 estimate/spot-check | 4096 may improve PP but reportedly costs ~37K KV | Do not trade away the 512K-520K envelope |
| `B12X_PCIE_DMA_FP8` / `F8_DMA` | lossless control; lossy opt-in only | Josh reports ~4% PP, but lossy transport may affect retrieval | Never promote without matched KLD, degeneration, and full needle gates |

Keep `CKV_GATHER_MAX_TOKENS=140000`, query split, owner exchange off, prefetch
depth 1, two-level fold off, and spec-extend-as-decode enabled unless the r28
dry-run shows that a setting is ignored or contradicts its measured TP4/DCP4
policy. Explicitly verify each setting in runtime logs; an exported variable is
not evidence that its code path ran.

### 4. Performance and memory matrix

For r28/3.36 and every final r28/3.42 candidate, collect the same harness JSON,
raw logs, timestamps, and power/thermal trace:

- uncached PP at 3K, 8K, 32K, 64K, 128K, and 256K;
- TG at C1/C2/C4/C8 for zero, 32K, and 128K context;
- one C16 throughput/correctness spot-check after C8 passes (production can
  remain capped at eight);
- MTP acceptance length/rate by concurrency and real-agent traffic;
- logical and physical KV blocks/tokens, model/scratch/graph/KV VRAM, free
  VRAM at idle and peak, LMCache worker overhead, OOM/preemption counts;
- per-GPU watts, clocks, temperature, throttling reason, PCIe traffic, and
  rank placement throughout prefill and decode.

After inference heats the cards, reassess cold-to-hot physical GPU ordering.
Compare repeated thermal/power traces, then bind rank 0 to the consistently
coldest card. Do not infer the order from idle temperature alone.

### 5. Correctness and quality gates

1. Repeat KLD for the current 3.36 baseline and 3.42 using the same BF16-logit
   reference, prompt windows, teacher-forced positions, KV mode, K6 policy,
   and run count. Add a 3.42 checkpoint-only/no-K6 cell only if it can be run
   without compromising the serving tests. Never compare KLD numbers obtained
   from different reference captures or window policies.
2. Run salted, uncached multi-needle retrieval at 8K, 32K, 96K, 128K, 162K,
   256K, 384K, 507,904, and 520,192 prompt tokens. Require all planted facts,
   not merely a coherent answer. The published “clean through 162K” result is
   not sufficient for our Hermes Agents requirement.
3. Run long-generation degeneration/repetition tests cold and warm, including
   after compile-cache and LMCache reuse.
4. Run deterministic repeats, standard chat, reasoning, preserve-thinking
   multi-turn, tool calls, structured JSON/XGrammar, streaming cancellation,
   and the `local-primary` alias.
5. Run a selected Aider failure/regression subset during maintenance. A full
   225-case Aider pass may run after promotion at concurrency four so it does
   not lengthen the service outage.

### 6. LMCache and maximum envelope

- Prove 125 GB DRAM plus a bounded 512 GB `/mnt/fast` NVMe tier.
- Verify cold miss, DRAM hit, NVMe hit, eviction at the size bound, restart
  reuse, hit-rate/bytes/latency metrics, and no cross-checkpoint cache reuse.
- Run the exact 520,192 prompt + 4,096 generation envelope with LMCache
  enabled, then C8 mixed small requests. Require no OOM, Xid, rank loss, or
  unrecovered request error.
- Soak briefly under real agent traffic and inspect every vLLM/SparkInfer/
  LMCache warning before promotion.

## Promotion and rollback

Promote 3.42 only after the exact final configuration passes all release gates.
Production must return to port 8000 with the two aliases, max sequences 8,
dynamic NVFP4 KV, FP8 RoPE, 524,288 model length, and the qualified LMCache
tiers. Preserve the stopped 3.36 control container, configuration, K6 cache,
and LMCache namespace until the 3.42 soak is complete.

If 3.42 fails a hard gate, restore the preserved 3.36 NVFP4 container without
rebuilding or changing its cache paths. If it is close to a stated threshold,
ask before deciding not to promote.

## Evidence bundle

Store, checksum, and summarize:

- image and checkpoint identities plus tensor manifest;
- full expanded environment and launch command for every accepted cell;
- cold/warm startup logs and runtime warning audit;
- benchmark JSON and raw request outputs;
- KLD inputs/results and per-position summary;
- needle prompts, salts, expected facts, and outputs;
- memory/KV tables, LMCache metrics, power/thermal traces, and topology;
- final promotion or rollback command and post-start health/traffic evidence.

## Sources

- [Gilded Gnosis v20 GLM-5.2 guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md)
- [r28 build repository](https://github.com/local-inference-lab/blackwell-llm-docker)
- [3.42bpw checkpoint](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.42bpw)
