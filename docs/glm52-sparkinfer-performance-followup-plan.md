# Historical GLM-5.2 r26 SparkInfer performance follow-up plan

Prepared 2026-08-04 for the next explicit AIBeast maintenance window. This
file records the performance A/B tests that survived the source audit of
SparkInfer `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`. It is a plan, not an
authorization to interrupt the production endpoint and not evidence that any
candidate is faster.

**Historical scope, preserved 2026-09-08:** the source audit, control settings,
and references to the "live" or "current" service below describe the August 4
r26 deployment, not today's appliance. This is retained research, not the next
authorized maintenance plan. Later work changed the checkpoint, shared-H
runtime, context envelope, and split candidates; consult the
[r28 qualification plan](AIBEAST_GLM52_342_QUALIFICATION_PLAN.md),
[August 9 maintenance plan](aibeast-next-prefill-maintenance-plan-2026-08-09.md),
and [September 8 AIBeast qualification](../TEST_RESULTS.md#aibeast-parity-cutover-and-accepted-soak-2026-09-08)
before reusing any hypothesis or control. These later records do not establish
that every experiment proposed here was completed.

The audit found one immediately testable GLM knob, two conditional MLA-kernel
experiments, and one conditional collective experiment. Other reported
opportunities were stale, already mitigated, outside the GLM path, or based on
the wrong 228 KiB shared-memory budget. RTX PRO 6000 Blackwell is compute
capability 12.x and provides 100 KiB of shared memory per SM (99 KiB per
block).

The review was extended with second- and third-pass findings after this plan
was first written. Those additions were checked against the installed r26
source, image labels, live worker environment, startup log, and PCIe
calibration artifact. The reconciled result is:

- the live policy is `query-split=1`, `owner-merge=0`, `indexer-shards=2`, and
  `prefetch-depth=0`; the review's `owner-merge=1` / `indexer-shards=0` claim is
  stale or taken from a different script;
- shared-expert streaming really is disabled in the production profile;
- the r26 integration contains SparkInfer PR #117, but its installed mixed
  Trellis ABI is version 5 and does **not** accept `broadcast_suh` or
  `broadcast_svh`. The current service is safe because the mixed path validates
  expert-sized rotation tables and successfully booted with those tables; the
  review's claim that PR #117 added ABI 6 broadcast support is false. A future
  shared-H mixed-Trellis implementation would still need an explicit patch and
  full quality qualification;
- NVFP4 is genuinely excluded from the native GLM H8 decode path. Supporting
  it is not a one-line gate removal: the present H8 branch calls GLM
  block-scaled math routines, so an NVFP4-aware H8 math path and SMEM/resource
  validation are required;
- dynamic-token NVFP4 S6 reloads per-token outer scales across V work. Hoisting
  may help, but it trades loads for registers and must prove itself first in a
  focused kernel benchmark;
- the vLLM integration already passes `attn_norm.variance_epsilon` and
  `ffn_norm.variance_epsilon` into the B12X mHC calls. The reported live
  `norm_eps=0` risk is not present;
- the saved PCIe calibration was incorrectly run with the 656-byte FP8 CKV
  default, while production dynamic-token NVFP4 records are 368 bytes. This
  overstates CKV wire volume by about 78%. It does not alter serving while
  prefetch is explicitly disabled, but the calibration must be corrected
  before reconsidering prefetch.

## Control configuration

Capture the live container inspect, image ID/digest, composed vLLM and
SparkInfer revisions, checkpoint revision, resolved configuration, driver,
CUDA version, GPU clocks/power limits, topology, and cache namespaces again at
the start of the window. The expected control is the qualified r26 production
shape:

```text
checkpoint  willfalco/GLM-5.2-EXL3-TR3-3.36bpw@8d9aa923a17502675ca23737349b67f2e66bb69d
GPU order   CUDA_VISIBLE_DEVICES=2,1,0,3
hardware    4x RTX PRO 6000 Blackwell, 280 W/card
topology    TP4 / DCP4
weights     EXL3 mixed K3/K4 + ONLINE_QUANT=exl3-b6
KV          dynamic-token NVFP4 MLA + FP8 RoPE
speculation native MTP3, probabilistic proposals, standard rejection
scheduler   MAX_NUM_SEQS=8 / MAX_NUM_BATCHED_TOKENS=3072
arena       VLLM_EXL3_PREFILL_CAPACITY=3072
graphs      GRAPH=32 / VLLM_EXL3_TRELLIS_MAX_M=32
context     MAX_MODEL_LEN=524288 / GPU_BLOCKS_OVERRIDE=2048
memory      GPU_MEMORY_UTILIZATION=0.957
offload     LMCache 125 GiB DRAM + bounded 512 GiB local NVMe
MLA splits  SPARKINFER_MLA_SM120_NUM_SPLITS=0 (shape-aware heuristic)
```

Do not change the image, checkpoint, GPU order, power cap, KV format, MTP
method, scheduler, active-KV pin, LMCache posture, or any other runtime knob
inside an A/B series. If a newer release is being qualified in the same
window, first establish its own unchanged control and perform this matrix
entirely on that release.

## Measurement contract

For every full-stack arm:

1. Use a deterministic, arm-specific compile-cache namespace. Warm all
   required CUDA graphs and CuTe/Triton objects before recording results.
   Record cold compile time separately; never mix first-use compilation into
   steady-state PP/TG.
2. Use the standard voipmonitor LLM decode benchmark and preserve its complete
   output. Use identical prompts, context lengths, output limits, sampling
   parameters, and seeds in every arm.
3. Run at least three recorded samples per cell. Report each sample, median,
   arithmetic mean, and spread; do not report only the best run.
4. Measure cold, unique-prefix PP at 3K, 32K, and 128K. Confirm GPU and external
   prefix-cache hits are zero for the measured prompt tokens.
5. Measure aggregate TG at C1, C2, C4, and C8 at zero/short, 32K, and 128K
   resident contexts where the harness supports it. Preserve per-request
   TTFT/TPOT and request failures/preemptions.
6. Record MTP acceptance, mean acceptance length, active GPU-KV, idle and peak
   free VRAM, GPU power/temperature/clocks, and post-ready compile misses.
7. Use the production MTP3 profile for the primary comparison. If acceptance
   variance changes the ordering or intervals overlap, repeat only the
   disputed cells at MTP0 to isolate target-model decode.
8. Alternate back to a control after the candidates so thermal drift, agent
   traffic, or a slowly changing host does not become the apparent gain.

Production traffic must not be mixed into the headline benchmark. Real agent
traffic is useful only after the synthetic winner has been selected and the
endpoint is deliberately returned to port 8000 for the soak gate.

## Experiment A — MLA decode split count

This is the only no-code performance candidate from the audit. At GLM TP4 C1,
the current heuristic selects 32 splits. Lower values reduce partial-result
and merge traffic, but also reduce attention CTAs and make each CTA process
more chunks. Faster execution is therefore a hypothesis, not a consequence of
the lower split count.

Run in this order:

| arm | `SPARKINFER_MLA_SM120_NUM_SPLITS` | purpose |
|---|---:|---|
| A0 | `0` | unchanged shape-aware production control |
| A1 | `16` | intermediate merge/parallelism trade-off |
| A2 | `8` | aggressive merge reduction |
| A3 | `0` | repeated control after candidates |

The target verification row count changes with MTP3. For the current
2,048-candidate, one-head-block GLM geometry on 188 SMs, the source heuristic
is expected to choose the following plans:

| request concurrency | target rows (`C * (1 + MTP3)`) | expected splits |
|---:|---:|---:|
| C1 | 4 | 32 |
| C2 | 8 | 16 |
| C4 | 16 | 11 |
| C8 | 32 | 11 |

These values must be confirmed from `LAST_DECODE_PLAN` or equivalent runtime
telemetry in every arm. A fixed value of 16 or 8 therefore changes the
high-concurrency cells too; there is no expected C8-equals-control shortcut.

Promotion threshold:

- at least 3% repeatable median TG improvement in the workload-relevant cells;
- no greater than 2% PP regression at any measured depth;
- no greater than 2% TG regression at C4 or C8;
- no new request failures, preemptions, OOMs, degeneration, compile misses, or
  material MTP-acceptance regression;
- active KV remains exactly 524,288 tokens and free-VRAM movement remains
  within normal boot variance.

Treat a result below 2% as noise and retain heuristic `0`. If a candidate is
close to these gates or trades C1 against C8 in a way that is not clearly
dominated, ask the user before declining to promote it.

## Experiment B — native GLM H8 for dynamic NVFP4 (development prerequisite)

The production dynamic-token NVFP4 path is excluded from `native_glm_h8` and
uses the generic eight-math-warp kernel. The native H8 path uses four math
warps, swapped-AB QK, and packed high/low-row PV, making this the largest new
decode opportunity in the updated review.

Do **not** benchmark a patch that merely removes the `ScaleFormat.NVFP4_E4M3`
gate. The existing native branch invokes block-scaled GLM math and is not an
NVFP4 implementation. Before a full model boot, an isolated successor must:

- add scale-format-aware NVFP4 H8 QK/PV math while preserving the 368-byte
  dynamic-token record and FP8-RoPE contract;
- prove its SMEM layout is below the real 99 KiB per-block limit;
- retain compiled resource reports for the generic and H8 kernels, including
  threads, registers, local spills, SMEM, and occupancy;
- pass focused numerical comparisons for static and per-token outer scales,
  masked/tail candidates, all split counts used by Experiment A, and repeated
  CUDA-graph replay;
- demonstrate at least a 5% median focused decode-kernel improvement with no
  numerical drift or spill regression.

Only after those gates pass should pristine and patched sources receive the
full measurement contract. Require at least 3% repeatable end-to-end TG gain
at C1 and no regression beyond 2% at C4/C8 or in PP before considering the
additional kernel complexity for promotion.

## Experiment C — dynamic-NVFP4 S6 outer-scale reuse (conditional)

Source inspection confirmed that dynamic-token NVFP4 S6 loads a candidate's
outer scale again as V dimensions are processed. The compiler/SMEM cache may
already make this cheap, while explicit reuse can increase register pressure
or spill. Do not consume a full model restart on this idea until a patch exists
and a focused kernel benchmark demonstrates a gain.

Preconditions:

- patch is isolated to V-scale value reuse and has CPU/static tests plus the
  existing MLA numerical tests;
- compiled resource reports are retained for baseline and candidate
  (registers, spills, static/dynamic shared memory, occupancy);
- the focused decode-kernel benchmark uses the production generic-NVFP4 S6
  shapes and
  demonstrates at least a 3% median kernel-time improvement without numerical
  drift.

If those gates pass, compare pristine source and the patch with the complete
Experiment A measurement contract, using the selected split policy from A.
Require at least 1% repeatable end-to-end TG improvement to justify the added
kernel complexity. Revert immediately on spills, lower occupancy, output
drift, or a regression outside noise.

## Experiment D — BF16 DMA RS/AG pipelining (conditional)

The lossless PCIe ring currently uses one copy stream across reduce-scatter
and all-gather. Separate streams might pipeline completed shards, but both
phases share dependencies and the same PCIe link/copy engines. This is a
long-prefill hypothesis with low expected decode impact.

Before a full model boot, compare pristine and patched collectives on the
exact TP4 topology and GPU order:

- payloads: 24, 32, 64, 128, and 256 MiB;
- warm iterations: enough to report median and p95 latency plus effective
  bandwidth;
- correctness: bitwise-equivalent BF16 all-reduce output, including repeated
  and back-to-back operations;
- evidence: stream/event timeline, power, failures, and the selected
  B12X/NCCL crossover.

Proceed to a full GLM A/B only if relevant 64--256 MiB cells improve by at
least 5% without instability. In the full stack, require at least 2% repeated
PP improvement at 32K or 128K, no TG regression beyond 2%, and no correctness
or lifecycle regression. Do not combine this patch with Experiments B or C
until each has independent attribution.

## Calibration prerequisite — use the actual NVFP4 CKV record size

Before testing CKV prefetch again, fix or override the PCIe overlap probe to
use `ckv_record_bytes=368` for `nvfp4_ds_mla` with dynamic-token scale and FP8
RoPE. Use a new calibration fingerprint/cache entry; never reinterpret the old
656-byte artifact as an NVFP4 result.

Re-run the calibration at 8K, 64K, 128K, and at least one depth representative
of the 512K serving envelope. Preserve isolated CKV, isolated TP, concurrent
wall time, wire bytes, effective bandwidth, and the accept/reject decision.
Keep production `prefetch-depth=0` unless the corrected calibration and a full
uncached-prefill A/B show a repeatable benefit without reducing stability or
KV capacity.

## Final release gates for any promoted combination

Only the isolated winner(s) may be combined. Re-run a control if stacking does
not approximately preserve the independently measured direction.

The final candidate must pass:

- the complete authenticated API/feature suite, including thinking,
  preserved multi-turn thinking, streaming, structured output, tools, and
  `local-primary`;
- 5/5 needles at 1/25/50/75/99% in an actual 521K--522K prompt, with no
  degeneration;
- exact 524,288 active GPU-KV tokens and no OOM at maximum context;
- C8 decode without failures or preemptions;
- LMCache DRAM reuse and cross-restart bounded-NVMe reuse;
- zero unexpected post-ready compiles and zero CUDA, distributed, XGrammar,
  worker-exit, or cache-allocation errors;
- a production restart on port 8000 followed by a real-agent soak recording
  PP, TG, MAL, cache hit rates, free VRAM, and power.

Restore the captured known-good control on port 8000 if no candidate passes.
Do not promote a theoretical kernel improvement or a microbenchmark-only win.

## Evidence layout

Retain credential-free evidence under a dated directory such as:

```text
/mnt/fast/build/sparkinfer-performance-followup-YYYYMMDD/
  control/
  split16/
  split8/
  h8-control/
  h8-nvfp4-candidate/
  outer-scale-control/
  outer-scale-candidate/
  dma-control/
  dma-candidate/
  pcie-calibration-368/
  final/
```

Each arm should contain the resolved configuration, container inspect, image
and source identities, startup and benchmark logs, raw JSON/CSV results,
`nvidia-smi` samples, compile-cache statistics, and a short manifest with
SHA-256 hashes. Summarize the final A/B in a new results document rather than
editing historical r26 measurements in place.

## Explicitly out of scope

Do not spend this GLM window on the paged-attention stage/registry findings,
Laguna specializations, generic MXFP8 BMM/linear kernels, thread-block
clusters, or compiler-fingerprint cleanup. They do not affect the selected
GLM path or are startup/developer-experience concerns rather than demonstrated
PP/TG opportunities. Qwen3.6 qualification currently uses the FlashInfer
decode wrapper, so any SparkInfer paged-attention investigation belongs in a
separate Qwen/backend experiment.
