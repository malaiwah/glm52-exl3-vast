# GLM-5.2 SIQ/QSRT serving research — 2026-08-09

## Outcome

SIQ and QSRT should be treated as related but distinct optimization programs:

- **SIQ is the mature serving baseline.** It uses ordinary EXL3 MCG/Trellis
  expert tensors, mixed K3/K4 tiers, hybrid BF16/EXL3 packaging, shared-H
  storage, and the already-qualified B12X mixed-tier paths. The full GLM-5.2
  SIQ/EXL3 stack has TP4/DCP4, MTP, long-context, concurrency, and production
  evidence.
- **QSRT is a newer codec and runtime contract.** It stores TP-independent
  fixed-payload atoms and selects expert-static K2/K3/K4 rate shifts. Its
  present end-to-end evidence is excellent but limited to the 5.04B Fruit
  proxy on one RTX 5090. It is not yet a qualified replacement for full
  GLM-5.2 SIQ on four RTX PRO 6000 cards.

The immediate QSRT opportunity is not another quantization search. It is to
remove conservative serving-policy limits and duplicated runtime work:

1. stop forcing block-M 8 for large-M W4A16 prefill;
2. raise the W4A8 decode ceiling from 16 rows through the C8/C12 MTP shapes;
3. qualify the already-supported per-request sparse-prefill splitting at
   C2/C4/C8 instead of enforcing `MAX_NUM_SEQS=1` in the launcher;
4. share the repeated preparation and accumulation around the two local I=256
   parts used by both Fruit TP1 and full GLM TP4;
5. preserve full-GLM shared-H storage and explicitly encode/serve MTP layer 78;
6. fix publication verification so a TP4 target+draft deployment does not hash
   the entire package up to eight times.

The highest-confidence short-term experiments are configuration/code A/Bs on
Fruit. The largest strategic task is creating a full GLM-5.2 QSRT artifact
from a full-model calibration corpus and qualifying its W4A16 oracle before
enabling W4A8, MTP, DCP, or 500K serving.

No production service, GitHub issue, or pull request was changed during this
research.

## Evidence labels

This document uses the following labels deliberately:

- **Measured:** retained GPU or artifact evidence exists at the cited
  revision.
- **Source-proven:** a behavior follows directly from the cited implementation
  but has not necessarily been measured end to end.
- **Estimated:** an engineering range to prioritize an A/B, not a performance
  claim.
- **Unknown:** the required artifact, hardware shape, or benchmark does not yet
  exist.

Fruit is a serving and CI proxy. It is Aider-contaminated and cannot establish
flagship intelligence. Codec KLD, reconstruction, and kernel parity on Fruit
are transferable evidence; model quality and full-scale performance are not.

## Current Local Inference Lab stack

```text
BF16 source checkpoint
        |
        v
KQuant calibration / QSRT encoder / publication seal
        |
        v
TP-independent atom-major safetensors
        |
        v
vLLM kquant_hybrid loader, planner, scheduler and model integration
        |
        v
B12X W4A16 prefill/reference + W4A8 small-M decode kernels
        |
        v
r31-derived Gilded Gnosis runtime image
```

| Responsibility | Current source of truth | Status on 2026-08-09 |
|---|---|---|
| Codec, calibration, artifacts | [local-inference-lab/kquant](https://github.com/local-inference-lab/kquant) | Generic codec merged; Fruit artifact work remains in [PR #4](https://github.com/local-inference-lab/kquant/pull/4) |
| GPU kernels and plans | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Generic QSRT merged; Fruit runtime remains in [PR #129](https://github.com/local-inference-lab/b12x/pull/129) |
| Loader and serving | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | Generic path is in draft [PR #243](https://github.com/local-inference-lab/vllm/pull/243); Fruit adapter remains in [PR #269](https://github.com/local-inference-lab/vllm/pull/269) |
| Base image | [blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) | Latest public standard base found is Gilded Gnosis v20 r31; QSRT uses a derived image |
| Operations and results | [rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) | QSRT appears in daily evidence, but has no full-GLM production runbook |

The latest standard image located was:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12xacee6e5-fi1ac6942-cu132-20260807-r31
sha256:3230c25ff95f8678a8eeb52a463f0d3b9f96f6ad550418cc51ea12177a55b41c
```

QSRT is not part of that immutable standard image. The current Fruit launcher
derives from r31 and injects exact open-PR vLLM and B12X sources.

### Revision caveat

The current heads contain meaningful commits newer than the GPU-qualified
revisions quoted in their descriptions:

| Open work | Head inspected |
|---|---|
| KQuant PR #4 | `5cdf270812f44dbadd6ed4e39d3fbd7edacda75d` |
| B12X PR #129 | `56d5a9063e7726d6799c87760e2070c38e479677` |
| vLLM PR #269 | `59ec754c37207d4e3edea4e762fbe40ed9f5f702` |

KQuant #4 includes later provenance and trust-chain hardening. B12X #129
includes shared multipart input preparation and a mixed-rate contract
correction. vLLM #269 includes runtime hardening and a newer KQuant pin.

Before drawing a release conclusion, rebuild one derivative image from the
exact three heads and repeat the Fruit gate. A clean or mergeable GitHub state
is not GPU qualification.

## SIQ and QSRT are not synonyms

### SIQ

In this project, SIQ describes a checkpoint/program using normal EXL3
MCG/Trellis expert tensors with model-specific tier metadata and hybrid
packaging. It is not a separate serialized bitstream. The mature full-model
path already has:

- mixed K3/K4 routed experts;
- one-grid mixed-tier B12X decode and prefill;
- shared-H quantization/storage;
- TP4/DCP4 and MTP serving;
- dynamic NVFP4 MLA KV, long-context gates, and production traffic.

The original large-M mixed K3/K4 problem in B12X issue
[#107](https://github.com/local-inference-lab/b12x/issues/107) is no longer an
open SIQ opportunity. PRs #112 and #117 delivered the one-grid path and
runtime-dynamic tier counts. Current r31 uses block-M 8 for decode and block-M
32 for qualified full-model prefill geometries.

### QSRT

QSRT means **Quantile-Stratified Rate-shifted Trellis codec**. Its current
profile is `qsrt_sqg_e4m3`, using the `sqg_xor_cheb_t12` codebook and SQG-E4M3
L16 trellis. Its defining properties are:

- K2/K3/K4 rate choices around a K3 budget;
- expert-static R0/R1/R2 rate-shift modes;
- paired P24/P33 records that keep pair payload size fixed;
- canonical TP-independent atom-major safetensors;
- disposable rank-local prepared caches;
- an optional X4T exact endpoint only when the source representation is
  compatible MXFP4, not arbitrary BF16 weights.

KQuant owns calibration, encoding, storage, and package validation. vLLM owns
loading, planning, scheduling, MTP/cache integration, and the API. B12X owns
the W4A16/W4A8 execution kernels and scratch contracts.

## What Fruit has actually proved

The source and artifacts are:

- [SIQ Fruit Instruct](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct)
- [BF16 Fruit source](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct-bf16)
- [QSRT Fruit Instruct artifact](https://huggingface.co/malaiwah/GLM-5.2-QSRT-Fruit-Instruct-exact/tree/82ef5eebc62faa442f815828ad2c77949a86f032)
- [annealed QSRT Fruit artifact](https://huggingface.co/malaiwah/GLM-5.2-QSRT-Fruit-exact/tree/da82206035557b14e0ec5e914562ad14f845995a)

### Artifact and quality results

| Result | Retained evidence |
|---|---:|
| Expert instances encoded | 2,816 |
| Atom files | 11 |
| Calibration documents | 256, document-disjoint |
| Routed calibration tokens | 46,223 |
| K3 NMSE | 0.013015 |
| K2 error / K3 | 4.254x |
| K4 error / K3 | 0.251x |
| QSRT artifact | 2.7095 GiB, 4.6177 effective bpw |
| Prior SIQ artifact | 2.8891 GiB, 4.9236 effective bpw |
| Whole-model reduction | 6.21% |
| Expert-payload reduction | 10.25% |
| Full-vocabulary KLD, six positions | mean 0.0017309, max 0.0034681 |
| Top-1 agreement | 100% |
| Top-10 overlap | 98.33% |

The allocation is almost uniform: 2,797 of 2,816 experts use R0/R0 and only
19 receive a nonzero rate shift. That demonstrates a sound codec and
publication pipeline, but it does not yet demonstrate that rate shifting is
materially better than a simpler uniform K3 checkpoint for this calibration
set. Full GLM needs a route-weighted census and held-out quality comparison.

Both BF16 and QSRT Fruit failed one exact-answer behavioral control identically.
That isolates the failure from the codec, but also reinforces that Fruit is a
serving proxy rather than a flagship quality judge.

### Runtime results

| TP1 RTX 5090 result | Throughput |
|---|---:|
| QSRT W4A8 decode | 38.0–38.5 tok/s |
| QSRT W4A16 decode | 28.3–29.1 tok/s |
| W4A8/W4A16 ratio | 1.325x |
| QSRT warm eager generation | 63.54 tok/s |
| BF16 warm eager generation | 77.45 tok/s |

The two throughput pairs come from different retained experiments and should
not be divided across rows. Within the matched QSRT comparison, W4A8 is a
clear win. The separate BF16 comparison shows that QSRT still leaves about
18% decode performance on the table on Fruit.

Physical validation is TP1. TP2 ownership is unit-tested, not physically
qualified. The rootless launcher currently enforces TP1, one sequence, 4K
context, 4K batching, and no speculative configuration.

## Transfer boundary to full GLM-5.2

Full GLM geometry is favorable but not proof:

| Geometry | Fruit TP1 | Full GLM TP4 per rank |
|---|---:|---:|
| Hidden size H | 1,024 | 6,144 |
| Expert intermediate I | 512 | 512 local (2,048 global / TP4) |
| Experts | 256 | 256 |
| Top-k | 8 | 8 |
| Local I=256 parts | 2 | 2 |

The atom loader partitions complete global atom slots by TP rank and then
prepares 256-channel pieces. Full GLM TP4 therefore presents the same two-part
local-I geometry as Fruit TP1. The existing kernels are parameterized over H,
and H=6144 satisfies their alignment contracts.

What changes is substantial:

- FC1 reduces over a hidden dimension six times deeper than Fruit.
- FC2 presents a six-times-wider H grid.
- register pressure, occupancy, spills, launch balance, and graph residency can
  all change;
- TP4/DCP4 adds physical collectives and rank imbalance;
- MTP layer 78 adds target/draft planning and effective query shapes;
- 500K context adds sparse-MLA, KV, cache/offload, and long-prefill pressure.

The right conclusion is “high structural reuse,” not “drop-in support.”

### First-order memory estimate

Full GLM has roughly 724.8B routed-expert parameters globally, or 181.2B per
rank at TP4. Moving only that payload from an average 3.42 to 3.0 bits would
save about 8.9 GiB/rank in ideal bit arithmetic before record metadata,
alignment, codebooks, kept/high-quality tiers, and runtime scratch. Fruit’s
measured 10.25% expert-payload reduction suggests an engineering estimate of
roughly **7–9 GiB/GPU**, not a result.

That headroom could fund more KV, safer late-allocation margin, or larger graph
coverage. It must not be booked until a full artifact is loaded and profiled.

## Ranked serving opportunities

| Rank | Candidate | Scope | Expected effect | Evidence status |
|---:|---|---|---|---|
| 1 | Shape-aware W4A16 block-M for prefill | QSRT | 10–30% PP hypothesis | Source-proven bad fixed policy; unmeasured transfer |
| 2 | Extend W4A8 from M<=16 through M=48 | QSRT | 5–15% full-stack TG hypothesis at C8/C12+MTP | Fruit W4A8 win measured; larger M unqualified |
| 3 | Qualify C2/C4/C8 sparse prefills | QSRT/vLLM | Unlock practical serving; scaling unknown | Source shows per-request splitting already exists |
| 4 | Native/shared two-part W4A16 preparation and accumulation | QSRT/B12X | 2–8% E2E PP estimate | Duplicated work source-proven |
| 5 | Same-expert route packing and adaptive W4A8 crossover | QSRT/B12X | 3–15% E2E TG estimate under C>1/MTP | Prior art; unmeasured here |
| 6 | Native/shared two-part W4A8 output path | QSRT/B12X | 1–5% E2E TG estimate | Remaining duplicate work source-proven |
| 7 | Shared-H full-model QSRT schema | QSRT/KQuant/vLLM | memory and input-transform reduction | SIQ benefit measured; QSRT transfer unqualified |
| 8 | Rank/target/draft-aware publication verification | QSRT/vLLM | startup/NFS/CPU reduction | Duplicate whole-package reads source-proven |
| 9 | Right-size kept-tier/X4T scratch | QSRT/vLLM | up to ~45 MiB/rank in the examined 96/256 case | Source arithmetic; only relevant to mixed kept tiers |
| 10 | Explicit runtime preparation/freeze | QSRT/vLLM/B12X | reliability; possible cold-start benefit | Lazy construction source-proven |
| 11 | P24 kernel tuning | QSRT/B12X | negligible on Fruit; unknown full GLM | Fruit weighted incidence only 0.29% |

Ranges overlap and are not additive.

## 1. Stop pinning W4A16 prefill to block-M 8

The current vLLM QSRT planner explicitly supplies:

```python
w4a16_block_size_m=8
```

for a plan sized to `max_num_tokens`, including large-M prefill. B12X accepts
larger M geometries. SIQ history already demonstrated that block-M 8 can be a
poor large-M choice, and current full-model SIQ uses block-M 32 for qualified
prefill shapes.

QSRT uses a different SQG decode path, so SIQ’s percentage cannot be copied.
The first experiment should be Fruit W4A16 at M=128/512/1024/2048/3072/4096
with block 8/16/32/48/64 and auto. Record kernel time, end-to-end prefill,
scratch, registers, spills, tensor utilization, and exact parity.

If the crossover repeats, implement a plan selector keyed by H, local I,
top-k, M, record mix, and device—not a new universal default.

## 2. Extend W4A8 through effective MTP concurrency shapes

The current dispatch uses W4A8 only when:

```text
decode_only and 0 < M <= 16
```

With MTP3, effective target/draft query rows can reach 32 at C8 and 48 at C12.
Those production-relevant shapes silently fall back to slower W4A16. The
additional full-model scratch for an M=48 backing allocation is estimated at
only 8–12 MiB/rank, but graphs, late allocation, and kernel occupancy must be
measured.

Test M=1/2/4/8/12/16/24/32/40/48. Prewarm every promoted graph shape before KV
sizing. Measure target and draft separately, then C1/C4/C8/C12 TG and mean
acceptance length. A faster target with a slower or lower-acceptance draft is
not a serving win.

## 3. Remove the launcher-only one-sequence restriction carefully

`MAX_NUM_SEQS=1` is a qualification guard, not a fundamental kernel limit.
The vLLM metadata builder already splits a multi-request B12X prefill into
single-request chunks, and the execution path iterates those chunks. Each
individual B12X call still receives the shared page table it requires.

This supports a low-code qualification path for C2/C4/C8, but it does not
promise linear scaling: simultaneous prefills currently serialize through
per-request indexer chunks. Required cases are:

- two and four simultaneous cold prefills;
- mixed long prefill plus active decode;
- prefix-cache hit plus cold miss;
- preemption/cancellation;
- structured outputs and prompt logprobs;
- graphs and eager parity;
- MTP target/draft identity and MAL.

Only after measuring chunk/launch overhead should the project consider packed
multi-request indexer work.

## 4. Share multipart work

Both Fruit TP1 and full GLM TP4 have local I=512 represented as two I=256
parts. The arithmetic for both parts is required; the surrounding H-sized work
is not always required twice.

### W4A16 prefill/reference

The current path invokes the complete backend for each part. That repeats route
preparation, H input transformation, fused MoE launch setup, H output
transformation/top-k reduction, and FP32 copy/add around the necessary part
arithmetic.

A shared-preparation/native multipart plan is the best code-level PP target
after the block-size A/B. Estimated effect: 3–10% of the routed kernel and
roughly 2–8% end-to-end PP.

### W4A8 decode

W4A8 already shares route-map construction and one input rotation/MXFP8
quantization. It still repeats FC1, activation/rotation/quantization, FC2, and
the H output transform/top-k reduction per part, followed by FP32 copy/add.

A native two-part accumulator or streamed FC2 reduction is estimated at
2–8% of the routed kernel and 1–5% end-to-end TG.

The final full-H dtype conversion/copy is another secondary fusion target.

## 5. Pack same-expert routes and use a measured crossover

Current W4A8 is a route-per-CTA anchor. Under concurrency and MTP, multiple
rows frequently route to the same expert. Sorting/compacting those rows can
increase reuse and reduce repeated expert setup.

Primary prior art exists in:

- [ExLlamaV3 route packing](https://github.com/turboderp-org/exllamav3/blob/4f8ad0121f483ba66a5336244a4c3b6d7210385e/exllamav3/modules/block_sparse_mlp.py#L1189-L1252)
- [ExLlamaV3 packed EXL3 MoE kernel](https://github.com/turboderp-org/exllamav3/blob/4f8ad0121f483ba66a5336244a4c3b6d7210385e/exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh#L47-L266)
- [llama.cpp GPU expert compaction](https://github.com/ggml-org/llama.cpp/blob/7ba604f1cb61cd14898138e9abc0b4ff2601f180/ggml/src/ggml-cuda/mmid.cu#L22-L168)
- [llama.cpp one-time token quantization/expert bounds](https://github.com/ggml-org/llama.cpp/blob/7ba604f1cb61cd14898138e9abc0b4ff2601f180/ggml/src/ggml-cuda/mmq.cu#L183-L256)

TensorRT-LLM also uses a shape-dependent B12X/grouped-GEMM crossover rather
than assuming one kernel wins everywhere. Its integration is useful design
evidence, not a drop-in implementation for this vLLM/B12X path.

Instrument rows per expert, duplicate-route percentage, route entropy, and
target/draft distributions. Benchmark M=1 through 48. Preserve the M=1
route-major fast path and switch only where packing repays its own cost.

Estimated effect is 10–35% of the routed kernel and 3–15% end-to-end TG under
C>1/MTP, with a likely neutral or negative result at M=1. This estimate must
not be promoted as a QSRT result until measured.

## 6. Preserve shared-H and define the high-quality endpoint

Full-model SIQ shared-H saves about 681 MiB/GPU in the MTP3 posture. B12X W4A8
also accepts shared input geometry; with top-k 8, a physical shared-H input
stage can avoid transforming up to eight route-expanded copies of an input.

The full-model QSRT schema should carry the shared-H contract explicitly.
Do not conflate an encoder’s shared Hessian with physically shared runtime
rotations: validate actual stored tensors and prepared rows.

The quality endpoint also needs a GLM-specific decision. W4A8 currently
rejects a layer containing an X4T kept tier, and X4T exactness applies only to
compatible MXFP4 source nibbles. A BF16-source GLM artifact must therefore:

- remain all-QSRT and accept the measured quality;
- define a K4/K6/BF16 kept tier and add a mixed W4A8 implementation; or
- use W4A16 for layers containing kept experts.

This is a quality/memory/performance Pareto decision, not merely a file-format
choice.

## 7. Treat MTP as part of the artifact

MTP layer 78 must have an explicit entry in `hybrid_bit_map`. In the generic
hybrid path, an absent layer can become all-kept NVFP4; the QSRT atom loader
then rejects an unmapped QSRT layer. A full artifact must co-calibrate and
publish the MTP layer rather than grafting an implicit draft after the fact.

The qualification matrix must report:

- target-only W4A16 KLD;
- target W4A8 KLD;
- draft W4A8/W4A16 parity;
- MAL and per-position acceptance;
- C1/C4/C8/C12 target and draft timings;
- graph coverage through `MAX_NUM_SEQS x (1 + MTP)`;
- no late compile or persistent allocation after profiling.

## 8. Make package verification scale to TP4 target+draft

The current publication verifier hashes every file in `MANIFEST.sha256` in
each quantization-config process. Its cache is local to one runtime object.
Target and MTP construct independent quantization configs.

At TP4, a QSRT target plus QSRT draft can therefore perform as many as eight
complete logical package reads. For a hundreds-of-GB full artifact this can
mean multiple TB of aggregate hashing work. A warm page cache may hide NFS
traffic, but it does not remove CPU and DRAM bandwidth consumption.

Recommended order:

1. reuse one process-global authenticated seal between target and draft;
2. coordinate verification across ranks and have each rank hash only files or
   authenticated ranges it owns;
3. retain authenticated file descriptors or use fs-verity/Merkle/invalidation
   metadata so “verified” cannot become a stale unchecked marker;
4. include X4T sidecars, which are currently reopened by pathname after the
   primary verification and therefore have a TOCTOU gap.

Measure cold-NFS and warm-cache startup separately, including bytes read,
hashing CPU time, DRAM bandwidth, and time to first healthy response.

## 9. Freeze runtime preparation before KV sizing

W4A16 planning correctly uses maximum tokens rather than the first incidental
M. MRv1 and MRv2 both profile the drafter. W4A8 allocates an M=16 backing
store and exposes allocation-free M=1..16 views; the current kernel family is
not M-specialized.

The remaining sharp edge is lifecycle: setup still occurs lazily through the
first `apply()` path. The production contract should expose an explicit
post-load preparation step covering all target/draft plans and graph rows,
then assert that capture/serving cannot JIT or persistently allocate.

This should build on the safeguards already integrated through the B12X
#102/#108/#110/#112 lineage rather than reviving an obsolete patch.

### Right-size optional kept-tier scratch

The current X4T preparation path allocates scale grids across all E experts.
For a full-model layer with only 96 of 256 experts in a kept tier, the examined
shape would reserve roughly 72 MiB/rank where about 27 MiB is active. A
subset-aware layout could recover approximately 45 MiB/rank. All-QSRT layers
avoid this allocation, so this is a conditional memory cleanup rather than a
general QSRT speed feature.

The logical-mid capture hooks also remain incomplete: W4A8 cannot presently
capture its logical middle representation, and multipart W4A16 skips that
capture. Restore these before using capture data for full-model calibration or
runtime parity claims.

## 10. Do not optimize P24 until the full artifact says to

Fruit’s exact manifest contains 20 P24 FC1 pair slots and 9 P24 FC2 pair slots
out of 5,632 in each corresponding population. Matrix-work weighting gives:

```text
(2 * 20 + 9) / (3 * 5632) = 0.2900%
```

Even if the observed P24 kernel were 14.4% slower than P33 and made free, the
uniform routed-kernel ceiling would be tiny. P24 cannot explain Fruit’s 18%
gap to BF16.

For full GLM, collect a **route-weighted** P24/P33 histogram before investing.
If traffic heavily favors the rare shifted experts, static record counts are
not enough.

## General vLLM/B12X work that remains applicable

QSRT does not replace the model-level opportunities already tracked for the
r31 SIQ stack:

- [vLLM issue #207](https://github.com/local-inference-lab/vllm/issues/207):
  hoist repeated sparse-MLA metadata work and address the >140K projection
  compaction path safely;
- [vLLM issue #272](https://github.com/local-inference-lab/vllm/issues/272):
  skip discarded logits/sampling/draft work for unfinished prefills while
  preserving MTP hydration;
- [vLLM issue #273](https://github.com/local-inference-lab/vllm/issues/273):
  preserve specialized prefill/decode paths in mixed batches;
- B12X #130 and vLLM #270: persistent-buffer ownership and target/draft sharing
  as a memory/reliability foundation.

Those changes should be qualified once on W4A16 and once on W4A8 only when the
kernel path differs. Avoid duplicating a format-independent patch solely for
QSRT.

## Full qualification plan

### Phase 0 — establish one reproducible Fruit control

1. Rebase or compose exact current heads of KQuant #4, B12X #129, and vLLM
   #269 on the immutable r31 base.
2. Pin the Fruit artifact revision and source/runtime manifests.
3. Use a fresh compile-cache namespace; then retain it for matched warm A/Bs.
4. Re-run publication checks, CPU suite, TP ownership tests, TP1 eager, graphs,
   W4A16/W4A8 correctness, KLD, and the retained 700-token decode fixture.
5. Capture startup allocations and assert no new JIT or persistent allocation
   after profiling.

### Phase 1 — low-cost serving-policy A/Bs on Fruit

Run counterbalanced A/B/A cells:

1. W4A16 block-M 8 versus 16/32/48/64/auto.
2. W4A8 ceiling 16 versus 24/32/40/48.
3. `MAX_NUM_SEQS` 1 versus 2/4/8, first pure prefill and then mixed traffic.
4. shared versus current multipart preparation, initially as instrumentation
   and then as a code patch if repeated work is material.
5. route-major versus same-expert packed W4A8 for M=1..48.

### Phase 2 — build the full-model artifact and W4A16 oracle

1. Select document-disjoint, multilingual, coding, tool-use, long-context, and
   MTP-aware calibration/validation sets.
2. Collect route-weighted expert activation and P24/P33 distributions.
3. Encode every target MoE layer and MTP layer 78 with explicit shared-H
   geometry and a documented high-quality-tier policy.
4. Seal one TP-independent publication.
5. Load TP1/TP2/TP4 and verify exact atom reconstruction at every partition
   boundary.
6. Qualify W4A16 first. It is the serving oracle for subsequent W4A8 work.

### Phase 3 — full-model performance integration

1. TP4/DCP4 W4A16 PP at 8K/32K/64K/128K/256K/500K.
2. Enable W4A8 at C1, then C4/C8/C12 with MTP3.
3. Compare MTP3 and MTP5, then report target/draft timing, MAL, graph coverage,
   and per-rank memory rather than assuming deeper speculation wins.
4. Qualify C2/C4/C8 cold prefills and mixed prefill/decode.
5. Add prefix-cache reuse and LMCache DRAM/NVMe offload.
6. Measure cold-NFS and warm-cache package verification and startup.

### Phase 4 — quality and release gates

- KLD on at least eight disjoint 2K windows, not one six-position sample;
- BF16 or best available teacher, W4A16, then W4A8 comparisons;
- multi-seed five-depth needles through the 500–520K envelope;
- degeneration and long free-generation checks;
- Aider/coding, multilingual, tools, structured outputs, prompt logprobs, and
  real Hermes-agent workloads;
- no OOM, restart, late JIT, graph miss, or unprofiled persistent allocation;
- exact/lossless PCIe transport unless a lossy path independently passes every
  quality gate;
- stable C1/C4/C8/C12 TG, PP, MAL, prefix-cache and offload metrics;
- rollback to the exact current SIQ production profile if any gate fails.

## Required instrumentation

Add NVTX/CUDA-event ranges around:

- route packing and expert-map construction;
- H input rotation and MXFP8 quantization;
- each I=256 part’s FC1, activation/rotation/quantization, and FC2;
- H output transform/top-k reduction;
- FP32 copy/add and final dtype conversion;
- sparse-indexer per-request chunking;
- target and every MTP draft step;
- publication hashing and atom preparation.

Collect:

- rows per expert, duplicate-route rate, route entropy;
- route-weighted P24/P33 and R0/R1/R2 incidence;
- DRAM/L2 bytes, tensor utilization, SM occupancy, registers and spills;
- launch counts and CPU dispatch gaps;
- per-rank atom bytes, plan scratch, graph pools, late allocations, minimum free
  VRAM, and logical KV;
- package bytes read and hashed per rank/process;
- power, clocks, temperature, and slowest-rank time;
- PP, TG, MAL, KLD, top-1/top-10, needles, and workload success.

## Promotion policy

Promote a QSRT change only when it improves an end-to-end metric outside run
noise and does not regress the following:

- exact reconstruction and publication validation;
- W4A16/W4A8 KLD and task quality;
- MTP acceptance and target/draft identity;
- 500–520K retrieval and degeneration;
- C1 TG while improving aggregate concurrency;
- memory accounting, graph capture, or late-allocation safety;
- cold and warm startup reliability.

For the full checkpoint, compare QSRT against the exact current SIQ/r31
production profile, not against BF16 Fruit or an obsolete SIQ image.

## Primary sources

### Local Inference Lab

- [KQuant repository](https://github.com/local-inference-lab/kquant)
- [KQuant Fruit artifact PR #4](https://github.com/local-inference-lab/kquant/pull/4)
- [KQuant Fruit pilot issue #3](https://github.com/local-inference-lab/kquant/issues/3)
- [B12X Fruit QSRT PR #129](https://github.com/local-inference-lab/b12x/pull/129)
- [vLLM Fruit adapter PR #269](https://github.com/local-inference-lab/vllm/pull/269)
- [vLLM generic QSRT draft PR #243](https://github.com/local-inference-lab/vllm/pull/243)
- [rtx6kpro August 7 summary](https://github.com/local-inference-lab/rtx6kpro/blob/master/daily-summaries/2026-08/2026-08-07.md)
- [rtx6kpro August 8 summary](https://github.com/local-inference-lab/rtx6kpro/blob/master/daily-summaries/2026-08/2026-08-08.md)

### External prior art

- [QTIP paper](https://arxiv.org/abs/2406.11235) and [official implementation](https://github.com/Cornell-RelaxML/qtip)
- [QServe](https://arxiv.org/abs/2405.04532)
- [Atom](https://arxiv.org/abs/2310.19102)
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [upstream vLLM Blackwell W4A8 issue #35439](https://github.com/vllm-project/vllm/issues/35439)
- [SGLang releases](https://github.com/sgl-project/sglang/releases)
- [TensorRT-LLM B12X integration](https://github.com/NVIDIA/TensorRT-LLM/blob/1d7c771da76abefbabb28674d10556a57805202f/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl_b12x.py)
- [DA-MoE route-aware dispatch](https://arxiv.org/abs/2607.23099)
