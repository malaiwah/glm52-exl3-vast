# GLM-5.2 Jarrel quant comparison and NCCL/B12X maintenance plan

Prepared and executed 2026-08-01 during an explicit AIBeast maintenance
window. The completed result and immutable evidence boundary are in
[the r17 maintenance report](glm52-r17-maintenance-results.md).

## Executive conclusions

1. Jarrel's current main hybrid is not simply a lower-bpw version of our
   3.25-bpw checkpoint. It uses a learned AQLM codebook for most routed experts,
   NVFP4 for a smaller, sensitivity-selected set, and a custom runtime. Our
   checkpoint uses EXL3 Trellis K3/K4 experts and the SparkInfer/B12X runtime.
1. Jarrel keeps more nominal precision in its selected hot experts, all routed
   MTP-78 experts, vision tower, projector, and published FP8 KV posture. Our
   checkpoint keeps more nominal precision in the majority cold-expert tier
   and, unlike Jarrel's current loader, does not convert attention output
   projections or shared experts from BF16 to FP8 at load time.
1. Both retain attention, the sparse indexer, dense layers 0-2, routing, and
   output-critical tensors at high precision on disk. Neither prunes experts.
1. Jarrel's `NCCL_MAX_NCHANNELS=4` plus `NCCL_BUFFSIZE=1048576` recipe is a
   useful hypothesis, not a portable default. On NCCL 2.30, the modern cap is
   `NCCL_MAX_CTAS=4`. The 1 MiB buffer is one quarter of NCCL's 4 MiB default.
1. B12X already intercepts small and very large traffic on AIBeast, but NCCL
   still carries the middle-sized TP traffic, large DCP exchanges, and parts of
   long prefill. NCCL tuning can therefore matter, but probably at specific
   message-size crossovers rather than uniformly.
1. GG v20-r17 must be rebased into the turnkey and qualified before the NCCL
   or B12X tuning campaign. Its vLLM #222 supersedes both our #210 prefill-
   capacity patch and #219 mixed-K patch; stacking either old patch over #222
   would invalidate the result.
1. The published r17 gate proves the immutable image and repaired custom PCIe
   path at MTP0, C1, graph 6, 262K. It does not qualify our production MTP5,
   C8, graph 48, 512K, LMCache posture. A matched r14/r17 production-shape A/B
   is therefore the first GPU gate.
1. The r17 candidate slowdown was packaging, not #222: the wheel omitted a
   local header required by the runtime-compiled PCIe extension and silently
   fell back to PyNCCL. SparkInfer #105 adds the header and packaging test; a
   valid r17 boot must select `B12X_PCIE_ONESHOT_DMA` and must not repeat that
   fallback.
1. After r17 passes, the cost-bounded core NCCL experiment is five boots in
   this order:
   control A1, CTA cap B, combined D, 1 MiB buffer C, control A2. Only after the
   NCCL winner is selected do we recalibrate the B12X/NCCL crossover and test
   B12X route choices.

## Evidence boundary

The comparison below is pinned so that moving model cards and runtime branches
cannot silently change its meaning.

| item | immutable boundary |
|---|---|
| Jarrel checkpoint | `jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid@c5d93567f1ff2de4dbba6018b58a653654c1309a` |
| Jarrel runtime used for format inspection | `jarrelscy/vllm-glm52-sm120@57955eed93621c6331db73226c755e33dcccb0e8` |
| appliance 3.25-bpw checkpoint | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@fa85cf3ad778795c3455913760ec6a7359270f4d`; weight objects are unchanged from the appliance-pinned `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` and AIBeast snapshot `61d2b6b757f6a4ac7098a78d861f2033497532dc` |
| current production/control runtime | GG v20-r14 plus vLLM #210 and #219, image `localhost/glm52-turnkey:r14-cap1024-shape-turnkey-v2`, ID `6ab5b03fd8669eab048e6b7836f026df7b4e77377f9b6b8914d283139b7384f6` |
| r17 candidate image | `voipmonitor/vllm:gilded-gnosis-v20-vllmdb29328-sib2bff71-fi801d57a-cu132-20260801-r17`, registry digest `sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727`, image ID `sha256:1384b6badaa6c7546c062ddc1c0543b0f657e2105a41a92ba6220372890b0e98` |
| r17 composed sources | vLLM `db293280d021d32db0552f3f6e4b95abbd9c69a1`; SparkInfer `b2bff719ba1be0a5d30cb39cba795f0812db0f3d`; LMCache `a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa`; reproducible build `f5ba50b0d986bb9c46c0270eea0bc8df72bafefe` |
| NCCL documentation reviewed | NVIDIA NCCL 2.30.7 documentation; AIBeast image currently loads the pinned local-inference NCCL 2.30.4 library |

Primary sources:

- [Jarrel model card](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid/blob/c5d93567f1ff2de4dbba6018b58a653654c1309a/README.md)
  and [configuration](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid/blob/c5d93567f1ff2de4dbba6018b58a653654c1309a/config.json)
- [Jarrel hybrid loader contract](https://github.com/jarrelscy/vllm-glm52-sm120/blob/57955eed93621c6331db73226c755e33dcccb0e8/vllm/model_executor/layers/quantization/nvfp4_aqlm_hybrid.py)
  and [load-time FP8 conversion](https://github.com/jarrelscy/vllm-glm52-sm120/blob/57955eed93621c6331db73226c755e33dcccb0e8/vllm/model_executor/layers/quantization/fp8_w8a16.py)
- [Willfalco model card](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw/blob/fa85cf3ad778795c3455913760ec6a7359270f4d/README.md),
  [configuration](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw/blob/fa85cf3ad778795c3455913760ec6a7359270f4d/config.json),
  and [per-expert tier bitmap](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw/blob/fa85cf3ad778795c3455913760ec6a7359270f4d/tier_bitmap.json)
- [AQLM paper](https://arxiv.org/abs/2401.06118)
- [NVIDIA NCCL environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [r17 vLLM #222](https://github.com/local-inference-lab/vllm/pull/222),
  [SparkInfer #105](https://github.com/local-inference-lab/sparkinfer/pull/105),
  [reproducible build review #14](https://github.com/local-inference-lab/blackwell-llm-docker/pull/14),
  [release checklist](https://github.com/local-inference-lab/rtx6kpro/issues/33),
  and [GLM-5.2 launch contract](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md)

Published benchmark claims from the two stacks are not treated as a matched
A/B. They use different runtimes, KV formats, context envelopes, speculative
settings, prompts, and evaluation harnesses.

## What AQLM means here

AQLM is additive vector quantization. Instead of assigning an independent
small scalar to every weight, it represents a group of weights using an index
into a learned vector codebook. Jarrel's cold tier uses one 65,536-entry FP16
codebook over groups of eight weights. One 16-bit index per eight weights is
exactly 2 bits/weight before metadata.

That is not equivalent to an ordinary scalar 2-bpw quant. A learned vector can
preserve correlated structure that a two-bit scalar cannot. Jarrel also keeps
an FP16 scale per expert/output channel and PV-tunes the cold codebooks and
scales against BF16 teacher outputs. During decode, a fused kernel performs
codebook lookup/GEMV; during prefill, bounded expert groups are dequantized to
FP16 and passed to per-expert GEMMs.

The current Jarrel schema implements an intermediate 2/4-bpw AQLM tier, but
the checkpoint does not use it: every layer reports `n_base=0`. Its actual
cold tier is therefore approximately 2.0093 bpw including scales, useful
codebooks, and the presently stored but unused `w2m` codebook. Its NVFP4 tier
uses four-bit values plus an FP8 scale per 16 weights, approximately 4.5 bpw
before negligible outer-scale overhead.

EXL3 takes a different approach. Our routed expert payload is exactly K3 or K4
Trellis data, plus per-rank/projection FP16 Hadamard scale/flip vectors and an
`int32` selector. That metadata adds approximately 0.03386 bpw. The codebook is
procedural rather than a large learned checkpoint payload.

## Exact routed-expert allocation

| allocation | Jarrel NVFP4+AQLM main hybrid | appliance 3.25-bpw EXL3 |
|---|---:|---:|
| layers 3-77 | 5,642 NVFP4 + 13,558 AQLM | 4,800 K4 + 14,400 K3 |
| per target layer | 29-144 NVFP4, average 75.23; remainder AQLM | exactly 64 K4 + 192 K3 |
| routed MTP layer 78 | all 256 NVFP4 | all 256 K3 Trellis |
| all routed experts | 5,898 NVFP4 + 13,558 AQLM | 4,800 K4 + 14,656 K3 |
| high/lower-tier share | 30.31% / 69.69% | 24.67% / 75.33% |
| nominal target-expert mean | 2.7346 bpw | 3.2500 bpw |
| nominal mean including MTP | 2.7579 bpw | 3.2467 bpw |
| approximate stored mean including format overhead | 2.7643 bpw including MTP | 3.2806 bpw |

Jarrel's older prose saying 65-96 NVFP4 experts per layer is stale after its
measured-error re-tier. The exact current range is 29-144. Jarrel first uses
routing mass and then measured error to allocate its variable hot tier. The
Willfalco quant uses exactly 64 K4 experts per layer selected with a four-axis
owner corpus covering general, legal, code/agentic, and reasoning-termination
behavior, with LDLQ/Hessian calibration.

Both formats preserve every one of the 256 experts in every routed layer. This
is precision allocation, not pruning.

## Which components received more accuracy budget

“Higher precision” in this table describes the stored or runtime format. It is
not a claim of lower end-to-end error without a matched BF16-relative test.

| component | Jarrel on disk | Jarrel at serving time | our 3.25-bpw checkpoint/runtime | nominal accuracy-budget observation |
|---|---|---|---|---|
| embeddings, final norm, LM head | BF16 | BF16 | BF16 | equal |
| attention Q/K/V and MLA absorption inputs | BF16 | BF16 | BF16 | equal |
| attention output projection | BF16 | selected tensors converted to E4M3 W8A16 by Jarrel's loader | BF16; `VLLM_USE_B12X_FP8_GEMM=1` selects eligible kernels but does not itself quantize a BF16 checkpoint tensor | our runtime keeps more nominal precision here |
| sparse DSA/indexer | BF16 | BF16 | BF16 | equal and correctness-critical |
| dense MLP layers 0-2 | BF16 | BF16 | BF16 | equal |
| shared experts | BF16 | gate-up/down converted to E4M3 W8A16 | BF16 | our runtime keeps more nominal precision here |
| routed experts, hot tier | NVFP4, approximately 4.5 bpw | fused NVFP4 kernels | K4 Trellis plus metadata | Jarrel gives selected experts a slightly larger nominal storage budget |
| routed experts, cold tier | learned AQLM, approximately 2.0093 bpw | fused lookup for decode; bounded FP16 dequantization for prefill | K3 Trellis plus metadata | our checkpoint gives the majority tier more raw bits; Jarrel's learned/PV-tuned codebook may compensate |
| routers, gates, correction biases | BF16 | BF16 | BF16 | equal |
| routed MTP-78 experts | all NVFP4 | NVFP4 | all K3 Trellis | Jarrel spends more nominal precision on draft prediction |
| MTP-78 attention, router, shared expert, `eh_proj`, norms | BF16 | BF16 | BF16 | equal |
| vision tower + projector | BF16, 932,887,040 bytes | BF16 | absent from this text-only checkpoint | Jarrel retains full-precision vision capability; ours spends no VRAM on it |
| published serving KV | `fp8_ds_mla`, BF16 RoPE, 656-byte record | FP8 | dynamic-token `nvfp4_ds_mla`, FP8 RoPE, 368-byte record | Jarrel spends more KV precision/capacity per token; ours favors a 512K fit |

The MTP detail corrects two easy misconceptions:

- Jarrel's routed layer 78 is NVFP4, not BF16 and not AQLM. Its AQLM map ends
  at layer 77; the actual layer-78 expert tensors are packed `U8` weights with
  FP8 block scales and FP32 outer scales.
- Willfalco's `modelopt_dispatch_note` says layer 78 remains BF16, but that note
  is stale. The README, quantized scope, tier bitmap, and actual safetensors
  index show routed layer-78 experts in K3 Trellis. Only the non-expert MTP
  tensors remain BF16.
- Runtime `MTP_TOKENS=3` or `5` is speculative depth. Both checkpoints store
  one next-token-prediction layer; MTP5 does not imply five stored heads.

## Size, quality, and performance interpretation

| item | Jarrel main hybrid | appliance 3.25-bpw |
|---|---:|---:|
| model safetensors | 292.492 GB / 272.405 GiB, including vision | 339.069 GB / 315.783 GiB, text only |
| text payload | 291.560 GB | 339.069 GB |
| disk-size difference | 46.577 GB smaller even with vision | baseline |
| approximate TP4 disk-byte difference | about 10-12 GB/rank before loader/runtime effects | baseline |

The disk difference suggests substantial potential VRAM headroom for Jarrel,
but is not a VRAM claim. Replicated tensors, dequantization buffers, codebooks,
CUDA graphs, and scratch arenas differ between the runtimes and must be
measured after load and at first request.

The quant author reports mean KLD `0.087711` with standard FP8 KV/BF16 RoPE
and `0.095971` with dynamic NVFP4/RoPE8. The appliance independently measured
the dynamic posture at `0.0927076684` over its standard 2,047 positions. There
is no matched Jarrel result using the same reference logits, positions, KV
format, prompts, and sampler. Bits alone therefore cannot rank overall
accuracy. Jarrel's published quality and context results are encouraging, but
they do not isolate the quant from its custom kernels and serving choices.

The defensible qualitative conclusion is:

- Jarrel concentrates accuracy on the experts that its calibration considers
  hot or sensitive, gives the full routed MTP layer NVFP4, and uses a less
  compressed published KV posture.
- Our 3.25-bpw checkpoint gives every cold routed expert K3 rather than a
  roughly 2-bpw tier, keeps selected output/shared paths BF16 at runtime, and
  uses a more compressed KV record to retain the full 512K envelope.
- A future quant A/B must use a common text-only, MTP-off, 64K envelope and
  matched BF16-reference positions. A comparison of each stack's preferred
  speculative/context configuration should be labelled an end-to-end stack
  comparison, not a quant-only result.

## Current AIBeast control to preserve

At the start of the maintenance window, capture the live container rather than
reconstructing it from profile defaults. The checked-in 3.25-bpw profile uses
MTP3/graph 32, while the currently soaked AIBeast production control was
explicitly launched with MTP5/graph 48. The A/B must reproduce the latter.

Observed control on 2026-08-01:

| item | control value |
|---|---|
| service/container | `glm52-turnkey-r14-cap1024-shape-prod`, port 8000, aliases `GLM-5.2` and `local-primary` |
| hardware | 4x RTX PRO 6000 Blackwell 96 GB; every GPU reports `NODE` and NUMA 0 |
| driver/runtime | NVIDIA driver 595.71.05, CUDA 13.2, 280 W/card |
| model | read-only Willfalco 3.25-bpw snapshot `61d2b6b757f6a4ac7098a78d861f2033497532dc` |
| topology | TP4/DCP4 |
| serving geometry | MTP5 probabilistic, max length 524,288, eight sequences, batch 2,048, EXL3 prefill capacity 1,024, graph/Trellis 48 |
| active GPU KV | exactly 2,048 blocks = `2,048 × 64 × DCP4 = 524,288` logical tokens |
| KV/cache | dynamic-token NVFP4 MLA; 125 GiB aggregate LMCache DRAM and bounded 512 GiB local-NVMe L2 |
| lossless transport | `F8_DMA=0`; current effective B12X DMA minimum 6 MiB; CKV gather maximum 140,000; CKV prefetch 0 |
| image | `localhost/glm52-turnkey:r14-cap1024-shape-turnkey-v2` |

Before stopping it, preserve:

```bash
export EVIDENCE_ROOT=/mnt/fast/build/nccl-b12x-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$EVIDENCE_ROOT"

podman inspect glm52-turnkey-r14-cap1024-shape-prod \
  >"$EVIDENCE_ROOT/production-control.inspect.json"
podman logs glm52-turnkey-r14-cap1024-shape-prod \
  >"$EVIDENCE_ROOT/production-control.log" 2>&1
curl -fsS http://127.0.0.1:8000/metrics \
  >"$EVIDENCE_ROOT/production-control.metrics.txt"
nvidia-smi -q >"$EVIDENCE_ROOT/production-control.nvidia-smi-q.txt"
nvidia-smi topo -m >"$EVIDENCE_ROOT/production-control.topology.txt"
```

Also copy the resolved `/state/.glm-config/runtime/config.env`, image ID,
checkpoint mount, cache fingerprint, GPU order, and LMCache configuration from
the running container. Those bytes, not the prose table, are the rollback
authority.

## Gate 0: rebase and qualify GG v20-r17

Do this before the NCCL/B12X experiments. If r17 cannot reproduce its release
gate and pass the production-shape A/B, restore r14 and end the window. Do not
spend the remaining window tuning a runtime that is about to be replaced.

### Reviewed r17 change set

The publication locks and current PR heads were rechecked on 2026-08-01. r17
is an immutable composition of reviewed open PR heads; it is not evidence that
all of those heads have merged into their upstream default branches.

| component | exact r17 content | consequence for this project |
|---|---|---|
| vLLM #222 at `6206ac40` | shape-aware mixed EXL3, bounded persistent prefill plan, one-grid mixed K3/K4 for decode/small M ≤ 32, serial homogeneous K3/K4 prefill with FP32 tier accumulation | supersedes both appliance patches #210 and #219; blank/unset capacity follows scheduler batch size, while a positive `VLLM_EXL3_PREFILL_CAPACITY` retains our explicit memory/PP tradeoff |
| SparkInfer #105 at `45033eeb` | PCIe replay/IPC hardening plus complete wheel packaging of local runtime-JIT headers | fixes the r17-candidate PyNCCL fallback; the release must select `B12X_PCIE_ONESHOT_DMA` |
| vLLM #216 at `52201611` | semantic PCIe graph channels | retain the r17 head; do not reapply the older appliance overlay |
| SparkInfer #110 at `01917253` | W4A16 planning, tails, and capture updates | replaces the earlier W4A16 field-review patch series in the r17 composition |
| vLLM #212/#213 | compressed MLA physical-page stride and skipped pre-KV attention during FlashInfer autotune | relevant to capacity/startup reliability; confirm their log-selected paths and 512K behavior |
| vLLM #217/#218 | shared native CPU KV region and SWA/MTP/shared-prefix retention fixes | relevant to cache correctness even though production uses LMCache; keep in the source ledger |
| LMCache #7-#17 | bounded errors/diagnostics, failed-retrieval recompute, exact-prefix handling, deletion locking, store results, direct-I/O alignment, current/legacy FS keys, durable stores, restart accounting, and bounded L1 writeback | directly relevant to the 125 GiB DRAM plus 512 GiB NVMe production posture; exercise cold, warm, restart, and capacity gates |
| build PR #14 at `f5ba50b0` | exact source locks, wheel-content rejection, and immutable r17 publication | pin the digest, not only the mutable tag |

vLLM #145 is retained for NVFP4 scale compatibility even though its upstream
PR is closed/held. vLLM #214 is a DS4 profile and is inapplicable to this GLM
EXL3 qualification. At review time, #222, SparkInfer #105, and build PR #14
were still open; release inclusion must not be described as an upstream merge.
Their heads matched the immutable release locks. #222 was `UNSTABLE` only
because its `pre-commit` job remained queued; the other reported checks had
passed. Recheck all heads and checks immediately before building.

### Published release gate to reproduce

The release result used the Willfalco checkpoint at `d7d79c2`, TP4/DCP4,
MTP0, one sequence, graph 6, batch 2,048, maximum length 262,144, GMU 0.95,
and dynamic NVFP4 MLA KV:

| metric | published r17 | matched post-fix reference | acceptance band |
|---|---:|---:|---:|
| active KV | 855,808 tokens | not separately reported | exact startup value, allowing only documented allocator rounding |
| PP 3K | 3,761 tok/s | 3,799 tok/s | within 3% |
| PP 32K | 3,670 tok/s | 3,677 tok/s | within 3% |
| PP 128K | 3,306 tok/s | 3,302 tok/s | within 3% |

The earlier candidate's approximately 3,424/3,313/3,007 tok/s was invalid: the
wheel contained the PCIe CUDA source but omitted its local header, runtime
compilation failed, and vLLM silently selected PyNCCL. Before accepting the
published-image reproduction:

1. Verify the pulled image digest is
   `sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727`.
1. Verify the SparkInfer wheel contains `ipc_handle_registry.h` adjacent to
   the runtime-compiled PCIe source. Retain the resolved path and SHA-256.
1. Require a successful PCIe extension build/cache load and the explicit
   `B12X_PCIE_ONESHOT_DMA` selection line. Reject a PyNCCL fallback attributed
   to extension/header/import failure.
1. Run response sanity and one clean 3K/32K/128K pass. This is a release
   preflight, not the production decision.

### Rebase the turnkey without double-patching

The present appliance Dockerfile is based on r14 and applies
`patches/field-review-r14/manifest.json` before the compatibility scripts.
That manifest overlays ancestors or predecessors of vLLM #222/#216,
SparkInfer #105/#110, and the LMCache lifecycle fixes. It must not run against
r17: hash failure is the safe outcome, while a partial/double application
would produce an unreviewable stack.

Build one r17 turnkey candidate as follows:

1. Pin the base by the r17 registry digest and record the base image ID.
1. Replace the r14 manifest with an r17 source ledger proving, patch by patch,
   `included`, `superseded`, or `not applicable`. Never stack #210 or #219 over
   #222.
1. Make the build fail closed unless the composed vLLM, SparkInfer, and
   LMCache source IDs match the immutable release boundary and the PCIe local
   header is present.
1. Keep native-feature probes such as `patch_exl3_mixk.py`, but prove they
   report `already native` and do not mutate r17 source.
1. Update image labels, build fingerprint, and compile-cache namespace from
   r14 to r17. Add local tests for clean native skip and partial-tree failure.
1. Use a fresh r17 compile cache for the first qualification. Never reuse r14
   AOT artifacts. Reuse that same validated r17 cache across later NCCL arms.

### Matched production-shape A/B

After the published gate passes, compare the captured r14 control to the r17
turnkey candidate while changing only the runtime image/source composition.
The r17 arm must use the same checkpoint bytes, GPU order, TP4/DCP4, MTP5,
graph/Trellis 48, C8, batch 2,048, positive prefill capacity 1,024, GMU,
2,048-block override, 524,288 model length, dynamic-token NVFP4 KV, lossless
transport variables, and 125 GiB/512 GiB LMCache tiers as production.

Run this economical sequence:

1. `R14-A`: capture one fresh matched PP/TG/MAL/memory control on the alternate
   port using the preserved production launch bytes.
1. `R17-B1`: cold r17 compile/start, 3K/32K/128K PP, C1/C2/C4/C8 TG, MAL,
   available KV, idle/peak VRAM, LMCache cold/warm reuse, feature suite, 256K
   needles, and a near-maximum sentinel.
1. `R17-B2`: second clean start from the validated r17 compile cache; repeat
   the performance matrix and full five-depth near-maximum needle gate.
1. Run the structured-output, thinking/preserved-thinking, tool-call, aliases,
   LMCache restart, and C8 soak gates before declaring r17 qualified.

Populate this table before the transport campaign:

| arm | image/digest | KV tokens | PP 3K | PP 32K | PP 128K | TG C1/C4/C8 | MAL C1/C8 | idle/peak VRAM | LMCache cold/warm/restart | errors |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| R14-A | captured production control | 524,288 logical | | | | | | | | |
| R17-B1 | r17 turnkey candidate | | | | | | | | | |
| R17-B2 | same r17 candidate/cache | | | | | | | | | |

Do not compare the published MTP0/262K/C1 numbers directly with the production
MTP5/512K/C8 result. The former is a packaging/path preflight; the latter is
the promotion evidence. If r17 passes all gates, bind its exact candidate image
to `QUALIFIED_IMAGE` and use only that image for the NCCL/B12X phases below.
If it fails, restore the captured r14 control to port 8000 and defer tuning.

## Why NCCL still matters when B12X is enabled

The current route is hybrid:

| message/operation | selected path |
|---|---|
| TP all-reduce up to 64 KiB | B12X one-shot |
| fused all-reduce + RMSNorm up to 84 KiB | B12X fused path |
| middle-sized TP traffic | PyNCCL |
| lossless TP traffic at or above 6 MiB | B12X PCIe DMA |
| DCP exchange up to 16 query tokens | B12X A2A |
| larger DCP exchange | `ag_rs` through NCCL |
| full-CKV prefill gather | B12X orchestration/borrowed workspace, but the all-gather transport remains PyNCCL |

For hidden size 6,144 in BF16, one query row is 12,288 bytes. The meaningful
TP boundaries are therefore five rows (60 KiB), six rows (72 KiB), seven rows
(84 KiB), and 512 rows (6 MiB). With MTP5, each request presents six query
rows: C1/C2/C4/C8 correspond to 6/12/24/48 rows. The current DCP cap keeps C1
and C2 on B12X A2A while C4 and C8 use NCCL `ag_rs`.

This explains the expected shape of an NCCL result: the largest effects should
appear in long prefill, middle-sized TP messages, and C4/C8 DCP traffic. A flat
speedup across every cell would be suspicious and should trigger a cache-hit or
measurement audit.

## NCCL hypothesis from Jarrel's recipe

Jarrel's serving recipe uses:

```text
NCCL_MAX_NCHANNELS=4
NCCL_BUFFSIZE=1048576
```

For our NCCL 2.30 stack:

- `NCCL_MAX_NCHANNELS` has been deprecated since NCCL 2.17. Use
  `NCCL_MAX_CTAS=4`; if both names are present, NCCL applies the lower limit.
- `NCCL_MAX_CTAS=4` may free SM/CTA resources for simultaneous model compute,
  but can reduce collective bandwidth. NCCL's default is topology- and
  operation-selected, not a fixed number.
- `NCCL_BUFFSIZE=1048576` lowers the pairwise communication buffer from the
  documented 4 MiB default to 1 MiB. It may recover memory and alter pipelining,
  but can hurt large all-gather/reduce-scatter bandwidth.
- Do not infer memory recovery by multiplying one buffer by ranks/channels.
  Communicator allocation and protocol selection are more complicated. Record
  measured VRAM, startup peaks, and NCCL's actual channel plan.

NVIDIA classifies these as experiment/debug tuning controls and warns against
leaving them pinned without workload-specific evidence.

## Invariants, safety, and contamination controls

The core NCCL factorial begins only after Gate 0 qualifies r17 and changes only
NCCL CTA cap and buffer size. Hold all of the following fixed:

- exact qualified r17 turnkey image, model bytes, GPU order, driver, 280 W
  power cap, clocks policy, TP4/DCP4, MTP5, and sampler seed;
- batch 2,048, EXL3 prefill capacity 1,024, graph/Trellis 48, C8, fixed 2,048
  blocks, and 524,288 maximum context;
- dynamic-token NVFP4 MLA KV, lossless `F8_DMA=0`, 125 GiB LMCache DRAM, and
  bounded 512 GiB NVMe;
- `PCIE_CALIBRATION=off` and explicit
  `PCIE_DMA_MIN_BYTES=6291456`, so the B12X crossover cannot move while NCCL is
  being compared;
- CKV gather 140,000, prefetch 0, query-split state, owner merge, shared-expert
  stream state, and every other vLLM/SparkInfer variable from the captured
  control environment.

Use the same fresh, r17-only persistent compile-cache volume for all arms. The
source, checkpoint, and graph shape do not change, so reusing a validated AOT
cache avoids paying a cold compile per boot. Each process still captures its
own CUDA graphs. Use a unique state volume and a fresh LMCache filesystem root
for each arm so that calibration state and external prefix records cannot
cross arms. Keep matched prompt text across arms but use a new prompt seed for
each of the three repetitions. Verify GPU and external prefix-cache hits are
zero for every timed prefill.

Prime every shape once before timing it. One-time JIT or autotune work is not a
throughput result. Wait for LMCache stores and GPU work to drain between cells.

If the A1/A2 control geometric means differ by more than 2%, the run is not
stationary. Repeat the control/winner pair before drawing a conclusion.

## Gate 1: prove every variable is effective

Before entering the maintenance window, inspect the exact composed source in
the image and produce a consumer ledger for every candidate variable. A string
in `envs.py` or a registration table is not a runtime consumer.

Classify each variable as:

- **effective**: parsed and changes a selected path in the current EXL3/DCP4
  geometry;
- **effective but inapplicable**: parsed, but the current checkpoint/route does
  not reach it;
- **default-equivalent**: parsed but equal to current source behavior;
- **stale/inert**: no consumer in the pinned composed tree.

The prior r14 audit found the following. Re-run every source-consumer check on
the exact r17 composed tree before deciding that an item is still inert or
default-equivalent:

| variable/group | finding before the window | action |
|---|---|---|
| `VLLM_EXL3_PREFILL_CAPACITY` | r17 #222 owns the native implementation; blank/unset follows scheduler batch size and a positive value bounds the persistent plan | keep production 1,024 for the r14/r17 A/B; do not reapply #210; revisit capacity only in a separate memory/PP experiment |
| `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE`, `VLLM_B12X_MLA_SPEC_DECODE_MAX_Q` | r17 consumes both directly in `b12x_mla_sparse.py`; `auto` is safe at MTP5/C8/row-48, but forced `1` can reserve 64 rows and is invalid for the row-48 graph | the `auto` arm was safe but produced no repeatable win, so production retains `0` |
| `VLLM_CPP_AR_*` | C++ all-reduce cutoffs are inert while the selected backend is `b12x` | no performance boot; retain only if another backend consumes them |
| `VLLM_USE_B12X_MOE=1` | does not make the EXL3 target routed experts use B12X MoE; they use Trellis | document the actual consumers before retaining it as a 3.25-bpw performance claim |
| `SPARKINFER_MLA_SM120_PREFILL_MG=1`, fused/direct sparse indexer paths | active, correctness-sensitive GLM paths with prior long-context evidence | do not disable casually; any candidate must run the full needle gate |
| `SPARKINFER_W4A16_SMALL_M_DIRECT=1` | source-default small-M route | test only if profiling shows material time in the eligible kernel |

Also verify that the #222 serial prefill and mixed small-M plans are selected,
that the #105 PCIe extension is loadable, and that the log-selected transport
matches the operation-size routing table. This source audit is where
contradictory older ledger entries must be corrected. An inert flag cannot be
“qualified” by equal benchmark numbers.

## Phase 1: ordered NCCL factorial

Use five primary boots in this order:

| order | arm | `NCCL_MAX_CTAS` | `NCCL_BUFFSIZE` | purpose |
|---:|---|---:|---:|---|
| 1 | A1 | unset/auto | unset/4 MiB default | fresh control |
| 2 | B | 4 | unset/4 MiB default | isolate CTA cap |
| 3 | D | 4 | 1,048,576 | test Jarrel-equivalent combined hypothesis |
| 4 | C | unset/auto | 1,048,576 | isolate buffer reduction |
| 5 | A2 | unset/auto | unset/4 MiB default | detect drift |

Use `NCCL_MAX_CTAS`, not `NCCL_MAX_NCHANNELS`. Explicitly unset the deprecated
name in every arm. If A1's NCCL diagnostics show that automatic maximum/active
channels are already four or fewer for every relevant communicator, B is a
no-op and B/D may be skipped after recording that evidence.

Early exits:

- If C recovers less than 64 MiB/GPU and loses more than 2% PP, skip D if D has
  not already run, or reject the buffer component immediately.
- If B loses more than 2% PP or TG and recovers no useful headroom, reject the
  cap component.
- Stop an arm immediately on OOM, communicator hang, CUDA/NCCL error, request
  preemption, or a change to the exact 524,288 active-KV envelope.

### Launch template

Use `scripts/run-local-podman.sh`; do not use
`scripts/field_review_full_gate.sh`, which hard-codes field-review calibration
and model choices. Fill the three host paths from the captured production
inspect. Do not put API keys in the evidence directory.

```bash
export QUALIFIED_IMAGE=localhost/glm52-turnkey:r17-qualified
export EXL3_MODEL_ROOT=/absolute/read-only/checkpoint/snapshot
export EXL3_MARKER=/absolute/read-only/.download-complete
export COMMON_COMPILE_CACHE=glm52-nccl-ab-r17-cache
export TEST_PORT=18000

# Set ARM to A1, B, C, D, or A2 and use a fresh directory for each arm.
export ARM=A1
export ARM_LMCACHE_ROOT="$EVIDENCE_ROOT/lmcache-$ARM"
mkdir -p "$ARM_LMCACHE_ROOT"

env \
  -u TUNE_NCCL_MAX_CTAS \
  -u TUNE_NCCL_MAX_NCHANNELS \
  -u TUNE_NCCL_BUFFSIZE \
  IMAGE="$QUALIFIED_IMAGE" \
  NAME="glm52-r17-nccl-$ARM" PORT="$TEST_PORT" AUTH=none LANDING_PAGE=0 \
  MODEL_PROFILE=glm52-exl3 MODEL_VARIANT=exl3-tr3-3.25bpw \
  SERVED_MODEL_NAME="GLM-5.2 local-primary" \
  MODEL_DIR_HOST="$EXL3_MODEL_ROOT" DOWNLOAD_MARKER_HOST="$EXL3_MARKER" \
  CACHE_VOLUME="$COMMON_COMPILE_CACHE" \
  STATE_VOLUME="glm52-nccl-state-$ARM" \
  LMCACHE_DISK_HOST="$ARM_LMCACHE_ROOT" \
  DCP=4 MTP_TOKENS=5 MAX_MODEL_LEN=524288 MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=2048 VLLM_EXL3_PREFILL_CAPACITY=1024 \
  GPU_MEMORY_UTILIZATION=0.957 GPU_BLOCKS_OVERRIDE=2048 \
  OFFLOAD_FRACTION=0.5 PREFIX_CACHE_BACKEND=lmcache PREFIX_CACHE_DISK_GB=512 \
  KV_CACHE_DTYPE=nvfp4_ds_mla KV_SCALE_MODE=dynamic-token \
  DCP_CKV_GATHER_MAX_TOKENS=140000 DCP_CKV_PREFETCH_DEPTH=0 \
  B12X_PCIE_DMA=1 F8_DMA=0 PCIE_CALIBRATION=off \
  PCIE_DMA_MIN_BYTES=6291456 \
  MAX_CUDAGRAPH_CAPTURE_SIZE=48 \
  CUDAGRAPH_CAPTURE_SIZES=4,8,12,16,20,24,28,32,36,40,44,48 \
  VLLM_EXL3_TRELLIS_MAX_M=48 \
  TUNE_NCCL_DEBUG=INFO \
  TUNE_NCCL_DEBUG_SUBSYS=ENV,INIT,GRAPH,TUNING \
  TUNE_NCCL_DEBUG_FILE="/state/nccl-$ARM.%h.%p.log" \
  bash scripts/run-local-podman.sh
```

Resolve and save `podman image inspect "$QUALIFIED_IMAGE"` before the first
arm; a locally rebuilt mutable tag is not a result identity. The template is
the A arm. For a candidate, insert only the applicable
assignment after the three `-u` lines and before `IMAGE=...`:

```bash
# B
TUNE_NCCL_MAX_CTAS=4

# C
TUNE_NCCL_BUFFSIZE=1048576

# D
TUNE_NCCL_MAX_CTAS=4 TUNE_NCCL_BUFFSIZE=1048576
```

Because shell exports can leak from one arm into the next, prefer an explicit
`env -u ...` invocation per arm over an interactive sequence of `export`
commands. Confirm the entrypoint's `>>> tuning override` lines and NCCL INFO
output before accepting any result.

### Telemetry and benchmark contract

Capture telemetry at 200 ms throughout prime and measure phases:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu,memory.used,memory.free \
  --format=csv -lms 200 >"$EVIDENCE_ROOT/$ARM-gpu.csv" &
export TELEMETRY_PID=$!
```

For each arm, prime the 3K/32K/128K prefill shapes and the C1/C2/C4/C8 decode
shapes once. Then run three matched repetitions. The first command measures
cold unique-prefix PP and near-zero-context decode. The second measures decode
with a 32K live context.

```bash
# Untimed per-process shape prime.
python3 scripts/benchmark_serving.py \
  --base-url "http://127.0.0.1:$TEST_PORT" --model GLM-5.2 \
  --input-tokens 256 --output-tokens 128 \
  --concurrency 1,2,4,8 --requests-per-level 1 \
  --prefill-tokens 3072,32768,128000 --warmup 1 \
  --temperature 1.0 --seed 20260801 --timeout 1800 \
  --prompt-seed "nccl-ab-prime-$ARM" \
  --metadata campaign=nccl-b12x-prime --metadata arm="$ARM" \
  --out "$EVIDENCE_ROOT/$ARM-prime.json"

python3 scripts/benchmark_serving.py \
  --base-url "http://127.0.0.1:$TEST_PORT" --model GLM-5.2 \
  --input-tokens 32768 --output-tokens 128 \
  --concurrency 1,2,4,8 --requests-per-level 1 \
  --prefill-tokens "" --warmup 0 \
  --temperature 1.0 --seed 20260801 --timeout 1800 \
  --prompt-seed "nccl-ab-32k-prime-$ARM" \
  --metadata campaign=nccl-b12x-32k-prime --metadata arm="$ARM" \
  --out "$EVIDENCE_ROOT/$ARM-prime-32k.json"

for REP in 1 2 3; do
  python3 scripts/benchmark_serving.py \
    --base-url "http://127.0.0.1:$TEST_PORT" --model GLM-5.2 \
    --input-tokens 256 --output-tokens 1024 \
    --concurrency 1,2,4,8 --requests-per-level 1 \
    --prefill-tokens 3072,32768,128000 --warmup 1 \
    --temperature 1.0 --seed 20260801 --timeout 1800 \
    --prompt-seed "nccl-ab-r$REP" \
    --metadata campaign=nccl-b12x --metadata arm="$ARM" \
    --metadata repetition="$REP" \
    --out "$EVIDENCE_ROOT/$ARM-r$REP-serving.json"

  python3 scripts/benchmark_serving.py \
    --base-url "http://127.0.0.1:$TEST_PORT" --model GLM-5.2 \
    --input-tokens 32768 --output-tokens 1024 \
    --concurrency 1,2,4,8 --requests-per-level 1 \
    --prefill-tokens "" --warmup 0 \
    --temperature 1.0 --seed 20260801 --timeout 1800 \
    --prompt-seed "nccl-ab-32k-r$REP" \
    --metadata campaign=nccl-b12x-32k --metadata arm="$ARM" \
    --metadata repetition="$REP" \
    --out "$EVIDENCE_ROOT/$ARM-r$REP-serving-32k.json"
done
```

The throughput harness deliberately disables thinking and uses `ignore_eos`
to obtain a controlled 1,024-token decode sample. That is appropriate for a
transport comparison. The final quality gate must separately give GLM ample
reasoning room: run a real thinking-enabled GPQA sample with a 32,768-token
completion cap after the winner is selected.

For every arm, retain:

- resolved launch environment, image/checkpoint digests, compile-cache hit
  lines, CUDA-graph capture lines, and NCCL channel/algorithm/protocol output;
- model-load peak, target/draft arena sizes, available KV memory, exact active
  blocks/tokens, and idle/runtime free VRAM per rank;
- PP, TTFT, TPOT, aggregate TG, MTP mean acceptance length, accepted/drafted
  counters, p50/p95 latency, preemptions, failures, power, clocks, and tokens/J;
- Prometheus snapshots before and after each cell, including GPU and external
  prefix-cache hit counters.

Stop telemetry cleanly after each arm:

```bash
kill "$TELEMETRY_PID"
wait "$TELEMETRY_PID" 2>/dev/null || true
podman logs "glm52-r17-nccl-$ARM" >"$EVIDENCE_ROOT/$ARM-server.log" 2>&1
python3 scripts/field_review_log_audit.py \
  --log "$EVIDENCE_ROOT/$ARM-server.log" \
  --out "$EVIDENCE_ROOT/$ARM-log-audit.json"
```

### Core result table

Populate this table from JSON and logs, not the periodic vLLM logger alone.

| arm | idle free MiB/GPU | available KV GiB | PP 3K | PP 32K | PP 128K | TG C1 | TG C4 | TG C8 | MAL C1/C8 | avg W PP/TG | errors/preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| A1 | | | | | | | | | | | |
| B | | | | | | | | | | | |
| D | | | | | | | | | | | |
| C | | | | | | | | | | | |
| A2 | | | | | | | | | | | |

## Phase 2: recalibrate the B12X/NCCL boundary

Changing NCCL changes the competitor curve used by the B12X calibrator. Do not
recalibrate before selecting the NCCL winner.

With the winning NCCL variables fixed:

1. Boot with `PCIE_CALIBRATION=force`, `PCIE_DMA_MIN_BYTES=-1`, query split
   pinned to its control state, and CKV prefetch pinned to 0.
1. Record the measured lossless-DMA threshold and calibration artifact.
1. Run a matched A/B between the production-fixed 6 MiB threshold and the new
   measured threshold. Repeat the 32K/128K PP and C1/C4/C8 TG cells.
1. Promote the calibrated threshold only if its paired result beats both the
   fixed control and A1/A2 drift. Cache it by the complete topology, GPU order,
   NCCL library, and winner variables.

Then run one short diagnostic boot with `B12X_PCIE_DMA=0`. This is not an
isolated DMA test: the entrypoint disables both
`VLLM_ENABLE_PCIE_ALLREDUCE` and `VLLM_USE_B12X_DCP_A2A`. It leaves the B12X
model kernels enabled. Measure only 32K/128K PP and C1/C4/C8 TG. Its purpose is
to quantify the combined communication value of B12X and to catch a crossover
that is worse than pure NCCL; it is not a candidate default by itself.

## Phase 3: challenge B12X route choices

Run these only after the NCCL and DMA winners are fixed. Change one route at a
time and restore the winner between candidates.

| priority | question | control | candidate sequence | required cells / interpretation |
|---:|---|---|---|---|
| 1 | DCP A2A crossover | 16 query tokens | 32, then 48 only if 32 wins; test 8 only if C2 regresses | MTP5 C1/C2/C4/C8 maps to 6/12/24/48 rows. 32 moves C4 to B12X; 48 also moves C8. Measure TG, MAL, tail latency, and power. |
| 2 | owner merge | `VLLM_DCP_TOPK_OWNER_MERGE=1` | `0`, the source default and earlier DCP2 winner | 32K/128K PP plus C1/C4/C8. Retain only if DCP4 benefits. |
| 3 | shared-expert overlap | stream disabled | enable at source-default threshold 256; if it wins, compare threshold 16 | Watch output lifetime, first-request memory, exact arithmetic, PP, and C1/C8. A disabled stream makes the threshold inert. |
| 4 | output projection | `VLLM_USE_B12X_WO_PROJECTION=1` | `0` | short sentinel first; expand only if a profile shows material time in this path. |
| 5 | eligible FP8 GEMM route | `VLLM_USE_B12X_FP8_GEMM=1` | `0` | This flag selects eligible kernels; it does not quantize BF16 tensors. Compare PP/TG and exact outputs, but do not call it a weight-precision A/B. |
| 6 | query split | captured control value | toggle only after confirming its exact current effective state and minimum-context threshold | 8K/32K/64K/128K PP. Keep CKV gather/prefetch fixed. |

Do not spend GPU time on a variable with no source consumer. Do not combine
route candidates until each has an isolated result. Once isolated winners are
known, run one combined boot because memory lifetimes and transport crossovers
can interact.

Keep these out of this maintenance window unless new evidence appears:

- `F8_DMA=ring` or another compressed collective mode. It is quality-affecting
  and needs a separate maximum-context KLD/retrieval campaign.
- `NCCL_NTHREADS`. Test only if profiling after CTA=4 shows NCCL SM pressure or
  low clocks that channel control did not resolve.
- disabling sparse-indexer, MG prefill, or fused-indexer correctness paths.
  These are long-context critical and require more than a throughput smoke.
- CKV prefetch. The current value 0 already won the prior AIBeast study; revisit
  only after a source change or materially recovered workspace.

## Phase 4: challenge current NCCL pins

These are separate cleanup/portability questions and must not be folded into
the core factorial.

| current choice | challenge | plan |
|---|---|---|
| `NCCL_PROTO=LL,LL128,Simple` | This is equivalent to the supported-protocol default on platforms supporting LL128, while NVIDIA discourages forcing it and warns that unsupported LL128 can corrupt data. | Add a test-only way for the entrypoint to truly unset the variable; compare NCCL route logs and the winner cells. If identical, remove the production pin so NCCL can reject unsupported protocols automatically. An empty environment string is not accepted as proof of an unset. |
| `NCCL_P2P_LEVEL=SYS` | All AIBeast GPUs report `NODE`/NUMA 0; NCCL auto likely chooses the same P2P reach. `SYS` is broader than required. | First compare startup topology/route logs with the variable truly unset. Run the timed matrix only if the selected route changes. Re-test separately on rental topologies. |
| `NCCL_IB_DISABLE=1` | Could disable a real fabric on another host. | Retain for this single-node, no-IB AIBeast profile. Make it topology-derived for future multi-node/provider profiles; no AIBeast A/B is useful. |
| custom NCCL 2.30.4 preload | ABI and tuning are image-coupled. | Record the actual loaded library and compare only when a new image changes it. Never mix system NCCL and local-inference NCCL inside one result. |
| 6 MiB fixed B12X DMA threshold | The optimal point depends on NCCL's winning competitor curve. | Recalibrate after Phase 1, then A/B fixed versus measured as Phase 2. |

## Promotion gates

A candidate can become the AIBeast production route only if all of these pass:

1. The r17 base digest, turnkey image ID, composed source commits, wheel header,
   and selected `B12X_PCIE_ONESHOT_DMA` route match Gate 0 evidence. No r14
   overlay has mutated the native r17 tree.
1. Two clean starts using the same validated r17 compile namespace: no lock wait,
   hang, CUDA/NCCL error, OOM, fallback surprise, or stale-compile symptom.
1. Exactly 2,048 GPU KV blocks and 524,288 active logical tokens remain
   available, with enough first-request headroom for 128K and near-maximum
   requests.
1. PP geometric mean at 3K/32K/128K is no worse than 2% versus the matched
   control; C1 and aggregate C2/C4/C8 TG are no worse than 1-2% unless the
   candidate recovers a clearly more valuable safety margin.
1. The candidate improves at least one material outcome: at least 2% PP/TG,
   at least 64 MiB/GPU measured headroom, or a demonstrable tail/power gain.
   A result inside A1/A2 spread is noise.
1. Matched MTP MAL changes by no more than 0.05 and draft-token acceptance by
   no more than one percentage point unless the throughput improvement is
   explained by that stochastic difference and reproduces with another seed.
1. Tokens/J is not worse than 2%. Report average and peak power separately for
   PP and TG; AIBeast's 280 W cap can hide a throughput change behind clocks.
1. The full API suite passes: ordinary/thinking chat, streamed usage,
   preserved-thinking multi-turn, structured JSON, tool call/result round
   trip, and both served aliases.
1. The winner passes 256K and near-maximum five-depth needle retrieval with no
   degeneration, then a C8 soak. Use `scripts/feature_suite.py`,
   `scripts/verify_serving.py`, and `scripts/needle_matrix.py`.
1. Run a thinking-enabled GPQA sample last with temperature 1.0 and
   `max_tokens=32768`. Record truncations separately; do not starve GLM's
   reasoning and mislabel the result as model quality.
1. Restore the selected stack to port 8000, verify `GLM-5.2` and
   `local-primary`, and observe real agent traffic. Keep the original control
   launch bytes ready for immediate rollback.

Suggested final gates:

```bash
python3 scripts/feature_suite.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --out "$EVIDENCE_ROOT/winner-feature-suite.json"

python3 scripts/needle_matrix.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --max-model-len 524288 --reserve-tokens 4096 \
  --sizes 262144,520192 --depths 0.01,0.25,0.5,0.75,0.99 \
  --seeds 20260801,20260802 --timeout 3600 \
  --out "$EVIDENCE_ROOT/winner-needle-matrix.json"
```

## Decision order and rollback

The project's priority remains PP > TG > KV, provided the exact 512K envelope
and correctness gates survive. Apply decisions in this order:

1. Qualify or reject the r17 turnkey rebase against r14.
1. NCCL CTA/buffer winner on the qualified r17 image.
1. B12X DMA crossover winner under that NCCL configuration.
1. DCP A2A crossover.
1. Other isolated B12X route winners.
1. One combined qualification and two-start proof.

Before r17 is qualified, rollback means restoring the exact captured r14
container and environment. After r17 is qualified, capture its complete
known-good launch bytes; route-tuning rollback means:

- unset `NCCL_MAX_CTAS`, `NCCL_MAX_NCHANNELS`, and `NCCL_BUFFSIZE`;
- restore the qualified r17 production-resolved B12X environment, fixed 6 MiB
  threshold, MTP5/graph 48, 2,048 blocks, and cache configuration;
- restart the preserved production container on port 8000;
- verify `/health`, both model aliases, one thinking response, one structured
  output, and resumed agent traffic.

Do not delete the per-arm evidence or cache roots during the maintenance
window. Review them first; cleanup can be a separate, explicit operation.

## Result handoff

Store raw evidence under a dated directory such as:

```text
/mnt/fast/build/r17-nccl-b12x-20260801T000000Z/
```

Use names that retain arm and repetition, for example
`B-r2-serving-32k.json`, `D-server.log`, and `A2-gpu.csv`. Add the final
narrative to `TEST_RESULTS.md` only after the window. Update profile defaults
or the README only if a candidate passes every promotion gate.

Related project evidence:

- [Flagship qualification contract](../TEST_PLAN.md#flagship-glm-52-qualification)
- [r14 ordered maintenance matrix](glm52-r14-maintenance-plan.md#ordered-cost-bounded-matrix)
- [r17 vLLM implementation and validation](https://github.com/local-inference-lab/vllm/pull/222)
- [r17 PCIe packaging/lifecycle fix](https://github.com/local-inference-lab/sparkinfer/pull/105)
- [r17 immutable build composition](https://github.com/local-inference-lab/blackwell-llm-docker/pull/14)
- [3.25-bpw active-KV/offload qualification](glm52-3.25-offload-qualification.md)
- [Current parameter rationale and deferred route A/Bs](glm52-tuning-rationale.md#next-maintenance-window-325-bpw-runtime-route-ab-plan)
- [Benchmark comparison rules](benchmarks.md#performance-compare-like-with-like)
- [Field-review reproduction commands](field-review-results/2026-07-30-vast-46335896/COUNTER-VALIDATION.md#reproduction-commands)
