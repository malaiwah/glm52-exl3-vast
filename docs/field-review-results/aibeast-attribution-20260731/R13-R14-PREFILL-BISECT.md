# r13 -> r14 GLM-5.2 prefill-regression bisect

Date: 2026-07-31

## Conclusion

The roughly 23% r13-to-r14 prefill regression is caused by the paired one-grid
mixed-K3/K4 Trellis feature introduced by vLLM `2258b4e0f0f5` and SparkInfer
`f27c41d25667`. It is not caused by vLLM #175, #198, the common consolidated
MoE API, KV assignment, sparse-prefill configuration, or vLLM #210.

An r14 hybrid retaining the complete r14 stack but replacing only its EXL3
mixed-K executor with the exact r13 serial two-tier implementation recovered
prefill to within 0.5-1.1% of r13 at all tested depths. It completed the full
35-request matrix with zero failures.

## Exact source boundary

The published image labels show that both releases use the same vLLM base
`f978d009fab996617f9a3cadef36ce727bcd83cd` and the same revisions of vLLM
#145, #175, and #198. Both include SparkInfer #92 at the same revision.

The relevant changes are:

| Project | r13 head / parent | r14 feature | r14 hardening |
|---|---|---|---|
| vLLM #190 | `e55627045ac6` | `2258b4e0f0f5` - execute mixed K3/K4 experts in one grid | `46c36c0f89a4` - harden integration contracts |
| SparkInfer #104 | not present | `f27c41d25667` - add one-grid mixed K3/K4 Trellis path | `241718d427e0` - harden launch contracts |

SparkInfer's base also advanced from `36cade0` to `9b852b2`, but the intervening
commits only optimize SM121 paged attention. AIBeast is SM120, and the r14
hybrid retains those newer commits while recovering performance.

The two hardening commits add validation, correct cache keys, bounds checks,
and explicit route-block plumbing; they do not replace the core kernel or its
tile geometry. The feature pair above is therefore the commit boundary to
filter first.

## What changed in the hot path

The r13 runtime is not a bare stock EXL3 implementation. Its deployed
site-packages file contains the appliance's `EXL3-MIXK-PATCH` series, which:

- partitions 192 K3 and 64 K4 experts into homogeneous tiers;
- executes two serial SparkInfer plans and accumulates their results;
- uses route block 8 for decode and block 64 for prefill;
- aliases scratch arenas across the two tiers.

r14 replaces that path with one cooperative grid spanning both bitrates. Its
current mixed launch uses route block 8 for decode and prefill and the safe
mixed tile `(128, 128, 32, 512)`.

## Isolated kernel evidence

The upstream SparkInfer mixed-Trellis benchmark did not represent the actual
r13 predecessor: its serial control used the mixed tile and block 8, rather
than the serial tile and r13's prefill block 64. With the benchmark corrected
to use serial tile `(64, 256, 64, 256)`, the relevant comparison on one AIBeast
RTX PRO 6000 Blackwell GPU was:

| Active rows | r13 serial block 64 | r14 mixed block 8 | Serial speedup |
|---:|---:|---:|---:|
| 512 | 689.22 us | 1,212.00 us | 1.76x |
| 2,048 | 1,917.54 us | 4,636.96 us | 2.42x |
| 3,072 | 2,815.07 us | 8,732.74 us | 3.10x |

At block 8 on both sides, the one-grid path is 1.65x faster at 32 rows and
1.03x faster at 3,072 rows. This explains why the new path can help decode and
why a synthetic comparison that omits the serial block-64 prefill geometry
misses the server regression.

Trying block 64 on the mixed path is not currently a valid fix. Compilation
fails on SM120 because SparkInfer has no register-count specialization for
`(256, 4, 8, 8, False)`.

## Full-model confirmation

All arms used the test contract in `RESULTS.md`: the same 3.25-bpw snapshot,
TP4/DCP4, MTP5, dynamic NVFP4 MLA KV, fixed 524,288 active KV tokens, 2,048
batched tokens, 8 sequences, CKV gather 140,000, vision off, and LMCache
125-GiB DRAM plus 512-GiB NVMe. Each arm had a fresh compile-cache and LMCache
namespace; every tested shape was warmed before the common measured corpus.

| Target prompt | r13 | pristine r14 | r14 + r13 serial executor | Hybrid vs r13 | Hybrid vs r14 |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 2,322.7 | 1,772.9 | 2,296.7 | -1.12% | +29.54% |
| 32,768 | 2,180.2 | 1,660.2 | 2,157.6 | -1.04% | +29.96% |
| 131,072 | 2,057.5 | 1,583.1 | 2,046.3 | -0.54% | +29.26% |

The independent prime corpus produced 2,350.5 / 2,247.2 / 2,105.0 tok/s,
which gives the same conclusion.

Decode is not the primary attribution signal because MTP acceptance varies
between stochastic runs. The measured hybrid C1/C2/C4/C8 aggregate rates were
103.4 / 139.2 / 201.5 / 269.1 tok/s with mean acceptance lengths 4.070 / 3.746 /
4.035 / 3.988. All requests completed without an exception, OOM, or preemption.

## Shape-aware implementation qualification

The production candidate implements the recommended split instead of copying
the complete r13 executor: one-grid/block-8 is retained through the decode
capture limit, then serial homogeneous K3/K4 block-64 plans handle prefill.
Both paths share one tier-ordered rotation allocation, and the two serial plans
share one scratch arena because they execute sequentially.

| Target prompt | Shape-aware r14 | vs pristine r14 | vs r13 |
|---:|---:|---:|---:|
| 3,072 | 2,283.2 tok/s | +28.78% | -1.70% |
| 32,768 | 2,146.6 tok/s | +29.30% | -1.54% |
| 131,072 | 2,033.1 tok/s | +28.43% | -1.18% |

The independent prime corpus measured 2,343.1 / 2,235.2 / 2,096.3 tok/s.
The common C1/C2/C4/C8 aggregate decode rates were 98.45 / 165.25 / 198.76 /
248.07 tok/s with mean acceptance lengths 3.518 / 4.164 / 3.926 / 3.836.
The full 35-request matrix completed with zero failures and zero preemptions.

Model loading fell from 83.53 GiB/GPU in pristine r14 to 82.81 GiB/GPU after
sharing the rotations. The server still exposed 524,288 active KV tokens and
retained 1.5 GiB/GPU after the full matrix. Exact retrieval gates passed at
128K and with a 509,022-token prompt: all needles at 10%, 50%, and 90% depth
were recovered without degeneration.

The focused EXL3 suite passed 20/20 tests in the exact r14 runtime image. Ruff,
`py_compile`, and `git diff --check` also passed.

Upstream review:

- vLLM implementation: https://github.com/local-inference-lab/vllm/pull/219
- SparkInfer kernel/benchmark follow-up:
  https://github.com/local-inference-lab/sparkinfer/issues/107

## Elimination of the alternative hypotheses

- **vLLM #175 / sparse query splitting:** r13 and r14 image labels pin the
  same #175 commit. The hybrid retains r14's entire attention and DCP path.
- **vLLM #198 / repeatable activation profiling:** both releases pin the same
  #198 commit. The hybrid retains r14 profiling. More importantly, every arm
  forces exactly 2,048 GPU KV blocks, so assigned KV cannot explain PP.
- **vLLM #145 / outer-scale wiring:** both releases pin the same #145 commit.
- **SparkInfer #92 / consolidated fused-MoE API:** both releases pin the same
  #92 commit. The recovered serial executor runs on the r14 SparkInfer base.
- **r14 hardening overhead:** checks occur around planning/launch, but the
  isolated kernel result exposes the large geometry difference itself. The
  hardening commits do not alter the core mixed kernel's block-8 geometry.
- **vLLM #210:** absent from both the pristine-r14 control and this hybrid.

## Recommended upstream next step

Do not revert the hardening checks. Keep the correctness contracts, then make
the mixed path shape-aware:

1. retain one-grid/block-8 for decode and small row counts;
2. restore the serial homogeneous-tier block-64 path for prefill until an
   efficient SM120 mixed block-64 specialization exists;
3. add the actual serial block-64 predecessor to the upstream A/B benchmark;
4. gate promotion on full-server 3K/32K/128K prefill, not only per-layer mixed
   before/after timing and decode throughput.

## Reproducibility artifacts

- Hybrid image: `localhost/glm52-turnkey:r14-serial-mixk-bisect-v1`, image ID
  `b8642f8402377a6cffef6a2882ce880d4df7683abd96b4563a2afcfa471230a6`.
- r13 runtime EXL3 SHA-256:
  `6fe311833b1a5f3b651881562c43e10f959933103f9290ece1cea006e5c6e646`.
- r14 runtime EXL3 SHA-256:
  `59a0915f6326a3db189455d962a565412065bd79494c3675734a0e24c1107df2`.
- Corrected microbenchmark SHA-256:
  `8ce6cc626dd2f859d835f1ddf6b5d71e747b287e4f0debef27bc27f97a90a7a2`.
- Common measured result: `r14-serial-measured.json`.
- Independent prime result: `r14-serial-prime.json`.
- Shape-aware image: `localhost/glm52-turnkey:r14-shape-aware-mixk-v2`, image
  ID `b30010a981283e6500283fc5d95261fbb1f6f950cb7d5a480abf8a1b0261a92a`.
- Shape-aware common result: `r14-shape-v2-measured.json`.
- Shape-aware independent prime result: `r14-shape-v2-prime.json`.
- Retrieval gates: `r14-shape-v2-needle-128k.json` and
  `r14-shape-v2-needle-508k.json`.
- Full server log: `r14-shape-v2-server.log`.
