# AIBeast r13 / r14 / vLLM #210 attribution

Date: 2026-07-31

## Postscript: r14 regression fix

The r13 rollback recorded below was the safe interim outcome before the
commit-level bisect completed. The one-grid mixed-K3/K4 path was subsequently
identified as the r14 prefill regression source, and a shape-aware r14 build
was qualified: one-grid/block-8 remains active for decode while serial
homogeneous K3/K4 block-64 plans serve prefill. It recovered 28.4-29.3% over
pristine r14 at 3K/32K/128K, passed the full 35-request matrix without errors
or preemptions, and retrieved all three needles in a 509,022-token prompt.

The implementation is under review in
https://github.com/local-inference-lab/vllm/pull/219, with the SparkInfer
kernel follow-up at https://github.com/local-inference-lab/sparkinfer/issues/107.
See `R13-R14-PREFILL-BISECT.md` and the `r14-shape-v2-*` artifacts for the
complete qualification.

## Outcome

- The r14 integration stack has a reproducible prefill regression versus the r13 production image under the matched GLM-5.2 profile: about 23% at 3K, 32K, and 128K.
- The follow-up commit-level bisect identifies r14's paired one-grid mixed-K3/K4 Trellis feature as the cause. An r14 hybrid changing only the EXL3 mixed-K executor recovered to within 0.5-1.1% of r13. See `R13-R14-PREFILL-BISECT.md`.
- Enabling `VLLM_EXL3_PREFILL_CAPACITY=1024` through vLLM #210 on otherwise-pristine r14 recovers 560 MiB of target-plus-draft Trellis arenas per rank and raises measured available-KV headroom by 0.55 GiB/GPU.
- The direct #210 prefill cost, measured with three paired cache-cold prompt sets, is a median 4.29% at 32K and 4.37% at 128K. The 3K result is noisier and has a 15.28% median loss.
- #210 did not produce a uniform decode penalty. The observed C1/C2/C4/C8 aggregate deltas were +6.39%, +2.81%, -7.40%, and +2.92%; the variation tracks speculative-acceptance differences.
- The r14 + #210 candidate is not promoted to the PP-first production profile. The exact r13 production container is restored on port 8000.
- Post-rollback verification: `/health` returned 200, `/v1/models` exposed both `GLM-5.2` and `local-primary`, and a non-thinking chat completion through `local-primary` returned exactly `READY`. Port 18000 was closed and no attribution container remained running.

## Test contract

- Host: 4x NVIDIA RTX PRO 6000 Blackwell, 96 GB each, 280 W/card power limit.
- Driver: 595.71.05; CUDA: 13.2.
- Model: `willfalco/GLM-5.2-EXL3-TR3-3.25bpw`, snapshot `61d2b6b757f6a4ac7098a78d861f2033497532dc`.
- TP4 / DCP4, native MTP5 probabilistic draft, dynamic NVFP4 MLA KV, FP8 RoPE.
- 524,288 max model length; fixed 2,048 GPU blocks = 524,288 active KV tokens.
- 8 max sequences; 2,048 max batched tokens; graph and Trellis capacity 48.
- CKV gather 140,000; DCP workspace 1,024 MiB; vision disabled.
- LMCache: 125 GiB DRAM plus bounded 512 GiB NVMe.
- Benchmark endpoint: isolated port 18000 with no user/agent traffic.
- Warm-up covered every tested prefill and C1/C2/C4/C8 decode shape before measurement.
- Measured decode: 8 requests per concurrency, 1,024-token inputs, 512 generated tokens, temperature 1.0, common seed and prompt corpus.
- Measured prefill: cache-cold prompts at 3,072, 32,768, and 131,072 target tokens.

## Source controls

| Arm | Image | Relevant source state |
|---|---|---|
| r13 control | `localhost/glm52-turnkey:r13-prod-v2` (`ccce56e0cd23...`) | Published r13 integration; no #210 knob |
| r14 control | `localhost/glm52-turnkey:r14-control-v1` (`6525268ebdbb...`) | Pristine r14 source plus only the required appliance ABI compatibility fix; all eight field-review patches absent |
| r14 + #210 | `localhost/glm52-turnkey:r14-vllm210-v1` (`d468b5b1c1fa...`) | Same r14 control plus only the vLLM #210/#211 manifest component and the identical ABI fix |

r13 and r14 share vLLM base commit `f978d009fab996617f9a3cadef36ce727bcd83cd`, but r14 advances the integrated vLLM #190 head and adds SparkInfer #104's mixed-K path. The r13-to-r14 comparison is therefore an integration-release comparison; the r14-to-r14+#210 comparison isolates #210.

## Prefill throughput

The first table is the common fully warmed measurement corpus.

| Target prompt | r13 tok/s | r14 tok/s | r14 + #210 tok/s | r13 -> r14 | r14 -> #210 |
|---:|---:|---:|---:|---:|---:|
| 3,072 | 2,322.7 | 1,772.9 | 1,502.0 | -23.67% | -15.28% |
| 32,768 | 2,180.2 | 1,660.2 | 1,597.0 | -23.85% | -3.80% |
| 131,072 | 2,057.5 | 1,583.1 | 1,527.8 | -23.06% | -3.49% |

Two additional A-B-A repetitions used paired cache-cold prompt corpora. Across all three paired observations, the direct #210 deltas were:

| Target prompt | Median | Range |
|---:|---:|---:|
| 3,072 | -15.28% | -21.61% to -4.32% |
| 32,768 | -4.29% | -4.83% to -3.80% |
| 131,072 | -4.37% | -5.40% to -3.49% |

The initial per-arm prime artifacts are excluded from attribution because first-use compilation affected early r14 shapes.

## Decode throughput and speculative acceptance

| Concurrency | r13 tok/s | r14 tok/s | r14 + #210 tok/s | r13 -> r14 | r14 -> #210 |
|---:|---:|---:|---:|---:|---:|
| C1 | 97.25 | 111.45 | 118.57 | +14.60% | +6.39% |
| C2 | 155.89 | 169.83 | 174.60 | +8.95% | +2.81% |
| C4 | 193.75 | 187.59 | 173.71 | -3.18% | -7.40% |
| C8 | 246.81 | 234.65 | 241.49 | -4.93% | +2.92% |

Mean acceptance length by C1/C2/C4/C8:

- r13: 3.789 / 4.270 / 3.855 / 3.824
- r14: 4.136 / 4.489 / 4.106 / 3.729
- r14 + #210: 4.397 / 4.605 / 3.693 / 4.142

All arms completed 35 measured requests with zero failures and zero preemptions. Since the #210 decode deltas are not monotonic and move with acceptance length, this matrix does not support a claim of a systematic TG loss from #210.

## Memory attribution

| Metric per GPU/rank | r13 | r14 | r14 + #210 |
|---|---:|---:|---:|
| Loaded model | 82.83 GiB | 83.53 GiB | 83.53 GiB |
| Target EXL3 buffers | 773.9 MiB arena | 538.3 MiB | 298.3 MiB |
| Draft EXL3 arena | 773.9 MiB | 734.1 MiB | 414.1 MiB |
| Retained graph pool | 0.58 GiB | 0.53 GiB | 0.54 GiB |
| Available KV before fixed override | 4.67 GiB | 4.29 GiB | 4.84 GiB |
| Active KV after override | 524,288 | 524,288 | 524,288 |

Direct #210 arena reduction on r14:

- Target: 538.3 - 298.3 = 240.0 MiB/rank.
- Draft: 734.1 - 414.1 = 320.0 MiB/rank.
- Total: 560.0 MiB/rank.
- Observed available-KV increase: 4.84 - 4.29 = 0.55 GiB/GPU.

The earlier roughly 1 GiB/rank headline referred to a larger combined patch campaign. It is not the measured gain from #210 alone in this MTP5 mixed-K profile.

## Decision and next experiment

Keep r13 production for the PP-first coding/agent workload. Do not enable capacity 1,024 by default on r14. The next useful maintenance experiment is:

1. Keep the r14 correctness hardening but route prefill through the serial homogeneous-tier block-64 path until the mixed kernel has an efficient SM120 large-row specialization.
2. Add the actual r13 serial block-64 predecessor to SparkInfer's mixed-Trellis benchmark.
3. Test #210 at an intermediate capacity such as 1,536 after the r14 prefill regression is fixed.
4. Repeat deterministic decode with acceptance held as constant as practical before assigning a small TG delta to #210.
