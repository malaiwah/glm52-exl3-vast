## AIBoss Blackwell GPU evidence (2026-08-07)

The corrected stack is GPU-qualified for both the real mixed-Trellis target path and its native rank-sliced MTP1 draft path on one RTX 5090.

### Provenance

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.57.04, 400 W limit
- Base: GGv20r28 `sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0` (CUDA 13.2.1)
- Model: `malaiwah/GLM-5.2-SIQ-Fruit-Instruct@acd11237ebe808efb8fe688ee50bf24c7fd466a0`
- Patch heads: b12x #126 `6022e6e7c7ea1199a06a27cf5a777c2804b13cfb`; vLLM #250 `2a34b0760f4d9bbd5d2ff6809238593098bd46ff`
- Derivative image ID: `49c108672a18fef0a0a0860d14e6ef0b7b4fc2746327ba95658430140ccbcedb`
- r28's legacy `sparkinfer` package name was identifier-normalized only; both PR diffs passed `git apply --check` against the exact installed sources.

### The first GPU run caught and fixed a real coverage miss

Warming only the maximum live row count for a power-of-two capacity bucket did not cover a five-row request: Triton also specializes the runtime `live_numel` scalar by divisibility/alignment. The first MTP extension then exposed a second false equivalence: the target uses `int32` route IDs while the native draft uses `int64`. The final b12x head warms both scalar classes and both ID dtypes, and keys the prewarm cache by dtype. Exact-r28 focused tests are 6/6 green.

### Matched 2K serving smoke (TP1/DCP1, eager, B12X sparse MLA, NVFP4 MLA KV, seqs=4, GMU=.75)

| | r28 control | patched | delta |
|---|---:|---:|---:|
| route variants pre-KV | 0 | 32 | +32 |
| measured module residency | lazy/unmeasured | 2.0 MiB | +2.0 MiB |
| logical KV tokens | 2,926,208 | 2,925,952 | -256 (-0.009%) |
| cold engine init | 24.44 s | 29.11 s | +4.67 s |
| CC1 decode smoke | 35.1 tok/s | 36.0 tok/s | no regression signal |
| post-start route-pack JIT | small-prefix + sort | none for 1/2/5/8/9 tokens | fixed |

Generated outputs were byte-identical. On populated JIT caches, both engine initializations were 7.04 s.

### Reverse-order warm-cache prefill repeat

Five post-warmup samples per cell:

| rows | control | patched | delta |
|---:|---:|---:|---:|
| 256 | 8,606 | 8,501 tok/s | -1.22% |
| 1,024 | 33,789 | 34,274 tok/s | +1.44% |
| 2,048 | 66,415 | 66,374 tok/s | -0.06% |
| 4,096 | 74,905 | 74,997 tok/s | +0.12% |
| 6,144 | 108,056 | 108,238 tok/s | +0.17% |

This is steady-state parity within noise; the value is predictable residency and elimination of live-serving route-pack JIT, not throughput.

### Long-context path gate

At max model length/batch 32,768, 40 variants warmed for 2.0 MiB, KV capacity was 2,343,680 tokens, and no `_pack_topk_routes_*` kernel JITed after engine start. A 4,026-token retrieval prompt returned the planted name `Lily`; `FRUIT-LONG-SERVE-OK`.

### Native MTP-draft closure

The native MTP1 run uses FP8 MLA KV, TP1/DCP1, eager mode, max model length/batch 2,048, max sequences 4, and GMU 0.75. The control target-only patch still lazily loaded `_pack_topk_routes_small_prefix_kernel` and `_pack_topk_routes_sort_kernel` from the distinct rank-sliced draft runtime after engine start.

| | target-only warmup | final target + draft warmup | delta |
|---|---:|---:|---:|
| logical KV tokens | 1,865,920 | 1,865,728 | -192 (-0.010%) |
| accounted module residency | draft lazy/unmeasured | +2.0 MiB | +2.0 MiB |
| post-start draft route-pack JIT | small-prefix + sort | none | fixed |
| accepted drafts | 451/571 (79.0%) | 451/571 (79.0%) | identical |
| mean accepted length | 1.790 | 1.790 | identical |
| output | expected story | byte-identical story | unchanged |

The cold final run took 43.04 s versus 30.27 s for the target-only image because the empty cache now compiles the draft specializations during initialization. With the compilation cache populated, the exact final image initialized in 6.80 s, served the same output, retained the same acceptance, emitted `FRUIT-MTP-OK`, and loaded no `_pack_topk_routes_*` kernels after engine start. The decode smoke was 58.8 versus 63.8 tok/s, but that single sample is clock/cache noise and is not presented as a speedup.

Raw-log SHA-256:

- cold final: `aebc5c2df64641a22e60b349362e3fc48857ced59c096746adf906fe2f8a6bc7`
- exact-head warm cache: `9628800b5137a1ecd1bb5f2a9f4fdf9f070b2fea448249ecc192b492809d9c11`

### Production corroboration

A read-only audit of the untouched AIBeast r28 production container found the exact failure family in an MTP3 TP4/DCP4 service. At 2026-08-07 10:11:29, TP3 entered `_apply_mixed_rank_sliced` -> `run_mixed_trellis` -> `pack_topk_routes_by_expert` -> `_pack_topk_routes_post_prefix_kernel`; Triton's `load_binary()` failed with `RuntimeError: Triton Error [CUDA]: out of memory`. Scheduler telemetry reported only 1.18% active KV usage and two running requests, excluding active-request KV pressure as the cause. The engine timed out and restarted. The new process then warned that `post_prefix` and `sort` were again JIT-loading during inference.

This trace establishes that late route-pack module residency is already capable of crashing the production MTP stack. It does not by itself identify that particular launch as target versus draft; the final patch closes both owners.

### Qualification boundary

This closes the measured target and native-MTP route-pack residency gaps for the TP1 eager Blackwell path. TP4/DCP4 and CUDA-graph capture remain pending; AIBeast production was not interrupted.
