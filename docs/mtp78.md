# GLM MTP78 draft: native quantized, with compatibility alternatives

Extracted from the README. The `glm52-exl3` profile's speculative layer-78
draft: measurements, history, and compatibility paths.

The complete one-by-one justification for every non-default CLI parameter and
runtime environment value in the balanced profile is kept in
[`docs/glm52-tuning-rationale.md`](glm52-tuning-rationale.md). It records
the retained benefit, cost, evidence level, current-source defaults, and
settings removed as unsupported rather than treating the launch environment
as folklore.

The pinned Brandon revision now serializes layer 78 in the same native
rank-sliced EXL3/TR3 format as the target experts, so the current default performs
no checkpoint surgery. The earlier Brandon revision shipped a 19.3 GB BF16
layer; the measured graft and standalone override remain available for that
checkpoint and for controlled comparisons. MadeBy561's native draft is
different again: serialized NVFP4 experts with eligible dense pieces receiving
its MXFP8 load-time overlay.

## Measured (4-arm QC run, same box / config / prompts, 2026-07-25)

| draft (layer 78) | MAL | accept | decode tok/s | KV mem/GPU |
|---|---|---|---|---|
| BF16 (stock, 19.3 GB) | 3.528 | 84.3% | 49.6 | 5.27 GiB |
| **EXL3 3bpw grafted** (3.7 GB) | 3.517 | 83.9% | 49.5 | **8.92 GiB** |
| **EXL3 3bpw override** (3.7 GB, no surgery) | **3.548** | **84.9%** | **49.9** | **8.92 GiB** |
| NVFP4 (`lukealonso/GLM-5.2-NVFP4` MTP shards) | 3.531 | 84.4% | 49.5 | 8.5 GiB |

All four drafts accept identically; the EXL3 3bpw draft is the smallest and
edges NVFP4 on batched decode. Prefill/decode deltas are inside run-to-run
noise. Full writeup, methodology and the two NVFP4 config gotchas:
**https://gist.github.com/malaiwah/4bbb16bef2e336e94af165076cdba955**

**Historical graft headroom:** available KV memory in that earlier comparison
went 5.27 -> 8.92 GiB/GPU (**+69%**). Do not apply its advertised pool to the
balanced profile: the v31 cross-provider control's DCP2, MTP-5, graph capture
through C8, InstantTensor and portable runtime headroom exposed a measured
523,264 logical tokens on Runpod at GMU 0.976. A new base must reproduce that
capacity and pass retrieval before promotion.

**Speculation depth:** the cheaper draft also moves the optimum. Measured
(GSM8K n=30, +-0.5% noise floor): MTP-2 42.9 tok/s, MTP-3 51.5,
**MTP-5 53.4** (+3.7%). MTP-5 used to lose ~22% with the 19.3 GB BF16 draft;
with a small quantized draft the extra proposals win. MTP-5 is now the
qualified default and the capture/trellis windows are both 64 so C8 remains
inside the captured path. After restoration on production port 8000, real
agent traffic held MTP-5 MAL between 3.90 and 5.13 (58.0–82.7% average draft
acceptance across populated ten-second windows). This is the expected field
sanity check; the matched GSM8K comparison, rather than unmatched traffic,
decides MTP-5 versus MTP-3.

## Historical pre-v31 comparison: EXL3 + grafted MTP78

Measured 2026-07-26 on owned hardware (4x RTX PRO 6000 Blackwell, **280 W cap**,
TP4/DCP4-a2a, MTP-3, 512K context, DRAM KV offload on, clean single-stream):

| historical fp8-KV comparison (pre-v29) | decode C1 | MAL / accept | KV/GPU | KV pool | 505K needle |
|---|---|---|---|---|---|
| GLM-5.2 NVFP4-NF3 hybrid (previous prod, calibrated nvfp4 KV) | 119.2 tok/s | ~3.5 / 0.83 | 4.64 GiB | 537,600 tok | 7/7 |
| **EXL3-TR3 3bpw + MTP78, fp8 KV** | 112.4 tok/s | 3.471 / 0.824 | **8.89 GiB** | **697,600 tok** | **6/6** |

**The historical trade was ~6% slower decode for ~30% more KV pool, and
vision.** The
EXL3 weights are ~7 GiB/rank smaller than the hybrid's, which is what pays for
the bigger pool even though fp8 KV costs ~1.7x the bytes per token that nvfp4
would. Long-context retrieval is verified clean (6/6 at depths to 490K inside a
505K request).

> **Why calibrated scales are non-negotiable.** Earlier uncalibrated
> `nvfp4_ds_mla` experiments failed long needles with degenerate output while
> short quality, vision, and structured-output checks still passed. The v29
> runtime ships the GLM-5.2-specific calibrated MLA outer-scale file and the
> entrypoint now selects it explicitly and verifies its SHA-256 before exporting
> `VLLM_NVFP4_MLA_SCALES_FILE`; the configurator
> refuses this KV dtype for model families without an equivalent calibration.
> Release qualification still runs cold long-context needles—calibration is
> not a reason to skip the causal test.

GG v20-r9 offers `KV_SCALE_MODE=dynamic-token`. That paired mode selects
the 368-byte FP8-RoPE record, stores an outer scale per token, and cannot be
combined with the static scale file. On AIBeast the appliance measured mean KLD
`0.1167701185` twice across 2,047 positions, then retrieved 5/5 needles without
degeneration from two exact 510,533-token prompts. It is therefore the flagship
EXL3 default; the final scheduler also passed at 521,275 tokens.
`static-calibrated` remains the compatibility choice for variants
that have not passed the same gate.

Vision consumed about 1.31 GiB/GPU in an earlier graft and 1.99 GiB/GPU in the
final v20 qualification. Treat it as a distinct opt-in profile: re-run the
configurator's memory and retrieval gates after enabling it rather than
assuming the text-only 524K/concurrency envelope is unchanged.

MTP acceptance remains healthy on short vision prompts, but the final v20
32K gate collapsed to MAL 1.25–1.50 alongside degenerate retrieval. Exposing
the nested `lm_head` is necessary, but it is not sufficient evidence of
long-context correctness.

Two settings are load-bearing in the v29 graft configuration:

- `ONLINE_QUANT=none` — serving presets that default to an mxfp8 online overlay
  make EXL3 refuse with `quantization_config is only supported when ...`. The
  entrypoint sets this explicitly.
- `VLLM_EXL3_TRELLIS_MIN_M` is **unset**. v29 stamps the draft role at
  construction: target layers retain m=4 while MTP draft layers advertise the
  capturable m=1 floor they require. A global override defeats that
  role-specific choice. Advanced A/B tests may still use
  `TUNE_VLLM_EXL3_TRELLIS_MIN_M`.

A trellis (or BF16) MTP draft also needs `moe_backend=triton` **separately from**
the target's backend — a rank-3 trellis tensor is not a fused expert weight.

## Separate EXL3 draft override (experimental; do not use for production)

The overlay works as a *separate draft directory* — leave the base checkpoint
untouched and add one field:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3,
                       "moe_backend":"triton","draft_sample_method":"greedy",
                       "model":"/path/to/GLM-5.2-EXL3-TR3-MTP78/3bpw-keep0"}'
```

v29 supports this separately rank-sliced draft by stamping its role during
construction. The turnkey still defaults to the in-place graft because it
avoids a second draft directory and is the release configuration exercised by
the automated boot verifier.

## Model quality (the target model, unrelated to the draft)

648 samples per quant, Z.ai eval settings (temp 1.0, top_p 0.95), pass@1:

| benchmark | Original BF16 | Hybrid MXFP8-NVFP4-NF3 | EXL3 3.0bpw |
|---|---|---|---|
| AIME 2026 (30x4) | 99.2 | 97.5 | **99.2** |
| HMMT Feb 2026 (33x4) | 92.5 | 97.0 | **95.5** |
| GPQA Diamond (198x2) | 91.2 | 89.4 | **91.4** |

Both quants are statistically indistinguishable from BF16 — every delta is
within sampling noise (~±3).

> **Note — vLLM patch requirement:** loading a rank-sliced EXL3 MTP overlay
> needs a one-hunk fix in vLLM's `deepseek_mtp.py` (`load_weights` misses the
> rank-sliced name normalization; upstream PR:
> [voipmonitor/vllm#11](https://github.com/voipmonitor/vllm/pull/11)). The
> entrypoint applies it to the image's vLLM automatically at boot
> (`scripts/patch_deepseek_mtp.py`, idempotent); if the anchor is missing in
> a future image build, the template falls back to the BF16 draft rather
> than fail.
