# GLM-5.2 flagship qualification evidence

Extracted from the README's flagship model card: the end-to-end evidence
behind the `glm52-exl3` profile and its variants. The 3.25-bpw active-KV and
offload matrix has its own report in
[glm52-3.25-offload-qualification.md](glm52-3.25-offload-qualification.md);
per-flag justifications are in
[glm52-tuning-rationale.md](glm52-tuning-rationale.md).

## Terminal-Bench reproduction

An independent
[Terminal-Bench 2.1 reproduction on this Brandon EXL3/TR3 checkpoint](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/discussions/1)
reported “69 pass · 15 model fail · 4 infrastructure error”: 78.4% over the
88 completed trials, or 82.1% after excluding infrastructure errors. Z.ai's
[original GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) reports
81.0% with Terminus-2. The quantized run is thus within 2.6 percentage points
of the vendor score and crosses it on the quality-only denominator. That is
strong end-to-end evidence that the quant retains the original model's
agentic capability, but not a controlled quantization-only A/B: serving
context, runtime, harness state, and error accounting differ, so the two
percentages must not be called statistically equivalent.

## 3.25-bpw variant, LMCache, and the r13/r14 narrative

The mixed 3.25-bpw profile is a qualified quality/performance trade, not a
new default. On the original AIBeast/r11 control it passed the complete OpenAI
feature suite and an actual 522,360-token five-depth retrieval with no
degeneration. Its isolated 32K/128K prefill was 2,407.9/2,262.1 tok/s;
aggregate decode was 128.0 tok/s at C1 and 285.3 at C8, with MTP5 mean
acceptance lengths 5.74/5.08. Against the 3.0-bpw control that is roughly 3–6%
slower prefill, 14% slower C1, and 26% slower C8.

The final r13 appliance image repeated the full API gate and the same exact
522,360-token 5/5 retrieval on the production endpoint. With ordinary agent
traffic sharing the service, unique-prefix PP was 2,197.1/2,062.8 tok/s at
32K/128K. Aggregate TG was 108.8/173.0/255.0/332.9 tok/s at C1/C2/C4/C8,
MTP5 mean acceptance length was 4.96/4.81/5.32/5.41, and there were no request
failures or preemptions. That remains the historical production result. A
later matched six-prompt field-review gate on the same retained rental selected
MTP3 for the candidate profile: draft acceptance was 84.10% versus 39.84%,
mean acceptance length was 3.523 versus 2.992, and mean/p95 TPOT was
16.962/23.647 ms versus 25.104/48.709 ms for MTP5. MTP5 produced four fast
samples but two repeatable ~48 ms outliers, so its median alone was misleading.
The corrected r14 field-repair successor has now completed that gate and is
the authoritative field-repair attribution result for the 131K text-only,
offload-disabled profile: **251,392 logical KV tokens** (**1.87 GiB/rank**),
13/13 API features on both initial and warm passes, 5/5 needles at 32K,
20/20 needles across cold 65K/126K probes, and two bounded C1/C2/C4/C8
matrices with no request failure or preemption. Fully warm prefill was
2,011.072 tok/s at 3K and 1,857.989 tok/s at 32K; warm aggregate decode was
29.119/36.191/40.063/62.338 tok/s at C1/C2/C4/C8. Final scorecards were
**96/100 GSM8K** and **44/50 GPQA**; every one of GPQA's 44 normal stops was
correct and its only six misses exhausted the 32K reasoning ceiling. TERM
released the wrapper within 2 seconds and all workers/GPU allocations within
19 seconds. Release appliance:
`ghcr.io/malaiwah/glm52-exl3-vast:fa52eda06ab516cd7e1a6628d915d2fd2478478f@sha256:1152b6e23604cd158017964c5ef14d6290779f7d1e67ced2ba4a39c8ec83a5c7`;
immutable evidence
[`5c76a253`](https://github.com/malaiwah/glm52-exl3-vast/tree/5c76a2536e7fc9a5f1cb6bf182531889f5385e65/docs/field-review-results/2026-07-30-vast-46335896/artifacts).
This gate does not supersede the separate 512K, vision, or cache-offload
qualification above. The GPU run used the manifest-matched imported runtime;
CI built the pinned appliance afterward from unchanged runtime files, but that
published digest was not separately GPU-booted.
The exact patch order and independent
reproduction contract are in the
[counter-validation guide](field-review-results/2026-07-30-vast-46335896/COUNTER-VALIDATION.md).

The checkpoint's reported dynamic-NVFP4 KLD is 0.095971 versus 0.119525 for
3.0 bpw. The appliance independently measured **0.0927076684** over the
standard 2,047 positions, versus its independent 3.0-bpw result of
0.1167701185. The reference bundle contains one window, so a displayed
standard deviation of zero is structural (`n=1`), not zero model variance.
The complete active-KV/offload matrix and exact release boundaries are in
[the 3.25-bpw qualification report](glm52-3.25-offload-qualification.md).

The original 3,072-token shape was not runtime-safe with LMCache: it booted,
then OOMed its first 128K request on a 36 MiB mixed-K output conversion. The
final profile keeps the full 2,048-block GPU pool while lowering the prefill
chunk to 2,048 and sending sparse-indexer folds above 64 MiB through the exact
streaming-carry path. With 125 GiB LMCache DRAM it passed two 128K requests,
then completed 520,001- and 524,012-token prefills. The latter reached 99.3%
GPU-KV use, leaving only the API/template margin below the 524,288 hard limit.
The r13 production run has already served 277,504 prompt tokens from external
KV, restored 548 chunks per rank, and completed 19 filesystem-L2 loads with no
dropped cache events. The bounded local-NVMe directory occupied 33 GiB of its
512 GiB ceiling after qualification.

At the matched 65,024-token eviction gate, native vLLM restored the prefix from
DRAM in 0.568 s and LMCache in 0.508 s, versus roughly 30 s recomputation.
Native retained about 0.83 GiB more idle VRAM/GPU, but prompt throughput was
within run variance. Adding a bounded 512 GiB local-NVMe LMCache tier did not
change idle VRAM, 128K PP, or MTP acceptance. After a complete engine/cache
restart, the identical 131,076-token prompt restored from NVMe in 1.254 s;
LMCache reported 6.51 GB/s NVMe-to-DRAM and 14.7–17.1 GB/s DRAM-to-GPU.
Filesystem L2 remains opt-in because derived KV may contain session material.

The hybrid's 2,048 chunk is intentional. On v20, a 3,072-token chunk with a 512 MiB
workspace passed three uncached 32K prefills and a C1/C2/C4/C8 sweep, but
immediately OOMed in the target NF3 MoE output allocation at a 520,192-token
prompt. A 1 GiB workspace was worse: it passed the first 32K gate and OOMed on
the next request. A configuration that only boots—or even passes one short
needle—is not a 512K profile.

## Feature status

The live hybrid suite passes authenticated model discovery, exact
tokenization, ordinary chat, thinking-content visibility, streaming with
usage, multi-turn with preserved reasoning, release-gating strict structured
JSON both with and without thinking,
one automatic tool call, and tool-result continuation. The former
`tool_choice=required` duplicate-call behavior came from XGrammar's GLM
structural-tag grammar allowing another tool tag, but no normal trailing text
or end-of-turn path. GG r9 retains r8's pinned XGrammar 0.2.5, whose
`tool_choice=required` grammar permits normal completion after one or more
calls while still requiring at least one. The probe remains optional until
this derived r9 image completes the live appliance gate; `tool_choice=auto`
remains release-required.

Thinking plus structured output also crosses an MTP-specific boundary. The
draft can have proposed several answer tokens before the reasoning-end marker
activates the grammar. Those pre-mask proposals are allowed to be rejected and
resampled; on GG r5's vLLM path, retained through r8, XGrammar logged each expected
rejection as
`Failed to advance FSM` at ERROR severity even when the request returned HTTP
200 with exact schema-valid JSON. The entrypoint applies an idempotent
compatibility patch that checks these post-marker probes against the packed
grammar bitmask row vLLM already filled, without probing or mutating the
matcher. Valid probes still advance the temporary FSM state, and invalid
*committed* tokens retain vLLM's original hard-error path. The automatic
serving verifier and feature suite now make strict JSON with thinking a release
gate rather than inferring correctness from HTTP 200. This complements
[vLLM #44993](https://github.com/vllm-project/vllm/pull/44993), whose reasoning
boundary fix is already present in GG r5 through r9.

The patched path was live-qualified on the full Qwen3.6-27B checkpoint with
MTP2: 4/4 concurrent strict-schema requests passed in each of thinking-off,
thinking-on and omitted/default-thinking modes, followed by a clean full
feature-suite pass. There were no `Failed to advance FSM` messages, HTTP 500s
or engine failures. XGrammar can still print a native post-EOS warning under
speculation; exact output, health, and the committed-token failure path remain
the release criteria.

[`preserve thinking`](https://docs.z.ai/guides/capabilities/thinking-mode)
means forwarding the assistant's complete, unmodified prior
`reasoning_content` in the next request. [Interleaved
thinking](https://docs.vllm.ai/en/latest/features/interleaved_thinking/) is the
model reasoning again between tool calls and tool results. They are related
history semantics, not synonyms; interleaved tool use needs its intervening
thinking blocks preserved, while general multi-turn preservation remains an
explicit landing-page option and defaults off.

## Vision qualification (v20)

The final v20 turnkey image was qualified on the same 4x RTX PRO 6000
AIBeast stack as the controls above:

| what | result |
|---|---|
| 5120x2880 Retina-style dashboard | 17/18 exact requested details in 11.40s |
| multimodal prompt accounting | 2,131 prompt tokens, including 2,074 multimodal tokens |
| follow-up reusing image history | exact `COBALT-917 / Mira Chen / 09:53` in 0.71s; 2,048 cached tokens |
| text-only regression in same process | exact in 0.19s |
| model-memory delta | ~1.99 GiB/GPU (79.65 vs 77.66 GiB) |
| DCP4, GMU 0.975 | 564,736 KV tokens; short suite stable |
| DCP4, GMU 0.98 | 610,560 KV tokens, but first request OOMed with 37.12 MiB free |
| **32K retrieval gate** | **0/3 with degenerate output; MTP MAL collapsed to 1.25–1.50** |

This verdict is deliberately scoped to the current EXL3/TR3 composition, not
to Baseten's tower. The published
[MadeBy561 vision merge](https://huggingface.co/chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid)
reports text/vision MTP parity, while an independent
[NVFP4+AQLM merge](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid)
passed mixed image-plus-needle retrieval at 130K. Conversely, another
[EXL3/TR3 vision merge](https://huggingface.co/0xSero/GLM-5.2-TR3-Vision)
reports MTP acceptance collapsing on long synthetic prompts.

The known MadeBy561 merge was also tested locally with MTP disabled. It passed
3/3 short controls and 3/3 32K text needles, yet extracted only 1/18 requested
values from the same 5K dashboard. Applying Jarrel's config-delegation and
load-only name-mapping pattern raised that to only 2/18. This disproves an
EXL3/TR3-only explanation and isolates MTP from the failure, but does not
contradict successful simpler upstream image tests.

That wrapper correction had no material speed effect: periodic 32K PP moved
from 2,457.4 to 2,471.2 tok/s (+0.6%), while steady C1 generation moved from
about 49.6–49.7 to 49.2 tok/s with MTP off. Both are ordinary run noise. The
appliance therefore keeps vision opt-in and unqualified as a general-purpose
profile. A passing result on another quant or a simple image does not waive
the detailed-image and long-context gates.
