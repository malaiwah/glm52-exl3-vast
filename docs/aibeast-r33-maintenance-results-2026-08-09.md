# AIBeast r33 maintenance qualification — 2026-08-09

Status: qualified and promoted to AIBeast production. This record intentionally
distinguishes clean cells from cells contaminated by live agent traffic.

## Candidate and invariant serving shape

- base: `voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33`
- model: `willfalco/GLM-5.2-EXL3-TR3-3.42bpw@a350292cb2038f2c31732569a711a89e5d72fd46`
- TP4 / DCP4 / MTP3; online EXL3 K6; dynamic NVFP4 DS-MLA KV with FP8 RoPE
- 520,192 request limit; C12; batch 3,072; graph/Trellis ceiling 48
- fixed KV bytes: 4,518,907,904 per rank
- LMCache: 125 GiB DRAM plus bounded 512 GiB NVMe
- GPU order: `2,1,0,3`; memory clock 16,365 MHz
- prompt-logprobs workspace chunk: 128

## Supersession audit

- vLLM #250 was removed because vLLM #228 contains the same stable patch.
- the separate B12X #126 overlay was removed because r33 contains its complete
  four-commit head, including the later scalar-alignment, native-MTP and route
  ID dtype fixes.
- B12X #130 and vLLM #258, #270 and #271 remain required overlays.
- the vLLM #258 exact head was corrected to
  `63b77c8031e02d23a2c464627196cd5663f7116f` in the appliance manifest.

## Power and thermal A/B

At 400 W, the 516K retrieval gate passed and the hottest GPU briefly reached
92 C without an NVIDIA hardware- or software-thermal-throttle indication.
Short 90--92 C excursions are accepted for this host; sustained temperature at
that level or an actual throttle flag is not. The selected 375 W soak cap
retained 98.6% of the 400 W 262K-prefill result and is currently the better
performance-per-watt point, rather than a safety-mandated ceiling.

| cap | 128K PP | 262K PP | thermal result |
|---:|---:|---:|---|
| 280 W | 2,038 tok/s | 1,788 tok/s | cool control |
| 400 W | — | 2,184 tok/s | GPU 3 briefly reached 92 C; no throttle |
| 375 W | 2,141 tok/s | 2,154 tok/s | selected soak cap; no thermal throttle |

The 375 W mixed 128K+C4 test completed 4/4 requests with aggregate prompt
throughput 2,246.8 tok/s, no preemption, and MTP mean acceptance length 3.3546.

## Memory-clock A/B at 375 W

| memory clock | 64K PP | 128K PP | delta at 128K |
|---:|---:|---:|---:|
| 16,365 MHz | 2,812 tok/s | 2,442 tok/s | control |
| 15,165 MHz | 2,753 tok/s | 2,341 tok/s | -4.2% |
| 13,965 MHz | 2,752 tok/s | 2,305 tok/s | -5.6% |

The production preference remains 16,365 MHz.

## vLLM #277 startup A/B

The direct mixed-Trellis tier-slab loader was tested against the same warm
checkpoint and compile cache.

| measurement | pristine r33 | r33 + #277 | delta |
|---|---:|---:|---:|
| entrypoint to API ready | 798.39 s | 683.55 s | -114.84 s (-14.4%) |
| matched TP3 model load | 671.98 s | 514.71 s | -157.26 s (-23.4%) |
| final model VRAM | 81.54 GiB | 81.55 GiB | parity |
| CUDA graph capture | 12 s / 0.75 GiB | 12 s / 0.75 GiB | parity |

The patched image passed the non-vision feature suite, 8K/64K prefill, C8
decode, a fresh five-depth 517,177-token needle suite, and degeneration checks.

Upstream: [issue #276](https://github.com/local-inference-lab/vllm/issues/276)
and [draft PR #277](https://github.com/local-inference-lab/vllm/pull/277).

## Lossless PCIe DMA threshold A/B

Five cold repetitions per arm were run under live agent traffic. Each prefill
repetition used a nonce that diverged at token zero; an accidentally warm
LMCache repetition was detected, rejected and replaced. Decode cells with
obvious unrelated traffic are retained as stress evidence but not represented
as uncontaminated single-workload claims.

| cell | 6 MiB median | 24 MiB median | 24 MiB delta |
|---|---:|---:|---:|
| PP 8K | 2,721 | 2,678 | -1.6% |
| PP 32K | 2,677 | 2,641 | -1.4% |
| PP 64K | 2,502 | 2,590 | +3.5% |
| PP 128K | 2,304 | 2,398 | +4.1% |
| PP 262K | 2,217 | 2,175 | -1.9% |
| C4 aggregate TG | 193.3 | 194.2 | parity |

No arm produced an OOM, restart or preemption. The 24 MiB default is retained:
6 MiB did not provide a consistent speed gain, while it expands the traffic
range sent through the custom PCIe DMA route.

## MHC prefill geometry crossover

Lowering `B12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS` from its 4,096-token
default to 3,072 makes the selected scheduler chunk use B12X's hidden-4096
M192/K64 projection geometry. Three fresh-prefix repetitions completed with no
failure, OOM, restart, or preemption. The third run overlapped substantial live
agent traffic; medians are therefore based on the two faster independent runs
and the slower contaminated run rather than discarded samples.

| cell | 4,096 control median | 3,072 candidate median | delta |
|---|---:|---:|---:|
| PP 3K | 2,542 | 2,672 | +5.1% |
| PP 8K | 2,678 | 3,061 | +14.3% |
| PP 64K | 2,590 | 2,880 | +11.2% |
| PP 128K | 2,398 | 2,736 | +14.1% |
| PP 262K | 2,175 | 2,352 | +8.1% |

Decode samples were traffic- and acceptance-sensitive and are not used to
claim a TG change; this dispatch knob is confined to the MHC prefill path. The
3,072 crossover is promoted for the 3.42 bpw / batch-3,072 profile.

## Asynchronous scheduling qualification

The candidate exposes a strict `ASYNC_SCHEDULING=on|off` self-service switch.
The appliance default remains `off`; the AIBeast service is intentionally
being left `on` until the next maintenance window to collect a longer sample
under real agent traffic.

The async-on arm passed thinking and streamed usage, preserved-thinking
multi-turn chat, structured JSON, single and required tool calls, tool-result
round trips, cancellation followed by immediate reuse, and four concurrent
approximately 16.4K prompts requesting 20 prompt logprobs. Every prompt-
logprobs response returned exactly the expected number of rows.

A pressure cell submitted twelve 65,725-token prompts with 128 generated
tokens each. All 12 requests completed in 393.23 seconds: 788,700 prompt
tokens at 2,005.7 tok/s aggregate, 1,536 generated tokens, MTP mean acceptance
length 2.0303, zero request errors, zero scheduler preemptions, zero OOMs and
zero restarts. The scheduler safely deferred excess work as KV became
available. Physical free VRAM remained approximately 865--973 MiB.

Decode observations are encouraging but live-traffic-sensitive: two async-on
C12 samples produced 309.1 and 279.6 aggregate tok/s with mean acceptance
length 3.025 and 2.904. The matched async-off C12 sample was 270.4 tok/s. This
supports an extended soak, not yet a universal speedup claim.

## KLD and final maximum-context gates

The exact retained 3.42 bpw KLD protocol reproduced a mean KLD of
`0.08988819671335253` across 2,047 positions. It is byte-for-byte identical to
the previous baseline value. The reference bundle contains one window, so its
reported standard deviation of zero is structural rather than evidence of
zero model variance.

With async scheduling active, a new seed (`20260815`) produced an actual
517,178-token prompt and recovered all five planted values at 1%, 25%, 50%,
75%, and 99% in 247.84 seconds, with no degeneration. The request reached
97.5% logical KV occupancy while unrelated live traffic queued safely. It held
965 MiB of physical VRAM free per GPU at the prefill activation peak, produced
no OOM, preemption, worker failure, or restart, and drained back to normal.

Two identical temperature-zero, thinking-disabled async requests returned
exactly `2, 3, 5, 7, 11, 13, 17, 19`; normalized response SHA-256 hashes were
identical (`ed0331a7d0ff41379794dc55eef60da2af100fc3d7df6b6548456c43cafe693e`).

## Promoted production posture

AIBeast is serving port 8000 from the r33 + memory-stack + vLLM #277 image with
both `GLM-5.2` and `local-primary` aliases, exact 520,192 logical KV capacity,
C12/batch-3,072/MTP3, the promoted 3,072-token MHC crossover, 24 MiB lossless
DMA threshold, 375 W power cap, and 16,365 MHz memory clock. Async scheduling
is deliberately enabled for an extended real-traffic soak until the next
maintenance window; the general turnkey default remains off during that soak.
