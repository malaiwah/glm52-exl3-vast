# Qwen3.6-27B NVFP4 qualification

Extracted from the README: the measured single-GPU envelope behind the
`qwen36-27b-nvfp4` profile.

The Qwen profile serves
[`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
with `--quantization modelopt`, the `qwen3` reasoning parser and
`qwen3_coder` tool parser. It defaults to its native vision encoder, one GPU,
192K context, FP8 KV selected from the checkpoint metadata, no DRAM KV
offload, and compiled MTP-off decoding. The full profile was qualified on one
575 W RTX 5090 32 GB: thinking/non-thinking, streaming, preserved multi-turn
reasoning, strict structured outputs, tools/tool-result continuation, a
192,290-token five-depth retrieval, and two 5120x2880 detailed-vision gates all
passed.

The upstream architecture supports 262,144 tokens, but 256K is not a safe
vision-enabled 32 GB appliance target. Text-only attempts approached it; the
vision encoder needs transient working memory. At 200K and 208K the detailed
image test survived with only 177 MiB and 29 MiB free respectively, while
nearby boots OOMed. The selected `GPU_MEMORY_UTILIZATION=0.90`,
`MAX_MODEL_LEN=196608`, `MAX_NUM_BATCHED_TOKENS=4096`, and 8,388,608-pixel
image cap retained about 511 MiB after the post-192K vision repetition. An
8,192-token scheduler batch reduced the profiled KV ceiling to about 168K and
could not start, so it is not a performance upgrade for this envelope.

| one RTX 5090, GG v20-r9 | measured result |
|---|---:|
| uncached PP @8K / 64K / 180K | 3,704 / 3,716 / 2,513 tok/s |
| aggregate TG C1 / C2 / C4 | 68.0 / 120.9 / 221.6 tok/s |
| KV pool / maximum request | 205,544 / 196,608 tokens |
| near-maximum retrieval | 5/5 at 192,290 tokens; no degeneration |
| 5K dashboard | 17–18/18 details, image follow-up and text regression passed |
| warm compatible-cache start | about 43 seconds to API readiness |

On GG v20-r9, Qwen MTP2 cannot use the compiled
FlashInfer decode wrapper: the
wrapper freezes `q_len_per_req=1`, while the draft step needs `3`, and the first
request kills the engine. The appliance therefore adds `--enforce-eager`
automatically when Qwen MTP is enabled. That compatibility path passed the live
feature gate but gives up torch compilation and CUDA graphs. It measured only
46/81/101 tok/s at C1/C2/C4 versus 68/121/222 without MTP, despite healthy
74–79% draft acceptance and mean acceptance length around 2.5. It also reduces
usable context. The fast qualified default therefore remains `MTP_TOKENS=0`.
The real OMP workload made the memory cost more explicit: with GMU 0.90, MTP2
failed 192K KV admission, and 172K, 155K and even 64K auto-KV trials OOMed in
the first substantial eager request because vLLM filled the remaining GMU
budget with KV. A 64K diagnostic passed after either lowering GMU to 0.85 or
pinning the equivalent per-GPU pool with
`KV_CACHE_MEMORY_BYTES=4981753856`. Across 38 OMP reporting intervals the
fixed-headroom shape averaged 2,656 prompt tok/s, reached 3,520 prompt tok/s,
and produced MAL 2.606 with 80.3% mean draft acceptance. Aggregate generation
averaged 22.8 tok/s and peaked at 95.3 while serving up to four requests—still
materially worse than compiled MTP-off. The fixed-pool knob is therefore an
advanced diagnostic/workspace control, not part of the Qwen production
profile.

N-gram speculation hit the same frozen-query-shape class in compiled mode and
failed its correctness gate in eager mode; EAGLE and DSpark require compatible
external draft checkpoints that this checkpoint does not publish.

The [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
documents the architecture, native context, text-only switch, Qwen parsers,
and its MTP module. The pinned vLLM runtime uses the current speculative method
name `mtp` (the older `qwen3_next_mtp` alias is deprecated). The
[NVIDIA NVFP4 checkpoint card](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
specifies the ModelOpt loader and 262K serving command. This image's
[pinned GG source and EXL3 integration](https://github.com/local-inference-lab/vllm/pull/190)
contains the required Qwen3.5 architecture, parser, speculative method, and
mixed-precision ModelOpt implementation. The single-GPU 192K vision profile is
the measured turnkey envelope; 262K remains an upstream model capability, not
a claim about safe operation on a 32 GB card.
