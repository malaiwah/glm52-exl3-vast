AIBeast GPU A/B is complete on the exact r33-derived turnkey stack.

Hardware/profile held constant:

- 4x RTX PRO 6000 Blackwell, TP4/DCP4/MTP3
- `willfalco/GLM-5.2-EXL3-TR3-3.42bpw@a350292cb2038f2c31732569a711a89e5d72fd46`
- online EXL3 K6, dynamic NVFP4 DS-MLA KV with FP8 RoPE
- C12, batch 3072, graph/trellis 48, max model length 520192
- identical persistent model/compile/JIT caches and launch environment
- 375 W cap during post-start performance/correctness gates

Only the vLLM runtime patch from `df004fb4a6bfb39d2f67c6eb96313f55cea8c65b` changed.

Warm startup A/B:

| Metric | pristine r33 | PR #277 | delta |
|---|---:|---:|---:|
| container start -> `Starting vLLM server` | 798.39 s | 683.55 s | **-114.84 s (-14.4%)** |
| vLLM `Model loading took` (TP3 matched) | 671.98 s | 514.71 s | **-157.26 s (-23.4%)** |
| model VRAM reported | 81.54 GiB/rank | 81.55 GiB/rank | parity |
| CUDA graph capture | 12 s / 0.75 GiB | 12 s / 0.75 GiB | parity |

The checkpoint-shard phase becomes longer because the tier placement work is done while loading, but eliminating the later duplicate grouping/stacking work produces a material end-to-end win. This is not merely a shifted log phase.

Patched-image serving gates:

- feature suite: chat, visible thinking, streaming, preserved-thinking multi-turn, structured JSON, tool calling, required tool choice, and tool-result round trip all passed
- short correctness: arithmetic/factual/instruction checks all passed
- uncached prefill: 8K **3,090 tok/s**, 64K **3,006 tok/s**, zero failures/preemptions
- C8 decode: **265.0 aggregate tok/s**, MAL **3.466**, zero failures/preemptions
- C1 clean attempt: **90.4 tok/s**; coarse metrics show concurrent agent traffic, so treat as traffic-contaminated rather than a matched microbenchmark
- fresh-seed maximum-context needle: **517,177 exact tokens**, 5/5 at 1/25/50/75/99%, no degeneration, 243.16 s
- physical free VRAM during the 517K run: about **877 MiB/GPU**; no OOM, traceback, worker death, or preemption

Evidence is retained under `/mnt/fast/build/20260809-glm52-prefill/startup-ab/{pristine,patched}/` on the test host. Based on this A/B, the patch is GPU-qualified for this mixed-Trellis/native-MTP production shape.
