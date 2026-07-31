# Cross-provider benchmarks and loader evidence

Extracted from the README's flagship model card. All numbers describe the
exact stacks named below; a driver, CUDA, base-image, or kernel refresh is a
requalification boundary.

## Performance: compare like with like

The provider comparison uses the balanced
TP4/DCP2/TR3-MTP5 flagship profile and `llm-inference-bench` v0.4.29 protocol.
All measured systems used four RTX PRO 6000 Blackwell 96 GB cards. Results are
aggregate output throughput; PP is a cold unique-prefix request, so prefix
cache hits do not inflate it. AIBeast is the final GG v20-r5 compute image;
GG r8 retained that exact compute stack. GG r9 changes the NVFP4/indexer
sources and is therefore a new qualification boundary. Vast and Runpod are
the immediately preceding v31 candidate; JarvisLabs is the July 30 GG r9
candidate with the appliance's no-P2P fallback. The later one-card Qwen
qualification also supports the exact
Runpod compatibility pair `590.48.01 / CUDA 13.2`; the pairwise admission
rule below intentionally does not admit `590.48.01 / CUDA 13.1`.

| environment | topology / power cap | PP 8K / 32K / 64K / 128K tok/s | TG C1 / C2 / C4 / C8 tok/s | exact long-context gate |
|---|---|---:|---:|---|
| **AIBeast (owned, GG r5)** | all `NODE`, 280 W/card | 2,853 / 2,749 / 2,658 / 2,504 | 106.7 / 145.6 / 207.9 / 284.5 | 510,535-token document, 5/5 depths |
| **Vast Community** | all `NODE`, 600 W/card | 3,046 / 2,939 / 2,875 / 2,700 | 78.5 / 140.8 / 210.1 / 330.8 | 517,176 tokens, 5/5 depths |
| **Runpod Secure** | two `NODE` pairs, cross-pair `SYS`, 600 W/card | 3,554 / 3,449 / 3,357 / 3,114 | 63.0 / 155.8 / 223.1 / 343.5 | 517,176 tokens, 5/5 depths |
| **JarvisLabs IN1 VM** | all `PHB`, CUDA P2P unavailable, 600 W/card | 2,417 / 2,444 / 2,354 / 2,228 | 79.6 / 120.8 / 180.2 / 272.3 | 517,177 tokens, 5/5 depths |
| **Runpod Community estimate** | host-dependent, commonly Vast-like | **2,500–3,500 / 2,400–3,400 / 2,300–3,300 / 2,100–3,100** | **55–95 / 125–165 / 185–230 / 280–350** | expected when the same 4x96 GB shape boots; run the gate |

The Runpod Community row is deliberately a planning range, not a benchmark:
its topology, host contention, registry route, storage and power policy vary
by offer. Secure Cloud is not automatically faster at low-concurrency decode;
the measured cross-socket topology made C1 slower than both all-`NODE` hosts.
Conversely, its network and storage made cold provisioning much faster.

GPU telemetry, not wall-outlet system power:

| phase | AIBeast GG r5 | Vast Community v31 | Runpod Secure v31 | JarvisLabs GG r9 |
|---|---:|---:|---:|---:|
| complete canonical run average | 1,056 W | 1,495 W | 1,457 W | 1,090 W |
| zero-context C1 | 1,084 W | 1,241 W | 1,012 W | 946 W |
| zero-context C8 | 1,110 W | 1,739 W | 1,743 W | 1,137 W |

AIBeast remains the efficiency reference: the rental power ceiling improves
prefill and high-concurrency aggregate throughput, but does not overcome
communication latency at C1. The measured drivers were **595.71.05 / CUDA
13.2** on AIBeast, **610.43.03 / CUDA 13.3 compatibility** on Vast,
**610.43.02 / CUDA 13.3 compatibility** on Runpod Secure, and
**595.58.03 / CUDA 13.2** on JarvisLabs. AIBeast's
`nvidia-smi` client reported **580.95.05** while the loaded driver reported
595.71.05.

These versions are part of the result. A driver, CUDA, base-image, or kernel
refresh is a requalification boundary: isolate incompatible compile caches,
repeat a cold 32K retrieval gate, confirm memory profiling and runtime
headroom, and rerun the compact performance matrix before comparing new
numbers with this table. AIBeast is scheduled for such a host refresh; until
that pass is recorded, these values describe the tested stack rather than the
future installation. The appliance now puts its persistent vLLM, Triton,
Torch-extension, and Inductor caches below the base image's immutable
`LOCAL_INFERENCE_CACHE_FINGERPRINT`: same-stack restarts remain warm, while an
r11-to-r13 change cannot accidentally execute stale compiled objects.

The current GG image is CUDA 13.2. The appliance therefore fails fast below
the qualified pair **NVIDIA driver 590.48.01 and reported CUDA 13.2**, before
it downloads model weights. Driver 595.45.04 remains the driver paired with
CUDA 13.2 GA in the
[official release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html),
but the lower pair is no longer speculative: a Runpod Secure RTX 5090 with
driver 590.48.01, a CUDA 13.2 report and `cuda-compat-13-2` present passed the
Qwen profile's complete feature suite, vision, long-context retrieval,
autonomous-appliance checks and a real cross-provider OMP workload. The pair
requirement remains intentional: a Runpod r580 host failed NCCL initialization
and Vast classified an earlier r590 offer as CUDA 13.1. Set
`ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1` only for another separately qualified
driver/CUDA combination.

## Safetensors, compiled-cache reuse, and the InstantTensor opt-in

Safetensors remains the flagship default. As a historical cold/warm control,
the exact immutable r11 image loaded
all 81 target shards in 91.36 seconds from a warm local store, completed model
load in 135.08 seconds, and exposed 542,208 logical KV tokens at GMU 0.957.
It passed the complete OpenAI feature suite and an exact 522,360-token
five-depth retrieval with 5/5 needles and no degeneration.

The same container image was then replaced and restarted against the same
persistent compile volume. The backbone loaded its AOT artifact in 0.55
seconds, the small speculative head compiled in 3.63 seconds, and output
remained correct. The warm run exposed 553,472 KV tokens; the variation is the
runtime memory profiler, not a different profile. This directly challenges
the historical cache-corruption and very-low-throughput failure modes:
same-stack reuse is enabled, but every compiled path is namespaced by the
immutable upstream source fingerprint plus the turnkey EXL3 patch ABI.

InstantTensor remains selectable. Earlier stacks often loaded it in
32.4–33.1 seconds versus 60.5–62.6 seconds for warm-page-cache safetensors,
with no systematic steady-state PP/TG change. It also stalled without reaching
GPU allocation in later cold qualification attempts and has repeatedly
changed the memory-admission boundary. It is therefore an experiment, not the
first-time-user default. Any loader change requires a cold start, a decode
check, and the near-maximum retrieval gate.

A seeded same-prompt matrix found no systematic steady-state change:

| loader | PP 8K / 32K | TG C1 / C2 / C4 / C8 | failures / preemptions |
|---|---:|---:|---:|
| safetensors | 2,794.8 / 2,680.2 | 170.9 / 227.7 / 318.7 / 409.1 | 0 / 0 |
| InstantTensor | 2,782.4 / 2,680.4 | 168.7 / 230.6 / 306.2 / 402.6 | 0 / 0 |

The mixed deltas range from +1.3% to -3.9%, consistent with run/output
variation rather than a loader-dependent kernel change. The older
safetensors 514,432-token sparse-indexer OOM remains useful historical
evidence that loader and runtime revisions alter the memory shape; the r11
522,360-token pass supersedes it for this exact image and profile.

Do not read a single periodic vLLM line as an end-to-end prefill benchmark.
The logger defaults to a 10-second interval and counts each scheduled chunk
when it completes. With a 2,048-token chunk, one completed chunk prints
`204.8 tok/s`, two print `409.6`, and a bucket with no completed chunk prints
`0`; `204.8, 0, 204.8, 0` is therefore ordinary boundary quantization. Use
exact prompt tokens divided by TTFT, with a unique prefix so prefix caching
cannot contaminate the result.
