# GLM-5.2 GG v20-r14 qualification plan

Prepared 2026-07-30 for the next AIBeast maintenance window. This is a
qualification plan, not a production-promotion claim. Until every promotion
gate below passes, the control remains the running r13 3.25-bpw service.

## Immutable release boundary

| item | reviewed value |
|---|---|
| image | `voipmonitor/vllm:gilded-gnosis-v20-vllm749050e-si8110e3e-fi801d57a-cu132-20260730-r14` |
| registry manifest | `sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2` |
| pulled image ID | `sha256:bdf8fe02d0d44f1d4704149363e86aa2265d42ac739b3200c4f58788586614c0` |
| built | `2026-07-30T21:12:56Z`; compressed manifest size `25,184,332,296` bytes |
| vLLM | base `f978d009fab9`; composed tree `749050edab1b`; PRs `#145@99bf3f1`, `#175@2c14a58`, `#190@46c36c0`, `#198@703af3f` |
| SparkInfer | base `9b852b281250`; composed tree `8110e3ea4177`; PRs `#92@06032ce`, `#104@241718d` |
| LMCache | base `9cebd405d0ca`; composed tree `a5aa59cc8edc`; `0.5.2+glm52dcp.4`, unchanged from the qualified durability stack |
| EXL3 extension | `brandonmmusic-max/exllamav3` `a1-retile-sm120@704aefd` |
| runtime | PyTorch `2.12.0+cu132`, CUDA `13.2.1`, FlashInfer `801d57a`, XGrammar `0.2.5` |
| checkpoint | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |

The new checkpoint revision changes only `README.md`. Its 97-file inventory,
`339,171,568,527`-byte Hub storage total, `config.json`,
`model.safetensors.index.json`, and all weight objects are identical to the
previously qualified `61d2b6b757f6a4ac7098a78d861f2033497532dc` revision.
AIBeast can therefore retain the immutable existing snapshot and marker; no
339 GB weight copy or mutation is required.

The upstream model page currently links a non-existent full build SHA
`2464cc0dbe6f...`. The public release archive actually resolves to
`local-inference-lab/blackwell-llm-docker@2464cc03d0298493bf345bbc797f77c4455efda8`
(`build(v20): publish mixed EXL3 r14 release`). Use the latter when reproducing
or auditing the archived r14 integration patches.

## What changed, and what remains unproven

r14 replaces the appliance's compatibility implementation, which executed the
K3 and K4 expert tiers serially, with SparkInfer #104's one-grid path. It packs
global routes once, rotates activations once, keeps the checkpoint's native
packed weights, and executes both tiers in one cooperative kernel. For this
checkpoint every routed layer must report exactly
`tiers=((3, 192), (4, 64))`; the GLM tile is `(128, 128, 32, 512)`.
No reconstruction, online quantization, or expert reorder is involved.

An exact composed-tree comparison found that r14 changes only vLLM
`exl3.py`; `envs.py` is byte-identical to r13. SparkInfer adds the native mixed
module. There are no new environment variables and no default changes, so the
queued runtime-route A/Bs retain their r13 meaning.

Published evidence:

| published case | result |
|---|---:|
| final-image MTP0 C1, zero context | `48.3 tok/s` |
| final-image MTP0 C1, 65,536 context | `46.0 tok/s` |
| same PR heads, MTP3 C1 | `105.23 tok/s` |
| same PR heads, MTP3 C4 aggregate | `260.6 tok/s` |
| same PR heads, MTP3 C8 aggregate | `390.9 tok/s` |
| mixed kernel, decode | `82.38 -> 54.11 us/layer` (`-34.3%`) |
| mixed kernel, prefill | `6.764 -> 6.528 ms/layer` (`-3.5%`) |

The final-image MTP0 gate used GMU `0.95`, graph cap 6 and exposed 747,776 KV
tokens. The MTP3 throughput came from the same exact PR heads before final
image assembly, not a pulled-image 512K/LMCache run. Upstream's longest r14
gate was 65,536 input tokens plus 256 output tokens. It did not test MTP5,
the appliance's 2,048-token safety shape, 512K retrieval, KLD, LMCache DRAM,
LMCache NVMe, or our four non-default runtime routes. Those are the reasons
for this maintenance campaign.

## Rental status

Inventory captured on 2026-07-30:

- JarvisLabs IN1 had four RTX-PRO6000 cards only in its managed-container
  pool. That pool cannot launch the custom turnkey OCI image. Its full-VM pool
  had three free cards, one short of TP4, so no Jarvis rental was started.
- RunPod Secure advertised low stock but could not allocate four colocated
  Server or Workstation cards. RunPod Community successfully created Pod
  `4r3db68ea3nwzx`: four RTX PRO 6000 Blackwell Server Edition GPUs, 503 GB
  host RAM, 56 vCPUs, a 1 TB `/workspace` volume, and a `$6.76/hour` GPU rate.
  The host actually supplied driver `580.82.07`; the candidate correctly
  refused it before downloading weights because this is the same unsupported
  driver line with prior NCCL-failure evidence. The Pod is stopped, not
  terminated, so its workspace remains available without GPU billing.
- Vast subsequently exposed a better `$6.57/hour` California offer with driver
  `610.43.02`, CUDA UMD `13.3`, 600 W/card, PCIe 5, about 9.1 GB/s local disk,
  940/930 Mb/s network and 600 GB allocated storage. Instance `46335896` is the
  active `args`/ENTRYPOINT-mode qualification rental. A first CLI attempt,
  `46335044`, demonstrated the documented SSH-mode trap and is stopped rather
  than destroyed for follow-up patch work.
- The r610 `nvidia-smi` header spells the field `CUDA UMD Version`, rather than
  the historical `CUDA Version`. The compatibility override was needed only
  because the candidate parser did not recognize that new spelling; direct
  inspection confirmed PyTorch `2.12.0+cu132` on a CUDA UMD 13.3 driver. The
  parser fix accepts both header formats and continues to fail closed when
  neither one is present.
- All four Vast GPUs report same-NUMA `NODE` topology and a 600 W ceiling, but
  CUDA peer access is false for every pair. The appliance therefore disabled
  B12X PCIe DMA and DCP A2A and selected its NCCL/shared-memory fallback before
  model download. Treat this rental as a compatibility/quality gate, not an
  AIBeast performance proxy.

Rental topology is a qualification result, not an AIBeast performance proxy.
Record driver, CUDA report, P2P matrix, NUMA, PCIe width, power ceiling, host
RAM and disk bandwidth before interpreting its PP/TG numbers.

### Vast rental qualification evidence

Instance `46335896` completed the cost-bounded compatibility and quality row
on 2026-07-31. It used the r14 candidate at appliance commit `6ad56a8`; that
candidate pins the immutable upstream r14 payload above. The later wrapper
commit `cb62061` changes only NVIDIA banner parsing, passed CI, and published
as `ghcr.io/malaiwah/glm52-exl3-vast:cb62061558913d4c46db6fcbae25c2a6426a7562`
(`sha256:af2edc5004b0b453845c4614ed03e6d79048f2c097dfe43ff4a1a881dbd426a2`).
The active rental was deliberately not recycled onto it because `/workspace`
is part of the container's 600 GB overlay on this host; recycling could discard
the 315.78 GiB checkpoint. The old candidate was allowed only after direct
inspection established driver `610.43.02`, CUDA UMD `13.3`, PyTorch
`2.12.0+cu132`, and NCCL `2.29.7`.

The host disappeared from Vast briefly at 40% of the checkpoint download.
Instance intent remained `running`; it recovered without intervention and the
authenticated Hugging Face download resumed at the preserved byte count. The
checkpoint then loaded all 81 target shards and the native MTP shard. Every
routed layer reported `tiers=((3, 192), (4, 64))`; no compatibility graft,
eager-parity path, or mixed-K reconstruction ran.

The release-request shape at GMU `0.95` failed safely at KV admission:
3.17 GiB/GPU was available, 3.90 GiB was required for 524,288, and vLLM
estimated a 425,472-token ceiling. Applying only GMU `0.96` through the
self-service editor restarted vLLM without touching the weights. Two
successive cached boots produced:

| rental startup measurement | first boot | repeat boot |
|---|---:|---:|
| available KV memory/GPU | 4.12 GiB | 4.12 GiB |
| logical KV tokens | 553,216 | 552,960 |
| full-524,288 concurrency | 1.06x | 1.05x |
| engine initialization | 67.86 s | 53.55 s |
| compilation inside initialization | 4.48 s | 0.63 s |
| idle free VRAM after serving | 294–514 MiB | 514–554 MiB |

The second boot also completed a first 194-token generation without the
historical post-start 36 MiB allocation OOM. GMU `0.96` is therefore a
repeatable rental setting for this exact MTP3/3,072/auto-KV shape, but its
small physical headroom is not evidence to replace AIBeast's pinned-block
production geometry.

Measured serving results at temperature `1.0`, with no request errors,
preemptions, or OOM:

| Vast Community, no CUDA peer access | result |
|---|---:|
| 8K prefill | 973.5 tok/s |
| 65K prefill | 1,670.9 tok/s |
| 1K-context C1 | 36.2 aggregate tok/s; 18.67 ms mean TPOT; MAL 3.11 |
| 1K-context C4 | 59.2 aggregate tok/s; MAL 3.48 |
| 1K-context C8 | 171.7 aggregate tok/s; MAL 3.11 |
| near-zero-context C1 | 38.8 aggregate tok/s; 24.57 ms mean TPOT; MAL 2.28 |
| near-zero-context C4 | 75.4 aggregate tok/s; MAL 2.40 |
| near-zero-context C8 | 247.8 aggregate tok/s; MAL 2.82 |
| exact near-maximum retrieval | 5/5 with a 523,458-token haystack, 364.55 s |

The five needles were at 1%, 25%, 50%, 75%, and 99%; the answer matched every
code and the degeneration detector remained empty. A subsequent arithmetic
request returned exactly `391`. The complete API suite also passed auth,
tokenization, ordinary/thinking chat, streamed usage, preserved-thinking
multi-turn, structured JSON with thinking both off and on, one tool call, and
the tool-result round trip. Vision was not requested because this checkpoint
is the text-only 3.25-bpw profile.

During the 523K prefill, cards averaged `367.7 / 334.7 / 339.8 / 350.0 W`
while at least 90% busy and peaked at `442.7 / 437.7 / 406.6 / 433.7 W`.
Their power ceiling was 600 W, versus AIBeast's deliberate 280 W/card cap.
The rental is not power-limited; its missing CUDA peer access and fallback to
NCCL/shared memory are the material topology differences. The alternating
1,228.8/1,536.0 tok/s long-prefill log rows correspond to four versus five
3,072-token chunks landing in a fixed ten-second metrics bucket. They are a
reporting-quantization artifact, not alternating slow hardware cycles.

Artifacts retained on the rental:

- `/workspace/benchmarks/r14-vast-feature.json`
- `/workspace/benchmarks/r14-vast-serving.json`
- `/workspace/benchmarks/r14-vast-zero-context.json`
- `/workspace/benchmarks/r14-vast-needles-522360.json`
- `/workspace/benchmarks/r14-vast-feature-power.csv`
- `/workspace/benchmarks/r14-vast-522k-power.csv`

The instance remains running for follow-up patches at `$6.57/hour`, as
requested. The first SSH-mode Vast instance and the r580 RunPod Community pod
remain stopped rather than destroyed.

## Turnkey compatibility preparation

The r14 candidate must:

1. pin the immutable r14 manifest, never the moving tag;
2. recognize the complete native mixed-K API and leave upstream `exl3.py`
   untouched; retain the two-grid v5 patch only as an r11-r13 compatibility
   fallback and fail closed on a partial native API;
3. pin the model-card revision above while accepting AIBeast's read-only,
   byte-identical older snapshot;
4. move persistent vLLM/Triton/Inductor/extension caches to an
   `exl3native1` namespace below r14's source fingerprint. Never reuse r13
   mixed-adapter objects;
5. keep the currently qualified MTP5/batch-2,048 profile as the candidate
   default until the A/B matrix selects a replacement.

The candidate build is acceptable only after the local suite, the no-op test
against the exact reconstructed r14 `exl3.py`, the container build, and image
label/digest inspection pass.

## AIBeast preflight and rollback

The control is `glm52-turnkey-r13-325-prod` on
`localhost/glm52-turnkey:r13-prod-v2`, serving port 8000 with both
`GLM-5.2` and `local-primary`. It uses the read-only Hub snapshot, exactly
2,048 batched tokens, 125 GiB LMCache DRAM, and a bounded 512 GiB filesystem
tier at `/mnt/fast/lmcache/glm52-3.25-r13-512g`.

Before stopping it, archive under
`/mnt/fast/build/r14-production-qualification-20260730/`:

- `podman inspect`, complete environment with secrets redacted, image digest,
  mounts, health and the last 10,000 log lines;
- `nvidia-smi -q`, `nvidia-smi topo -m`, P2P/NUMA calibration output, current
  power limits, free DRAM and `/mnt/fast` capacity;
- the current compile-cache namespace and LMCache statistics;
- a clean control C1/C4/C8 decode row and 32K/128K PP row if live traffic
  permits.

Do not remove the control container, volumes, or cache. Rollback is simply:
stop the candidate, start `glm52-turnkey-r13-325-prod`, wait for `/health`,
confirm both model aliases, and run one ordinary plus one structured-output
request. Another user's `priceless_banach` container is out of scope and must
not be stopped or altered.

Run stale-lock recovery only after proving no process owns the extension lock.
Do not blindly delete an active lock. Preserve every failed candidate's logs
before a restart.

## Ordered, cost-bounded matrix

Every row gets an immutable config dump, startup memory lines, active KV,
idle/free VRAM, PP, TG, MAL, power and error scan. Reuse only the new r14
compile namespace after the first cold run.

### 1. Reproduce the release request

Start the pulled r14 turnkey with TP4/DCP4, MTP3, seqs 8, graph/trellis 32,
batch 3,072, GMU 0.95, auto KV and no external offload. Confirm:

- `tiers=((3, 192), (4, 64))` and the native compatibility-patch no-op;
- target and draft compile/capture without an eager-parity message;
- no first-request 36 MiB output-conversion OOM;
- 8/8 short correctness plus 65,536+256;
- C1/C4/C8 and MAL against `105.23 / 260.6 / 390.9`.

This is the upstream reproduction row, not a production candidate.

### 2. Select MTP depth and prefill batch

Use the complete production posture: exact 2,048 GPU blocks / 524,288 logical
tokens, dynamic-token NVFP4 MLA KV, fold budget 64 MiB, LMCache DRAM 125 GiB
and NVMe limit 512 GiB.

1. r14 MTP5 / batch 2,048 / graph 48: exact r13-production-equivalent control.
2. r14 MTP3 / batch 2,048 / graph 32: isolate draft depth.
3. Winner / batch 3,072: determine whether native one-grid execution removed
   the former 36 MiB first-request OOM and whether 32K/128K PP improves without
   reducing C1/C8 decode.

Use the same prompts and at least three decode samples. Prefer MTP3 only if its
real-workload MAL and aggregate results beat MTP5 after variance; otherwise the
user-approved MTP5/2,048 candidate remains selected. A 3,072-token batch is
promotable only after two cold 128K requests, C8 and near-maximum context all
complete without OOM. Priority remains PP, then TG, then surplus KV; exact
512K capacity and C1 at or above 100 tok/s are hard gates.

### 3. One-variable runtime-route A/Bs

Hold the winning geometry fixed and restart for one change at a time:

| order | variable | control | candidate | primary measurement |
|---:|---|---:|---:|---|
| 1 | `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE` | `0` | `auto` | C1/C4/C8, MAL, verifier row count and transient VRAM |
| 2 | `VLLM_DCP_TOPK_OWNER_MERGE` | `1` | `0` | 32K/128K PP, exact retrieval and top-k correctness |
| 3 | `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `1` | `0` | C1/C4/C8 plus 128K PP |
| 3a | `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | `16` | `256` | only if streaming wins step 3 |
| 4 | `VLLM_DCP_A2A_MAX_TOKENS` | `16` | `0` | small decode and 32K/128K prefill; optionally bracket 8/32 |

Do not test the inert unprefixed `DCP_A2A_MAX_TOKENS`, and do not add
`B12X_MHC_MAX_TOKENS`; the composed source has no GLM runtime consumer for
either spelling. With MTP5 at C8, `auto` plans exactly 48 verifier rows, so
graph and Trellis ceilings must remain at least 48. Never force
`SPEC_EXTEND_AS_DECODE=1`, which reserves the full operator limit rather than
only genuine verifier rows.

### 4. Offload, quality and feature gates

For the selected configuration:

- compare GPU-only, LMCache DRAM, then LMCache DRAM+NVMe startup/free-VRAM;
- restore one 65K prefix from DRAM, restart and restore one 128K prefix from
  NVMe, record transfer bandwidth, hit rates, dropped events and L2 size;
- run the standard 2,047-position KLD capture. The r13 result was
  `0.0927076684`; require no material regression and keep the candidate below
  `0.10`. The reference bundle has one window, so standard deviation zero is
  structural;
- run the exact 522,360-token five-depth needle document and the 524,012
  boundary request. Require 5/5 exact retrieval, no degeneration, no OOM and
  normal queued service afterward;
- run ordinary/thinking chat, streaming usage, preserved-thinking multi-turn,
  strict structured JSON with and without thinking, and the complete
  tool-call/result loop;
- review the complete vLLM, SparkInfer, LMCache, CUDA/NCCL and appliance logs,
  not just request status.

### 5. Log-derived follow-ups

The Vast full-log review added four items to tonight's interpretation:

1. vLLM warns that MTP3, eight sequences and a 3,072-token scheduler budget
   may under-provision draft slots. This is another reason to retain the
   planned matched 2,048/3,072 prefill A/B; do not assume that the larger
   number is automatically faster.
2. First use compiled several CuTeDSL/Triton shapes that startup warmup did not
   cover, including unified prefill, fused W4A16 MoE, DCP correction and
   route-packing kernels. Compare first-cold and warmed 32K/128K latency. Only
   extend appliance warmup for shapes whose latency benefit justifies added
   startup time and memory pressure.
3. Keep CUDA-graph memory profiling enabled. r14 reports that GMU `0.96`
   behaves like `0.9549` under the older unprofiled accounting; disabling the
   profiler would hide graph memory from admission rather than create safe
   headroom.
4. The `top_p=0.95` startup notice comes from the checkpoint's
   `generation_config.json`, not an appliance-injected request. Do not add
   `--generation-config vllm` merely to silence it; clients and the benchmark
   explicitly supply their desired sampling parameters.

The repeated “unknown vLLM environment variable” messages remain cosmetic:
the variables are consumed by the composed GLM/SparkInfer integration outside
upstream `envs.py`. The no-peer `SymmMemCommunicator` warnings are expected on
this rental and correctly coincide with the appliance disabling B12X PCIe DMA
and DCP A2A. The one ACME transient check recovered through the bounded
background retry and produced the working public certificate before serving.

## Promotion decision

Promote r14 only if it serves both aliases, retains exact 512K active context,
clears C1 100 tok/s, improves or preserves PP/TG against the r13 control,
passes every quality/feature gate, survives C8 and agent traffic without OOM,
and shows no unexplained warning or cache corruption. Then update the profile,
qualification report and README with measured—not upstream-estimated—values.

If any hard gate fails, restore the untouched r13 container and retain the r14
artifacts for diagnosis. A successful rental remains stopped rather than
terminated so its 1 TB volume and downloaded weights can be reused for the
next patch round.
