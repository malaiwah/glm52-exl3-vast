# Appliance test results

Cost-controlled execution of [TEST_PLAN.md](TEST_PLAN.md) from 2026-07-26
through 2026-08-04. Provider credentials and generated appliance tokens were
kept in process-local environment variables and are not included here.

Per-release GLM-5.2 model qualification details (throughput tables, KLD
measurements, memory ladders) are in
[docs/glm52-qualification.md](docs/glm52-qualification.md),
[docs/glm52-3.25-offload-qualification.md](docs/glm52-3.25-offload-qualification.md),
and the per-release `docs/glm52-rXX-*.md` files. This file records provider
integration evidence, bug discoveries and fixes, cost tracking, and the
decision history that explains why the codebase is the way it is.

Current release: **GG v20-r26** (qualified 2026-08-04 on AIBeast).

## Contents

- [Current qualification status](#current-qualification-status)
- [Provider integration](#provider-integration)
- [Qwen3.6-27B NVFP4 qualification](#qwen36-27b-nvfp4-qualification-vast-rtx-5090)
- [GLM-5.2 qualification history](#glm-52-qualification-history)
- [GLM-5.2 design decisions](#glm-52-design-decisions)
- [Local and browser checks](#local-and-browser-checks)
- [Artifacts](#artifacts)
- [Coverage summary](#coverage-summary)
- [Next economical live pass](#next-economical-live-pass)

## Current qualification status

| profile | release | hardware | status | section |
|---|---|---|---|---|
| GLM-5.2 EXL3 TR3 3.0bpw (default) | GG v20-r26 | 4x RTX PRO 6000 (AIBeast) | Qualified | [r26 gate](#gg-v20-r26-tp4dcp4-policy-gate-aibeast-2026-08-04-current) |
| GLM-5.2 EXL3 TR3 3.36bpw | GG v20-r26 | 4x RTX PRO 6000 (AIBeast) | Qualified | [r26 gate](#gg-v20-r26-tp4dcp4-policy-gate-aibeast-2026-08-04-current) |
| GLM-5.2 EXL3 TR3 3.25bpw mixed-K | GG v20-r17 | 4x RTX PRO 6000 (AIBeast) | Qualified | [r17 gate](#gg-v20-r17-native-mixed-k-production-gate-2026-08-01) |
| Qwen3.6-27B NVFP4 vision | GG v20-r9 | 1x RTX 5090 (Vast) | Qualified | [Qwen section](#qwen36-27b-nvfp4-qualification-vast-rtx-5090) |
| JarvisLabs VM (NCCL fallback) | GG v20-r9 | 4x RTX PRO 6000 (IN1) | Qualified | [JarvisLabs](#jarvislabs-in1-flagship-qualification-2026-07-30) |
| Runpod 590.48.01 / CUDA 13.2 | GG v20-r9 | 1x RTX 5090 (Secure) | Qualified | [Runpod compat](#runpod-5904801--cuda-132-compatibility-2026-07-29) |
| GLM-5.2 vision (opt-in) | GG v20 | 4x RTX PRO 6000 (AIBeast) | Short-context only | [Vision section](#v20-vision-qualification-2026-07-27) |

## Provider integration

### Secure Vast composite retest (2026-07-29)

A clean `ghcr.io/malaiwah/glm52-exl3-vast:latest` boot at commit `87fd42d`
repeated the Qwen server + separate OMP client exercise on Vast. The server was
a verified RTX 5090 host with driver `595.71.05`, CUDA 13.2 compatibility and a
575 W power ceiling; the client was a low-cost RTX 4060 Ubuntu rental used only
as a first-user SSH/OMP machine.

- deSEC DNS and Let's Encrypt direct TLS came up automatically. The checkpoint
  download took 78 seconds; the complete engine, compile and verifier path took
  about seven minutes from container start.
- The built-in gate passed health, short answers, structured output with
  reasoning, and 3/3 retrieval at 32,773 tokens. The independent feature suite
  then passed bad-key rejection, tokenize, thinking on/off, streaming usage,
  preserved-thinking multi-turn, structured JSON with thinking, tools and tool
  continuation, and vision.
- OMP 17.1.8 discovered the landing-page YAML at its current
  `~/.omp/agent/models.yml` path, advertised images/tools/reasoning and the
  196,608-token context correctly, and completed an exact secure-endpoint smoke.
- SOUL levels 1, 2 and 3 were exercised. Level 1 produced a structured,
  tool-free journal; level 2 used only the bounded shell tool and correctly
  identified a marked synthetic OOM as a test artifact; level 3's authenticated
  exact canary returned `SOUL-CANARY-OK`. The appliance was returned to its
  shipping level 0 afterward.

A post-run semantic audit found the escalation coherent and useful but exposed
one continuity gap: incident updates and recovery shared a stable Nanobot
session, while the same check failing again after recovery received a new
incident id and could miss the earlier episode. Every analysis now receives a
bounded digest of the eight newest redacted journal entries; daily synthesis
continues to receive up to 50. The next composite protocol exercises levels
1→2→3 early, leaves level 3 active for the rest of the workload, and uses the
landing page's typed, session-erasing provider termination instead of returning
SOUL to level 0 before dashboard teardown.

The combined secure test exposed one real cross-feature defect: SOUL still used
plain loopback HTTP after vLLM had become a direct-TLS listener. The controller
now uses the certificate hostname on the local engine port, and PID 1 maps that
hostname to loopback. This retains normal CA and hostname validation without a
provider NAT hairpin. The same pass raised Nanobot's bounded report allowance
from 1,200 to 2,400 tokens (preventing structured JSON truncation), disables
thinking for the 16-token exact canary, handles `content: null`, and makes the
landing listener's TLS 1.2 minimum explicit. A live certificate test rejected
TLS 1.1 and accepted TLS 1.2 with verification success.

Real OMP load produced 46 successful chat requests with no OOM, preemption,
traceback or engine error:

| reporting metric | composite result |
|---|---:|
| prompt throughput, non-zero mean / median / peak | 2,185 / 2,264 / 3,763 tok/s |
| aggregate generation, non-zero mean / peak | 44.0 / 153.4 tok/s |
| maximum running / waiting requests | 3 / 2 |
| maximum KV use | 96.3% |
| GPU utilization, 120 two-second samples | 92.8% mean / 100% peak |
| GPU power | 507.8 W mean / 586.3 W transient peak |
| GPU temperature | 72 C mean / 79 C peak |

MAL is not applicable: the qualified Qwen profile intentionally runs compiled
MTP-off decoding. Three broad tool-using OMP reviews respected the configured
four-request provider ceiling and kept the server healthy, but hit their
eight-minute client deadline without final reports. Two file-attached,
no-tool reviews then completed in about 90 seconds. Their findings were useful
as leads rather than verdicts: human validation rejected several false
positives and retained the explicit TLS floor hardening.

The two rentals ran for about 43 minutes. Credit moved from approximately
`$35.3211` to `$34.3461` (about `$0.975`, including the unusually material
image/checkpoint ingress charge on this host). Both exact instance IDs were
destroyed, Vast returned an empty instance inventory, and the generated deSEC A
and ACME TXT records were deleted and verified absent.

### Vast.ai live execution (2026-07-26)

The first candidate was terminated when its image pull remained incomplete
for more than 17 minutes. A faster RTX 5090 host then started a cached image in
about 51 seconds; the 0.8B checkpoint downloaded in about 6 seconds and vLLM
became ready after its normal model/JIT initialization.

Passed on the fast host:

- Docker-entrypoint launch mode and one-GPU custom profile.
- Provider detection, mapped API/dashboard/SSH ports, and READY label.
- GPU guard, checkpoint download marker, and model-neutral landing content.
- Dashboard missing/wrong-token rejection and valid-token rendering.
- API `401` without its key and success with the persisted key.
- `/health`, `/v1/models`, `/metrics`, non-streaming chat, SSE streaming,
  final usage chunk, Qwen reasoning parser, and forced tool-call parsing.
- Key-only SSH and an authenticated API request through an SSH tunnel.
- Engine-child termination followed by recovery and acceptance of the same
  persisted API key.
- Provider stop/start with the same checkpoint marker and byte size, no
  weight re-download, and the same API key.

Live evidence exposed two issues that were fixed afterward:

1. Current vLLM uses `reasoning`; the UI only read the legacy
   `reasoning_content` field.
2. The upstream image exports removed `VLLM_CACHE_DIR`; the entrypoint now
   unsets it while retaining supported `VLLM_CACHE_ROOT`.

Vast credit moved from approximately `$21.8431` to `$21.6435`, for about
`$0.20` total including the deliberately aborted slow-host attempts. The final
Vast API check returned zero instances.

### deSEC dynamic DNS and direct TLS (2026-07-26)

The supplied DNS token exposed one zone. A unique documentation-address RRset
was created using the appliance's atomic bulk `PUT`, resolved from deSEC's
authoritative nameserver, updated, deleted, and confirmed absent through both
the API and authoritative DNS.

A second short Vast RTX 5090 run exercised the appliance path itself:

- `model-<instance-id>` registered to the observed public IPv4 address.
- lego created the DNS-01 TXT challenge, Let's Encrypt validated it, and lego
  removed the TXT RRset afterward.
- The issued certificate had the generated hostname as both CN and SAN, a
  trusted Let's Encrypt chain, and a 90-day validity window.
- The direct mapped HTTPS endpoint passed `/health`, authenticated
  `/v1/models`, and a Qwen chat completion.
- The token-gated dashboard accepted TLS on its mapped port; an unauthenticated
  request correctly returned 403.
- The Vast label became
  `Qwen3.5-0.8B-Dynamic-DNS-Smoke READY https://.../v1`, making readiness and
  the final endpoint visible in the console.

The follow-up used about `$0.06` of Vast credit. The rental was destroyed, its
A record was deleted, the record returned API 404, and the final Vast inventory
again returned zero instances.

### Runpod live execution (2026-07-26)

Initial attempts established that advertised machine bandwidth does not predict
Runpod's private registry-provisioning time: earlier Community placements
failed before allocation, and two Secure Pods were terminated before their
containers started. A follow-up tested the hybrid networking revision.

The first follow-up allocation accidentally targeted an RTX 4090 and was
destroyed after 39 seconds when the `sm120+` Blackwell requirement was caught.
A Secure RTX 5090 attempt was then stopped at the original five-minute image
pull limit. At the user's direction, one final retry on the same machine was
given ten minutes:

- The public IP and TCP mappings appeared around 9 minutes.
- The token-gated landing page and deSEC A record proved the entrypoint had
  started around 10 minutes even though Runpod REST still reported
  `runtime: null`.
- The 0.8B model became ready at approximately 15 minutes 47 seconds from Pod
  allocation.

Passed on the Secure RTX 5090:

- Blackwell GPU guard, provider detection, model download, and vLLM/JIT boot.
- Hybrid ports: dashboard on `1111/http`, secure API fallback on `8000/http`,
  and direct appliance TLS on `8443/tcp`.
- deSEC public-IP registration, DNS-01 validation, and ACME TXT cleanup.
- Trusted Let's Encrypt certificate with the per-Pod hostname as CN and SAN.
- Direct and proxy `/health` returned 200.
- Both routes returned 401 without the API key and authenticated
  `/v1/models` returned `qwen-smoke`.
- Direct-TLS and proxy chat completions both returned the requested response.
- The token-gated dashboard and `/chat` rendered through Runpod HTTPS, exposed
  the direct endpoint, and included the multi-turn `Preserve thinking` option.
- Key-only SSH over the mapped public port.

The successful retry ran 1,172 seconds at `$0.99/hour` (about `$0.32`). The
preceding five-minute RTX 5090 attempt was about `$0.09`; the 39-second 4090
correction was negligible. Including earlier placement experiments, documented
Runpod exposure was approximately `$0.55` before storage rounding.

The final Pod was deleted, its A record returned API 404, the ACME TXT count
was zero, authoritative DNS no longer answered, and Runpod inventory returned
zero Pods.

### Authenticated HF Xet throughput (2026-07-26)

A fresh 80 GB Runpod volume on the same Secure RTX 5090 host exercised the
production `qwen36-27b-nvfp4` download with a process-local Hugging Face token:

- The token and the `nvidia/Qwen3.6-27B-NVFP4` model API both returned HTTP
  200. No credential material was written to the repository or retained after
  the test.
- The immutable runtime image supplied `huggingface_hub 1.24.0` and
  `hf-xet 1.5.2`; the downloader set `HF_XET_HIGH_PERFORMANCE=1` before import
  and used its default `MODEL_DOWNLOAD_WORKERS=16`.
- The cold model directory grew from about 20 MB of metadata to
  21,941,628,570 bytes (20.435 GiB). All three safetensor shards were present,
  their index declared 21,921,428,072 weight bytes, and Xet logged zero ERROR
  entries.
- The appliance's `.download-model` to `.download-complete` markers measured
  1,094 seconds (18 minutes 14 seconds): 20.056 MB/s, or 160.451 Mbit/s,
  averaged across metadata and payload.
- Xet adaptive concurrency reduced the transfer to four concurrent ranges
  because it classified the route as struggling. Authentication and the
  high-performance transport therefore worked, but the host-to-CAS path did
  not approach the host's advertised multi-gigabit bandwidth.

The follow-up cost about `$0.30` at `$0.99/hour`. The Pod was deleted
immediately after the completion marker, its ephemeral volume was discarded,
and final Runpod inventory returned zero Pods.

### Remaining economical qualification follow-up (2026-07-26)

Two additional Secure RTX 5090 Runpod attempts were each given the full
ten-minute startup grace. Both landed on the same `EUR-IS-2` machine but
returned no runtime, public address, or port mappings before the cutoff. They
were destroyed without reaching appliance code, and Runpod inventory returned
zero Pods. This added about `$0.33` before storage rounding.

The already-validated Vast RTX 5090 host then completed the provider-neutral
feature rows against the same immutable image:

- Native Qwen vision booted with the multimodal encoder and correctly counted
  two cats in a public test image. The 0.8B model returned the answer through
  `reasoning` with `content: null`, independently confirming the UI
  compatibility case.
- A key-only SSH tunnel carried a 2,520-token prefill to the private API and
  returned `TUNNEL_OK`.
- Terminating the vLLM server made health unavailable; the supervisor started
  a new server, authenticated `/v1/models` returned HTTP 200 with the same
  key, and health recovered.
- A provider stop/start preserved the checkpoint marker byte-for-byte and did
  not redownload the model.
- A separate text-only MTP boot resolved `Qwen3_5MTP`, shared the target
  embeddings and LM head with the draft, and reached health. After first-use
  compilation, an eight-token completion returned in one second; metrics
  recorded six draft tokens and five accepted tokens.

The pinned runtime warned that `qwen3_next_mtp` is a deprecated alias for
`mtp`; the profile and test plan now use the current method name. A cleanup
audit also caught two Vast destroy commands waiting for interactive
confirmation. Both contracts and the final MTP contract were then explicitly
destroyed, and final Vast inventory returned zero instances. The Vast
follow-up was approximately `$0.65` before bandwidth/storage rounding,
including the two contracts that remained live until the confirmation issue
was caught.

### Runpod 590.48.01 / CUDA 13.2 compatibility (2026-07-29)

The same Qwen profile was then exercised on a Runpod Secure RTX 5090 pod whose
`nvidia-smi` reported driver **590.48.01** and CUDA **13.2**. The container had
`/usr/local/cuda/compat/libcuda.so.1`; the run was intentionally admitted with
the old override while the pair was still unqualified.

This was not a boot-only smoke:

- the full appliance feature suite passed, including thinking/non-thinking
  chat, streaming, preserved reasoning, strict structured output, tools and
  native vision;
- the 32K verifier retrieved 3/3 needles without degeneration;
- uncached 8K/32K prefill measured about 2,687/3,885 tok/s and a controlled C1
  decode measured about 64.8 tok/s;
- a separate Vast client installed OMP 17.1.8 from scratch and completed a
  real repository-review workload against the direct deSEC TLS endpoint;
- SOUL levels 1, 2 and 3 were exercised, including a synthetic incident,
  journal continuity and a level-3 canary;
- the authenticated Runpod landing page, proxy fallback, Markdown chat,
  reasoning stream and client-install controls were exercised live.

The appliance then exercised its destructive teardown path. It stopped vLLM,
securely erased 199 session files (7.5 MiB; public weights deliberately kept),
and attempted provider deletion. This Runpod deployment allowed the scoped key
to read its own Pod but refused both REST deletion (HTTP 403) and GraphQL
`podTerminate` (HTTP 403, provider code 1010). Its preinstalled `runpodctl` was
an older release for which current `pod delete` was unknown; account-side
legacy `runpodctl remove pod` immediately removed the Pod. The appliance now
tries both CLI grammars after the two HTTP APIs and passes the key through the
child environment, which remains available even after secure erase removes
`~/.runpod/config.toml`. Unit coverage reproduces the exact fallback sequence.
Live confirmation of the corrected worker is deferred with the GLM
cross-provider pass described below.

The runtime selected its documented Marlin fallback for some FP4 weight-only
components, so these results do not claim a new native-FP4 kernel path. They
do establish the driver/CUDA pair as a supported turnkey configuration. The
entrypoint admission floor is therefore driver `590.48.01` **and** reported
CUDA `13.2`: the earlier Vast `590.48.01 / CUDA 13.1` offer still fails closed,
as does the r580 host that failed NCCL initialization. This one-card result is
not silently presented as a four-GPU GLM performance qualification; each model
profile retains its own feature, quality and performance evidence.

### Current-image Runpod GLM pass deferred (2026-07-29)

The planned four-GPU GLM half of this cross-provider exercise was stopped at
the user's direction because suitable Runpod stock was scarce and cold image
availability was consuming rental time without reaching the container. Three
Secure RTX PRO 6000 Blackwell Server Edition allocations exposed neither SSH
nor container uptime within the 20-minute allowance, including one placement
requested with the provider's `minDownload=200` control. A final allocation
was still pulling the GG v20-r9 bootstrap image when the experiment was
cancelled. No GLM weights were downloaded and no current-image feature,
quality, or performance claim is made from these attempts.

All four server allocations and the separate Vast OMP client were destroyed,
and both provider inventories were verified empty. The completed Runpod Qwen
qualification above remains valid. The earlier v31 Runpod GLM matrix remains
historical evidence only; qualifying the current r9 turnkey image on Runpod is
explicitly deferred until suitable four-card stock is available.

### JarvisLabs IN1 flagship qualification (2026-07-30)

The current GG v20-r9 turnkey appliance was qualified on a JarvisLabs full VM
in IN1 with four RTX PRO 6000 Blackwell Server Edition GPUs, 640 GiB host
memory, a 500 GB root disk, NVIDIA driver 595.58.03 and reported CUDA 13.2.
Jarvis exposed a 600 W limit per card and billed this shape at $7.56/hour,
per minute. Every GPU pair was in the same NUMA node and reported `PHB`, but
CUDA peer reads and writes were unavailable for every pair. The stock image
therefore failed during B12X PCIe/DCP CUDA-IPC collective initialization.

The appliance now performs the peer-access check before calibration and falls
back to NCCL/shared-memory collectives when the advertised GPUs cannot open
peer mappings. With that provider-neutral safety gate, the same flagship
profile reached a verified API without changing its model, DCP2, MTP5,
dynamic-token NVFP4 MLA KV, 524,288-token limit, or 50% DRAM-offload posture.
The first successful boot took 6m57s after the container started, including a
fresh DNS-01 certificate; the already-downloaded checkpoint, persisted
certificate and AOT cache reduced the next engine start to about 3m22s.
Cold image transfer was about seven minutes and the 309 GiB checkpoint about
ten minutes on this host.

The OpenAI-compatible feature gate passed authentication rejection,
tokenization, thinking and non-thinking chat, streamed usage, preserved
multi-turn reasoning, strict structured output both with and without thinking,
automatic and required tool calls, duplicate-tool suppression, and tool-result
continuation. The GLM EXL3 production profile intentionally has vision off, so
this run makes no Jarvis vision claim.

The canonical `llm-inference-bench` v0.4.29 protocol produced:

| metric | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold prefill (tok/s) | 2,417 | 2,444 | 2,354 | 2,228 |

| zero-context sustained decode | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 79.6 | 120.8 | 180.2 | 272.3 |

The canonical run averaged 1,090 W across all four GPUs, peaked at 1,394 W
against a 2,400 W combined limit, and averaged 93% GPU utilization. Its C1 and
C8 phases averaged about 946 W and 1,137 W. A separate request/metrics harness
measured MTP5 mean acceptance lengths of 3.17, 4.48, 3.30 and 2.40 at
C1/C2/C4/C8 respectively; the aggregate decode rates from that shorter
256-output-token matrix were 68.7/103.0/129.2/124.1 tok/s.

The exact long-context gate created a 517,177-token prompt and recovered all
five needles at 1%, 10%, 50%, 90% and 99% in 302.5 seconds without
degeneration. The endpoint remained healthy. A separate 65,533-token
prefix-cache experiment measured 27.90s cold, 0.346s as a GPU hit and 0.427s
after GPU eviction when reloaded from host DRAM. The connector reported
65,280 external-prefix hit tokens, 4.17 GB loaded in 0.328s, and zero
allocation failures.

The generated landing token persisted mode 0600 outside the replaced
container. The unauthenticated page remained read-only, a bad config token
returned 403, and the valid token unlocked chat, configuration and the
multiply-gated termination page. deSEC registration and a trusted Let's
Encrypt certificate worked on the VM's direct public ports. Credential-free
raw evidence is retained under
`test-results/jarvislabs-2026-07-30/`.

The final live gate submitted the landing page's secure-erase option, exact
machine id and permanent-destruction acknowledgement. The worker stopped the
engine, erased the session and destroyed VM 460920 through JarvisLabs. The CLI
subsequently reported zero VMs, zero attached File Storage and a $60.0603
remaining balance; the instance's deSEC A and ACME TXT records were absent.

## Qwen3.6-27B NVFP4 qualification (Vast RTX 5090)

The full pinned `nvidia/Qwen3.6-27B-NVFP4` checkpoint at `0893e160…` was
qualified with the r9-derived turnkey image on Vast instance `46196547`, one
RTX 5090 32 GB with a 575 W limit. The selected profile is native vision on,
TP1, `GPU_MEMORY_UTILIZATION=0.90`, `MAX_MODEL_LEN=196608`,
`MAX_NUM_BATCHED_TOKENS=4096`, `MAX_NUM_SEQS=4`, compiled MTP-off decoding,
checkpoint-selected FP8 E4M3 KV, and `max_pixels=8388608`.

The context/memory sweep separated "booted once" from a safe appliance
default:

| candidate | result |
|---|---|
| native 262,144 at GMU 0.97 / batch 2,048 | KV admission failed; estimated maximum 249,312 |
| native 262,144 at GMU 0.98 | still short at about 260,288 and left no credible vision workspace |
| 221,184 at GMU 0.95 | text and 216K retrieval passed, but full-resolution vision OOMed |
| 204,800 at GMU 0.91 | detailed 5K vision passed, only 177 MiB remained |
| 212,992 at GMU 0.92 | detailed 5K vision passed, only 29 MiB remained |
| **196,608 at GMU 0.90** | repeated vision and near-maximum text passed; about 511 MiB remained after the final image |

The checkpoint processor's 16,777,216-pixel default OOMed on a 5120x2880
dashboard. Capping it to 8,388,608 pixels retained useful detail: two final
runs recovered 17–18 of 18 small dashboard facts, remembered the image in a
follow-up turn, and passed a text-only regression. The second run followed the
near-maximum prefill, so it also checked that long-context activity did not
poison the vision/text path.

The selected scheduler produced:

| measurement | result |
|---|---:|
| uncached prefill 8K / 64K / 180K | 3,703.6 / 3,716.1 / 2,513.3 tok/s |
| aggregate decode C1 / C2 / C4 | 68.0 / 120.9 / 221.6 tok/s |
| KV pool / maximum request | 205,544 / 196,608 tokens |
| exact near-maximum retrieval | 5/5 depths at 192,290 actual tokens, no degeneration |

The 8,192-token scheduler candidate was not an improvement: its larger
activation profile left only 5.42 GiB for KV, an estimated 167,776 tokens, and
failed startup admission for 196,608. A compatible cached restart loaded the
AOT artifact in 1.04 seconds and reached the API in about 43 seconds including
model load, graph capture, and multimodal warmup.

The final feature suite passed bad-key rejection, tokenization,
thinking/non-thinking chat, streamed usage, preserved multi-turn reasoning,
strict JSON with thinking both off and on, one automatic tool call,
`tool_choice=required`, tool-result continuation, and native vision. The final
serve log contained zero engine errors, tracebacks, HTTP 500s, preemptions,
OOMs, FSM failures, or post-EOS matcher warnings.

Speculative controls were rejected rather than exposed as optimistic defaults:

- eager MTP2 had healthy 74–79% draft acceptance and mean acceptance length
  around 2.5, but measured only 46/81/101 tok/s at C1/C2/C4 versus
  68/121/222 compiled without MTP and reduced the context envelope;
- compiled MTP2 hit the FlashInfer frozen `q_len_per_req=1` wrapper when the
  verification step needed 3, so the compatibility path must be eager;
- compiled n-gram hit the same query-shape class and eager n-gram failed its
  acceptance/correctness gate;
- EAGLE and DSpark require compatible trained draft checkpoints that this
  NVIDIA checkpoint does not publish.

The runtime reports mixed ModelOpt NVFP4/FP8/MXFP8 weights and FP8 E4M3 KV.
It warns that the checkpoint does not publish a separate Q scale and therefore
uses the K scale/1.0; the five-depth 192K retrieval and degeneration gates are
the empirical quality evidence for this exact record. It also warns that some
FP4 weight-only components fall back to Marlin on SM120. Those warnings are
documented upstream/runtime characteristics, not hidden by the appliance.

## GLM-5.2 qualification history

### v20 MadeBy561 qualification (2026-07-27)

The production-scale pass began on Vast instance `45997603`, four RTX PRO 6000
Blackwell 96 GB GPUs and an 850 GB disk, then pivoted to the owned AIBeast
four-GPU host for absolute performance and maximum-context work. The rental
was destroyed after the pivot. It was useful for memory-fault reproduction
and same-host A/B tests, but not as an absolute performance reference:

- GPU 0 reaches the other three GPUs through `SYS`; GPUs 1–3 are `NODE` and
  GPUs 2–3 are `PIX`.
- The machine has two CPU sockets and four NUMA nodes.
- CUDA peer reads/writes work, while native peer atomics do not.
- The provider charged `$7.248/hour`, including the enlarged disk.

The extensive AIBeast results in this section are pinned to the following
tested envelope:

| component | tested value |
|---|---|
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB reported per GPU |
| topology | all four GPU paths `NODE`, one NUMA node |
| power limit | 280 W per card |
| CPU / host RAM | AMD Ryzen Threadripper 9970X, 32 cores / 64 threads; 251 GiB |
| host OS / kernel | Ubuntu 24.04.4 LTS; Linux 6.8.0-136-generic |
| container runtime | Podman 4.9.3 |
| NVIDIA driver | 595.71.05 |
| driver-reported CUDA compatibility | 13.2 |
| `nvidia-smi` client banner | 580.95.05 |

The CUDA value is the compatibility level printed by `nvidia-smi`, not a
claim about every toolkit bundled in the container. A forthcoming AIBeast
driver/CUDA refresh is a requalification boundary: compile caches must be
isolated or invalidated as appropriate, then cold retrieval, memory headroom,
features, and the compact performance matrix must be repeated before new
measurements are compared with this record.

Both checkpoints fit simultaneously: approximately 303 GB for EXL3 and
341 GB for `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. The authenticated
MadeBy561 Hugging Face Xet transfer completed in 3 minutes 45 seconds. Local
safetensor reads take about 34–36 seconds, but full engine restarts take
roughly 4–7 minutes after compilation, warmup, memory profiling, and CUDA
graph capture. The appliance's 32K retrieval gate adds about four minutes on
this topology.

#### Preserved AIBeast control

The owned-production control is the same MadeBy561 checkpoint at revision
`68babde27a97a4c980c2494e830dd424975cd5a3` on the v19 image. Its four GPU
paths are all `NODE` on one NUMA node. The production launch uses
TP4/DCP4/MTP3, probabilistic proposals, a 3,072-token prefill chunk,
`nvfp4_ds_mla` KV, a 537,600-token pool, and 128 GiB of host KV offload.
It also uses the compressed `F8_DMA=ring` path and clamps the checkpoint's
1,048,576-position metadata to the served 524,288-token context so unused
target/draft BF16 RoPE rows are not allocated.

Trusted isolated measurements recorded in its handoff are 2,299 prompt
tokens/s at 8K, 2,192 at 64K, and 119.2 output tokens/s at C1. Production logs
observed during ordinary traffic showed roughly 92–96 output tokens/s and mean
speculative acceptance length around 2.94–3.12. Prior seeded retrieval was
clean at 490K and 505K.

Before experimentation, the exact v19 image was tagged locally, its launcher,
override, patches, scales and checksums were copied under
`/mnt/vault/llm/vllm+lmcache/control-backups/20260727-control-v19`, and the
stopped control container was retained. The turnkey mounted the same NFS
checkpoint and completion marker read-only, with separate writable cache/state
volumes. `SOUL_AUTONOMY_LEVEL` and its startup ceiling remained zero.

#### v20 MadeBy561 memory search

Every candidate used TP4/DCP4, MTP3, the checkpoint-native draft (serialized
NVFP4 experts with the checkpoint's MXFP8 dense overlay), calibrated
`nvfp4_ds_mla` KV, synchronous scheduling, B12X MLA/MoE, and a maximum request
length of 524,288. The v20 topology calibrator selected lossless PCIe DMA at a
393,216-byte crossover, enabled DCP query splitting from 8,192 context tokens,
and disabled CKV prefetch overlap on this cross-socket host; its microbenchmark
found DMA about 61–63% faster than NCCL above the crossover.

| candidate | result |
|---|---|
| auto pool, GMU 0.96, batch 3,072 | startup KV admission failure |
| auto pool 551,680 tokens, GMU 0.98, batch 3,072 | first 32K request OOM; only 24.75 MiB free for a 36 MiB NF3 target allocation |
| pinned 524,288-token pool, batch 2,048, workspace 1,024 MiB | 32K 3/3 and full feature suite passed |
| same pin, batch 3,072, workspace 1,024 MiB | one 32K pass, then later 32K/benchmark OOM |
| same pin, batch 3,072, workspace 512 MiB | three uncached 32K passes and C1–C8 passed, but a 520,192-token request immediately OOMed |
| same pin, batch 2,048, workspace 512 MiB, lossless | 32K repeated 3/3; 521,276 tokens retrieved 5/5 depths |
| same pin and transport crossovers, FP8 ring | selected candidate; 521,277 tokens retrieved 5/5 depths |

The failure shape is important. Startup admission and one short request are
not sufficient evidence for this target: the NF3/MTP transient allocation is
not fully represented by the apparent KV headroom. The configurator therefore
supplies the pool, chunk, workspace, utilization, native draft, and proposal
method as one `madeby561-hybrid` variant default while retaining explicit
per-knob overrides.

The selected candidate's required feature suite passes discovery/auth mode,
exact tokenization, ordinary and thinking chat, SSE usage, multi-turn with
preserved reasoning, one automatic tool call, and tool-result continuation.
Structured JSON also passed but remains informational. Forced
`tool_choice=required` emitted five duplicate calls on this build; automatic
tool choice emitted exactly one, so normal agentic workloads remain a release
gate while forced mode does not.

#### AIBeast absolute performance

AIBeast has four all-`NODE` RTX PRO 6000 Blackwell 96 GB GPUs on one NUMA node.
All cards were power-limited to 280 W. Driver 595.71.05 reported CUDA 13.2;
the `nvidia-smi` client banner was 580.95.05. Each concurrency level used
eight unique 1K prompts requesting 512 output tokens:

| target / candidate | unique prefill | aggregate decode / MAL |
|---|---|---|
| v19 daily-driver control | 2,299 tok/s @8K; 2,192 @64K | C1 119.2 |
| v20, lossless DMA | 2,474 @1K; 2,581 @8K; 2,142 @32K; 1,925 @66K | C1 121.8 / 3.789; C2 142.6 / 3.668; C4 205.8 / 3.899; C8 267.6 / 3.876 |
| **v20, FP8 ring selected** | **2,286 @1K; 2,701 @8K; 2,176 @32K; 1,987 @66K** | **C1 121.6 / 3.941; C2 142.3 / 3.657; C4 208.4 / 3.954; C8 269.7 / 3.913** |

Both v20 sweeps completed without request failure or preemption. The selected
profile also passed three fresh 32K probes (15/15 needles total). The corrected
live-tokenizer harness built a 521,277-token haystack and recovered five codes
at depths 1%, 15%, 50%, 90% and 99% while KV occupancy reached 99.5%.

For reference, the discarded Vast results were 271 @1K, 392 @8K, 141 @32K
and 12–18 tok/s C1 across candidates. Its mixed topology explains the order-of-
magnitude gap; those measurements remain useful only as a same-host A/B.

Periodic vLLM logger values were not used as these throughput measurements.
Its default ten-second logger increments prompt tokens only when a scheduled
chunk completes, then resets the interval. A 2,048-token chunk therefore
prints `204.8`, `0`, `204.8`, `0` when successive chunks straddle alternating
buckets, and `409.6` when two land together. Exact unique prompt tokens divided
by end-to-end prefill time are the comparable metric.

### v20 vision qualification (2026-07-27)

The read-only EXL3 derivative installed its vision plugin from a writable
temporary copy, leaving the checkpoint untouched. DCP4 at
`GPU_MEMORY_UTILIZATION=0.975` loaded 79.65 GiB/GPU versus 77.66 GiB/GPU for
the text target and exposed 564,736 KV tokens. A deterministic 5120x2880
Retina-style dashboard then produced:

| probe | result |
|---|---|
| detailed first turn | 17/18 exact requested details; 11.395s |
| prompt accounting | 2,131 tokens, including 2,074 multimodal tokens |
| image-history follow-up | exact three requested values; 0.712s and 2,048 cached tokens |
| text-only regression | exact; 0.185s |
| cold 32K retrieval gate | **0/3; degenerate output** |
| speculative behavior at failure | MAL 1.25–1.50, average draft acceptance about 5–10% |

Capacity and correctness were separate failure modes. At utilization 0.98,
the same DCP4 wrapper exposed 610,560 KV tokens but left 37.12 MiB free and
OOMed on its first 48 MiB verification allocation. At 0.975 it served short
vision reliably, and persistent AOT reuse reduced the main-backbone compile
phase from about 103 seconds to a 2.03-second cache load, but long-context
retrieval still failed. Vision remains opt-in and short-context-only; the
flagship text profile keeps it off.

Preserved artifacts on AIBeast:

- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/glm52-vision-5k-result.json`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/glm52-vision-5k-dashboard.png`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/vision-dcp4-gmu0975-safetensors.log`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/vision-dcp4-gmu0975-safetensors.inspect.json`

This failure does not establish that the Baseten MoonViT/PatchMerger graft is
inherently unsafe. Upstream evidence narrows the diagnosis:

- Chronarion's published
  [MadeBy561 vision merge](https://huggingface.co/chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid)
  uses byte-identical hybrid text shards and reports MTP-5 text/vision parity.
  Its plugin is the source already selected by this appliance.
- Jarrel's
  [NVFP4+AQLM vision merge](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid)
  reports exact mixed image-plus-needle retrieval at 130K with healthy MTP.
  Its runtime delegates missing top-level wrapper fields to `text_config`,
  preserves standalone inner-model names, and remaps checkpoint names only
  while loading weights.
- 0xSero's
  [EXL3/TR3 vision merge](https://huggingface.co/0xSero/GLM-5.2-TR3-Vision)
  shows the same class of long-prompt MTP collapse, although its published
  long probes are repeated-token performance tests rather than retrieval
  correctness tests.

The quant-family discriminator was then run against the published MadeBy561
wrapper with MTP completely disabled. At `GPU_MEMORY_UTILIZATION=0.94` it
passed all three short text controls and recovered all three 32K needles in
14.56 seconds, but recovered only 1/18 requested values from the same 5K
dashboard and hallucinated repeated fields. This rules out speculative
drafting and makes EXL3 expert ordering insufficient as the sole explanation.
It does not prove that the tower cannot perform simpler image tasks.

Jarrel's independently successful wrapper differs in two relevant ways: it
delegates absent top-level configuration fields to `text_config`, and confines
checkpoint-name remapping to weight loading rather than leaving a mapper on
the live inner language model. A transient read-only copy of those corrections
was applied to the MadeBy561 plugin. It booted cleanly and retained 3/3 short
and 3/3 32K text retrieval, but improved the detailed dashboard by only one
field, to 2/18. The correction was therefore not promoted as a vision-quality
fix.

The wrapper change also produced the expected performance null result:

| MadeBy561 vision wrapper | periodic 32K PP | steady C1 TG |
|---|---:|---:|
| published Chronarion plugin | 2,457.4 tok/s | 49.6–49.7 tok/s |
| Jarrel-style mapper/prefix correction | 2,471.2 tok/s | 49.2 tok/s |

The +0.6% PP and roughly -1% TG movements are run noise, not a throughput
change. A seeded benchmark of the corrected MTP-off arm measured 2,652.4 and
2,566.9 tok/s PP at 8K and 32K. This is deliberately reported separately from
the flagship MTP-on profile: disabling its native speculative draft explains
the lower C1 generation rate.

The evidence now supports a narrower conclusion. The EXL3 composition is good
at detailed short-image extraction but fails long text; the currently
published MadeBy561 composition is text-safe at 32K but fails detailed image
extraction. Neither is a generally qualified multimodal flagship. Successful
upstream simple-image and mixed image-plus-needle results remain credible, but
they do not substitute for this appliance's Retina-dashboard gate.

Additional preserved artifacts on AIBeast:

- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/madeby-vision-mtpoff-5k-result.json`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/madeby-vision-wrapperfix-5k-result.json`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/madeby-vision-wrapperfix-benchmark.json`
- `/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/madeby-vision-wrapperfix.log`

### v31 cross-provider flagship (2026-07-28)

#### Runpod Secure: full release matrix

The release candidate
`ghcr.io/malaiwah/glm52-exl3-vast:v31-rental-candidate-20260728` ran on a
Runpod Secure Pod with four RTX PRO 6000 Blackwell Workstation cards. The
cards exposed 97,887 MiB each, driver 610.43.02 and a 600 W/card power limit.
GPU 0/1 and 2/3 were separate `NODE` islands with `SYS` paths between them.

Cold click-to-health was 25m13s:

- registry-image pull: about 8m24s;
- authenticated 309 GiB Brandon checkpoint download: 3m35s, roughly
  1.3 GiB/s;
- InstantTensor target load: 169.23s; draft load: 3.81s; complete model load:
  180.47s;
- first backbone/draft compilation: 105.69s + 7.34s.

A restart reused the persisted AOT artifacts. The target/draft loaded in
192.71s + 3.68s, while the compile phases fell to 2.13s + 4.74s. Short and
long output remained correct and throughput remained in family, disproving
the prior cache-corruption concern on this exact runtime/cache key.

`GPU_MEMORY_UTILIZATION=0.978` admitted 534,272 KV tokens but left only
7.94 MiB free; the first benchmark request needed a 36 MiB EXL3 transient
allocation and OOMed. This is a runtime-headroom failure even though boot and
KV admission passed. At 0.976 the profile exposed 523,264 KV tokens, retained
about 564 MiB steady-state free memory (about 98 MiB under C8), and completed
the matrix without OOM.

The required feature suite passed authentication, tokenization, ordinary
chat, visible thinking content, streaming usage, multi-turn history with
optional preserved reasoning, structured JSON, one automatic tool call and
tool-result continuation. `tool_choice=required` still emitted five duplicate
calls and remains an optional diagnostic. Vision was deliberately off.

The exact 516,096-token target produced a 517,176-token request and recovered
all five needles at 1%, 10%, 50%, 90% and 99% in 196.17s without degeneration.
The engine remained healthy.

Canonical `llm-inference-bench` v0.4.29 results:

| measurement | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 3,316 | 3,443 | 3,152 | 3,126 |
| focused prefill tok/s | 3,554 | 3,449 | 3,357 | 3,114 |
| focused total GPU power | 1,728.93 W | 1,785.56 W | 1,923.56 W | 1,803.15 W |

| decode context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| zero | 63.0 | 155.8 | 223.1 | 343.5 |
| 32K | 56.7 | 149.4 | 217.1 | 334.3 |
| 128K | 61.1 | 136.4 | capacity-limited | capacity-limited |

The complete benchmark averaged 1,457.24 W and peaked at 1,949.96 W under a
2,400 W aggregate card limit. Zero-context C1 averaged 1,012.15 W; C8 averaged
1,742.53 W. These are GPU telemetry totals, not wall-outlet system power.

Both Runpod routes passed external TLS and authenticated chat: the managed
HTTPS proxy, and the direct deSEC/Let's Encrypt TCP endpoint on mapped port
8443. The latter avoids the managed proxy's documented 100-second connection
limit. The Pod was deleted after qualification and a follow-up API read
returned 404.

Artifacts are preserved under
`evidence/runpod-secure-20260728/` in the qualification workspace.

#### Release inputs and upstream audit

Docker Hub reported v31 as the newest `verdictai/glm52-exl3-sparkinfer`
publication, at the exact pinned digest
`sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff`.
The runtime identifies itself as vLLM `0c79e41`, SparkInfer integration
`c3828fd`, FlashInfer integration `801d57a`, CUDA 13.2.

The Hugging Face API reported that all selected revisions were current:
BrandonMusic `9297b9f1…`, MadeBy561 `68babde2…`, and NVIDIA Qwen NVFP4
`0893e160…`. The Qwen revision is now pinned too, even though that 27B profile
remains a lower-cost residual GPU test.

The same-day upstream review found two relevant draft/opt-in changes:
per-token outer scaling for NVFP4 KV
(`local-inference-lab/vllm#189` + `sparkinfer#86`) and deterministic
oldest-boundary sparse-indexer tie handling (`sparkinfer#84`). Neither was a
release-ready default. The current calibrated per-layer scale path already
passed the exact ~517K gate, so adopting either change requires a separate
cold five-depth quality and performance comparison. The v31 image already
contains the runtime-stride, page-table-offset, PCIe output-lifetime,
query-split crossover and partial-indexer topology work relevant to this
profile.

#### Vast Community: full release matrix

Vast instance `46068195` used four RTX PRO 6000 Blackwell Workstation cards,
driver 610.43.03, CUDA compatibility 13.3, 600 W/card limits and an all-`NODE`
GPU topology. The candidate image was cached but the Brandon checkpoint was
not. Click-to-health was approximately 55 minutes: about 48 minutes for the
authenticated 309 GiB checkpoint transfer at roughly 0.9 Gbit/s, then about
six minutes for model load, profiling, compilation and graph capture.
InstantTensor loaded the target in 42.35 seconds; compilation took 85.51
seconds. The stable profile exposed 523,264 KV tokens.

All required external-path feature checks passed through the deSEC hostname
and mapped TLS port. Normal automatic tool choice emitted one call;
`tool_choice=required` reproduced the same optional five-call duplication seen
on Runpod. Vision was deliberately disabled.

The exact near-maximum prompt contained 517,176 tokens and recovered all five
needles at 1%, 10%, 50%, 90% and 99% in 244.10 seconds. No degeneration was
detected and the engine remained healthy.

Canonical `llm-inference-bench` v0.4.29:

| measurement | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 3,001 | 2,968 | 2,916 | 2,716 |
| focused prefill tok/s | 3,046 | 2,939 | 2,875 | 2,700 |
| focused total GPU power | 1,615.27 W | 1,673.14 W | 1,686.95 W | 1,695.67 W |

| decode context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| zero | 78.5 | 140.8 | 210.1 | 330.8 |
| 32K | 73.4 | 144.8 | 210.1 | 313.5 |
| 128K | 75.4 | 140.8 | capacity-limited | capacity-limited |

Zero-context decode power averaged 1,241.19 W at C1, 1,535.20 W at C2,
1,592.34 W at C4 and 1,739.02 W at C8. The all-`NODE` topology materially
improved C1 over the two-island Runpod Secure host. The rental and its deSEC
A/TXT records were deleted after artifacts were copied; the final provider and
DNS API reads returned no resource.

Artifacts are preserved under
`evidence/vast-community-20260728/` in the qualification workspace.

#### Same-day GG v20-r5 requalification boundary

After the v31 provider matrix completed, Docker Hub published a newer common
upstream image:

`voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5`

at manifest
`sha256:7b230b45991d93065d99c863fdb9ae030fb49592b59fa3c930cc00bfde09e51d`.
Unlike the older common GG image, r5 carries the EXL3/Trellis source
integration directly. Its image labels record GG base `4247d676`, vLLM
integration tree `936ed48` including EXL3 PR #190, SparkInfer integration tree
`f532ec9` including Trellis PR #49, and FlashInfer `801d57a`.

The turnkey candidate was rebased to this immutable r5 manifest. The full
local contract suite passed before publishing a GPU-test branch image. The
v31 provider numbers above remain historical evidence and are not silently
relabelled as r5 results; r5 requires its own cold boot, feature, retrieval,
memory and compact performance gates before promotion.

The first r5 rental attempts found a provider-admission issue before model
qualification:

- Runpod Secure assigned four Server Edition cards on driver 580.126.09. The
  image and checkpoint became available, but all four workers failed their
  initial NCCL all-reduce with `unhandled cuda error`; the Pod was terminated
  after about 11 minutes.
- The next available Vast offer reported driver 590.48.01 and
  `cuda_max_good=13.1`; it was terminated during image pull, before checkpoint
  download.

CUDA 13.2 GA is paired with Linux driver 595.45.04. The entrypoint now rejects
an older driver before downloading weights, unless an operator explicitly sets
`ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1` for a separately qualified
`cuda-compat-13-2` installation. Unit coverage includes the rejected r580/r590
and accepted r595/r610 branches. Neither rejected rental remains allocated.

This was the conservative admission posture at that point in the chronology.
The July 29 Runpod Qwen qualification below later promoted the exact
`590.48.01 / CUDA 13.2` pair and changed admission to test driver and reported
CUDA together. The earlier Vast `590.48.01 / CUDA 13.1` shape remains rejected.

### GG v20-r5 AIBeast flagship (2026-07-28)

The final candidate
`ghcr.io/malaiwah/glm52-exl3-vast:c083aa6a1d84bc6030a76236ee4f80bb4a2b6881`
was qualified on four RTX PRO 6000 Blackwell 96 GB cards. AIBeast used loaded
driver 595.71.05, CUDA 13.2, an all-`NODE` peer topology, the effective NVIDIA
P2P override, and a 280 W/card power limit. The runtime reported GG v20-r5,
vLLM `936ed48`, SparkInfer `f532ec9`, FlashInfer `801d57a`, and native EXL3
integration from PR #190 / Trellis PR #49.

The qualified shape was TP4/DCP2, external rank-sliced TR3 MTP-5,
probabilistic draft sampling, standard rejection sampling, 3,072 batched
tokens, eight sequences, CUDA graph and Trellis capture through 64, calibrated
`nvfp4_ds_mla` KV, InstantTensor, 140,000-token full-CKV gather, 50% aggregate
DRAM prefix offload, utilization 0.976, and vision off.

GG r5 safely accounts for a measured 0.81 GiB/GPU retained CUDA-graph pool.
At the former 520,192 request limit, KV validation correctly failed: 8.97 GiB
was required and 8.88 GiB was available, for an estimated 514,944-token
ceiling. The selected `MAX_MODEL_LEN=513536` preserves 1,408 tokens of
admission margin without disabling the new graph estimator.

The read-only target and draft loaded in about 33 seconds with InstantTensor.
A fresh compile-cache boot reached health in 4m35s. Reusing the same cache
loaded the backbone AOT directly (`torch.compile` 0.45s) and reached health in
2m25s; a subsequent production-port restart took 2m02s. The draft head still
spent about two seconds compiling. Output correctness and performance remained
normal after reuse.

All required feature-suite checks passed both on isolated port 18000 and the
final production port 8000: tokenization, thinking and non-thinking chat,
streaming usage, multi-turn with preserved reasoning, structured JSON
(informational), one automatic tool call, and tool-result continuation.
`tool_choice=required` continued to emit five calls and remains an optional
diagnostic. Vision was deliberately disabled.

Fresh-cache compact results:

| measurement | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| prefill tok/s | 1,890 cold-first-cell | 2,732 | 2,650 | 2,496 |

| decode context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| ~1K | 142.4 | 150.2 | 223.9 | 285.9 |
| MTP mean acceptance length | 5.59 | 5.03 | 4.74 | 5.56 |

Cache-reused compact results:

| measurement | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| prefill tok/s | 2,853 | 2,749 | 2,658 | 2,504 |

| decode context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| ~1K | 138.8 | 189.8 | 223.8 | 312.1 |
| MTP mean acceptance length | 5.34 | 5.41 | 5.04 | 5.39 |

Canonical `llm-inference-bench` v0.4.29 at commit `86cf05c`:

| measurement | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold unique-prefix prefill tok/s | 2,827 | 2,535 | 2,652 | 2,369 |

| decode context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| zero | 106.7 | 145.6 | 207.9 | 284.5 |
| 32K | 103.9 | 144.3 | 203.2 | 274.1 |
| 128K | 102.1 | 135.8 | capacity-limited | capacity-limited |

The capacity-limited cells are expected: the product requirement is one solo
near-maximum session plus useful C1-C8 aggregate throughput, not four or eight
simultaneous 128K sessions. Whole-run GPU power averaged 1,055.67 W and peaked
at 1,127.91 W under a 1,120 W aggregate configured limit. Zero-context C1
averaged 1,084.14 W and C8 averaged 1,109.73 W.

Two independent uncached near-maximum requests used different seeds and exact
served-tokenizer counts of 510,534 and 510,535 document tokens. Both recovered
all five needles at 1%, 10%, 50%, 90% and 99%, with no degeneration, in 231.06
and 234.97 seconds. The precise envelope probe completed a 507,902-token served
prompt plus 4,096 generated tokens (511,998 total) in 267.15 seconds. TTFT was
224.70s, decode TPOT 10.37ms (about 96.5 tok/s), and MTP mean acceptance length
was 3.43.

The benchmark prompt builder previously accepted a three-percent token-count
error, which could overshoot a 500K request by thousands of tokens. It now
iterates to 0.01% with a 32-token floor; the exact-envelope result above used
that corrected harness.

After the final profile returned to production port 8000, live agent traffic
kept MTP-5 mean acceptance length between 3.90 and 5.13, with 58.0–82.7%
average draft acceptance across populated ten-second windows. Historical
MadeBy561/DCP4 MTP-3 field windows were 2.70–3.53 MAL and 56.8–84.4% average
draft acceptance. Those targets and runtime shapes differ, so this is a
real-workload sanity check rather than the matched A/B; the controlled GSM8K
result (51.5 tok/s at MTP-3 versus 53.4 at MTP-5) remains the selection
evidence.

Artifacts and SHA-256 checksums are preserved under
`evidence/aibeast-r5-20260728/` in the qualification workspace. The r5
container remains healthy on AIBeast port 8000 with zero restarts; the prior
production container is retained stopped as an exact rollback.

### GG v20-r8 upstream refresh smoke (2026-07-28)

The published appliance at commit `7b0bc08a16807d6de78ede0e6f8987be4bddd8c6`
and manifest `sha256:8ea41df6a4102ebf8a5b58e5cd1e7589897790c14bb974dac11e58ef4a0353fd`
was cold-started on Vast instance `46142351`: one RTX 5090, NVIDIA driver
595.71.05, CUDA 13.2 compatibility, and the `Qwen/Qwen3.5-0.8B` custom-family
smoke checkpoint. The exact immutable GG base was v20-r8. The container reached
the running state after about 7.9 minutes and the API health endpoint passed
after 13 minutes 25 seconds. The model download itself took about four seconds
and weight loading 0.82 seconds; most of the remaining interval was the cold
image pull followed by first-use compilation and kernel warm-up.

The runtime banner reported the expected r8 build, model architecture
resolution and structured-output/speculative-decode patch preflight all
succeeded, and no engine fatal error occurred before health. This is a base
runtime/startup smoke only: the harness treated the landing page's expected
authenticated HTTP 403 as a dashboard failure and therefore released the
rental before running the feature suite. It must not be cited as GLM quality,
throughput, required-tool-choice, or full-feature evidence. Both Vast and
RunPod inventories were verified empty afterward; AIBeast production was not
restarted or modified.

### GG v20-r9 dynamic-token production (2026-07-28/29)

The r9 appliance image
`ghcr.io/malaiwah/glm52-exl3-vast:171083b3f89534a4d36f202bf70fb11bd19ce1ea`
was qualified on four RTX PRO 6000 Blackwell GPUs at 280 W/card, NVIDIA driver
595.71.05 and CUDA 13.2. The production shape was TP4/DCP2, MTP5, 3,072
batched tokens, InstantTensor, 513,536 maximum model length, 50% host-DRAM
prefix offload, dynamic-token `nvfp4_ds_mla` with FP8 RoPE, and vision off.

Two independent prompt-logit comparisons against the BF16 reference each
reported mean KLD `0.1167701184931591` across 2,047 positions. This is about
2.3% below D-Rock's reported single-run TR3 value of 0.119525; it is a
repeatability and relative-divergence result, not proof of semantic equivalence.

The memory ladder found that boot admission alone was insufficient:

| GMU | exposed KV | result |
|---|---:|---|
| 0.976 | 677,504 | first real request OOMed on a 36 MiB transient with 25 MiB free |
| 0.9675 | 623,232 | exact long-context gate plus a live client OOMed on a 108 MiB all-gather with 91 MiB free |
| 0.9600 | 575,488 | 510,533-token five-depth retrieval passed 5/5; complete feature suite passed |
| **0.9550** | **532,224** | 510,533-token five-depth retrieval passed 5/5 with no degeneration while ordinary clients queued; about 1 GiB/GPU remained late in prefill |

The final 0.9550 shape also passed tokenization, thinking/no-thinking chat,
streaming usage, multi-turn preserve-thinking, strict JSON with and without
thinking, automatic and required tool calls, and tool-result round-trip.
Observed long-prefill log windows were roughly 1,843–2,765 tok/s, mostly
2,150–2,458 tok/s. Live decode windows reached 152.3 aggregate tok/s; MTP5
mean acceptance length ranged with workload and reached 4.74. The complete
final log contained zero OOMs, tracebacks, engine ERRORs, XGrammar FSM errors,
or overlap-probe events.

A separate startup failure was traced to the persistent zero-byte
`sparkinfer_pcie_dma_ext/lock` left by an interrupted PyTorch extension build.
Rebooting did not clear the volume. Atomically renaming the ownerless sentinel
unwedged startup immediately; a later clean build created and removed its own
lock normally. The appliance now performs a bounded, process-aware preflight
that only quarantines this known sentinel and never touches `.ninja_lock`.

#### Full-512K scheduler and CKV-gather selection (2026-07-29)

The same immutable r9 appliance was retested on AIBeast at GMU 0.955 to
convert remaining VRAM headroom into a real binary 512K request limit. The
selected DCP2 shape is `MAX_MODEL_LEN=524288`,
`MAX_NUM_BATCHED_TOKENS=3072`, and
`DCP_CKV_GATHER_MAX_TOKENS=140000`. It exposed 543,488 GPU KV tokens.

Matched unique-prefix prefill results favored the existing scheduler:

| candidate | 8K | 64K | 128K | 180K | result |
|---|---:|---:|---:|---:|---|
| batch 3,072 / gather 140K | 2,822.8 | 2,634.9 | 2,483.0 | 2,444.1 | selected |
| batch 3,200 / gather 140K | 2,784.9 | 2,599.3 | 2,457.3 | 2,362.8 | slower at every measured length |
| batch 3,072 / gather 192K | — | 2,600.4 | 2,461.9 | 2,335.3 | slower despite covering the 180K prompt |

The 3,200-token candidate also reduced exposed KV capacity to 533,248 and
increased the EXL3 Trellis arena from 1,054.2 to 1,094.2 MiB. The 192K gather
candidate exposed 539,904 KV tokens and consumed another 56 MiB/GPU. Matched
decode sweeps did not show a systematic gather-dependent change: the longer
C8 controls were 284.78 tok/s at 140K and 284.89 tok/s at 192K. Lower
concurrency varied with MTP acceptance length. Batch 3,200 provided no
consistent decode gain and its longer C8 result was 253.19 tok/s, so neither
candidate justified its PP/capacity cost.

The selected shape then passed a 521,275-token prompt in 234.3 seconds,
retrieving all five needles at 1%, 10%, 50%, 90%, and 99% with no
degeneration. KV use peaked at 94.1%; 576 MiB/GPU remained after the gate.
The complete API suite passed on isolated port 18000 and again after
promotion to production port 8000. Production advertises
`max_model_len: 524288`; startup and the post-promotion suite contained no
OOM, worker loss, stale-lock, XGrammar FSM, or engine errors.

### GG v20-r11 flagship + LMCache (2026-07-30)

The r11 appliance was qualified on AIBeast's four RTX PRO 6000 Blackwell
96 GB cards at 280 W/card, driver 595.71.05 and CUDA 13.2. The exact base
reports vLLM `0.11.2.dev280+…r11`, LMCache `0.5.2+glm52dcp.4`, XGrammar
0.2.5, and source fingerprint
`vllm4247d67653-b12xf9be272495-164b028d1c7c1c6b`.

The selected 3.0bpw control is TP4/DCP2, external rank-sliced TR3 MTP5,
probabilistic draft sampling, dynamic-token NVFP4 MLA KV with FP8 RoPE,
3,072 batched tokens, 140,000-token CKV gather, GMU 0.957, safetensors,
LMCache over 50% aggregate host DRAM, and a 524,288-token request limit.
Vision is off.

Cold startup loaded all 81 target shards in 91.36 seconds, completed model
load in 135.08 seconds, compiled the backbone in 39.15 seconds and the
speculative head in 4.05 seconds, then captured graphs in about 30 seconds.
The API reached health in about 9m46s including cold extensions/AOT and
exposed 542,208 logical KV tokens.

The complete feature suite passed tokenization, thinking/no-thinking,
streaming usage, preserve-thinking multi-turn, strict JSON in both reasoning
modes, automatic and required tool calls, and tool-result continuation. An
exact 522,360-token request retrieved all five needles at 1%, 10%, 50%, 90%
and 99% without degeneration in 233.97 seconds. Steady prefill log windows
were mostly 2,150–2,458 tok/s.

A matched temperature-one benchmark measured:

| metric | result |
|---|---:|
| uncached PP 32K / 128K | 2,479.9 / 2,393.1 tok/s |
| aggregate TG C1 / C8 | 149.3 / 387.5 tok/s |
| MTP5 mean acceptance length C1 / C8 | 5.16 / 5.74 |
| average draft acceptance C1 / C8 | 83.2% / 94.8% |

One unrelated client overlapped part of that matrix, so it is a loaded
production observation rather than an isolated absolute record.

The final source-exact image (including the parity and mixed-K loader
patches) was then run on a private `:18000` maintenance endpoint to exclude
agent traffic. It passed the complete feature suite and measured 2,665.3 /
2,484.5 tok/s at 32K / 128K, 156.0 tok/s at C1 and 353.1 tok/s at C8. MTP5
MAL was 5.36 / 5.29 with 87.1% / 85.8% draft acceptance; all requests
completed without preemption or error. A preceding `:8000` sample was
discarded because server logs proved unrelated requests were running and
waiting throughout it.

The same immutable image then warm-started against its existing compiled
cache. The backbone AOT artifact loaded in 0.55 seconds, the small
speculative head compiled in 3.63 seconds, graph capture took about 11
seconds, and the exact response `CACHE-REUSE-OK` was returned. The API was
healthy in about 6m22s despite concurrent checkpoint traffic through the same
storage cache, and exposed 553,472 logical KV tokens. There was no corruption,
degeneration, or historical low-throughput cache failure. Compiled artifacts
are therefore retained only inside the immutable runtime fingerprint plus
turnkey EXL3 patch-ABI namespace.

The dynamic-NVFP4 KLD protocol independently reproduced mean
`0.1167701184931591` over 2,047 positions. The reference bundle contains one
window; standard deviation zero is therefore structural (`n=1`), not a claim
of zero variance.

LMCache's four-worker 125 GiB DRAM tier remained healthy through the feature,
performance and exact maximum-context gates. Its optional filesystem L2 was
separately tested at `/mnt/fast/lmcache/glm52-r11` with a 32 GiB hard limit.
The first 64K write cost 166.9 seconds, so disk is not a latency-free default;
after complete engine replacement, the identical prefix restored from L2 in
0.465 seconds and proved persistence. The directory consumed about 3.0 GiB.
The production default is DRAM-only; filesystem L2 is opt-in, local RAID0
NVMe only, never the NFS checkpoint store.

Credential-free artifacts are retained under
`/mnt/fast/build/r11-qualification/` on AIBeast.

### GG v20-r11 mixed 3.25-bpw (2026-07-30)

`willfalco/GLM-5.2-EXL3-TR3-3.25bpw` revision `61d2b6b7…` is a
315.9 GiB mixed-K checkpoint: 192 routed experts per layer use K=3 and 64 use
K=4, including the checkpoint-native layer-78 draft. The stock r11 loader
retained obsolete rank-sliced scale backings while preparing every tier and
would deterministically OOM. The appliance's hash-pinned mixed-K patch drops
each consumed source backing and releases the allocator cache after a prepared
layer; the final target plus native MTP5 loaded at 82.93 GiB/GPU.

The qualified shape is TP4/DCP4, native MTP5 with probabilistic proposals,
dynamic-token NVFP4 MLA KV plus FP8 RoPE, batch 3,072, CKV gather 140,000,
GMU 0.957, safetensors, vision off, and exactly 2,048 pinned blocks. On this
stack, `2,048 × 64 × DCP4 = 524,288` logical KV tokens. The first attempted
pin of 8,448 blocks incorrectly requested 2,162,688 tokens and OOMed; the
configurator and documentation now state the unit relationship explicitly.

The backbone reused its namespaced AOT artifact in 0.54 seconds. Model
preparation/loading took 545.4–672.2 seconds across warm NFS runs; graph
capture took 29 seconds and 0.20 GiB/GPU. After first-use 32K kernel
compilation, about 560–642 MiB/GPU remained.

The exact no-offload shape passed tokenization, thinking/no-thinking,
streaming usage, preserve-thinking multi-turn, strict JSON with and without
reasoning, automatic/required tool calls, duplicate suppression, and tool
result continuation. Its isolated temperature-one matrix measured:

| metric | 3.25 bpw | 3.0-bpw control | 3.25 delta |
|---|---:|---:|---:|
| uncached PP 32K | 2,407.9 tok/s | 2,479.9 | -2.9% |
| uncached PP 128K | 2,262.1 tok/s | 2,393.1 | -5.5% |
| aggregate TG C1 | 128.0 tok/s | 149.3 | -14.3% |
| aggregate TG C8 | 285.3 tok/s | 387.5 | -26.4% |
| MTP5 MAL C1 / C8 | 5.74 / 5.08 | 5.16 / 5.74 | workload-dependent |
| draft acceptance C1 / C8 | 94.8% / 81.5% | 83.2% / 94.8% | workload-dependent |

An actual 522,360-token prompt then recovered all five needles at 1%, 10%,
50%, 90%, and 99% with no degeneration in 291.55 seconds. The endpoint stayed
healthy and the log contained no engine error.

External prefix cache changes the memory result. LMCache over 50% host DRAM
passed the short feature suite and booted with the same 524,288-token pool,
but its first 32K prefill OOMed when PCIe all-reduce requested another 36 MiB.
The higher-fidelity profile therefore uses `OFFLOAD_FRACTION=0`; the default
3.0-bpw flagship retains qualified LMCache DRAM/NVMe support.

The independent dynamic-NVFP4 KLD run reported **0.09270766841426936** over
2,047 positions. The same harness reported 0.1167701184931591 for the
3.0-bpw control; the quant author's published dynamic values were 0.095971
and 0.119525 respectively. The reference bundle has one window, so the
standard deviation is structurally zero (`n=1`) rather than evidence of zero
model variance.

Power sampled once per second across all four 280 W-capped cards:

| phase | mean/GPU | four-GPU mean | peak single GPU | mean utilization |
|---|---:|---:|---:|---:|
| 32K + 128K uncached prefill | 271.6 W | 1,086.5 W | 289.0 W | 97.6% |
| C1 + C8 decode | 251.9 W | 1,007.6 W | 288.9 W | 97.4% |
| exact 522,360-token retrieval | 272.5 W | 1,090.1 W | 288.9 W | 98.6% |

Credential-free KLD, power, feature, benchmark, needle, and failure artifacts
are retained under `/mnt/fast/build/r11-qualification/`.

### GG v20-r14 + vLLM #210/#219 production gate (2026-08-01)

The source-exact turnkey image composes #210's bounded/sliced EXL3 arena and
#219's shape-aware mixed-K executor. The latter recovered pristine r14's
prefill regression: before slicing, 3K/32K/128K PP was 2,283.2/2,146.6/
2,033.1 tok/s, 28.4-29.3% above pristine r14 and within 1.2-1.7% of the
matched r13 control.

The selected 1,024-row arena returned about 665 MiB/GPU. A 1,536-row arm
returned only about 332 MiB/GPU but paid the same roughly 10-11% PP cost, so
it was rejected. On the final immutable image, unique-prefix cold PP was
2,121.3/1,931.9/1,837.2 tok/s at 3K/32K/128K. Aggregate MTP5 TG was
93.0/126.4/161.8/240.5 tok/s at C1/C2/C4/C8, with MAL
3.31/3.17/3.11/3.70 and no failure or preemption.

The final shape retained exactly 524,288 active GPU-KV tokens plus 125 GiB
aggregate LMCache DRAM and a bounded 512 GiB NVMe tier. It retrieved 3/3
needles from an actual 509,022-token prompt without degeneration, exposed
both `GLM-5.2` and `local-primary`, and completed with no exception or CUDA
OOM. Full provenance and the reason for keeping MTP3 in the shipping profile
while the AIBeast control used MTP5 are recorded in
[`docs/glm52-3.25-offload-qualification.md`](docs/glm52-3.25-offload-qualification.md).

### GG v20-r17 native mixed-K production gate (2026-08-01)

The immutable r17 base replaced the superseded #210/#219 overlays with native
vLLM #222 and included SparkInfer #105's complete custom-PCIe wheel. A matched
production-shape r14/r17 comparison retained identical 82.81 GiB weights,
1.49 GiB activation peak, 0.54 GiB non-Torch allocation, 0.20 GiB graphs, and
the fixed 524,288-token KV pool. PP geometric mean changed by -1.4%, within the
matched gate, while the runtime selected `B12X_PCIE_ONESHOT_DMA` rather than
the r17-candidate packaging fallback.

The AIBeast route campaign selected a 1 MiB NCCL buffer, 6 MiB lossless-DMA
crossover, DCP A2A cap 48, and source-default owner merge 0 for the explicit
PP-first agent workload. The resulting cold PP was 2,230/2,052/1,950 tok/s at
3K/32K/128K. It passed all API/tool/structured-output checks, exposed both
served aliases, recovered 5/5 needles from exact 261,195- and 521,276-token
prompts, and showed no degeneration, preemption, OOM, CUDA, NCCL, structured-
FSM, or process error. A clean production restart then repeated the exact
521,276-token matrix with a new seed in 312.9 seconds; all 27 post-ready kernel
lookups were persistent disk-cache hits, including the twelve long-row
`m=120` artifacts, and none recompiled. Full candidate tables, rejected
microbenchmark choices, power/memory boundaries, and raw-artifact locations are recorded in
[`docs/glm52-r17-maintenance-results.md`](docs/glm52-r17-maintenance-results.md).

### GG v20-r25 / SparkInfer #117 production gate (2026-08-03)

The immutable r25 image and exact SparkInfer #117 head were verified by hash.
The real 3.36-bpw layer shapes—206 K3 + 50 K4 at layer 3 and 160 K3 + 96 K4
at layers 4–77—loaded in one process and completed CUDA graph capture through
32. The selected TP4/DCP4, online-K6, native-MTP3, dynamic-NVFP4 profile kept
exactly 524,288 GPU-KV tokens plus 125 GiB LMCache DRAM and bounded 512 GiB
NVMe tiers.

Cold unique-prefix PP was 2,362.8/2,284.9/2,143.8 tok/s at 3K/32K/128K.
Aggregate TG was 100.7/162.2/240.1/297.0 tok/s at C1/C2/C4/C8 with MAL
3.13/3.43/3.41/3.02 and zero request failures or preemptions. Strict JSON with
thinking passed, and all five needles were recovered from an actual
521,275-token prompt without degeneration or OOM.

A matched FP8-KV comparator improved mean KLD from 0.0825070 to 0.0686692,
but only a 512-row arena was stable at a 512,000-token pool. That shape passed
5/5 needles at 509,010 actual tokens, yet lost 18–21% PP and 22.9% C8
throughput and retained only tens of MiB of post-benchmark headroom. It was
not promoted. Full configuration, memory, and precision evidence is in
[`docs/glm52-r25-3.36-qualification.md`](docs/glm52-r25-3.36-qualification.md).

### GG v20-r26 TP4/DCP4 policy gate (AIBeast, 2026-08-04, current)

The immutable r26 image was tested first as a matched official-image A/B and
then through the complete turnkey overlay. On the 3.36-bpw/K6/MTP3 shape, r26
improved old-auto r25 PP by 10.5/12.2/13.8% at 8K/64K/128K. Its second exact
indexer shard reduced the auto-sized logical pool from 771,584 to 717,312
tokens; both remain comfortably above the product's 524,288-token gate.

The appliance had already disabled owner merge, so its production-shape gain
was smaller and repeatable: 2,453--2,458 / 2,350--2,370 /
2,197--2,238 tok/s at 3K/32K/128K, about 2.5--4.4% over r25. The profile kept
exactly 524,288 active tokens, passed the full chat/thinking/streaming/
structured-output/tool suite, and retrieved all five needles from an actual
522,359-token request in 295.96 seconds. There were no request failures,
preemptions, degeneration, CUDA errors, or OOMs.

Qualification found and fixed one appliance integration issue: the PCIe
calibration subprocess explicitly requested automatic query split but retained
hard-coded owner and indexer values, bypassing r26's two-shard policy. All
three policy inputs now resolve through r26's helper, and the 3.36-bpw profile
keeps its measured 24 MiB lossless DMA crossover. The cold r26 namespace did
compile first-use shapes after `/health`; the subsequent feature run had zero
post-ready compiles and zero XGrammar FSM/runtime findings. Full results are in
[`docs/glm52-r26-3.36-qualification.md`](docs/glm52-r26-3.36-qualification.md).

## GLM-5.2 design decisions

### InstantTensor loader selection

The first exact 524,288-token InstantTensor launch did not merely flap once:
two consecutive engine attempts loaded successfully, then both failed KV
admission. InstantTensor reported 77.70 GiB/GPU resident model memory versus
77.66 GiB/GPU for safetensors, leaving 9.03 GiB KV when 9.04 GiB was required.
The supervisor's retry behavior was treated as failed startup, not eventual
success.

With `MAX_MODEL_LEN=520192` and the same 0.978 utilization:

- five launches booted on their first engine attempt;
- every built-in 32K gate retrieved 3/3 codes;
- cached starts reached a verified verdict in 132.8 seconds;
- target+draft load took 32.4–33.1 seconds, versus 60.5–62.6 seconds for
  safetensors;
- the required feature suite passed, including thinking visibility,
  streaming usage, preserved-thinking multi-turn, one automatic tool call,
  and tool-result continuation;
- two independent ~517K prompts recovered 5/5 needles at 1%, 15%, 50%, 90%,
  and 99% in 233.7–237.4 seconds with no degeneration.

The benchmark harness gained `--prompt-seed` after an unseeded comparison
showed that per-run UUID text could materially change speculative acceptance.
The corrected identical-prompt matrix was:

| loader | PP 8K / 32K | TG C1 / C2 / C4 / C8 | MAL C1 / C2 / C4 / C8 |
|---|---:|---:|---:|
| safetensors | 2,794.8 / 2,680.2 | 170.9 / 227.7 / 318.7 / 409.1 | 5.82 / 5.82 / 5.83 / 5.70 |
| InstantTensor | 2,782.4 / 2,680.4 | 168.7 / 230.6 / 306.2 / 402.6 | 5.71 / 5.98 / 5.79 / 5.62 |

All 64 decode requests completed with zero failure or preemption. The mixed
throughput deltas (+1.3% to -3.9%) do not support a steady-state loader
regression.

A later exact near-maximum challenge changed the default decision. Safetensors
failed three times at 514,432 computed tokens when the sparse indexer requested
a contiguous 352 MiB allocation: standard verification with an auto 530,304
pool, block verification with that same pool, and standard verification with
an exact 520,192-token pool. InstantTensor completed the same request on the
current image with 5/5 retrieval and no degeneration. The balanced profile
therefore defaults to InstantTensor plus `MAX_MODEL_LEN=520192`; safetensors
remains selectable but is not qualified for this profile's near-max envelope.

Artifacts:

- `instanttensor-currentimage-nearmax-five-depth.json`
- `instanttensor-currentimage-final.log`
- `standard-auto-blocks-nearmax-oom.log`
- `safetensors-4064blocks-nearmax-oom.log`
- `block-verification-full.log`

### Standard vs block speculative verification

The two verification methods were tested on the same InstantTensor/DCP2/MTP5
control with temperature 1, identical seeded prompts, and two repetitions per
concurrency. Values below are the two-run means:

| verifier | C1 TG / MAL | C2 TG / MAL | C4 TG / MAL | C8 TG / MAL |
|---|---:|---:|---:|---:|
| standard | 95.48 / 2.936 | 134.56 / 3.253 | 187.46 / 3.545 | 194.95 / 2.964 |
| block | 109.23 / 3.494 | 124.43 / 2.991 | 167.04 / 3.162 | 197.90 / 3.135 |
| block TG delta | +14.4% | -7.5% | -10.9% | +1.5% |

Block verification also completed the exact 517,178-token five-depth retrieval
with 5/5 needles, no degeneration, and output identical to the standard arm.
The long-context failure seen in a separate block run reproduced under
standard verification and was isolated to safetensors fragmentation, not the
verifier. Because the throughput result reverses sign across the supported
concurrency range, `standard` remains the production default and `block`
remains an explicit experiment.

Additional artifacts:

- `block-verification-standard-run1.json`
- `block-verification-standard-run2.json`
- `block-verification-block-run1.json`
- `block-verification-block-run2.json`
- `instanttensor-block-nearmax-five-depth.json`

### 3,072 vs 4,096 batched-token admission

The balanced profile's 3,072-token batch was challenged with a single-variable
4,096 cold boot at the same DCP2, MTP5, 520,192 maximum, GMU 0.978, graph
window, KV dtype and offload settings. The larger shape compiled under a
distinct cache key, then failed deterministic KV admission:

| batch | model/runtime memory | available KV | 520,192 admission |
|---:|---:|---:|---|
| 3,072 | 77.66 GiB/GPU | 9.15 GiB / 530,304 tokens | pass |
| 4,096 | 77.83 GiB/GPU | 7.57 GiB / ~438,656 tokens | fail; 8.97 GiB required |

The 4,096 Trellis arena was 1,374.2 MiB. Because this arm could not meet the
512K–520K context requirement, no PP/TG benchmark was used to rationalize it.
The profile retains 3,072.

### DRAM prefix-cache eviction and reuse

DRAM offload was tested as a second-level prefix cache, not as a way to enlarge
the active GPU KV pool. Source inspection found that native vLLM's
`cpu_bytes_to_use` is already an aggregate TP-world budget: dividing it by TP
in the appliance made a requested 125 GiB tier only ~31 GiB in practice. That
version stored 71.7 GB across the test stream but had evicted the original
prefix from CPU by the final request, which recomputed in 58.48 seconds.

After passing the full 125 GiB aggregate value, the same TP4/DCP2 service used
`OFFLOAD_FRACTION=0.5` on a 251 GiB host and completed this matrix:

| phase | prompt | TTFT | GPU hit | external DRAM hit | transfer |
|---|---:|---:|---:|---:|---:|
| target, cold | 133,731 tokens | 52.47s | 0 | 0 | stored 9.90 GB |
| target, immediate repeat | 133,731 | 0.59s | 133,504 | 0 | none |
| five unique eviction prompts | ~133,734 each | 54.43–58.03s | 0 | 0 | stored 9.90 GB each |
| target, after GPU eviction | 133,731 | **0.69s** | **0** | **133,504** | **9.89 GB CPU→GPU** |

The final load completed across four workers in 0.87 seconds aggregate with no
allocation failures, making DRAM reuse about 76x faster than recomputing this
prefix. Host memory stayed flat at roughly 199 GiB used and 51 GiB available.
That makes 50% a qualified setting for large reusable agentic prefixes on this
256 GiB host. A 70% tier was not attempted because it would consume almost all
of the observed operating margin; a larger host may qualify it separately.

The warning that native P2P atomics are unavailable was present in both v19
and v20. Peer reads/writes pass, but PyTorch symmetric-memory barriers require
system-scope atomic CAS and disable that one/two-shot communicator. Separate
v20 logs confirm B12X PCIe fused all-reduce and B12X DCP collectives are active,
so the warning is not an all-P2P fallback.

### MadeBy561 native MTP identification

The selected MadeBy561 path was initially described as a "stock BF16 MTP
draft." Direct inspection of the exact revision disproved that label. Layer
78 expert weights are serialized `U8` with FP8 E4M3 `weight_scale` plus FP32
secondary/input scales. The v20 `nvfp4_nf3_hybrid` loader treats an MTP layer
outside the target hybrid bit-map as uniform NVFP4; eligible dense BF16 pieces
receive the checkpoint's MXFP8 load-time overlay. In other words, this is the
checkpoint-native optimized NVFP4/MXFP8 path, not a BF16 draft.

An apples-to-apples substitution with a standalone draft assembled from
`lukealonso/GLM-5.2-NVFP4`'s MTP-only shards booted successfully with the B12X
ModelOpt FP4 draft backend, but regressed on the same AIBeast shape:

| MadeBy561 MTP path | model load/GPU | auto-profile blocks | prefill @8K / 32K / 66K | aggregate decode / MAL |
|---|---:|---:|---|---|
| **native checkpoint path** | **83.97 GiB** | 2,720 | **2,701 / 2,176 / 1,987 tok/s** | **C1 121.6 / 3.941; C2 142.3 / 3.657; C4 208.4 / 3.954; C8 269.7 / 3.913** |
| Luke external ModelOpt NVFP4 | 84.09 GiB | 2,817 | 1,968 / 1,905 / 1,868 tok/s | C1 87.1 / 3.536; C2 102.7 / 3.276; C4 161.1 / 3.616; C8 174.1 / 3.662 |

Both runs pinned the served pool to 2,048 blocks; the auto-profile values are
reported only to expose the loader/workspace difference. Luke provided 97
additional profiled blocks (24,832 logical tokens) but slightly increased
resident model memory and lost 28–35% of aggregate decode throughput. It is
therefore supported as an advanced external-draft experiment, not a
recommended MadeBy561 profile. The draft-memory savings previously measured
for Luke apply when replacing Brandon's genuinely BF16 MTP layer; they do not
describe the already-quantized MadeBy561 native path.

### Structured output + MTP compatibility

Oh My Pi traffic exposed repeated XGrammar
`Failed to advance FSM ... Please file an issue` messages during structured
requests. Source inspection confirmed GG r5 already includes the substantive
reasoning-boundary work from vLLM #44993. In the remaining path, draft tokens
after a reasoning-end marker were proposed before the JSON bitmask existed;
the manager deliberately tolerated their rejection and the verifier resampled
them, but `accept_tokens` logged the expected speculative rejection as an
engine ERROR before returning `False`.

The unmodified AIBeast control was exercised without restarting or modifying
production:

- 8/8 concurrent strict-schema requests passed with thinking disabled;
- 8/8 passed with thinking enabled;
- 8/8 passed with the thinking option omitted;
- the updated full feature suite passed both strict-JSON modes, including an
  exact `{"answer":42}` document and a populated reasoning field;
- the engine returned HTTP 200, remained healthy with zero restarts, and still
  emitted the diagnostic FSM errors during the thinking request.

This establishes the current messages as false-severity diagnostics for the
observed Chat Completions path, not evidence that Oh My Pi received malformed
JSON. It does not generalize to every FSM error: upstream reports include
genuine HTTP 500 and request-termination variants, so committed-token failures
must remain fatal.

The candidate entrypoint now inspects the already-filled packed grammar
bitmask for only those post-reasoning speculative probes. It accepts an
allowed probe temporarily so later masks in the same MTP window remain exact,
skips the noisy backend call for expected-invalid speculation, and leaves the
original accept/assert path intact for committed tokens. A prior validator-
based version was made automatically upgradeable after log review showed that
even a non-mutating matcher probe could produce a native post-EOS warning.

The immutable candidate image
`ghcr.io/malaiwah/glm52-exl3-vast:b806ee6fb8700f43068cd87558f72d09fd93bc9e`
then ran on Vast instance `46116148`, one RTX 5090 with driver 595.71.05,
CUDA 13.2 compatibility and a 575 W limit at `$0.637/hour`. The image pulled
in about four minutes and the 20.4 GiB pinned Qwen checkpoint downloaded in
about 20 seconds on this unusually well-connected host. After the first log
review, the rental source was upgraded in place from the image's validator
preflight to the final bitmask implementation; the committed migration handles
both source shapes idempotently.

The rental exposed a separate Qwen MTP qualification bug before the grammar
test: GG v20-r5's compiled FlashInfer wrapper was planned with
`q_len_per_req=1`, while MTP2 needed `3`. The first verifier request killed the
engine, and the supervisor retry reproduced the same fatal error. With the
same checkpoint, source patch and MTP2 configuration in `--enforce-eager`
mode:

- the final bitmask implementation passed 4/4 concurrent strict-schema
  requests with thinking disabled, 4/4 with thinking enabled and 4/4 with the
  option omitted;
- the full required feature suite passed tokenization, both chat modes,
  streaming usage, preserved multi-turn reasoning, both strict-JSON modes,
  one automatic tool call and tool-result continuation;
- all structured documents were exact `{"answer":42}` values, thinking
  requests populated the reasoning field, and the engine recorded zero FSM
  ERRORs, zero HTTP 500s and zero fatal errors;
- MTP2 reported roughly 2.5–2.7 mean acceptance length and about 93–101
  aggregate generation tok/s during four-way decode windows.

XGrammar still printed a native “matcher has terminated ... token 198” warning
after some speculative stop tokens. It is distinct from the removed vLLM FSM
ERROR path: outputs remained exact and the engine remained healthy. The
compiled, MTP-off default was therefore tested separately. Its first uncached
shape compile reached health in about three minutes, the complete feature
suite passed, and its log had zero FSM errors, zero post-EOS matcher warnings,
zero HTTP 500s and zero fatal errors. The Qwen profile now keeps that fast
compiled mode as the default and automatically adds `--enforce-eager` only
when `MTP_TOKENS>0`, with an explicit configurator warning.

Qwen occasionally used 820 completion tokens before crossing the reasoning
boundary on the tiny strict-schema probe. The verifier budget was raised from
768 to 1,024 so it measures schema correctness rather than truncation. Prompt
wording also mattered: “at most one short sentence” could remain in thinking
for the full 2,048-token control, whereas the simpler “Think briefly, then”
crossed the boundary and returned exact JSON.

Both thinking and non-thinking strict JSON remain required release gates.
Local patch/idempotence and legacy-upgrade tests, unknown-runtime behavior,
verifier, feature-suite, Python and shell tests passed. Per the operator's
instruction, the patch was **not** loaded into or restarted on AIBeast; its
production endpoint remained online throughout.

### Startup timing

Authenticated Hugging Face Xet downloaded the 341 GiB checkpoint on Vast in
3m45s. On AIBeast NFS plus `cachefilesd`, target shards loaded in 128–142s and
the complete model reported loaded in 161–179s. The first turnkey
configuration reached health in 9m43s and was verified in 10m44s; with AOT
caches populated, the ring configuration reached health in 6m31s and was
verified in 6m49s. With the ring-specific CuTe kernels cached too, the final
repeat was verified in exactly 5m00s. Its 32K gate took 17–19s.

## Local and browser checks

Passed:

- ShellCheck, Actionlint, Python bytecode compilation, JSON parsing, and
  whitespace/error checks.
- Profile-only resolution for GLM-5.2, Qwen3.6-27B NVFP4, and a custom model.
- Secure/insecure landing-page render conditions and JavaScript syntax.
- Desktop and 390-pixel mobile browser rendering with no horizontal overflow.
- Token-gated dashboard behavior.
- Two-turn mocked streaming chat.
- GLM `reasoning_content` and current vLLM/Qwen `reasoning` deltas.
- Reasoning-only Qwen output becomes a visible, non-empty assistant turn.
- `Preserve thinking` defaults off, can be enabled, and does not duplicate a
  reasoning-only response in later context.
- Clear resets both the visible conversation and request history.

## Artifacts

- Branch: `codex/appliance-live-test`
- Final runtime-image commit:
  `09c14e07f5529d18380572830e2b4e47bb16cc49`
- Final immutable image:
  `ghcr.io/malaiwah/glm52-exl3-vast:09c14e07f5529d18380572830e2b4e47bb16cc49`
- Final image digest:
  `sha256:9d4114dae30953e19eaf2e7af221f2259e0277147c9257764091de479582a528`
- Registry payload remains approximately 11.6 GiB compressed and 30 GB
  unpacked; the follow-up adds the small `socat` TLS-forwarder package.
- Final build:
  <https://github.com/malaiwah/glm52-exl3-vast/actions/runs/30225872241>

The final build passed lint and image publication with SHA-pinned Node 24
actions, least-privilege permissions, per-ref concurrency, timeouts, maximum
provenance, and an SBOM. Manual dispatch published the immutable SHA tag and
did not replace `latest`.

## Coverage summary

| Area | Result |
|---|---|
| Build, profiles, manifests, local UI | Passed |
| Vast provider integration and baseline appliance features | Passed |
| deSEC API lifecycle, DNS-01 cleanup, and trusted direct TLS | Passed on Vast |
| Qwen reasoning/tool parser with small live model | Passed on Vast |
| UI reasoning compatibility and multi-turn behavior | Fixed and browser-tested |
| Runpod template/API schema and placement handling | Passed |
| Runpod hybrid proxy/direct-TLS runtime compatibility | Passed on Secure RTX 5090 |
| Authenticated HF Xet cold download (Qwen 27B) | Passed; 20.435 GiB in 18m14s |
| Small-model native vision and MTP | Passed on Vast RTX 5090 |
| GLM TP4/DCP4, EXL3/MTP78, 512K, production vision/offload | Explicit release qualification gap |
| Final provider resources | Vast: 0; Runpod: 0 |

## Next economical live pass

Prefer a Runpod Secure Blackwell machine known to cache the pinned image; the
validated host took roughly nine minutes merely to expose mappings despite
advertising multi-gigabit networking. The authenticated Qwen download also
showed that provider bandwidth does not predict the host-to-Hugging-Face CAS
route. Reuse the 0.8B profile and execute only the still-uncovered
restart-persistence, vision, MTP, and full-profile qualification rows from
`TEST_PLAN.md`.
