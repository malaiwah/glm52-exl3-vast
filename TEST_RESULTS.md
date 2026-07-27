# Appliance test results — 2026-07-26

This report records the cost-controlled execution of `TEST_PLAN.md`. Provider
credentials and generated appliance tokens were kept in process-local
environment variables and are not included here.

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

## Vast.ai live execution

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

### deSEC dynamic DNS and direct TLS follow-up

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

## Runpod live execution

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

### Authenticated Hugging Face Xet throughput follow-up

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

### Remaining economical qualification follow-up

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

## GLM-5.2 flagship qualification — 2026-07-27

The production-scale pass uses Vast instance `45997603`, four RTX PRO 6000
Blackwell 96 GB GPUs, and an 850 GB disk. The rental is useful for
correctness, memory-fault reproduction, and same-host A/B tests, but not as an
absolute performance reference:

- GPU 0 reaches the other three GPUs through `SYS`; GPUs 1–3 are `NODE` and
  GPUs 2–3 are `PIX`.
- The machine has two CPU sockets and four NUMA nodes.
- CUDA peer reads/writes work, while native peer atomics do not.
- The provider charged `$7.248/hour`, including the enlarged disk.

Both checkpoints fit simultaneously: approximately 303 GB for EXL3 and
341 GB for `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. The authenticated
MadeBy561 Hugging Face Xet transfer completed in 3 minutes 45 seconds. Local
safetensor reads take about 34–36 seconds, but full engine restarts take
roughly 4–7 minutes after compilation, warmup, memory profiling, and CUDA
graph capture. The appliance's 32K retrieval gate adds about four minutes on
this topology.

### Read-only AIBeast control

The owned-production control is the same MadeBy561 checkpoint at revision
`68babde27a97a4c980c2494e830dd424975cd5a3` on the v19 image. Its four GPU
paths are all `NODE` on one NUMA node. The production launch uses
TP4/DCP4/MTP3, probabilistic proposals, a 3,072-token prefill chunk,
`nvfp4_ds_mla` KV, a 537,600-token pool, and 128 GiB of host KV offload.

Trusted isolated measurements recorded in its handoff are 2,299 prompt
tokens/s at 8K, 2,192 at 64K, and 119.2 output tokens/s at C1. Production logs
observed during ordinary traffic showed roughly 92–96 output tokens/s and mean
speculative acceptance length around 2.94–3.12. Prior seeded retrieval was
clean at 490K and 505K. The endpoint was inspected read-only and was not
restarted or benchmarked while serving the owner.

### v20 MadeBy561 memory search

Every candidate used TP4/DCP4, MTP3, the stock BF16 draft, calibrated
`nvfp4_ds_mla` KV, synchronous scheduling, B12X MLA/MoE, and a maximum request
length of 524,288. The v20 topology calibrator selected lossless PCIe DMA at a
393,216-byte crossover and disabled CKV prefetch overlap on this cross-socket
host; its microbenchmark found DMA about 61–63% faster than NCCL above the
crossover.

| candidate | result |
|---|---|
| auto pool, GMU 0.96, batch 3,072 | startup KV admission failure |
| auto pool 551,680 tokens, GMU 0.98, batch 3,072 | first 32K request OOM; only 24.75 MiB free for a 36 MiB NF3 target allocation |
| pinned 524,288-token pool, batch 2,048, workspace 1,024 MiB | 32K 3/3 and full feature suite passed |
| same pin, batch 3,072, workspace 1,024 MiB | one 32K pass, then later 32K/benchmark OOM |
| same pin, batch 3,072, workspace 512 MiB | three uncached 32K passes and C1–C8 passed, but a 520,192-token request immediately OOMed |
| same pin, batch 2,048, workspace 512 MiB | selected maximum-context candidate |

The failure shape is important. Startup admission and one short request are
not sufficient evidence for this target: the NF3/MTP transient allocation is
not fully represented by the apparent KV headroom. The configurator therefore
supplies the pool, chunk, workspace, utilization, stock draft, and proposal
method as one `madeby561-hybrid` variant default while retaining explicit
per-knob overrides.

The selected candidate's required feature suite passes authenticated discovery,
exact tokenization, ordinary and thinking chat, SSE usage, multi-turn with
preserved reasoning, one automatic tool call, and tool-result continuation.
Structured JSON also passed but remains informational. Forced
`tool_choice=required` emitted five duplicate calls on this build; automatic
tool choice emitted exactly one, so normal agentic workloads remain a release
gate while forced mode does not.

### Same-host performance

The mixed-topology host is slow in absolute terms. Its purpose here is to
compare configurations without changing hardware:

| target / candidate | unique prefill | aggregate decode |
|---|---|---|
| EXL3, MTP3, batch 3,072 | 286 tok/s @1K; 379 @8K | C1 18.9, C2 26.3, C4 29.8, C8 28.0 |
| MadeBy561, batch 2,048, workspace 1,024 | 271 @1K; 392 @8K; 141 @32K | C1 12.4, C2 22.7, C4 32.0, C8 25.6 |
| MadeBy561, batch 3,072, workspace 512, greedy | about 135 tok/s sustained long prefill | C1 14.5, C2 27.4, C4 31.2, C8 29.9 |
| MadeBy561, batch 3,072, workspace 512, probabilistic | same prefill arm | C1 17.5, C2 23.0, C4 19.9, C8 29.6 |
| MadeBy561, MTP off, batch 6,144 | — | C1 16.5, C2 21.5, C4 25.2, C8 36.6 |

Periodic vLLM logger values were not used as these throughput measurements.
Its default ten-second logger increments prompt tokens only when a scheduled
chunk completes, then resets the interval. A 2,048-token chunk therefore
prints `204.8`, `0`, `204.8`, `0` when successive chunks straddle alternating
buckets, and `409.6` when two land together. Exact unique prompt tokens divided
by end-to-end prefill time are the comparable metric.
