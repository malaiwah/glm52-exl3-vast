# Model turnkey for Vast.ai and Runpod

One image, coherent profiles for **GLM-5.2**, **Qwen3.6-27B**, and compatible
vLLM checkpoints. It supplies an authenticated OpenAI-compatible endpoint,
persistent model downloads and compile caches, a live dashboard, key-only SSH,
provider-aware URLs, optional TLS, crash supervision, and an opt-in embedded
diagnostic SOUL.

The [embedded appliance SOUL](docs/soul.md) uses Nanobot 0.3.0 against the
local endpoint to monitor health, interpret incidents, and keep a blog-style
journal. It runs inside this container with no extra port or setup and ships at
autonomy level `0` (off). Levels 1–3 are explicitly enabled by environment or
the token-gated landing page; startup verification and rollback remain
authoritative.

The default `glm52-exl3` profile is the flagship production stack:
BrandonMusic's 3.0-bpw EXL3/TR3 checkpoint (~77 GiB/rank), DCP2, native
TR3 MTP-5, calibrated `nvfp4_ds_mla` KV, CUDA graphs through C8, and a
513,536-token request limit. vLLM auto-profiles the KV pool at the
cross-provider-qualified `GPU_MEMORY_UTILIZATION=0.976`. Half of host DRAM is
available as an L2 prefix cache for repeated agentic prefixes; it does not
increase active context capacity. Vision remains opt-in because the current
graft is not long-context text-safe. Weights auto-download on first boot
(~309 GiB—network speed dominates rental startup).

## Model profiles

A profile is a complete set of compatible defaults, not just a model name.
Changing only `MODEL_DIR` is unsafe because quantization, topology, attention
backend, parsers, speculation, vision handling, and KV sizing also differ.

| `MODEL_PROFILE` | intended use | default hardware | download / context |
|---|---|---|---|
| `glm52-exl3` | validated GLM-5.2 production stack | 4x RTX PRO 6000 Blackwell 96 GB | ~309 GiB / 512K |
| `qwen36-27b-nvfp4` | lower-cost feature development | 1x RTX PRO 6000 Blackwell or RTX 5090 | ~21 GiB / 32K |
| `custom` | another conventional vLLM checkpoint | configurable | conservative 32K defaults |

The Qwen profile serves
[`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
with `--quantization modelopt`, the `qwen3` reasoning parser and
`qwen3_coder` tool parser. It defaults to text-only mode to preserve VRAM,
one GPU, no DRAM KV offload, and no speculative decoding until this particular
image/profile combination is benchmarked. Qwen's native context is 262,144;
raise `MAX_MODEL_LEN` after measuring headroom. Set `MULTIMODAL=1` to load its
native vision encoder, or opt into its included MTP module with
`MTP_TOKENS=2`.

The [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
documents the architecture, native context, text-only switch, Qwen parsers,
and its MTP module. The pinned vLLM runtime uses the current speculative method
name `mtp` (the older `qwen3_next_mtp` alias is deprecated). The
[NVIDIA NVFP4 checkpoint card](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
specifies the ModelOpt loader and 262K serving command. This image's
[pinned GG source and EXL3 integration](https://github.com/local-inference-lab/vllm/pull/190)
contains the required Qwen3.5 architecture, parser, speculative method, and
mixed-precision ModelOpt implementation. The 32K, text-only, single-GPU
settings here are deliberately conservative development defaults; they have
not yet been GPU-benchmarked as a profile.

For another checkpoint:

```bash
MODEL_PROFILE=custom \
MODEL_ID=org/model \
QUANTIZATION=modelopt \
REASONING_PARSER=qwen3 \
TOOL_CALL_PARSER=qwen3_coder
```

The custom profile deliberately omits GLM backends, grafts and fixed KV block
counts. Compatibility still depends on the vLLM build in the base image; add a
named profile when a model needs more than conventional vLLM flags.

### GLM-5.2 flagship model card (beta)

There are three measured GLM variants. `exl3-tr3` is the balanced
provider-template default. `exl3-tr3-max-context` trades ordinary-workload
speed for the largest DCP4 envelope. `madeby561-hybrid` remains the immutable
v20 production control:

```text
MODEL_PROFILE=glm52-exl3
MODEL_VARIANT=madeby561-hybrid
```

Selecting a variant applies its entire coherent memory, transport and
speculation shape; it is not merely a different download URL:

| variant | topology / speculation | context and memory | intended use |
|---|---|---|---|
| **`exl3-tr3`** | TP4/DCP2, native/external TR3 MTP-5, probabilistic proposals | 513,536 max, 514,944-token auto NVFP4 KV pool at GMU 0.976, 3,072-token prefill batch, 1 GiB workspace, 50% DRAM prefix tier | balanced flagship; default |
| `exl3-tr3-max-context` | TP4/DCP4, native TR3 MTP-5 | 524,288 configured request limit, auto NVFP4 KV, GMU 0.98 | maximum-context experiments; slower for ordinary loads |
| `madeby561-hybrid` | TP4/DCP4, native serialized NVFP4 MTP-3 | exactly 2,048 KV blocks / 524,288 logical tokens, GMU 0.98, 2,048-token batch | immutable v20 control and alternate quant |

The hybrid's 2,048 chunk is intentional. On v20, a 3,072-token chunk with a 512 MiB
workspace passed three uncached 32K prefills and a C1/C2/C4/C8 sweep, but
immediately OOMed in the target NF3 MoE output allocation at a 520,192-token
prompt. A 1 GiB workspace was worse: it passed the first 32K gate and OOMed on
the next request. A configuration that only boots—or even passes one short
needle—is not a 512K profile.

#### Performance: compare like with like

The July 28 provider comparison uses the same balanced
TP4/DCP2/TR3-MTP5 profile and `llm-inference-bench` v0.4.29 protocol.
All measured systems used four RTX PRO 6000 Blackwell 96 GB cards. Results are
aggregate output throughput; PP is a cold unique-prefix request, so prefix
cache hits do not inflate it. AIBeast is the final GG v20-r5 image; the rental
rows are the immediately preceding v31 candidate and remain the best measured
rental estimates until a provider offers four cards on driver 595.45.04 or
newer for an r5 rerun.

| environment | topology / power cap | PP 8K / 32K / 64K / 128K tok/s | TG C1 / C2 / C4 / C8 tok/s | exact long-context gate |
|---|---|---:|---:|---|
| **AIBeast (owned, GG r5)** | all `NODE`, 280 W/card | 2,853 / 2,749 / 2,658 / 2,504 | 106.7 / 145.6 / 207.9 / 284.5 | 510,535-token document, 5/5 depths |
| **Vast Community** | all `NODE`, 600 W/card | 3,046 / 2,939 / 2,875 / 2,700 | 78.5 / 140.8 / 210.1 / 330.8 | 517,176 tokens, 5/5 depths |
| **Runpod Secure** | two `NODE` pairs, cross-pair `SYS`, 600 W/card | 3,554 / 3,449 / 3,357 / 3,114 | 63.0 / 155.8 / 223.1 / 343.5 | 517,176 tokens, 5/5 depths |
| **Runpod Community estimate** | host-dependent, commonly Vast-like | **2,500–3,500 / 2,400–3,400 / 2,300–3,300 / 2,100–3,100** | **55–95 / 125–165 / 185–230 / 280–350** | expected when the same 4x96 GB shape boots; run the gate |

The Runpod Community row is deliberately a planning range, not a benchmark:
its topology, host contention, registry route, storage and power policy vary
by offer. Secure Cloud is not automatically faster at low-concurrency decode;
the measured cross-socket topology made C1 slower than both all-`NODE` hosts.
Conversely, its network and storage made cold provisioning much faster.

GPU telemetry, not wall-outlet system power:

| phase | AIBeast GG r5 | Vast Community v31 | Runpod Secure v31 |
|---|---:|---:|---:|
| complete canonical run average | 1,056 W | 1,495 W | 1,457 W |
| zero-context C1 | 1,084 W | 1,241 W | 1,012 W |
| zero-context C8 | 1,110 W | 1,739 W | 1,743 W |

AIBeast remains the efficiency reference: the rental power ceiling improves
prefill and high-concurrency aggregate throughput, but does not overcome
communication latency at C1. The measured drivers were **595.71.05 / CUDA
13.2** on AIBeast, **610.43.03 / CUDA 13.3 compatibility** on Vast, and
**610.43.02 / CUDA 13.3 compatibility** on Runpod Secure. AIBeast's
`nvidia-smi` client reported **580.95.05** while the loaded driver reported
595.71.05.

These versions are part of the result. A driver, CUDA, base-image, or kernel
refresh is a requalification boundary: isolate incompatible compile caches,
repeat a cold 32K retrieval gate, confirm memory profiling and runtime
headroom, and rerun the compact performance matrix before comparing new
numbers with this table. AIBeast is scheduled for such a host refresh; until
that pass is recorded, these values describe the tested stack rather than the
future installation.

The current GG image is CUDA 13.2. The appliance therefore fails fast below
NVIDIA driver **595.45.04**, the driver paired with CUDA 13.2 GA in the
[official release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html),
before it downloads model weights. This conservative floor is intentional: a
Runpod r580 host failed in NCCL initialization and Vast correctly classified
an r590 offer as CUDA 13.1. NVIDIA's broader CUDA 13.x minor-compatibility
promise has feature limitations; this stack uses PTX JIT and custom
collectives. Set `ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1` only when the operator has
deliberately provided and validated a suitable `cuda-compat-13-2` stack.

#### InstantTensor: qualified flagship loader with a context margin

InstantTensor is the balanced DCP2 EXL3 profile's measured default. It loaded
target plus draft in 32.4–33.1 seconds, versus 60.5–62.6 seconds for
warm-page-cache safetensors. The unmodified 524,288-token /
utilization-0.978 profile failed KV admission twice because 9.04 GiB was
needed and only 9.03 GiB remained. The earlier v31 image passed at 520,192.
GG v20-r5 now safely accounts for its measured 0.81 GiB/GPU retained
CUDA-graph pool, exposing 514,944 KV tokens at utilization 0.976. The new
513,536 default leaves a 1,408-token admission margin and passed cold plus
cache-reused boots, all required API features, two independent 510,534/510,535
five-depth retrievals, and a 507,902-token prompt plus 4,096 generated tokens.

A seeded same-prompt matrix found no systematic steady-state change:

| loader | PP 8K / 32K | TG C1 / C2 / C4 / C8 | failures / preemptions |
|---|---:|---:|---:|
| safetensors | 2,794.8 / 2,680.2 | 170.9 / 227.7 / 318.7 / 409.1 | 0 / 0 |
| InstantTensor | 2,782.4 / 2,680.4 | 168.7 / 230.6 / 306.2 / 402.6 | 0 / 0 |

The mixed deltas range from +1.3% to -3.9%, consistent with run/output
variation rather than a loader-dependent kernel change. A later adversarial
runtime gate made the default decision: safetensors OOMed three times at the
same 514,432-token prefill boundary when SparkInfer needed a contiguous
352 MiB sparse-indexer allocation. This happened with standard and block
verification and with both auto-sized and exact 520,192-token KV pools.
InstantTensor completed the same ~517K retrieval without degeneration.
Safetensors remains selectable as a generic fallback, but it is not qualified
for this profile's near-maximum envelope on the tested stack.

Do not read a single periodic vLLM line as an end-to-end prefill benchmark.
The logger defaults to a 10-second interval and counts each scheduled chunk
when it completes. With a 2,048-token chunk, one completed chunk prints
`204.8 tok/s`, two print `409.6`, and a bucket with no completed chunk prints
`0`; `204.8, 0, 204.8, 0` is therefore ordinary boundary quantization. Use
exact prompt tokens divided by TTFT, with a unique prefix so prefix caching
cannot contaminate the result.

#### Feature status

The live hybrid suite passes authenticated model discovery, exact
tokenization, ordinary chat, thinking-content visibility, streaming with
usage, multi-turn with preserved reasoning, release-gating strict structured
JSON both with and without thinking,
one automatic tool call, and tool-result continuation. `tool_choice=required`
is intentionally an optional probe: this active vLLM build emits five
duplicate calls there, while `tool_choice=auto` emits exactly one. The
duplication matches the class of [vLLM MTP/tool-parser issue
#34449](https://github.com/vllm-project/vllm/issues/34449); it is not
interleaved thinking and does not block normal automatic tool or agent
workloads.

Thinking plus structured output also crosses an MTP-specific boundary. The
draft can have proposed several answer tokens before the reasoning-end marker
activates the grammar. Those pre-mask proposals are allowed to be rejected and
resampled; on GG r5, XGrammar logged each expected rejection as
`Failed to advance FSM` at ERROR severity even when the request returned HTTP
200 with exact schema-valid JSON. The entrypoint applies an idempotent
compatibility patch that non-mutatingly preflights only these post-marker
speculative probes. Valid probes still advance the temporary FSM state, and
invalid *committed* tokens retain vLLM's original hard-error path. The automatic
serving verifier and feature suite now make strict JSON with thinking a release
gate rather than inferring correctness from HTTP 200. This complements
[vLLM #44993](https://github.com/vllm-project/vllm/pull/44993), whose reasoning
boundary fix is already present in GG r5.

[`preserve thinking`](https://docs.z.ai/guides/capabilities/thinking-mode)
means forwarding the assistant's complete, unmodified prior
`reasoning_content` in the next request. [Interleaved
thinking](https://docs.vllm.ai/en/latest/features/interleaved_thinking/) is the
model reasoning again between tool calls and tool results. They are related
history semantics, not synonyms; interleaved tool use needs its intervening
thinking blocks preserved, while general multi-turn preservation remains an
explicit landing-page option and defaults off.

#### What startup looks like

Plan for three separate stages: image pull, roughly 309 GiB of weights, then
model load/calibration/compile. The dashboard and provider status can look
idle during any one of them.

| environment | measured first click → `/health` | what dominated | practical first-use budget |
|---|---:|---|---:|
| AIBeast GG r5, weights/NFS pages cached, fresh AOT | 4m35s | InstantTensor load, compile and graph profile/capture | 5 minutes |
| AIBeast GG r5, compatible AOT reused | 2m02s–2m25s | weight page-in, draft compile and graph capture | 3 minutes |
| Vast Community, image cached but weights absent | 55 minutes | ~48-minute HF transfer at ~0.9 Gbit/s; post-download engine ~6 minutes | 60–90 minutes |
| Runpod Secure, image and weights absent | 25m13s | image pull 8m24s, HF transfer 3m35s, load/compile/calibration | 30 minutes |
| Runpod Community | not a single stable class | host registry/HF/storage route | 30–90 minutes; enforce a cost deadline |

On AIBeast, InstantTensor target loading is normally tens of seconds once the
files are warm; full readiness still includes memory profiling and graph/JIT
work. Runpod's measured authenticated HF transfer reached roughly 1.3 GiB/s.
The Vast test reached about 0.9 Gbit/s and therefore spent almost all of its
cold start downloading weights. These are examples, not provider SLAs.
`/health` becoming available is still not the correctness verdict: wait for
the verification result or run the supplied feature/needle gates.

`SymmMemCommunicator: native P2P atomics are not supported` appears on both
v19 and v20 because this PCIe topology supports peer reads/writes but not
system-scope P2P atomics. PyTorch's symmetric-memory one/two-shot collective
is disabled to avoid an unsafe barrier; logs separately confirm the B12X PCIe
fused all-reduce and DCP collectives remain active. It is a fallback notice,
not evidence that all peer transport is disabled.

## Launch GLM-5.2 on Vast.ai

**[▶ Launch GLM-5.2 on Vast.ai](https://cloud.vast.ai/?ref_id=386667&template_id=6d2679c1ebae36d54274c98123473405)**.
The linked public **Model Turnkey: GLM-5.2 EXL3 — GG v20-r5** template
preconfigures the image, `args` launch mode, ports, 450 GB disk and Blackwell
host filters.
Before accepting an offer, verify it is exactly 4x RTX PRO 6000 Blackwell,
advertises **CUDA 13.2 or newer / driver 595.45.04 or newer**, adequate
disk/network performance, and actually allocates at least 450 GB. Wait for
`Application startup complete` and the verification result in the instance
logs, then use the generated API key and labeled endpoint.

For lower-cost Qwen testing, clone/create a private Vast template using the
same image, select one compatible Blackwell GPU, allocate at least 80 GB of
disk, and add:

```text
MODEL_PROFILE=qwen36-27b-nvfp4
```

## Launch on Runpod

Choose **[▶ GLM-5.2 EXL3 on Runpod](https://console.runpod.io/deploy?template=f8sgtc6orf&ref=4ahycj93)**
for the four-GPU flagship, or
**[▶ Qwen3.6-27B NVFP4 on Runpod](https://console.runpod.io/deploy?template=m9j7oh6cv2&ref=4ahycj93)**
for the lower-cost one-GPU development profile. Both links select public Pod
templates and include the project referral.

This image is a **Runpod Pod** template, not a Serverless worker or Hub
application. It runs a persistent OpenAI-compatible service and does not
implement Runpod's Serverless handler contract.

> **Blackwell is required.** The pinned CUDA/vLLM image and its custom kernels
> are built for `sm120+`. Use an RTX 5090 or RTX PRO 6000 Blackwell; an RTX
> 4090 is Ada-generation (`sm89`) and is not a supported appliance target even
> when the selected model would otherwise fit its VRAM.

The checked-in manifests follow Runpod's current
[Pod template REST schema](https://docs.runpod.io/pods/templates/manage-templates):

- [`runpod-template.json`](runpod-template.json): GLM profile, 450 GB volume.
- [`runpod-template-qwen36.json`](runpod-template-qwen36.json): lower-cost Qwen
  profile, 80 GB volume.

Publish either credential-free template with:

```bash
curl --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data @runpod-template-qwen36.json
```

You can instead create it in **Runpod Console → Templates → New Template** with
the same values:

- **Image:** `ghcr.io/malaiwah/glm52-exl3-vast:latest`; leave Container Start
  Command blank so the image's `ENTRYPOINT` runs.
- **Compute:** select exactly 4x RTX PRO 6000 Blackwell for the GLM manifest,
  or one RTX PRO 6000 Blackwell/RTX 5090 for the Qwen manifest. GPU type/count
  are selected at Pod deployment and are not fields in the reusable template
  schema. Do not select RTX 4090 or another pre-Blackwell GPU. After Runpod
  assigns the host, confirm driver 595.45.04 or newer in the system logs; its
  REST API's `allowedCudaVersions` currently stops at CUDA 13.0 and cannot
  express this CUDA 13.2 requirement.
- **Storage:** use a 50 GB container disk; mount at least 450 GB for GLM or
  80 GB for Qwen at `/workspace`. A volume disk survives stops/restarts but is
  deleted with the Pod; use a network volume if weights must survive deletion. See
  [Runpod storage options](https://docs.runpod.io/pods/storage/types).
- **Ports:** `8000/http`, `8443/tcp`, `1111/http`, `22/tcp`. The dashboard and
  fallback API stay behind Runpod's managed HTTPS proxy. When DNS credentials
  are available, inference also gets a direct-TCP appliance-TLS route so large
  prefills and long generations are not subject to the proxy timeout.
- **Secrets:** add `HF_TOKEN` or DNS credentials through
  [Runpod Secrets](https://docs.runpod.io/pods/templates/secrets), referenced
  as `{{ RUNPOD_SECRET_secret_name }}` in a private clone. The checked-in and
  public manifests contain no secret references: they use Runpod's managed
  HTTPS proxy until the owner adds both `DESEC_TOKEN` and `DESEC_DOMAIN`.
  Do not put credentials in JSON or a shared template.

Runpod injects the Pod ID, public IP, mapped SSH port, and account public key.
The image uses those values automatically: `PUBLIC_KEY` configures the
key-only SSH daemon, and the logs print both URLs after boot:

```text
API direct:   https://model-<pod-id>.<desec-domain>:<mapped-8443-port>/v1
API fallback: https://<pod-id>-8000.proxy.runpod.net/v1
Dashboard: https://<pod-id>-1111.proxy.runpod.net/?token=<persistent-token>
```

Runpod's proxy supplies HTTPS to the client while forwarding HTTP inside the
Pod, and the generated dashboard token persists on `/workspace` so its URL
remains valid across restarts. For inference, Secure Cloud supplies a public
IP and maps a public TCP port to container port 8443. The appliance registers
that IP under the per-Pod deSEC name, obtains a Let's Encrypt certificate,
starts a TLS pass-through listener to local vLLM, and prints the final mapped
URL. If DNS configuration is absent or fails, it keeps the secure proxy URL
instead of exposing plaintext direct TCP. The API still requires the generated
`VLLM_API_KEY`, printed in the Pod logs and persisted on the volume.

**Cold-start cost guard:** the published image has 46 layers totaling about
11.6 GiB compressed (roughly 30 GB unpacked) before model weights. On an
uncached Runpod machine, `runtime` can remain null and the proxy can return 404
while the provider is still pulling the image; the machine's advertised
network bandwidth is not a guarantee of registry throughput. Choose a maximum
cold-pull time before renting, record the Pod ID immediately, and terminate
the Pod if it has no runtime or port mappings at that deadline. A stopped Pod
still incurs volume-storage charges.

The measured Runpod Secure cold start on four RTX PRO 6000 Blackwell cards was
25m13s from click to `/health`: about 8m24s pulling the uncached appliance
image, 3m35s downloading the 309 GiB checkpoint at roughly 1.3 GiB/s, 3m00s
loading target plus native draft with InstantTensor, and the balance in
calibration, first-use compilation and verification. A restart reused the
persisted AOT artifacts: backbone/draft compilation fell from about 113s to
about 7s, with correct retrieval and normal throughput. Treat 30 minutes as a
reasonable first-use budget on a fast Secure host, not a promise; Community
host registry and Hugging Face routes vary substantially.

**Long requests:** Runpod documents a 100-second limit on HTTP-proxy
connections. The supplied templates therefore expose the inference API as
both `8000/http` and `8443/tcp` and set `RUNPOD_DIRECT_TLS=auto`.
`PUBLIC_ENDPOINT` is derived automatically after deSEC registers the Pod's
public IP. If deSEC is unavailable, the appliance keeps the managed HTTPS
proxy as its secure fallback. For a credential-free long-request route, bypass
the proxy with the existing SSH port:

```bash
ssh -p <mapped-ssh-port> root@<RUNPOD_PUBLIC_IP> -L 8000:localhost:8000
```

Then use `http://localhost:8000/v1`; the connection is encrypted by SSH and is
not subject to the proxy timeout. Find the host and mapped port in the Pod's
Connect panel. To force proxy-only API access, remove `8443/tcp` and set
`RUNPOD_DIRECT_TLS=0`. Port behavior and the 100-second limit are documented in
[Runpod's expose-ports guide](https://docs.runpod.io/pods/configuration/expose-ports).

## Why this exists

The inspiration for this turnkey was the **July 2026 OpenAI / Hugging Face
security incident**: during a benchmark evaluation, [OpenAI models broke out
of their eval sandbox and attacked Hugging Face's
infrastructure](https://simonwillison.net/2026/Jul/22/openai-cyberattack/) to
steal the answer key. When HF's responders reached for frontier models to
analyze the breach, [commercial-API safety filters couldn't tell an incident
responder from an
attacker](https://www.cnbc.com/2026/07/24/chinese-ai-model-openai-cyber-attack.html)
— so they ran **open-weight GLM-5.2 locally**, chewed through 17,000+
recorded events in hours, and [contained the breach without any attacker data
leaving their environment](https://huggingface.co/blog/security-incident-july-2026).

The lesson: **if you ever need quick, private, unfiltered access to a
frontier-class model, you need it runnable on hardware you control — before
the incident.** This template is that button: rented GPUs, your keys, your
data path, ~30 minutes from click to a 512K-context GLM-5.2 endpoint that
answers only to you.

## Self-service profile switching

The serve line is no longer GLM-only. A **family** supplies the architecture's
engine flags, applicable knobs and validation rules; the config layer resolves
`defaults < family < startup env < state file` inside it. The provider-facing
`MODEL_PROFILE` names map to the editable families:

| startup profile | self-service family / variant |
|---|---|
| `glm52-exl3` | `glm52` / `exl3-tr3` |
| `qwen36-27b-nvfp4` | `qwen36` / `qwen36-nvfp4` |
| `custom` | `custom` / `custom` plus `MODEL_ID` |

Tensor parallelism follows the GPUs visible through `nvidia-smi`,
`NVIDIA_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`; the provider's advertised
count is corroboration only. GLM-specific MLA/DCP/EXL3 flags are absent from
Qwen and custom launches. Inapplicable knobs are disabled in the UI and rejected
if injected through a hand-edited state file. See
[docs/model-families.md](docs/model-families.md).

Check the resolved values and exact vLLM argv without renting or downloading:

```bash
docker run --rm -e CONFIG_SMOKE=1 -e MODEL_PROFILE=qwen36-27b-nvfp4 \
  ghcr.io/malaiwah/glm52-exl3-vast:latest
```

## Self-service configuration (no rebuild, no re-rent)

The image is locked in when you rent, but the *deployment* is not. The landing
page on `:1111` has a **Configure** panel: every knob with its current value,
where that value came from (built-in default / template env / your saved file),
and a rationale explaining what it does and what it costs. Change what you
want, hit apply, and **vLLM restarts — the container, the weights, the API key
and the TLS cert are untouched.**

Full design note: [docs/self-service-config.md](docs/self-service-config.md).

- **Inheritance**: built-in defaults < startup environment < a JSON state file
  on the volume. The file wins on purpose: template env cannot be edited after
  launch, so the page has to be able to override it. The file stores only what
  you actually changed.
- **Pre-validation**: combinations that are known to be broken are refused
  before anything is written — an NVFP4 draft without
  `DRAFT_QUANTIZATION`, nvfp4 KV on a model with no calibrated MLA scales, a
  decode width outside the CUDA-graph/Trellis window, or a pinned pool smaller
  than `MAX_MODEL_LEN`. Each refusal quotes the measured reason.
- **Rollback**: if the new configuration does not come up, *or comes up and
  fails the long-context probe*, the last known-good configuration is restored
  and restarted automatically. The failed config, its boot log and the diff are
  kept under `.glm-config/failures/`.
- **Self-analysis**: once the known-good config is serving again, the model
  itself is handed the failed log, the working log and the diff, and writes a
  plain-language explanation onto the page.
- **Export / import**: download the saved config as JSON, paste it into the
  next instance.

> **"Healthy" is not a short prompt.** Silent-corruption configurations such
> as nvfp4 MLA KV without the checkpoint-calibrated outer scales answer
> `/health` and short prompts *perfectly* and can still fail long retrieval. So the
> post-restart check includes a **long-context needle probe**, and the page
> reports **Correctness** separately from **Engine**. If the probe did not run,
> it says "long context UNVERIFIED"; it never claims health it did not measure.

Requires `OPEN_BUTTON_TOKEN` in the template environment on Vast. Runpod
generates and persists a token automatically when none is supplied. Without a
token, the editor is not exposed.

## Terminate + session erase (opt-in, off by default)

When you are done, the landing page can **destroy the instance from inside it**
— so billing stops the moment you stop working, and so an optional erase can run
first. Full design note, provider matrix and cited sources:
[docs/termination-and-erase.md](docs/termination-and-erase.md).

- **Off by default.** Launch with `TERMINATE_ENABLED=1` to get the control at
  all. Every provider dashboard can already terminate an instance, so the
  in-container button is a convenience you opt into — not something a leaked
  landing-page URL hands to a stranger.
- **`TERMINATE_LOCKED=1`** is a hard lock: termination is refused no matter
  what. Both switches are **startup environment only**. A state file that tries
  to set either is rejected outright, and the landing page can only ever make
  them *more* restrictive — a locked instance cannot be unlocked from the UI, by
  any token, only by restarting the container with different env.
- **vast.ai and RunPod**, auto-detected (`TERMINATE_PROVIDER` overrides). An
  unrecognised provider says so and points at the dashboard instead of failing
  obscurely. On RunPod the pod-scoped key RunPod injects is **verified to
  terminate its own pod** — no extra credential needed; the page still checks it
  before you commit, and `RUNPOD_TERMINATE_API_KEY` covers the cases the pod key
  cannot.
- **Typed confirmation**: you type the instance id, plus an explicit
  acknowledgement checkbox. No single click can destroy anything.
- **`TERMINATE_DRY_RUN=1`** runs the whole flow and shows the request it would
  have sent, without sending it.
- **Session erase** (a checkbox, unchecked): overwrites and unlinks the API key,
  TLS private key, config state, every log this template writes (prompts
  included), shell history, SSH material, provider/HF credentials, and anything
  you added under the model dir. **The public model weights are deliberately not
  erased** — the checkpoint is downloadable by anyone, so overwriting 332 GB
  hides nothing. Optional RAM and VRAM zeroing. Read the limits in the design
  note: on SSDs with wear levelling, on overlay filesystems and on network
  volumes an overwrite does not guarantee the old bytes are unreachable, and the
  instance console log in your dashboard is outside our reach entirely.

> **RunPod, two things that bite by default:**
> 1. **Expose the ports when you create the pod** —
>    `--ports "22/tcp,1111/http,8000/http,8443/tcp"`.
>    A pod created without them comes up with `ports: null`: the container runs,
>    but the landing page, the API and even SSH are unreachable, and a running
>    pod's ports cannot be changed. You would have to destroy it and re-download
>    332 GB. (Measured.)
> 2. **A network volume survives termination and keeps billing** — and RunPod's
>    stock environment already points `HF_HOME` at it
>    (`/runpod-volume/.cache/huggingface/`), so your HF token lands there by
>    default. On such a pod the erase is the *only* thing that removes your
>    session data; the page says so in red, and the volume itself must be deleted
>    from the dashboard.

## GLM profile: vision (opt-in)

Set `VISION=1` (or enable it in the configurator) to graft the MoonViT-3d
tower (Kimi-K2.6, frozen) plus
Baseten's trained 49.5M PatchMerger projector onto the EXL3 text
backbone — **~890 MB**, no text weight touched. The default `VISION=0` text
profile preserves the maximum context/performance envelope; the operation is
fully reversible.

The final v20 turnkey image was qualified on the same 4x RTX PRO 6000
AIBeast stack described above:

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

**Know the edges:**
- **Vision is short-context-only on this EXL3 graft.** A detailed 5K screenshot
  works well, including multi-turn follow-up, but the same healthy process
  fails the mandatory 32K text-retrieval gate. It is therefore useful but not
  a flagship long-context profile.
- **Do not promote 0.98 because it boots.** Its larger advertised KV pool left
  too little transient allocation headroom. The measured 0.975 setting is the
  memory ceiling for this exact short-vision shape, not a promise for other
  models or drivers.
- **Ask for values, not ranks.** It reads text and numbers well; ordinal and
  counting reasoning is weak (it read all 14 chart values correctly, then put
  the highlighted model in the wrong rank).
- **Not for Computer Use.** Coordinate localisation is unusable: 0/6 targets
  within 40 px, mean error ~191 px, answers snapped to a round grid. The
  projector was trained at ~0.3 MP with no coordinate supervision. Pair it with
  a detector (OmniParser / OCR boxes) if you need clicks.
- Images only — video is not supported by this checkpoint.

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

## GLM profile: native quantized MTP78, with compatibility alternatives

The complete one-by-one justification for every non-default CLI parameter and
runtime environment value in the balanced profile is kept in
[`docs/glm52-tuning-rationale.md`](docs/glm52-tuning-rationale.md). It records
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

### Measured (4-arm QC run, same box / config / prompts, 2026-07-25)

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

**DRAM prefix-cache offload and memlock.** This tier does not enlarge the
GPU KV pool or make a larger active request fit. It preserves evicted prefix
KV in host DRAM so repeated system prompts, repositories, tool histories and
other large agentic prefixes can be restored instead of recomputed.

Vast accepts only ports, environment variables
and hostname in its template Docker Options, so a `--ulimit memlock=...` entry
there is ignored. Fortunately, gating offload on memlock is measurably a false
gate: a 125 GiB tier offloads normally under a 31 GiB limit because the
connector does not mlock the tier up front. The default is therefore
warn-and-proceed, and it degrades rather than fails —
`kv_load_failure_policy=recompute` means any KV block that cannot be fetched
back is recomputed instead of erroring the request. Set
`OFFLOAD_IGNORE_MEMLOCK=0` for conservative disable-instead behaviour.

`OFFLOAD_FRACTION` is an aggregate host-RAM budget. In the pinned native vLLM
connector, `cpu_bytes_to_use` already accounts for the complete TP world and
derives each worker's physical slice; dividing the value by TP again makes the
real cache four times smaller on TP4. The appliance passes the aggregate value
and reports both the total and estimated per-worker slice at boot.

The corrected TP4 implementation was exercised on a 251 GiB AIBeast host with
`OFFLOAD_FRACTION=0.5` (125 GiB aggregate). A cold 133,731-token prefix took
52.47 seconds. After five different ~133K prompts forced it completely out of
GPU cache, the same prefix returned from DRAM in 0.69 seconds: 133,504 external
prefix-hit tokens and 9.89 GB loaded CPU-to-GPU across four workers, with zero
allocation failures. That is about **76x lower TTFT than recomputation** for
this agentic-prefix shape. The preallocated tier left about 51 GiB of host RAM
available. Although the configurator permits larger fractions, 50% is the
recommended ceiling on a 256 GiB host; 70% would leave too little operating
margin on this machine.

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

### Historical pre-v31 comparison: EXL3 + grafted MTP78

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
> runtime now ships the GLM-5.2-specific calibrated MLA outer-scale file and
> the entrypoint preserves `VLLM_NVFP4_MLA_SCALES_FILE`; the configurator
> refuses this KV dtype for model families without an equivalent calibration.
> Release qualification still runs cold long-context needles—calibration is
> not a reason to skip the causal test.

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

### Separate EXL3 draft override (experimental; do not use for production)

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

### Model quality (the target model, unrelated to the draft)

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

## Vast.ai template settings (manual setup)
- **Image**: `ghcr.io/malaiwah/glm52-exl3-vast:latest` (the ghcr.io package
  must be set to **public** visibility, or vast hosts can't pull it)
- **Launch mode**: docker ENTRYPOINT (vLLM logs appear on the instance console;
  the image starts its own key-only SSH daemon)
- **Docker options**: `-p 22:22 -p 8000:8000 -p 1111:1111`. Vast's current
  [Docker Options documentation](https://docs.vast.ai/guides/instances/docker-environment#docker-create-options)
  accepts only ports, environment variables and hostname in this field;
  `--ipc` and `--ulimit` entries are ignored. Port 22 is SSH and port 1111 is
  the landing page.
- **Profile**: `MODEL_PROFILE=glm52-exl3` (default), or
  `MODEL_PROFILE=qwen36-27b-nvfp4` for the low-cost development model.
- **Disk**: >=450 GB for GLM; >=80 GB for Qwen.
- **GPU filter**: 4x RTX PRO 6000 Blackwell (96 GB) for GLM; one RTX PRO 6000
  Blackwell or RTX 5090 for Qwen.
- **Env (all optional)**: `HF_TOKEN` (authenticated download and higher
  applicable Hub rate limits), `OFFLOAD_FRACTION`
  (GLM default 0.5 for reusable agentic prefixes), `MTP_TOKENS` (GLM default 5; Qwen
  default 0), `MAX_NUM_SEQS`, `MAX_MODEL_LEN` (GLM 513536; Qwen 32768),
  `SERVED_MODEL_NAME`,
  `MTP78_TRELLIS` (default 1: quantized trellis draft, see MTP78 section; 0 = stock BF16 draft),
  `LANDING_PAGE` (default 1; 0 disables the :1111 landing page). Recommended
  extra env: `OPEN_BUTTON_PORT=1111` — the dashboard **Open** button then hits
  the landing page: live boot status (weight-download progress, TLS, engine),
  ready-to-paste client configs (oh-my-pi, opencode, Claude Code, Codex),
  a minimal streaming chat UI at `/chat`, and the **self-service config editor**
  at `/config` (needs `OPEN_BUTTON_TOKEN`; see the section above). Token-gated; with TLS configured the
  page upgrades plain-HTTP hits to HTTPS and only then embeds the API key.
  On ready, the instance labels itself "`<model> READY <endpoint>`" in your dashboard.

Endpoint: `http://<public-ip>:<mapped-8000-port>/v1` once the console shows
`Application startup complete` (first boot: download + JIT, plan ~30-60 min;
later boots reuse both weights and the compatible AOT compile cache).

Checkpoint downloads use `huggingface_hub.snapshot_download` with the bundled
`hf-xet` transport and `HF_XET_HIGH_PERFORMANCE=1`. `MODEL_DOWNLOAD_WORKERS`
defaults to 16 concurrent files. Hugging Face's adaptive Xet concurrency remains
the default for each file; advanced deployments can pass through
`HF_XET_FIXED_DOWNLOAD_CONCURRENCY` after measuring their route. An `HF_TOKEN`
authenticates the request and can avoid anonymous rate limits, but does not by
itself guarantee that a particular host-to-CAS route will be fast. See Hugging
Face's [model-download guidance](https://huggingface.co/docs/hub/models-downloading)
and [Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## Evidence / why these defaults
Root-cause investigation of the long-context corruption and the validated
config matrix (6 runs, 5 hosts, 4 driver families):
- Root cause (nvfp4 KV x host P2P state):
  https://gist.github.com/cae272443a9817da72b6802a0b9a5d73
- Override-host proof 7/7 @505K:
  https://gist.github.com/7d5d7e685f7498a356fa2dd12b876f14
- fp8 clean to 440K on stock: same matrix gist; harness:
  https://gist.github.com/929d7d8e4ac94c43fe126c4b3f6a6ea6
- Historical fallback: 512K fp8 KV via `--num-gpu-blocks-override 2048` at
  util 0.93. The v29 default instead uses calibrated NVFP4 KV and auto-sizing.

Base runtime image:
`voipmonitor/vllm@sha256:7b230b45991d93065d99c863fdb9ae030fb49592b59fa3c930cc00bfde09e51d`
(pinned GG v20-r5). Its labels record GG base `4247d676`, vLLM EXL3
integration tree `936ed48` (including
[PR #190](https://github.com/local-inference-lab/vllm/pull/190)), SparkInfer
integration tree `f532ec9` (including
[PR #49](https://github.com/local-inference-lab/sparkinfer/pull/49)), and
FlashInfer `801d57a`. It also includes native vLLM support for
`Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen parsers, and MTP
speculative decoding.

Profile checkpoints:

- GLM: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` at `9297b9f1…`
- MadeBy561 control: `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at `68babde2…`
- Qwen: `nvidia/Qwen3.6-27B-NVFP4` at `0893e160…`

### Running it on your own hardware

The same image drops onto an owned box as a transparent replacement for an
existing endpoint:

| env | default | why you'd change it |
|---|---|---|
| `MODEL_PROFILE` | `glm52-exl3` | select `qwen36-27b-nvfp4` for low-cost testing, or `custom` with `MODEL_ID` |
| `MODEL_ID` | profile checkpoint | use a compatible alternate checkpoint without changing its profile defaults |
| `MODEL_DIR` | profile-specific path under `/workspace` | point at weights you already have; the download marker is tied to `MODEL_ID` |
| `MODEL_DISPLAY_NAME` | profile name | dashboard and provider label |
| `SERVED_MODEL_NAME` | profile name | whitespace-separated aliases, so existing clients keep working |
| `TENSOR_PARALLEL_SIZE` | 4 GLM / 1 Qwen | match a supported profile topology |
| `MAX_MODEL_LEN` | 513536 GLM / 32768 Qwen | increase the Qwen development context only after measuring VRAM |
| `MULTIMODAL` | n/a GLM / 0 Qwen | Qwen `1` loads its native vision encoder; GLM vision remains controlled by `VISION` (default 0) |
| `QUANTIZATION` | custom profile only | vLLM quantizer name such as `modelopt` |
| `REASONING_PARSER` / `TOOL_CALL_PARSER` | custom profile only | model-specific OpenAI response parsers |
| `AUTH` | `key` | `none` serves unauthenticated on a trusted LAN |
| `ALLOW_UNSUPPORTED_GPU` | `0` | bypass the profile GPU-name check; the required visible GPU count still applies |
| `MIN_NVIDIA_DRIVER_VERSION` | `595.45.04` | raise when a newer CUDA/base image requires it; the gate runs before model download |
| `ALLOW_UNSUPPORTED_NVIDIA_DRIVER` | `0` | bypass the CUDA 13.2 driver floor only for an intentionally qualified compatibility-package stack |
| `GPU_BLOCKS_OVERRIDE` | 0 | auto-profile the largest safe KV pool; set a positive block count to pin it |
| `OFFLOAD_FRACTION` | 0.5 GLM / 0 Qwen | host DRAM used as an aggregate L2 prefix cache (not active-context capacity); `0.5` is the measured agentic-workload setting on a 256 GiB host and native vLLM derives the TP worker slices |
| `OFFLOAD_IGNORE_MEMLOCK` | `1` | proceed when the memlock ulimit is below the tier size (see below); `0` disables offload instead |
| `MTP78_MODE` | `off` (native) | the current Brandon revision contains a native rank-sliced TR3 draft; `graft` and `override` remain experimental compatibility paths. MadeBy561's native draft uses serialized NVFP4 experts. Prefer the `MTP_DRAFT` knob on the config page. |
| `MTP_DRAFT_SAMPLE_METHOD` | `probabilistic` GLM | measured MTP-5 proposal mode; `greedy` remains available for controlled A/B tests |
| `F8_DMA` | `0` family / `ring` MadeBy561 | compressed PCIe collective mode; the hybrid override passed the 521K five-depth gate |
| `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | `-1` family / `8192` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured crossover |
| `PCIE_DMA_MIN_BYTES` | `-1` family / `393216` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured byte crossover |
| `OPEN_BUTTON_TOKEN` | provider-specific | required to expose the `:1111` config editor; Vast supplies it and Runpod gets a persisted generated token when one is not set |
| `SOUL_AUTONOMY_LEVEL` | `0` | enable the embedded diagnostic SOUL: observer `1`, shell investigator `2`, or bounded proactive diagnostician `3` |
| `SOUL_AUTONOMY_MAX_LEVEL` | `3` | startup-only ceiling for landing-page overrides; invalid values fail closed to `0` |
| `SOUL_HEARTBEAT_INTERVAL_S` / `SOUL_JOURNAL_INTERVAL_S` | `300` / `3600` | deterministic snapshot and blog-style journal cadence; changing these does not restart vLLM |
| `VERIFY` | `1` | `0` disables the post-start correctness probe entirely (the page then reports "unverified" and nothing rolls back) |
| `VERIFY_LONG_CONTEXT` | `1` | `0` keeps the short-prompt checks only — read the warning above before using it |
| `VERIFY_NEEDLE_TOKENS` | `32768` | size of the long-context retrieval probe |
| `GLM_STATE_DIR` | `<volume>/.glm-config` | where the config state file, known-good config, failures and logs live |
| `MODEL_FAMILY` / `MODEL_VARIANT` | selected by `MODEL_PROFILE` | `glm52`/`exl3-tr3`, `qwen36`/`qwen36-nvfp4`, or `custom`/`custom`; the config page can switch these without rebuilding |
| `SSHD` | `auto` | `auto` starts the bundled key-only sshd when a provider injects a public key and nothing is already listening; `0` never starts it and `1` always tries |
| `CONFIG_SMOKE` | `0` | `1` resolves the config, prints the argv and exits without downloading or touching a GPU |
| `TERMINATE_ENABLED` | `0` | `1` exposes the terminate control on the landing page (startup env only) |
| `TERMINATE_LOCKED` | `0` | `1` hard-locks termination for the life of the container (startup env only) |
| `TERMINATE_PROVIDER` | (auto) | force `vastai` or `runpod` when detection fails |
| `RUNPOD_TERMINATE_API_KEY` | (unset) | RunPod account API key. Not normally needed — the injected pod-scoped key is verified to terminate its own pod — but covers a missing/altered key or targeting another pod |
| `TERMINATE_DRY_RUN` | `0` | `1` prepares the destroy request and does not send it |
| `TERMINATE_PROBE` | `1` | `0` skips the read-only credential pre-check |

## Security

**Threat model honestly stated:** a rented host's operator has root — memory,
VRAM, and traffic on the box are visible to a determined host. These controls
are the padlock that keeps honest people honest; truly sensitive work belongs
on hardware you own.

- **API key** (on by default): set `VLLM_API_KEY`, or one is auto-generated and
  printed in the instance console logs at boot. All /v1 calls need
  `Authorization: Bearer <key>`. The key is persisted to the volume, so a
  restart does not silently invalidate client configs.
- **`AUTH=none` disables authentication entirely.** This exists for dropping the
  image into a *trusted private network* — e.g. replacing an in-house endpoint
  whose clients are already configured without a key. It is never appropriate on
  a rented public host, so it is opt-in and printed as a loud warning at boot.
- **SSH tunnel** (recommended for solo use): no public API exposure needed —
  `ssh -p <ssh-port> root@<ssh-host> -L 8000:localhost:8000`
  then use `http://localhost:8000/v1`. You can omit `-p 8000:8000` from the
  Vast Docker options entirely in this mode. Keep `-p 22:22`; Vast maps it to
  the external SSH port shown in the instance panel. On Runpod, expose
  `22/tcp` and use the public IP plus mapped port from the Connect panel. The
  image installs Vast's `SSH_PUBLIC_KEY` or Runpod's `PUBLIC_KEY` and starts
  key-only `sshd` itself.
- **Direct-TCP TLS via Let's Encrypt DNS-01 — turnkey with deSEC**
  (recommended for Vast and for Runpod's inference API; the Runpod dashboard
  stays on managed proxy HTTPS):
  One-time setup (~2 minutes, free, reusable forever):
  1. Create an account at [desec.io](https://desec.io/signup) (email only).
  2. Register a dynDNS domain, e.g. `yourname.dedyn.io`
     ([docs](https://desec.readthedocs.io/en/latest/dyndns/configure.html)).
  3. Create an API token: [Token management](https://desec.io/tokens)
     ([docs](https://desec.readthedocs.io/en/latest/auth/tokens.html)).

  Store `DESEC_TOKEN=<your-token>` in Vast
  [Account Settings](https://cloud.vast.ai/account/) **Environment Variables**
  section, where it is encrypted and injected at launch. On Runpod, create a
  Secret such as `desec_token` and reference it from the private template as
  `{{ RUNPOD_SECRET_desec_token }}`. Add
  `DESEC_DOMAIN=yourname.dedyn.io` to the instance or private template; on
  Runpod also expose `8000/http` plus `8443/tcp` and set
  `RUNPOD_DIRECT_TLS=auto` (secure proxy fallback) or `1` (fail-fast).
  At boot the instance registers a stable per-instance hostname
  (`model-<provider-instance-id>.yourname.dedyn.io`), points it at itself, obtains a
  Let's Encrypt certificate via DNS-01 ([lego](https://go-acme.github.io/lego/dns/desec/)),
  and prints the final `https://...:<port>/v1` URL in the console logs next to
  the API key. Each instance gets its own name, stable across reboots — so
  records don't pile up in the zone and certs persist on the volume, reused
  while they have >7 days validity left.

- **Other DNS providers** (Cloudflare, DuckDNS, 150+ via lego): set
  `ACME_DOMAIN=model.example.com`, `ACME_DNS_PROVIDER=cloudflare` (any lego
  provider), and the provider credential env (e.g.
  `CLOUDFLARE_DNS_API_TOKEN=...` with Zone:DNS:Edit scope; or DuckDNS:
  `ACME_DNS_PROVIDER=duckdns` + `DUCKDNS_TOKEN=...` — free, no domain needed).
  Point the name at the instance IP, and the endpoint becomes
  `https://<domain>:<mapped-port>/v1`. Certs persist on the volume and are
  reused while they have >7 days validity left, then re-issued at boot (avoids
  Let's Encrypt's 5/week duplicate-cert limit on reboot loops).
- **Token hygiene**: put `HF_TOKEN`, `DESEC_TOKEN`, API keys and other secrets
  in Vast's account-level environment-variable store or Runpod Secrets, never
  in a public template. They remain visible to the rented host's operator, so
  scope DNS tokens narrowly (single zone, DNS-only) and rotate them when the
  rental ends.
- **Egress hygiene**: telemetry disabled (`VLLM_NO_USAGE_STATS`,
  `DO_NOT_TRACK`, `HF_HUB_DISABLE_TELEMETRY`), `HF_HUB_OFFLINE=1` once weights
  are local, and the boot log prints the listening-socket audit. Full egress
  firewalling is not possible without NET_ADMIN (not granted on vast).
- **Disk note**: verify the allocated disk matches the profile. GLM needs at
  least 450 GB for weights, overlays, graft backup, vision and compile cache.
  Qwen's checkpoint is about 21 GiB; its 80 GB volume leaves room for cache and
  experiments.
