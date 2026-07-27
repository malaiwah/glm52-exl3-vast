# Model turnkey for Vast.ai and Runpod

One image, coherent profiles for **GLM-5.2**, **Qwen3.6-27B**, and compatible
vLLM checkpoints. It supplies an authenticated OpenAI-compatible endpoint,
persistent model downloads and compile caches, a live dashboard, key-only SSH,
provider-aware URLs, optional TLS, and crash supervision.

The default `glm52-exl3` profile remains the validated production stack:
EXL3 trellis weights (~77 GiB/rank — fits commodity 95.01-GiB cards),
**fp8 KV cache** (correct on stock drivers — the nvfp4 default silently
corrupts >~150K context without a host driver P2P override; see Evidence),
MTP speculative decoding **with a quantized trellis draft** (see MTP78 below),
and DRAM KV offload auto-sized to 70% of the instance's RAM allocation
(cgroup-aware — partial rentals don't oversize it). Weights auto-download on
first boot (~332 GB — pick a fast-net host).

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
[pinned vLLM source](https://github.com/voipmonitor/vllm/tree/551719766029e78824a30d97ae6ac63917405b5f)
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

## Launch GLM-5.2 on Vast.ai

**[▶ Launch on vast.ai](https://cloud.vast.ai/?ref_id=386667&template_id=ccab1ea5b390cec1bb615a79840baa40)** —
public template with the image, ports, launch mode, disk, and host filters
(4x RTX PRO 6000, >=1Gbps net) pre-configured. Before renting, verify the
offer allocates **at least 450 GB of disk**. Then rent, wait for
"Application startup complete" in the instance logs, grab the API key from the
same logs, done.

For lower-cost Qwen testing, clone/create a private Vast template using the
same image, select one compatible Blackwell GPU, allocate at least 80 GB of
disk, and add:

```text
MODEL_PROFILE=qwen36-27b-nvfp4
```

## Launch on Runpod

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

Create either private template with:

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
  schema. Do not select RTX 4090 or another pre-Blackwell GPU.
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
  as `{{ RUNPOD_SECRET_secret_name }}`. The checked-in manifests expect a
  secret named `desec_token`. Do not put credentials in the JSON or a shared
  template.

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

## GLM profile: vision (default ON)

Images work out of the box: the MoonViT-3d tower (Kimi-K2.6, frozen) plus
Baseten's trained 49.5M PatchMerger projector are grafted onto the EXL3 text
backbone at first boot — **~890 MB**, no text weight touched. `VISION=0` serves
pure text (fully reversible).

Measured on 4x RTX PRO 6000 (full writeup:
https://gist.github.com/malaiwah/c004c8b48bb177203f56cb29107f8540):

| what | result |
|---|---|
| smoke (chart values / OCR / counting / diagram) | 4/4 exact |
| multi-image 2 / 4 / 6 / 8 | all correct, ~151 tokens per image |
| 6 near-identical charts (cross-image binding) | 4/4 |
| dense 14-model bar chart | all labels + values exact |
| **MTP speculation under vision** | **MAL 3.526 / 84.2% — unchanged vs text-only** |
| KV cost of vision | ~1.3 GiB/GPU (still +2.36 GiB vs a BF16 draft) |

**Know the edges:**
- **Downscale screenshots to <= 4096 px.** 5K images are accepted but unreliable;
  everything >= 2560 px hits the same ~4250-token patch cap anyway.
- **Ask for values, not ranks.** It reads text and numbers well; ordinal and
  counting reasoning is weak (it read all 14 chart values correctly, then put
  the highlighted model in the wrong rank).
- **Not for Computer Use.** Coordinate localisation is unusable: 0/6 targets
  within 40 px, mean error ~191 px, answers snapped to a round grid. The
  projector was trained at ~0.3 MP with no coordinate supervision. Pair it with
  a detector (OmniParser / OCR boxes) if you need clicks.
- Images only — video is not supported by this checkpoint.

## GLM profile: MTP78 quantized speculative-draft layer (default ON)

The MTP draft layer ships in BF16 (19.3 GB). This template grafts a
**3.0bpw EXL3 trellis version** (`malaiwah/GLM-5.2-EXL3-TR3-MTP78`,
all-256-expert, calibrated on a 7.3M-token full-corpus capture) at first
boot — validated at **BF16-parity acceptance (MAL 3.06 vs 3.05)** while
freeing **~3.8 GB/GPU straight into KV cache**. Set `MTP78_TRELLIS=0` to
revert to the stock BF16 draft (the graft is fully reversible; `.orig`
backups are kept).

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

**DRAM offload and memlock.** Vast accepts only ports, environment variables
and hostname in its template Docker Options, so a `--ulimit memlock=...` entry
there is ignored. Fortunately, gating offload on memlock is measurably a false
gate: a 125 GiB tier offloads normally under a 31 GiB limit because the
connector does not mlock the tier up front. The default is therefore
warn-and-proceed, and it degrades rather than fails —
`kv_load_failure_policy=recompute` means any KV block that cannot be fetched
back is recomputed instead of erroring the request. Set
`OFFLOAD_IGNORE_MEMLOCK=0` for conservative disable-instead behaviour.

**KV headroom:** available KV memory goes 5.27 -> 8.92 GiB/GPU (**+69%**). This
template pins the pool at 512K (`--num-gpu-blocks-override 2048`), so the
headroom is unspent by default — at ~10.3 KiB/token it is worth roughly
**+355K tokens of pool (~880K context)** if you lift the override, or the same
512K with much more concurrency margin.

**Speculation depth:** the cheaper draft also moves the optimum. Measured
(GSM8K n=30, +-0.5% noise floor): MTP-2 42.9 tok/s, **MTP-3 51.5** (default),
**MTP-5 53.4** (+3.7%). MTP-5 used to lose ~22% with the 19.3 GB BF16 draft;
at 3.7 GB the extra draft tokens are cheap enough to win. Try `MTP_TOKENS=5`
if you want that last few percent (not yet the default — wants a larger run).

### The converged stack: EXL3 + MTP78 + vision, one image

Measured 2026-07-26 on owned hardware (4x RTX PRO 6000 Blackwell, **280 W cap**,
TP4/DCP4-a2a, MTP-3, 512K context, DRAM KV offload on, clean single-stream):

| stack (fp8 KV — what this template ships) | decode C1 | MAL / accept | KV/GPU | KV pool | 505K needle |
|---|---|---|---|---|---|
| GLM-5.2 NVFP4-NF3 hybrid (previous prod, calibrated nvfp4 KV) | 119.2 tok/s | ~3.5 / 0.83 | 4.64 GiB | 537,600 tok | 7/7 |
| **EXL3-TR3 3bpw + MTP78, fp8 KV** | 112.4 tok/s | 3.471 / 0.824 | **8.89 GiB** | **697,600 tok** | **6/6** |

**The honest trade: ~6% slower decode for ~30% more KV pool, and vision.** The
EXL3 weights are ~7 GiB/rank smaller than the hybrid's, which is what pays for
the bigger pool even though fp8 KV costs ~1.7x the bytes per token that nvfp4
would. Long-context retrieval is verified clean (6/6 at depths to 490K inside a
505K request).

> **What this table used to say, and why it was wrong.** An earlier revision
> reported 127.4 tok/s and an 860,928-token pool for this stack — "faster *and*
> bigger". Those numbers were measured with `--kv-cache-dtype nvfp4_ds_mla`,
> which this template does not ship. That config fails the same needle test
> **0/6 with degenerate output** (`".,, while.,, and and while,,,,"`) while
> GSM8K, vision and structured output all still pass, so nothing short of a
> long-context retrieval test catches it. nvfp4 KV needs *per-checkpoint
> calibrated MLA outer scales*: the hybrid checkpoint has them, the EXL3
> checkpoint does not, and borrowing the hybrid's would be meaningless.
> If a quantization change ever looks like a free lunch on both axes at once,
> measure retrieval before believing it.

With vision resident, fp8 sizes the usable ceiling at ~420K rather than 512K
(vision costs ~1.31 GiB/GPU and ~13% of *text* decode). Set `MAX_MODEL_LEN`
to 384K-420K for a vision deployment, or `VISION=0` to keep the full 512K.

**MTP survives the graft** (MAL 3.528 with vision vs 3.533 without). Reports of
*zero* draft acceptance on comparable hybrid grafts come from the speculator not
seeing the nested `lm_head`; the plugin used here exposes it.

Two knobs are load-bearing in the validated graft configuration:

- `ONLINE_QUANT=none` — serving presets that default to an mxfp8 online overlay
  make EXL3 refuse with `quantization_config is only supported when ...`. The
  entrypoint sets this explicitly.
- `VLLM_EXL3_TRELLIS_MIN_M=4` — this is the validated lower bound for the
  in-checkpoint graft. Do **not** lower it to 1 to make the separate EXL3 draft
  override boot: that clears a CUDA-graph error by moving unvalidated m=1..3
  shapes onto the trellis kernel and was measured to cause silent long-context
  corruption.

A trellis (or BF16) MTP draft also needs `moe_backend=triton` **separately from**
the target's backend — a rank-3 trellis tensor is not a fused expert weight.

### Separate EXL3 draft override (experimental; do not use for production)

The overlay works as a *separate draft directory* — leave the base checkpoint
untouched and add one field:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3,
                       "moe_backend":"triton","draft_sample_method":"probabilistic",
                       "model":"/path/to/GLM-5.2-EXL3-TR3-MTP78/3bpw-keep0"}'
```

This proves that the draft loads independently and recovers the expected KV
memory, but it is **not a production-safe path on the current image**: its
speculator captures at m=3, outside the validated `[4,32]` trellis window.
Lowering the window to 1 produced silent retrieval corruption. The turnkey
therefore defaults to the validated in-place graft.

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
  (GLM default 0.70; Qwen default 0), `MTP_TOKENS` (GLM default 3; Qwen
  default 0), `MAX_NUM_SEQS`, `MAX_MODEL_LEN` (GLM 524288; Qwen 32768),
  `SERVED_MODEL_NAME`,
  `MTP78_TRELLIS` (default 1: quantized trellis draft, see MTP78 section; 0 = stock BF16 draft),
  `LANDING_PAGE` (default 1; 0 disables the :1111 landing page). Recommended
  extra env: `OPEN_BUTTON_PORT=1111` — the dashboard **Open** button then hits
  the landing page: live boot status (weight-download progress, TLS, engine),
  ready-to-paste client configs (oh-my-pi, opencode, Claude Code, Codex), and
  a minimal streaming chat UI at `/chat`. Token-gated; with TLS configured the
  page upgrades plain-HTTP hits to HTTPS and only then embeds the API key.
  On ready, the instance labels itself "`<model> READY <endpoint>`" in your dashboard.

Endpoint: `http://<public-ip>:<mapped-8000-port>/v1` once the console shows
`Application startup complete` (first boot: download + JIT, plan ~30-60 min;
later boots only pay JIT).

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
- 512K fp8 KV via `--num-gpu-blocks-override 2048` validated at util 0.93.

Base runtime image:
`verdictai/glm52-exl3-sparkinfer@sha256:2bb9e804a283d1da3b7e3425ff87375121285141d0d0a40d3dc09d41bf881a10`
(pinned). It contains the specialized GLM extensions, but also includes native
vLLM support for `Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen
parsers, and MTP speculative decoding.

Profile checkpoints:

- GLM: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`
- Qwen: `nvidia/Qwen3.6-27B-NVFP4`

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
| `MAX_MODEL_LEN` | 524288 GLM / 32768 Qwen | increase the Qwen development context only after measuring VRAM |
| `MULTIMODAL` | 1 GLM / 0 Qwen | Qwen `1` loads its native vision encoder; GLM vision remains controlled by `VISION` |
| `QUANTIZATION` | custom profile only | vLLM quantizer name such as `modelopt` |
| `REASONING_PARSER` / `TOOL_CALL_PARSER` | custom profile only | model-specific OpenAI response parsers |
| `AUTH` | `key` | `none` serves unauthenticated on a trusted LAN |
| `ALLOW_UNSUPPORTED_GPU` | `0` | bypass the profile GPU-name check; the required visible GPU count still applies |
| `GPU_BLOCKS_OVERRIDE` | 2048 GLM / 0 otherwise | `0` lets vLLM size KV normally |
| `OFFLOAD_FRACTION` | 0.70 GLM / 0 otherwise | fraction of RAM for the DRAM KV tier |
| `OFFLOAD_IGNORE_MEMLOCK` | `1` | proceed when the memlock ulimit is below the tier size (see below); `0` disables offload instead |
| `MTP78_MODE` | `graft` | validated in-place trellis draft; `off` restores the stock BF16 draft; `override` is experimental and currently unsafe at long context |

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
