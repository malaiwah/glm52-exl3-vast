# Model turnkey for Vast.ai, Runpod, and JarvisLabs

One image, coherent profiles for **full GLM-5.3 (non-Flash)**,
**GLM-5.2**, **Qwen3.6-27B**, and compatible vLLM checkpoints. It supplies an authenticated
OpenAI-compatible endpoint, persistent model downloads and compile caches, a
live dashboard, key-only SSH, provider-aware URLs, optional TLS, crash
supervision, and an opt-in embedded diagnostic [SOUL](docs/soul.md).

The next-release default is **`MODEL_PROFILE=glm53-3.42bpw-500k`**, the full 755B
`glm_moe_dsa` model, not Flash. It retains 3.42bpw and AIBeast's **520,192-token**
total request limit with reduced workspace. The spot-container run retrieved
3/3 facts from a **501,086-token haystack**; this is positive >=500K evidence,
not the full maintenance matrix or exact OCI-runtime qualification. See
[current results](TEST_RESULTS.md#jarvislabs-spot-container-proof-2026-09-08).
**No AIBeast production restart, final image promotion or `latest` update
was performed.** Changing the bare-launch default does not switch production.

The runtime base is local-inference-lab's **Gilded Gnosis v20 r34**
(`docker.io/voipmonitor/vllm@sha256:820181fb…`: vLLM `e2666d9a65` integration
tree `4d006a43`, B12X/SparkInfer `cd3ce190`, LMCache `0.5.2+glm52dcp.4`,
FlashInfer `1ac69427`, NCCL 2.30.4, Torch 2.12.0+cu132, CUDA 13.2.1), plus the
reviewed r34 maintenance sources and this repository's fail-closed overlays.
The base and selected runtime files are hash-pinned; this is not a complete
filesystem or supply-chain equivalence claim. The Gilded refresh uses necessary
patches re-derived for that base, not proof that all 27 Verdict overlays were
already present.

The full `glm53-3.42bpw` profile keeps its historical 393,216-token envelope;
that older qualification does not transfer to the refreshed Gilded image.
`glm52-exl3` remains an explicit alternative. **GLM-5.3-Flash is omitted from
this build** because the base lacks its GLM5Next pooled-indexer runtime, not
because a port has been proved impossible. The separate
[VerdictAI-derived reference](https://github.com/malaiwah/glm52-exl3-vast/pkgs/container/glm52-exl3-vast)
is `ghcr.io/malaiwah/glm52-exl3-vast@sha256:9d7ab60a3ad666edb8d38812ec6709909e6752a78fad464c842c7b17659f5d5b`;
its GPU qualification does not transfer here.
Existing hosted provider links may still select their historical profiles; inspect the profile and image
pin before renting. Exact runtime and checkpoint pins are in the
[changelog](CHANGELOG.md).

## Contents

- [Why this exists](#why-this-exists)
- [Quick start](#quick-start)
  - [Self-serve with Docker or Podman](#self-serve-with-docker-or-podman)
- [Model profiles](#model-profiles)
- [What startup looks like](#what-startup-looks-like)
- [Launch GLM-5.2 on Vast.ai](#launch-glm-52-on-vastai)
- [Launch on Runpod](#launch-on-runpod)
- [Launch on JarvisLabs](#launch-on-jarvislabs)
- [Running it on your own hardware](#running-it-on-your-own-hardware)
- [Self-service profile switching](#self-service-profile-switching)
- [Self-service configuration](#self-service-configuration-no-rebuild-no-re-rent)
- [Terminate + session erase](#terminate--session-erase-opt-in-off-by-default)
- [GLM profile: vision (opt-in)](#glm-profile-vision-opt-in)
- [GLM profile: MTP78 draft](#glm-profile-mtp78-draft)
- [Vast.ai template settings (manual setup)](#vastai-template-settings-manual-setup)
- [Configuration reference](#configuration-reference)
- [Evidence / why these defaults](#evidence--why-these-defaults)
- [Security](#security)

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

## Quick start

| profile | provider | launch | hardware | disk | first-boot budget |
|---|---|---|---|---|---|
| GLM-5.3 full 3.42bpw **500K+ candidate** | spot container / own host / authorized maintenance | [spot rootfs guide](#launch-on-jarvislabs) / [docker/podman](#self-serve-with-docker-or-podman) | 4x RTX PRO 6000 Blackwell 96 GB | **600 GB**, extra space for optional L2 | >=500K spot retrieval passed; cold-start budget not measured |
| GLM-5.3 full 3.42bpw, historical 393K profile | self-serve / JarvisLabs VM | [docker/podman](#self-serve-with-docker-or-podman) | 4x RTX PRO 6000 Blackwell 96 GB | **600 GB** | historical image: up to 90 min cold; ~9 min with warm runtime cache |
| GLM-5.2 flagship (3.42bpw) | Vast.ai | [▶ Launch](https://cloud.vast.ai/?ref_id=386667&template_id=6d2679c1ebae36d54274c98123473405) | 4x RTX PRO 6000 Blackwell 96 GB | **600 GB** | 60–90 min |
| GLM-5.2 flagship (3.42bpw) | Runpod | [▶ Launch](https://console.runpod.io/deploy?template=f8sgtc6orf&ref=4ahycj93) | 4x RTX PRO 6000 Blackwell 96 GB | **600 GB** | ~30 min (Secure) |
| GLM-5.2 flagship (3.42bpw) | JarvisLabs | [▶ VM guide](#launch-on-jarvislabs) | 4x RTX-PRO6000 VM | **650 GB** | ~30 min |
| Qwen3.6 vision (low-cost) | Vast.ai | [▶ Launch](https://cloud.vast.ai/?ref_id=386667&template_id=214d2e120a6718558fa2078d4579d4316) | 1x RTX 5090 32 GB | 100 GB | ~6 min on a well-connected host |
| Qwen3.6 vision (low-cost) | Runpod | [▶ Launch](https://console.runpod.io/deploy?template=7ufac3b4zw&ref=4ahycj93) | 1x RTX 5090 32 GB | 100 GB | ~30 min |
| GLM-5.2 flagship (3.42bpw) | self-serve (own hardware) | [▶ docker/podman](#self-serve-with-docker-or-podman) | 4x RTX PRO 6000 Blackwell 96 GB | **600 GB** | 2–12 min once weights are local |
| Qwen3.6 vision (low-cost) | self-serve (own hardware) | [▶ docker/podman](#self-serve-with-docker-or-podman) | 1x RTX 5090 32 GB | 100 GB | ~1 min warm |

**Requirements that fail fast:** Blackwell (`sm120+`) GPUs only, and the
qualified pair **NVIDIA driver 590.48.01+ / CUDA 13.2+** — both are checked
before any weights download. First boot downloads the checkpoint, so set a
cold-start cost deadline before renting. Wait for `Application startup
complete` plus the verification result in the instance logs, then use the
generated API key and labeled endpoint. Measured per-provider timings are
under [What startup looks like](#what-startup-looks-like).

### Self-serve with Docker or Podman

Use a separately staged candidate image on idle, authorized hardware. Set
`IMAGE` to the candidate's immutable published `repo@sha256:…` reference, not
`latest`; the existing production image does not acquire this profile by
changing an environment variable. With Docker and the NVIDIA container toolkit,
the following downloads weights into a dedicated workspace and prints keys in
the container logs. **Do not launch alongside the production GPU workload.**

```bash
: "${IMAGE:?Set IMAGE to the published candidate repository@sha256:digest}"
sudo docker run -d --name glm53-342-candidate \
  --restart no \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1:-1 \
  -e MODEL_PROFILE=glm53-3.42bpw-500k -e PORT=8001 \
  -v /srv/turnkey-glm53-candidate:/workspace \
  "$IMAGE"
```

This Docker shape was used in the historical JarvisLabs VM qualification,
not the current spot-container rootfs graft. Select
`MODEL_PROFILE=glm53-3.42bpw` for the historical 393K full-model profile,
`MODEL_PROFILE=glm52-exl3` for the legacy four-GPU profile, or
`MODEL_PROFILE=qwen36-27b-nvfp4` for the one-GPU profile. For authenticated
downloads, read `HF_TOKEN` with a silent prompt, export it and pass only
`-e HF_TOKEN`, or use a permission-restricted environment file; never put
the token value in command arguments or shell history.
Follow first boot with `sudo docker logs -f glm53-342-candidate`. This example's endpoint is
`http://localhost:8001/v1` and the tokenized dashboard is on `:1111`; keep
both behind your LAN or an SSH tunnel, or configure TLS as described in
[Security](#security).

**Host-driver maintenance only — NVIDIA P2P override.** The commands below
unload NVIDIA modules and must not run while production or another GPU client
is active. Schedule a separate authorized maintenance window; this is not a
candidate-staging prerequisite to execute automatically.
On direct-attach multi-GPU systems where `nvidia-smi topo -m` shows `PHB` or
`NODE` between GPUs, create `/etc/modprobe.d/nvidia-p2p-override.conf`:

```bash
echo 'options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"' \
  | sudo tee /etc/modprobe.d/nvidia-p2p-override.conf
echo 'options nvidia_uvm uvm_disable_hmm=1' \
  | sudo tee /etc/modprobe.d/uvm.conf
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia && sudo modprobe nvidia_uvm
```

Without this, CUDA kernels that issue direct peer-GPU memory loads silently
fall back to SysMem staging (~15× slower for PCIe allreduce). See the
[RTX PRO 6000 wiki](https://github.com/local-inference-lab/rtx6kpro) for
full PCIe topology and tuning guidance.

**Using docker-compose.** Create a `docker-compose.yml`:

```yaml
services:
  glm53-candidate:
    image: ${IMAGE:?Set an immutable published candidate image}
    container_name: glm53-342-candidate
    restart: "no"
    ipc: host
    network_mode: host
    ulimits:
      memlock: -1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - MODEL_PROFILE=glm53-3.42bpw-500k
      - PORT=8001
      # - MODEL_PROFILE=glm53-3.42bpw  # historical 393216-token full-model profile
      # - HF_TOKEN                       # inherit a silently supplied environment value
      # - PREFIX_CACHE_DISK_GB=32         # LMCache L2 disk tier (GiB)
      # - PREFIX_CACHE_DISK_FRACTION=0.5  # ...or fraction of free disk
    volumes:
      - /srv/turnkey-glm53-candidate:/workspace
```

Then:

```bash
docker compose up -d
docker compose logs -f
```

For rootless Podman against a checkpoint already on disk — no download, the
checkpoint mounted read-only — use the checked-in runner:

```bash
export MODEL_PROFILE=glm53-3.42bpw-500k
export GPU_DEVICE_MODE=manual  # Podman 4.9: explicit devices, not newer CDI syntax
export NAME=glm53-342-candidate
export PORT=8001
export RESTART_POLICY=no
export MODEL_DIR_HOST=/path/to/complete/checkpoint
export DOWNLOAD_MARKER_HOST=/path/to/flags/.download-complete
bash scripts/run-local-podman.sh
```

Cache volumes, draft/vision mounts, and the LMCache NVMe tier are documented
in [Running it on your own hardware](#running-it-on-your-own-hardware). Both
paths have a dry run that prints the resolved vLLM argv and exits without
downloading anything or touching a GPU: add `-e CONFIG_SMOKE=1` to the Docker
command, or prefix the runner with `CONFIG_SMOKE=1`.

## Model profiles

A profile is a complete set of compatible defaults, not just a model name.
Changing only `MODEL_DIR` is unsafe because quantization, topology, attention
backend, parsers, speculation, vision handling, and KV sizing also differ.

| `MODEL_PROFILE` | intended use | default hardware | download / context |
|---|---|---|---|
| **`glm53-3.42bpw-500k`** | **next-release default; >=500K spot retrieval passed, full maintenance matrix open** | 4x RTX PRO 6000 Blackwell 96 GB | ~331 GiB / 520,192 total-token limit |
| `glm53-3.25bpw` | optional lower-bit capacity experiment, NOT GPU-qualified | 4x RTX PRO 6000 Blackwell 96 GB | ~317 GiB / planned 524,288 total tokens |
| ~~`glm53-k6`~~ / ~~`glm53-k8`~~ | GLM-5.3-Flash: **refused** on this base; served by the VerdictAI-derived lineage `ghcr.io/malaiwah/glm52-exl3-vast@sha256:9d7ab60a…` | 4x RTX PRO 6000 Blackwell 96 GB | n/a here |
| `glm53-3.42bpw` | historical full-model mixed K3/K4 EXL3 qualification; not transferred to refreshed GG image | 4x RTX PRO 6000 Blackwell 96 GB | ~331 GiB / 393,216 |
| `glm52-exl3` | validated GLM-5.2 production stack | 4x RTX PRO 6000 Blackwell 96 GB | ~309 GiB / 512K |
| `qwen36-27b-nvfp4` | vision-enabled, lower-cost production/development | 1x RTX 5090 32 GB | ~21 GiB / 192K |
| `custom` | another conventional vLLM checkpoint | configurable | conservative 32K defaults |

### GLM-5.3 full-model 3.42bpw 500K: candidate to official release

The primary profile is `glm53-3.42bpw-500k`, variant
`exl3-tr3-glm53-3.42bpw-500k`, architecture family **`glm52`**. The family
reflects the unchanged full-model architecture, not the checkpoint version.
It pins
[`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@99c6f951333d2b38f1efefa533c7afadf0d376e3`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/tree/99c6f951333d2b38f1efefa533c7afadf0d376e3).
All 81 weight-file LFS SHA-256/size pairs match evaluated revision
`8bef807a0fcdd180e984a26b50e731cdba9a8ff2`. The newer chat template passed
the spot feature suite; the full parser/continuation maintenance matrix remains
open. Validate metadata against pinned HF Git object IDs; do not bypass stale
manifest entries indiscriminately.

The [header accounting](maintenance/glm53-aibeast-500k/memory-comparison.json)
compares every tensor in the live willfalco 5.2 and davidsyoung 5.3 3.42bpw
checkpoints. Their BF16 carrier and router-bias storage are identical.
The new quant has **0.848 GiB/rank** more tensor payload: approximately
0.665 GiB of per-expert rotation vectors and 0.182 GiB of Trellis data.
The index totals were not directly comparable: 5.2 counts shard headers and
5.3 counts only tensor bytes. This is quantization/layout overhead, not a
larger model architecture.

The live 5.2 boot recorded 81.73 GiB model loading and 0.75 GiB graphs/rank;
the older 5.3 qualification recorded 82.42 GiB and 0.61 GiB on a different
runtime. Those are not a controlled same-runtime A/B. The historical 5.3
520K failure does not establish that GLM-5.3 requires smaller workspaces on
the current runtime.

**AIBeast maintenance is parity-first, as explicitly requested by the user:**
try the live GLM-5.2 resource/tuning settings before reducing to the exercised
rental floor. The read-only baseline comes from production `Config.Env` and
the actual serving argv of `glm52-turnkey-r34-maint-20260815-v1`
([production baseline](maintenance/glm53-aibeast-500k/production-baseline.json)).

| resource | first trial: `MAINTENANCE_TRIAL=parity` (default) | explicit fallback: `MAINTENANCE_TRIAL=rental-floor` |
|---|---|---|
| sequences / scheduler tokens / EXL3 prefill arena | 12 / 3072 / 3072 | 8 / 2048 / 1024 |
| GPU memory utilization | 0.95 | 0.93 |
| maximum CUDA graph capture / Trellis maximum M | 48 / 48 | 32 / 32 |
| graph capture sizes | 4,8,12,16,20,24,28,32,36,40,44,48 | 4,8,12,16,20,24,28,32 |
| LMCache initial RAM arena | 125 GiB | 20 GiB |

Both trials preserve full **3.42bpw**, the pinned checkpoint, **520,192
total tokens**, TP4/DCP4, interleave 64, native probabilistic MTP3, online K6,
dynamic NVFP4 KV with FP8 RoPE, and `KV_CACHE_MEMORY_BYTES=4518907904` per GPU.
Both retain a 125 GiB LMCache RAM ceiling and 384 GiB disk tier; external
prefix caching does not enlarge active GPU context. No token-limit or precision
reduction, optional 3.25bpw substitution, or 750K claim is part of this policy.

The general `MODEL_PROFILE=glm53-3.42bpw-500k` defaults remain the exercised
rental floor; the maintenance stage overrides them for the parity first trial.
Only after the operator records a parity failure or insufficient measured
margin may they explicitly select `rental-floor`. There is **no automatic
shrink or fallback**. Each trial has its own default name, caches, state and
stage manifest; fallback appends `-rental-floor` to the parity name. The selector
is part of stage identity, not a way to silently retune an existing stage.

This policy **requires a new image build**: the prior image validator rejects
3072 scheduler/prefill settings. Image `200b1841…` below is GPU-exercised only
at the rental floor, not at parity. A newly built image needs its own recorded
immutable digest and parity GPU qualification; neither is supplied by the old
receipts. **The AIBeast maintenance window has not opened.**

The exercised image is
[`ghcr.io/malaiwah/glm52-exl3-vast@sha256:200b1841453b6a46c91f0b7a2866589cda7e52625b29590bb7a69fe90948e6c9`](https://github.com/malaiwah/glm52-exl3-vast/pkgs/container/glm52-exl3-vast),
built from [`e96231359fc5dca2df9d4a397d7a814ba9deae30`](https://github.com/malaiwah/glm52-exl3-vast/commit/e96231359fc5dca2df9d4a397d7a814ba9deae30)
by [CI 34168581945](https://github.com/malaiwah/glm52-exl3-vast/actions/runs/34168581945);
the release work is [PR #58](https://github.com/malaiwah/glm52-exl3-vast/pull/58).
On resumed JarvisLabs spot container **500157** (predecessor **483634**), four GPUs served a
501,086-token haystack with 3/3 facts in 361.42 s, followed by 512
temperature-1 output tokens in 7.76 s and a passing feature suite (vision
skipped). A 131,409-token repeated prefix fell from 62.4 s to 1.3 s using
native GPU cache only. An independent **501,098-token** pressure probe then
retrieved 3/3 facts in 360.891 s; replaying the original prefix took **3.115 s**
with **115,200 new external-cache hit tokens** and 15,872 new native-hit
tokens. This positively measures **LMCache DRAM retrieval after GPU pressure**,
not a pure DRAM-only request, L2/disk persistence, restart or fault recovery.
The image was unpacked and grafted into the provider container. Final
verification matched **19 critical sources, seven module initializers and seven
image-recorded native libraries**. Provider NCCL 2.23.4 files remained on disk;
live process maps showed the image's **NCCL 2.30.4** loaded. These are scoped
hash/load observations, not complete filesystem or OCI isolation/security
equivalence. Later source changes are not retroactively GPU-qualified.
The resumed instance **500157 was paused after evidence capture**, preserving
1200 GB of user storage (billable while paused). Subsequently, on the user's
explicit destruction request, the provider destroy API returned success for
500157. A subsequent `jl list` returned `[]` and all six resource counts were
zero ([separate destroy receipt](maintenance/glm53-aibeast-500k/evidence-spot-20260908/destroy.json));
the historical pause evidence remains unchanged.

The [maintenance assets](maintenance/glm53-aibeast-500k/) stage a separate
name, port, state and cache using Podman 4.9-compatible device mappings.
`start` refuses while production is running. Safe staging/CPU inspection uses
the newly built image's recorded `IMAGE=repository@sha256:…`, not the old
floor-only digest:

```bash
# Set IMAGE to the new published immutable digest before these commands.
: "${IMAGE:?Set IMAGE to the new parity-capable image digest}"
export IMAGE
MAINTENANCE_TRIAL=parity uv run --no-project --python 3.12 maintenance/glm53-aibeast-500k/candidate.py stage
MAINTENANCE_TRIAL=parity uv run --no-project --python 3.12 maintenance/glm53-aibeast-500k/candidate.py config-smoke
```

These commands do not stop production or open a GPU window. After a justified
parity failure only, use `MAINTENANCE_TRIAL=rental-floor` consistently for a
new isolated stage and its subsequent commands; never reuse parity state.
An authorized window must prove:

- exact published image without runtime-source mounts and all pinned model
  files verified;
- the initial cold engine and first temperature-1 sampler, without a hidden
  warmed retry; automatic startup verification is disabled only in this
  isolated qualification stage so the runner owns the first API sample;
- strict output, tools/continuation and tokenizer-exact multi-depth retrieval
  with **at least 500,000 real input tokens plus useful output** inside 520,192;
- C1/C4/C8 and measured long-prefill/active-decode overlap, with both stream
  timelines and physical memory/error evidence;
- LMCache cold/warm/L2 hits, bounded eviction, restart and failure recovery.

Deadline ownership is tested on the actual installed adapter using controlled
CPU futures; this is not a claimed 180-second GPU DMA-hang injection. Real
cache round trips and engine-group recovery remain separate GPU gates.
Rollback retains the old service/image/state and revokes candidate reboot
eligibility before stopping it. Neither staging nor qualification automatically
cuts over production or promotes `latest`.

### GLM-5.3 full-model 3.42bpw

`MODEL_PROFILE=glm53-3.42bpw` pins
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@8bef807a0fcdd180e984a26b50e731cdba9a8ff2`.
The **historical** live-qualified appliance pin is
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:6e2475d0568fd110eeaa1193157c7662747e096b476b05ed71ab247e081e9b82`.
The measurements in this subsection belong to that lineage, not the refreshed
Gilded r34 candidate.
This is the 755B `glm_moe_dsa` GLM-5.3 model, not GLM-5.3-Flash. Its complete
structural configuration matches GLM-5.2, so it reuses the qualified GLM-5.2
B12X sparse-MLA, DCP4, native MTP-3, mixed-K EXL3 and online-K6 runtime family.
The payload itself is not the old shared-H encoding: every routed layer stores
148 K3 and 108 K4 experts with per-expert transforms.

Qualification rejected the inherited 520,192-token/4.21-GiB KV envelope. It
passed startup and 32K retrieval, then OOMed in the first temperature-1 sampler
request with only 3 MiB free on one rank. The profile therefore pins a
3.18-GiB/GPU dynamic-NVFP4 pool and a 393,216-token request limit. The final
arm loaded 82.42 GiB of tensors per rank, used 0.61 GiB/rank for CUDA graphs,
and retained at least 665 MiB of observed physical free memory after first-use
compilation and C8 sampling.

Unique-prefix prefill measured 2,465 / 2,444 / 2,380 / 2,282 tok/s at
8K / 32K / 64K / 128K. Aggregate decode from a 1,024-token raw-prompt target
(1,027 API prompt tokens after chat templating) to 512 temperature-1 output
tokens measured 60.40 / 153.93 / 227.55 tok/s at C1 / C4 / C8; all 72 requests
completed without failure, preemption, or prefix reuse. The complete OpenAI
feature suite passed, as did 15/15 retrieval facts at five depths through
131,407, 261,192, and 389,959 tokenizer-exact document tokens. A post-stress
short/structured gate was clean. The cold log did record expected first-use
CuTeDSL/Triton JIT after readiness; its remaining structured-FSM, CUDA,
distributed, process-failure, and error-level audit categories were clean.
A corrected audit of the warmed line 3,030–3,469 window found zero findings in
all categories.

Post-qualification workflow review release
`55194ff329271ada376b691b2733ab35012cbd68` repeated the read-only,
no-source-mount boot on the same four-GPU rental with zero restarts. The
image-owned startup gate passed the three short checks, strict structured
output, a complete 512-token temperature-1 sample with a post-sample health
check, and 3/3 retrieval facts in a tokenizer-exact 32,853-token prompt. The
independent OpenAI feature suite also passed. Public port 8000 remained
available with the API key (401 without it, 200 with it). LMCache bound only
to `127.0.0.1:8089`, exposed only its six read-only information/version paths,
and had no credential-like variables; a synthetic SOUL launch completed the
post-hardening key handshake and reaped a detached child for the `soul` uid.

### GLM-5.3-Flash K6 and K8: a separate lineage, not this image

This image refuses `glm53-k6` and `glm53-k8`. Flash needs the GLM5Next model,
pooled-indexer and KV-pool warmup runtime, which the Gilded Gnosis base does
not contain; the VerdictAI-derived image
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:9d7ab60a3ad666edb8d38812ec6709909e6752a78fad464c842c7b17659f5d5b`
remains the Flash lineage. The measurements below are the retained evidence for
that lineage, kept because they set the family boundary this repository still
reasons about.

Those profiles pin checkpoint revisions
`be51877455a8786ebdd5f96053aff6dc74a0996f` (K6) and
`b5ef443adce36ba5a10f2d5aa682fc9f2f0d0fae` (K8). Do not transplant only a
model directory into a GLM-5.2 launch: Glm5Next routing, EXL3 encoding, sparse
MLA, Triton MoE, NOPE/index-pool compression, calibrated KV scales, topology,
and scheduler bounds are one contract. Both profiles intentionally disable
speculative decoding.

K6 uses the qualified B12X fused Trellis kernel and remains that lineage's
production kernel. K8 cannot safely widen that decoder: eight overlapping 16-bit MCG
windows span 72 bits, while its two-word path represents only 64. K8 therefore
uses ExLlamaV3's native K8 routed-expert kernel in eager mode with a 512-token
scheduler and parity arena.

On the same four-GPU host, K8 used 78.94–78.97 GiB/GPU for weights plus
non-torch allocations, 2.25 GiB for peak activations, no graph memory, and
7.10–7.14 GiB/GPU for KV. It exposed 6,610,733 logical KV tokens, or 14.41
maximum-length requests at the advertised cap. K6 used 63.74–63.78 GiB,
3.02 GiB, 0.45–0.46 GiB, and 21.52–21.56 GiB respectively, exposing
20,043,933 logical tokens. K8 therefore
adds about 15.2 GiB of non-KV memory and gives up about 14.4 GiB of KV per GPU.

K8 unique-prefix prefill measured 2,684 / 2,825 / 2,938 / 2,986 tok/s at
8K / 32K / 64K / 128K. Aggregate target-only decode measured
10.29 / 38.79 / 75.66 tok/s at C1 / C4 / C8 with a 256-token input,
8.42 / 23.43 / 28.46 at 32K, and 5.42 / 12.05 / 13.25 at 128K. No request
failed or preempted. K6 is 5.3–7.3× faster at short context and 8.3–24.4×
faster when the measured 32K/128K prefill cost is included.
K8's reason to exist is fidelity: its panel KLD is 0.012384 versus K6's
0.013723, at about 77 GB / 30% more checkpoint bytes.

The common 458,752-token cap is deliberate. K6 passed two independent 448K
trials at 449,461 and 449,462 document tokens. K8 passed two more independent
trials at 449,461 tokens, retrieving all three facts in 170.775 and 175.086
seconds; short and structured-output checks still passed afterwards. K6's
480K answer-budget failures and concurrent 505K corruption remain the family
boundary: the larger auto-profiled KV pools are concurrency capacity, not
evidence that a single 500K request is correct.

The Qwen profile serves
[`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
with `--quantization modelopt`, the `qwen3` reasoning parser, the
`qwen3_coder` tool parser, its native vision encoder, and a 192K-context
envelope qualified on one RTX 5090. The measured envelope, throughput, and
the MTP/speculation analysis are in
[docs/qwen36-qualification.md](docs/qwen36-qualification.md).

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

### GLM flagship variant matrix (beta)

The following matrix records measured GLM-5.2/GLM-5.3 alternatives; it does
not qualify the new full 3.42bpw reduced-workspace candidate above. `exl3-tr3` is the historical
GLM-5.2 provider-template selection. `exl3-tr3-max-context` trades ordinary-workload
speed for context experiments. `madeby561-hybrid` remains the immutable v20 control:

```text
MODEL_PROFILE=glm52-exl3
MODEL_VARIANT=madeby561-hybrid
```

Selecting a variant applies its entire coherent memory, transport and
speculation shape; it is not merely a different download URL:

| variant | topology / speculation | context and memory | intended use |
|---|---|---|---|
| **`exl3-tr3`** | TP4/DCP2, native/external TR3 MTP-5, probabilistic proposals | 524,288 max, 542,208-token cold r11 KV pool at GMU 0.957 on AIBeast, 3,072-token prefill batch, 140,000-token CKV gather, 1 GiB workspace, LMCache over 50% host DRAM | balanced flagship; default |
| `exl3-tr3-3.25bpw` | TP4/DCP4, native mixed-K TR3 MTP-3, probabilistic proposals; one-grid decode plus serial K3/K4 block-64 prefill | exactly 2,048 KV blocks / 524,288 logical tokens, GMU 0.957, 2,048-token scheduler with a reusable 1,024-row EXL3 arena, 64 MiB exact-fold budget, LMCache over 50% host DRAM | higher fidelity; ~22 GiB larger download and slower than the default |
| `exl3-tr3-3.36bpw` | TP4/DCP4, mixed checkpoint + online Trellis K6, native MTP-3; r26 exact query-split/full-CKV policy with two indexer shards and owner merge off | exactly 2,048 KV blocks / 524,288 logical tokens, GMU 0.957, 3,072-token scheduler/arena, 125 GiB LMCache DRAM + bounded 512 GiB NVMe | previous high-fidelity profile; dynamic-NVFP4 KLD 0.082507, 2,453--2,458 / 2,350--2,370 / 2,197--2,238 tok/s PP at 3K/32K/128K, 5/5 needles in an actual 522,359-token prompt |
| **`exl3-tr3-3.42bpw`** | TP4/DCP4, shared-H checkpoint + online Trellis K6, native MTP-3 with probabilistic proposals; r28 lossless query-split/full-CKV policy | exactly 2,032 KV blocks / 520,192 logical tokens, GMU 0.95, 3,072-token scheduler/arena, 125 GiB LMCache DRAM + bounded 512 GiB NVMe | selected high-fidelity profile; K6/dynamic-NVFP4 KLD 0.089888 (native weights 0.082039), PP 2,367 / 2,263 / 2,137 tok/s at 3K/32K/128K, complete 45/45 five-depth needles through a 516,096-token prompt plus 4,096-token reserve |
| **`exl3-tr3-glm53-3.42bpw`** | GLM-5.3 TP4/DCP4 per-expert mixed K3/K4 + online Trellis K6, native probabilistic MTP-3 | exactly 393,216 logical KV tokens, fixed 3.18 GiB/GPU dynamic-NVFP4 pool, GMU 0.93, 3,072-token scheduler, C8 | selected full 755B GLM-5.3 profile; use `MODEL_PROFILE=glm53-3.42bpw` |
| `exl3-tr3-max-context` | TP4/DCP4, native TR3 MTP-5 | 524,288 configured request limit, auto NVFP4 KV, GMU 0.98 | maximum-context experiments; slower for ordinary loads |
| `madeby561-hybrid` | TP4/DCP4, native serialized NVFP4 MTP-3 | exactly 2,048 KV blocks / 524,288 logical tokens, GMU 0.98, 2,048-token batch | immutable v20 control and alternate quant |

The 3.42 profile deliberately keeps dynamic NVFP4 MLA KV with FP8 RoPE. A
matched 2,047-position KLD arm improved from `0.089888` to `0.077949` when the
same K6 model used FP8 KV with BF16 RoPE, and the healed Aider run also favored
FP8. That higher-fidelity KV record is 656 bytes/token versus 368 for NVFP4 and
reduced the practical serving envelope to about 295K. FP8 is therefore a useful
quality/latency experiment, not the flagship default: it cannot meet the
520K-context Hermes Agents requirement on four 96 GB cards.

An independent Terminal-Bench 2.1 reproduction on the Brandon checkpoint
scored within 2.6 points of Z.ai's vendor result. GG r17 introduced the native
[bounded, shape-aware mixed-K implementation](https://github.com/local-inference-lab/vllm/pull/222)
and the [complete custom-PCIe package](https://github.com/local-inference-lab/sparkinfer/pull/105);
the appliance does not stack the superseded #210/#219 overlays. The r17
AIBeast production gate retained exactly 524,288 active tokens, passed the
full API suite, and recovered 5/5 needles at both 256K and an actual
521,276-token prompt without degeneration. A clean production restart reused
all 27 first-use kernels from disk and repeated the maximum-context gate with
a second seed. The exact r14/r17 A/B and the
NCCL/B12X route campaign are in
[docs/glm52-r17-maintenance-results.md](docs/glm52-r17-maintenance-results.md).
The broader evidence —
feature gates, KLD measurements, LMCache qualification, and release
boundaries — is in
[docs/glm52-qualification.md](docs/glm52-qualification.md) and
[docs/glm52-3.25-offload-qualification.md](docs/glm52-3.25-offload-qualification.md);
the r20/3.36 campaign, including the mixed-tier cache-key repair, is in
[docs/glm52-r20-3.36-qualification-plan.md](docs/glm52-r20-3.36-qualification-plan.md).
The r25 qualification of SparkInfer #117, plus the matched NVFP4-versus-FP8
KV/KLD comparison, is in
[docs/glm52-r25-3.36-qualification.md](docs/glm52-r25-3.36-qualification.md);
the r26 policy A/B, turnkey integration repair, and repeated 512K production
gate are in
[docs/glm52-r26-3.36-qualification.md](docs/glm52-r26-3.36-qualification.md);
the r28 shared-H 3.42-bpw capacity, KLD, API, and 45-cell maximum-context gate
are in
[docs/glm52-r28-3.42-qualification.md](docs/glm52-r28-3.42-qualification.md);
cross-provider throughput, power, and loader tables are in
[docs/benchmarks.md](docs/benchmarks.md).

## What startup looks like

Plan for three separate stages: image pull, roughly 309--328 GiB of weights
depending on the GLM variant, then model load/calibration/compile. The
dashboard and provider status can look idle during any one of them.

| environment | measured first click → `/health` | what dominated | practical first-use budget |
|---|---:|---|---:|
| AIBeast GG r28, 3.42-bpw NFS checkpoint + online-K6/JIT cache reused | 12--15.5 min | 5--7 min NFS shard stream plus mixed-Trellis hydration; online-K6 artifacts were cache hits | 17 minutes |
| AIBeast GG r28, 3.42-bpw cold online-K6 encode | ~89 min | ~64 min weight/K6 pass plus ~25 min mixed-Trellis hydration | 90–105 minutes; health start grace is 90 minutes |
| AIBeast GG r26, 3.36-bpw NFS checkpoint + online-K6/JIT cache reused | ~12 min to `/health`, ~13 min through verification | ~4 min NFS shard stream plus ~7 min mixed-Trellis hydration; online-K6 and CuTe artifacts were cache hits | 15 minutes |
| AIBeast GG r17, safetensors + compatible r17 AOT reused | 12–14 min | ~252s shard load plus ~7 min mixed-Trellis hydration; AOT reconstruction under 1s | 15 minutes |
| AIBeast GG r20, 3.36-bpw online K6 cache reused | ~5m10s | 79s shard load, ~45s mixed-Trellis hydration, compile/profile/graphs and the built-in 32K correctness gate | 6 minutes |
| AIBeast GG r11, safetensors, cold patched EXL3/AOT cache | 9m46s | 91s weight load plus cold extensions, AOT, profiling and graph capture | 10–12 minutes |
| AIBeast GG r11, same-stack AOT reused during heavy storage traffic | 6m22s | NFS/cachefilesd contention while another checkpoint populated the RAID0 cache; backbone AOT loaded in 0.55s | 7 minutes |
| AIBeast GG r5, weights/NFS pages cached, fresh AOT | 4m35s | InstantTensor load, compile and graph profile/capture | 5 minutes |
| AIBeast GG r5, compatible AOT reused | 2m02s–2m25s | weight page-in, draft compile and graph capture | 3 minutes |
| Vast Community, image cached but weights absent | 55 minutes | ~48-minute HF transfer at ~0.9 Gbit/s; post-download engine ~6 minutes | 60–90 minutes |
| Runpod Secure, image and weights absent | 25m13s | image pull 8m24s, HF transfer 3m35s, load/compile/calibration | 30 minutes |
| Runpod Community | not a single stable class | host registry/HF/storage route | 30–90 minutes; enforce a cost deadline |
| JarvisLabs IN1 VM, image and weights absent | ~24 minutes from measured stages | ~7-minute image pull, ~10-minute HF transfer, 6m57s first successful engine/TLS start | 30 minutes |
| JarvisLabs IN1 VM, checkpoint/AOT/cert reused | 3m22s | 60-second InstantTensor load, 4.9-second cached compile, graph capture and DRAM tier allocation | 4 minutes |

During post-start verification, one loopback request intentionally calls
`GET /v1/models` with a known-wrong key. Its `127.0.0.1 ... 401 Unauthorized`
access-log line is a passing authentication test, not a failed health probe;
the boot log prints that explanation immediately before the request.

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
The linked public **Model Turnkey: GLM-5.2 EXL3** template
preconfigures the image, `args` launch mode, ports, 600 GB disk and Blackwell
host filters.
Before accepting an offer, verify it is exactly 4x RTX PRO 6000 Blackwell,
advertises **CUDA 13.2 or newer / driver 590.48.01 or newer**, adequate
disk/network performance, and actually allocates at least 600 GB. Wait for
`Application startup complete` and the verification result in the instance
logs, then use the generated API key and labeled endpoint.

For the one-GPU alternative, use
**[▶ Qwen3.6-27B NVFP4 Vision on Vast.ai](https://cloud.vast.ai/?ref_id=386667&template_id=214d2e120a6718558fa207d4579d4316)**.
It selects the same credential-free image in Docker `args`/ENTRYPOINT mode,
one RTX 5090, 100 GB of disk, and `MODEL_PROFILE=qwen36-27b-nvfp4`.

> **Do not select Vast's SSH launch mode.** The appliance starts its own
> key-only SSH service. Vast SSH mode replaces the image entrypoint, so the
> dashboard, model download, verifier, and vLLM never start. Use the linked
> template or an `args` template with no start command. This applies to both
> the GLM and Qwen profiles.

For a direct CLI launch, explicitly select that mode by putting an empty
`--args` at the very end of the command (the Vast CLI otherwise defaults to
SSH mode):

```bash
vastai create instance <offer-id> \
  --image ghcr.io/malaiwah/glm52-exl3-vast:latest \
  --disk 600 --label glm52-turnkey \
  --env '-p 22:22 -p 8000:8000 -p 8443:8443 -p 1111:1111 -e MODEL_PROFILE=glm52-exl3' \
  --cancel-unavail --args ''
```

`--args` consumes every remaining CLI token, so nothing may follow it. Check
`image_runtype=args` in `vastai show instance <id> --raw` before waiting for a
large checkpoint download.

### Low-cost first run: Qwen + Oh My Pi

The Qwen template is the quickest way to learn the complete appliance flow
before renting the four-GPU flagship. A live July 29 qualification on a Vast
RTX 5090 reached the verified API in about six minutes on a well-connected
host: roughly two minutes for the appliance image, 57 seconds for the 21 GiB
checkpoint, then model load, compile/autotune and the long-context gate. On a
different Community host, the image had not completed a single new layer after
20 minutes; that rental was released. Treat visible completed layers—not an
advertised network number or an animated `loading` line—as progress, and set a
cold-start cost deadline before clicking Rent.

For a secure first run, configure `DESEC_TOKEN` in Vast's account environment
and `DESEC_DOMAIN=yourname.dedyn.io` in a private clone of the template before
launch. Open the tokenized dashboard from the instance label/logs and wait for:

1. **Weights: ready**
2. **TLS / DNS: `https://model-<instance-id>.<domain>`**
3. **Engine: serving**
4. **Correctness: verified including long context**

If TLS / DNS still says `not configured`, the direct Vast endpoint is plain
HTTP. Do not send its bearer key over the public Internet. Fix the template and
relaunch, or use the encrypted SSH-tunnel route documented below.


Install [Oh My Pi](https://github.com/can1357/oh-my-pi) and copy the **Oh My
Pi (OMP 17+)** YAML shown by the secure landing page. Concurrency limits,
review scoping, and the measured client-timeout guidance are in
[docs/qwen-omp-guide.md](docs/qwen-omp-guide.md).

The dashboard keeps a short client-side history of prompt/generation
throughput, running/waiting requests, KV pressure and prefix-cache hits. Boot
highlights are UTC timestamped so a first-time user can distinguish real
progress from a stalled pull or warm-up. This capture is the secure Vast
qualification while OMP was reviewing this repository:

![Qwen3.6 27B Vast dashboard under bounded OMP load](docs/images/qwen36-vast-omp-live-dashboard.png)

## Launch on Runpod

Choose **[▶ GLM-5.2 EXL3 on Runpod](https://console.runpod.io/deploy?template=f8sgtc6orf&ref=4ahycj93)**
for the four-GPU flagship, or
**[▶ Qwen3.6-27B NVFP4 Vision on Runpod](https://console.runpod.io/deploy?template=7ufac3b4zw&ref=4ahycj93)**
for the qualified one-GPU profile. Both links select public Pod templates and
include the project referral.

This image is a **Runpod Pod** template, not a Serverless worker or Hub
application. It runs a persistent OpenAI-compatible service and does not
implement Runpod's Serverless handler contract.

> **Blackwell is required.** The pinned CUDA/vLLM image and its custom kernels
> are built for `sm120+`. Use an RTX 5090 or RTX PRO 6000 Blackwell; an RTX
> 4090 is Ada-generation (`sm89`) and is not a supported appliance target even
> when the selected model would otherwise fit its VRAM.

The checked-in manifests follow Runpod's current
[Pod template REST schema](https://docs.runpod.io/pods/templates/manage-templates):

- [`runpod-template.json`](runpod-template.json): GLM profile, 600 GB volume.
- [`runpod-template-qwen36.json`](runpod-template-qwen36.json): vision-enabled
  Qwen profile, 100 GB volume.

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
  or one RTX 5090 32 GB for the Qwen manifest. GPU type/count
  are selected at Pod deployment and are not fields in the reusable template
  schema. Do not select RTX 4090 or another pre-Blackwell GPU. After Runpod
  assigns the host, confirm CUDA 13.2 or newer and driver 590.48.01 or newer in
  the system logs; its
  REST API's `allowedCudaVersions` currently stops at CUDA 13.0 and cannot
  express this CUDA 13.2 requirement.
- **Storage:** use a 50 GB container disk; mount at least 600 GB for GLM or
  100 GB for Qwen at `/workspace`. A volume disk survives stops/restarts but is
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

## Launch on JarvisLabs

Choose **[▶ RTX PRO 6000 Blackwell VM on JarvisLabs](https://jarvislabs.ai/dashboard/vm)**,
then select exactly four `RTX-PRO6000` GPUs, at least 650 GB of disk, and the
current Ubuntu VM image. JarvisLabs' managed Templates page is a fixed
provider catalog; as of July 2026 its documentation and authenticated
dashboard expose neither user-published Docker templates nor self-service
referral links. The direct VM link is therefore the honest launch link—there
is no template/referral id to append.

The live inventory is visible without renting anything:

```bash
uv tool install jarvislabs
jl setup
jl gpus
```

JarvisLabs documents the current
[`jarvislabs` 0.2.x CLI](https://docs.jarvislabs.ai/cli), its separate
[VM/container template catalog](https://docs.jarvislabs.ai/templates/), the
[Python SDK/API surface](https://docs.jarvislabs.ai/sdk), the
[pause/destroy lifecycle](https://docs.jarvislabs.ai/getting_started), and
[current pricing](https://jarvislabs.ai/pricing). The CLI's `jl gpus --json`
is the source of truth for stock because availability differs between VM and
container workloads.

The historical on-demand VM quote was `$1.89/GPU-hour`, or `$7.56/hour`
for four cards. The 2026-09-08 proof instead resumed the pre-existing
**483634** container as **500157**, with **four spot RTX PRO 6000 GPUs**, at an approximate
**$3.96/hour GPU quote**, retaining its 1200 GB `/home` storage. Stock and
pricing remain live values, not promises. A VM supports Docker; the spot
container path below unpacks a custom image without a container runtime.
**500157 was paused after evidence capture, preserving the user's existing
storage, then explicitly authorized for destruction; its destroy API returned
success; `jl list` then returned `[]` and all six resource counts were zero
([destroy receipt](maintenance/glm53-aibeast-500k/evidence-spot-20260908/destroy.json)).** The pause
receipt remains historical: pausing released GPU compute but retained billable
storage. Resume changed the machine ID; always resolve the current ID with
`jl list` before lifecycle actions rather than using predecessor 483634.

A managed-container probe was also completed rather than merely inferred from
the catalog. Its four RTX PRO 6000 GPUs had peer reads/writes between every
pair, and its 1.2 TB `/home` volume persisted, but `/workspace` lived on the
ephemeral root filesystem. The template supplies neither Docker nor Podman,
keeps `CAP_SYS_CHROOT` and `CAP_MKNOD` but drops `CAP_SYS_ADMIN`, and blocks
user namespaces, so no mount, no bind and no unprivileged namespace is
available: a chroot there cannot be given `/proc` or `/dev`.
`scripts/jarvislabs_container_rootfs.sh` therefore takes the remaining honest
route. It fetches the published digest with `skopeo`, unpacks it with `umoci`
(pinned by hash, so whiteouts and opaque directories are handled correctly),
then *grafts* the image over the rented container - the appliance's `/opt`
trees by symlink, the OS layer including its newer glibc by `rsync`, with the
runtime-injected driver files excluded so the host driver keeps winning - and
runs the real entrypoint in the container's own namespace, where `/proc`,
`/sys`, `/dev/nvidia*` and a 320 GiB `/dev/shm` already exist. Before anything
launches, the helper checks the scoped installed runtime provenance. In the
exercised graft, final verification matched 19 critical source hashes, seven
module-initializer hashes and ten recorded native-library hashes. Extra
provider NCCL 2.23.4 files were retained and disclosed; live process maps showed
image NCCL 2.30.4 in use. This is **not** exact OCI-runtime, filesystem or
security qualification: provider namespace, mounts and host driver remain.
The graft irreversibly mutates the provider OS, so it demands `TURNKEY_GRAFT=1`;
use only a specifically authorized container and preserve persistent user data.
The current source helper additionally checks recorded NCCL/ExLlama native
libraries; that check passed against the running graft before pause, not as
proof of a fresh GPU boot with every later helper change.
It requires a container marker and canonical dedicated `TURNKEY_ROOT` and
`TURNKEY_WORKSPACE` paths without symlink components, below `/home`, `/mnt`,
`/srv`, `/tmp` or `/var/tmp`. The workspace cannot be inside the root's
`bundle` or `oci` directories. `/opt` runtime targets plus `/workspace`,
`/cache` and `/state` must be absent or already matching links. Conflicting
directories or links are
refused rather than deleted. Repeating the same-image graft verifies without
rewriting; do not remove provider/user paths to force admission.
Its `stop` command refuses to shut down a grafted provider container: stop the
launch supervisor/process group deliberately, then pause the reused instance.

The qualified IN1 VM reported four same-NUMA `PHB` cards but no CUDA peer
reads or writes between any pair. The unmodified pre-Jarvis image reached
model warmup and then failed its SparkInfer DCP all-gather. The current
appliance detects that capability boundary before calibration, disables B12X
PCIe DMA/DCP A2A, and serves through the lossless NCCL/shared-memory fallback.
That is why this shape passes the full 517K quality gate but trails all-`NODE`
hosts in the performance table. Driver `595.58.03`, CUDA 13.2, Ubuntu 24.04,
and the 600 W/card power ceiling are part of the measured result; a different
Jarvis host remains a fresh topology gate.

<details>
<summary><b>JarvisLabs full VM + Docker (the measured flagship path)</b></summary>

Create and connect with the current CLI:

```bash
jl create --gpu RTX-PRO6000 --vm --num-gpus 4 --storage 650 \
  --region IN1 --name glm52-turnkey --yes
jl list
ssh ubuntu@<public-ip>
```

On the VM, export the numeric id and region shown by `jl list`. Add the two
optional download/TLS credentials without putting them in a shared script or
shell history, then run the checked-in launcher:

```bash
export JARVISLABS_MACHINE_ID=<numeric-id>
export JARVISLABS_REGION=IN1
export DESEC_DOMAIN=<your-zone>.dedyn.io
read -rsp "Hugging Face token (optional): " HF_TOKEN; export HF_TOKEN; echo
read -rsp "deSEC token (optional): " DESEC_TOKEN; export DESEC_TOKEN; echo

curl -fsSL \
  https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/main/scripts/jarvislabs_vm_bootstrap.sh \
  | bash
```

The launcher writes credentials to a mode-0600 env file, stores weights and
compile caches under persistent `/home/turnkey`, pulls the appliance, and
starts it with host networking, host IPC, all GPUs, and unlimited memlock.
Follow first boot with:

```bash
sudo docker logs -f glm52-turnkey
```

JarvisLabs VMs expose their public IP directly; there is no managed HTTPS
proxy. Configure deSEC for trusted TLS on
`https://model-<machine-id>.<zone>:8000/v1` and the tokenized dashboard on
`:1111`. Without DNS credentials, use an SSH tunnel rather than sending the
bearer key over public HTTP:

```bash
ssh ubuntu@<public-ip> -L 8000:localhost:8000 -L 1111:localhost:1111
```

The launcher always creates `/home/turnkey` before binding it to the
container's `/workspace`. The OCI image is weights-free; the public checkpoint
downloads into that persistent directory on first boot and is not captured in
a VM image.

</details>

<details>
<summary><b>JarvisLabs container instance (spot) without any container runtime</b></summary>

The exercised path resumed predecessor **483634** as **500157**, not a new rental.
Do not create a second instance for that workflow. For a separately authorized
new disposable rental, the creation shape is:

```bash
jl create --gpu RTX-PRO6000 --num-gpus 4 --storage 700 --spot \
  --region IN1 --http-ports 8000,1111 --name glm53-500k --yes
jl list                     # take the machine id, status and public IP
ssh -o StrictHostKeyChecking=no root@<public-ip>
```

On the instance, downloads and state live under persistent `/home`; grafting
also changes the ephemeral provider OS. The helper in this checkout includes
source changes newer than the exercised `e9623135` image; record its revision
separately and do not treat the image receipt as proof of those later changes.
Use a reviewed checkout of PR #58 to supply the helper rather than a moving
`main` script:

```bash
export TURNKEY_IMAGE=ghcr.io/malaiwah/glm52-exl3-vast@sha256:200b1841453b6a46c91f0b7a2866589cda7e52625b29590bb7a69fe90948e6c9
export TURNKEY_ROOT=/home/turnkey TURNKEY_GRAFT=1
# Copy scripts/jarvislabs_container_rootfs.sh from the reviewed checkout
# to /home/turnkey/rootfs.sh before running these commands.
bash /home/turnkey/rootfs.sh prepare   # skopeo fetch + umoci unpack
bash /home/turnkey/rootfs.sh graft     # install over the container, then verify
MODEL_PROFILE=glm53-3.42bpw-500k SSHD=0 \
  bash /home/turnkey/rootfs.sh smoke   # GPU-free resolved-argv check
read -rsp "Hugging Face token (optional): " HF_TOKEN; export HF_TOKEN; echo
MODEL_PROFILE=glm53-3.42bpw-500k SSHD=0 \
  setsid nohup bash /home/turnkey/rootfs.sh run >/home/turnkey/serve.log 2>&1 &
```

Use a silent prompt as above or a permission-restricted credential file, never
literal token values in command arguments, examples, logs or shell history.
Environment values are still visible to privileged processes and the provider.
Spot instances can be reclaimed at any time, so copy evidence off-host as it is
produced. **The resumed 500157 is now paused, with its old storage preserved.**
Storage continues while paused (the recorded `$0.00014/GB-hour` rate is about
`$4/day` for 1.2 TB; check current billing). Destroy only a separately created
disposable rental after explicit authorization to delete its data.

</details>

JarvisLabs does not inject a VM-scoped API key. Appliance self-termination is
therefore off by default. The provider dashboard is safest; advanced users may
set `TERMINATE_ENABLED=1` and
`JARVISLABS_TERMINATE_API_KEY` before launching, understanding that this is an
account-scoped credential. The landing page identifies that distinction,
requires the exact VM id plus acknowledgement, and issues a destroy—not a
pause—after the optional session erase.

## Running it on your own hardware

Stage the replacement independently; a running endpoint is not a disposable
container. Keep its image, launch environment, state and cache for rollback.

<details>
<summary><b>AIBeast / owned Linux host + rootless Podman</b></summary>

The checked-in runner preserves the same appliance entrypoint used by rentals.
Point it at an existing read-only Hugging Face checkpoint, a writable cache,
and a tiny writable flag directory. No weights are downloaded or mutated:

```bash
: "${IMAGE:?Set IMAGE to the published candidate repository@sha256:digest}"
export IMAGE
export MODEL_PROFILE=glm53-3.42bpw-500k
export MODEL_DIR_HOST=/mnt/vault/llm/huggingface/hub/\
models--davidsyoung--GLM-5.3-EXL3-TR3-3.42bpw/snapshots/\
99c6f951333d2b38f1efefa533c7afadf0d376e3
export DOWNLOAD_MARKER_HOST=/mnt/fast/glm53-candidate-flags/.download-complete
export NAME=glm53-342-candidate
export CACHE_VOLUME=glm53-candidate-cache
export STATE_VOLUME=glm53-candidate-state
export PORT=8001
export GPU_DEVICE_MODE=manual
export RESTART_POLICY=no

bash scripts/run-local-podman.sh
```

The runner uses host networking/IPC, passes the NVIDIA and DRI devices,
mounts the checkpoint read-only, and keeps compilation output outside the
checkpoint. Podman 4.9 uses manual NVIDIA/DRI devices; `GPU_DRM_DEVICES`
can specify the inventoried DRM render devices. The generic runner refuses an
existing name by default. `REPLACE_EXISTING=1` plus a fresh `ROLLBACK_NAME`
explicitly stops and renames that named container rather than deleting it;
do not use this as an implicit production cutover. Prefer the maintenance
assets for AIBeast: their `start` refuses while production runs, and `rollback`
stops only the candidate before restarting the preserved old service.
Never reuse old state/cache volumes for the candidate: stored overrides win
over new defaults, and candidate writes would contaminate rollback.
To print the resolved configuration without running the GPU service:

```bash
CONFIG_SMOKE=1 bash scripts/run-local-podman.sh
```

LMCache DRAM is selected by the GLM profiles; historical GLM-5.2 offload
measurements are not full GLM-5.3 qualification. A positive
`PREFIX_CACHE_DISK_GB` additionally stores bounded derived KV under the
writable LMCache mount. On an owned host, create a dedicated local NVMe
directory and bind it at the appliance's secure-erase-aware path:

```bash
mkdir -p /mnt/fast/lmcache/glm53-candidate
export LMCACHE_DISK_HOST=/mnt/fast/lmcache/glm53-candidate
export PREFIX_CACHE_BACKEND=lmcache
export PREFIX_CACHE_DISK_GB=512
bash scripts/run-local-podman.sh
```

The `PREFIX_CACHE_DISK_GB` limit is enforced by LMCache even when the backing
filesystem is larger. Do not point this path into the read-only checkpoint;
cached KV may contain session material, and secure termination only provides a
best-effort erase on flash storage. The 512 GiB setting needs at least that much
free local space plus operational margin; AIBeast had 748 GiB free before the
qualification. The tier does not preallocate its limit.

LMCache's internal HTTP listener is fixed to `127.0.0.1`; the production image
registers only its read-only information router. Administrative `/env`,
`/run_script`, cache mutation, quota, configuration, and backend-reconfiguration
routes are not present, and credential-like environment variables are removed
from the LMCache process. This boundary is independent of authentication on the
public model API at port 8000.

</details>

## Self-service profile switching

The serve line is no longer GLM-only. A **family** supplies the architecture's
engine flags, applicable knobs and validation rules; the config layer resolves
`defaults < family < variant < startup env < state file` inside it. The provider-facing
`MODEL_PROFILE` names map to the editable families:

| startup profile | self-service family / variant |
|---|---|
| `glm53-3.42bpw-500k` (candidate default) | `glm52` / `exl3-tr3-glm53-3.42bpw-500k` |
| `glm53-3.25bpw` | `glm52` / `exl3-tr3-glm53-3.25bpw` |
| `glm53-3.42bpw` | `glm52` / `exl3-tr3-glm53-3.42bpw` |
| ~~`glm53-k6`~~ / ~~`glm53-k8`~~ | withdrawn with the `glm53` Flash family; refused, not remapped |
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

- **Inheritance**: built-in defaults < family < variant < startup environment
  < a JSON state file on the volume. The file wins on purpose: template env
  cannot be edited after launch, so the page has to be able to override it.
  The file stores only what you actually changed.
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

Requires `OPEN_BUTTON_TOKEN` in the template environment on Vast. Runpod and
JarvisLabs generate and persist a token automatically when none is supplied.
Without a token, the editor is not exposed.

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
- **vast.ai, RunPod, and JarvisLabs**, auto-detected (`TERMINATE_PROVIDER`
  overrides). Vast
  normally injects the instance-scoped `CONTAINER_API_KEY`; an explicitly
  supplied account `VAST_API_KEY` is also supported. An
  unrecognised provider says so and points at the dashboard instead of failing
  obscurely. On RunPod the injected pod-scoped key has terminated its own pod
  on some deployments and has been refused on another; the appliance therefore
  tries REST, GraphQL, and both current and legacy `runpodctl` delete commands.
  `RUNPOD_TERMINATE_API_KEY` supplies an account key when the scoped key lacks
  delete permission. JarvisLabs VMs provide no instance-scoped key; opt-in
  self-termination requires the numeric machine id, region, and a deliberately
  supplied account key, and the confirmation page warns about that broader
  credential.
- **Typed confirmation**: you type the instance id, plus an explicit
  acknowledgement checkbox. No single click can destroy anything.
- **`TERMINATE_DRY_RUN=1`** runs the whole flow and shows the request it would
  have sent, without sending it.
- **Session erase** (a checkbox, unchecked): overwrites and unlinks the API key,
  TLS private key, config state, every log this template writes (prompts
  included), shell history, SSH material, provider/HF credentials, and anything
  you added under the model dir. **The public model weights are deliberately not
  erased** — the checkpoint is downloadable by anyone, so overwriting the full
  checkpoint hides nothing. Optional RAM and VRAM zeroing. Read the limits in the design
  note: on SSDs with wear levelling, on overlay filesystems and on network
  volumes an overwrite does not guarantee the old bytes are unreachable, and the
  instance console log in your dashboard is outside our reach entirely.

> **RunPod, two things that bite by default:**
> 1. **Expose the ports when you create the pod** —
>    `--ports "22/tcp,1111/http,8000/http,8443/tcp"`.
>    A pod created without them comes up with `ports: null`: the container runs,
>    but the landing page, the API and even SSH are unreachable, and a running
>    pod's ports cannot be changed. You would have to destroy it and re-download
>    the checkpoint — ~21 GiB for Qwen or ~309 GiB / 332 GB for GLM. (Measured.)
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

The v20 qualification tables and the comparison against other published
GLM-5.2 vision merges are in
[docs/glm52-qualification.md](docs/glm52-qualification.md).

## GLM profile: MTP78 draft

Full GLM-5.3 profiles use the checkpoint's native EXL3/TR3 MTP draft with
MTP3 and reject GLM-5.2 graft/override paths. For the explicit GLM-5.2 Brandon
profile, the native quantized layer-78 draft retains its historical MTP5
qualification. `MTP78_TRELLIS=0` selects `MTP_DRAFT=native`, not BF16;
use the registry's `MTP_DRAFT` selector rather than inferring dtype from this
legacy boolean.
The 4-arm draft comparison, speculation-depth measurements, historical graft
evidence, and the experimental separate-draft override are in
[docs/mtp78.md](docs/mtp78.md); every non-default flag is justified in
[docs/glm52-tuning-rationale.md](docs/glm52-tuning-rationale.md).

## Vast.ai template settings (manual setup)
- **Image**: an immutable published candidate digest for qualification; a
  qualified release digest for production. Do not promote the candidate to
  `latest` before the GPU gate. The GHCR package must be pullable by the host.
- **Launch mode**: docker ENTRYPOINT (vLLM logs appear on the instance console;
  the image starts its own key-only SSH daemon)
- **Docker options**: `-p 22:22 -p 8000:8000 -p 1111:1111`. Vast's current
  [Docker Options documentation](https://docs.vast.ai/guides/instances/docker-environment#docker-create-options)
  accepts only ports, environment variables and hostname in this field;
  `--ipc` and `--ulimit` entries are ignored. Port 22 is SSH and port 1111 is
  the landing page.
- **Profile**: `MODEL_PROFILE=glm53-3.42bpw-500k` (candidate default);
  `glm53-3.42bpw`, `glm53-3.25bpw` and `glm52-exl3` are explicit alternatives,
  or choose `qwen36-27b-nvfp4` for the one-GPU vision model.
- **Disk**: >=600 GB for GLM (3.42bpw weights ~339 GB + image ~39 GB + JIT caches ~15 GB + margin); >=100 GB for Qwen.
- **GPU filter**: 4x RTX PRO 6000 Blackwell (96 GB) for GLM; one RTX PRO 6000
  Blackwell or RTX 5090 for Qwen.
- **Env (all optional)**: `HF_TOKEN` (authenticated download and higher
  applicable Hub rate limits; supply via silent prompt or protected file),
  `OFFLOAD_FRACTION` (aggregate host-DRAM prefix cache; GLM 0.5, Qwen 0),
  `MTP_TOKENS` (full GLM-5.3 3; Qwen 0), `MAX_NUM_SEQS`,
  `MAX_MODEL_LEN` (full 3.42 candidate 520192; historical full 3.42 393216;
  optional 3.25 experiment 524288; Qwen 196608),
  `VLLM_EXL3_PREFILL_CAPACITY` (GLM-only reusable EXL3 arena; the mixed
  3.25-bpw profile selects 1024 rows inside its 2048-token scheduler chunk;
  the r20/3.36-bpw K6 profile selects its qualified 3072/3072 PP-first shape),
  `SERVED_MODEL_NAME`,
  `MTP_DRAFT` (native for full GLM-5.3; `off` disables speculation),
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

## Configuration reference

The most common first-launch knobs:

| env | default | purpose |
|---|---|---|
| `MODEL_PROFILE` | `glm53-3.42bpw-500k` | full non-Flash 520K candidate; other models must be selected explicitly |
| `HF_TOKEN` | (unset) | authenticated downloads and higher Hub rate limits |
| `DESEC_TOKEN` / `DESEC_DOMAIN` | (unset) | turnkey TLS via deSEC DNS-01 (see [Security](#security)) |
| `OPEN_BUTTON_TOKEN` | provider-specific | exposes the `:1111` landing page and config editor |
| `MAX_MODEL_LEN` | profile-specific | full 3.42 candidate 520192 (>=500K spot retrieval passed; matrix open); historical full 3.42 393216; 3.25 experiment 524288; Qwen 196608 |
| `MTP_TOKENS` | full GLM-5.3 3 / Qwen 0 | speculation depth; GLM-5.2 variant-specific |
| `OFFLOAD_FRACTION` | 0.5 GLM / 0 Qwen | host-DRAM prefix cache (not active-context capacity) |
| `TERMINATE_ENABLED` | `0` | expose the in-container terminate control |
| `AUTH` | `key` | `none` only on a trusted private network |
| `CONFIG_SMOKE` | `0` | `1` prints the resolved argv and exits |

Every knob — including KV sizing, offload/memlock behaviour, verification,
SOUL autonomy, and termination — is documented with its default and rationale
in [docs/configuration.md](docs/configuration.md). A final supported `TUNE_*`
runtime environment layer is intended for advanced A/B experiments and is not
exposed in the self-service UI. It does not bypass family/profile safety
validation: full-model launches reject incompatible Flash flags and unsafe
KV/context combinations.

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

Relocated deep-dive records:

- [docs/benchmarks.md](docs/benchmarks.md) — cross-provider performance and
  power tables, loader matrix, driver/CUDA admission evidence
- [docs/glm52-qualification.md](docs/glm52-qualification.md) — flagship
  feature gates, KLD, LMCache, vision, and the r14 field repair
- [docs/qwen36-qualification.md](docs/qwen36-qualification.md) — Qwen 192K
  envelope and speculation analysis
- [docs/mtp78.md](docs/mtp78.md) — MTP layer-78 draft measurements
- [docs/glm52-tuning-rationale.md](docs/glm52-tuning-rationale.md) — per-flag
  deviation ledger
- [docs/configuration.md](docs/configuration.md) — complete environment
  reference
- [docs/glm52-r14-maintenance-plan.md](docs/glm52-r14-maintenance-plan.md),
  [docs/glm52-r17-maintenance-results.md](docs/glm52-r17-maintenance-results.md),
  [docs/glm52-r20-3.36-qualification-plan.md](docs/glm52-r20-3.36-qualification-plan.md),
  [docs/glm52-r25-3.36-qualification.md](docs/glm52-r25-3.36-qualification.md),
  [docs/glm52-r26-3.36-qualification.md](docs/glm52-r26-3.36-qualification.md) —
  per-release GLM-5.2 maintenance and qualification campaigns
- [docs/glm52-3.25-offload-qualification.md](docs/glm52-3.25-offload-qualification.md)
  — 3.25-bpw active-KV and DRAM/NVMe offload qualification
- [docs/glm52-nccl-b12x-jarrel-maintenance-plan.md](docs/glm52-nccl-b12x-jarrel-maintenance-plan.md)
  — Jarrel quant comparison and NCCL/B12X recalibration plan

## Security

**Threat model honestly stated:** a rented host's operator has root — memory,
VRAM, and traffic on the box are visible to a determined host. These controls
are the padlock that keeps honest people honest; truly sensitive work belongs
on hardware you own.

**Open hardening scope:** container PID 1 still runs as root
([#20](https://github.com/malaiwah/glm52-exl3-vast/issues/20)). Rootless Podman
maps that identity into a host user namespace; it is not a non-root application
and still exposes that user's mounted files and devices. Host networking/IPC
remain in owned-host examples ([#21](https://github.com/malaiwah/glm52-exl3-vast/issues/21)):
loopback cache listeners and API authentication do not isolate NCCL/worker
sockets or other host services. Use host firewall/SSH boundaries; bridge TP4/DCP4
and a non-root entrypoint require separate compatibility qualification.
Parent ML dependency index selection
([#24](https://github.com/malaiwah/glm52-exl3-vast/issues/24)) and complete
hash-locked core ML requirements
([#25](https://github.com/malaiwah/glm52-exl3-vast/issues/25)) remain open.
The hash-locked SOUL venv and immutable parent digest narrow their respective
boundaries; they do not retroactively make the parent's dependency build
hash-verified or first-index-only.

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

  DNS and ACME are deliberately **not an unbounded startup dependency**. The
  appliance gives the first registration/issuance attempt 150 seconds, then
  continues engine startup and retries every five minutes in the background.
  A successful background retry persists the certificate and reports that one
  restart/apply is needed to put it on the listener. Tune those bounds with
  `ACME_ATTEMPT_TIMEOUT_S` and `ACME_BACKGROUND_RETRY_S`; do not make the
  foreground deadline long enough to hide model-download or engine progress.

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
- **Disk note**: GLM-5.2 3.42bpw needs at least **600 GB** to avoid
  ENOSPC during inference. The real budget: ~339 GB weights + ~39 GB image
  + ~15 GB SparkInfer/CuTe JIT compile caches + ~10 GB container writable
  layer + operational margin. A 450 GB disk fills to 100% and crashes the
  engine when JIT caches write. JarvisLabs VMs should use `--storage 650`
  to leave room; Vast/Runpod should provision 600 GB. If you must use a
  smaller disk, disable the LMCache L2 tier (`PREFIX_CACHE_DISK_GB=0`,
  which is the default) and be aware JIT compilation may still fail.
  Qwen's checkpoint is about 21 GiB; its 100 GB volume leaves room for
  cache and experiments.

## Attribution, licenses and runtime provenance

This repository's [MIT license](LICENSE) covers its own licensed appliance
code, not every file in the image or the model weights. Preserve bundled
third-party copyright/license/notice files when redistributing. The parent
runtime combines vLLM, B12X/SparkInfer, ExLlamaV3 and CUDA/NCCL components;
their licenses remain separate. In particular, the
[parent runtime](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw)
declares **`LicenseRef-ShapleyMCG-1.0`**; retain its Shapley/MCG license and
notices alongside the [ExLlamaV3 lineage](https://github.com/turboderp-org/exllamav3)
attribution. An appliance OCI label saying MIT is not a license grant for
that parent or its codebook implementation.

Full GLM-5.3 weights use the custom
[GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/LICENSE),
as the [quant model card](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
declares. Encoder/tooling MIT attribution does not change that model license.
Review its notice, use and commercial MaaS conditions before deployment.

The build generates `/opt/runtime-provenance.json` from the **installed**
filesystem after all overlays. It records the parent digest, actual critical
source and native-library hashes, package/module locations and versions, and
model pins from the registry. It replaces the stale r26 ledger, not historical
r26 evidence. The output cannot know the image's own future registry digest:
record the pushed manifest digest separately alongside the extracted JSON.
Source hashes alone are not GPU, model-capacity, or supply-chain qualification.
