# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 (96GB)

One-click 512K-context GLM-5.2 OpenAI-compatible endpoint on rented GPUs:
EXL3 trellis weights (~77 GiB/rank — fits commodity 95.01-GiB cards),
**fp8 KV cache** (correct on stock drivers — the nvfp4 default silently
corrupts >~150K context without a host driver P2P override; see Evidence),
MTP speculative decoding **with a quantized trellis draft** (see MTP78 below),
and DRAM KV offload auto-sized to 70% of the instance's RAM allocation
(cgroup-aware — partial rentals don't oversize it). Weights auto-download on
first boot (~332 GB — pick a fast-net host).

## One-click launch

**[▶ Launch on vast.ai](https://cloud.vast.ai/?ref_id=386667&template_id=ccab1ea5b390cec1bb615a79840baa40)** —
public template with the image, ports, launch mode, disk, and host filters
(4x RTX PRO 6000, >=400GB disk, >=1Gbps net) pre-configured. Rent, wait for
"Application startup complete" in the instance logs, grab the API key from the
same logs, done.

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

## Vision (default ON)

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

## MTP78: quantized speculative-draft layer (default ON)

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

**KV headroom:** available KV memory goes 5.27 -> 8.92 GiB/GPU (**+69%**). This
**Speculation depth:** the cheaper draft also moves the optimum. Measured
(GSM8K n=30, +-0.5% noise floor): MTP-2 42.9 tok/s, **MTP-3 51.5** (default),
**MTP-5 53.4** (+3.7%). MTP-5 used to lose ~22% with the 19.3 GB BF16 draft;
at 3.7 GB the extra draft tokens are cheap enough to win. Try `MTP_TOKENS=5`
if you want that last few percent (not yet the default — wants a larger run).

template pins the pool at 512K (`--num-gpu-blocks-override 2048`), so the
headroom is unspent by default — at ~10.3 KiB/token it is worth roughly
**+355K tokens of pool (~880K context)** if you lift the override, or the same
512K with much more concurrency margin.

### Deploying it elsewhere (no checkpoint surgery)

The overlay works as a *separate draft directory* — leave the base checkpoint
untouched and add one field:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3,
                       "moe_backend":"triton","draft_sample_method":"probabilistic",
                       "model":"/path/to/GLM-5.2-EXL3-TR3-MTP78/3bpw-keep0"}'
```

(Validated: KV showed 8.92 GiB with the base checkpoint reverted to stock BF16 —
proof the trellis draft really loads from the override dir.)

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

## vast.ai template settings (manual setup)
- **Image**: `ghcr.io/malaiwah/glm52-exl3-vast:latest` (the ghcr.io package
  must be set to **public** visibility, or vast hosts can't pull it)
- **Launch mode**: docker ENTRYPOINT (vLLM logs appear on the instance console;
  SSH works per vast standards)
- **Docker options**: `-p 8000:8000 -p 1111:1111 --ipc=host --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576` (memlock is REQUIRED for DRAM offload; :1111 is the landing page)
- **Disk**: >= 400 GB
- **GPU filter**: 4x RTX PRO 6000 Blackwell (96 GB), CUDA >= 13.0
- **Env (all optional)**: `HF_TOKEN` (faster download), `OFFLOAD_FRACTION`
  (default 0.70; 0 disables), `MTP_TOKENS` (default 3; 0 disables),
  `MAX_NUM_SEQS`, `MAX_MODEL_LEN` (default 524288), `SERVED_MODEL_NAME`,
  `MTP78_TRELLIS` (default 1: quantized trellis draft, see MTP78 section; 0 = stock BF16 draft),
  `LANDING_PAGE` (default 1; 0 disables the :1111 landing page). Recommended
  extra env: `OPEN_BUTTON_PORT=1111` — the dashboard **Open** button then hits
  the landing page: live boot status (weight-download progress, TLS, engine),
  ready-to-paste client configs (oh-my-pi, opencode, Claude Code, Codex), and
  a minimal streaming chat UI at `/chat`. Token-gated; with TLS configured the
  page upgrades plain-HTTP hits to HTTPS and only then embeds the API key.
  On ready, the instance labels itself "GLM-5.2 READY <endpoint>" in your dashboard

Endpoint: `http://<instance>:8000/v1` once the console shows
`Application startup complete` (first boot: download + JIT, plan ~30-60 min;
later boots only pay JIT).

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

Base image: `verdictai/glm52-exl3-sparkinfer@sha256:bfd6d667...` (pinned).
Checkpoint: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`.

## Security

**Threat model honestly stated:** a rented host's operator has root — memory,
VRAM, and traffic on the box are visible to a determined host. These controls
are the padlock that keeps honest people honest; truly sensitive work belongs
on hardware you own.

- **API key** (always on): set `VLLM_API_KEY`, or one is auto-generated and
  printed in the instance console logs at boot. All /v1 calls need
  `Authorization: Bearer <key>`.
- **SSH tunnel** (recommended for solo use): no public exposure needed —
  `ssh -p <ssh-port> root@<ssh-host> -L 8000:localhost:8000`
  then use `http://localhost:8000/v1`. You can omit `-p 8000:8000` from the
  docker options entirely in this mode.
- **TLS via Let's Encrypt DNS-01 — turnkey with deSEC** (recommended):
  One-time setup (~2 minutes, free, reusable forever):
  1. Create an account at [desec.io](https://desec.io/signup) (email only).
  2. Register a dynDNS domain, e.g. `yourname.dedyn.io`
     ([docs](https://desec.readthedocs.io/en/latest/dyndns/configure.html)).
  3. Create an API token: [Token management](https://desec.io/tokens)
     ([docs](https://desec.readthedocs.io/en/latest/auth/tokens.html)).

  Then, when launching the template, add two environment variables on the
  launch page: `DESEC_TOKEN=<your-token>` and `DESEC_DOMAIN=yourname.dedyn.io`.
  At boot the instance registers a stable per-instance hostname
  (`glm-<container-id>.yourname.dedyn.io`), points it at itself, obtains a
  Let's Encrypt certificate via DNS-01 ([lego](https://go-acme.github.io/lego/dns/desec/)),
  and prints the final `https://...:<port>/v1` URL in the console logs next to
  the API key. Each instance gets its own name, stable across reboots — so
  records don't pile up in the zone and certs persist on the volume, reused
  while they have >7 days validity left.

- **Other DNS providers** (Cloudflare, DuckDNS, 150+ via lego): set
  `ACME_DOMAIN=glm.example.com`, `ACME_DNS_PROVIDER=cloudflare` (any lego
  provider), and the provider credential env (e.g.
  `CLOUDFLARE_DNS_API_TOKEN=...` with Zone:DNS:Edit scope; or DuckDNS:
  `ACME_DNS_PROVIDER=duckdns` + `DUCKDNS_TOKEN=...` — free, no domain needed).
  Point the name at the instance IP, and the endpoint becomes
  `https://<domain>:<mapped-port>/v1`. Certs persist on the volume and are
  reused while they have >7 days validity left, then re-issued at boot (avoids
  Let's Encrypt's 5/week duplicate-cert limit on reboot loops).
- **Token hygiene**: anything in template env is visible to the host operator
  and to anyone you share the template with — scope DNS tokens narrowly
  (single zone, DNS-only), rotate them when the rental ends, and never
  publish a template with tokens baked in.
- **Egress hygiene**: telemetry disabled (`VLLM_NO_USAGE_STATS`,
  `DO_NOT_TRACK`, `HF_HUB_DISABLE_TELEMETRY`), `HF_HUB_OFFLINE=1` once weights
  are local, and the boot log prints the listening-socket audit. Full egress
  firewalling is not possible without NET_ADMIN (not granted on vast).
- **Disk note**: verify the instance actually allocated >=400 GB — some hosts
  under-allocate silently; first boot needs ~332 GB for weights.
