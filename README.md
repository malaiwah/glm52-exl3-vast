# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 (96GB)

One-click 512K-context GLM-5.2 OpenAI-compatible endpoint on rented GPUs:
EXL3 trellis weights (~77 GiB/rank — fits commodity 95.01-GiB cards),
**fp8 KV cache** (correct on stock drivers — the nvfp4 default silently
corrupts >~150K context without a host driver P2P override; see Evidence),
MTP speculative decoding, and DRAM KV offload auto-sized to 70% of the
instance's RAM allocation (cgroup-aware — partial rentals don't oversize it).
Weights auto-download on first boot (~332 GB — pick a fast-net host).

## One-click launch

**[▶ Launch on vast.ai](https://cloud.vast.ai/?ref_id=386667&template_id=697166835ebda4fe5de506047576f45d)** —
public template with the image, ports, launch mode, disk, and host filters
(4x RTX PRO 6000, >=400GB disk, >=1Gbps net) pre-configured. Rent, wait for
"Application startup complete" in the instance logs, grab the API key from the
same logs, done.

## vast.ai template settings (manual setup)
- **Image**: `ghcr.io/malaiwah/glm52-exl3-vast:latest` (the ghcr.io package
  must be set to **public** visibility, or vast hosts can't pull it)
- **Launch mode**: docker ENTRYPOINT (vLLM logs appear on the instance console;
  SSH works per vast standards)
- **Docker options**: `-p 8000:8000 --ipc=host --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576` (memlock is REQUIRED for DRAM offload)
- **Disk**: >= 400 GB
- **GPU filter**: 4x RTX PRO 6000 Blackwell (96 GB), CUDA >= 13.0
- **Env (all optional)**: `HF_TOKEN` (faster download), `OFFLOAD_FRACTION`
  (default 0.70; 0 disables), `MTP_TOKENS` (default 3; 0 disables),
  `MAX_NUM_SEQS`, `MAX_MODEL_LEN` (default 524288), `SERVED_MODEL_NAME`. Recommended extra env: `OPEN_BUTTON_PORT=8000` (dashboard Open button targets the API). On ready, the instance labels itself "GLM-5.2 READY <endpoint>" in your dashboard

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
