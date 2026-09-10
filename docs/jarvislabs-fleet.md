# JarvisLabs fleet deployment: two-instance pattern with a shared filesystem

The single-instance quickstart downloads 331 GB of GLM-5.3 weights onto the
GPU instance you are paying GPU rates for, and throws them away when the
instance is destroyed. The fleet pattern splits the roles:

```
┌──────────────────┐     ┌──────────────────────────────┐
│ persistent FS     │◀───│ 1-GPU spot "populator"       │
│ glm53-weights     │     │ fills the FS once (~$0.17)  │
│ 500 GB, IN1       │     └──────────────────────────────┘
│ /home/jl_fs       │     ┌──────────────────────────────┐
│  weights          │◀───│ 4-GPU spot replicas (any num) │
│  shared caches    │     │ boot from the warm FS:       │
│  + image tree     │     │ weights, quant cache,        │
└──────────────────┘     │ image tree, AOT cache        │
                          └──────────────────────────────┘
                                  ▲
                                  │ HTTPS proxy (public)
                    ┌──────────────────────────────┐
                    │ CPU VM router ($0.05/hr)      │
                    │ LiteLLM, session affinity      │
                    └──────────────────────────────┘
```

Three gains, all measured on live spot instances (Sept 9, 2026):

1. **Cost.** Weights are downloaded once by a 1-GPU spot instance
   ($0.99/hr × ~10 min ≈ $0.17) instead of by every 4-GPU launch
   ($3.96/hr × 12 min ≈ $0.79 per launch). The populate pays for itself on
   the first relaunch.
2. **Reusability.** The filesystem outlives every instance. A relaunch
   finds the weights, the vLLM AOT compile cache, the unpacked appliance
   image tree, and — with `VLLM_EXL3_ONLINE_CACHE_DIR` pinned to the
   filesystem — the ~12 GiB online-quant cache, all already present.
3. **Scalability.** JarvisLabs filesystems attach to any number of your
   instances at once; each replica boots from the same weights read-only and
   keeps its own config state on its own disk.

## Measured time-to-first-serve (4×RTX PRO 6000 spot, IN1)

| Path | Time to serving | GPU cost of the wait |
|---|---|---|
| Cold: everything on the serve instance (quickstart alone) | ~36 min | ~$2.38 |
| Warm weights (FS attached, quant cache on instance disk) | 25.6 min | ~$1.69 |
| Warm weights + shared AOT cache (second replica) | 19.1 min | ~$1.26 |
| **Fully warm (shared quant + image trees on the FS)** | **~11–14 min** | **~$0.73–0.92** |

Fully warm relaunches save ~$1.45 and ~22 minutes per launch versus cold,
indefinitely, for a one-time ~$0.17 populate. First-token latency after
serving starts: ~1–3 s through the full public chain
(client → router → proxy → replica), including TLS.

## The commands

Everything is driven by `scripts/jarvislabs_fleet.sh` from your machine
(requires the `jl` CLI, `ssh`, `python3`):

```bash
# one-time: the persistent filesystem (IN1 — the PRO 6000 region)
jl filesystem create --name glm53-weights --storage 500 --region IN1

curl -fsSL .../scripts/jarvislabs_fleet.sh -o jarvislabs_fleet.sh

./jarvislabs_fleet.sh populate   # 1-GPU spot fills the FS, then destroys itself
./jarvislabs_fleet.sh serve      # 4-GPU spot replica, boots warm, prints endpoint
./jarvislabs_fleet.sh router     # CPU VM LiteLLM router with session affinity
./jarvislabs_fleet.sh manager    # autonomous fleet manager on the router VM
./jarvislabs_fleet.sh scale      # second replica + router pick-up
./jarvislabs_fleet.sh status     # instances + filesystem
./jarvislabs_fleet.sh teardown   # destroy replica slots + helpers, KEEP FS and router
./jarvislabs_fleet.sh teardown --all  # also destroy the router VM
```

The script keeps its instance map in `~/.jarvis-fleet.json`; `serve` is
repeatable and each replica is independent. `teardown` destroys every live
`glm53-serve-*` slot (manager-created ones included — they are not in the
instance map) and never touches the filesystem or, by default, the router
VM (the LE certificate, master key and manager config exist only there);
pass `--all` to take the router down too.

## How each part works


**Fleet manager (autonomous).** `manager` turns the router VM into a
closed-loop fleet manager (`scripts/fleet_manager.py`, systemd-supervised):

- **Replicas are autonomous.** The serve recipe is registered as a
  JarvisLabs startup script (fleet API key baked in at registration);
  registration is idempotent — the script is updated in place by name, so
  re-running `manager` never accumulates copies. Every replica created
  with `--script-id` grafts itself, pins `MODEL_DIR` and the shared caches
  to the filesystem, and waits for its own health — no outside SSH involved.
- **Manager-owned slots are numeric.** Only `glm53-serve-<N>` slots are
  adopted. Manually created replicas (`glm53-serve-a` from `serve`/`scale`)
  keep their own appliance-generated API key, which the fleet key cannot
  authenticate against — do not mix the two under one router.
- **Reap recovery, with boot-time awareness.** A slot that died — spot
  reclamation, crash — is recreated. A young slot that is not serving yet
  is *booting*, not broken (`boot_timeout_seconds`, default 2400): the
  manager never reaps an alive slot younger than that, so a slow cold boot
  cannot trigger a destroy-and-recreate thrash loop.
- **Cold boots are serialized.** `create_slot` refuses to create while any
  alive slot is younger than the boot timeout: two cold boots at once would
  write the one shared quantization cache concurrently, and this also makes
  every creation path respect `MAX_REPLICAS`.
- **Load-based scaling.** Each healthy replica is scraped at `/metrics`
  (unauthenticated through the proxy): queueing (`vllm:num_requests_waiting`
  above the threshold) or a full batch window scales up toward
  `MAX_REPLICAS`. Scale-down fires on LIGHT load — fewer than 4 requests in
  flight across the fleet for 15 minutes — retires the least-loaded
  replica: litellm is rewired without it first, a drain window passes, and
  only then is it destroyed, so in-flight requests and pinned sessions are
  not killed. A 5-minute cooldown bounds churn.
- **litellm rewiring, verified.** When the healthy set changes, the manager
  regenerates `config.yaml` from verified endpoints (an endpoint counts
  only if its `/metrics` body is vLLM's — the Jupyter lab URL answers 200
  to everything and must never be routed to) and restarts litellm. A failed
  restart is retried next cycle; the config is never rewritten to an empty
  model list while replicas are briefly down.

Watch it: `ssh ubuntu@<router> journalctl -u fleet-manager -f`.

**Populate.** A 1×RTX PRO 6000 spot container attaches the filesystem
(`--fs-id`), runs one resumable `snapshot_download` of
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw` at the qualified revision into
`/home/jl_fs/GLM-5.3-EXL3-TR3-3.42bpw`, and destroys itself. Measured:
331 GB in 607 s (~550 MB/s). CPU-only instances cannot attach filesystems
on JarvisLabs (`--fs-id is not supported with CPU VMs`), so the cheapest
attachable instance in IN1 is a single PRO 6000.

**Serve.** Each replica is a 4×PRO 6000 spot container with the filesystem
attached. The quickstart grafts the appliance with:

- `MODEL_DIR=/home/jl_fs/GLM-5.3-EXL3-TR3-3.42bpw` — weights on the FS;
  `snapshot_download` verifies the complete snapshot in seconds and never
  transfers.
- `GLM_STATE_DIR=/home/turnkey/workspace/.glm-config` (instance disk) —
  config state, failures log, and the auto-generated API key stay private
  per replica. The API key deliberately does NOT live next to the weights:
  a weights-adjacent key file on a shared filesystem would be written by
  every replica and shared by all of them.
- `VLLM_EXL3_ONLINE_CACHE_DIR=/home/jl_fs/.runtimes/exl3-online` — the
  ~12 GiB online-quantization cache, one directory shared by all slots.
  The cache content is a pure function of (weights revision, quant
  algorithm, GPU arch), so per-slot copies would waste 12 GiB per replica;
  concurrency safety comes from the manager serializing cold boots (one
  non-adult slot at a time). Boot logs confirm `Online EXL3 K6 cache hit`
  for all 1644 entries and skip re-quantization entirely.
- `TURNKEY_ROOT=/home/jl_fs/.image/qual` — the unpacked appliance image
  tree on the FS. The image must be digest-pinned
  (`repo@sha256:...`), and the tree carries a marker naming the digest it
  was unpacked from; a replica whose image digest does not match the tree
  fails closed rather than grafting a mismatch. The slow GHCR fetch+unpack
  (up to ~14 min from IN1) therefore happens once per image version, not
  once per replica.

Replicas also run AIBeast's selected runtime envs, so the rental fleet
behaves like the qualified appliance: prefill fairness at a 60% compute
share, `MAX_NUM_SEQS=12`, batched tokens 3072, prefill capacity 2048,
0.95 GPU memory utilization, and the 48-token capture/Trellis window that
12 sequences x 4 MTP tokens require.

vLLM's AOT compilation cache lands beside the model dir on the FS and is
shared by all replicas; that is what made the second replica boot 6 minutes
faster than the first.

**Router.** A 2-vCPU/8-GB CPU VM ($0.05/hr) runs LiteLLM under systemd
(`sudo systemd-run`, because the VM has no user lingering and an ssh-spawned
process dies with the session). The fleet script regenerates its config
from live instances — each replica is probed first (authenticated
`/v1/models` through the JarvisLabs proxy) and only a verified URL is routed
to. Session stickiness:

```yaml
router_settings:
  routing_strategy: simple-shuffle
  model_group_affinity_config:
    GLM-5.3: [deployment_affinity, session_affinity]
  deployment_affinity_ttl_seconds: 3600
```

**How sessions are identified.** LiteLLM pins in this order: a session id
— any of the `session-id` / `session_id` / `thread-id` /
`conversation_id` headers, a W3C `Baggage: session.id=...` header, or a
`session_id` body field — and, failing that, a hash of the incoming API
key. Pins expire after one idle hour and refresh on every request. Since
the router enforces a master key, all anonymous clients share one hash: to
get per-conversation spread across replicas, send a `session-id` header
per conversation (agent frameworks like Codex emit these natively) or
issue distinct virtual keys via `/key/generate`.

**TLS (optional).** Export `DESEC_TOKEN` + `DESEC_DOMAIN` (your deSEC
zone) + `ACME_EMAIL` (the Let's Encrypt registration address) before
running `router`, and the router gets a real Let's Encrypt
certificate: the script registers `glm53-router.<zone>` → the VM's public
IP and issues via lego DNS-01 — the same deSEC path the appliance uses,
guard included. litellm then serves TLS on 443 (the non-root systemd unit
gets the ambient bind capability). Endpoint: `https://glm53-router.<zone>/v1`.
Verified live: `glm53-router.malaiwah.dedyn.io` issued in 65 s. Note LE's
5-duplicate-certificates/week limit: the certificate lives on the VM, so
recreating the router more than ~5× a week re-issues and can hit it.

## Networking facts (measured, not documented by the provider)

- JarvisLabs containers expose `--http-ports` ONLY through their HTTPS proxy
  (`https://<6-hostname-chars><machine-id><index>.notebooksc.jarvislabs.net`);
  raw TCP on the public IP is unreachable by design.
- The router CPU VM's port 4000 is directly reachable on its public IP
  (plain HTTP unless DESEC TLS is configured), and it does not support
  `--http-ports` at all.
- CPU VMs cannot attach filesystems (`--fs-id is not supported with CPU
  VMs`), which is why the populator is a 1-GPU spot instance.
- Containers CAN reach each other privately: replica A hit replica B at
  `10.200.74.31:8000` directly (200, authenticated). A colocated router
  (LiteLLM on one of the serve containers, private URLs) is the
  low-latency variant; it trades the stable public entrypoint for speed.
  GPU VMs (not containers) accept `--vpc-id`, so an all-VM fleet can do
  fully private networking.

## Considered and rejected (measured)

- **User-space weight cache in vLLM** (local copy populated on first read,
  cachefilesd-style): the FS reads at 753 MB/s cold (1.9 GB/s re-read)
  versus 634 MB/s for the local rbd, so a local copy makes the ~7-minute
  331 GB load slower, not faster. `cachefilesd` itself is impossible anyway
  (the FS is a FUSE client, not NFS; FS-Cache cannot attach). Dropped.
- **Pause instead of destroy for scale-down**: a paused instance keeps the
  graft and skips allocation, but the FS already removed those phases, so
  resume saves only ~2 of the ~11 minutes to serving — and spot restore odds
  are the same as a fresh create. Dropped.

## Costs at a glance (spot, IN1)

| Resource | Rate |
|---|---|
| Filesystem 500 GB | billed by JarvisLabs separately (kept between fleets) |
| Populator 1×PRO 6000 | $0.99/hr, ~10 min once |
| Serve replica 4×PRO 6000 | $3.96/hr while up |
| Router CPU VM 2 vCPU/8 GB | $0.05/hr |

Spot instances can be reclaimed at any time; that is precisely what the
persistent filesystem defends against — a reclaimed fleet is a ~15-minute,
~$1 relaunch away from fully warm.
