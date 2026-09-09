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
│  per-slot runtime │     │ boot from the warm FS        │
│  caches           │     └──────────────────────────────┘
└──────────────────┘              ▲
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
   finds the weights, the vLLM AOT compile cache, and — with
   `VLLM_EXL3_ONLINE_CACHE_DIR` pinned per slot — the ~12 GiB online-quant
   cache, all already present.
3. **Scalability.** JarvisLabs filesystems attach to any number of your
   instances at once; each replica boots from the same weights read-only and
   keeps its own config state on its own disk.

## Measured time-to-first-serve (4×RTX PRO 6000 spot, IN1)

| Path | Time to serving | GPU cost of the wait |
|---|---|---|
| Cold: everything on the serve instance (quickstart alone) | ~36 min | ~$2.38 |
| Warm weights (FS attached, quant cache on instance disk) | 25.6 min | ~$1.69 |
| Warm weights + shared AOT cache (second replica) | 19.1 min | ~$1.26 |
| **Fully warm (slot-pinned quant cache on the FS)** | **~14 min** | **~$0.92** |

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
./jarvislabs_fleet.sh scale      # second replica + router pick-up
./jarvislabs_fleet.sh status     # instances + filesystem
./jarvislabs_fleet.sh teardown   # destroy instances, KEEP the filesystem
```

The script keeps its instance map in `~/.jarvis-fleet.json`; `serve` is
repeatable and each replica is independent. `teardown` never touches the
filesystem — keep it so the next fleet is warm on the first launch.

## How each part works

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
- `VLLM_EXL3_ONLINE_CACHE_DIR=/home/jl_fs/.runtimes/<slot>/exl3-online` —
  the ~12 GiB online-quantization cache, keyed by slot name so a slot
  relaunch reuses it while two slots never race on the same files. Boot
  logs confirm `Online EXL3 K6 cache hit` for all 1644 entries and skip
  re-quantization entirely.

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

Requests from the same client key (or carrying a session id) keep landing on
the replica whose LMCache prefix tier already holds their conversation; the
shuffle only spreads NEW sessions. Verified live: repeated calls with one
client key all hit the same `x-litellm-model-id`, a different key pinned to
the other replica.

## Networking facts (measured, not documented by the provider)

- JarvisLabs containers expose `--http-ports` ONLY through their HTTPS proxy
  (`https://<6-hostname-chars><machine-id><index>.notebooksc.jarvislabs.net`);
  raw TCP on the public IP is unreachable by design.
- The router CPU VM's port 4000 IS directly reachable on its public IP
  (plain HTTP; put TLS in front if this matters to you), and it does not
  support `--http-ports` at all.
- There is no network route between CPU VMs (10.0.0.x) and containers
  (10.200.74.x), so the VM router reaches replicas via the public proxy
  (~0.2–0.3 s overhead per request).
- Containers CAN reach each other privately: replica A hit replica B at
  `10.200.74.31:8000` directly (200, authenticated). A colocated router
  (LiteLLM on one of the serve containers, private URLs) is the
  low-latency variant; it trades the stable public entrypoint for speed.
  GPU VMs (not containers) accept `--vpc-id`, so an all-VM fleet can do
  fully private networking.

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
