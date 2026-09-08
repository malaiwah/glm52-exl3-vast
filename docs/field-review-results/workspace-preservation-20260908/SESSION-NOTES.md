# Turnkey Appliance Build & Deploy — Session Notes

## Session date: 2026-08-28

---

## 1. Workspace setup

### Directory layout at `~/turnkey/`

```
HANDOFF.md              K6 mission brief
glm52-exl3-vast/        canonical appliance repo (main, pushed)
vllm/                   local-inference-lab/vllm (dev/gilded-gnosis)
b12x/                   local-inference-lab/b12x (master)
rtx6kpro/               local-inference-lab/rtx6kpro (master)
llm-inference-bench/    local-inference-lab/llm-inference-bench (main)
```

### What was cleaned up
- Deleted 3 superseded repo snapshots (`glm52-exl3-turnkey`, `glm52-exl3-vast-r11`, `glm52-exl3-vast-field-combined`)
- Deleted Venti LoRA quantization project (cancelled)
- Deleted all cache/pycache directories
- Archived all unique GLM-5.2-era docs, evidence, r34 maintenance recipe, PR bodies to canonical repo (commit `fbe0d27`)
- Deleted 7 old source trees (24 git repos), replaced with 4 fresh clones
- Pushed 1 unpublished commit (SparkInfer w4a16 fix `5df3310`) to `malaiwah/sparkinfer` branch `codex/fix-w4a16-direct-boundaries`

---

## 2. Docker data-root moved to ZFS tank pool

The root filesystem only had 57 GB free; the vLLM base image alone is ~30 GB.

### Steps
```bash
# Check ZFS pool
zpool list
# tank  1.36T  51.8G  1.31T ...

# Create dataset
sudo zfs create tank/docker

# Stop Docker
sudo systemctl stop docker docker.socket

# Update daemon.json
sudo python3 -c "
import json
with open('/etc/docker/daemon.json') as f:
    cfg = json.load(f)
cfg['data-root'] = '/tank/docker'
with open('/etc/docker/daemon.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"

# Start Docker
sudo systemctl start docker

# Verify
sudo docker info | grep 'Docker Root Dir'
# Docker Root Dir: /tank/docker
```

### Resulting `/etc/docker/daemon.json`
```json
{
  "insecure-registries": ["10.15.0.109:30500"],
  "data-root": "/tank/docker"
}
```

### IMPORTANT: containerd root also needs to be on tank

Docker 29 uses containerd's snapshotter integration (`io.containerd.snapshotter.v1`).
Setting `data-root` in `daemon.json` only moves Docker's metadata/buildkit state —
the actual image layers and snapshots are stored by containerd at its own root
(`/var/lib/containerd` by default), which stays on the root disk.

**Symptom**: `du -sh /var/lib/containerd` shows 63 GB while `tank/docker` shows only 37 MB.

**Fix** (do this BEFORE the next build, not during a running build):

```bash
# Create containerd dataset on tank
sudo zfs create tank/containerd

# Stop Docker + containerd
sudo systemctl stop docker docker.socket containerd

# Generate default containerd config if it doesn't exist
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml > /dev/null

# Set root to tank
sudo sed -i 's|root = "/var/lib/containerd"|root = "/tank/containerd"|' /etc/containerd/config.toml

# Move existing data (optional — can also start fresh)
# sudo rsync -a /var/lib/containerd/ /tank/containerd/

# Start containerd + Docker
sudo systemctl start containerd
sudo systemctl start docker

# Verify
sudo docker info | grep -i 'containerd\|data root'
sudo du -sh /tank/containerd
```

**Lesson**: When moving Docker storage to a new pool, you must move BOTH:
1. Docker `data-root` (in `/etc/docker/daemon.json`)
2. containerd `root` (in `/etc/containerd/config.toml`)

The current build is finishing on the root disk (21 GB free was enough). Fix
containerd's root before the next build or push.

### IMPORTANT: This machine runs k3s Kubernetes

The build machine (`mbelleau-buildbox`) runs a k3s cluster (v1.36.3) with
CI runners and agent sandboxes. **Do NOT stop/restart Docker or containerd**
carelessly — it will disrupt k3s pods.

The containerd root fix must be done either:
1. Via a bind mount / symlink approach (no service restart), or
2. During a planned maintenance window

k3s uses `tank/k3s` at `/var/lib/rancher` for its state, separate from
Docker's containerd at `/var/lib/containerd`.


## Variant selection: 3.42bpw instead of default 3.0bpw

The default variant `exl3-tr3` uses `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` (~309 GiB).
We switched to `exl3-tr3-3.42bpw` which uses `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` (~328 GiB).

```bash
# In the appliance.env file:
MODEL_VARIANT=exl3-tr3-3.42bpw
```

This is the r28 shared-H quality profile: "EXL3-TR3 shared-H 3.42bpw + online K6 — high-fidelity 520K".
It inherits from 3.36bpw (r26-qualified) which inherits from 3.25bpw (r11-qualified).
Key differences from default 3.0bpw: higher fidelity, online K6 quantization, 520K context envelope,
DCP4 (vs DCP2), 512 GiB LMCache L2 disk, different graph capture sizes.
---

## Log review findings (UX improvements for next image)

1. **Config dump printed twice** — the 40+ line effective configuration block appears twice in boot logs (once in apply_config, once in serve_once). Suppress the second print.
2. **Warnings truncated** — `warn:` lines are cut with `…`; full text matters (KLD re-run, session material).
3. **`NVIDIA_VISIBLE_DEVICES=void`** — displays literal "void" for unset; should say "unset" or "(not set)".
4. **pynvml deprecation warning** leaks to stdout during GPU probing; suppress with `PYTHONWARNINGS=ignore::FutureWarning` in the GPU detection phase.
5. **Download progress at 10% intervals** — for a 328 GiB download, 1.5 min between updates on a paid rental is anxiety-inducing. Increase to 5% or 30-second intervals.
6. **Provider detection: `generic` not `jarvislabs`** — VM hostname `jl-vm-NNN` not recognized; means JarvisLabs-specific TLS/termination features don't auto-activate. Functional consequence.
7. **`DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS = -1`** — correct internally but looks like an error; display as `disabled` or `off`.

## 3. Docker image build

### Image naming convention
- Existing convention: `ghcr.io/malaiwah/glm52-exl3-vast:<tag>` (GHCR)
- Docker Hub authenticated as `malaiwah` (GHCR token lacks `write:packages` scope)
- For this build: tagged as `malaiwah/glm52-exl3-vast:fbe0d27` and `malaiwah/glm52-exl3-vast:latest` (Docker Hub)

### Build command
```bash
cd ~/turnkey/glm52-exl3-vast
sudo docker build -t malaiwah/glm52-exl3-vast:fbe0d27 -t malaiwah/glm52-exl3-vast:latest .
```

### Dockerfile overview
- Base: `voipmonitor/vllm@sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0` (~30 GB pull)
- Installs: huggingface_hub, hf-xet, dnspython, nvtop, htop, curl, openssh-server, socat, python3-venv, lego (ACME client)
- Creates isolated SOUL nanobot venv (`/opt/nanobot-venv`)
- Freeze-diffs system env before/after venv install (fails on any diff)
- Applies 3 patches: `patch_vllm_serial_spec_warning.py`, `patch_exl3_parity_abi.py`, `patch_exl3_mixk.py`
- SHA-256 verifies patch scripts and base image
- Runs `verify_r28_base.py` immutable-source gate

### Build context
- 62 MB (no `.dockerignore` needed — small repo)
- Largest subdirs: `docs/` (31 MB), `maintenance/` (1.8 MB)

### Push to Docker Hub
```bash
sudo docker push malaiwah/glm52-exl3-vast:fbe0d27
sudo docker push malaiwah/glm52-exl3-vast:latest
```

---

## 4. JarvisLabs VM provisioning

### Create VM via `jl` CLI
```bash
jl create --gpu RTX-PRO6000 --num-gpus 4 --vm --region IN1 --storage 450 --name glm52-baseline --json -y
```

### Result
- Machine ID: 485913
- IP: 151.185.34.24
- 4× RTX PRO 6000 Blackwell Server Edition (97,887 MiB each)
- 640 GB RAM, 112 CPUs, 450 GB SSD
- SSH: `ssh ubuntu@151.185.34.24`
- Docker pre-installed (v29.5.1)
- On-demand pricing: $1.89/hr (spot not available for VMs, only containers)

### GPU topology (before P2P config)
```
PHB between all GPUs (PCIe Host Bridge — direct-attach, no switches)
All GPUs on NUMA node 0
```

### Note on spot vs on-demand
- Spot pricing ($0.99/hr) is only available for **container** instances, not VMs
- VMs are on-demand only ($1.89/hr for 4× RTX-PRO6000)
- The appliance needs a VM (SSH access, custom Docker images, `--network host`)

---

## 5. JarvisLabs VM hardening (P2P + kernel params + dist-upgrade)

### NVIDIA P2P driver override

Critical for PHB (direct-attach) topology: without it, CUDA kernels that issue
direct remote-GPU loads through IPC/BAR1 silently fall back to SysMem staging
(~15× slower for PCIe oneshot allreduce).

Source: `rtx6kpro/hardware/pcie-bandwidth.md` — "NVIDIA P2P Driver Override (ForceP2P)"

```bash
# /etc/modprobe.d/nvidia-p2p-override.conf
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
```

### UVM HMM disable

```bash
# /etc/modprobe.d/uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

### GRUB kernel parameters for PCIe stability

```bash
# /etc/default/grub — GRUB_CMDLINE_LINUX_DEFAULT
# Added: pcie_aspm=off pcie_port_pm=off
sudo update-grub
```

Full recommended line from rtx6kpro docs (Festr's Turin system):
```
GRUB_CMDLINE_LINUX="rd.auto=1 rd.md=1 rd.md.conf=1 mitigations=off spectre_v2=off spec_store_bypass_disable=off l1tf=off mds=off tsx_async_abort=off srbds=off mmio_stale_data=off retbleed=off amd_iommu=off iommu=off"
```
We only added `pcie_aspm=off pcie_port_pm=off` for now (JarvisLabs VM, not a bare-metal Turin).

### Apply order
1. Write modprobe configs (no reboot needed for files)
2. Write GRUB params and `update-grub`
3. Run `apt-get dist-upgrade`
4. Reboot to apply GRUB + reload NVIDIA driver with new params

### Verify after reboot
```bash
# Check P2P params loaded
grep -E 'RegistryDwords|ForceP2P|EnableResizableBar' /proc/driver/nvidia/params

# Check GRUB cmdline
cat /proc/cmdline | grep pcie_aspm

# Check topology
nvidia-smi topo -m
```

### Dist-upgrade
```bash
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
  -o Dpkg::Options::="--force-confdef" \
  -o Dpkg::Options::="--force-confold" \
  dist-upgrade
```
- 144 packages needed upgrading on a fresh JarvisLabs VM
- Use `--force-confdef --force-confold` to keep existing config files

---

## 6. Deploy appliance on JarvisLabs VM

### Using the bootstrap script
The repo includes `scripts/jarvislabs_vm_bootstrap.sh` designed for `curl | bash`:

```bash
# On the VM:
export TURNKEY_IMAGE=malaiwah/glm52-exl3-vast:latest
export MODEL_PROFILE=glm52-exl3
export JARVISLABS_MACHINE_ID=485913
export JARVISLABS_REGION=IN1
curl -fsSL https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/main/scripts/jarvislabs_vm_bootstrap.sh | bash
```

### What the bootstrap script does
1. Carries forward persisted credentials from `/home/turnkey/.secrets/appliance.env`
2. Creates `/home/turnkey` workspace (survives container replacement)
3. Writes env file with MODEL_PROFILE, landing page config, optional TLS/termination
4. Pulls the image: `sudo docker pull $TURNKEY_IMAGE`
5. Removes old container if present: `sudo docker rm -f glm52-turnkey`
6. Runs container with `--gpus all --ipc=host --network host --env-file ... -v /home/turnkey:/workspace`

### Manual deploy (if bootstrap script not used)
```bash
sudo docker pull malaiwah/glm52-exl3-vast:latest
sudo docker run -d \
  --name glm52-turnkey \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --network host \
  --ulimit memlock=-1:-1 \
  -e MODEL_PROFILE=glm52-exl3 \
  -e LANDING_PAGE=1 \
  -v /home/turnkey:/workspace \
  malaiwah/glm52-exl3-vast:latest
```

### SSH tunnel for secure access (no TLS configured)
```bash
ssh -L 8000:localhost:8000 -L 1111:localhost:1111 ubuntu@151.185.34.24
# Then open http://localhost:1111/ with the token from the logs
```

---

## 7. Verification

### Health check
```bash
curl -sf http://localhost:8000/health
# Returns 200 when vLLM is ready
```

### Full verify_serving.py gate
```bash
python3 scripts/verify_serving.py --endpoint http://localhost:8000 --api-key <key>
```
- Short probes: arithmetic, factual, instruction
- Structured-output probe
- Long-context needle probe (tokenizer-calibrated)
- Exit 0 = all pass

### Manual test
- Dashboard at `http://localhost:1111/` (via SSH tunnel)
- Chat completions via `curl` or the dashboard
- Check `sudo docker logs -f glm52-turnkey` for errors

---

## 8. Key safety constraints

- **AIBeast production (10.15.0.166)**: NEVER touch. No restarts, no image changes, no GPU maintenance. The inference serving this conversation depends on it.
- **JarvisLabs cost**: tear down VM promptly after verification. Use spot where possible. Monitor idle time.
- **Image registry**: Docker Hub as `malaiwah/glm52-exl3-vast` (GHCR token lacks write:packages scope)

---

## 9. Pending tasks (this session)

- [ ] Docker build completes
- [ ] Push image to Docker Hub
- [ ] Reboot JarvisLabs VM (apply GRUB + P2P + dist-upgrade)
- [ ] Deploy appliance on VM
- [ ] Pass verify_serving.py full gate
- [ ] Michel's manual test checkpoint
- [ ] Create K6-SPILL-OVER.md for Phase 2

---

## 10. Phase 2 preview (K6 GLM-5.3 Flash)

Documented in HANDOFF.md. Will be detailed in `K6-SPILL-OVER.md` after baseline checkpoint.

Key points:
- Checkpoint: `malaiwah/GLM-5.3-Flash-TR3-6bpw` (~253.5 GB, 120 shards)
- Architecture: Glm5NextForConditionalGeneration, 45 layers, 288 routed experts
- Target: TP4 on 4×96 GB SM120
- Base: Brandon v84 bring-up + local-inference-lab PR #28 final
- PR chain: B12X #243→#245, vLLM #314→#316→#318→#397
- Release codename: Auric Apotheosis (recommended)


---

## 11. SPEED-RUN GUIDE — JarvisLabs GLM-5.2 3.42bpw baseline deploy

All lessons from the 2026-08-28 deploy cycle, compressed for the next test cycle.
Follow in order; each step has exact commands and known gotchas.



### Prerequisites on build machine (`mbelleau-buildbox`)

- `sudo docker` needs root's Docker Hub auth: `sudo cp ~/.docker/config.json /root/.docker/config.json`
- Docker data-root is on `/tank/docker` (ZFS). containerd root still on `/var/lib/containerd` (root disk, 63 GB) — k3s runs, don't stop containerd carelessly.
- Image: `malaiwah/glm52-exl3-vast:latest` (Docker Hub, not GHCR — gh token lacks `write:packages`)
- Repo at `~/turnkey/glm52-exl3-vast`, branch `main`, 3 unpushed commits: `6981d2e`, `53e6dce`, `03d3448`



### Step 1: Create JarvisLabs VM (~2 min)

```bash

jl create --gpu RTX-PRO6000 --num-gpus 4 --vm --region IN1 --storage 450 --name glm52-baseline --json -y

# Record machine ID and IP from JSON output

# SSH: ssh ubuntu@<IP>

```

- On-demand only for VMs ($1.89/hr). Spot ($0.99) is container-only.

- 4× RTX PRO 6000 (97,887 MiB each), 640 GB RAM, 112 CPU, 435 GB disk.



### Step 2: Harden VM (~10 min, runs while you prep other things)

```bash

# 2a. Dist-upgrade

sudo apt-get update -qq

sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade



# 2b. P2P driver override (CRITICAL for PHB topology)

echo 'options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"' | sudo tee /etc/modprobe.d/nvidia-p2p-override.conf

echo 'options nvidia_uvm uvm_disable_hmm=1' | sudo tee /etc/modprobe.d/uvm.conf



# 2c. GRUB kernel params

# Edit /etc/default/grub: add pcie_aspm=off pcie_port_pm=off to GRUB_CMDLINE_LINUX_DEFAULT

sudo update-grub



# 2d. *** CRITICAL: Disable nvidia-dcgm.service BEFORE first boot ***

# nv-hostengine binds tcp://127.0.0.1:5555 which conflicts with vLLM/LMCache ZMQ

sudo systemctl mask nvidia-dcgm.service

sudo systemctl mask nv-hostengine.service

sudo kill $(pgrep nv-hostengine) 2>/dev/null || true



# 2e. Reboot to apply GRUB + NVIDIA driver reload

sudo reboot

```



### GOTCHA: Port 5555 conflict (cost us ~15 min this cycle)

`nvidia-dcgm.service` auto-starts `nv-hostengine` on port 5555. vLLM's LMCache

multiprocess adapter also needs ZMQ on `tcp://127.0.0.1:5555`. Symptom:

`zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:5555')`.

Fix: mask `nvidia-dcgm.service` + kill the process. It auto-restarts if you only

kill without masking. **Do this in Step 2 before deploying the container.**



### Step 3: Deploy container (~2 min)

```bash

# Pull image

sudo docker pull malaiwah/glm52-exl3-vast:latest



# Create env file (weights already on volume if reusing same VM)

cat > /home/turnkey/.secrets/appliance.env << 'EOF'

MODEL_PROFILE=glm52-exl3

MODEL_VARIANT=exl3-tr3-3.42bpw

LANDING_PAGE=1

OPEN_BUTTON_PORT=1111

MODEL_DISPLAY_NAME=GLM-5.2 3.42bpw JarvisLabs Baseline

HF_TOKEN=[REDACTED]

PREFIX_CACHE_DISK_GB=32

EOF



# Run container

sudo docker run -d --name glm52-turnkey --restart unless-stopped \

  --gpus all --ipc=host --network host --ulimit memlock=-1:-1 \

  --env-file /home/turnkey/.secrets/appliance.env \

  -v /home/turnkey:/workspace \

  malaiwah/glm52-exl3-vast:latest

```



### Step 4: Monitor boot phases (know the timeline)

| Phase | Duration | Log signature |

|-------|----------|---------------|

| Config resolve + PCIe calibrate | ~30s | `>>> profile runtime:` lines |

| Checkpoint reconcile | ~5s | `>>> reconcile:` lines |

| LMCache init | ~5s | `>>> LMCache prefix tier:` |

| **401 probe notice** | instant | `>>> Verification: one intentional bad-key GET /v1/models will log 401` — **THIS IS EXPECTED, NOT A CRASH** |

| Weight download (first boot only) | ~20 min | `>>> Download progress: N%` at 5% intervals |

| Weight loading (81 shards, online K6) | ~15-28 min | `Loading safetensors checkpoint shards: N%` + `Online EXL3 K6 encoded model.layers.N.*` |

| torch.compile + AOT | ~30s | `torch.compile took X s` |

| LMCache KV register | ~5s | `LMCache INFO: Registering kv caches` |

| CUDA graph capture (8 graphs × 3 types) | ~2-3 min | `Capturing CUDA graphs (PIECEWISE/FULL):` + `Capturing prefill CUDA graphs` |

| Speculator capture | ~30s | `Capturing model for speculator...` |

| verify_serving.py (auto) | ~2 min | `>>> Verify:` probe results |

| **Total cold boot (first)** | ~50 min | download + load + compile + verify |

| **Total warm boot (weights cached)** | ~25 min | load + compile + verify |



### Key monitoring commands

```bash

# Quick status

sudo docker logs --tail 10 glm52-turnkey



# Health check

curl -sf http://localhost:8000/health && echo "HEALTH OK" || echo "not ready"



# Search for errors

sudo docker logs glm52-turnkey 2>&1 | grep -i 'error\|traceback\|OOM\|killed\|crash\|5555'



# Dashboard token (from logs)

sudo docker logs glm52-turnkey 2>&1 | grep -i 'token\|dashboard\|landing'

```



### Step 5: SSH tunnel for manual testing

```bash

# On your local machine:

ssh -L 8000:localhost:8000 -L 1111:localhost:1111 ubuntu@<VM_IP>

# Dashboard: http://localhost:1111/ (token from logs)

# API: http://localhost:8000/v1/chat/completions

```



### Step 6: Teardown

```bash

jl destroy <machine_id>

```



### Code changes made this session (3 commits, not yet pushed to GitHub)

| Commit | Description |

|--------|-------------|

| `6981d2e` | `PREFIX_CACHE_DISK_FRACTION` knob + all variants `PREFIX_CACHE_DISK_GB` default → 0 |

| `53e6dce` | UX: suppress duplicate config dump, fix `void` message, 5% download intervals, JarvisLabs hostname detection, `-1` as `disabled` |

| `03d3448` | README: docker-compose example + P2P prerequisite docs |



### Image vs code gap

The deployed image (`fbe0d27`) does NOT include the 3 new commits. They will be

in the next image rebuild. The 3 commits are config/UX/docs only — no runtime

behavior change that affects this deployment.



### What to check if boot fails

1. **Port 5555**: `sudo ss -tlnp | grep 5555` — if `nv-hostengine`, mask the service

2. **OOM during K6 encoding**: 0.95 GMU online K6 is tight on 4×96GB; watch `nvidia-smi`

3. **EXL3 load errors**: check for `fallback proxy error` (normal) vs `Traceback` (bad)

4. **Disk space**: 328 GiB weights + 38 GB image + 32 GB L2 = ~398 GB needed; 435 GB disk

5. **401 in logs**: EXPECTED — it's the intentional bad-key auth probe, not a crash


---

## 12. AIBeast production salvage audit (2026-08-28)

Compared AIBeast production container env vs turnkey appliance resolved env.
Goal: find production tuning missing from the turnkey image.

### Already in our entrypoint.sh (exported globally)
- `NCCL_IB_DISABLE=1`, `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=LL,LL128,Simple` (line 1515)
- `VLLM_ENABLE_PCIE_ALLREDUCE`, `VLLM_PCIE_ALLREDUCE_BACKEND` (line 1537)
- `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB`, `VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0` (line 1544)
- `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel` (line 1550)
- `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (line 1603-1604)
- `VLLM_RTX6K_FUSED_ALLREDUCE_ADD` unset (line 1524-1525)
- `HF_HUB_OFFLINE=1` exported after download (line 2076)
- `MODEL_READ_ONLY` handled (line 892, 1296)

### Was missing — fixed this session
- `VLLM_PROMPT_LOGPROBS_CHUNK_SIZE=128` — knob existed in glm_config.py defaults and
  as a KNOB, but was NOT in the variant runtime_env dict, so it wasn't exported via
  PROFILE_RUNTIME_ENV. Added to base exl3-tr3 variant runtime_env (commit `6568ff6`).

### Value differences (expected — AIBeast-specific production overrides)
These are deployment-specific overrides in production.env, not missing from the appliance:
- `CUDAGRAPH_CAPTURE_SIZES`: AIBeast=4..48, turnkey=4..32 (variant default)
- `MAX_CUDAGRAPH_CAPTURE_SIZE`: AIBeast=48, turnkey=32
- `VLLM_EXL3_TRELLIS_MAX_M`: AIBeast=48, turnkey=32
- `MAX_NUM_SEQS`: AIBeast=12, turnkey=8
- `GPU_BLOCKS_OVERRIDE`: AIBeast=0, turnkey=2032
- `KV_CACHE_MEMORY_BYTES`: AIBeast=4518907904, turnkey=0 (auto)
- `PCIE_CALIBRATION`: AIBeast=off, turnkey=auto (auto is better for rentals)
- `PCIE_DMA_MIN_BYTES`: AIBeast=25165824, turnkey=-1 (auto)
- `PREFIX_CACHE_DISK_GB`: AIBeast=384, turnkey=0 (we intentionally defaulted to 0)
- `SERVED_MODEL_NAME`: AIBeast='GLM-5.2 local-primary', turnkey='GLM-5.2'
- `LMCACHE_L1_INIT_GB`: AIBeast=125, turnkey=min(20, L1) (lazy growth default)
- `SUPERVISOR_MAX_RESTARTS`: AIBeast=2, turnkey=5 (appliance default)

### L2 eviction (already ported earlier this session)
- `LMCACHE_L2_EVICTION_POLICY`, `LMCACHE_L2_EVICTION_TRIGGER_WATERMARK`,
  `LMCACHE_L2_EVICTION_RATIO`, `LMCACHE_RETRIEVE_TIMEOUT_SECONDS` — ported from
  r34 maintenance overlay (commit `f657dbc`)

### Conclusion
The turnkey appliance now has all production hardening from AIBeast. The remaining
differences are intentional deployment-specific overrides that operators set via
their env file. The entrypoint and glm_config.py handle all the global tuning vars.

### Commits pushed this session (5 total, all on GitHub)
| Commit | Description |
|--------|-------------|
| `6981d2e` | PREFIX_CACHE_DISK_FRACTION knob + default PREFIX_CACHE_DISK_GB → 0 |
| `53e6dce` | UX: suppress duplicate config dump, fix void message, 5% download intervals, JarvisLabs detection |
| `03d3448` | README: docker-compose example + P2P prerequisite docs |
| `f657dbc` | Port L2 eviction + retrieve_timeout from r34 maintenance overlay |
| `6568ff6` | Add VLLM_PROMPT_LOGPROBS_CHUNK_SIZE to GLM52 runtime env |
