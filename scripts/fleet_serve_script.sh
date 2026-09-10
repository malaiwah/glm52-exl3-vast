#!/usr/bin/env bash
# Autonomous GLM-5.3 replica boot for JarvisLabs container instances.
#
# Registered once as a JarvisLabs startup script (jl scripts add); every
# container created with --script-id <id> boots itself with the fleet's
# pinned configuration and NO outside SSH involvement:
#
#   - weights come from the shared filesystem (--fs-id at create time)
#   - the online-quantization cache is ONE shared directory on that same
#     filesystem (/home/jl_fs/.runtimes/exl3-online), not per-slot: its
#     content is identical for every replica, and safety comes from the
#     manager serializing COLD BOOTS — it never creates a second slot
#     while one is still booting
#   - config state and the API key stay on the instance's own disk
#   - the API key is the fleet-wide key (the manager, which is the only
#     thing that needs to know it, bakes it in at registration time)
#
# The slot name arrives as the MACHINE_NAME environment variable.
set -Eeuo pipefail

FLEET_KEY="${GLM_FLEET_KEY:?GLM_FLEET_KEY must be set by the registration step}"

export MODEL_DIR=/home/jl_fs/GLM-5.3-EXL3-TR3-3.42bpw
export GLM_STATE_DIR=/home/turnkey/workspace/.glm-config
export TURNKEY_WORKSPACE=/home/turnkey/workspace
export VLLM_API_KEY="$FLEET_KEY"
# The unpacked appliance image lives on the shared filesystem too: the
# first boot to see it pays the registry fetch + unpack, every later
# instance reuses the tree (fetch_and_unpack honors the .unpacked digest
# marker). The host driver re-injection is idempotent within a region.
export TURNKEY_ROOT=/home/jl_fs/.image/qual
# The online-quantization cache is one shared directory on that same
# filesystem: identical weights, algorithm and GPU architecture make its
# content replica-independent, and the manager serializes cold boots
# so no two replicas ever write it at once.
mkdir -p /home/jl_fs/.runtimes
export VLLM_EXL3_ONLINE_CACHE_DIR=/home/jl_fs/.runtimes/exl3-online
export VLLM_EXL3_ONLINE_CACHE_MODE=readwrite

# The appliance's selected runtime (AIBeast parity): prefill fairness at a
# 60% compute share and the tuned batching shape, with the 48-token
# capture/Trellis window that 12 sequences x 4 MTP tokens require.
export PREFILL_FAIRNESS_ENGINE=compute_share
export PREFILL_COMPUTE_SHARE=0.6
export MAX_NUM_SEQS=12
export MAX_NUM_BATCHED_TOKENS=3072
export VLLM_EXL3_PREFILL_CAPACITY=2048
export GPU_MEMORY_UTILIZATION=0.95
# LMCache prefix tier (AIBeast parity): 125 GiB aggregate host DRAM L1
# plus a bounded NVMe L2 on the instance's own disk (MODEL_ROOT resolves
# to the instance-local workspace, never the shared filesystem). The L2
# has a hard capacity limit with LRU eviction (watermark 0.90, evict 10%),
# so disk usage stays bounded without operator intervention.
export PREFIX_CACHE_BACKEND=lmcache
export LMCACHE_L1_MAX_GB=125
export PREFIX_CACHE_DISK_GB=384
# trellis windows must be raised together or decode silently leaves the
# captured fast path under concurrency (glm_config rule concurrency-window).
export CUDAGRAPH_CAPTURE_SIZES=4,8,12,16,20,24,28,32,36,40,44,48
export MAX_CUDAGRAPH_CAPTURE_SIZE=48
export VLLM_EXL3_TRELLIS_MAX_M=48
# Download to a file and exec only on success: process substitution
# discards curl's exit status, so a truncated download would exec a
# truncated script. Bounded, fast retries: the startup script fires the
# moment the instance boots, when DNS may not be up yet; curl's default
# backoff would then sit in minutes-long sleeps between retries.
curl -fsSL --connect-timeout 10 --max-time 60 --retry 10 --retry-delay 2 \
  --retry-all-errors \
  "https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/cdfbbe41c2913b161f20e3e42ce897b705d66667/scripts/jarvislabs_quickstart.sh" \
  -o /tmp/quickstart.sh || exit 1
exec bash /tmp/quickstart.sh
