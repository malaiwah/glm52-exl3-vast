#!/usr/bin/env bash
set -euo pipefail

readonly root=/mnt/fast/build/r34-aibeast-read-lease-20260821
readonly name=glm52-turnkey-r34-read-lease-20260821-v2
readonly image=localhost/glm52-turnkey:r34-aibeast-read-lease-v2
readonly env_file="$root/production.env"
readonly state="$root/runtime/state"
readonly cache="$root/runtime/compile-cache"
readonly lmcache="$root/runtime/lmcache"

mkdir -p "$state/.glm-config" "$cache" "$lmcache/l2"
if [[ ! -f "$state/.glm-config/checkpoint-baseline.json" ]]; then
  cp /mnt/fast/build/20260809-glm52-prefill/state-r33/.glm-config/checkpoint-baseline.json \
    "$state/.glm-config/checkpoint-baseline.json"
fi

test -f "$env_file"
test -d /mnt/vault/llm/huggingface/models--willfalco--GLM-5.2-EXL3-TR3-3.42bpw
test -f /mnt/fast/turnkey-flags/glm52-342-a350292.download-complete
podman container exists "$name" && {
  echo "FATAL: candidate container already exists: $name" >&2
  exit 1
}

podman run -d \
  --name "$name" \
  --device /dev/nvidia0 --device /dev/nvidia1 \
  --device /dev/nvidia2 --device /dev/nvidia3 \
  --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
  --device /dev/dri/card0 --device /dev/dri/card1 \
  --device /dev/dri/card2 --device /dev/dri/card3 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --device /dev/dri/renderD130 --device /dev/dri/renderD131 \
  --network host --ipc host --init --restart no \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  --health-cmd 'curl -sf http://localhost:8000/health || exit 1' \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 90m \
  --env-file "$env_file" \
  -v "$state:/state:rw" \
  -v /mnt/vault/llm/huggingface/models--willfalco--GLM-5.2-EXL3-TR3-3.42bpw:/models/hf-checkpoint:ro \
  -v /mnt/fast/turnkey-flags/glm52-342-a350292.download-complete:/models/hf-checkpoint/snapshots/a350292cb2038f2c31732569a711a89e5d72fd46/.download-complete:ro \
  -v "$lmcache:/workspace/.lmcache:rw" \
  -v "$cache:/cache:rw" \
  "$image"
