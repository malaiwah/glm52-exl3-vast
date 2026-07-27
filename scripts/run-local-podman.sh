#!/usr/bin/env bash
# Run the same turnkey image on an owned four-GPU Blackwell host without
# downloading or modifying a shared checkpoint.
#
# Required:
#   MODEL_DIR_HOST=/path/to/a/complete/checkpoint
#   DOWNLOAD_MARKER_HOST=/path/to/.download-complete
#
# The checkpoint directory and completion marker are separate read-only bind
# mounts. All mutable state and compilation caches live in Podman volumes.
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/malaiwah/glm52-exl3-vast:latest}"
NAME="${NAME:-glm52-turnkey}"
PORT="${PORT:-8000}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:?set MODEL_DIR_HOST to the prepared checkpoint}"
MODEL_MOUNT_HOST="${MODEL_MOUNT_HOST:-$MODEL_DIR_HOST}"
MODEL_DIR_CONTAINER="${MODEL_DIR_CONTAINER:-/models/checkpoint-root}"
DOWNLOAD_MARKER_HOST="${DOWNLOAD_MARKER_HOST:?set DOWNLOAD_MARKER_HOST}"
HF_CACHE_HOST="${HF_CACHE_HOST:-}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
CACHE_VOLUME="${CACHE_VOLUME:-glm52-turnkey-cache}"
STATE_VOLUME="${STATE_VOLUME:-glm52-turnkey-state}"

[ -d "$MODEL_DIR_HOST" ] || {
  echo "FATAL: checkpoint directory does not exist: $MODEL_DIR_HOST" >&2
  exit 4
}
[ -d "$MODEL_MOUNT_HOST" ] || {
  echo "FATAL: checkpoint mount root does not exist: $MODEL_MOUNT_HOST" >&2
  exit 4
}
[ -f "$DOWNLOAD_MARKER_HOST" ] || {
  echo "FATAL: completion marker does not exist: $DOWNLOAD_MARKER_HOST" >&2
  exit 4
}

cache_mounts=()
if [ -n "$HF_CACHE_HOST" ]; then
  [ -d "$HF_CACHE_HOST" ] || {
    echo "FATAL: Hugging Face cache does not exist: $HF_CACHE_HOST" >&2
    exit 4
  }
  cache_mounts=(-v "$HF_CACHE_HOST:/root/.cache/huggingface:ro")
fi

podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --replace --restart="${RESTART_POLICY:-unless-stopped}" \
  --name "$NAME" \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --network host \
  -e MODEL_PROFILE="${MODEL_PROFILE:-glm52-exl3}" \
  -e MODEL_VARIANT="${MODEL_VARIANT:-madeby561-hybrid}" \
  -e MODEL_DIR="$MODEL_DIR_CONTAINER" \
  -e MODEL_READ_ONLY=1 \
  -e HF_HUB_OFFLINE=1 \
  -e GLM_STATE_DIR=/state/.glm-config \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e AUTH="${AUTH:-none}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2 local-primary}" \
  -e PORT="$PORT" -e LANDING_PAGE="${LANDING_PAGE:-0}" \
  -e SUPERVISOR="${SUPERVISOR:-1}" \
  -e SOUL_AUTONOMY_LEVEL="${SOUL_AUTONOMY_LEVEL:-0}" \
  -e SOUL_AUTONOMY_MAX_LEVEL="${SOUL_AUTONOMY_MAX_LEVEL:-0}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}" \
  -e DCP_PREFILL_WORKSPACE_MIB="${DCP_PREFILL_WORKSPACE_MIB:-512}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}" \
  -e GPU_BLOCKS_OVERRIDE="${GPU_BLOCKS_OVERRIDE:-2048}" \
  -e MTP_DRAFT="${MTP_DRAFT:-bf16}" \
  -e MTP_TOKENS="${MTP_TOKENS:-3}" \
  -e MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}" \
  -e LOAD_FORMAT="${LOAD_FORMAT:-safetensors}" \
  -e VISION="${VISION:-0}" \
  -e B12X_PCIE_DMA="${B12X_PCIE_DMA:-1}" \
  -e F8_DMA="${F8_DMA:-ring}" \
  -e PCIE_CALIBRATION="${PCIE_CALIBRATION:-off}" \
  -e DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-0}" \
  -e DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-8192}" \
  -e PCIE_DMA_MIN_BYTES="${PCIE_DMA_MIN_BYTES:-393216}" \
  -e CLAMP_ROPE_TABLES="${CLAMP_ROPE_TABLES:-1}" \
  ${TUNE_VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:+-e TUNE_VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="$TUNE_VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS"} \
  ${TUNE_VLLM_PCIE_DMA_MIN_BYTES:+-e TUNE_VLLM_PCIE_DMA_MIN_BYTES="$TUNE_VLLM_PCIE_DMA_MIN_BYTES"} \
  -v "$MODEL_MOUNT_HOST:/models/checkpoint-root:ro" \
  -v "$DOWNLOAD_MARKER_HOST:$MODEL_DIR_CONTAINER/.download-complete:ro" \
  -v "$CACHE_VOLUME:/cache" \
  -v "$STATE_VOLUME:/state" \
  "${cache_mounts[@]}" \
  "$IMAGE"

echo "Launched $NAME from immutable checkpoint $MODEL_DIR_HOST on port $PORT."
echo "Rollback remains the responsibility of the host's preserved control launcher."
