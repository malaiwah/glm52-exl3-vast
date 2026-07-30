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
DRAFT_MODEL_HOST="${DRAFT_MODEL_HOST:-}"
DRAFT_MODEL_CONTAINER="${DRAFT_MODEL_CONTAINER:-/models/mtp78-draft}"
VISION_ASSET_HOST="${VISION_ASSET_HOST:-}"
VISION_MARKER_HOST="${VISION_MARKER_HOST:-}"
TOKENIZER_JSON_HOST="${TOKENIZER_JSON_HOST:-}"
SHARED_MODEL_STORE_HOST="${SHARED_MODEL_STORE_HOST:-}"
SHARED_MODEL_STORE_CONTAINER="${SHARED_MODEL_STORE_CONTAINER:-$SHARED_MODEL_STORE_HOST}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
CACHE_VOLUME="${CACHE_VOLUME:-glm52-turnkey-cache}"
STATE_VOLUME="${STATE_VOLUME:-glm52-turnkey-state}"
LMCACHE_DISK_HOST="${LMCACHE_DISK_HOST:-}"
restart_policy="${RESTART_POLICY:-unless-stopped}"
if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  restart_policy=no
fi

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

draft_mounts=()
draft_env=()
if [ -n "$DRAFT_MODEL_HOST" ]; then
  [ -f "$DRAFT_MODEL_HOST/config.json" ] &&
    [ -f "$DRAFT_MODEL_HOST/model.safetensors.index.json" ] || {
    echo "FATAL: external draft is incomplete: $DRAFT_MODEL_HOST" >&2
    exit 4
  }
  draft_mounts=(-v "$DRAFT_MODEL_HOST:$DRAFT_MODEL_CONTAINER:ro")
  draft_env=(-e DRAFT_MODEL="$DRAFT_MODEL_CONTAINER")
fi

vision_mounts=()
if [ -n "$VISION_ASSET_HOST" ] || [ -n "$VISION_MARKER_HOST" ]; then
  [ -n "$VISION_ASSET_HOST" ] && [ -n "$VISION_MARKER_HOST" ] || {
    echo "FATAL: set both VISION_ASSET_HOST and VISION_MARKER_HOST" >&2
    exit 4
  }
  [ -d "$VISION_ASSET_HOST" ] || {
    echo "FATAL: prepared vision derivative does not exist: $VISION_ASSET_HOST" >&2
    exit 4
  }
  [ -f "$VISION_MARKER_HOST" ] || {
    echo "FATAL: vision marker does not exist: $VISION_MARKER_HOST" >&2
    exit 4
  }
  vision_mounts=(
    -v "$VISION_ASSET_HOST:$MODEL_DIR_CONTAINER/.vision:ro"
    -v "$VISION_MARKER_HOST:$MODEL_DIR_CONTAINER/.vision-enabled:ro"
  )
fi

tokenizer_mounts=()
if [ -n "$TOKENIZER_JSON_HOST" ]; then
  [ -f "$TOKENIZER_JSON_HOST" ] || {
    echo "FATAL: tokenizer serialization does not exist: $TOKENIZER_JSON_HOST" >&2
    exit 4
  }
  tokenizer_mounts=(-v "$TOKENIZER_JSON_HOST:$MODEL_DIR_CONTAINER/tokenizer.json:ro")
fi

shared_store_mounts=()
if [ -n "$SHARED_MODEL_STORE_HOST" ]; then
  [ -d "$SHARED_MODEL_STORE_HOST" ] || {
    echo "FATAL: shared model store does not exist: $SHARED_MODEL_STORE_HOST" >&2
    exit 4
  }
  case "$SHARED_MODEL_STORE_CONTAINER" in
    /*) ;;
    *)
      echo "FATAL: SHARED_MODEL_STORE_CONTAINER must be an absolute path" >&2
      exit 4
      ;;
  esac
  shared_store_mounts=(-v "$SHARED_MODEL_STORE_HOST:$SHARED_MODEL_STORE_CONTAINER:ro")
fi

lmcache_mounts=()
if [ -n "$LMCACHE_DISK_HOST" ]; then
  case "$LMCACHE_DISK_HOST" in
    /*) ;;
    *)
      echo "FATAL: LMCACHE_DISK_HOST must be an absolute path" >&2
      exit 4
      ;;
  esac
  [ -d "$LMCACHE_DISK_HOST" ] || {
    echo "FATAL: LMCache disk directory does not exist: $LMCACHE_DISK_HOST" >&2
    exit 4
  }
  # Keep the persistent tier beneath the appliance workspace. The secure
  # termination planner already recognizes /workspace/.lmcache as derived,
  # potentially sensitive KV and erases it on an opted-in teardown.
  lmcache_mounts=(-v "$LMCACHE_DISK_HOST:/workspace/.lmcache:rw")
fi

# Advanced runtime A/Bs use the entrypoint's TUNE_<engine-env> convention.
# Forward every explicitly supplied tuning override without teaching this local
# launcher a second, inevitably incomplete list of SparkInfer/vLLM variables.
tuning_env=()
while IFS='=' read -r tune_name tune_value; do
  case "$tune_name" in
    TUNE_[A-Z0-9_]*)
      tuning_env+=(-e "$tune_name=$tune_value")
      ;;
  esac
done < <(env)

# Profile knobs are optional here. Passing a launcher-side fallback for every
# one used to pin the historical MadeBy561 shape over whichever variant the
# operator selected, so local and rental runs silently exercised different
# configurations. The config resolver owns defaults; this helper forwards only
# explicit operator overrides.
config_env=()
for config_name in \
  TENSOR_PARALLEL_SIZE DCP MAX_MODEL_LEN MAX_NUM_SEQS \
  MAX_NUM_BATCHED_TOKENS DCP_PREFILL_WORKSPACE_MIB \
  GPU_MEMORY_UTILIZATION GPU_BLOCKS_OVERRIDE OFFLOAD_FRACTION \
  OFFLOAD_IGNORE_MEMLOCK PREFIX_CACHE_BACKEND PREFIX_CACHE_DISK_GB \
  MTP_DRAFT DRAFT_QUANTIZATION MTP_TOKENS \
  MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD CUDAGRAPH_CAPTURE_SIZES \
  MAX_CUDAGRAPH_CAPTURE_SIZE VLLM_EXL3_TRELLIS_MAX_M \
  KV_CACHE_DTYPE KV_SCALE_MODE LOAD_FORMAT VISION B12X_PCIE_DMA F8_DMA \
  PCIE_CALIBRATION DCP_CKV_PREFETCH_DEPTH DCP_CKV_GATHER_MAX_TOKENS \
  DCP_KV_CACHE_INTERLEAVE_SIZE DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS \
  PCIE_DMA_MIN_BYTES CLAMP_ROPE_TABLES TRANSFORMERS_VERBOSITY
do
  if [ -n "${!config_name:-}" ]; then
    config_env+=(-e "$config_name=${!config_name}")
  fi
done
unset config_name

podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --replace --restart="$restart_policy" \
  --name "$NAME" \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --network host \
  -e MODEL_PROFILE="${MODEL_PROFILE:-glm52-exl3}" \
  -e MODEL_VARIANT="${MODEL_VARIANT:-exl3-tr3}" \
  -e MODEL_DIR="$MODEL_DIR_CONTAINER" \
  -e MODEL_READ_ONLY=1 \
  -e HF_HUB_OFFLINE=1 \
  -e GLM_STATE_DIR=/state/.glm-config \
  -e AUTH="${AUTH:-none}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2 local-primary}" \
  -e PORT="$PORT" -e LANDING_PAGE="${LANDING_PAGE:-0}" \
  -e SUPERVISOR="${SUPERVISOR:-1}" \
  -e SOUL_AUTONOMY_LEVEL="${SOUL_AUTONOMY_LEVEL:-0}" \
  -e SOUL_AUTONOMY_MAX_LEVEL="${SOUL_AUTONOMY_MAX_LEVEL:-0}" \
  -e CONFIG_SMOKE="${CONFIG_SMOKE:-0}" \
  "${config_env[@]}" \
  "${tuning_env[@]}" \
  -v "$MODEL_MOUNT_HOST:/models/checkpoint-root:ro" \
  -v "$DOWNLOAD_MARKER_HOST:$MODEL_DIR_CONTAINER/.download-complete:ro" \
  "${vision_mounts[@]}" \
  "${tokenizer_mounts[@]}" \
  "${shared_store_mounts[@]}" \
  "${lmcache_mounts[@]}" \
  -v "$CACHE_VOLUME:/cache" \
  -v "$STATE_VOLUME:/state" \
  "${cache_mounts[@]}" \
  "${draft_mounts[@]}" \
  "${draft_env[@]}" \
  "$IMAGE"

echo "Launched $NAME from immutable checkpoint $MODEL_DIR_HOST on port $PORT."
echo "Rollback remains the responsibility of the host's preserved control launcher."
