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
MODEL_MOUNT_CONTAINER="${MODEL_MOUNT_CONTAINER:-/models/checkpoint-root}"
MODEL_DIR_CONTAINER="${MODEL_DIR_CONTAINER:-$MODEL_MOUNT_CONTAINER}"
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

# A Hugging Face snapshot is normally a directory of relative symlinks into
# ../../blobs. Binding only snapshots/<revision> makes those targets disappear
# inside the container even though every file resolves correctly on the host.
# When the caller supplied the simple/default mount contract, recognize that
# layout and bind the immutable model repository root instead. Explicit
# MODEL_MOUNT_* settings remain authoritative for nonstandard stores.
if [ "$MODEL_MOUNT_HOST" = "$MODEL_DIR_HOST" ] &&
   [ "$MODEL_MOUNT_CONTAINER" = "/models/checkpoint-root" ] &&
   [ "$MODEL_DIR_CONTAINER" = "$MODEL_MOUNT_CONTAINER" ]; then
  case "$MODEL_DIR_HOST" in
    */models--*/snapshots/*)
      hf_model_root="${MODEL_DIR_HOST%%/snapshots/*}"
      hf_snapshot_relative="${MODEL_DIR_HOST#"$hf_model_root"/}"
      if [ -d "$hf_model_root/blobs" ]; then
        MODEL_MOUNT_HOST="$hf_model_root"
        MODEL_MOUNT_CONTAINER=/models/hf-checkpoint
        MODEL_DIR_CONTAINER="$MODEL_MOUNT_CONTAINER/$hf_snapshot_relative"
        echo ">>> local runner: preserving Hugging Face snapshot symlinks via $MODEL_MOUNT_CONTAINER" >&2
      fi
      ;;
  esac
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
# Existence is not integrity. An HF snapshot is a directory of relative
# symlinks into ../../blobs; a pruned blob store (HF cache GC, partial
# download) passes the directory checks above while the container sees
# dangling links and fails minutes later inside vLLM load — the exact failure
# the rebind logic exists to prevent. With -L, `find -type l` reports exactly
# the links whose targets are gone.
[ -r "$MODEL_DIR_HOST/config.json" ] || {
  echo "FATAL: $MODEL_DIR_HOST/config.json is missing or unreadable — this is" >&2
  echo "       not a complete checkpoint (pruned blobs or a broken snapshot)." >&2
  exit 4
}
# No `| head` here: under `set -o pipefail` an early-exiting head SIGPIPEs
# find, the pipeline reports 141, and the guard silently passes in exactly the
# mass-pruned case it exists to catch. -print -quit stops at the first hit.
_dangling="$(find -L "$MODEL_DIR_HOST" -type l -print -quit 2>/dev/null || true)"
if [ -n "$_dangling" ]; then
  echo "FATAL: the checkpoint contains dangling symlinks (pruned HF blobs?):" >&2
  echo "       $_dangling" >&2
  echo "       Re-download the snapshot or point MODEL_DIR_HOST at complete bytes." >&2
  exit 4
fi
unset _dangling

cache_mounts=()
if [ -n "$HF_CACHE_HOST" ]; then
  [ -d "$HF_CACHE_HOST" ] || {
    echo "FATAL: Hugging Face cache does not exist: $HF_CACHE_HOST" >&2
    exit 4
  }
  cache_mounts=(-v "$HF_CACHE_HOST:/root/.cache/huggingface:ro")
fi

draft_mounts=()
# Clear any draft path baked into the image unless this launcher also mounts
# that exact path. Otherwise a native-checkpoint run can silently inherit a
# stale /models/mtp78-draft value from an older image revision.
draft_env=(-e DRAFT_MODEL=)
if [ -n "$DRAFT_MODEL_HOST" ]; then
  if [ ! -f "$DRAFT_MODEL_HOST/config.json" ] ||
      [ ! -f "$DRAFT_MODEL_HOST/model.safetensors.index.json" ]; then
    echo "FATAL: external draft is incomplete: $DRAFT_MODEL_HOST" >&2
    exit 4
  fi
  draft_mounts=(-v "$DRAFT_MODEL_HOST:$DRAFT_MODEL_CONTAINER:ro")
  draft_env=(-e DRAFT_MODEL="$DRAFT_MODEL_CONTAINER")
fi

vision_mounts=()
if [ -n "$VISION_ASSET_HOST" ] || [ -n "$VISION_MARKER_HOST" ]; then
  if [ -z "$VISION_ASSET_HOST" ] || [ -z "$VISION_MARKER_HOST" ]; then
    echo "FATAL: set both VISION_ASSET_HOST and VISION_MARKER_HOST" >&2
    exit 4
  fi
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
  MAX_NUM_BATCHED_TOKENS VLLM_EXL3_PREFILL_CAPACITY DCP_PREFILL_WORKSPACE_MIB \
  GPU_MEMORY_UTILIZATION GPU_BLOCKS_OVERRIDE KV_CACHE_MEMORY_BYTES \
  OFFLOAD_FRACTION \
  OFFLOAD_IGNORE_MEMLOCK PREFIX_CACHE_BACKEND PREFIX_CACHE_DISK_GB \
  LMCACHE_L1_INIT_GB \
  MTP_DRAFT DRAFT_QUANTIZATION MTP_TOKENS \
  MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD CUDAGRAPH_CAPTURE_SIZES \
  MAX_CUDAGRAPH_CAPTURE_SIZE VLLM_EXL3_TRELLIS_MAX_M \
  KV_CACHE_DTYPE KV_SCALE_MODE ONLINE_QUANT LOAD_FORMAT VISION B12X_PCIE_DMA F8_DMA \
  PCIE_CALIBRATION DCP_CKV_PREFETCH_DEPTH DCP_CKV_GATHER_MAX_TOKENS \
  DCP_KV_CACHE_INTERLEAVE_SIZE DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS \
  PCIE_DMA_MIN_BYTES CLAMP_ROPE_TABLES TRANSFORMERS_VERBOSITY
do
  if [ -n "${!config_name:-}" ]; then
    config_env+=(-e "$config_name=${!config_name}")
  fi
done
unset config_name

# LMCache and vLLM both need time to drain background stores and release CUDA
# workers. Podman's default force-removal grace is only ten seconds, which is
# too short for a four-GPU GLM process and can leave extension locks or partial
# L2 records behind. Stop an existing appliance cleanly before replacing it.
if podman container exists "$NAME"; then
  podman stop -t "${STOP_TIMEOUT:-120}" "$NAME" >/dev/null 2>&1 || true
  podman rm -f "$NAME" >/dev/null 2>&1 || true
fi
# The smoke contract is "no GPU touched": requesting devices would fail at
# container create on any host without the NVIDIA container stack, before the
# resolved config ever printed.
gpu_args=(--gpus "\"device=${GPU_DEVICES}\"")
if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  gpu_args=()
fi
# CDI/--gpus controls which devices the container may access, but it does not
# preserve the caller's list as CUDA's logical-rank ordering.  Export the same
# list so GPU_DEVICES can provide deterministic TP-rank placement (for example
# cold-to-hot on an owned host) instead of silently reverting to PCI order.
podman run -d --replace --restart="$restart_policy" \
  --name "$NAME" \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  ${gpu_args[@]+"${gpu_args[@]}"} --ipc=host --network host \
  -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
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
  -v "$MODEL_MOUNT_HOST:$MODEL_MOUNT_CONTAINER:ro" \
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
