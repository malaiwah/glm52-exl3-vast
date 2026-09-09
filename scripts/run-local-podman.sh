#!/usr/bin/env bash
# Run the same turnkey image on an owned four-GPU Blackwell host without
# downloading or modifying a shared checkpoint.
#
# Required:
#   MODEL_DIR_HOST=/path/to/a/complete/checkpoint
#   DOWNLOAD_MARKER_HOST=/path/to/.download-complete
#
# The checkpoint directory and completion marker are separate read-only bind
# mounts. Mutable state and compilation caches use caller-selected volumes/binds.
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/malaiwah/glm52-exl3-vast:latest}"
NAME="${NAME:-glm53-candidate}"
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
GPU_DEVICES="${GPU_DEVICES:-2,1,0,3}"
GPU_DEVICE_MODE="${GPU_DEVICE_MODE:-auto}"
CACHE_VOLUME="${CACHE_VOLUME:-$NAME-cache}"
STATE_VOLUME="${STATE_VOLUME:-$NAME-state}"
LMCACHE_DISK_HOST="${LMCACHE_DISK_HOST:-}"
restart_policy="${RESTART_POLICY:-no}"
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
  MODEL_ID QUANTIZATION REASONING_PARSER TOOL_CALL_PARSER MULTIMODAL MM_MAX_PIXELS \
  REASONING_EFFORT_DEFAULT PREFILL_SCHEDULE_INTERVAL TRUST_REMOTE_CODE VERIFY \
  PREFILL_FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE \
  TENSOR_PARALLEL_SIZE DCP MAX_MODEL_LEN MAX_NUM_SEQS \
  MAX_NUM_BATCHED_TOKENS VLLM_EXL3_PREFILL_CAPACITY DCP_PREFILL_WORKSPACE_MIB \
  GPU_MEMORY_UTILIZATION GPU_BLOCKS_OVERRIDE KV_CACHE_MEMORY_BYTES \
  OFFLOAD_FRACTION \
  OFFLOAD_IGNORE_MEMLOCK PREFIX_CACHE_BACKEND PREFIX_CACHE_DISK_GB \
  LMCACHE_L1_INIT_GB LMCACHE_L1_READ_TTL \
  LMCACHE_L1_MAX_GB \
  LMCACHE_RETRIEVE_TIMEOUT_SECONDS LMCACHE_L2_EVICTION_POLICY \
  LMCACHE_L2_EVICTION_TRIGGER_WATERMARK LMCACHE_L2_EVICTION_RATIO \
  SUPERVISOR_MAX_RESTARTS \
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

# Podman 4.9 supports CDI and explicit host devices, not Docker's --gpus.
# CONFIG_SMOKE must not even discover GPU devices or invoke the NVIDIA hook.
gpu_args=()
if [ "${CONFIG_SMOKE:-0}" != "1" ]; then
  [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "FATAL: GPU_DEVICES must be comma-separated physical GPU indices" >&2
    exit 4
  }
  IFS=',' read -r -a gpu_indices <<<"$GPU_DEVICES"
  cdi_devices=""
  case "$GPU_DEVICE_MODE" in
    auto|cdi)
      if command -v nvidia-ctk >/dev/null 2>&1; then
        cdi_devices="$(nvidia-ctk cdi list 2>/dev/null || true)"
      fi
      cdi_complete=1
      for gpu in "${gpu_indices[@]}"; do
        if ! grep -Fxq "nvidia.com/gpu=$gpu" <<<"$cdi_devices"; then
          cdi_complete=0
        fi
      done
      if [ "$cdi_complete" = 1 ]; then
        GPU_DEVICE_MODE=cdi
      elif [ "$GPU_DEVICE_MODE" = cdi ]; then
        echo "FATAL: requested NVIDIA CDI GPU devices are not registered" >&2
        exit 4
      else
        GPU_DEVICE_MODE=manual
      fi
      ;;
    manual) ;;
    *) echo "FATAL: GPU_DEVICE_MODE must be auto, cdi, or manual" >&2; exit 4 ;;
  esac
  if [ "$GPU_DEVICE_MODE" = cdi ]; then
    for gpu in "${gpu_indices[@]}"; do
      gpu_args+=(--device "nvidia.com/gpu=$gpu")
    done
  else
    host_devices=(/dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset)
    for gpu in "${gpu_indices[@]}"; do
      host_devices+=("/dev/nvidia$gpu")
    done
    # DRM numbering is host-specific; only expose the explicitly inventoried
    # nodes, never an entire /dev directory or a privileged container.
    if [ -n "${GPU_DRM_DEVICES:-}" ]; then
      IFS=',' read -r -a drm_devices <<<"$GPU_DRM_DEVICES"
      for device in "${drm_devices[@]}"; do
        [[ "$device" =~ ^/dev/dri/(card[0-9]+|renderD[0-9]+)$ ]] || {
          echo "FATAL: invalid GPU_DRM_DEVICES entry: $device" >&2; exit 4;
        }
        host_devices+=("$device")
      done
    fi
    for device in "${host_devices[@]}"; do
      [ -c "$device" ] || {
        echo "FATAL: NVIDIA host device is missing: $device (or configure CDI)" >&2
        exit 4
      }
      gpu_args+=(--device "$device")
    done
  fi
fi
# Preserve the physical list as CUDA's logical TP-rank order.
profile_env=(-e MODEL_PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}")
if [ -n "${MODEL_VARIANT:-}" ]; then
  profile_env+=(-e MODEL_VARIANT="$MODEL_VARIANT")
fi
served_name_env=()
if [ -n "${SERVED_MODEL_NAME:-}" ]; then
  served_name_env=(-e SERVED_MODEL_NAME="$SERVED_MODEL_NAME")
fi
profile_identity="${MODEL_VARIANT:-${MODEL_PROFILE:-glm53-3.42bpw-500k}}"

health_start_period="${HEALTH_START_PERIOD:-}"
if [ -z "$health_start_period" ]; then
  case "$profile_identity" in
    exl3-tr3-3.42bpw|exl3-tr3-glm53-3.42bpw|glm53-3.42bpw|exl3-tr3-glm53-3.42bpw-500k|glm53-3.42bpw-500k|exl3-tr3-glm53-3.25bpw|glm53-3.25bpw)
      health_start_period=90m
      ;;
    *) health_start_period=45m ;;
  esac
fi
# Refuse collisions before any destructive action, including config smoke.
# An explicit replacement retains the stopped original under a required name.
existing=0
if podman container exists "$NAME"; then
  existing=1
  if [ "${CONFIG_SMOKE:-0}" = 1 ] || [ "${REPLACE_EXISTING:-0}" != 1 ]; then
    echo "FATAL: container already exists: $NAME; choose a fresh NAME" >&2
    exit 4
  fi
  : "${ROLLBACK_NAME:?explicit replacement requires a fresh ROLLBACK_NAME}"
  if [ "$ROLLBACK_NAME" = "$NAME" ]; then
    echo "FATAL: ROLLBACK_NAME must differ from NAME" >&2
    exit 4
  fi
  if podman container exists "$ROLLBACK_NAME"; then
    echo "FATAL: ROLLBACK_NAME must name an unused container" >&2
    exit 4
  else
    rollback_status=$?
    [ "$rollback_status" = 1 ] || {
      echo "FATAL: Podman could not check rollback name (status $rollback_status)" >&2
      exit 4
    }
  fi
else
  exists_status=$?
  [ "$exists_status" = 1 ] || {
    echo "FATAL: Podman could not check container existence (status $exists_status)" >&2
    exit 4
  }
fi
podman image exists "$IMAGE" || {
  echo "FATAL: image is not present locally: $IMAGE; prepare it separately" >&2
  exit 4
}
if [ "${LAUNCH_PREFLIGHT:-0}" = 1 ]; then
  echo "Preflight passed; no container started or modified."
  exit 0
fi
if [ "$existing" = 1 ]; then
  podman stop -t "${STOP_TIMEOUT:-120}" "$NAME"
  podman rename "$NAME" "$ROLLBACK_NAME"
  echo "Preserved rollback container: $ROLLBACK_NAME" >&2
fi
# CDI supplies driver libraries itself. Config smoke must not invoke GPU hooks.
# Manual device mapping still needs the legacy NVIDIA hook to inject the host
# driver libraries and nvidia-smi, just as the existing AIBeast launch does.
nvidia_visible_devices="$GPU_DEVICES"
if [ "${CONFIG_SMOKE:-0}" = "1" ] || [ "$GPU_DEVICE_MODE" = cdi ]; then
  hooks_dir="$(mktemp -d)"
  trap 'rmdir "$hooks_dir"' EXIT
  gpu_args+=(--hooks-dir="$hooks_dir")
  nvidia_visible_devices=void
fi
podman run -d --pull=never --restart="$restart_policy" \
  --name "$NAME" \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period "$health_start_period" \
  ${gpu_args[@]+"${gpu_args[@]}"} --ipc=host --network host \
  --init --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" -e NVIDIA_VISIBLE_DEVICES="$nvidia_visible_devices" \
  "${profile_env[@]}" \
  -e MODEL_DIR="$MODEL_DIR_CONTAINER" \
  -e MODEL_READ_ONLY=1 \
  -e HF_HUB_OFFLINE=1 \
  -e GLM_STATE_DIR=/state/.glm-config \
  -e AUTH="${AUTH:-none}" \
  "${served_name_env[@]}" \
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
