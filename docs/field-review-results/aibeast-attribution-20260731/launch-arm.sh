#!/usr/bin/env bash
set -euo pipefail

: "${ARM_IMAGE:?set ARM_IMAGE}"
: "${ARM_NAME:?set ARM_NAME}"
: "${CACHE_VOLUME:?set CACHE_VOLUME}"
: "${STATE_VOLUME:?set STATE_VOLUME}"
: "${LMCACHE_ROOT:?set LMCACHE_ROOT}"

PORT="${PORT:-18000}"
PREFILL_CAPACITY="${PREFILL_CAPACITY:-unset}"
MODEL_REPOSITORY=/mnt/vault/llm/huggingface/models--willfalco--GLM-5.2-EXL3-TR3-3.25bpw
MODEL_SNAPSHOT=/models/hf-checkpoint/snapshots/61d2b6b757f6a4ac7098a78d861f2033497532dc
DOWNLOAD_MARKER=/mnt/fast/turnkey-flags/brandon-exl3.download-complete

case "$PREFILL_CAPACITY" in
  unset) prefill_env=() ;;
  *[!0-9]*|""|0)
    echo "PREFILL_CAPACITY must be unset or a positive integer" >&2
    exit 2
    ;;
  *) prefill_env=(-e "VLLM_EXL3_PREFILL_CAPACITY=$PREFILL_CAPACITY") ;;
esac

mkdir -p "$LMCACHE_ROOT"
podman volume exists "$CACHE_VOLUME" || podman volume create "$CACHE_VOLUME" >/dev/null
podman volume exists "$STATE_VOLUME" || podman volume create "$STATE_VOLUME" >/dev/null

if podman container exists "$ARM_NAME"; then
  podman stop -t 120 "$ARM_NAME" >/dev/null 2>&1 || true
  podman rm -f "$ARM_NAME" >/dev/null
fi

podman run -d --restart=no \
  --entrypoint /usr/local/bin/model-turnkey-entry.sh \
  --name "$ARM_NAME" \
  --health-cmd "curl -sf http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s --health-timeout 10s --health-retries 3 \
  --health-start-period 45m \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host \
  -e MODEL_PROFILE=glm52-exl3 \
  -e MODEL_VARIANT=exl3-tr3-3.25bpw \
  -e MODEL_DIR="$MODEL_SNAPSHOT" \
  -e MODEL_READ_ONLY=1 -e HF_HUB_OFFLINE=1 \
  -e GLM_STATE_DIR=/state/.glm-config \
  -e AUTH=none -e SERVED_MODEL_NAME='GLM-5.2 local-primary' \
  -e PORT="$PORT" -e LANDING_PAGE=0 -e SUPERVISOR=1 \
  -e SOUL_AUTONOMY_LEVEL=0 -e SOUL_AUTONOMY_MAX_LEVEL=0 \
  -e TENSOR_PARALLEL_SIZE=4 -e DCP=4 \
  -e MAX_MODEL_LEN=524288 -e MAX_NUM_SEQS=8 \
  -e MAX_NUM_BATCHED_TOKENS=2048 \
  -e GPU_MEMORY_UTILIZATION=0.957 -e GPU_BLOCKS_OVERRIDE=2048 \
  -e OFFLOAD_FRACTION=0.5 \
  -e PREFIX_CACHE_BACKEND=lmcache -e PREFIX_CACHE_DISK_GB=512 \
  -e MTP_DRAFT=native -e MTP_TOKENS=5 \
  -e MTP_DRAFT_SAMPLE_METHOD=probabilistic \
  -e CUDAGRAPH_CAPTURE_SIZES=4,8,12,16,20,24,28,32,36,40,44,48 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=48 -e VLLM_EXL3_TRELLIS_MAX_M=48 \
  -e KV_CACHE_DTYPE=nvfp4_ds_mla -e KV_SCALE_MODE=dynamic-token \
  -e LOAD_FORMAT=safetensors -e VISION=0 \
  -e B12X_PCIE_DMA=1 -e F8_DMA=0 -e PCIE_CALIBRATION=off \
  -e PCIE_DMA_MIN_BYTES=-1 -e CLAMP_ROPE_TABLES=1 \
  -e DCP_CKV_GATHER_MAX_TOKENS=140000 \
  -e DCP_PREFILL_WORKSPACE_MIB=1024 \
  -e DCP_CKV_PREFETCH_DEPTH=0 \
  -e DCP_KV_CACHE_INTERLEAVE_SIZE=64 \
  -e DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=-1 \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  "${prefill_env[@]}" \
  -v "$MODEL_REPOSITORY:/models/hf-checkpoint:ro" \
  -v "$DOWNLOAD_MARKER:$MODEL_SNAPSHOT/.download-complete:ro" \
  -v "$LMCACHE_ROOT:/workspace/.lmcache:rw" \
  -v "$CACHE_VOLUME:/cache" -v "$STATE_VOLUME:/state" \
  "$ARM_IMAGE"

echo "$ARM_NAME launched on :$PORT from $ARM_IMAGE (prefill capacity: $PREFILL_CAPACITY)"
