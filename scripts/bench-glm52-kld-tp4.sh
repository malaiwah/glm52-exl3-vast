#!/usr/bin/env bash
# Reproduce the local-inference-lab GLM-5.2 prompt-logit KLD protocol on one
# four-GPU Blackwell node. This is an offline LLM run: do not share its GPUs
# with a serving engine.
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE to the immutable candidate image}"
RUN_NAME="${RUN_NAME:?set RUN_NAME, for example r9-dynamic-run1}"
SCALE_MODE="${SCALE_MODE:-dynamic-token}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/fast/models/GLM-5.2-EXL3-TR3-3.0bpw}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/vault/llm/speculator/kld-ref/reference-logits}"
KLD_RUNNER="${KLD_RUNNER:-/mnt/fast/build/rtx6kpro-kld-master/models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/prefill_kld_fallback.py}"
KLD_PYDEPS="${KLD_PYDEPS:-/mnt/fast/build/kld-pydeps}"
KLD_OUTPUT_ROOT="${KLD_OUTPUT_ROOT:-/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260729-r9-kld}"
HF_DATASET_CACHE="${HF_DATASET_CACHE:-/mnt/fast/build/kld-hf-cache}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/fast/build/kld-cache-r9}"
EXL3_PY_PATCH="${EXL3_PY_PATCH:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
EXL3_PREFILL_CHUNK="${EXL3_PREFILL_CHUNK:-128}"

case "$SCALE_MODE" in
  uncalibrated-static)
    KV_FP8_ROPE=0
    VLLM_NVFP4_MLA_DYNAMIC_SCALE=0
    VLLM_NVFP4_MLA_SCALES_FILE=""
    ;;
  static-calibrated)
    KV_FP8_ROPE=0
    VLLM_NVFP4_MLA_DYNAMIC_SCALE=0
    VLLM_NVFP4_MLA_SCALES_FILE=/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json
    ;;
  dynamic-token)
    KV_FP8_ROPE=1
    VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
    VLLM_NVFP4_MLA_SCALES_FILE=""
    ;;
  *)
    echo "SCALE_MODE must be uncalibrated-static, static-calibrated, or dynamic-token" >&2
    exit 2
    ;;
esac

mkdir -p "$CACHE_ROOT"
for required in "$MODEL_ROOT" "$REFERENCE_ROOT/logits_0.safetensors" \
                "$KLD_RUNNER" "$KLD_PYDEPS"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

OUTPUT_DIR="$KLD_OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR" "$HF_DATASET_CACHE"
chmod 700 "$OUTPUT_DIR"

PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"$PATTERN\"}"
LLM_EXTRA='{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true}'

printf '%s\n' \
  "image=$IMAGE" \
  "scale_mode=$SCALE_MODE" \
  "model=$MODEL_ROOT" \
  "reference=$REFERENCE_ROOT" \
  "tp=4" \
  "dcp=1" \
  "mtp=0" \
  "kv_cache_dtype=nvfp4_ds_mla" \
  "kv_fp8_rope=$KV_FP8_ROPE" \
  "dynamic_scale=$VLLM_NVFP4_MLA_DYNAMIC_SCALE" \
  "scales_file=$VLLM_NVFP4_MLA_SCALES_FILE" \
  "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
  "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
  "exl3_prefill_chunk=$EXL3_PREFILL_CHUNK" \
  >"$OUTPUT_DIR/config.env"

container_name="glm52-kld-${RUN_NAME//[^A-Za-z0-9_.-]/-}"
podman rm -f "$container_name" >/dev/null 2>&1 || true

GPU_DEVICE_ARGS=()
for gpu_device in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 \
                  /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm \
                  /dev/nvidia-uvm-tools /dev/dri/card0 /dev/dri/card1 \
                  /dev/dri/card2 /dev/dri/card3 /dev/dri/renderD128 \
                  /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131; do
  if [ -e "$gpu_device" ]; then
    GPU_DEVICE_ARGS+=(--device "$gpu_device")
  fi
done

PATCH_MOUNT_ARGS=()
if [ -n "$EXL3_PY_PATCH" ]; then
  if [ ! -f "$EXL3_PY_PATCH" ]; then
    echo "missing EXL3 Python patch: $EXL3_PY_PATCH" >&2
    exit 2
  fi
  PATCH_MOUNT_ARGS+=(
    -v "$EXL3_PY_PATCH:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro"
  )
fi

podman run --rm --name "$container_name" \
  "${GPU_DEVICE_ARGS[@]}" \
  "${PATCH_MOUNT_ARGS[@]}" \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "$MODEL_ROOT:/models/checkpoint-root:ro" \
  -v "$REFERENCE_ROOT:/kld/ref:ro" \
  -v "$KLD_RUNNER:/kld/prefill_kld_fallback.py:ro" \
  -v "$KLD_PYDEPS:/kld-pydeps:ro" \
  -v "$OUTPUT_DIR:/kld/output:rw" \
  -v "$HF_DATASET_CACHE:/root/.cache/huggingface:rw" \
  -v "$CACHE_ROOT:/cache:rw" \
  -e PYTHONPATH=/kld-pydeps \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
  -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SAFETENSORS_FAST_GPU=1 \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_DCP_A2A_MAX_TOKENS=16 \
  -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB \
  -e VLLM_PCIE_DMA_FP8=0 \
  -e B12X_PCIE_DMA_FP8=0 \
  -e SPARKINFER_PCIE_DMA_FP8=0 \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_W4A8_TINY_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_EXL3_PREFILL_BLOCK_M=64 \
  -e VLLM_EXL3_PREFILL_CHUNK="$EXL3_PREFILL_CHUNK" \
  -e SPARKINFER_INDEXER_TWO_LEVEL_FOLD=auto \
  -e SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB=256 \
  -e KV_FP8_ROPE="$KV_FP8_ROPE" \
  -e VLLM_NVFP4_MLA_DYNAMIC_SCALE="$VLLM_NVFP4_MLA_DYNAMIC_SCALE" \
  -e VLLM_NVFP4_MLA_SCALES_FILE="$VLLM_NVFP4_MLA_SCALES_FILE" \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
python3 /kld/prefill_kld_fallback.py \
  --model /models/checkpoint-root \
  --tokenizer /models/checkpoint-root \
  --reference-logits /kld/ref \
  --context-length 2048 \
  --stride 512 \
  --max-windows 1 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --dtype bfloat16 \
  --kv-cache-dtype nvfp4_ds_mla \
  --load-format instanttensor \
  --max-model-len 4096 \
  --max-num-batched-tokens '$MAX_NUM_BATCHED_TOKENS' \
  --max-num-seqs 1 \
  --quantization exl3 \
  --attention-backend B12X_MLA_SPARSE \
  --hf-overrides '$HF_OVERRIDES' \
  --llm-extra-json '$LLM_EXTRA' \
  --kld-chunk-rows 32 2>&1 | tee /kld/output/kld.log"

python3 - "$OUTPUT_DIR/kld.log" "$OUTPUT_DIR/summary.json" <<'PY'
import json
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
match = re.search(r"fallback_prefill_kld_done\s+(\{.*\})",
                  log_path.read_text(errors="replace"))
if not match:
    raise SystemExit(f"missing KLD result in {log_path}")
summary = json.loads(match.group(1))
summary["log"] = str(log_path)
count = int(summary.get("num_windows", summary.get("n", 1)))
summary["reference_windows"] = count
if count <= 1:
    summary["stddev_note"] = (
        "The reference bundle contains one window; a reported standard "
        "deviation of 0 is structural (n=1), not evidence of zero model variance."
    )
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
