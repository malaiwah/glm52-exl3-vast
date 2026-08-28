#!/usr/bin/env bash
# Reproduce the local-inference-lab GLM-5.2 prompt-logit KLD protocol on one
# four-GPU Blackwell node. This is an offline LLM run: do not share its GPUs
# with a serving engine.
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE to the immutable candidate image}"
RUN_NAME="${RUN_NAME:?set RUN_NAME, for example r9-dynamic-run1}"
SCALE_MODE="${SCALE_MODE:-dynamic-token}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
ONLINE_QUANT="${ONLINE_QUANT:-none}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/fast/models/GLM-5.2-EXL3-TR3-3.0bpw}"
MODEL_MOUNT_ROOT="$MODEL_ROOT"
MODEL_MOUNT_CONTAINER=/models/checkpoint-root
MODEL_CONTAINER_ROOT=/models/checkpoint-root
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/vault/llm/speculator/kld-ref/reference-logits}"
KLD_RUNNER="${KLD_RUNNER:-/mnt/fast/build/rtx6kpro-kld-runner/models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/prefill_kld_fallback.py}"
KLD_PYDEPS="${KLD_PYDEPS:-/mnt/fast/build/kld-pydeps-py312-v2}"
KLD_OUTPUT_ROOT="${KLD_OUTPUT_ROOT:-/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260729-r9-kld}"
HF_DATASET_CACHE="${HF_DATASET_CACHE:-/mnt/fast/build/kld-hf-cache}"
EXL3_PY_PATCH="${EXL3_PY_PATCH:-}"
SPARKINFER_MIXED_TRELLIS_PY="${SPARKINFER_MIXED_TRELLIS_PY:-}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KLD_DCP="${KLD_DCP:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
EXL3_PREFILL_CHUNK="${EXL3_PREFILL_CHUNK:-128}"
EXL3_PREFILL_BLOCK_M="${EXL3_PREFILL_BLOCK_M:-64}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
KLD_CACHE_KEY="$(printf '%s\n%s\n' "$IMAGE" "$MODEL_ROOT" |
  sha256sum | cut -c1-16)"
CACHE_ROOT="${CACHE_ROOT:-/mnt/fast/build/kld-cache-$KLD_CACHE_KEY}"
GPU_ORDER="${GPU_ORDER:-0,1,2,3}"

# Hugging Face snapshots contain relative links into ../../blobs. Binding only
# the revision directory makes a complete host checkpoint look empty inside
# the container. Preserve the repository root and point the runner back at the
# selected immutable snapshot.
case "$MODEL_ROOT" in
  */models--*/snapshots/*)
    hf_model_root="${MODEL_ROOT%%/snapshots/*}"
    hf_snapshot_relative="${MODEL_ROOT#"$hf_model_root"/}"
    if [ -d "$hf_model_root/blobs" ]; then
      MODEL_MOUNT_ROOT="$hf_model_root"
      MODEL_MOUNT_CONTAINER=/models/hf-checkpoint
      MODEL_CONTAINER_ROOT="/models/hf-checkpoint/$hf_snapshot_relative"
    fi
    ;;
esac

case "$CONTAINER_RUNTIME" in
  podman)
    RUNTIME=(podman)
    ;;
  docker)
    if docker info >/dev/null 2>&1; then
      RUNTIME=(docker)
    else
      RUNTIME=(sudo docker)
    fi
    ;;
  *)
    echo "CONTAINER_RUNTIME must be podman or docker" >&2
    exit 2
    ;;
esac

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

case "$KV_CACHE_DTYPE" in
  nvfp4_ds_mla) ;;
  fp8)
    if [ "$KV_FP8_ROPE" != 0 ] ||
       [ "$VLLM_NVFP4_MLA_DYNAMIC_SCALE" != 0 ] ||
       [ -n "$VLLM_NVFP4_MLA_SCALES_FILE" ]; then
      echo "KV_CACHE_DTYPE=fp8 requires BF16 RoPE and no NVFP4 scale mode" >&2
      echo "use SCALE_MODE=uncalibrated-static" >&2
      exit 2
    fi
    ;;
  *)
    echo "KV_CACHE_DTYPE must be nvfp4_ds_mla or fp8" >&2
    exit 2
    ;;
esac

ONLINE_QUANT_CONFIG=""
ONLINE_TRELLIS_BITS=""
ONLINE_CACHE_MODE="off"
ONLINE_ABSORB_BMM=1
case "$ONLINE_QUANT" in
  none)
    ;;
  mxfp8)
    ONLINE_QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","lm_head"]}'
    ;;
  exl3-b6)
    ONLINE_QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
    ONLINE_TRELLIS_BITS=6
    ONLINE_CACHE_MODE=readwrite
    ONLINE_ABSORB_BMM=0
    ;;
  *)
    echo "ONLINE_QUANT must be none, mxfp8, or exl3-b6" >&2
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
if [ -n "$ONLINE_QUANT_CONFIG" ]; then
  LLM_EXTRA="{\"decode_context_parallel_size\":$KLD_DCP,\"moe_backend\":\"b12x\",\"enforce_eager\":true,\"quantization_config\":$ONLINE_QUANT_CONFIG}"
else
  LLM_EXTRA="{\"decode_context_parallel_size\":$KLD_DCP,\"moe_backend\":\"b12x\",\"enforce_eager\":true}"
fi

printf '%s\n' \
  "image=$IMAGE" \
  "scale_mode=$SCALE_MODE" \
  "online_quant=$ONLINE_QUANT" \
  "model=$MODEL_ROOT" \
  "reference=$REFERENCE_ROOT" \
  "tp=4" \
  "dcp=$KLD_DCP" \
  "mtp=0" \
  "kv_cache_dtype=$KV_CACHE_DTYPE" \
  "kv_fp8_rope=$KV_FP8_ROPE" \
  "dynamic_scale=$VLLM_NVFP4_MLA_DYNAMIC_SCALE" \
  "scales_file=$VLLM_NVFP4_MLA_SCALES_FILE" \
  "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
  "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
  "exl3_prefill_chunk=$EXL3_PREFILL_CHUNK" \
  "exl3_prefill_block_m=$EXL3_PREFILL_BLOCK_M" \
  "sparkinfer_mixed_trellis_py=$SPARKINFER_MIXED_TRELLIS_PY" \
  "cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING" \
  "load_format=$LOAD_FORMAT" \
  "cache_key=$KLD_CACHE_KEY" \
  "gpu_order=$GPU_ORDER" \
  "container_runtime=$CONTAINER_RUNTIME" \
  >"$OUTPUT_DIR/config.env"

container_name="glm52-kld-${RUN_NAME//[^A-Za-z0-9_.-]/-}"
"${RUNTIME[@]}" rm -f "$container_name" >/dev/null 2>&1 || true

GPU_DEVICE_ARGS=()
if [[ "$CONTAINER_RUNTIME" == docker ]]; then
  GPU_DEVICE_ARGS=(--gpus all)
else
  for gpu_device in /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 \
                    /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm \
                    /dev/nvidia-uvm-tools /dev/dri/card0 /dev/dri/card1 \
                    /dev/dri/card2 /dev/dri/card3 /dev/dri/renderD128 \
                    /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131; do
    if [ -e "$gpu_device" ]; then
      GPU_DEVICE_ARGS+=(--device "$gpu_device")
    fi
  done
fi

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
if [ -n "$SPARKINFER_MIXED_TRELLIS_PY" ]; then
  if [ ! -f "$SPARKINFER_MIXED_TRELLIS_PY" ]; then
    echo "missing SparkInfer mixed-Trellis patch: $SPARKINFER_MIXED_TRELLIS_PY" >&2
    exit 2
  fi
  PATCH_MOUNT_ARGS+=(
    -v "$SPARKINFER_MIXED_TRELLIS_PY:/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py:ro"
  )
fi

ONLINE_ENV_ARGS=()
if [ "$ONLINE_QUANT" = "exl3-b6" ]; then
  ONLINE_ENV_ARGS=(
    -e VLLM_EXL3_ONLINE_TRELLIS_BITS="$ONLINE_TRELLIS_BITS"
    -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3
    -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online
    -e VLLM_EXL3_ONLINE_CACHE_MODE="$ONLINE_CACHE_MODE"
    -e VLLM_B12X_ABSORB_BMM="$ONLINE_ABSORB_BMM"
  )
elif [ "$ONLINE_QUANT" = "mxfp8" ]; then
  ONLINE_ENV_ARGS=(-e VLLM_B12X_ABSORB_BMM="$ONLINE_ABSORB_BMM")
fi

"${RUNTIME[@]}" run --rm --name "$container_name" \
  "${GPU_DEVICE_ARGS[@]}" \
  "${PATCH_MOUNT_ARGS[@]}" \
  "${ONLINE_ENV_ARGS[@]}" \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "$MODEL_MOUNT_ROOT:$MODEL_MOUNT_CONTAINER:ro" \
  -v "$REFERENCE_ROOT:/kld/ref:ro" \
  -v "$KLD_RUNNER:/kld/prefill_kld_fallback.py:ro" \
  -v "$KLD_PYDEPS:/kld-pydeps:ro" \
  -v "$OUTPUT_DIR:/kld/output:rw" \
  -v "$HF_DATASET_CACHE:/root/.cache/huggingface:rw" \
  -v "$CACHE_ROOT:/cache:rw" \
  -e PYTHONPATH=/kld-pydeps \
  -e CUDA_VISIBLE_DEVICES="$GPU_ORDER" \
  -e CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
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
  -e VLLM_EXL3_PREFILL_BLOCK_M="$EXL3_PREFILL_BLOCK_M" \
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
  --model '$MODEL_CONTAINER_ROOT' \
  --tokenizer '$MODEL_CONTAINER_ROOT' \
  --reference-logits /kld/ref \
  --context-length 2048 \
  --stride 512 \
  --max-windows 1 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
  --dtype bfloat16 \
  --kv-cache-dtype '$KV_CACHE_DTYPE' \
  --load-format '$LOAD_FORMAT' \
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
