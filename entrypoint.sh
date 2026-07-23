#!/bin/bash
# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 Blackwell (96GB), TP4/DCP4,
# 512K context, fp8 KV (correct on stock drivers — see evidence gists in labels),
# MTP speculative decode, DRAM KV offload auto-sized to a fraction of host RAM.
# All logs go to stdout (vast.ai console). SSH per vast standards works alongside.
set -e

echo "=== GLM-5.2 EXL3 turnkey ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -4 || true
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 4 ] || { echo "FATAL: need 4 GPUs, found $NGPU"; exit 1; }

MODEL_DIR="${MODEL_DIR:-/workspace/GLM-5.2-EXL3-TR3-3.0bpw}"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo ">>> First boot: downloading EXL3 weights (~332 GB) to $MODEL_DIR"
  echo ">>> (set HF_TOKEN env for higher rate limits)"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw', local_dir='$MODEL_DIR', max_workers=16)"
  echo ">>> Weights ready."
fi

# DRAM KV offload: OFFLOAD_FRACTION of host RAM (default 0.70); OFFLOAD_FRACTION=0 disables.
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-0.70}"
KVT_ARGS=()
if [ "$OFFLOAD_FRACTION" != "0" ]; then
  MEMLOCK_KB=$(ulimit -l)
  if [ "$MEMLOCK_KB" != "unlimited" ] && [ "$MEMLOCK_KB" -lt 67108864 ] 2>/dev/null; then
    echo "!!! WARNING: memlock ulimit is ${MEMLOCK_KB}KB — too low to pin a DRAM KV pool."
    echo "!!! Add '--ulimit memlock=-1:-1' to the template Docker options to enable offload."
    echo "!!! Continuing WITHOUT DRAM offload."
    OFFLOAD_FRACTION=0
  fi
fi
if [ "$OFFLOAD_FRACTION" != "0" ]; then
  MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  OFF_BYTES=$(python3 -c "print(int($MEM_KB*1024*$OFFLOAD_FRACTION))")
  echo ">>> DRAM KV offload: $((OFF_BYTES/1073741824)) GiB (${OFFLOAD_FRACTION} of host RAM)"
  KVT_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF_BYTES}}")
  # OffloadingConnector rejects expandable_segments (VMM can remap pinned KV pages)
  export PYTORCH_CUDA_ALLOC_CONF=""
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

export CUDA_DEVICE_MAX_CONNECTIONS=32 CUTE_DSL_ARCH=sm_120a OMP_NUM_THREADS=16
export SAFETENSORS_FAST_GPU=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple
export VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_USE_B12X_FP8_GEMM=1 VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_B12X_MOE=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
export VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0
export VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0
export VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_FUSED_MOE_GROUPED_TOPK=1
export VLLM_USE_B12X_MHC=1 B12X_MHC_MAX_TOKENS=16384 VLLM_USE_B12X_WO_PROJECTION=1
export B12X_MLA_SM120_UNIFIED=1 B12X_DENSE_SPLITK_TURBO=1 B12X_W4A16_TC_DECODE=1 B12X_MOE_FORCE_A16=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0 VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=8
export VLLM_USE_B12X_DCP_A2A=1 VLLM_DCP_A2A_MAX_TOKENS=16 VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
export VLLM_DCP_GLOBAL_TOPK=1 VLLM_DCP_SHARD_DRAFT=1 VLLM_DCP_QUERY_SPLIT=0
export VLLM_B12X_MLA_CKV_GATHER=1 VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=512 VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=16384
export VLLM_EXL3_TRELLIS_MIN_M=4 VLLM_EXL3_TRELLIS_MAX_M=32 VLLM_EXL3_TRELLIS_BLOCK_M=8 VLLM_EXL3_PREFILL_CHUNK=128
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export TORCH_CUDA_ARCH_LIST=12.0a FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=2400
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

# API key: use VLLM_API_KEY env, or auto-generate and print to console (vast UI logs)
if [ -z "${VLLM_API_KEY:-}" ]; then
  VLLM_API_KEY="sk-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  echo "=================================================================="
  echo ">>> API KEY (auto-generated; set VLLM_API_KEY env to override):"
  echo ">>> $VLLM_API_KEY"
  echo "=================================================================="
fi
export VLLM_API_KEY

# TLS via Let's Encrypt DNS-01 (optional): set ACME_DOMAIN + ACME_DNS_PROVIDER
# (lego provider name, e.g. cloudflare, duckdns) + the provider's cred envs
# (e.g. CLOUDFLARE_DNS_API_TOKEN, or DUCKDNS_TOKEN). Cert issued each boot.
TLS_ARGS=()
if [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ] && command -v lego >/dev/null; then
  echo ">>> Issuing LetsEncrypt cert for $ACME_DOMAIN via DNS-01 ($ACME_DNS_PROVIDER)"
  lego --accept-tos --email "${ACME_EMAIL:-admin@$ACME_DOMAIN}"        --dns "$ACME_DNS_PROVIDER" --domains "$ACME_DOMAIN"        --path /workspace/.lego run || echo "!!! ACME issuance failed; continuing WITHOUT TLS"
  CRT="/workspace/.lego/certificates/${ACME_DOMAIN}.crt"
  KEY="/workspace/.lego/certificates/${ACME_DOMAIN}.key"
  [ -f "$CRT" ] && TLS_ARGS=(--ssl-certfile "$CRT" --ssl-keyfile "$KEY") && echo ">>> TLS enabled: https://$ACME_DOMAIN:<mapped-port>/v1"
fi

# Egress hygiene: no telemetry; offline mode once weights are local
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
echo ">>> Listening sockets at boot (expect only vllm on ${PORT:-8000} + vast ssh):"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | head -8 || true

MTP_TOKENS="${MTP_TOKENS:-3}"
SPEC_ARGS=()
if [ "$MTP_TOKENS" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS,\"moe_backend\":\"triton\",\"draft_sample_method\":\"probabilistic\"}")
fi

exec vllm serve "$MODEL_DIR" \
  --served-model-name "${SERVED_MODEL_NAME:-GLM-5.2}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
  --tensor-parallel-size 4 --decode-context-parallel-size 4 \
  --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 64 \
  --quantization exl3 --kv-cache-dtype fp8 \
  --attention-backend B12X_MLA_SPARSE --moe-backend b12x --load-format safetensors \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":true}}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}" \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-3072}" \
  --max-cudagraph-capture-size 32 \
  --num-gpu-blocks-override 2048 \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --no-async-scheduling \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  --api-key "$VLLM_API_KEY" \
  "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}"
