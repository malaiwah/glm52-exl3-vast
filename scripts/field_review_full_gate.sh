#!/usr/bin/env bash
# Reproducible one-process GLM-5.2 EXL3 field-review gate.
#
# This intentionally starts one server and never supervises it.  The caller
# owns health polling, the workload, evidence capture, and teardown.  Use a
# distinct RUN_LABEL and JIT_ROOT for every baseline/candidate/capacity run.
set -euo pipefail

RUN_LABEL="${RUN_LABEL:?set RUN_LABEL to a unique evidence label}"
MODE="${MODE:-baseline}"
PORT="${PORT:-18000}"
JIT_ROOT="${JIT_ROOT:-/workspace/field-review-tests/jit/$RUN_LABEL}"
MODEL_ROOT="${MODEL_ROOT:-/workspace}"
CANDIDATE_PYTHONPATH="${CANDIDATE_PYTHONPATH:-}"
PREFILL_CAPACITY="${PREFILL_CAPACITY:-unset}"
MTP_TOKENS_VALUE="${MTP_TOKENS_VALUE:-0}"
GPU_MEMORY_UTILIZATION_VALUE="${GPU_MEMORY_UTILIZATION_VALUE:-0.90}"
MAX_MODEL_LEN_VALUE="${MAX_MODEL_LEN_VALUE:-131072}"
MAX_NUM_SEQS_VALUE="${MAX_NUM_SEQS_VALUE:-1}"
MAX_NUM_BATCHED_TOKENS_VALUE="${MAX_NUM_BATCHED_TOKENS_VALUE:-3072}"
MAX_CUDAGRAPH_CAPTURE_SIZE_VALUE="${MAX_CUDAGRAPH_CAPTURE_SIZE_VALUE:-6}"
CUDAGRAPH_CAPTURE_SIZES_VALUE="${CUDAGRAPH_CAPTURE_SIZES_VALUE:-1,2,3,4,5,6}"
VLLM_EXL3_TRELLIS_MAX_M_VALUE="${VLLM_EXL3_TRELLIS_MAX_M_VALUE:-6}"
GPU_BLOCKS_OVERRIDE_VALUE="${GPU_BLOCKS_OVERRIDE_VALUE:-0}"
OFFLOAD_FRACTION_VALUE="${OFFLOAD_FRACTION_VALUE:-0}"
PREFIX_CACHE_BACKEND_VALUE="${PREFIX_CACHE_BACKEND_VALUE:-lmcache}"
PREFIX_CACHE_DISK_GB_VALUE="${PREFIX_CACHE_DISK_GB_VALUE:-0}"
VISION_VALUE="${VISION_VALUE:-0}"
FIELD_REVIEW_SCRIPTS_DIR="${FIELD_REVIEW_SCRIPTS_DIR:-/workspace/field-review-tests/turnkey-scripts}"
FIELD_REVIEW_STATE_DIR="${FIELD_REVIEW_STATE_DIR:-/tmp/field-review-$RUN_LABEL/state}"

# An interactive SSH shell on a rental does not necessarily inherit the OCI
# image's virtualenv-first PATH even though PID 1 did.  Pin it here so the
# candidate import and the entrypoint's Python helpers use the same torch/vLLM
# environment as serving.
export PATH="/opt/venv/bin:$PATH"
if [ -d "$FIELD_REVIEW_SCRIPTS_DIR" ]; then
  export SCRIPTS_DIR="$FIELD_REVIEW_SCRIPTS_DIR"
fi

case "$MODE" in
  baseline)
    if [ -n "$CANDIDATE_PYTHONPATH" ]; then
      echo "FATAL: baseline mode cannot use CANDIDATE_PYTHONPATH" >&2
      exit 2
    fi
    ;;
  candidate)
    if [ -z "$CANDIDATE_PYTHONPATH" ] ||
       [ ! -d "$CANDIDATE_PYTHONPATH/vllm" ]; then
      echo "FATAL: candidate mode needs a vLLM worktree in CANDIDATE_PYTHONPATH" >&2
      exit 2
    fi
    ;;
  *)
    echo "FATAL: MODE must be baseline or candidate" >&2
    exit 2
    ;;
esac

case "$PREFILL_CAPACITY" in
  unset) unset VLLM_EXL3_PREFILL_CAPACITY ;;
  *[!0-9]*|"")
    echo "FATAL: PREFILL_CAPACITY must be unset or a positive integer" >&2
    exit 2
    ;;
  0)
    echo "FATAL: PREFILL_CAPACITY must be greater than zero" >&2
    exit 2
    ;;
  *) export VLLM_EXL3_PREFILL_CAPACITY="$PREFILL_CAPACITY" ;;
esac

case "$MTP_TOKENS_VALUE" in
  *[!0-9]*|"")
    echo "FATAL: MTP_TOKENS_VALUE must be a non-negative integer" >&2
    exit 2
    ;;
esac

for positive_setting in \
  MAX_MODEL_LEN_VALUE \
  MAX_NUM_SEQS_VALUE \
  MAX_NUM_BATCHED_TOKENS_VALUE \
  MAX_CUDAGRAPH_CAPTURE_SIZE_VALUE \
  VLLM_EXL3_TRELLIS_MAX_M_VALUE; do
  positive_value="${!positive_setting}"
  case "$positive_value" in
    *[!0-9]*|""|0)
      echo "FATAL: $positive_setting must be a positive integer" >&2
      exit 2
      ;;
  esac
done

case "$GPU_BLOCKS_OVERRIDE_VALUE" in
  *[!0-9]*|"")
    echo "FATAL: GPU_BLOCKS_OVERRIDE_VALUE must be a non-negative integer" >&2
    exit 2
    ;;
esac

case "$PREFIX_CACHE_DISK_GB_VALUE" in
  *[!0-9]*|"")
    echo "FATAL: PREFIX_CACHE_DISK_GB_VALUE must be a non-negative integer" >&2
    exit 2
    ;;
esac

case "$VISION_VALUE" in
  0|1) ;;
  *)
    echo "FATAL: VISION_VALUE must be 0 or 1" >&2
    exit 2
    ;;
esac

if ! awk -v value="$GPU_MEMORY_UTILIZATION_VALUE" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 1) }'; then
  echo "FATAL: GPU_MEMORY_UTILIZATION_VALUE must be greater than 0 and at most 1" >&2
  exit 2
fi

if ! awk -v value="$OFFLOAD_FRACTION_VALUE" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 0.9) }'; then
  echo "FATAL: OFFLOAD_FRACTION_VALUE must be between 0 and 0.9" >&2
  exit 2
fi

if [ "$PREFILL_CAPACITY" != unset ] &&
   [ "$PREFILL_CAPACITY" -gt "$MAX_NUM_BATCHED_TOKENS_VALUE" ]; then
  echo "FATAL: PREFILL_CAPACITY cannot exceed MAX_NUM_BATCHED_TOKENS_VALUE" >&2
  exit 2
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader |
   grep -q '[0-9]'; then
  echo "FATAL: a GPU process is already running; refusing a polluted gate" >&2
  exit 3
fi

mkdir -p "$JIT_ROOT" \
  "$FIELD_REVIEW_STATE_DIR" \
  "/tmp/field-review-$RUN_LABEL/runtime"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_EXTENSIONS_DIR="$JIT_ROOT/torch-extensions"
export VLLM_CACHE_ROOT="$JIT_ROOT/vllm"
export TRITON_CACHE_DIR="$JIT_ROOT/triton"
export XDG_CACHE_HOME="$JIT_ROOT/xdg"
export GLM_STATE_DIR="$FIELD_REVIEW_STATE_DIR"
export GLM_RUNTIME_DIR="/tmp/field-review-$RUN_LABEL/runtime"

# SSH-launched processes inside the retained appliance do not inherit every
# OCI environment value from PID 1. Reconstitute the two immutable runtime
# paths needed by the release image so an MTP/source gate exercises the same
# EXL3 extension and NCCL shim as a normal appliance boot.
if [ -z "${VLLM_EXL3_EXT_PATH:-}" ] &&
   [ -d /opt/exllamav3 ]; then
  export VLLM_EXL3_EXT_PATH=/opt/exllamav3
fi
if [ -z "${LD_PRELOAD:-}" ] &&
   [ -e /opt/libnccl-local-inference.so.2.30.4 ]; then
  export LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
fi

if [ "$MODE" = "candidate" ]; then
  export PYTHONPATH="$CANDIDATE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
  # Preserve the exact reviewed vLLM tree. The turnkey compatibility patch is
  # enabled by default in every appliance boot, but mutating a PR worktree
  # would invalidate exact-SHA attribution for this source-review gate.
  export STRUCTURED_OUTPUT_SPEC_PATCH=0
else
  unset PYTHONPATH
  unset STRUCTURED_OUTPUT_SPEC_PATCH
fi

# Isolate the benchmark from all rental-facing auxiliaries and credentials.
export MODEL_ROOT MODEL_READ_ONLY=1
export MODEL_PROFILE=glm52-exl3 MODEL_FAMILY=glm52
export MODEL_VARIANT=exl3-tr3-3.25bpw
export MODEL_ID=
export MODEL_REVISION=d7d79c2d14599dfce7a5d12b85f7ad73f40e623d
export MODEL_REPO=willfalco/GLM-5.2-EXL3-TR3-3.25bpw
export TENSOR_PARALLEL_SIZE=4 DCP=4
export MTP_TOKENS="$MTP_TOKENS_VALUE" MTP_DRAFT_SAMPLE_METHOD=probabilistic
export MAX_MODEL_LEN="$MAX_MODEL_LEN_VALUE"
export MAX_NUM_SEQS="$MAX_NUM_SEQS_VALUE"
export MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS_VALUE"
export MAX_CUDAGRAPH_CAPTURE_SIZE="$MAX_CUDAGRAPH_CAPTURE_SIZE_VALUE"
export CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES_VALUE"
export VLLM_EXL3_TRELLIS_MAX_M="$VLLM_EXL3_TRELLIS_MAX_M_VALUE"
export GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION_VALUE"
export GPU_BLOCKS_OVERRIDE="$GPU_BLOCKS_OVERRIDE_VALUE"
export KV_CACHE_DTYPE=nvfp4_ds_mla KV_SCALE_MODE=dynamic-token
export OFFLOAD_FRACTION="$OFFLOAD_FRACTION_VALUE"
export PREFIX_CACHE_BACKEND="$PREFIX_CACHE_BACKEND_VALUE"
export PREFIX_CACHE_DISK_GB="$PREFIX_CACHE_DISK_GB_VALUE"
export VISION="$VISION_VALUE"
export DCP_CKV_GATHER_MAX_TOKENS=140000
export DCP_PREFILL_WORKSPACE_MIB=1024
export DCP_CKV_PREFETCH_DEPTH=0
export PCIE_CALIBRATION=auto
export LOAD_FORMAT=safetensors
export AUTH=none SERVED_MODEL_NAME=GLM-5.2
export LANDING_PAGE=0 SSHD=0 SOUL_LEVEL=0
export SUPERVISOR=0 VERIFY=0
export STATUS_FILE="/tmp/field-review-$RUN_LABEL/status.json"
export PORT

# The retained r14 appliance predates the r610 "CUDA UMD Version" parser fix
# already present in this repository, so its installed entrypoint reports this
# qualified 610.43.02 / 13.3 host as CUDA "unknown".  The field gate records
# the actual nvidia-smi header and bypasses only that stale admission parser.
export ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1

unset DESEC_TOKEN DESEC_DOMAIN ACME_DOMAIN ACME_DNS_PROVIDER
unset CONTAINER_API_KEY RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset VLLM_API_KEY

echo ">>> field-review run: label=$RUN_LABEL mode=$MODE capacity=$PREFILL_CAPACITY mtp=$MTP_TOKENS"
echo ">>> serving shape: model_len=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS batch=$MAX_NUM_BATCHED_TOKENS graph=$MAX_CUDAGRAPH_CAPTURE_SIZE"
echo ">>> cache shape: gpu_blocks=$GPU_BLOCKS_OVERRIDE offload_fraction=$OFFLOAD_FRACTION disk_gib=$PREFIX_CACHE_DISK_GB"
echo ">>> JIT/cache root: $JIT_ROOT"
echo ">>> EXL3 extension path: ${VLLM_EXL3_EXT_PATH:-unset}"
echo ">>> NCCL preload: ${LD_PRELOAD:-unset}"
if [ "$MODE" = "candidate" ]; then
  python3 - <<'PY'
import lmcache
import sparkinfer
import vllm
print(f">>> candidate LMCache import: {lmcache.__file__}")
print(f">>> candidate SparkInfer import: {sparkinfer.__file__}")
print(f">>> candidate vLLM import: {vllm.__file__}")
PY
fi

# Keep the complete entrypoint/vLLM process tree in a dedicated process group
# so the caller can tear a gate down by signaling the recorded wrapper PID.
# The stock entrypoint does not forward TERM to its serving child when it is
# launched directly from an SSH/nohup gate.
command -v setsid >/dev/null || {
  echo "FATAL: setsid is required for an isolated field-review process group" >&2
  exit 4
}
setsid /usr/local/bin/model-turnkey-entry.sh &
service_pid=$!
trap 'kill -TERM -- "-$service_pid" 2>/dev/null || true' TERM INT
wait "$service_pid"
