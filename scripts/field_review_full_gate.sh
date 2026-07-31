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
FIELD_REVIEW_SCRIPTS_DIR="${FIELD_REVIEW_SCRIPTS_DIR:-/workspace/field-review-tests/turnkey-scripts}"

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

if nvidia-smi --query-compute-apps=pid --format=csv,noheader |
   grep -q '[0-9]'; then
  echo "FATAL: a GPU process is already running; refusing a polluted gate" >&2
  exit 3
fi

mkdir -p "$JIT_ROOT" \
  "/tmp/field-review-$RUN_LABEL/state" \
  "/tmp/field-review-$RUN_LABEL/runtime"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_EXTENSIONS_DIR="$JIT_ROOT/torch-extensions"
export VLLM_CACHE_ROOT="$JIT_ROOT/vllm"
export TRITON_CACHE_DIR="$JIT_ROOT/triton"
export XDG_CACHE_HOME="$JIT_ROOT/xdg"
export GLM_STATE_DIR="/tmp/field-review-$RUN_LABEL/state"
export GLM_RUNTIME_DIR="/tmp/field-review-$RUN_LABEL/runtime"

if [ "$MODE" = "candidate" ]; then
  export PYTHONPATH="$CANDIDATE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
else
  unset PYTHONPATH
fi

# Isolate the benchmark from all rental-facing auxiliaries and credentials.
export MODEL_ROOT MODEL_READ_ONLY=1
export MODEL_PROFILE=glm52-exl3 MODEL_FAMILY=glm52
export MODEL_VARIANT=exl3-tr3-3.25bpw
export MODEL_ID=
export MODEL_REVISION=d7d79c2d14599dfce7a5d12b85f7ad73f40e623d
export MODEL_REPO=willfalco/GLM-5.2-EXL3-TR3-3.25bpw
export TENSOR_PARALLEL_SIZE=4 DCP=4
export MTP_TOKENS=0 MTP_DRAFT_SAMPLE_METHOD=probabilistic
export MAX_MODEL_LEN=131072 MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=3072
export MAX_CUDAGRAPH_CAPTURE_SIZE=6
export CUDAGRAPH_CAPTURE_SIZES=1,2,3,4,5,6
export VLLM_EXL3_TRELLIS_MAX_M=6
export GPU_MEMORY_UTILIZATION=0.90
export GPU_BLOCKS_OVERRIDE=0
export KV_CACHE_DTYPE=nvfp4_ds_mla KV_SCALE_MODE=dynamic-token
export OFFLOAD_FRACTION=0 PREFIX_CACHE_BACKEND=lmcache
export PREFIX_CACHE_DISK_GB=0 VISION=0
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

echo ">>> field-review run: label=$RUN_LABEL mode=$MODE capacity=$PREFILL_CAPACITY"
echo ">>> JIT/cache root: $JIT_ROOT"
if [ "$MODE" = "candidate" ]; then
  python3 - <<'PY'
import vllm
print(f">>> candidate vLLM import: {vllm.__file__}")
PY
fi

exec /usr/local/bin/model-turnkey-entry.sh
