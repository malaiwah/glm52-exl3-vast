#!/bin/bash
# Multi-model Blackwell vLLM turnkey for Vast.ai and Runpod Pods.
# MODEL_PROFILE is the provider-template compatibility name; MODEL_FAMILY and
# MODEL_VARIANT are the self-service configuration names persisted by the UI.
#
# SELF-SERVICE CONFIG (docs/self-service-config.md). Every tunable is resolved
# from three layers — built-in defaults < startup environment < the JSON state
# file the landing page writes on the volume — re-resolved on every vLLM start,
# so the user can change the deployment from the :1111 page without rebuilding
# the image or destroying the rental. A candidate config that fails to come up,
# or that comes up and fails the long-context probe, is rolled back to the last
# known-good one automatically, with its config and log preserved for analysis.
set -e

SCRIPTS_DIR="${SCRIPTS_DIR:-/opt/scripts}"
MODEL_PROFILE="${MODEL_PROFILE:-}"
case "$MODEL_PROFILE" in
  ""|glm52-exl3)
    export MODEL_FAMILY="${MODEL_FAMILY:-glm52}"
    export MODEL_VARIANT="${MODEL_VARIANT:-exl3-tr3}"
    MODEL_PROFILE="${MODEL_PROFILE:-glm52-exl3}"
    ;;
  glm53-k6|glm53-k8)
    export MODEL_FAMILY="${MODEL_FAMILY:-glm53}"
    export MODEL_VARIANT="${MODEL_VARIANT:-$MODEL_PROFILE}"
    ;;
  qwen36-27b-nvfp4)
    export MODEL_FAMILY="${MODEL_FAMILY:-qwen36}"
    export MODEL_VARIANT="${MODEL_VARIANT:-qwen36-nvfp4}"
    ;;
  custom)
    export MODEL_FAMILY="${MODEL_FAMILY:-custom}"
    export MODEL_VARIANT="${MODEL_VARIANT:-custom}"
    ;;
  *)
    echo "FATAL: unknown MODEL_PROFILE=$MODEL_PROFILE"
    echo "FATAL: choose glm53-k6, glm53-k8, glm52-exl3, qwen36-27b-nvfp4, or custom"
    exit 1
    ;;
esac
export MODEL_PROFILE
echo "=== vLLM turnkey (profile $MODEL_PROFILE; family $MODEL_FAMILY) ==="
NVIDIA_DRIVER_VERSION=""
NVIDIA_CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INVENTORY="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"
  printf '%s\n' "$GPU_INVENTORY" | head -4
  NVIDIA_DRIVER_VERSION="$(printf '%s\n' "$GPU_INVENTORY" | head -1 | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')"
  NVIDIA_CUDA_VERSION="$(nvidia-smi 2>/dev/null | python3 -c "
import sys
sys.path.insert(0, '${SCRIPTS_DIR:-/opt/scripts}')
from gpu_detect import parse_cuda_version
print(parse_cuda_version(sys.stdin.read()))" || true)"
fi
# What can this container ACTUALLY use? nvidia-smi intersected with
# NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES — see scripts/gpu_detect.py.
# This is the only place that shells out to nvidia-smi; the config layer reads
# the result from GLM_GPU_COUNT, so nothing else needs to see a GPU.
GPU_JSON=$(python3 "${SCRIPTS_DIR:-/opt/scripts}/gpu_detect.py" --json 2>/dev/null || echo '{}')
NGPU=$(printf '%s' "$GPU_JSON" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('count', 0))
except Exception: print(0)")
export GLM_GPU_COUNT="$NGPU"
printf '%s' "$GPU_JSON" | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
for n in d.get('notes', []): print('>>> gpu: ' + n)" || true
GPU_NAME=$(printf '%s' "$GPU_JSON" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('name', ''))
except Exception: print('')" 2>/dev/null || echo "")
export GLM_PROVIDER
GLM_PROVIDER=$(python3 -c "
import sys; sys.path.insert(0, '${SCRIPTS_DIR:-/opt/scripts}')
try:
    import provider; print(provider.detect())
except Exception: print('unknown')" 2>/dev/null || echo unknown)
case "$GLM_PROVIDER" in
  vastai) PLATFORM=vast ;;
  runpod) PLATFORM=runpod ;;
  jarvislabs) PLATFORM=jarvislabs ;;
  *) PLATFORM=generic ;;
esac
INSTANCE_ID="$(printf '%s' \
  "${JARVISLABS_MACHINE_ID:-${CONTAINER_ID:-${RUNPOD_POD_ID:-}}}" |
  tr -cd '[:alnum:]-' | cut -c1-40)"
echo ">>> GPUs visible to this container: $NGPU${GPU_NAME:+ ($GPU_NAME)}  provider: $PLATFORM"
# GG v20 is built on CUDA 13.2 and uses PTX JIT/custom collectives. NVIDIA pairs
# CUDA 13.2 GA with Linux driver 595.45.04, but the full Qwen profile, feature
# suite, long-context retrieval, vision, SOUL and real OMP workload also passed
# on Runpod driver 590.48.01 advertising CUDA 13.2 with cuda-compat present.
# The pair matters: an earlier 590.48.01 / CUDA 13.1 Vast offer remains outside
# the supported envelope, as does the r580 Runpod host that failed NCCL init.
MIN_NVIDIA_DRIVER_VERSION="${MIN_NVIDIA_DRIVER_VERSION:-590.48.01}"
MIN_NVIDIA_CUDA_VERSION="${MIN_NVIDIA_CUDA_VERSION:-13.2}"
if [ "${CONFIG_SMOKE:-0}" != "1" ] && [ "$NGPU" -gt 0 ] &&
   [ "${ALLOW_UNSUPPORTED_NVIDIA_DRIVER:-0}" != "1" ]; then
  if ! python3 - "$NVIDIA_DRIVER_VERSION" "$MIN_NVIDIA_DRIVER_VERSION" \
      "$NVIDIA_CUDA_VERSION" "$MIN_NVIDIA_CUDA_VERSION" "$SCRIPTS_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[5])
from gpu_detect import version_meets_minimum
ok = (version_meets_minimum(sys.argv[1], sys.argv[2])
      and version_meets_minimum(sys.argv[3], sys.argv[4]))
raise SystemExit(0 if ok else 1)
PY
  then
    echo "FATAL: NVIDIA driver ${NVIDIA_DRIVER_VERSION:-unknown} / CUDA ${NVIDIA_CUDA_VERSION:-unknown} is outside the tested floor."
    echo "FATAL: choose a host advertising CUDA >=${MIN_NVIDIA_CUDA_VERSION} / driver >=${MIN_NVIDIA_DRIVER_VERSION};"
    echo "FATAL: refusing to download model weights for a host likely to fail during CUDA/NCCL initialization."
    echo "FATAL: set ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1 only for a separately qualified compatibility stack."
    exit 1
  fi
fi
case "${DESEC_TOKEN:-}" in
  *RUNPOD_SECRET_*) DESEC_TOKEN="" ;;
esac
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" = "auto" ]; then
  if [ -n "${RUNPOD_TCP_PORT_8443:-}" ] &&
     { { [ -n "${DESEC_TOKEN:-}" ] && [ -n "${DESEC_DOMAIN:-}" ]; } ||
       { [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ]; }; }; then
    RUNPOD_DIRECT_TLS=1
  else
    RUNPOD_DIRECT_TLS=0
    echo ">>> Runpod direct TLS not configured; using the managed HTTPS proxy fallback."
  fi
fi
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" = "1" ]; then
  [ -n "${RUNPOD_TCP_PORT_8443:-}" ] || {
    echo "FATAL: RUNPOD_DIRECT_TLS=1 requires exposing container port 8443 as TCP."
    exit 1
  }
  if ! { [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ]; } &&
     ! { [ -n "${DESEC_TOKEN:-}" ] && [ -n "${DESEC_DOMAIN:-}" ]; }; then
    echo "FATAL: RUNPOD_DIRECT_TLS=1 requires ACME_DOMAIN + ACME_DNS_PROVIDER,"
    echo "FATAL: or DESEC_TOKEN + DESEC_DOMAIN."
    exit 1
  fi
fi
if [ "$NGPU" = "0" ]; then
  if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
    case "$MODEL_PROFILE" in
      qwen36-27b-nvfp4|custom) NGPU="${TENSOR_PARALLEL_SIZE:-1}" ;;
      *) NGPU="${TENSOR_PARALLEL_SIZE:-4}" ;;
    esac
    export GLM_GPU_COUNT="$NGPU"
    echo ">>> config smoke: simulating $NGPU visible GPU(s); no GPU will be touched"
  else
  # Never silently fall back to 4. A wrong TP is a boot failure either way; the
  # only thing that changes is whether the user can tell why.
  echo "FATAL: no usable GPU detected."
  echo "       nvidia-smi reported nothing, or NVIDIA_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES"
  echo "       narrowed the set to zero. This image cannot serve a model without a GPU."
  echo "       If you believe this is wrong, run:  python3 ${SCRIPTS_DIR:-/opt/scripts}/gpu_detect.py --json"
  echo "       and set TENSOR_PARALLEL_SIZE explicitly to override the detected count."
  exit 1
  fi
fi
# The GPU-count gate used to be `-ge 4` right here, before any configuration was
# read, because the serve line hard-coded --tensor-parallel-size 4. That made the
# image refuse to start on a 1-GPU host even for a model that fits on one card.
# The check now belongs to the resolved TENSOR_PARALLEL_SIZE and runs after
# apply_config.

# Repair SSH key permissions before anything else. vast injects the account's
# public key into /root/.ssh/authorized_keys at container start; when it lands
# group- or world-writable, sshd refuses it with
#   "Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys"
# and every `ssh root@...` fails with a bare "Permission denied (publickey)".
# That silently breaks the SSH tunnel this README recommends as the safest way
# to reach the endpoint, and it looks exactly like a bad-host/bad-key problem,
# so it gets misdiagnosed as rental bad luck. Cheap to make unconditional.
if [ -d /root/.ssh ]; then
  chown -R root:root /root/.ssh 2>/dev/null || true
  chmod 700 /root /root/.ssh 2>/dev/null || true
  if [ -f /root/.ssh/authorized_keys ]; then
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
  fi
  echo ">>> SSH: repaired /root/.ssh ownership+modes (sshd rejects loose perms)"
fi

# ---- self-service config plumbing -------------------------------------------
# MODEL_ROOT is the volume root; the config state dir lives beside the weights so
# it survives a container replacement. The runtime dir must NOT be on the volume:
# a stale restart flag or verify verdict surviving a container swap would fire
# once more against a config that was never applied.
if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  MODEL_ROOT="${MODEL_ROOT:-/tmp/model-turnkey-smoke}"
else
  MODEL_ROOT="${MODEL_ROOT:-/workspace}"
fi
mkdir -p "$MODEL_ROOT"
export GLM_STATE_DIR="${GLM_STATE_DIR:-$MODEL_ROOT/.glm-config}"
export GLM_RUNTIME_DIR="${GLM_RUNTIME_DIR:-/tmp/glm-runtime}"
GLM_LOG_DIR="$GLM_STATE_DIR/logs"
SERVE_LOG="$GLM_LOG_DIR/serve-current.log"
RESTART_FLAG="$GLM_RUNTIME_DIR/restart-request"
VERIFY_FILE="$GLM_RUNTIME_DIR/verify.json"
VERIFY_LAST="$GLM_STATE_DIR/verify-last.json"
CONFIG_ENV="$GLM_RUNTIME_DIR/config.env"
# Termination handshake files. TERMINATE_FLAG is written by the terminate
# worker; the supervisor answers by stopping vLLM for good and touching
# ENGINE_STOPPED. Both live in the runtime dir, so a replaced container starts
# with a clean slate and can never inherit a stale "please terminate".
TERMINATE_FLAG="$GLM_RUNTIME_DIR/terminate-in-progress"
ENGINE_STOPPED="$GLM_RUNTIME_DIR/engine-stopped"
BOOT_NOTES="$GLM_RUNTIME_DIR/boot-notes.log"
SOUL_STATE_DIR="$GLM_STATE_DIR/soul"
SOUL_RUNTIME_DIR="$GLM_RUNTIME_DIR/soul"
SOUL_LOG="$SOUL_STATE_DIR/logs/controller.log"
mkdir -p "$GLM_STATE_DIR" "$GLM_RUNTIME_DIR" "$GLM_LOG_DIR" "$GLM_STATE_DIR/failures" \
  "$SOUL_STATE_DIR" "$SOUL_RUNTIME_DIR" "$SOUL_STATE_DIR/logs"
: > "$BOOT_NOTES"
# Everything important that happens before the engine is up goes BOTH to stdout
# and to a file the landing page renders. On RunPod stdout is unreachable —
# there is no console view and no CLI log command — so a boot warning that only
# reaches stdout may as well not exist. That is how three "silent" defects were
# diagnosed from a rented pod instead of from a log line.
boot_note() {
  echo "$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" \
    >> "$BOOT_NOTES" 2>/dev/null || true
}
rm -f "$RESTART_FLAG" "$VERIFY_FILE" "$TERMINATE_FLAG" "$ENGINE_STOPPED"

# MODEL_DIR set in the template env pins the weights regardless of the model
# variant knob; unset, the variant chooses the directory name.
MODEL_DIR_PINNED="${MODEL_DIR:-}"

# Freeze the environment layer exactly once, before this script exports anything
# else. Everything downstream reads the snapshot, so the layering can never be
# perturbed by a variable the entrypoint itself sets later.
python3 "$SCRIPTS_DIR/config_cli.py" snapshot-env || \
  echo "!!! config: could not snapshot the startup env (continuing on defaults)"

# apply_config: re-resolve defaults < env < state file and export the result.
# Runs before EVERY vLLM start, which is what makes an apply take effect without
# replacing the container. A state file that fails pre-validation is rolled back
# here rather than handed to the engine.
apply_config() {
  if python3 "$SCRIPTS_DIR/config_cli.py" env > "$CONFIG_ENV.tmp"; then
    mv "$CONFIG_ENV.tmp" "$CONFIG_ENV"
  else
    echo "!!! config: resolution failed; keeping the previously loaded configuration" >&2
    rm -f "$CONFIG_ENV.tmp"
  fi
  if [ -s "$CONFIG_ENV" ]; then
    # shellcheck disable=SC1090
    . "$CONFIG_ENV"
  fi
  case "${MODEL_FAMILY:-glm52}" in
    glm52) MODEL_PROFILE=glm52-exl3 ;;
    glm53)
      case "${MODEL_VARIANT:-glm53-k6}" in
        glm53-k8) MODEL_PROFILE=glm53-k8 ;;
        *) MODEL_PROFILE=glm53-k6 ;;
      esac
      ;;
    qwen36) MODEL_PROFILE=qwen36-27b-nvfp4 ;;
    custom) MODEL_PROFILE=custom ;;
  esac
  export MODEL_PROFILE
  if ! python3 "$SCRIPTS_DIR/config_cli.py" validate --quiet; then
    echo "!!! config: the resolved configuration FAILS pre-validation:"
    python3 "$SCRIPTS_DIR/config_cli.py" validate || true
    if python3 "$SCRIPTS_DIR/config_cli.py" should-rollback; then
      echo "!!! config: refusing to start the engine on it — rolling back"
      python3 "$SCRIPTS_DIR/config_cli.py" rollback --reason "failed pre-validation at boot" || true
      python3 "$SCRIPTS_DIR/config_cli.py" clear-restart || true
      if python3 "$SCRIPTS_DIR/config_cli.py" env > "$CONFIG_ENV.tmp"; then
        mv "$CONFIG_ENV.tmp" "$CONFIG_ENV"
        # shellcheck disable=SC1090
        . "$CONFIG_ENV"
      fi
    else
      echo "!!! config: no known-good configuration to fall back to — starting anyway."
      echo "!!! The findings above are measured failure modes; expect them."
    fi
  fi
  # SparkInfer's PCIe DMA and B12X DCP A2A channels exchange CUDA IPC handles.
  # Some otherwise attractive VM offers expose all cards under one PHB but
  # disable peer reads/writes in the hypervisor. On those hosts the overlap
  # calibrator cannot merely choose a conservative crossover: every custom
  # channel fails cudaIpcOpenMemHandle. Detect that hard capability boundary
  # before calibration and retain vLLM's NCCL/SHM fallbacks.
  if { [ "${MODEL_FAMILY:-glm52}" = "glm52" ] ||
       [ "${MODEL_FAMILY:-glm52}" = "glm53" ]; } &&
     [ "${B12X_PCIE_DMA:-1}" = "1" ] &&
     command -v nvidia-smi >/dev/null 2>&1; then
    _p2p_matrix="$(nvidia-smi topo -p2p r 2>/dev/null || true)"
    # nvidia-smi prints the unsupported status abbreviations again in its
    # legend. Inspect only actual GPU matrix rows; otherwise every fully
    # supported host is misclassified because the legend contains NS/CNS/etc.
    _p2p_matrix_rows="$(
      grep -E '^[[:space:]]*GPU[0-9]+[[:space:]]' <<<"$_p2p_matrix" || true
    )"
    if grep -Eq '(^|[[:space:]])(NS|CNS|GNS|TNS)([[:space:]]|$)' \
         <<<"$_p2p_matrix_rows"; then
      export B12X_PCIE_DMA=0
      boot_note "WARNING: GPU peer access is unavailable; disabled B12X PCIe DMA and DCP A2A, using NCCL/SHM collectives."
    fi
    unset _p2p_matrix _p2p_matrix_rows
  fi
  if [ -z "${_CONFIG_SHOWN:-}" ]; then
    python3 "$SCRIPTS_DIR/config_cli.py" show || true
    _CONFIG_SHOWN=1
    export _CONFIG_SHOWN
  fi
  # MODEL_DIR follows the variant unless the template pinned it.
  if [ -n "$MODEL_DIR_PINNED" ]; then
    MODEL_DIR="$MODEL_DIR_PINNED"
    if [ -n "${MODEL_DIRNAME:-}" ] && [ "$(basename "$MODEL_DIR")" != "$(basename "$MODEL_DIRNAME")" ]; then
      echo "!!! MODEL_DIR is pinned to $MODEL_DIR but the resolved model is"
      echo "!!! ${MODEL_REPO:-?} (expected directory '$MODEL_DIRNAME'). Serving from the"
      echo "!!! pinned path anyway — unset MODEL_DIR to let the family choose it."
    fi
  else
    MODEL_DIR="$MODEL_ROOT/${MODEL_DIRNAME:-GLM-5.2-EXL3-TR3-3.0bpw}"
  fi
  export MODEL_DIR
  return 0
}

apply_config

# A resolved MODEL_FAMILY the rest of this script does not implement must be a
# LOUD failure, not a silent fallback. Silently serving GLM-5.2 to someone who
# asked for another family is the worst outcome available: it downloads hundreds
# of GB they did not ask for, onto a disk that may not hold it, at rental rates,
# with the UI reporting progress as though everything were fine.
case "${FAMILY_ENV_BLOCK:-}" in
  glm52|glm53|generic) ;;
  *)
    echo "FATAL: MODEL_FAMILY='${MODEL_FAMILY:-?}' resolved to an engine environment"
    echo "       block ('${FAMILY_ENV_BLOCK:-}') that this entrypoint does not implement."
    echo "       Refusing to fall back to GLM-5.2 silently. Known families:"
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPTS_DIR')
import glm_config as gc
for k, v in gc.FAMILIES.items():
    print('         %-8s %s' % (k, v['label']))" || true
    exit 1
    ;;
esac
if [ -z "${MODEL_REPO:-}" ]; then
  echo "FATAL: no model repository resolved for MODEL_FAMILY='${MODEL_FAMILY:-?}'."
  echo "       Refusing to guess. Set MODEL_VARIANT or MODEL_DIR explicitly."
  exit 1
fi

if [ "$NGPU" -lt "${TENSOR_PARALLEL_SIZE:-4}" ]; then
  echo "FATAL: tensor-parallel size ${TENSOR_PARALLEL_SIZE:-4} needs that many GPUs, found $NGPU."
  echo "       TENSOR_PARALLEL_SIZE defaults to the detected count, so this means it was"
  echo "       set explicitly (env or config state) to more than this host has."
  echo "       Unset it to use $NGPU, or rent a host with more GPUs."
  exit 1
fi

# CONFIG_SMOKE=1: resolve everything, print what WOULD be served, and exit
# before the first byte is downloaded or a GPU is touched. This exists because
# knobs shipped that the entrypoint did not honour end-to-end, and each was
# found by renting a GPU instead of by running this.
#     docker run --rm -e CONFIG_SMOKE=1 -e MODEL_FAMILY=qwen36 <image>
# The argv half runs further down, once serve_once and the arg builders exist.
if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  echo "=== CONFIG SMOKE (nothing will be downloaded, no GPU will be touched) ==="
  if [ -z "${_CONFIG_SHOWN:-}" ]; then
    python3 "$SCRIPTS_DIR/config_cli.py" show || true
  fi
  echo ">>> model repo    : ${MODEL_REPO:-<unset>}"
  echo ">>> model dir     : ${MODEL_DIR:-<unset>}"
  echo ">>> family env    : ${FAMILY_ENV_BLOCK:-<unset>}"
  echo ">>> spec method   : ${SPEC_METHOD:-<unset>}"
  echo ">>> GPUs visible  : $NGPU (tensor-parallel ${TENSOR_PARALLEL_SIZE:-4})"
fi

# ---- optional sshd ----------------------------------------------------------
# MEASURED on a live RunPod pod: this image does NOT start sshd, and RunPod does
# not start one for you — their SSH works only because their own base images run
# one. With this image as PID 1 the pod reports RUNNING, `runpodctl ssh info`
# happily resolves, and every connection is refused. Combined with RunPod's other
# trap (ports must be requested at pod creation), a misconfigured pod becomes a
# black box: no shell, no landing page, only the console log.
#
# So: start sshd when the provider handed us keys and something is there to run.
# It is `auto` rather than always-on because on vast.ai sshd is already running
# and re-starting it would be at best redundant; and it is skippable entirely
# (SSHD=0) for anyone who wants the smaller surface. Key-only auth, using the
# key material the provider itself injected — this grants no access that the
# provider was not already handing out.
SSHD_STATE="not started"
setup_sshd() {
  local mode="${SSHD:-auto}"
  if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
    return 0
  fi
  if [ "$mode" = "0" ]; then
    SSHD_STATE="disabled (SSHD=0)"
    echo ">>> sshd: disabled (SSHD=0)"
    return 0
  fi
  # RunPod injects PUBLIC_KEY; Vast injects SSH_PUBLIC_KEY.
  local keys="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
  if [ "$mode" = "auto" ] && [ -z "$keys" ]; then
    SSHD_STATE="not needed (no PUBLIC_KEY; the provider runs its own)"
    return 0
  fi
  # Install the keys FIRST, unconditionally. If some other sshd is already
  # listening it is still the injected key that has to be in authorized_keys for
  # the user to get in — skipping the install because a daemon exists was
  # backwards, and would have left a running sshd that rejects the only key the
  # provider handed out.
  if [ -n "$keys" ]; then
    mkdir -p /root/.ssh
    touch /root/.ssh/authorized_keys
    # append only what is missing, so a re-run does not grow the file
    printf '%s\n' "$keys" | while IFS= read -r k; do
      [ -n "$k" ] || continue
      if ! grep -qxF "$k" /root/.ssh/authorized_keys 2>/dev/null; then
        printf '%s\n' "$k" >> /root/.ssh/authorized_keys
      fi
    done
    chmod 700 /root/.ssh 2>/dev/null || true
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
    _nkeys=$(grep -c . /root/.ssh/authorized_keys 2>/dev/null) || _nkeys=0
    boot_note ">>> sshd: ${_nkeys} authorized key(s) installed from PUBLIC_KEY"
  fi
  if (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -q ':22 '; then
    boot_note ">>> sshd: something is already listening on :22, leaving it alone"
    SSHD_STATE="already running"
    return 0
  fi
  if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then
    SSHD_STATE="no sshd binary in this image"
    boot_note "!!! sshd: NO sshd BINARY IN THIS IMAGE. There is no shell access on"
    boot_note "!!! providers that expect the container to run one (RunPod), and the"
    boot_note "!!! landing page on :1111 is then the ONLY way in — make sure that port"
    boot_note "!!! was exposed AT POD CREATION, because it cannot be added later."
    boot_note "!!! Rebuild with openssh-server, or set SSHD_INSTALL=1 to apt-get it."
    if [ "${SSHD_INSTALL:-0}" = "1" ]; then
      boot_note ">>> sshd: installing openssh-server (SSHD_INSTALL=1)"
      if apt-get update -qq; then
        apt-get install -y -qq openssh-server || true
      fi
    else
      return 0
    fi
  fi
  ssh-keygen -A >/dev/null 2>&1 || true
  mkdir -p /run/sshd
  if ${SSHD_BIN:-/usr/sbin/sshd}; then
    SSHD_STATE="listening on :22"
    boot_note ">>> sshd: started on :22 (key-only; the port must have been exposed at"
    boot_note ">>> pod/instance creation to reach it)"
  else
    SSHD_STATE="failed to start"
    boot_note "!!! sshd: FAILED to start; the landing page is the only way in"
  fi
  return 0
}
setup_sshd

# Boot-status snapshot for the landing page; rewritten at each milestone.
# Holds the API key once generated -> keep it root-only.
STATUS_FILE="${STATUS_FILE:-/tmp/glm-boot-status.json}"
EP_URL="" TLS_STATE="not configured" HTTPS_HOSTPORT="" CERT_PATH="" KEY_PATH="" OFFLOAD_STATE="off"
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" != "1" ]; then
  EP_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"
  TLS_STATE="https:// managed by Runpod proxy"
fi
status_update() {
  # Compose the JSON with json.dumps, not printf: a double quote or backslash
  # in any interpolated value (API key, served name, GPU name) would corrupt
  # the file the landing page depends on for the whole session. Values travel
  # via the child's environment, never argv.
  # Staged then renamed: `> "$STATUS_FILE"` truncated before python ran, so a
  # spawn failure under memory pressure blanked the dashboard's only status
  # source. umask 077 because the staged file carries the API key too.
  _su_written=0
  ( umask 077
    STATUS_PHASE="$1" STATUS_EP_URL="$EP_URL" STATUS_TLS="$TLS_STATE" \
    STATUS_HTTPS_HOSTPORT="$HTTPS_HOSTPORT" STATUS_CERT="$CERT_PATH" \
    STATUS_KEYFILE="$KEY_PATH" STATUS_OFFLOAD="$OFFLOAD_STATE" \
    STATUS_API_KEY="${VLLM_API_KEY:-}" STATUS_MODEL_DIR="$MODEL_DIR" \
    STATUS_SERVED="${SERVED_MODEL_NAME:-model}" \
    STATUS_FAMILY="${MODEL_FAMILY:-unknown}" STATUS_AUTH="${AUTH:-key}" \
    STATUS_PORT="${PORT:-8000}" STATUS_GPUS="${GLM_GPU_COUNT:-?}" \
    STATUS_GPU_NAME="${GPU_NAME:-}" STATUS_SSHD="${SSHD_STATE:-not started}" \
    STATUS_PROVIDER="${GLM_PROVIDER:-unknown}" \
    python3 - > "$STATUS_FILE.tmp" <<'PY'
import json, os
e = os.environ.get
print(json.dumps({
    "phase": e("STATUS_PHASE", ""),
    "endpoint": e("STATUS_EP_URL", ""),
    "tls": e("STATUS_TLS", ""),
    "https_hostport": e("STATUS_HTTPS_HOSTPORT", ""),
    "cert": e("STATUS_CERT", ""),
    "keyfile": e("STATUS_KEYFILE", ""),
    "offload": e("STATUS_OFFLOAD", ""),
    "api_key": e("STATUS_API_KEY", ""),
    "model_dir": e("STATUS_MODEL_DIR", ""),
    "served_model": e("STATUS_SERVED", "model"),
    "model_family": e("STATUS_FAMILY", "unknown"),
    "auth": e("STATUS_AUTH", "key"),
    "port": e("STATUS_PORT", "8000"),
    "gpus": e("STATUS_GPUS", "?"),
    "gpu_name": e("STATUS_GPU_NAME", ""),
    "sshd": e("STATUS_SSHD", "not started"),
    "provider": e("STATUS_PROVIDER", "unknown"),
}))
PY
  ) && _su_written=1
  if [ "$_su_written" = "1" ] && mv -f "$STATUS_FILE.tmp" "$STATUS_FILE" 2>/dev/null; then
    :
  else
    rm -f "$STATUS_FILE.tmp" 2>/dev/null || true
    echo "!!! status_update failed for phase $1 (keeping the previous snapshot)" >&2
  fi
  # The full status file carries the API key in plaintext, so it stays root-only
  # (600) even when the soul user exists. SOUL runs deprivileged and must not be
  # able to read the key: publish a SANITIZED copy (api_key stripped) for the
  # soul group instead, and point the controller at it. The controller still
  # receives the key it needs for probes through its own environment, which it
  # scrubs before spawning any exec shell — so nothing the model can reach ever
  # contains the key.
  chmod 600 "$STATUS_FILE" 2>/dev/null || true
  if id soul >/dev/null 2>&1; then
    _soul_status="$STATUS_FILE.soul"
    ( umask 077
      SOUL_STATUS_SRC="$STATUS_FILE" SOUL_STATUS_DEST="$_soul_status" \
        python3 - <<'PY' 2>/dev/null || true
import json, os
src, dst = os.environ["SOUL_STATUS_SRC"], os.environ["SOUL_STATUS_DEST"]
try:
    doc = json.load(open(src))
except Exception:
    doc = {}
doc.pop("api_key", None)
tmp = dst + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f)
os.replace(tmp, dst)
PY
    )
    chgrp soul "$_soul_status" 2>/dev/null || true
    chmod 640 "$_soul_status" 2>/dev/null || true
  fi
}

# Landing page (:1111, dual-protocol TLS+plain or behind Runpod's HTTPS proxy).
# Started before the weight download so status is visible from minute one.
# Needs OPEN_BUTTON_PORT=1111 env + '-p 1111:1111' in the template.
# It also hosts the config editor, so it gets the state/runtime dirs and the
# scripts dir (for the shared knob registry in glm_config.py).
if [ "${LANDING_PAGE:-1}" != "0" ] && [ -f /opt/landing.py ] \
   && [ "${CONFIG_SMOKE:-0}" != "1" ]; then
  # Whether the operator set it explicitly, captured BEFORE we default it: on
  # Runpod's managed-proxy mode we default proxy-trust ON, but an operator who
  # exposes the dashboard port as a DIRECT TCP mapping can set
  # LANDING_TRUST_PROXY_HTTPS=0 to refuse a forgeable X-Forwarded-Proto — honor
  # that explicit choice instead of forcing 1.
  _lthp_explicit="${LANDING_TRUST_PROXY_HTTPS+set}"
  LANDING_TRUST_PROXY_HTTPS="${LANDING_TRUST_PROXY_HTTPS:-0}"
  if [ "$PLATFORM" = "runpod" ] || [ "$PLATFORM" = "jarvislabs" ]; then
    if [ "$PLATFORM" = "runpod" ] && [ -z "$_lthp_explicit" ]; then
      LANDING_TRUST_PROXY_HTTPS=1
    fi
    if [ -z "${OPEN_BUTTON_TOKEN:-}" ]; then
      OPEN_BUTTON_TOKEN_FILE="${OPEN_BUTTON_TOKEN_FILE:-/workspace/.model-turnkey-landing-token}"
      mkdir -p "$(dirname "$OPEN_BUTTON_TOKEN_FILE")"
      if [ -s "$OPEN_BUTTON_TOKEN_FILE" ]; then
        OPEN_BUTTON_TOKEN="$(cat "$OPEN_BUTTON_TOKEN_FILE")"
      else
        OPEN_BUTTON_TOKEN="lp-$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        ( umask 077; printf '%s\n' "$OPEN_BUTTON_TOKEN" > "$OPEN_BUTTON_TOKEN_FILE" )
      fi
      chmod 600 "$OPEN_BUTTON_TOKEN_FILE" 2>/dev/null || true
    fi
    export OPEN_BUTTON_TOKEN
  fi
  status_update booting
  # Supervise the landing page: it is the appliance's recovery surface (on
  # RunPod, with no console log and possibly no sshd, the :1111 page is the ONLY
  # way in), so a process death during the long weight download or engine load
  # must not leave it gone for the rest of the session. Run it in a subshell
  # that restarts it with bounded backoff. The page is the subshell's child, so
  # the subshell's own TERM/INT trap reaps it on shutdown; on_term stops this
  # supervisor (a PID-1 job, so it is in `jobs -p`), which fires that trap. The
  # token is NOT re-minted on restart — OPEN_BUTTON_TOKEN is already resolved
  # and exported above, so every restart serves the same tokenized URL.
  (
    trap '[ -z "${_lp:-}" ] || kill "$_lp" 2>/dev/null || true; exit 0' TERM INT
    _lp=""; _lp_fails=0
    while :; do
      if [ -z "$_lp" ] || ! kill -0 "$_lp" 2>/dev/null; then
        if [ -n "$_lp" ]; then
          boot_note "!!! landing: dashboard process exited; restarting the :1111 recovery surface"
          _lp_fails=$((_lp_fails + 1))
          case "$_lp_fails" in 1) sleep 2 ;; 2) sleep 5 ;; *) sleep 15 ;; esac
        fi
        MODEL_DIR="$MODEL_DIR" STATUS_FILE="$STATUS_FILE" GLM_SCRIPTS_DIR="$SCRIPTS_DIR" \
          GLM_STATE_DIR="$GLM_STATE_DIR" GLM_RUNTIME_DIR="$GLM_RUNTIME_DIR" \
          MODEL_PROFILE="$MODEL_PROFILE" LANDING_TRUST_PROXY_HTTPS="$LANDING_TRUST_PROXY_HTTPS" \
          LANDING_API_PORT="${PORT:-8000}" python3 /opt/landing.py &
        _lp=$!
      fi
      sleep 5
    done
  ) &
  LANDING_SUPERVISOR_PID=$!
  echo ">>> Landing page (Open button) live on :1111 (supervised)"
  if [ "$PLATFORM" = "runpod" ]; then
    echo ">>> Runpod dashboard: https://${RUNPOD_POD_ID}-1111.proxy.runpod.net/?token=${OPEN_BUTTON_TOKEN}"
  elif [ "$PLATFORM" = "jarvislabs" ]; then
    echo ">>> JarvisLabs dashboard token is ready; the trusted URL is printed after DNS/TLS setup."
  fi
fi

# ---- embedded SOUL supervision ---------------------------------------------
# SOUL is intentionally a sibling of the vLLM supervisor: a failed diagnostic
# controller must never spend the engine's restart budget or change endpoint
# state. It starts early, observes boot state, and waits for phase=serving before
# asking the local model to do any work.
SOUL_PID=""
SOUL_FAILURES=0
SOUL_NEXT_START=0
SOUL_STARTED_AT=0
SOUL_PERMISSIONS_PREPARED=0
SOUL_PERMISSIONS_LAST=0
SOUL_SUPERVISOR_PID=""
SOUL_LEVEL_CACHE=""
SOUL_LEVEL_CHECK_AT=0
SOUL_CONFIG_MTIME=""

soul_level() {
  python3 "$SCRIPTS_DIR/soul_config.py" level 2>/dev/null || printf '0\n'
}

# The 5s reconcile loop asked soul_config.py for the level on EVERY tick — a
# full interpreter spawn 17k times a day just to be told SOUL is off (the
# default). Re-derive only when something could actually have changed: a
# landing-page write drops the reconcile marker (and SIGUSR1s PID 1), the
# config file's mtime moves, or a 60s fallback elapses. The marker keeps the
# "reconcile remains authoritative" property while making the idle path free.
#
# This sets $SOUL_LEVEL_CACHE IN PLACE and must be called DIRECTLY, not via
# command substitution: `level=$(soul_level_cached)` would run it in a subshell
# whose assignments to the cache globals are discarded, so the cache would never
# populate and the interpreter would spawn every tick anyway. reconcile_soul
# runs directly in the persistent supervisor subshell, so a direct call's
# assignments survive across ticks.
soul_level_cached() {
  local now mtime=""
  now=$(date +%s 2>/dev/null || echo 0)
  [ -f "$SOUL_STATE_DIR/config.json" ] &&
    mtime=$(stat -c %Y "$SOUL_STATE_DIR/config.json" 2>/dev/null || echo "")
  if [ -z "$SOUL_LEVEL_CACHE" ] ||
     [ -f "$SOUL_RUNTIME_DIR/reconcile" ] ||
     [ "$mtime" != "$SOUL_CONFIG_MTIME" ] ||
     [ $((now - SOUL_LEVEL_CHECK_AT)) -ge 60 ]; then
    SOUL_LEVEL_CACHE=$(soul_level)
    SOUL_LEVEL_CHECK_AT="$now"
    SOUL_CONFIG_MTIME="$mtime"
  fi
}

soul_prepare_permissions() {
  # The dedicated user owns only its persistent subtree. Rollback evidence is
  # root-owned and world-readable; SOUL records its analysis under its own
  # incidents directory, so it never needs write access elsewhere.
  if id soul >/dev/null 2>&1; then
    local _soul_permissions_now
    _soul_permissions_now=$(date +%s)
    mkdir -p "$SOUL_STATE_DIR" "$SOUL_RUNTIME_DIR" "$SOUL_STATE_DIR/logs"
    chown soul:soul "$SOUL_STATE_DIR" "$SOUL_RUNTIME_DIR" "$SOUL_STATE_DIR/logs" \
      2>/dev/null || true
    if [ "$SOUL_PERMISSIONS_PREPARED" = "0" ]; then
      chown -R soul:soul "$SOUL_STATE_DIR" "$SOUL_RUNTIME_DIR" 2>/dev/null || true
      SOUL_PERMISSIONS_PREPARED=1
    fi
    # The autonomy override is written by root (the landing page) and only
    # READ by the soul user, so root-own it: that stops an in-place edit.
    # It is NOT a complete barrier — soul owns this directory (it writes
    # journal.jsonl and status.json here), so a determined shell can still
    # unlink and replace the file. The real bound on autonomy is
    # SOUL_AUTONOMY_MAX_LEVEL, which is startup-env only and cannot be
    # raised from inside the container at all.
    if [ -f "$SOUL_STATE_DIR/config.json" ]; then
      chown root:soul "$SOUL_STATE_DIR/config.json" 2>/dev/null || true
      chmod 640 "$SOUL_STATE_DIR/config.json" 2>/dev/null || true
    fi
    if [ $((_soul_permissions_now - SOUL_PERMISSIONS_LAST)) -lt 60 ]; then
      return 0
    fi
    SOUL_PERMISSIONS_LAST="$_soul_permissions_now"
    for _soul_read in config.json known-good.json apply-state.json verify-last.json \
      checkpoint-baseline.json; do
      if [ -f "$GLM_STATE_DIR/$_soul_read" ]; then
        chgrp soul "$GLM_STATE_DIR/$_soul_read" 2>/dev/null || true
        chmod g+r "$GLM_STATE_DIR/$_soul_read" 2>/dev/null || true
      fi
    done
    for _soul_tree in "$GLM_STATE_DIR/failures" "$GLM_STATE_DIR/logs"; do
      [ -d "$_soul_tree" ] || continue
      find "$_soul_tree" -type d -exec chmod g+rx {} + 2>/dev/null || true
      find "$_soul_tree" -type f -exec chgrp soul {} + 2>/dev/null || true
      find "$_soul_tree" -type f -exec chmod g+r {} + 2>/dev/null || true
    done
    for _soul_runtime_read in startup-env.json config.env verify.json; do
      if [ -f "$GLM_RUNTIME_DIR/$_soul_runtime_read" ]; then
        chgrp soul "$GLM_RUNTIME_DIR/$_soul_runtime_read" 2>/dev/null || true
        chmod g+r "$GLM_RUNTIME_DIR/$_soul_runtime_read" 2>/dev/null || true
      fi
    done
  fi
}

start_soul() {
  [ "${CONFIG_SMOKE:-0}" != "1" ] || return 0
  [ "$(soul_level)" -gt 0 ] 2>/dev/null || return 0
  [ -x /opt/nanobot-venv/bin/python ] || {
    boot_note "!!! SOUL: isolated Nanobot runtime is missing; controller not started"
    return 0
  }
  soul_prepare_permissions
  echo ">>> SOUL: starting embedded controller at autonomy level $(soul_level)"
  nice -n 10 ionice -c 3 runuser -u soul -- env -i \
    HOME="$SOUL_STATE_DIR/workspace" \
    PATH="/opt/nanobot-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG="C.UTF-8" TERM="dumb" USER="soul" PYTHONUNBUFFERED="1" \
    GLM_STATE_DIR="$GLM_STATE_DIR" GLM_RUNTIME_DIR="$GLM_RUNTIME_DIR" \
    STATUS_FILE="$STATUS_FILE.soul" PORT="${PORT:-8000}" \
    VLLM_API_KEY="${VLLM_API_KEY:-}" SOUL_PROMPT="/opt/soul/SOUL.md" \
    SOUL_AUTONOMY_LEVEL="${SOUL_AUTONOMY_LEVEL:-0}" \
    SOUL_AUTONOMY_MAX_LEVEL="${SOUL_AUTONOMY_MAX_LEVEL:-3}" \
    SOUL_HEARTBEAT_INTERVAL_S="${SOUL_HEARTBEAT_INTERVAL_S:-300}" \
    SOUL_JOURNAL_INTERVAL_S="${SOUL_JOURNAL_INTERVAL_S:-3600}" \
    SOUL_JOURNAL_RETENTION_DAYS="${SOUL_JOURNAL_RETENTION_DAYS:-90}" \
    SOUL_EVIDENCE_RETENTION_DAYS="${SOUL_EVIDENCE_RETENTION_DAYS:-7}" \
    SOUL_MAX_STATE_MB="${SOUL_MAX_STATE_MB:-256}" \
    SOUL_TIMEZONE="${SOUL_TIMEZONE:-UTC}" \
    VERIFY_NEEDLE_TOKENS="${VERIFY_NEEDLE_TOKENS:-32768}" \
    NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    /opt/nanobot-venv/bin/python "$SCRIPTS_DIR/soul_controller.py" \
      --status-file "$STATUS_FILE.soul" --port "${PORT:-8000}" \
      >> "$SOUL_LOG" 2>&1 &
  SOUL_PID=$!
  SOUL_STARTED_AT=$(date +%s)
  SOUL_NEXT_START=0
}

stop_soul() {
  if [ -n "${SOUL_PID:-}" ] && kill -0 "$SOUL_PID" 2>/dev/null; then
    echo ">>> SOUL: stopping embedded controller"
    kill "$SOUL_PID" 2>/dev/null || true
    for _soul_wait in $(seq 1 20); do
      kill -0 "$SOUL_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$SOUL_PID" 2>/dev/null || true
    wait "$SOUL_PID" 2>/dev/null || true
  fi
  SOUL_PID=""
}

reconcile_soul() {
  local level now delay_index
  now=$(date +%s)
  # Direct call (NOT $()): soul_level_cached sets $SOUL_LEVEL_CACHE in place, and
  # a subshell would discard the cache — see its header. This function runs
  # directly in the persistent supervisor subshell, so the cache survives ticks.
  soul_level_cached
  level="$SOUL_LEVEL_CACHE"
  rm -f "$SOUL_RUNTIME_DIR/reconcile" 2>/dev/null || true
  if [ "${level:-0}" -le 0 ] 2>/dev/null; then
    stop_soul
    return 0
  fi
  soul_prepare_permissions
  if [ -n "${SOUL_PID:-}" ] && kill -0 "$SOUL_PID" 2>/dev/null; then
    if [ "${SOUL_STARTED_AT:-0}" -gt 0 ] &&
       [ $((now - SOUL_STARTED_AT)) -ge 600 ]; then
      SOUL_FAILURES=0
    fi
    return 0
  fi
  if [ -n "${SOUL_PID:-}" ]; then
    wait "$SOUL_PID" 2>/dev/null || true
    SOUL_PID=""
    SOUL_FAILURES=$((SOUL_FAILURES + 1))
    delay_index="$SOUL_FAILURES"
    case "$delay_index" in
      1) SOUL_NEXT_START=$((now + 5)) ;;
      2) SOUL_NEXT_START=$((now + 15)) ;;
      3) SOUL_NEXT_START=$((now + 60)) ;;
      *) SOUL_NEXT_START=$((now + 300)) ;;
    esac
    echo "!!! SOUL: controller exited; independent restart is scheduled" >&2
  fi
  if [ "${SOUL_NEXT_START:-0}" -le "$now" ]; then
    start_soul
  fi
}

start_soul_supervisor() {
  [ "${CONFIG_SMOKE:-0}" != "1" ] || return 0
  if [ -n "${SOUL_SUPERVISOR_PID:-}" ] &&
     kill -0 "$SOUL_SUPERVISOR_PID" 2>/dev/null; then
    return 0
  fi
  (
    trap 'stop_soul; exit 0' TERM INT
    trap 'stop_soul' EXIT
    while :; do
      reconcile_soul
      sleep 5
    done
  ) &
  SOUL_SUPERVISOR_PID=$!
}

stop_soul_supervisor() {
  if [ -n "${SOUL_SUPERVISOR_PID:-}" ] &&
     kill -0 "$SOUL_SUPERVISOR_PID" 2>/dev/null; then
    kill "$SOUL_SUPERVISOR_PID" 2>/dev/null || true
    wait "$SOUL_SUPERVISOR_PID" 2>/dev/null || true
  fi
  SOUL_SUPERVISOR_PID=""
}

# A UI config write sends SIGUSR1 only as a wake-up hint. The normal five-second
# reconcile remains authoritative, and this trap prevents the signal from
# terminating PID 1 on minimal shells.
trap ':' USR1
trap 'stop_soul_supervisor; [ -z "${LANDING_SUPERVISOR_PID:-}" ] || kill "$LANDING_SUPERVISOR_PID" 2>/dev/null || true; if [ -n "${SRV_PID:-}" ]; then kill -9 -- "-$SRV_PID" 2>/dev/null || true; fi' EXIT

# This script is container PID 1, and PID 1 ignores default-disposition
# signals: an untrapped `docker stop` / provider stop would hang for the full
# grace period and end in SIGKILL with no orderly release. Forward TERM to
# the serve process group and the background children, give vLLM a bounded
# grace to drop its GPU allocations, then exit; the EXIT trap's kill -9 of
# the serve group is the backstop for anything that ignored TERM.
# shellcheck disable=SC2317,SC2329  # invoked indirectly via the trap below
on_term() {
  trap ':' TERM INT
  echo ">>> stop requested: releasing serve process group and children" >&2
  stop_soul_supervisor 2>/dev/null || true
  if [ -n "${SRV_PID:-}" ]; then
    kill -TERM -- "-$SRV_PID" 2>/dev/null || true
  fi
  for _child in $(jobs -p); do
    kill -TERM "$_child" 2>/dev/null || true
  done
  if [ -n "${SRV_PID:-}" ]; then
    for _ in $(seq 1 30); do
      kill -0 "$SRV_PID" 2>/dev/null || break
      sleep 0.5
    done
  fi
  exit 143
}
trap on_term TERM INT

fetch_weights() {
  _wanted_checkpoint="${MODEL_REPO:-}"
  if [ -n "${MODEL_REVISION:-}" ]; then
    _wanted_checkpoint="${MODEL_REPO:-}@${MODEL_REVISION}"
  fi
  # Which repo do the bytes in this directory actually belong to? A model dir is
  # keyed by name, but MODEL_DIR can be pinned by the template while MODEL_REPO
  # moves with the family — and downloading one model on top of another produces
  # a directory that is neither, with a boot failure a long way from the cause.
  _repo_marker="$MODEL_DIR/.download-repo"
  _complete_marker="$MODEL_DIR/.download-complete"
  _complete_repo=""
  [ -f "$_complete_marker" ] && _complete_repo="$(cat "$_complete_marker" 2>/dev/null || true)"
  # A checkpoint-identity mismatch must RETURN failure, not `exit`: fetch_weights
  # is invoked as `if ! fetch_weights` (prepare_checkpoint), and an `exit` inside
  # a function called as an if-condition still terminates PID 1 — bypassing the
  # supervisor's verify/rollback loop and crash-looping the rental with no
  # recovery (the state file survives on the volume). Returning 1 lets the caller
  # roll back to the last known-good config, as the header contract promises.
  if [ -f "$_repo_marker" ]; then
    _have=$(cat "$_repo_marker" 2>/dev/null || echo "")
    if [ -n "$_have" ] && [ "$_have" != "${MODEL_REPO:-}" ]; then
      echo "!!! $MODEL_DIR already holds '$_have' but this configuration wants"
      echo "    '${MODEL_REPO:-?}'. Refusing to download one model on top of another."
      echo "    Point MODEL_DIR somewhere else, or delete that directory first."
      return 1
    fi
  elif [ -n "$_complete_repo" ] &&
       [ "${_complete_repo%%@*}" != "${MODEL_REPO:-}" ]; then
    echo "!!! $MODEL_DIR holds '$_complete_repo' but this configuration wants"
    echo "    '${MODEL_REPO:-?}'. Refusing to mix checkpoints in one directory."
    return 1
  fi
  # A local/NFS bind mount is an explicit immutable checkpoint selection. Do
  # not try to update it to the profile's download revision: that would fail on
  # a read-only mount and would violate the request to serve these exact bytes.
  # The repository mismatch checks above still prevent a different model from
  # being mistaken for this profile.
  if [ "${MODEL_READ_ONLY:-0}" = "1" ] && [ -f "$MODEL_DIR/config.json" ]; then
    if [ -n "$_complete_repo" ] &&
       [ "$_complete_repo" != "$_wanted_checkpoint" ]; then
      echo ">>> Checkpoint: immutable local marker '$_complete_repo' supersedes"
      echo ">>>             profile download revision '$_wanted_checkpoint'."
    else
      echo ">>> Checkpoint: using immutable local bytes at $MODEL_DIR"
    fi
    return 0
  fi
  # Gate on a completion marker, not config.json: small files land early in the
  # parallel download, so config.json existing does not mean the shards made it.
  # snapshot_download resumes/verifies incrementally, so re-running is safe.
  if [ -f "$_complete_marker" ] &&
     { [ -z "$_complete_repo" ] || [ "$_complete_repo" = "$_wanted_checkpoint" ]; }; then
    return 0
  fi
  status_update downloading-weights
  # Stamp the directory's identity BEFORE any bytes land: a partial download
  # followed by a family/variant switch must hit the mismatch gate above, not
  # interleave two checkpoints' identically-named shards in one directory.
  mkdir -p "$MODEL_DIR"
  printf '%s' "${MODEL_REPO:-}" > "$_repo_marker" 2>/dev/null || true
  boot_note ">>> Downloading ${MODEL_REPO:-?} (${MODEL_FAMILY:-glm52} family) to $MODEL_DIR"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo ">>> (HF_TOKEN detected: authenticated download)"
  else
    echo ">>> (set HF_TOKEN env for higher rate limits)"
  fi
  # HF_HUB_OFFLINE is exported later in the boot; a variant switch at runtime has
  # to be able to reach the hub again, so clear it for the download only.
  # set -e is unreliable here: fetch_weights is invoked as `if ! fetch_weights`
  # by its caller, and bash suppresses errexit for a function's ENTIRE body
  # when the function is the tested command of an if/while condition. A
  # failure inside must therefore be checked explicitly, not left to errexit,
  # or it is silently swallowed and the completion marker below gets written
  # for a download that never actually finished.
  # tqdm redraws one terminal line with carriage returns. Runpod and several
  # Docker log collectors buffer that line until its final newline, making an
  # active download look stuck. Disable that display and emit durable,
  # newline-delimited deciles from a small sidecar monitor instead.
  _download_gib="${MODEL_DOWNLOAD_GIB:-}"
  if [ -z "$_download_gib" ]; then
    _download_gib="$(python3 - "$SCRIPTS_DIR" "${MODEL_VARIANT:-}" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1])
import glm_config as gc
print(gc.VARIANTS.get(sys.argv[2], {}).get("download_gib", 0))
PY
)"
  fi
  case "$_download_gib" in
    ""|*[!0-9.]*|*.*.*) _download_gib=0 ;;
  esac
  HF_HUB_OFFLINE=0 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MODEL_REPO="${MODEL_REPO:-brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw}" \
    MODEL_DIR="$MODEL_DIR" python3 -u -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_REPO"], revision=os.environ.get("MODEL_REVISION") or "main", local_dir=os.environ["MODEL_DIR"], max_workers=16)' &
  _download_pid=$!
  python3 -u "$SCRIPTS_DIR/download_progress.py" \
    --pid "$_download_pid" --directory "$MODEL_DIR" \
    --expected-gib "$_download_gib" --boot-notes "$BOOT_NOTES" &
  _progress_pid=$!
  _download_rc=0
  # `wait` returns 128+signal when a trapped signal arrives — the dashboard's
  # USR1 wake-up hint lands here during long downloads. Treating that as a
  # download failure killed healthy multi-hundred-GiB transfers and respawned
  # a second concurrent download; re-wait until the child has actually exited.
  #
  # The status MUST be captured in the else branch: after `fi`, `$?` is the
  # exit status of the *if statement* (0 when an untested-else condition
  # fails), which silently turned every real download failure into a success
  # and stamped the completion marker for an incomplete checkpoint.
  while :; do
    if wait "$_download_pid"; then
      _download_rc=0
      break
    else
      _download_rc=$?
    fi
    # Only a signal-shaped status (>128) with a live child means "interrupted,
    # keep waiting". A genuine nonzero exit breaks out immediately.
    if [ "$_download_rc" -gt 128 ]; then
      if kill -0 "$_download_pid" 2>/dev/null; then
        continue
      fi
      # Reaped between the interrupted wait and this probe: bash retains the
      # real status for exactly one more wait. Anything ambiguous stays
      # nonzero so the caller retries (snapshot_download resumes safely).
      if wait "$_download_pid" 2>/dev/null; then
        _download_rc=0
      else
        _download_rc=$?
      fi
    fi
    break
  done
  # The monitor normally notices the reaped child within five seconds. Stop it
  # immediately so successful startup is not delayed by a reporting helper.
  kill "$_progress_pid" 2>/dev/null || true
  wait "$_progress_pid" 2>/dev/null || true
  if [ "$_download_rc" -ne 0 ]; then
    echo "!!! weights download failed for ${MODEL_REPO:-?}"
    return 1
  fi
  boot_note ">>> Download progress: 100% (snapshot verified)"
  printf '%s\n' "$_wanted_checkpoint" > "$_complete_marker"
  printf '%s' "${MODEL_REPO:-}" > "$MODEL_DIR/.download-repo" 2>/dev/null || true
  boot_note ">>> Weights ready (${MODEL_REPO:-?})."
  return 0
}

# MTP78 trellis draft (default ON) — the MTP draft layer quantized to 3.0bpw EXL3
# (all-256-expert, full-corpus calibrated), validated at BF16 acceptance PARITY
# while freeing ~3.8 GB/GPU for KV cache.
#
# MTP78_MODE=graft (default) does in-place surgery on layer 78 of the target.
# It avoids a second draft directory and has the longest appliance evidence.
#
#   MTP78_MODE=override  SEPARATE draft dir via --speculative-config. Leaves the
#     target byte-identical and measured slightly better on acceptance (MAL 3.548 /
#     84.9% vs 3.517 / 83.9%). v29 stamps it as a draft at construction and
#     gives its m=1..3 graph shapes the capturable Trellis plan they require.
#   MTP78_MODE=off       native in-checkpoint draft
# MTP78_TRELLIS=0 is still honoured as a synonym for off (it is folded into the
# MTP_DRAFT knob by the config resolver, which also accepts MTP78_MODE directly).
MTP78_DRAFT_DIR=""

fetch_mtp78_overlay() {
  local _overlay="$MODEL_DIR/.mtp78-overlay/3bpw-keep0"
  # Gate on the payload shard, not the directory: snapshot_download creates the
  # directory before the shards land, so a kill/blip mid-download leaves a dir
  # that every later boot treats as complete and never re-fetches — permanently
  # serving the degraded native BF16 draft with no recovery. Re-invoking
  # snapshot_download resumes safely.
  if ls "$_overlay"/*.safetensors >/dev/null 2>&1; then
    return 0
  fi
  HF_HUB_OFFLINE=0 MODEL_DIR="$MODEL_DIR" python3 -c '
import os
from huggingface_hub import snapshot_download
snapshot_download("malaiwah/GLM-5.2-EXL3-TR3-MTP78", allow_patterns=["3bpw-keep0/*"], local_dir=os.environ["MODEL_DIR"] + "/.mtp78-overlay", max_workers=8)' || return 1
  # Confirm the shard actually landed before declaring the overlay ready.
  ls "$_overlay"/*.safetensors >/dev/null 2>&1
}

prepare_nvfp4_mtp78_draft() {
  [ "${MTP_DRAFT:-}" = "nvfp4" ] || return 0
  if [ -f "${DRAFT_MODEL:-}/config.json" ] &&
     [ -f "${DRAFT_MODEL:-}/model.safetensors.index.json" ]; then
    echo ">>> MTP78: external NVFP4 draft ready at $DRAFT_MODEL"
    return 0
  fi
  # The Luke checkpoint publishes its MTP layer in two dedicated payloads, so
  # do not download the other ~400 GiB merely to build a 5.6 GiB draft.  Keep
  # the partial source and the derived directory outside the target checkpoint:
  # the same path works after a variant switch and never mutates shared weights.
  local draft_source="${DRAFT_SOURCE_DIR:-${MODEL_ROOT:-/workspace}/.draft-sources/lukealonso-GLM-5.2-NVFP4}"
  echo ">>> MTP78: downloading Luke NVFP4 draft payloads only (~5.6 GiB)"
  mkdir -p "$(dirname "$draft_source")" "$(dirname "$DRAFT_MODEL")"
  HF_HUB_OFFLINE=0 HF_XET_HIGH_PERFORMANCE=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'lukealonso/GLM-5.2-NVFP4',
    allow_patterns=[
        'config.json', 'generation_config.json', 'tokenizer.json',
        'tokenizer_config.json', 'chat_template.jinja',
        'model.safetensors.index.json', 'model-mtp.safetensors',
        'model-mtp-inputscales.safetensors',
    ],
    local_dir='$draft_source',
    max_workers=8,
)"
  python3 "$SCRIPTS_DIR/build_nvfp4_mtp_draft.py" \
    "$draft_source" "$DRAFT_MODEL"
}

prepare_mtp78() {
  MTP78_DRAFT_DIR="$MODEL_DIR/.mtp78-draft"
  # DRAFT_MODEL may be supplied from the environment (or the state file) to point
  # --speculative-config at an EXTERNAL draft checkpoint, independently of
  # MTP78_MODE. This is how a non-EXL3 draft (e.g. the NVFP4 MTP draft from
  # lukealonso/GLM-5.2-NVFP4, model-mtp.safetensors, 5.6 GiB) is used: it recovers
  # most of the KV that the BF16 fallback draft gives up, while still avoiding
  # _apply_rank_sliced, which is useful for comparing quantization formats.
  if [ -n "${DRAFT_MODEL:-}" ]; then
    prepare_nvfp4_mtp78_draft
    if [ ! -f "$DRAFT_MODEL/config.json" ] ||
       [ ! -f "$DRAFT_MODEL/model.safetensors.index.json" ]; then
      echo "FATAL: external draft is incomplete: $DRAFT_MODEL"
      exit 1
    fi
    echo ">>> Draft: external checkpoint $DRAFT_MODEL (MTP78_MODE=$MTP78_MODE ignored for draft selection)"
    MTP78_MODE=off
  fi

  if [ "$MTP78_MODE" = "override" ]; then
    status_update preparing-mtp78
    if python3 /opt/scripts/patch_deepseek_mtp.py; then
      if fetch_mtp78_overlay && python3 /opt/scripts/build_mtp78_draft.py \
           "$MODEL_DIR" "$MODEL_DIR/.mtp78-overlay/3bpw-keep0" "$MTP78_DRAFT_DIR"; then
        DRAFT_MODEL="$MTP78_DRAFT_DIR"
        echo ">>> MTP78: trellis draft via --speculative-config override (target untouched)"
      else
        echo "!!! MTP78: draft build failed — falling back to the native in-checkpoint draft"
      fi
    else
      echo "!!! MTP78: vLLM patch anchor missing in this image — keeping the native draft"
    fi
  elif [ "$MTP78_MODE" = "graft" ]; then
    if [ ! -f "$MODEL_DIR/.mtp78-grafted" ]; then
      status_update grafting-mtp78
      fetch_mtp78_overlay
      if python3 /opt/scripts/patch_deepseek_mtp.py; then
        if python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" "$MODEL_DIR/.mtp78-overlay/3bpw-keep0"; then
          echo ">>> MTP78: trellis draft active (grafted in place)"
        else
          echo "!!! MTP78 graft failed — reverting to the native draft"
          python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" --revert || true
        fi
      else
        echo "!!! MTP78: vLLM patch anchor missing in this image — keeping the native draft"
      fi
    else
      python3 /opt/scripts/patch_deepseek_mtp.py || true  # image may have been re-pulled
      echo ">>> MTP78: trellis draft already grafted"
    fi
  else
    echo ">>> MTP78 modification disabled — native in-checkpoint draft (${NATIVE_MTP_FORMAT:-checkpoint format})"
  fi
  # A checkpoint grafted by an earlier boot must be reverted before override mode
  # can work, otherwise the target still carries the trellis layer 78.
  if [ "$MTP78_MODE" != "graft" ] && [ -f "$MODEL_DIR/.mtp78-grafted" ] &&
     [ "${NATIVE_MTP_FORMAT:-}" != "exl3-tr3" ]; then
    echo ">>> Reverting a previous in-place graft (MTP78_MODE=$MTP78_MODE)"
    python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" --revert || true
  elif [ "$MTP78_MODE" != "graft" ] &&
       [ "${NATIVE_MTP_FORMAT:-}" = "exl3-tr3" ]; then
    echo ">>> Native checkpoint already carries EXL3-TR3 layer-78 experts; no graft/revert required"
  fi
  return 0
}

# Vision (opt-in): MoonViT-3d tower (Kimi-K2.6, frozen) + Baseten's trained
# 49.5M-param PatchMerger projector, bolted onto our EXL3 text backbone. Both
# vision parts are BF16 and checkpoint-agnostic — only ~1 GB, no text weight is
# touched. VISION=0 serves pure text.
# The EXL3 loader is multimodal-aware (it collapses `language_model.` prefixes),
# so the rank-sliced text weights still load under the Glm5v wrapper.
#
# KNOWN BROKEN AT LONG CONTEXT on the EXL3 target (measured 2026-07-26): 32K
# needle 0/2 with degenerate text on both the v20 and pre-v20 bases, while
# short-prompt and vision smoke tests pass 6/6. The config layer warns; the
# post-restart probe fails it by design.
install_vision_plugin() {
  local plugin_dir="$MODEL_DIR/.vision/plugins/glm5v_nf3"
  local install_dir
  if [ ! -d "$plugin_dir" ]; then
    echo "!!! Vision: vLLM plugin source is missing at $plugin_dir"
    return 1
  fi

  # pip/setuptools refreshes source-side egg-info even when the package is
  # already built. That fails when a local/NFS checkpoint is intentionally
  # mounted read-only, and the suppressed install error otherwise surfaces
  # later as the misleading "Glm5vForConditionalGeneration is not supported".
  # Install from a disposable writable copy and avoid network build isolation.
  install_dir="$(mktemp -d /tmp/glm5v-install.XXXXXX)" || return 1
  cp -a "$plugin_dir/." "$install_dir/" || return 1
  rm -rf "$install_dir/build" "$install_dir/glm5v_nf3.egg-info"
  pip install -q --no-build-isolation --no-deps "$install_dir"
}

prepare_vision() {
  VISION_REPO="${VISION_REPO:-chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid}"
  if [ "${VISION:-0}" = "1" ]; then
    # Gate on what config.json ACTUALLY says, not just the marker file. An MTP78 graft
    # REVERT rewrites config.json back to its pre-vision form, which silently strips the
    # Glm5v wrapper while leaving .vision-enabled in place -- the installer then reported
    # "already installed", skipped re-wrapping, and the server answered image requests
    # with "GLM-5.2 is not a multimodal model" (400). Measured on AIBeast 2026-07-26.
    # reconcile_checkpoint.py now closes that hole from the other side, by deriving the
    # wrapper from the tensors on disk instead of from a snapshot.
    if [ -f "$MODEL_DIR/.vision-enabled" ] && \
       ! python3 -c "
import json,sys
c=json.load(open('$MODEL_DIR/config.json'))
sys.exit(0 if 'text_config' in c or any('Glm5v' in a for a in c.get('architectures') or []) else 1)" 2>/dev/null; then
      echo ">>> Vision: marker present but config.json is text-only (graft revert clobbers it) -- reinstalling"
      rm -f "$MODEL_DIR/.vision-enabled"
    fi
    if [ ! -f "$MODEL_DIR/.vision-enabled" ]; then
      status_update installing-vision
      echo ">>> Vision: downloading tower + projector + processor (~1 GB) from $VISION_REPO"
      vision_ok=1
      HF_HUB_OFFLINE=0 VISION_REPO="$VISION_REPO" MODEL_DIR="$MODEL_DIR" python3 -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["VISION_REPO"], local_dir=os.environ["MODEL_DIR"] + "/.vision",
  allow_patterns=["vision_tower.safetensors","mm_projector.safetensors","config.json",
                  "configuration_glm5v.py","kimi_k25_processor.py","kimi_k25_vision_processing.py",
                  "media_utils.py","preprocessor_config.json","chat_template.jinja",
                  "plugins/**"], max_workers=8)' || vision_ok=0
      if [ "$vision_ok" = "1" ]; then
        cp "$MODEL_DIR/.vision"/vision_tower.safetensors "$MODEL_DIR/.vision"/mm_projector.safetensors "$MODEL_DIR/" || vision_ok=0
        for f in configuration_glm5v.py kimi_k25_processor.py kimi_k25_vision_processing.py media_utils.py preprocessor_config.json; do
          if [ -f "$MODEL_DIR/.vision/$f" ]; then
            cp "$MODEL_DIR/.vision/$f" "$MODEL_DIR/"
          fi
        done
        # The vision chat template is MANDATORY: the text-only template never emits
        # <|begin_of_image|><|image|><|end_of_image|>, so the multimodal processor
        # fails with "Failed to apply prompt replacement for mm_items['vision_chunk']".
        # Keep the text-only one for VISION=0 restore.
        if [ -f "$MODEL_DIR/.vision/chat_template.jinja" ]; then
          if [ -f "$MODEL_DIR/chat_template.jinja" ] && [ ! -f "$MODEL_DIR/chat_template.jinja.text-only" ]; then
            cp "$MODEL_DIR/chat_template.jinja" "$MODEL_DIR/chat_template.jinja.text-only"
          fi
          cp "$MODEL_DIR/.vision/chat_template.jinja" "$MODEL_DIR/chat_template.jinja"
        else
          vision_ok=0
        fi
        install_vision_plugin || vision_ok=0
      fi
      if [ "$vision_ok" = "1" ] && python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" "$MODEL_DIR/.vision" && python3 /opt/scripts/index_add_vision.py "$MODEL_DIR"; then
        echo ">>> Vision: ENABLED (image input active; VISION=0 to disable)"
      else
        echo "!!! Vision install failed — falling back to text-only for this boot"
        python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" --revert || true
        python3 /opt/scripts/index_add_vision.py "$MODEL_DIR" --revert || true
        # The install did NOT succeed, so the achieved state is text-only. Force
        # VISION=0 for the rest of this boot: prepare_checkpoint runs the
        # reconciler next with the REQUESTED VISION value, and reconciling to
        # --vision 1 would immediately re-wrap config.json to Glm5v and recreate
        # .vision-enabled over a checkpoint whose vLLM plugin never registered —
        # the exact "Glm5vForConditionalGeneration is not supported" crash-loop
        # this fallback exists to prevent. A later boot retries the install.
        VISION=0
      fi
    else
      # The marker lives on the VOLUME but the plugin is installed into the
      # CONTAINER's site-packages, so a replaced container (restart, image bump)
      # skips the block above and starts with a config.json that asks for
      # Glm5vForConditionalGeneration while the plugin that registers it is gone:
      #   Model architectures ['Glm5vForConditionalGeneration'] are not supported
      # vLLM then crash-loops until the supervisor gives up. Re-install every boot;
      # it is a ~50 KB local wheel and pip is a no-op when already satisfied.
      if [ -d "$MODEL_DIR/.vision/plugins/glm5v_nf3" ]; then
        if install_vision_plugin; then
          echo ">>> Vision: already installed (plugin re-registered in this container)"
        else
          echo "FATAL: Vision plugin installation failed."
          exit 1
        fi
      else
        echo "FATAL: Vision marker is present but $MODEL_DIR/.vision/plugins/glm5v_nf3 is missing."
        echo "       The Glm5v architecture cannot register. Remove $MODEL_DIR/.vision-enabled"
        echo "       to reinstall, or set VISION=0 to serve text-only."
        exit 1
      fi
    fi
  elif [ -f "$MODEL_DIR/.vision-enabled" ]; then
    echo ">>> VISION=0: reverting to text-only config"
    python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" --revert || true
    python3 /opt/scripts/index_add_vision.py "$MODEL_DIR" --revert || true
    if [ -f "$MODEL_DIR/chat_template.jinja.text-only" ]; then
      # copy, not mv: the backup has to survive so a later VISION=1 -> 0 cycle
      # still has a text-only template to restore.
      cp "$MODEL_DIR/chat_template.jinja.text-only" "$MODEL_DIR/chat_template.jinja" || true
    fi
  fi
  return 0
}

# Every checkpoint transition ends in the reconciler. Both legacy revert paths
# restore a config SNAPSHOT, and a snapshot goes stale the moment anything else
# changes: revert a graft after vision was installed and the restored config
# still claims hybrid_tr3_tail.moe_layers=[3,78] while layer 78 is BF16, which
# is exactly the
#   KeyError: model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight
# crash. reconcile_checkpoint.py re-derives moe_layers, the layer-78 ignore
# patterns, the architectures/wrapper and the weight index from the TENSORS ON
# DISK, so a transition cannot leave the config describing weights that are not
# there. It is idempotent and cheap (safetensors headers only).
prepare_checkpoint() {
  # Fix 4: a termination request during checkpoint prep (tens of minutes on
  # first boot) would otherwise be invisible — the supervisor only checks
  # TERMINATE_FLAG inside the post-serve monitoring loop. Abort prep early so
  # the supervisor loop can handle the termination cleanly.
  if [ -f "$TERMINATE_FLAG" ]; then return 0; fi
  # Fix 3: fetch_weights runs under `set -e`; a download failure (network
  # blip, bad HF_TOKEN, invalid MODEL_ID) would kill PID 1 immediately,
  # bypassing the supervisor's restart/rollback loop. Catch it here instead.
  if ! fetch_weights; then
    echo "!!! checkpoint download failed; supervisor will retry"
    return 1
  fi
  # The MTP78 graft/overlay and the Glm5v vision wrapper are surgery on a GLM-5.2
  # checkpoint's layer 78 and config.json. They are meaningless — and would be
  # destructive — anywhere else, so they are gated on the family rather than on
  # their own knobs, which the config layer has already marked inapplicable.
  if [ "${MODEL_FAMILY:-glm52}" = "glm52" ]; then
    if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then
      # Local/NFS deployments can expose a checkpoint directly from their shared
      # Hugging Face cache. Never run graft, vision, or reconciliation writes on
      # that source: a turnkey experiment must not mutate the canonical weights.
      # A complete external draft makes MTP78_MODE irrelevant: prepare_mtp78
      # validates that separate read-only mount and normalizes the mode to off.
      # Only reject modes that would mutate the target when no external draft
      # has been supplied.
      if { [ "${MTP78_MODE:-off}" != "off" ] && [ -z "${DRAFT_MODEL:-}" ]; }; then
        echo "FATAL: MODEL_READ_ONLY=1 cannot graft MTP78 into the target."
        echo "       Supply a complete external DRAFT_MODEL or prepare a writable derivative."
        exit 1
      fi
      if [ "${VISION:-0}" = "1" ] && [ ! -f "$MODEL_DIR/.vision-enabled" ]; then
        echo "FATAL: MODEL_READ_ONLY=1 cannot install vision into an unprepared target."
        echo "       Mount a derivative containing .vision-enabled, or use a writable copy."
        exit 1
      fi
      if [ "${VISION:-0}" = "1" ]; then
        install_vision_plugin || {
          echo "FATAL: read-only vision derivative cannot register its vLLM plugin."
          exit 1
        }
        echo ">>> Vision: read-only derivative plugin registered"
      fi
      if [[ "${MODEL_VARIANT:-exl3-tr3}" == exl3-tr3* ]]; then
        python3 "$SCRIPTS_DIR/reconcile_checkpoint.py" "$MODEL_DIR" \
          --vision "${VISION:-0}" --dry-run --quiet || {
          echo "FATAL: read-only EXL3 checkpoint does not pass reconciliation."
          exit 1
        }
      fi
      # An explicit external draft is independent of the immutable target.  It
      # may itself be a read-only bind mount (local/NFS) or may be assembled in
      # a separate writable volume (rental).  Validate/prepare it here while
      # continuing to skip every operation that could touch MODEL_DIR.
      if [ -n "${DRAFT_MODEL:-}" ]; then
        prepare_mtp78
      fi
      echo ">>> Checkpoint is read-only; mutation steps skipped."
    else
      prepare_mtp78
      prepare_vision
      python3 "$SCRIPTS_DIR/reconcile_checkpoint.py" "$MODEL_DIR" --vision "${VISION:-0}" || \
        echo "!!! reconcile: failed — the checkpoint config may not match the weights on disk"
    fi
  elif [ "${MODEL_FAMILY:-}" = "glm53" ]; then
    if ! MODEL_DIR="$MODEL_DIR" MODEL_VARIANT="${MODEL_VARIANT:-glm53-k6}" \
         MAX_MODEL_LEN="${MAX_MODEL_LEN:-458752}" python3 - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
config = json.loads((model_dir / "config.json").read_text())
variant = os.environ["MODEL_VARIANT"]
expected_bits = {"glm53-k6": 6, "glm53-k8": 8}.get(variant)
if expected_bits is None:
    raise SystemExit(f"FATAL: unsupported GLM-5.3 variant {variant!r}")
architectures = config.get("architectures") or []
if architectures != ["Glm5NextForConditionalGeneration"]:
    raise SystemExit(
        "FATAL: GLM-5.3 profile requires Glm5NextForConditionalGeneration; "
        f"checkpoint declares {architectures!r}"
    )
quant = config.get("quantization_config") or {}
if (
    str(quant.get("quant_method", "")).lower() != "exl3"
    or str(quant.get("codebook", "")).lower() != "mcg"
    or int(quant.get("bits", 0)) != expected_bits
):
    raise SystemExit(
        f"FATAL: {variant} requires EXL3/MCG K{expected_bits}; "
        f"checkpoint declares {quant!r}"
    )
text = config.get("text_config") or {}
native_context = int(text.get("max_position_embeddings", 0))
requested_context = int(os.environ["MAX_MODEL_LEN"])
if native_context < requested_context:
    raise SystemExit(
        f"FATAL: requested context {requested_context} exceeds checkpoint native "
        f"limit {native_context}"
    )
if not (
    text.get("mla_use_nope") is True
    and int(text.get("index_kpool", 0)) == 4
    and text.get("index_kpool_compress") is True
):
    raise SystemExit(
        "FATAL: GLM-5.3 sparse-MLA NOPE/K-pool contract is missing from config.json"
    )
print(
    f">>> GLM-5.3 checkpoint verified: {architectures[0]}, "
    f"EXL3/MCG K{expected_bits}, native context {native_context}"
)
PY
    then
      echo "!!! checkpoint verification failed; supervisor will retry"
      return 1
    fi
  elif [ "${MODEL_FAMILY:-}" = "qwen36" ]; then
    if ! MODEL_DIR="$MODEL_DIR" MTP_TOKENS="${MTP_TOKENS:-0}" python3 - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
config = json.loads((model_dir / "config.json").read_text())
architectures = config.get("architectures") or []
if "Qwen3_5ForConditionalGeneration" not in architectures:
    raise SystemExit(
        "FATAL: qwen36-nvfp4 requires Qwen3_5ForConditionalGeneration; "
        f"checkpoint declares {architectures!r}"
    )
quant = config.get("quantization_config") or {}
method = str(quant.get("quant_method", "")).lower()
algorithm = str(quant.get("quant_algo", "")).upper()
if method != "modelopt" or algorithm not in {
    "MIXED_PRECISION", "NVFP4", "W4A16_NVFP4"
}:
    raise SystemExit(
        "FATAL: qwen36-nvfp4 requires a ModelOpt NVFP4-compatible checkpoint; "
        f"found quant_method={method!r}, quant_algo={algorithm!r}"
    )
if int(os.environ["MTP_TOKENS"]) > 0:
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = (
        json.loads(index_path.read_text()).get("weight_map") if index_path.is_file() else {}
    ) or {}
    if not any(name.startswith("mtp.") for name in weight_map):
        raise SystemExit(
            "FATAL: Qwen MTP was enabled but the checkpoint has no mtp.* weights"
        )
print(f">>> Qwen checkpoint verified: {architectures[0]}, ModelOpt {algorithm}")
PY
    then
      echo "!!! checkpoint verification failed; supervisor will retry"
      return 1
    fi
  else
    echo ">>> ${MODEL_FAMILY:-glm52}: no checkpoint surgery (MTP78 graft and the vision"
    echo ">>> wrapper are GLM-5.2-specific); serving the downloaded weights as they are."
  fi
  return 0
}

# DRAM KV offload: OFFLOAD_FRACTION of the instance's RAM allocation;
# OFFLOAD_FRACTION=0 (the default) disables. Sized from min(cgroup limit, MemTotal) —
# inside a container /proc/meminfo shows the whole host's RAM, but a partial
# rental (e.g. 4 of 8 GPUs) only gets a slice of it.
KVT_ARGS=()
compute_offload() {
  KVT_ARGS=()
  export LMCACHE_MODE=off
  local off_fraction="${OFFLOAD_FRACTION:-0}"
  local cache_backend="${PREFIX_CACHE_BACKEND:-native}"
  if [ "$off_fraction" != "0" ] && [ ! -r /proc/meminfo ]; then
    echo ">>> DRAM KV offload sizing skipped: /proc/meminfo is unavailable on this smoke host"
    off_fraction=0
  fi
  if [ "$off_fraction" != "0" ]; then
    MEM_BYTES=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') * 1024 ))
    CG_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
    if [ "$CG_LIMIT" != "max" ] && [ "$CG_LIMIT" -lt "$MEM_BYTES" ] 2>/dev/null; then
      MEM_BYTES=$CG_LIMIT
    fi
    # Current native OffloadingConnector interprets cpu_bytes_to_use as the
    # AGGREGATE cache budget. CPUOffloadingSpec multiplies one worker's block
    # bytes by world_size before deriving its shared block count; each worker
    # then allocates only its rank-local slice. Dividing here silently divides
    # the effective tier by TP (measured live: OFFLOAD_FRACTION=0.5 advertised
    # 125 GiB but retained only ~31 GiB, so a 71 GiB store stream had no hits).
    # Keep OFF_BYTES as the physical per-worker estimate for the memlock check
    # and operator display, but pass OFF_TOTAL_BYTES to the connector.
    OFF_TOTAL_BYTES=$(python3 -c "print(int($MEM_BYTES*$off_fraction))")
    OFFLOAD_RANKS="${TENSOR_PARALLEL_SIZE:-1}"
    [ "$OFFLOAD_RANKS" -gt 0 ] 2>/dev/null || OFFLOAD_RANKS=1
    OFF_BYTES=$((OFF_TOTAL_BYTES / OFFLOAD_RANKS))
    MEMLOCK_KB=$(ulimit -l)
    if [ "$MEMLOCK_KB" != "unlimited" ] && [ "$MEMLOCK_KB" -lt "$((OFF_BYTES / 1024))" ] 2>/dev/null; then
      echo "!!! WARNING: memlock ulimit (${MEMLOCK_KB} KB) is below the $((OFF_BYTES/1073741824)) GiB KV pool to pin."
      if [ "${OFFLOAD_IGNORE_MEMLOCK:-1}" = "1" ]; then
        # Measured 2026-07-26 on an owned box: a 128 GiB offload tier runs fine with
        # memlock capped at ~31 GiB (kv_offload_total_bytes climbs normally), because
        # the connector does not mlock the whole tier up front. Rootless podman also
        # cannot raise memlock past the user's hard limit, so '--ulimit memlock=-1:-1'
        # is a no-op there and the check would disable a working feature outright.
        # Opt-in, because on an unknown rental host the conservative default is right.
        echo "!!! Proceeding anyway (warn-and-proceed is the default; OFFLOAD_IGNORE_MEMLOCK=0 to"
        echo "!!! disable offload instead). This degrades rather than fails: the connector does not"
        echo "!!! mlock the tier up front (measured: a 125 GiB tier offloading normally under a"
        echo "!!! 31 GiB memlock), and kv_load_failure_policy=recompute means any KV that cannot be"
        echo "!!! brought back is simply recomputed. Verify kv_offload_* metrics climb."
      else
        echo "!!! Add '--ulimit memlock=-1:-1' to the template Docker options to enable offload,"
        echo "!!! or set OFFLOAD_IGNORE_MEMLOCK=1 if you have verified offload works at this limit."
        echo "!!! Continuing WITHOUT DRAM offload."
        off_fraction=0
      fi
    fi
  fi
  if [ "$off_fraction" != "0" ]; then
    if [ "$cache_backend" = "lmcache" ]; then
      export LMCACHE_L1_GB="$((OFF_TOTAL_BYTES/1073741824))"
      [ "$LMCACHE_L1_GB" -gt 0 ] || export LMCACHE_L1_GB=1
      # Fraction-based L2 sizing: if PREFIX_CACHE_DISK_FRACTION is set (0-1),
      # compute L2 as that fraction of free disk on the model root filesystem.
      # This takes precedence over the fixed PREFIX_CACHE_DISK_GB value.
      if [ -n "${PREFIX_CACHE_DISK_FRACTION:-}" ] && \
         python3 -c "import sys; sys.exit(0 if float('${PREFIX_CACHE_DISK_FRACTION}') > 0 else 1)" 2>/dev/null; then
        _free_disk_bytes="$(df -B1 --output=avail "${MODEL_ROOT:-/workspace}" 2>/dev/null | tail -1 | tr -d '[:space:]')"
        _fraction="$(python3 -c "print(float('${PREFIX_CACHE_DISK_FRACTION}'))" 2>/dev/null || echo 0)"
        _computed_l2_gb="$(python3 -c "print(int(${_free_disk_bytes:-0} * ${_fraction} / 1073741824))" 2>/dev/null || echo 0)"
        export PREFIX_CACHE_DISK_GB="$_computed_l2_gb"
        echo ">>> LMCache L2: PREFIX_CACHE_DISK_FRACTION=${_fraction} of free disk (${_free_disk_bytes:-unknown} bytes) -> ${_computed_l2_gb} GiB"
        unset _free_disk_bytes _fraction _computed_l2_gb
      fi
      export LMCACHE_L2_GB="${PREFIX_CACHE_DISK_GB:-0}"
      if [ "$LMCACHE_L2_GB" -gt 0 ] 2>/dev/null; then
        export LMCACHE_MODE=disk
        export LMCACHE_L2_PATH="${LMCACHE_L2_PATH:-$MODEL_ROOT/.lmcache/l2}"
        OFFLOAD_STATE="LMCache ${LMCACHE_L1_GB} GiB DRAM + ${LMCACHE_L2_GB} GiB NVMe"
        echo ">>> LMCache prefix tier: ${LMCACHE_L1_GB} GiB aggregate DRAM + ${LMCACHE_L2_GB} GiB bounded NVMe"
        echo "!!! Cached KV can contain session material. Enable secure termination; disk erase is best effort."
      else
        export LMCACHE_MODE=ram
        OFFLOAD_STATE="LMCache ${LMCACHE_L1_GB} GiB DRAM"
        echo ">>> LMCache prefix tier: ${LMCACHE_L1_GB} GiB aggregate DRAM (no disk tier)"
      fi
      # The r11 helper rejects CUDA VMM/expandable segments for its pinned pages.
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    else
      OFFLOAD_STATE="$((OFF_TOTAL_BYTES/1073741824)) GiB native DRAM prefix cache"
      echo ">>> Native DRAM KV offload: $((OFF_TOTAL_BYTES/1073741824)) GiB total (${off_fraction} of instance RAM allocation),"
      echo ">>>                          aggregate connector budget; ~$((OFF_BYTES/1073741824)) GiB physical slice x ${OFFLOAD_RANKS} TP workers"
      # kv_load_failure_policy=recompute: a KV block that cannot be fetched back from
      # the DRAM tier is recomputed instead of failing the request. The vLLM default
      # is "fail", which would turn any offload hiccup into a terminal error — not an
      # acceptable trade for a cache tier that is a pure optimisation.
      KVT_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF_TOTAL_BYTES}}")
      # OffloadingConnector rejects expandable_segments (VMM can remap pinned KV pages)
      export PYTORCH_CUDA_ALLOC_CONF=""
    fi
  else
    OFFLOAD_STATE="off"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  fi
  return 0
}

# Vision serve args: bound the encoder reserve (chronarion's guidance) and cap
# media per request so a screenshot burst can't blow the profile.
VISION_ARGS=()
compute_vision_args() {
  VISION_ARGS=()
  if [ "${MODEL_FAMILY:-glm52}" = "glm52" ] && [ "${VISION:-0}" = "1" ] \
     && [ -f "$MODEL_DIR/.vision-enabled" ]; then
    VISION_ARGS=(--limit-mm-per-prompt "{\"vision_chunk\":${VISION_CHUNKS:-8}}")
  fi
  return 0
}

# ---- engine environment -----------------------------------------------------
# Split in two. The COMMON block is true of any model on this hardware. The
# GLM-5.2 block below it is the "each one is a measured decision" list — b12x
# kernels, the MLA/DCP path, the EXL3 trellis — and it is guarded by family
# because forcing those onto a dense non-MLA model is exactly how you get a
# configuration that cannot work and does not say why.
#
# The block stays HERE rather than in glm_config.py on purpose: it has to run
# after the image's own ENV in order to override it (see the TUNE_ note below),
# and that ordering is not expressible from the config layer.
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUTE_DSL_ARCH=sm_120a OMP_NUM_THREADS=16
export SAFETENSORS_FAST_GPU=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple
export TORCH_CUDA_ARCH_LIST=12.0a FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_FLASHINFER_SAMPLER=1
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
# The base image still carries three historical launcher switches that no
# pinned vLLM/SparkInfer source reads. Leaving them inherited achieves nothing
# and makes vLLM report misleading "unknown environment variable" warnings.
unset VLLM_USE_B12X_PCIE_DMA
unset VLLM_RTX6K_FUSED_ALLREDUCE_ADD
unset VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER

refresh_family_runtime_env() {
if [ "${FAMILY_ENV_BLOCK:-glm52}" = "glm52" ]; then
# These are profile-owned r20 online-quantization controls. Clear them on every
# supervisor start so changing exl3-b6 -> none cannot retain the previous
# conversion/cache contract in the same long-lived PID 1 environment.
unset VLLM_EXL3_ONLINE_TRELLIS_BITS VLLM_EXL3_ENCODER_SOURCE
unset VLLM_EXL3_ONLINE_CACHE_DIR VLLM_EXL3_ONLINE_CACHE_MODE
export VLLM_USE_B12X_FP8_GEMM=1 VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_B12X_MOE=1 VLLM_USE_V2_MODEL_RUNNER=1
if [ "${B12X_PCIE_DMA:-1}" = "1" ]; then
  export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  export VLLM_USE_B12X_DCP_A2A=1
else
  export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_USE_B12X_DCP_A2A=0
  unset VLLM_PCIE_ALLREDUCE_BACKEND
fi
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
export VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0
export VLLM_USE_AOT_COMPILE=1 VLLM_USE_MEGA_AOT_ARTIFACT=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_FUSED_MOE_GROUPED_TOPK=1
export VLLM_USE_B12X_MHC=1 B12X_MHC_MAX_TOKENS=16384 VLLM_USE_B12X_WO_PROJECTION=1
export B12X_MLA_SM120_UNIFIED=1 B12X_DENSE_SPLITK_TURBO=1
export B12X_W4A16_TC_DECODE=1 B12X_W4A8_TINY_DECODE=1 B12X_MOE_FORCE_A16=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0 VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=8
export VLLM_DCP_A2A_MAX_TOKENS=16 VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
export VLLM_DCP_GLOBAL_TOPK=1 VLLM_DCP_SHARD_DRAFT=1 VLLM_DCP_QUERY_SPLIT=1
export VLLM_DCP_TOPK_OWNER_MERGE=1 VLLM_DCP_INDEXER_SHARDS=0
export VLLM_B12X_MLA_CKV_GATHER=1 VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=512
export VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS="${DCP_CKV_GATHER_MAX_TOKENS:-140000}"
export VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-auto}"
export VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB="${DCP_PREFILL_WORKSPACE_MIB:-1024}"
# The borrowed-workspace implementation supports TP4/DCP4, but not TP4/DCP2
# (the latter has 16/32/32 local/input/kernel heads and is rejected by
# _validate_dcp_prefill_workspace_contract). Keep the fast workspace path for
# DCP4 and select the supported ordinary gather/project path for DCP1/DCP2.
if [ "${DCP:-4}" = "4" ]; then
  export VLLM_DCP_PROJECT_BEFORE_MERGE=1
  export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1
else
  export VLLM_DCP_PROJECT_BEFORE_MERGE=0
  export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0
fi
export VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS=1024
export VLLM_PCIE_DMA_FP8="${F8_DMA:-0}"
export B12X_PCIE_DMA_FP8="${F8_DMA:-0}"
export SPARKINFER_PCIE_DMA_FP8="${F8_DMA:-0}"
if [ "${MODEL_VARIANT:-exl3-tr3}" = "madeby561-hybrid" ]; then
  export VLLM_B12X_ABSORB_BMM=1 VLLM_NF3_GRID188_DECODE=1
else
  unset VLLM_B12X_ABSORB_BMM VLLM_NF3_GRID188_DECODE
fi
# r20's native path retains role-aware target/draft ownership and
# makes m=1 capturable for both roles. Keep the variable absent with MTP so the
# backend owns the minimum. The explicit MTP-off value preserves the same m=1
# contract for older compatible GG bases and makes that otherwise implicit
# choice visible in diagnostics.
if [ "${MTP_TOKENS:-3}" = "0" ]; then
  export VLLM_EXL3_TRELLIS_MIN_M=1
else
  unset VLLM_EXL3_TRELLIS_MIN_M
fi
# config.env has already replaced any inherited image value with the resolved
# self-service value (32 by default). Preserve it here so the validation,
# warning, EXL3 planner and CUDA-graph ceiling all describe the same window.
export VLLM_EXL3_TRELLIS_MAX_M="${VLLM_EXL3_TRELLIS_MAX_M:-32}"
export VLLM_EXL3_TRELLIS_BLOCK_M=8 VLLM_EXL3_PREFILL_CHUNK=128
# PR #210 separates the reusable EXL3 prefill arena from the scheduler chunk.
# config.env supplies the family/variant value; this explicit export both keeps
# it visible in EngineCore workers and makes the otherwise indirect env
# consumer auditable by the knob-wiring gate.
export VLLM_EXL3_PREFILL_CAPACITY="${VLLM_EXL3_PREFILL_CAPACITY:-${MAX_NUM_BATCHED_TOKENS:-3072}}"
# Keep the graph-aware, attention-aware profiler as the safe default, while
# honoring an explicit expert override.  A previous unconditional export made
# `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` appear accepted in the container
# environment but silently replaced it in EngineCore workers.
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN="${VLLM_MEMORY_PROFILE_INCLUDE_ATTN:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
unset VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

# GG v20-r9 has two mutually exclusive NVFP4 MLA record contracts. Our
# entrypoint bypasses the upstream family helper, so it must select the
# contract explicitly rather than assuming the parent image exported it.
# The appliance downloads the immutable, metadata-corrected public sidecar;
# verify its exact reviewed payload before using it. Dynamic-token mode owns
# [292,296) in the 368-byte FP8-RoPE record and therefore must not see a static
# scales file.
_GLM_NVFP4_SCALE_FILE=/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json
_GLM_NVFP4_SCALE_SHA256=ac68fe6af3056ec35299361293c9ae568769d21696756548493f67ff17881ece
if [ "${KV_CACHE_DTYPE:-nvfp4_ds_mla}" = "nvfp4_ds_mla" ]; then
  case "${KV_SCALE_MODE:-static-calibrated}" in
    dynamic-token)
      export KV_FP8_ROPE=1
      export VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
      unset VLLM_NVFP4_MLA_SCALES_FILE
      echo ">>> NVFP4 MLA scale mode: dynamic-token (FP8-RoPE 368-byte record)"
      ;;
    static-calibrated)
      export KV_FP8_ROPE=0
      unset VLLM_NVFP4_MLA_DYNAMIC_SCALE
      export VLLM_NVFP4_MLA_SCALES_FILE="$_GLM_NVFP4_SCALE_FILE"
      if [ "${CONFIG_SMOKE:-0}" != "1" ]; then
        _actual_scale_sha256="$(
          sha256sum "$VLLM_NVFP4_MLA_SCALES_FILE" 2>/dev/null | awk '{print $1}'
        )"
        if [ "$_actual_scale_sha256" != "$_GLM_NVFP4_SCALE_SHA256" ]; then
          echo "FATAL: the pinned GLM NVFP4 MLA scale artifact is missing or changed:" >&2
          echo "FATAL: $VLLM_NVFP4_MLA_SCALES_FILE" >&2
          return 1
        fi
        unset _actual_scale_sha256
      fi
      echo ">>> NVFP4 MLA scale mode: static-calibrated (verified GLM-5.2 artifact)"
      ;;
    *)
      echo "FATAL: KV_SCALE_MODE must be static-calibrated or dynamic-token." >&2
      return 1
      ;;
  esac
else
  unset KV_FP8_ROPE VLLM_NVFP4_MLA_DYNAMIC_SCALE VLLM_NVFP4_MLA_SCALES_FILE
fi
export SPARKINFER_INDEXER_TWO_LEVEL_FOLD="${SPARKINFER_INDEXER_TWO_LEVEL_FOLD:-auto}"
export SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB="${SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB:-256}"
unset _GLM_NVFP4_SCALE_FILE _GLM_NVFP4_SCALE_SHA256

  # v20's calibrator measures the lossless DMA crossover, DCP query split and
  # CKV prefetch overlap before model load, then caches the result by topology.
  # Compressed wire modes remain explicit experiments and the image launcher
  # deliberately skips BF16 calibration for them.
  if [ "${CONFIG_SMOKE:-0}" != "1" ] &&
     [ -x /usr/local/bin/serve-glm52-v19.sh ]; then
    _cal_gpus="$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))"
    if _cal_env="$(
      env -u DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS -u VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS \
        -u PCIE_DMA_MIN_BYTES -u VLLM_PCIE_DMA_MIN_BYTES \
        TP="${TENSOR_PARALLEL_SIZE:-4}" DCP="${DCP:-4}" GPUS="$_cal_gpus" \
        DCP_QUERY_SPLIT=auto DCP_TOPK_OWNER_MERGE=auto \
        DCP_INDEXER_SHARDS=auto \
        DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-auto}" \
        DCP_CKV_PREFETCH_WORKSPACE_MIB="${DCP_PREFILL_WORKSPACE_MIB:-1024}" \
        B12X_PCIE_DMA="${B12X_PCIE_DMA:-1}" F8_DMA="${F8_DMA:-0}" \
        PCIE_CALIBRATION="${PCIE_CALIBRATION:-auto}" PCIE_CALIBRATION_ONLY=1 \
        PCIE_CALIBRATION_CACHE_DIR="$GLM_STATE_DIR/pcie-calibration" \
        /usr/local/bin/serve-glm52-v19.sh
    )"; then
      while IFS= read -r _cal_line; do
        case "$_cal_line" in
          VLLM_DCP_QUERY_SPLIT=*|VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=*|\
          VLLM_B12X_MLA_CKV_GATHER=*|VLLM_DCP_TOPK_OWNER_MERGE=*|\
          VLLM_DCP_INDEXER_SHARDS=*|VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=*|\
          VLLM_PCIE_DMA_MIN_BYTES=*|PCIE_CALIBRATION_STATUS=*|\
          PCIE_CALIBRATION_CACHE=*)
            # A plain export assigns the KEY=VALUE held in the variable without
            # re-evaluating its contents ($(...), ;) the way eval would.
            # shellcheck disable=SC2163
            export "$_cal_line"
            ;;
        esac
      done <<< "$_cal_env"
      echo ">>> PCIe calibration: ${PCIE_CALIBRATION_STATUS:-unknown}; DMA minimum ${VLLM_PCIE_DMA_MIN_BYTES:-unknown}; CKV prefetch ${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH:-unknown}"
    else
      echo "!!! PCIe calibration launcher failed; retaining conservative runtime values."
    fi
    unset _cal_gpus _cal_env _cal_line
  elif [ "${DCP_CKV_PREFETCH_DEPTH:-auto}" = "auto" ]; then
    export VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0
  fi
  # Named profiles may pin measured crossovers after the topology calibrator.
  # -1 keeps the calibrated value; explicit values make a qualified profile
  # reproducible on providers where the calibration helper is unavailable.
  if [ "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:--1}" != "-1" ]; then
    export VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="$DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS"
  fi
  if [ "${PCIE_DMA_MIN_BYTES:--1}" != "-1" ]; then
    export VLLM_PCIE_DMA_MIN_BYTES="$PCIE_DMA_MIN_BYTES"
  fi
elif [ "${FAMILY_ENV_BLOCK:-}" = "glm53" ]; then
  # Exact environment used by the four-card K6 qualification. The pinned parent
  # carries the Glm5Next/B12X stack; these exports make its load-bearing OCI
  # defaults explicit and make a runtime switch back from another family safe.
  export PYTHONPATH=/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x:/opt/exllamav3
  export OMP_NUM_THREADS=1 LLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_EXL3_EXT_PATH=/opt/exllamav3
  export VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3
  export VLLM_EXL3_ENCODER_REVISION=704aefd743b390af4bd0fb429d1906f9b964c7d8
  export VLLM_EXL3_ONLINE_CACHE_DIR="$MODEL_ROOT/.exl3-online"
  export VLLM_EXL3_ONLINE_CACHE_MODE=readwrite
  export VLLM_EXL3_PREFILL_BLOCK_M=128 VLLM_EXL3_PREFILL_TRELLIS=1
  export B12X_GL53_ROUTE128_WIDE=1 B12X_GL53_ROUTE128_HYBRID_TAIL=1
  export VLLM_B12X_GLM_NOPE_NVFP4=1
  export VLLM_USE_AOT_COMPILE=0 VLLM_USE_MEGA_AOT_ARTIFACT=1
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  export VLLM_USE_B12X_FP8_GEMM=1 VLLM_USE_B12X_MOE=0
  export VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0
  export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
  export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
  export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
  export VLLM_B12X_ABSORB_BMM=0 B12X_MOE_FORCE_A16=1
  if [ "${B12X_PCIE_DMA:-1}" = "1" ]; then
    export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x
    export VLLM_USE_B12X_DCP_A2A=1
  else
    export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_USE_B12X_DCP_A2A=0
    unset VLLM_PCIE_ALLREDUCE_BACKEND
  fi
  _glm53_nccl=/opt/local-inference/nccl/lib/libnccl.so.2.31.2
  if [ -f "$_glm53_nccl" ]; then
    export NCCL_LOCAL_INFERENCE_PATH="$_glm53_nccl"
    export NCCL_PR2127_PATH="$_glm53_nccl"
    export VLLM_NCCL_SO_PATH="$_glm53_nccl"
    export LD_PRELOAD="$_glm53_nccl"
  elif [ "${CONFIG_SMOKE:-0}" = "1" ]; then
    unset NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH VLLM_NCCL_SO_PATH LD_PRELOAD
  else
    echo "FATAL: pinned local-inference NCCL library is missing: $_glm53_nccl" >&2
    return 1
  fi
  unset _glm53_nccl
  export VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json
  export VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 KV_FP8_ROPE=0
  if [ "${CONFIG_SMOKE:-0}" != "1" ]; then
    _glm53_scale_sha="$(
      sha256sum "$VLLM_NVFP4_MLA_SCALES_FILE" 2>/dev/null | awk '{print $1}'
    )"
    if [ "$_glm53_scale_sha" != "f10b6ee1116d71c4b61c4603d38cb257b3f0dcfde9bcc0847839e48ac9baeb1d" ]; then
      echo "FATAL: the pinned GLM-5.3 NVFP4 MLA scale artifact is missing or changed." >&2
      return 1
    fi
    unset _glm53_scale_sha
  fi
  unset VLLM_EXL3_ONLINE_TRELLIS_BITS VLLM_EXL3_ABI_SHIM
  unset VLLM_USE_FUSED_MOE_GROUPED_TOPK VLLM_USE_B12X_MHC B12X_MHC_MAX_TOKENS
  unset VLLM_USE_B12X_WO_PROJECTION B12X_MLA_SM120_UNIFIED B12X_DENSE_SPLITK_TURBO
  unset B12X_W4A16_TC_DECODE B12X_W4A8_TINY_DECODE
  unset VLLM_DISABLE_SHARED_EXPERTS_STREAM VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE
  unset VLLM_B12X_MLA_SPEC_DECODE_MAX_Q VLLM_DCP_A2A_MAX_TOKENS
  unset VLLM_DCP_A2A_LARGE_BACKEND VLLM_DCP_GLOBAL_TOPK VLLM_DCP_SHARD_DRAFT
  unset VLLM_DCP_QUERY_SPLIT VLLM_DCP_TOPK_OWNER_MERGE VLLM_DCP_INDEXER_SHARDS
  unset VLLM_B12X_MLA_CKV_GATHER VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS
  unset VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS VLLM_B12X_MLA_CKV_PREFETCH_DEPTH
  unset VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB
  unset VLLM_DCP_PROJECT_BEFORE_MERGE VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE
  unset VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS
  unset VLLM_PCIE_DMA_FP8 B12X_PCIE_DMA_FP8 SPARKINFER_PCIE_DMA_FP8
  unset SPARKINFER_INDEXER_TWO_LEVEL_FOLD SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB
  echo ">>> GLM-5.3 runtime: TP4/DCP4 B12X sparse MLA, Triton MoE, EXL3, NVFP4 MLA KV"
else
  echo ">>> ${MODEL_FAMILY:-?}: skipping the GLM-5.2 engine environment (b12x kernels,"
  echo ">>> MLA/DCP path, EXL3 trellis). Only the common block above is exported."
  # The base image carries GLM-specific OCI environment values. Remove them for
  # Qwen/custom so a profile switch cannot inherit MLA, EXL3, or custom NCCL
  # behavior merely because the container image was built for GLM.
  unset VLLM_EXL3_EXT_PATH VLLM_EXL3_ABI_SHIM VLLM_EXL3_PREFILL_CAPACITY
  unset VLLM_EXL3_ONLINE_TRELLIS_BITS VLLM_EXL3_ENCODER_SOURCE
  unset VLLM_EXL3_ONLINE_CACHE_DIR VLLM_EXL3_ONLINE_CACHE_MODE
  unset VLLM_NVFP4_MLA_SCALES_FILE
  unset VLLM_NVFP4_MLA_DYNAMIC_SCALE KV_FP8_ROPE
  unset SPARKINFER_INDEXER_TWO_LEVEL_FOLD SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB
  unset VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND
  unset VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS
  unset VLLM_DISABLE_SHARED_EXPERTS_STREAM VLLM_DISABLED_KERNELS
  unset VLLM_PCIE_DMA_FP8 B12X_PCIE_DMA_FP8
  unset SPARKINFER_PCIE_DMA_FP8 VLLM_PCIE_DMA_MIN_BYTES
  unset VLLM_B12X_ABSORB_BMM VLLM_NF3_GRID188_DECODE
  unset NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH VLLM_NCCL_SO_PATH LD_PRELOAD
fi
}

# ---- tuning overrides -------------------------------------------------------
# TUNE_<NAME>=<value> exports <NAME>=<value> here, i.e. AFTER the authoritative
# block above. This is what makes a knob A/B-able without rebuilding the image.
#
# Why a TUNE_ prefix instead of rewriting the block as ${NAME:-default}: the
# block deliberately OVERRIDES image-level ENV, and for two knobs the image ships
# a different value than we want --
#     VLLM_PCIE_ALLREDUCE_BACKEND        image=cpp  -> we force b12x
#     VLLM_DISABLE_SHARED_EXPERTS_STREAM image=0    -> we force 1
# so ${NAME:-default} would let the image's value win and silently change the
# allreduce backend. The prefix cannot collide with an inherited name.
#
# Values are echoed so a bench result can always be traced back to the exact
# configuration that produced it, e.g.
#     podman run -e TUNE_VLLM_EXL3_PREFILL_CHUNK=256 ...
#
# VLLM_EXL3_TRELLIS_MIN_M deliberately is not a self-service knob on v29:
# leaving it absent is the role-aware safe setting. Advanced experiments can
# still set TUNE_VLLM_EXL3_TRELLIS_MIN_M explicitly.
apply_tuning_overrides() {
while IFS='=' read -r _k _v; do
  case "$_k" in
    TUNE_*)
      _n="${_k#TUNE_}"
      [ -n "$_n" ] || continue
      export "$_n=$_v"
      echo ">>> tuning override: $_n=$_v"
      ;;
  esac
done < <(env)
unset _k _v _n
}

# A measured model variant may carry a coherent engine-route bundle that is
# too low-level to expose as twenty independent first-time-user controls.
# Apply it after the family defaults; explicit TUNE_<NAME> values remain the
# final layer and can still override every entry for controlled experiments.
apply_profile_runtime_env() {
for _assignment in "${PROFILE_RUNTIME_ENV[@]}"; do
  _n="${_assignment%%=*}"
  _v="${_assignment#*=}"
  [ -n "$_n" ] || continue
  export "$_n=$_v"
  echo ">>> profile runtime: $_n=$_v"
done
unset _assignment _n _v
}

# API key: VLLM_API_KEY env > persisted key on the volume > freshly generated.
# Persisting matters: a restart (supervisor, reboot, manual) that minted a NEW
# key would silently invalidate every client config the user already pasted.
#
# AUTH=none disables authentication entirely. This exists for trusted private
# networks (e.g. replacing an in-house endpoint whose clients are already
# configured without a key); it is NEVER appropriate on a rented public host,
# so it is opt-in and loudly announced. Default stays "always authenticated".
AUTH="${AUTH:-key}"
KEYFILE="${MODEL_DIR%/*}/.vllm-api-key"
if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  AUTH=none            # a smoke run must not mint or persist a key
fi
if [ "$AUTH" = "none" ]; then
  VLLM_API_KEY=""
  echo "=================================================================="
  echo ">>> AUTH=none - the endpoint is UNAUTHENTICATED."
  echo ">>> Only do this on a trusted private network. Anyone who can reach"
  echo ">>> port ${PORT:-8000} can use this model."
  echo "=================================================================="
fi
if [ "$AUTH" != "none" ] && [ -z "${VLLM_API_KEY:-}" ] && [ -s "$KEYFILE" ]; then
  VLLM_API_KEY="$(cat "$KEYFILE")"
  echo ">>> API key: reusing the persisted key from $KEYFILE"
fi
if [ "$AUTH" != "none" ] && [ -z "${VLLM_API_KEY:-}" ]; then
  VLLM_API_KEY="sk-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  ( umask 077; printf '%s' "$VLLM_API_KEY" > "$KEYFILE" ) 2>/dev/null || true
  echo "=================================================================="
  echo ">>> API KEY (auto-generated, persisted for restarts;"
  echo ">>>          set VLLM_API_KEY env to override):"
  echo ">>> $VLLM_API_KEY"
  echo "=================================================================="
fi
export VLLM_API_KEY
status_update configuring
start_soul_supervisor

# TLS via Let's Encrypt DNS-01 (optional): set ACME_DOMAIN + ACME_DNS_PROVIDER
# (lego provider name, e.g. cloudflare, duckdns) + the provider's cred envs
# (e.g. CLOUDFLARE_DNS_API_TOKEN, or DUCKDNS_TOKEN). Certs persist on the
# volume and are reused while valid >7 days, else re-issued at boot.
# Turnkey auto-DNS (deSEC): set DESEC_TOKEN + DESEC_DOMAIN (your *.dedyn.io zone).
# A per-instance name — stable across reboots (keyed to the provider instance
# ID) so records do not pile up and the LE certificate can be reused.
current_public_ip() {
  local live_ip=""
  live_ip="$(curl -fsS -m 10 https://api.ipify.org 2>/dev/null || true)"
  printf '%s' "${live_ip:-${PUBLIC_IPADDR:-${RUNPOD_PUBLIC_IP:-}}}"
}
PUBLIC_IP_CURRENT="$(current_public_ip)"
PUBLIC_API_PORT="${VAST_TCP_PORT_8000:-${RUNPOD_TCP_PORT_8443:-${JARVISLABS_PUBLIC_PORT_8000:-${PORT:-8000}}}}"
PUBLIC_DASHBOARD_PORT="${VAST_TCP_PORT_1111:-${JARVISLABS_PUBLIC_PORT_1111:-1111}}"

# Wire an ISSUED certificate into the serving path: vLLM's --ssl-* args (or the
# Runpod direct-TLS socat listener), the internal /etc/hosts alias + verify
# flag, and the endpoint/status/LOCAL_BASE picture. Idempotent and safe to call
# again — this is what lets a certificate that only arrives from the BACKGROUND
# acme retry, minutes into the session, become usable on the next vLLM restart
# instead of requiring a whole container replacement. Returns 0 iff TLS is
# active afterwards. CRT/KEY/ACME_DIRECT are globals set by the ACME block below.
wire_tls_endpoint() {
  if [ "${TLS_ENABLED:-0}" != "1" ] && [ "${ACME_DIRECT:-0}" = "1" ] &&
     [ -n "${CRT:-}" ] && [ -n "${KEY:-}" ] &&
     [ -f "$CRT" ] && [ -f "$KEY" ] && [ "${CONFIG_SMOKE:-0}" != "1" ]; then
    TLS_ENABLED=1
    if [ "$PLATFORM" = "runpod" ]; then
      if [ -z "${TLS_PROXY_PID:-}" ] || ! kill -0 "${TLS_PROXY_PID:-0}" 2>/dev/null; then
        socat "OPENSSL-LISTEN:8443,cert=${CRT},key=${KEY},verify=0,reuseaddr,fork" \
          "TCP4:127.0.0.1:${PORT:-8000}" &
        TLS_PROXY_PID=$!
        sleep 1
        kill -0 "$TLS_PROXY_PID" 2>/dev/null || {
          echo "!!! Runpod direct-TLS proxy failed to start; using managed HTTPS proxy"
          TLS_ENABLED=0
        }
      fi
    else
      TLS_ARGS=(--ssl-certfile "$CRT" --ssl-keyfile "$KEY")
      # Reach the engine by the certificate hostname (resolved to loopback) so
      # internal TLS clients verify the cert; anchor the /etc/hosts match so a
      # sub.$ACME_DOMAIN entry does not suppress installing the real name.
      case "$ACME_DOMAIN" in
        ""|*[!A-Za-z0-9.-]*)
          echo "!!! Unsafe ACME hostname; internal TLS alias was not installed"
          ;;
        *)
          if ! grep -Eq "(^|[[:space:]])${ACME_DOMAIN}([[:space:]]|\$)" /etc/hosts 2>/dev/null; then
            printf '127.0.0.1\t%s\n' "$ACME_DOMAIN" >> /etc/hosts
          fi
          TLS_INTERNAL_ALIAS=1
          ;;
      esac
    fi
    [ "$TLS_ENABLED" = "1" ] &&
      echo ">>> TLS enabled: https://$ACME_DOMAIN:${PUBLIC_API_PORT}/v1"
  fi
  if [ "${TLS_ENABLED:-0}" = "1" ]; then
    EP_URL="https://${ACME_DOMAIN}:${PUBLIC_API_PORT}"
    TLS_STATE="https://${ACME_DOMAIN}"
    [ "$PLATFORM" != "runpod" ] && HTTPS_HOSTPORT="${ACME_DOMAIN}:${PUBLIC_DASHBOARD_PORT}"
    CERT_PATH="$CRT" KEY_PATH="$KEY"
    LOCAL_SCHEME="http"; LOCAL_HOST="localhost"
    if [ "$PLATFORM" != "runpod" ]; then
      LOCAL_SCHEME="https"
      if [ "${TLS_INTERNAL_ALIAS:-0}" = "1" ]; then
        LOCAL_HOST="$ACME_DOMAIN"
        export GLM_INTERNAL_TLS_VERIFY=1
      fi
    fi
    LOCAL_BASE="$LOCAL_SCHEME://$LOCAL_HOST:${PORT:-8000}"
    return 0
  fi
  return 1
}

# Runpod's HTTP proxy terminates TLS and forwards plain HTTP to the container.
# App-level TLS is only appropriate on the separately exposed direct TCP port.
ACME_DIRECT=1
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" != "1" ]; then
  ACME_DIRECT=0
  if [ -n "${ACME_DOMAIN:-}${DESEC_TOKEN:-}${DESEC_DOMAIN:-}" ]; then
    echo "!!! Runpod proxy mode ignores ACME/deSEC settings; the proxy already terminates TLS."
    echo "!!! Expose 8443/tcp and set RUNPOD_DIRECT_TLS=auto or 1 for direct TCP TLS."
  fi
fi

if [ "$ACME_DIRECT" = "1" ] && [ -n "${DESEC_TOKEN:-}" ] &&
   [ -n "${DESEC_DOMAIN:-}" ] && [ -z "${ACME_DOMAIN:-}" ] &&
   [ "${CONFIG_SMOKE:-0}" != "1" ]; then
  _dns_id="${INSTANCE_ID:-}"
  if [ -z "$_dns_id" ]; then
    # No provider instance id (generic host): persist a random suffix on the
    # volume so reboots reuse ONE DNS name and ONE LE certificate instead of
    # minting a fresh pair per boot — records pile up in the zone and the LE
    # duplicate-certificate limit (5/week) burns down in a reboot loop.
    _sub_file="$GLM_STATE_DIR/.dns-suffix"
    _dns_id="$(cat "$_sub_file" 2>/dev/null || true)"
    if [ -z "$_dns_id" ]; then
      _dns_id="$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
      mkdir -p "$GLM_STATE_DIR" 2>/dev/null || true
      printf '%s' "$_dns_id" > "$_sub_file" 2>/dev/null || true
    fi
  fi
  SUB="model-${_dns_id}"
  MYIP="$PUBLIC_IP_CURRENT"
  export ACME_DOMAIN="${SUB}.${DESEC_DOMAIN}" ACME_DNS_PROVIDER=desec DESEC_TOKEN
  if [ -z "$MYIP" ]; then
    echo "!!! Could not determine public IP; deSEC/ACME will retry in the background"
  else
    # ttl 3600 = deSEC's account minimum; lower values are rejected with HTTP 400
    echo ">>> Registering ${SUB}.${DESEC_DOMAIN} -> ${MYIP} via deSEC"
    # The token travels in a 0600 header file, not argv: /proc/*/cmdline is
    # world-readable inside the container (including by the soul account).
    # Every step degrades to the background retry rather than aborting the
    # boot — this whole block is explicitly best-effort.
    _desec_hdr=""
    if _desec_hdr="$(umask 077 && mktemp "${GLM_RUNTIME_DIR:-/tmp}/desec-hdr.XXXXXX")" &&
       printf 'Authorization: Token %s\n' "$DESEC_TOKEN" > "$_desec_hdr"; then
      curl -sf -X PUT "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/" \
        -H @"$_desec_hdr" -H "Content-Type: application/json" \
        -d "[{\"subname\":\"${SUB}\",\"type\":\"A\",\"ttl\":3600,\"records\":[\"${MYIP}\"]}]" >/dev/null \
        && echo ">>> Registered. Endpoint will be: https://${ACME_DOMAIN}:${PUBLIC_API_PORT}/v1" \
        || echo "!!! deSEC registration failed; continuing startup and retrying in the background"
    else
      echo "!!! Could not stage the deSEC auth header; retrying in the background"
    fi
    [ -z "$_desec_hdr" ] || rm -f "$_desec_hdr"
  fi
fi

TLS_ARGS=()
TLS_ENABLED=0
TLS_INTERNAL_ALIAS=0
if [ "$ACME_DIRECT" = "1" ] && [ -n "${ACME_DOMAIN:-}" ] &&
   [ -n "${ACME_DNS_PROVIDER:-}" ] && command -v lego >/dev/null &&
   [ "${CONFIG_SMOKE:-0}" != "1" ]; then
  CRT="$MODEL_ROOT/.lego/certificates/${ACME_DOMAIN}.crt"
  KEY="$MODEL_ROOT/.lego/certificates/${ACME_DOMAIN}.key"
  # MODEL_ROOT persists across reboots: reuse a cert with >7 days left instead of
  # re-issuing every boot (LE duplicate-cert limit is 5/week; a reboot loop burns it)
  if [ -f "$CRT" ] && [ -f "$KEY" ] &&
     openssl x509 -checkend 604800 -noout -in "$CRT" >/dev/null 2>&1; then
    echo ">>> Reusing persisted cert for $ACME_DOMAIN (>7 days validity left)"
  else
    echo ">>> ACME/deSEC gets one bounded foreground attempt; serving is not held by later retries."
    export ACME_PUBLIC_IP="$PUBLIC_IP_CURRENT" SCRIPTS_DIR MODEL_ROOT
    if ! bash "$SCRIPTS_DIR/acme_retry.sh" --once; then
      echo "!!! TLS is not ready; continuing appliance startup now."
      echo "!!! ACME/deSEC will retry every ${ACME_BACKGROUND_RETRY_S:-300}s in the background."
      bash "$SCRIPTS_DIR/acme_retry.sh" --retry &
    fi
  fi
  # Wire the cert into the serving path if it is ready now (idempotent; the
  # supervisor loop calls this again so a late background cert is picked up).
  wire_tls_endpoint || true
fi

# Feed the final endpoint/TLS picture to the landing page's status file.
# wire_tls_endpoint set the TLS endpoint/LOCAL_BASE when a cert came up; fill in
# the non-TLS fallback (Runpod managed proxy, or plain HTTP) otherwise.
if [ "${TLS_ENABLED:-0}" != "1" ]; then
  if [ "$PLATFORM" = "runpod" ]; then
    EP_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"
    TLS_STATE="https:// managed by Runpod proxy"
  else
    EP_URL="http://${PUBLIC_IP_CURRENT:-localhost}:${PUBLIC_API_PORT}"
  fi
  LOCAL_SCHEME="http"
  LOCAL_HOST="localhost"
  LOCAL_BASE="$LOCAL_SCHEME://$LOCAL_HOST:${PORT:-8000}"
fi

if [ "$PLATFORM" = "runpod" ] && [ "$TLS_ENABLED" != "1" ]; then
  echo ">>> Runpod proxy endpoint: ${EP_URL}/v1"
  echo "!!! Runpod's HTTP proxy has a 100-second connection limit; use the SSH tunnel"
  echo "!!! documented in README.md for long generations and large-context requests."
elif [ "$PLATFORM" = "runpod" ]; then
  echo ">>> Runpod direct-TCP TLS endpoint: ${EP_URL}/v1"
elif [ "$PLATFORM" = "jarvislabs" ] && [ "$TLS_ENABLED" = "1" ]; then
  echo ">>> JarvisLabs dashboard: https://${ACME_DOMAIN}:${PUBLIC_DASHBOARD_PORT}/?token=${OPEN_BUTTON_TOKEN}"
elif [ "$PLATFORM" = "jarvislabs" ]; then
  echo "!!! JarvisLabs dashboard has no trusted public TLS URL."
  echo ">>> Use an SSH tunnel, then open: http://localhost:1111/?token=${OPEN_BUTTON_TOKEN}"
fi

# Egress hygiene: no telemetry; offline mode once weights are local
# Keep torch.compile, Triton and extension caches on the persistent volume.
# The upstream image exports /cache/jit paths, but /cache is not a mounted
# volume on Vast or Runpod, so a replaced container recompiles the
# backbone and the eagle head from scratch — ~100 s of every boot, for ~1.6 GB of
# artifacts. VLLM_DISABLE_COMPILE_CACHE is NOT set here (that guard belongs to the
# gilded-gnosis fork's cache bug); verify decode tok/s after enabling this.
#
# Do not share compiled objects across immutable runtime stacks. The parent
# fingerprint covers its base repositories, while the appliance's 21 K6
# overlays also change Python/Triton/AOT source consumed by Inductor and the
# runtime JITs. Keep an explicit overlay-generation suffix: restarts of this
# exact appliance remain warm, but old GG-v20 and unoverlaid GLM-5.3 artifacts
# cannot be reused.
CACHE_NAMESPACE="${LOCAL_INFERENCE_CACHE_FINGERPRINT:-turnkey-unversioned}-turnkey-glm53-k6-o21-v1"
case "$CACHE_NAMESPACE" in
  *[!A-Za-z0-9_.-]*|"")
    echo "FATAL: unsafe runtime cache fingerprint: $CACHE_NAMESPACE" >&2
    exit 2
    ;;
esac
unset VLLM_CACHE_DIR
case "${VLLM_CACHE_ROOT:-}" in
  ""|/cache/jit*)
    if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then
      export VLLM_CACHE_ROOT="/cache/$CACHE_NAMESPACE/vllm"
    else
      export VLLM_CACHE_ROOT="$MODEL_DIR/.vllm-cache/$CACHE_NAMESPACE/vllm"
    fi
    ;;
esac
case "${TRITON_CACHE_DIR:-}" in
  ""|/cache/jit*)
    if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then
      export TRITON_CACHE_DIR="/cache/$CACHE_NAMESPACE/triton"
    else
      export TRITON_CACHE_DIR="$MODEL_DIR/.vllm-cache/$CACHE_NAMESPACE/triton"
    fi
    ;;
esac
case "${TORCH_EXTENSIONS_DIR:-}" in
  ""|/cache/jit*)
    if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then
      export TORCH_EXTENSIONS_DIR="/cache/$CACHE_NAMESPACE/torch_extensions"
    else
      export TORCH_EXTENSIONS_DIR="$MODEL_DIR/.vllm-cache/$CACHE_NAMESPACE/torch_extensions"
    fi
    ;;
esac
case "${TORCHINDUCTOR_CACHE_DIR:-}" in
  ""|/cache/jit*)
    if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then
      export TORCHINDUCTOR_CACHE_DIR="/cache/$CACHE_NAMESPACE/torchinductor"
    else
      export TORCHINDUCTOR_CACHE_DIR="$MODEL_DIR/.vllm-cache/$CACHE_NAMESPACE/torchinductor"
    fi
    ;;
esac
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

recover_sparkinfer_extension_lock() {
  # A killed build can strand PyTorch's zero-byte FileBaton sentinel on the
  # persistent cache. The helper waits for a live compiler and only quarantines
  # the known SparkInfer PCIe DMA sentinel when it is ownerless. It never
  # removes Ninja's own lock or clears unrelated extensions.
  bash "$SCRIPTS_DIR/recover_torch_extension_lock.sh" "$TORCH_EXTENSIONS_DIR"
}

export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
echo ">>> Listening sockets at boot (expect only vllm on ${PORT:-8000} + vast ssh):"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | head -8 || true

# Concurrency vs the trellis/capture window. At decode the engine runs
#   m = MAX_NUM_SEQS * (1 + MTP_TOKENS)
# query tokens per step. m must stay inside BOTH the CUDA-graph capture window
# and the EXL3 trellis window [MIN_M, MAX_M], or decode silently falls off the
# captured trellis fast path onto the eager path once concurrency rises -- a
# cliff that no boot-time error announces and that only shows up as lost tok/s
# under real multi-stream load. This is now a pre-validation ERROR in
# glm_config.py (rule `concurrency-window`), so an apply from the landing page
# cannot reach the engine with it; the check below stays for env-only boots.
#
# This is why MAX_NUM_SEQS defaults to 8 and not 32: 8*(1+3)=32 exactly fills the
# default window. The old default of 32 gave m=128, four times outside it.
# To genuinely serve more streams, raise all three together, e.g. 16 seqs:
#   -e MAX_NUM_SEQS=16 \
#   -e CUDAGRAPH_CAPTURE_SIZES=8,16,24,32,40,48,56,64 \
#   -e MAX_CUDAGRAPH_CAPTURE_SIZE=64 -e TUNE_VLLM_EXL3_TRELLIS_MAX_M=64
# (more captured sizes cost capture time and some VRAM, so measure rather than
# assume it wins).
warn_capture_window() {
  # EXL3 trellis window: GLM family only. A dense model has no trellis and this
  # arithmetic would be meaningless noise on its boot log.
  if [ "${MODEL_FAMILY:-glm52}" != "glm52" ]; then
    return 0
  fi
  _DECODE_M=$(( ${MAX_NUM_SEQS:-8} * (1 + ${MTP_TOKENS:-3}) ))
  _CAP_MAX=${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}
  _TRELLIS_MAX=${VLLM_EXL3_TRELLIS_MAX_M:-32}
  if [ "$_DECODE_M" -gt "$_CAP_MAX" ] || [ "$_DECODE_M" -gt "$_TRELLIS_MAX" ]; then
    echo "!!! WARNING: MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} x (1+MTP_TOKENS=${MTP_TOKENS:-3}) = $_DECODE_M query tokens/step,"
    echo "!!!          but the cudagraph capture window tops out at $_CAP_MAX and the EXL3 trellis"
    echo "!!!          window at $_TRELLIS_MAX. Decode will leave the captured trellis path once"
    echo "!!!          enough streams are concurrent. Raise CUDAGRAPH_CAPTURE_SIZES,"
    echo "!!!          MAX_CUDAGRAPH_CAPTURE_SIZE and TUNE_VLLM_EXL3_TRELLIS_MAX_M to >= $_DECODE_M,"
    echo "!!!          or lower MAX_NUM_SEQS to $(( _CAP_MAX / (1 + ${MTP_TOKENS:-3}) ))."
  fi
  return 0
}

SPEC_ARGS=()
# DRAFT_QUANTIZATION: the draft's own quantization method. Without it the draft
# INHERITS the target's --quantization (exl3 here), so a non-EXL3 draft is loaded
# through the EXL3 path and dies in _apply_rank_sliced during speculator capture
# with the m=3 trellis-window error -- which looks identical to the v20 EXL3-draft
# bug but has a completely different cause. Measured on AIBeast 2026-07-26 with the
# NVFP4 draft from lukealonso/GLM-5.2-NVFP4 (quant_algo NVFP4, quant_method modelopt):
# vLLM's method name for it is "modelopt_fp4".
build_spec_args() {
  SPEC_ARGS=()
  if [ "${MTP_TOKENS:-3}" = "0" ]; then
    return 0
  fi
  # Speculation is not GLM-only, but its shape is family-specific: GLM-5.2 uses
  # method "mtp" with a triton MoE backend and an optional draft checkpoint;
  # Qwen uses the current vLLM "mtp" method without the GLM draft apparatus.
  if [ "${MODEL_FAMILY:-glm52}" != "glm52" ]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"${SPEC_METHOD:-mtp}\",\"num_speculative_tokens\":${MTP_TOKENS:-2}}")
    return 0
  fi
  _SPEC_QUANT=""
  _DRAFT_MOE_BACKEND="${DRAFT_MOE_BACKEND:-}"
  if [ -z "$_DRAFT_MOE_BACKEND" ]; then
    if [ "${DRAFT_QUANTIZATION:-}" = "modelopt_fp4" ]; then
      _DRAFT_MOE_BACKEND="b12x"
    else
      _DRAFT_MOE_BACKEND="triton"
    fi
  fi
  if [ -n "${DRAFT_QUANTIZATION:-}" ]; then
    _SPEC_QUANT=",\"quantization\":\"${DRAFT_QUANTIZATION}\""
    echo ">>> Draft quantization: ${DRAFT_QUANTIZATION} (overrides the target's --quantization)"
  fi
  if [ -n "${DRAFT_MODEL:-}" ]; then
    echo ">>> Draft MoE backend: $_DRAFT_MOE_BACKEND"
    SPEC_ARGS=(--speculative-config "{\"model\":\"$DRAFT_MODEL\",\"method\":\"${SPEC_METHOD:-mtp}\",\"num_speculative_tokens\":${MTP_TOKENS:-3},\"moe_backend\":\"${_DRAFT_MOE_BACKEND}\",\"draft_sample_method\":\"${MTP_DRAFT_SAMPLE_METHOD:-greedy}\",\"rejection_sample_method\":\"${MTP_REJECTION_SAMPLE_METHOD:-standard}\"${_SPEC_QUANT}}")
  else
    SPEC_ARGS=(--speculative-config "{\"method\":\"${SPEC_METHOD:-mtp}\",\"num_speculative_tokens\":${MTP_TOKENS:-3},\"moe_backend\":\"triton\",\"draft_sample_method\":\"${MTP_DRAFT_SAMPLE_METHOD:-greedy}\",\"rejection_sample_method\":\"${MTP_REJECTION_SAMPLE_METHOD:-standard}\"}")
  fi
  return 0
}

# Best-effort: surface readiness + endpoint into the vast.ai dashboard label
if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
  ( until curl -skf "$LOCAL_BASE/health" >/dev/null 2>&1; do sleep 20; done
    MODEL_LABEL="${SERVED_MODEL_NAME%% *} READY ${EP_URL}/v1"
    LABEL_JSON="$(MODEL_LABEL="$MODEL_LABEL" python3 -c \
      'import json,os; print(json.dumps({"label": os.environ["MODEL_LABEL"]}))')"
    # The account-scoped Vast key travels in a 0600 header file, not argv:
    # /proc/*/cmdline is world-readable inside the container.
    _lbl_hdr="$(umask 077 && mktemp "${GLM_RUNTIME_DIR:-/tmp}/vast-label-hdr.XXXXXX")"
    printf 'Authorization: Bearer %s\n' "$CONTAINER_API_KEY" > "$_lbl_hdr"
    curl -s -X PUT "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
      -H @"$_lbl_hdr" \
      -H "Content-Type: application/json" -d "$LABEL_JSON" >/dev/null 2>&1 || true
    rm -f "$_lbl_hdr"
  ) &
fi

# Supervisor: on a rental, an engine crash must not leave the box idle burning
# money. Restart vLLM (up to SUPERVISOR_MAX_RESTARTS, default 5) with backoff;
# a crash-loop that exceeds the budget exits so the instance is visibly failed
# rather than thrashing. SUPERVISOR=0 disables (single exec, legacy behaviour).
serve_once() {
# --served-model-name accepts several aliases; SERVED_MODEL_NAME is split on
# whitespace so a drop-in replacement can answer to the names existing clients
# already use (e.g. "GLM-5.2 local-primary") without touching those clients.
read -r -a SERVED_NAMES <<< "${SERVED_MODEL_NAME:-model}"
# AUTH=none -> omit --api-key entirely (passing an empty one still enforces auth).
# NB: `[ test ] && ARR=(...)` returns non-zero when the test is false, which
# under `set -e` kills serve_once outright — precisely in the AUTH=none and
# GPU_BLOCKS_OVERRIDE=0 cases these branches exist to support. Use if/then.
# vLLM reads VLLM_API_KEY from its environment (exported above; emptied for
# AUTH=none), so no --api-key argv is needed. Passing it on the argv exposed
# the secret in /proc/*/cmdline to every local user — including the sandboxed
# soul account — for the server's whole lifetime. The boot-time bad-key probe
# below verifies auth is actually enforced, so an environment-plumbing
# regression cannot go unnoticed.
AUTH_ARGS=()
# Pool sizing: auto-profile by default. On the v29 EXL3/MTP3 release this is
# about 1.1M tokens on four 96 GB cards, enough for two full 524K requests.
# A positive override remains available for reproducible smaller pools.
BLOCKS_ARGS=()
if [ "${GPU_BLOCKS_OVERRIDE:-0}" != "0" ]; then
  BLOCKS_ARGS=(--num-gpu-blocks-override "${GPU_BLOCKS_OVERRIDE:-0}")
fi
# stdout still goes to the vast console; the tee copy is what a rollback
# preserves as the failed boot's error log. Process substitution (not a pipe) so
# that $! stays the vllm process — a pipeline's exit would otherwise wait on tee,
# which surviving VLLM:: workers can hold open long after the engine is dead.
# GENERIC flags only. Everything architecture-specific — the quantization, the
# MLA/DCP flags, the attention and MoE backends, the tool/reasoning parsers, the
# sparse-indexer hf_overrides, the compilation config — comes from
# FAMILY_SERVE_ARGS, which the config layer built for the selected MODEL_FAMILY.
# That is what makes a second model family possible without a second serve line,
# and what removed `--tensor-parallel-size 4` as a literal.
# GLM-5.3 qualification used the parent image's system Python plus the pinned
# source-tree PYTHONPATH. Its /opt/venv console script has a different Python
# environment, so do not silently cross that runtime boundary.
if [ "${FAMILY_ENV_BLOCK:-}" = "glm53" ]; then
  _VLLM_LAUNCH=(python3 -m vllm.entrypoints.cli.main)
elif command -v vllm >/dev/null 2>&1; then
  _VLLM_LAUNCH=(vllm)
else
  _VLLM_LAUNCH=(python3 -m vllm.entrypoints.cli.main)
fi
_SERVE_CMD=("${_VLLM_LAUNCH[@]}" serve "$MODEL_DIR" \
  --served-model-name "${SERVED_NAMES[@]}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-nvfp4_ds_mla}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.96}" \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-3072}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}" \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --enable-force-include-usage \
  --no-async-scheduling \
  ${FAMILY_SERVE_ARGS[@]+"${FAMILY_SERVE_ARGS[@]}"} \
  ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
  ${BLOCKS_ARGS[@]+"${BLOCKS_ARGS[@]}"} ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
  "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}")
if [ "${LMCACHE_MODE:-off}" != "off" ]; then
  _SERVE_CMD=(bash "$SCRIPTS_DIR/glm52_lmcache_wrapper.sh" "${_SERVE_CMD[@]}")
fi
# SERVE_PRINT_ONLY: show the exact argv and return, for CONFIG_SMOKE. Printing
# the ARRAY rather than a reconstructed string is the point — it is the same
# array the engine gets, quoting and all.
if [ "${SERVE_PRINT_ONLY:-0}" = "1" ]; then
  printf '    %q\n' "${_SERVE_CMD[@]}"
  return 0
fi
"${_SERVE_CMD[@]}" > >(tee -a "$SERVE_LOG") 2>&1
}

if [ "${CONFIG_SMOKE:-0}" = "1" ]; then
  refresh_family_runtime_env
  apply_profile_runtime_env
  apply_tuning_overrides
  compute_offload
  compute_vision_args
  warn_capture_window
  # prepare_mtp78 is skipped in smoke mode, but the override draft's location
  # is deterministic: fill it in so the printed argv matches what the real
  # boot executes instead of omitting the speculative "model" key.
  if [ "${MTP78_MODE:-off}" = "override" ] && [ -z "${DRAFT_MODEL:-}" ]; then
    DRAFT_MODEL="$MODEL_DIR/.mtp78-draft"
  fi
  build_spec_args
  echo ">>> vllm argv that would be executed:"
  SERVE_PRINT_ONLY=1 serve_once
  echo "=== CONFIG SMOKE OK — nothing was downloaded, no GPU was touched ==="
  exit 0
fi

# vLLM's reasoning-aware structured-output manager tolerates MTP drafts that
# were sampled before the grammar became active, but XGrammar logs those
# expected rejected probes as engine ERRORs. Preflight only that boundary case;
# committed tokens retain the backend's normal validation/error path. Failure is
# a warning rather than a boot blocker because a future runtime may have fixed
# or refactored the code independently; the serving verifier below still gates
# actual structured-output correctness.
if ! python3 "$SCRIPTS_DIR/patch_structured_output_spec.py"; then
  echo "!!! structured-output/spec-decode compatibility patch was not applied."
  echo "!!! The verifier will test correctness, but inspect vLLM release changes"
  echo "!!! before accepting repeated 'Failed to advance FSM' engine errors."
fi

if [ "${SUPERVISOR:-1}" = "0" ]; then
  refresh_family_runtime_env
  apply_profile_runtime_env
  apply_tuning_overrides
  if ! prepare_checkpoint; then
    echo "!!! checkpoint preparation failed (SUPERVISOR=0: no retry loop here)" >&2
    exit 1
  fi
  compute_offload
  compute_vision_args
  warn_capture_window
  build_spec_args
  status_update starting-engine
  serve_once
  exit $?
fi

# Supervisor v2 (QC-hardened). v1 relied on the foreground child exiting and on
# pattern-based cleanup; a real crash leaves VLLM:: workers alive holding all
# VRAM, so the parent's exit is not a reliable death signal and name-pattern
# kills miss the survivors. v2: run the server in its OWN SESSION (setsid), so
# the whole tree can be killed by process group, and detect death by polling
# both the process and /health.
#
# `|| true` on every kill is load-bearing under `set -e`: when the tree is
# ALREADY gone (the ordinary case after a clean exit, and every case in the
# rollback path) both `kill` and `pkill` exit 1, and an unguarded failure here
# would take the supervisor down with it instead of restarting the engine.
kill_server_tree() {
  if [ -n "${SRV_PID:-}" ]; then
    kill -9 -- "-$SRV_PID" 2>/dev/null || true   # negative PID = whole group
  fi
  pkill -9 -f 'VLLM:[:]' 2>/dev/null || true
  pkill -9 -f 'EngineCor[e]' 2>/dev/null || true
  pkill -9 -f 'vllm serv[e]' 2>/dev/null || true
  for _ in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-0}" -lt 3000 ] 2>/dev/null && break
    sleep 5
  done
}

# ---- correctness verification ------------------------------------------------
# A short-prompt health check is NOT evidence of correctness: earlier
# uncalibrated NVFP4 KV and vision experiments passed /health and a two-line
# completion while failing long retrieval. So every start is followed by a
# probe that includes a long-context
# needle retrieval, and the landing page reports the two results separately.
# VERIFY=0 turns the probe off entirely (the UI then says so, it does not claim
# health); VERIFY_LONG_CONTEXT=0 keeps the short checks only, with the same
# honesty requirement.
VERIFY_PID=""
start_verifier() {
  VERIFY_PID=""
  if [ "${VERIFY:-1}" = "0" ]; then
    echo ">>> Verification disabled (VERIFY=0): the engine will be reported as "
    echo ">>> 'unverified' — no correctness claim is made and nothing rolls back."
    return 0
  fi
  _skip_lc=()
  if [ "${VERIFY_LONG_CONTEXT:-1}" = "0" ]; then
    _skip_lc=(--skip-long-context)
  fi
  read -r -a _names <<< "${SERVED_MODEL_NAME:-model}"
  if [ "${AUTH:-key}" != "none" ]; then
    echo ">>> Verification: one intentional bad-key GET /v1/models will log 401 from 127.0.0.1."
    echo ">>> That 401 proves authentication rejects an invalid credential; it is expected."
    # Actually send that probe (bounded: give up when the engine never gets
    # healthy — the verifier reports that failure itself). A non-401 answer is
    # surfaced as a loud boot note: it means requests are NOT authenticated.
    ( _i=0
      until curl -skf -o /dev/null "$LOCAL_BASE/health" 2>/dev/null; do
        _i=$((_i + 1))
        [ "$_i" -ge 720 ] && exit 0
        sleep 5
      done
      _code="$(curl -sk -o /dev/null -w '%{http_code}' \
        -H 'Authorization: Bearer definitely-wrong' \
        "$LOCAL_BASE/v1/models" 2>/dev/null || true)"
      if [ "$_code" = "401" ]; then
        echo ">>> auth probe: bad key rejected with 401, as required"
      else
        boot_note "!!! auth probe: expected 401 for a bad key, got HTTP ${_code:-none} — the API may be serving UNAUTHENTICATED"
      fi ) &
  fi
  python3 "$SCRIPTS_DIR/verify_serving.py" \
    --base-url "$LOCAL_BASE" --api-key-env VLLM_API_KEY --model "${_names[0]}" \
    --out "$VERIFY_FILE" --pid "$SRV_PID" \
    --max-model-len "${MAX_MODEL_LEN:-524288}" \
    --needle-tokens "${VERIFY_NEEDLE_TOKENS:-32768}" \
    --timeout "${VERIFY_HEALTH_TIMEOUT_S:-3600}" \
    ${_skip_lc[@]+"${_skip_lc[@]}"} >/dev/null 2>&1 &
  VERIFY_PID=$!
  return 0
}

stop_verifier() {
  if [ -n "$VERIFY_PID" ]; then
    kill "$VERIFY_PID" 2>/dev/null || true
  fi
  VERIFY_PID=""
  return 0
}

verdict_ok() {
  python3 -c 'import json,sys
try:
    sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)
except Exception:
    sys.exit(1)' "$1"
}

verdict_reason() {
  python3 -c 'import json,sys
try:
    print(json.load(open(sys.argv[1])).get("reason",""))
except Exception:
    print("verdict unreadable")' "$1"
}

maybe_self_analyze() {
  # The model that IS running explains the config that is not. Runs only once
  # the known-good config is serving, so the analysis never competes with a boot.
  local fdir
  fdir=$(python3 "$SCRIPTS_DIR/config_cli.py" pending-analysis 2>/dev/null) || return 0
  if [ -z "$fdir" ]; then
    return 0
  fi
  if [ "$(soul_level)" -gt 0 ] 2>/dev/null; then
    soul_prepare_permissions
    echo ">>> SOUL: queued rollback evidence from $fdir for incident analysis"
    return 0
  fi
  read -r -a _names <<< "${SERVED_MODEL_NAME:-model}"
  echo ">>> Self-analysis: asking the running model why $fdir failed"
  python3 "$SCRIPTS_DIR/analyze_failure.py" --dir "$fdir" \
    --base-url "$LOCAL_BASE" --api-key-env VLLM_API_KEY --model "${_names[0]}" \
    >/dev/null 2>&1 &
  return 0
}

MAXR="${SUPERVISOR_MAX_RESTARTS:-5}"
LOG_CAP_MB="${SERVE_LOG_MAX_MB:-64}"
# Job control: gives every background job its own process group, so the
# supervisor can reap the whole vLLM tree without spawning a detached bash.
set -m
attempt=0
while :; do
  if [ "$attempt" -gt 0 ]; then
    status_update restarting
    echo "!!! restarting vLLM — attempt $attempt/$MAXR in $((attempt*15))s" >&2
    sleep $((attempt*15))
    kill_server_tree
  fi

  # Clear the request flag BEFORE reading the config, not after: preparing a
  # checkpoint can take minutes, and an apply that lands during that window must
  # leave the flag set so this start is followed by another one. Clearing it
  # afterwards would swallow that second change silently.
  python3 "$SCRIPTS_DIR/config_cli.py" clear-restart || true
  # Re-resolve the config on EVERY start: this is what makes an apply from the
  # landing page take effect without replacing the container.
  apply_config
  # Pick up a certificate that the BACKGROUND acme retry issued after boot: this
  # re-evaluates TLS every restart, so a late cert becomes usable without
  # replacing the container. Idempotent and a no-op once TLS is active or when
  # no cert exists yet. It updates EP_URL/CERT_PATH/LOCAL_BASE in place; the
  # loop's own status_update below then publishes them.
  if [ "${TLS_ENABLED:-0}" != "1" ] && wire_tls_endpoint; then
    boot_note ">>> TLS: certificate is now available; serving with direct TLS after this restart"
  fi
  # Family-specific environment and topology calibration must be refreshed too:
  # a self-service apply can change family, DMA mode, DCP or prefetch policy
  # without replacing PID 1.
  refresh_family_runtime_env
  apply_profile_runtime_env
  apply_tuning_overrides
  if ! prepare_checkpoint; then
    attempt=$((attempt + 1))
    # A checkpoint that fails preparation on a config that DIFFERS from the last
    # known-good one is treated like a failed boot: roll back rather than burn the
    # entire restart budget on a config that can never succeed (e.g. a
    # checkpoint-identity mismatch after a self-service variant switch). One
    # retry first absorbs a transient download blip, which snapshot_download
    # resumes; with no known-good config, should-rollback declines and the
    # ordinary crash-loop budget applies.
    if [ "$attempt" -gt 1 ] && python3 "$SCRIPTS_DIR/config_cli.py" should-rollback; then
      status_update rolling-back
      echo "!!! checkpoint preparation keeps failing — rolling back to the last known-good configuration" >&2
      if python3 "$SCRIPTS_DIR/config_cli.py" rollback --log "$SERVE_LOG" --reason "checkpoint preparation failed"; then
        attempt=0
        continue
      fi
    fi
    if [ "$attempt" -gt "$MAXR" ]; then
      break
    fi
    continue
  fi
  compute_offload
  compute_vision_args
  warn_capture_window
  build_spec_args
  rm -f "$VERIFY_FILE"
  # A failed engine used to be followed immediately by `: > serve-current.log`
  # on the next supervisor attempt. That erased the only traceback before an
  # operator could inspect it (and made a costly rental look like an unexplained
  # restart). Crash restarts are the attempts with a non-zero counter; preserve
  # a bounded set before truncating the live log. Config-request restarts keep
  # attempt=0 and already preserve verified boots in last-good.log.
  if [ "$attempt" -gt 0 ] && [ -s "$SERVE_LOG" ]; then
    _failed_log="$GLM_LOG_DIR/serve-failed-attempt-${attempt}-$(date -u +%Y%m%dT%H%M%SZ).log"
    cp -- "$SERVE_LOG" "$_failed_log" 2>/dev/null || true
    # Keep the five newest failures. Paths are internal, fixed-prefix files;
    # avoid an unbounded log leak across a persistent rental volume.
    mapfile -t _old_failed_logs < <(
      find "$GLM_LOG_DIR" -maxdepth 1 -type f -name 'serve-failed-attempt-*.log' \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | tail -n +6 | cut -d' ' -f2-
    )
    if [ "${#_old_failed_logs[@]}" -gt 0 ]; then
      rm -f -- "${_old_failed_logs[@]}"
    fi
  fi
  : > "$SERVE_LOG"

  status_update starting-engine
  if ! recover_sparkinfer_extension_lock; then
    echo "!!! SparkInfer extension lock preflight did not clear safely; retrying before vLLM launch." >&2
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAXR" ]; then
      break
    fi
    continue
  fi
  # Run in a SUBSHELL of this shell, not `setsid bash -c`. A fresh bash only
  # inherits exported variables, and bash cannot export arrays at all — so the
  # old form started vLLM with an empty MODEL_DIR (`vllm serve ""` ->
  # HFValidationError) and, had that been fixed, would have silently dropped
  # SPEC_ARGS / KVT_ARGS / VISION_ARGS / TLS_ARGS: no speculative decode, no
  # DRAM KV offload, no vision, no TLS, on a server that looks like it booted.
  # `set -m` (job control, enabled above) puts each background job in its own
  # process group, so kill_server_tree's `kill -- -$SRV_PID` still reaps the
  # whole tree — which was the only reason setsid was used.
  serve_once &
  SRV_PID=$!
  start_verifier

  # Death = the session leader is gone. (Health is not a liveness proxy: the
  # engine legitimately takes ~15 min of JIT/cudagraph work before it answers.)
  reason=""
  verified=0   # a verdict has been consumed
  good=0       # ... and it was a PASS (only then is the log safe to truncate)
  while :; do
    if [ -f "$TERMINATE_FLAG" ]; then
      reason="terminating"
      break
    fi
    if [ -f "$RESTART_FLAG" ]; then
      reason="requested"
      break
    fi
    # A certificate the BACKGROUND acme retry issued after the engine came up
    # (it touches .lego/restart-required on success) is otherwise never served:
    # consume the marker and request a restart so the engine comes back up with
    # direct TLS. Only while TLS is not already active.
    if [ "${TLS_ENABLED:-0}" != "1" ] &&
       [ -f "$MODEL_ROOT/.lego/restart-required" ]; then
      rm -f "$MODEL_ROOT/.lego/restart-required" 2>/dev/null || true
      boot_note ">>> TLS: background certificate issuance completed; restarting to serve TLS"
      reason="requested"
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      reason="died"
      break
    fi
    if [ "$verified" = "0" ] && [ -f "$VERIFY_FILE" ]; then
      verified=1
      # VERIFY_FILE lives in the runtime tmpfs and VERIFY_LAST on the volume:
      # a plain cross-device mv is copy+unlink (readers can observe a torn
      # file), and a failed move must not hand verdict_ok the PREVIOUS boot's
      # verdict. Stage next to the destination, replace atomically, and fall
      # back to judging the fresh file in place if persisting it failed.
      _verdict_src="$VERIFY_LAST"
      if cp -f "$VERIFY_FILE" "$VERIFY_LAST.tmp" 2>/dev/null &&
         mv -f "$VERIFY_LAST.tmp" "$VERIFY_LAST" 2>/dev/null; then
        rm -f "$VERIFY_FILE"
      else
        echo "!!! could not persist the verification verdict to $VERIFY_LAST" >&2
        rm -f "$VERIFY_LAST.tmp" 2>/dev/null || true
        _verdict_src="$VERIFY_FILE"
      fi
      if verdict_ok "$_verdict_src"; then
        echo ">>> Verified: $(verdict_reason "$_verdict_src")"
        python3 "$SCRIPTS_DIR/config_cli.py" mark-good --log "$SERVE_LOG" || true
        status_update serving
        attempt=0
        good=1
        maybe_self_analyze
      else
        echo "!!! VERIFICATION FAILED: $(verdict_reason "$_verdict_src")" >&2
        if python3 "$SCRIPTS_DIR/config_cli.py" should-rollback; then
          reason="verify-failed"
          break
        fi
        # Nothing better to fall back to: keep serving, but never call it healthy.
        echo "!!! No known-good configuration to roll back to — the engine keeps"
        echo "!!! serving and the landing page reports it as UNVERIFIED." >&2
        python3 -c "
import sys; sys.path.insert(0, '$SCRIPTS_DIR')
import glm_config as gc
gc.set_apply_state('degraded', detail='verification failed and there is no known-good config')" || true
        status_update serving-unverified
      fi
    fi
    # Once verified, the boot log has already been copied to last-good.log, so a
    # long-lived instance's stats spam can be capped without losing diagnostics.
    if [ "$good" = "1" ] && [ -f "$SERVE_LOG" ]; then
      _sz=$(stat -c %s "$SERVE_LOG" 2>/dev/null || echo 0)
      if [ "$_sz" -gt $((LOG_CAP_MB * 1024 * 1024)) ] 2>/dev/null; then
        : > "$SERVE_LOG"
        echo "--- log truncated at ${LOG_CAP_MB} MB (boot log preserved in logs/last-good.log) ---" >> "$SERVE_LOG"
      fi
    fi
    sleep 5
  done
  stop_verifier

  case "$reason" in
    terminating)
      # The terminate worker owns the box from here. Stop the engine for good —
      # restarting it would rewrite the very logs the worker is erasing and
      # re-allocate the VRAM it wants to zero — then keep PID 1 alive so the
      # worker can finish and issue the destroy call. Exiting here would kill
      # the container mid-erase, which on some providers just restarts it.
      echo ">>> TERMINATION REQUESTED — stopping vLLM and standing down." >&2
      status_update terminating
      stop_soul_supervisor
      kill_server_tree
      : > "$ENGINE_STOPPED"
      while [ -f "$TERMINATE_FLAG" ]; do
        sleep 5
      done
      echo ">>> Termination was cancelled (flag cleared) — resuming." >&2
      rm -f "$ENGINE_STOPPED"
      start_soul_supervisor
      attempt=0
      continue
      ;;
    requested)
      echo ">>> Config change requested — restarting vLLM with the new configuration"
      status_update applying-config
      kill_server_tree
      attempt=0
      continue
      ;;
    verify-failed|died)
      if [ "$reason" = "died" ]; then
        echo "!!! vLLM exited (attempt $attempt)" >&2
      fi
      kill_server_tree
      # Rollback is only meaningful when the running config differs from the last
      # known-good one; otherwise this is an ordinary crash and the restart budget
      # applies.
      if python3 "$SCRIPTS_DIR/config_cli.py" should-rollback; then
        status_update rolling-back
        echo "!!! Rolling back to the last known-good configuration" >&2
        if python3 "$SCRIPTS_DIR/config_cli.py" rollback --log "$SERVE_LOG" --reason "$reason"; then
          attempt=0
          continue
        fi
        echo "!!! Rollback found nothing to restore" >&2
      fi
      attempt=$((attempt + 1))
      if [ "$attempt" -gt "$MAXR" ]; then
        break
      fi
      continue
      ;;
  esac
done
echo "!!! vLLM crash-looped past $MAXR restarts — giving up (destroy or debug the instance)" >&2
status_update failed
exit 1
