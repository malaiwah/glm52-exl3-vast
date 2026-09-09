#!/usr/bin/env bash
# Launch the turnkey appliance on a JarvisLabs full VM.
#
# Run this as the normal VM user, not as root. JarvisLabs VM images provide
# passwordless sudo and Docker; the checkpoint and all reusable caches live in
# /home/turnkey so a container replacement does not download them again.
#
# The whole script executes through main() at the very bottom: this launcher
# is documented as `curl | bash`, and a connection drop mid-transfer must not
# execute a destructive prefix (the `docker rm -f` of the running appliance)
# without the relaunch that follows it.
set -Eeuo pipefail

main() {
  IMAGE="${TURNKEY_IMAGE:-docker.io/malaiwah/glm52-exl3-vast:latest}"
  CONTAINER_NAME="${TURNKEY_CONTAINER_NAME:-glm52-turnkey}"
  WORKSPACE="${TURNKEY_WORKSPACE:-/home/turnkey}"
  REGION="${JARVISLABS_REGION:-}"
  MACHINE_ID="${JARVISLABS_MACHINE_ID:-}"
  ENV_FILE="$WORKSPACE/.secrets/appliance.env"

  # A re-run replaces the container but must not silently drop credentials the
  # previous launch persisted: an image refresh from a fresh shell without
  # re-exported HF_TOKEN/DESEC_TOKEN would rotate the API key and break cert
  # renewal weeks later. Preserve the termination switch and previously stored
  # key even when explicitly disarmed, so re-arming does not lose credentials.
  # An explicitly supplied value (including empty) wins over persisted state.
  CARRIED=()
  carry_forward() {
    local key="$1" prior
    [[ "${!key+x}" == x ]] && return 0
    [[ -f "$ENV_FILE" ]] || return 0
    prior="$(grep -m1 "^${key}=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || true)"
    if [[ -n "$prior" ]]; then
      printf -v "$key" '%s' "$prior"
      CARRIED+=("$key")
    fi
  }
  for key in HF_TOKEN DESEC_TOKEN DESEC_DOMAIN VLLM_API_KEY \
             TERMINATE_ENABLED MODEL_PROFILE MODEL_FAMILY MODEL_VARIANT \
             JARVISLABS_API_KEY JARVISLABS_TERMINATE_API_KEY; do
    carry_forward "$key"
  done
  if [[ -n "${MODEL_PROFILE:-}" ]]; then
    PROFILE="$MODEL_PROFILE"
  elif [[ -n "${MODEL_FAMILY:-}${MODEL_VARIANT:-}" ]]; then
    PROFILE=""
  else
    PROFILE=glm53-3.42bpw-500k
  fi
  if [[ "${#CARRIED[@]}" -gt 0 ]]; then
    echo ">>> Carried forward from the previous launch: ${CARRIED[*]}"
    echo ">>> (export a new value before running to replace one)"
  fi

  if [[ -z "$MACHINE_ID" && "$(hostname)" =~ ^jl-vm-([0-9]+)$ ]]; then
    MACHINE_ID="${BASH_REMATCH[1]}"
  fi
  if [[ ! "$MACHINE_ID" =~ ^[0-9]+$ ]]; then
    echo "FATAL: set JARVISLABS_MACHINE_ID to the numeric VM id shown by 'jl list'." >&2
    exit 2
  fi
  if [[ -z "${JARVISLABS_REGION:-}" ]]; then
    REGION="IN1"
    echo ">>> JARVISLABS_REGION not set; defaulting to IN1 (RTX-PRO6000 4-GPU VMs)."
    echo ">>> Export JARVISLABS_REGION=IN2/EU1 explicitly if your VM is elsewhere."
  fi
  case "${REGION^^}" in
    IN1|IN2|EU1) REGION="${REGION^^}" ;;
    *)
      echo "FATAL: set JARVISLABS_REGION=IN1, IN2, or EU1 to match the VM." >&2
      exit 2
      ;;
  esac
  if ! command -v docker >/dev/null 2>&1; then
    echo "FATAL: Docker is not installed; select JarvisLabs' current Ubuntu VM image." >&2
    exit 2
  fi
  if ! sudo -n true 2>/dev/null; then
    echo "FATAL: this launcher needs JarvisLabs' passwordless sudo." >&2
    exit 2
  fi

  # JarvisLabs VM images auto-start nvidia-dcgm (nv-hostengine), which binds
  # tcp://127.0.0.1:5555 and collides with the appliance's LMCache ZMQ
  # adapter. Mask the service and stop the daemon before the first launch;
  # killing it without masking just makes it restart.
  if systemctl list-unit-files 2>/dev/null | grep -q '^nvidia-dcgm\.service'; then
    sudo systemctl mask nvidia-dcgm.service 2>/dev/null || true
    sudo systemctl mask nv-hostengine.service 2>/dev/null || true
    sudo pkill -x nv-hostengine 2>/dev/null || true
    echo ">>> Masked nvidia-dcgm (port 5555 freed for LMCache ZMQ)."
  fi

  PASSTHROUGH_KEYS=(
    MODEL_FAMILY MODEL_VARIANT MODEL_ID MODEL_DISPLAY_NAME SERVED_MODEL_NAME
    TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
    VLLM_EXL3_PREFILL_CAPACITY
    GPU_MEMORY_UTILIZATION KV_CACHE_MEMORY_BYTES GPU_BLOCKS_OVERRIDE
    KV_CACHE_DTYPE KV_SCALE_MODE MTP_DRAFT MTP_TOKENS
    MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD
    DCP DCP_CKV_GATHER_MAX_TOKENS DCP_CKV_PREFETCH_DEPTH
    DCP_KV_CACHE_INTERLEAVE_SIZE DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
    DCP_PREFILL_WORKSPACE_MIB LOAD_FORMAT OFFLOAD_FRACTION
    OFFLOAD_IGNORE_MEMLOCK PREFIX_CACHE_BACKEND PREFIX_CACHE_DISK_GB
    VISION VISION_CHUNKS FEATURE_TEST_LEVEL
    ACME_ATTEMPT_TIMEOUT_S ACME_BACKGROUND_RETRY_S
    PORT PREFILL_FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE PREFILL_SCHEDULE_INTERVAL
  )

  for key in HF_TOKEN DESEC_TOKEN JARVISLABS_API_KEY \
             JARVISLABS_TERMINATE_API_KEY VLLM_API_KEY \
             "${PASSTHROUGH_KEYS[@]}"; do
    value="${!key:-}"
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
      echo "FATAL: $key contains a newline and cannot be written to a Docker env file." >&2
      exit 2
    fi
  done
  if [[ "${TERMINATE_ENABLED:-0}" == "1" &&
        -z "${JARVISLABS_TERMINATE_API_KEY:-${JARVISLABS_API_KEY:-}}" ]]; then
    echo "FATAL: TERMINATE_ENABLED=1 requires a JarvisLabs API key." >&2
    exit 2
  fi

  PUBLIC_IP="${PUBLIC_IPADDR:-}"
  if [[ -z "$PUBLIC_IP" ]]; then
    PUBLIC_IP="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
  fi
  if [[ ! "$PUBLIC_IP" =~ ^[0-9a-fA-F:.]+$ ]]; then
    echo "FATAL: could not discover the VM public IP; set PUBLIC_IPADDR explicitly." >&2
    exit 2
  fi

  # Stock Ubuntu keeps /home root-owned 0755, so first-run creation of
  # /home/turnkey needs the sudo this script already requires — but the tree
  # must belong to the invoking user so later unprivileged runs still work.
  if ! install -d -m 700 "$WORKSPACE" "$WORKSPACE/.secrets" 2>/dev/null; then
    sudo install -d -m 700 -o "$(id -u)" -g "$(id -g)" \
      "$WORKSPACE" "$WORKSPACE/.secrets"
  fi
  umask 077
  ENV_TMP="$(mktemp "$WORKSPACE/.secrets/appliance.env.tmp.XXXXXX")"
  trap 'rm -f "$ENV_TMP"' EXIT
  {
    printf 'JARVISLABS_MACHINE_ID=%s\n' "$MACHINE_ID"
    printf 'JARVISLABS_REGION=%s\n' "$REGION"
    printf 'PUBLIC_IPADDR=%s\n' "$PUBLIC_IP"
    printf 'MODEL_PROFILE=%s\n' "$PROFILE"
    printf 'MODEL_DISPLAY_NAME=%s\n' "${MODEL_DISPLAY_NAME:-GLM-5.3 JarvisLabs}"
    printf 'LANDING_PAGE=1\n'
    printf 'OPEN_BUTTON_PORT=1111\n'
    printf 'DESEC_DOMAIN=%s\n' "${DESEC_DOMAIN:-}"
    printf 'TERMINATE_ENABLED=%s\n' "${TERMINATE_ENABLED:-0}"
    for key in HF_TOKEN DESEC_TOKEN VLLM_API_KEY; do
      value="${!key:-}"
      [[ -z "$value" ]] || printf '%s=%s\n' "$key" "$value"
    done
    for key in "${PASSTHROUGH_KEYS[@]}"; do
      [[ "$key" == MODEL_DISPLAY_NAME ]] && continue
      value="${!key:-}"
      [[ -z "$value" ]] || printf '%s=%s\n' "$key" "$value"
    done
    # Retain a previously persisted key while disarmed; do not introduce a new
    # account credential into an unarmed appliance.
    if [[ "${TERMINATE_ENABLED:-0}" == "1" ||
          " ${CARRIED[*]} " == *" JARVISLABS_TERMINATE_API_KEY "* ||
          " ${CARRIED[*]} " == *" JARVISLABS_API_KEY "* ]]; then
      if [[ -n "${JARVISLABS_TERMINATE_API_KEY:-}" ]]; then
        printf 'JARVISLABS_TERMINATE_API_KEY=%s\n' \
          "$JARVISLABS_TERMINATE_API_KEY"
      elif [[ -n "${JARVISLABS_API_KEY:-}" ]]; then
        printf 'JARVISLABS_API_KEY=%s\n' "$JARVISLABS_API_KEY"
      fi
    fi
  } >"$ENV_TMP"
  chmod 600 "$ENV_TMP"
  mv -f "$ENV_TMP" "$ENV_FILE"
  trap - EXIT

  echo ">>> Pulling $IMAGE"
  sudo docker pull "$IMAGE"
  if sudo docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo ">>> Replacing container $CONTAINER_NAME; persistent $WORKSPACE is kept"
    sudo docker rm -f "$CONTAINER_NAME" >/dev/null
  fi

  sudo docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    --network host \
    --ulimit memlock=-1:-1 \
    --env-file "$ENV_FILE" \
    -v "$WORKSPACE:/workspace" \
    "$IMAGE"

  wait_and_summarize
}

wait_and_summarize() {
  local port="${PORT:-8000}" waited=0 health="" key="" token=""
  echo ">>> Waiting for the endpoint (image layers + 331 GB weights + model load;"
  echo ">>> typically 25-45 minutes on a fresh VM, ~10 minutes with cached weights)."
  while (( waited < 3600 )); do
    health="$(curl --max-time 5 -s -o /dev/null -w '%{http_code}' \
      "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
    [[ "$health" == "200" ]] && break
    if ! sudo docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" \
        2>/dev/null | grep -q true; then
      sudo docker logs --tail 30 "$CONTAINER_NAME" >&2 || true
      echo "FATAL: the appliance container exited before serving." >&2
      exit 1
    fi
    sleep 30; waited=$(( waited + 30 ))
    (( waited % 300 == 0 )) && \
      echo ">>> still booting... ${waited}s elapsed (last health: ${health:-none})"
  done
  [[ "$health" == "200" ]] || { echo "FATAL: not healthy after 60 minutes; inspect: sudo docker logs $CONTAINER_NAME" >&2; exit 1; }
  key="$(sudo docker exec "$CONTAINER_NAME" sh -c 'cat /workspace/../.vllm-api-key 2>/dev/null || true')"
  [[ -z "$key" && -r "$WORKSPACE/.vllm-api-key" ]] && key="$(cat "$WORKSPACE/.vllm-api-key")"
  [[ -z "$key" && -r "$WORKSPACE/../.vllm-api-key" ]] && key="$(cat "$WORKSPACE/../.vllm-api-key")"
  token="$(sudo docker logs "$CONTAINER_NAME" 2>&1 | grep -oE 'token=[a-zA-Z0-9-]+' | head -1 | cut -d= -f2 || true)"
  local models
  models="$(curl --max-time 10 -s -H "Authorization: Bearer $key" \
    "http://127.0.0.1:${port}/v1/models" | tr ',' '\n' | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ' || true)"
  cat <<SUMMARY

==================================================================
 GLM-5.3 is serving on VM $MACHINE_ID ($REGION)
==================================================================
 Endpoint (inside the VM)  : http://127.0.0.1:${port}/v1
 Endpoint (from your machine): ssh -L 8000:localhost:8000 ubuntu@$PUBLIC_IP
                                then http://localhost:8000/v1
 API key                    : ${key:-(printed once in 'docker logs' at first boot)}
                              send it as "Authorization: Bearer <key>"
 Model name                 : ${models:-GLM-5.3}
 Dashboard                  : ssh -L 1111:localhost:1111 ubuntu@$PUBLIC_IP
                              then http://localhost:1111/?token=${token:-(see docker logs)}
 Logs                       : sudo docker logs -f $CONTAINER_NAME

 Try it:
   curl http://127.0.0.1:${port}/v1/chat/completions \\
     -H "Authorization: Bearer $key" \\
     -H 'Content-Type: application/json' \\
     -d '{"model":"GLM-5.3","messages":[{"role":"user","content":"Say hi"}]}'

 Optional one-time host tuning (P2P override + pcie_aspm=off, needs a
 reboot): see the README JarvisLabs section. The appliance serves safely
 without it (NCCL fallback).

 Billing continues while the VM exists. When done: jl destroy $MACHINE_ID
==================================================================
SUMMARY
}

main "$@"
