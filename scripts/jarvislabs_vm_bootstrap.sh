#!/usr/bin/env bash
# Launch the turnkey appliance on a JarvisLabs full VM.
#
# Run this as the normal VM user, not as root. JarvisLabs VM images provide
# passwordless sudo and Docker; the checkpoint and all reusable caches live in
# /home/turnkey so a container replacement does not download them again.
set -Eeuo pipefail

IMAGE="${TURNKEY_IMAGE:-ghcr.io/malaiwah/glm52-exl3-vast:latest}"
PROFILE="${MODEL_PROFILE:-glm52-exl3}"
CONTAINER_NAME="${TURNKEY_CONTAINER_NAME:-glm52-turnkey}"
WORKSPACE="${TURNKEY_WORKSPACE:-/home/turnkey}"
REGION="${JARVISLABS_REGION:-}"
MACHINE_ID="${JARVISLABS_MACHINE_ID:-}"

if [[ -z "$MACHINE_ID" && "$(hostname)" =~ ^jl-vm-([0-9]+)$ ]]; then
  MACHINE_ID="${BASH_REMATCH[1]}"
fi
if [[ ! "$MACHINE_ID" =~ ^[0-9]+$ ]]; then
  echo "FATAL: set JARVISLABS_MACHINE_ID to the numeric VM id shown by 'jl list'." >&2
  exit 2
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

PASSTHROUGH_KEYS=(
  MODEL_VARIANT MODEL_ID MODEL_DISPLAY_NAME SERVED_MODEL_NAME
  TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  GPU_MEMORY_UTILIZATION KV_CACHE_MEMORY_BYTES GPU_BLOCKS_OVERRIDE
  KV_CACHE_DTYPE KV_SCALE_MODE MTP_DRAFT MTP_TOKENS
  MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD
  DCP DCP_CKV_GATHER_MAX_TOKENS DCP_CKV_PREFETCH_DEPTH
  DCP_KV_CACHE_INTERLEAVE_SIZE DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
  DCP_PREFILL_WORKSPACE_MIB LOAD_FORMAT OFFLOAD_FRACTION
  OFFLOAD_IGNORE_MEMLOCK PREFIX_CACHE_BACKEND PREFIX_CACHE_DISK_GB
  VISION VISION_CHUNKS FEATURE_TEST_LEVEL
  ACME_ATTEMPT_TIMEOUT_S ACME_BACKGROUND_RETRY_S
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

install -d -m 700 "$WORKSPACE" "$WORKSPACE/.secrets"
ENV_FILE="$WORKSPACE/.secrets/appliance.env"
umask 077
ENV_TMP="$(mktemp "$WORKSPACE/.secrets/appliance.env.tmp.XXXXXX")"
trap 'rm -f "$ENV_TMP"' EXIT
{
  printf 'JARVISLABS_MACHINE_ID=%s\n' "$MACHINE_ID"
  printf 'JARVISLABS_REGION=%s\n' "$REGION"
  printf 'PUBLIC_IPADDR=%s\n' "$PUBLIC_IP"
  printf 'MODEL_PROFILE=%s\n' "$PROFILE"
  printf 'MODEL_DISPLAY_NAME=%s\n' "${MODEL_DISPLAY_NAME:-GLM-5.2 JarvisLabs}"
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
  # JarvisLabs credentials are account-scoped, unlike Vast's instance key.
  # Only place one in the appliance when self-termination is explicitly on.
  if [[ "${TERMINATE_ENABLED:-0}" == "1" ]]; then
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

echo ">>> Appliance launched on JarvisLabs VM $MACHINE_ID ($REGION)"
echo ">>> Follow startup: sudo docker logs -f $CONTAINER_NAME"
if [[ -n "${DESEC_DOMAIN:-}" && -n "${DESEC_TOKEN:-}" ]]; then
  echo ">>> The logs will print the trusted dashboard and API URLs after DNS/TLS issuance."
else
  echo ">>> DNS/TLS is not configured; keep credentials off public HTTP."
  echo ">>> Secure fallback: ssh -L 8000:localhost:8000 -L 1111:localhost:1111 ubuntu@$PUBLIC_IP"
  echo ">>> Then open http://localhost:1111/ with the persisted token from the logs."
fi
