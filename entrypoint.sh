#!/bin/bash
# Multi-model vLLM turnkey for Vast.ai and Runpod Pods.
# MODEL_PROFILE selects a complete, internally consistent launch contract:
#   glm52-exl3          validated 4-GPU GLM-5.2 EXL3 production stack
#   qwen36-27b-nvfp4   low-cost 1-GPU Qwen3.6-27B NVFP4 development stack
#   custom              conservative vLLM defaults for another checkpoint
# All logs go to stdout (the provider's Pod/instance logs). The image starts its
# own key-only SSH daemon because Docker ENTRYPOINT launches do not inject one.
set -e

if [ -n "${RUNPOD_POD_ID:-}" ]; then
  PLATFORM=runpod
elif [ -n "${CONTAINER_ID:-}" ]; then
  PLATFORM=vast
else
  PLATFORM=generic
fi
INSTANCE_ID="$(printf '%s' "${CONTAINER_ID:-${RUNPOD_POD_ID:-}}" | tr -cd '[:alnum:]-' | cut -c1-40)"

MODEL_PROFILE="${MODEL_PROFILE:-glm52-exl3}"
case "$MODEL_PROFILE" in
  glm52-exl3)
    MODEL_ID="${MODEL_ID:-brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw}"
    MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-GLM-5.2}"
    PROFILE_MODEL_DIR="/workspace/GLM-5.2-EXL3-TR3-3.0bpw"
    PROFILE_SERVED_NAME="GLM-5.2"
    PROFILE_DOWNLOAD_GIB=309
    PROFILE_REQUIRED_GPUS=4
    PROFILE_GPU_PATTERN='RTX PRO 6000 Blackwell'
    PROFILE_TP_SIZE=4
    PROFILE_MAX_MODEL_LEN=524288
    PROFILE_OUTPUT_LIMIT=131072
    PROFILE_MAX_NUM_SEQS=8
    PROFILE_MAX_BATCHED_TOKENS=3072
    PROFILE_GPU_MEMORY_UTILIZATION=0.93
    PROFILE_GPU_BLOCKS=2048
    PROFILE_OFFLOAD_FRACTION=0.70
    PROFILE_MTP_TOKENS=3
    PROFILE_MULTIMODAL=1
    PROFILE_FEATURES="512K context · EXL3/fp8 KV · MTP-3 · TP4/DCP4"
    ;;
  qwen36-27b-nvfp4)
    MODEL_ID="${MODEL_ID:-nvidia/Qwen3.6-27B-NVFP4}"
    MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-Qwen3.6-27B}"
    PROFILE_MODEL_DIR="/workspace/Qwen3.6-27B-NVFP4"
    PROFILE_SERVED_NAME="Qwen3.6-27B"
    PROFILE_DOWNLOAD_GIB=21
    PROFILE_REQUIRED_GPUS=1
    PROFILE_GPU_PATTERN='RTX PRO 6000 Blackwell|GeForce RTX 5090'
    PROFILE_TP_SIZE=1
    PROFILE_MAX_MODEL_LEN=32768
    PROFILE_OUTPUT_LIMIT=8192
    PROFILE_MAX_NUM_SEQS=4
    PROFILE_MAX_BATCHED_TOKENS=4096
    PROFILE_GPU_MEMORY_UTILIZATION=0.90
    PROFILE_GPU_BLOCKS=0
    PROFILE_OFFLOAD_FRACTION=0
    PROFILE_MTP_TOKENS=0
    PROFILE_MULTIMODAL=0
    PROFILE_FEATURES="32K dev context · NVFP4 · text-only · 1× Blackwell GPU"
    ;;
  custom)
    [ -n "${MODEL_ID:-}" ] || {
      echo "FATAL: MODEL_PROFILE=custom requires MODEL_ID=<Hugging Face repo>"
      exit 1
    }
    MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-${MODEL_ID##*/}}"
    _MODEL_DIR_NAME="$(printf '%s' "$MODEL_ID" | tr '/:' '--' | tr -cd '[:alnum:]._-')"
    PROFILE_MODEL_DIR="/workspace/models/${_MODEL_DIR_NAME}"
    PROFILE_SERVED_NAME="$MODEL_DISPLAY_NAME"
    PROFILE_DOWNLOAD_GIB="${MODEL_DOWNLOAD_GIB:-0}"
    PROFILE_REQUIRED_GPUS="${REQUIRED_GPUS:-1}"
    PROFILE_GPU_PATTERN="${GPU_NAME_PATTERN:-}"
    PROFILE_TP_SIZE="${TENSOR_PARALLEL_SIZE:-$PROFILE_REQUIRED_GPUS}"
    PROFILE_MAX_MODEL_LEN=32768
    PROFILE_OUTPUT_LIMIT=8192
    PROFILE_MAX_NUM_SEQS=4
    PROFILE_MAX_BATCHED_TOKENS=4096
    PROFILE_GPU_MEMORY_UTILIZATION=0.90
    PROFILE_GPU_BLOCKS=0
    PROFILE_OFFLOAD_FRACTION=0
    PROFILE_MTP_TOKENS=0
    PROFILE_MULTIMODAL=1
    PROFILE_FEATURES="custom checkpoint · conservative vLLM defaults"
    ;;
  *)
    echo "FATAL: unknown MODEL_PROFILE=$MODEL_PROFILE"
    echo "FATAL: choose glm52-exl3, qwen36-27b-nvfp4, or custom"
    exit 1
    ;;
esac

MODEL_DIR="${MODEL_DIR:-$PROFILE_MODEL_DIR}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$PROFILE_SERVED_NAME}"
MODEL_DOWNLOAD_GIB="${MODEL_DOWNLOAD_GIB:-$PROFILE_DOWNLOAD_GIB}"
REQUIRED_GPUS="${REQUIRED_GPUS:-$PROFILE_REQUIRED_GPUS}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-$PROFILE_GPU_PATTERN}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$PROFILE_TP_SIZE}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$PROFILE_MAX_MODEL_LEN}"
MODEL_OUTPUT_LIMIT="${MODEL_OUTPUT_LIMIT:-$PROFILE_OUTPUT_LIMIT}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$PROFILE_MAX_NUM_SEQS}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$PROFILE_MAX_BATCHED_TOKENS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$PROFILE_GPU_MEMORY_UTILIZATION}"
GPU_BLOCKS_OVERRIDE="${GPU_BLOCKS_OVERRIDE:-$PROFILE_GPU_BLOCKS}"
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-$PROFILE_OFFLOAD_FRACTION}"
MTP_TOKENS="${MTP_TOKENS:-$PROFILE_MTP_TOKENS}"
MULTIMODAL="${MULTIMODAL:-$PROFILE_MULTIMODAL}"
LANDING_FEATURES="${LANDING_FEATURES:-$PROFILE_FEATURES}"

echo "=== Model turnkey: $MODEL_DISPLAY_NAME ==="
echo ">>> Profile: $MODEL_PROFILE; checkpoint: $MODEL_ID"
echo ">>> Platform: $PLATFORM${INSTANCE_ID:+ ($INSTANCE_ID)}"
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" = "1" ]; then
  [ -n "${PUBLIC_ENDPOINT:-}" ] || {
    echo "FATAL: RUNPOD_DIRECT_TLS=1 requires PUBLIC_ENDPOINT=https://domain:mapped-port"
    exit 1
  }
  if ! { [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ]; } &&
     ! { [ -n "${DESEC_TOKEN:-}" ] && [ -n "${DESEC_DOMAIN:-}" ]; }; then
    echo "FATAL: RUNPOD_DIRECT_TLS=1 requires ACME_DOMAIN + ACME_DNS_PROVIDER,"
    echo "FATAL: or DESEC_TOKEN + DESEC_DOMAIN."
    exit 1
  fi
fi
if [ "${PROFILE_VALIDATE_ONLY:-0}" = "1" ]; then
  printf 'MODEL_PROFILE=%s\nMODEL_ID=%s\nMODEL_DIR=%s\nSERVED_MODEL_NAME=%s\n' \
    "$MODEL_PROFILE" "$MODEL_ID" "$MODEL_DIR" "$SERVED_MODEL_NAME"
  printf 'REQUIRED_GPUS=%s\nTENSOR_PARALLEL_SIZE=%s\nMAX_MODEL_LEN=%s\n' \
    "$REQUIRED_GPUS" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN"
  printf 'GPU_BLOCKS_OVERRIDE=%s\nOFFLOAD_FRACTION=%s\nMTP_TOKENS=%s\nMULTIMODAL=%s\n' \
    "$GPU_BLOCKS_OVERRIDE" "$OFFLOAD_FRACTION" "$MTP_TOKENS" "$MULTIMODAL"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head "-$REQUIRED_GPUS" || true
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge "$REQUIRED_GPUS" ] || {
  echo "FATAL: profile $MODEL_PROFILE needs $REQUIRED_GPUS visible GPU(s), found $NGPU"
  exit 1
}
SUPPORTED_GPU_COUNT="$REQUIRED_GPUS"
if [ -n "$GPU_NAME_PATTERN" ]; then
  SUPPORTED_GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
    head "-$REQUIRED_GPUS" | grep -Ec "$GPU_NAME_PATTERN" || true)"
fi
if [ "$SUPPORTED_GPU_COUNT" -lt "$REQUIRED_GPUS" ] && [ "${ALLOW_UNSUPPORTED_GPU:-0}" != "1" ]; then
  echo "FATAL: profile $MODEL_PROFILE expects $REQUIRED_GPUS GPU(s) matching: $GPU_NAME_PATTERN"
  echo "FATAL: choose compatible hardware, or set ALLOW_UNSUPPORTED_GPU=1 for an experimental port."
  exit 1
fi
BOOT_MEMLOCK="$(ulimit -l)"
BOOT_NOFILE="$(ulimit -n)"
BOOT_SHM="$(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')"
echo ">>> Runtime limits: memlock=$BOOT_MEMLOCK, nofile=$BOOT_NOFILE, /dev/shm=${BOOT_SHM:-unknown}"

# Vast exposes the account key as SSH_PUBLIC_KEY; Runpod exposes a Pod SSH key
# as PUBLIC_KEY. Neither Docker ENTRYPOINT path installs the key or starts SSH
# for this image. Install either key, generate per-container host keys on first
# boot, and start our key-only daemon. Also preserve a pre-injected
# authorized_keys file for compatibility with older hosts. Loose permissions
# make sshd refuse the key with
#   "Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys"
# and every `ssh root@...` fails with a bare "Permission denied (publickey)".
install -d -m 700 /root/.ssh
PLATFORM_SSH_KEY="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [ -n "$PLATFORM_SSH_KEY" ]; then
  ( umask 077; printf '%s\n' "$PLATFORM_SSH_KEY" > /root/.ssh/authorized_keys )
fi
chown -R root:root /root/.ssh 2>/dev/null || true
chmod 700 /root /root/.ssh 2>/dev/null || true
if [ -f /root/.ssh/authorized_keys ]; then
  chmod 600 /root/.ssh/authorized_keys
fi
if command -v ssh-keygen >/dev/null && [ -x /usr/sbin/sshd ]; then
  install -d -m 755 /run/sshd
  ssh-keygen -A >/dev/null
  if /usr/sbin/sshd; then
    if [ -s /root/.ssh/authorized_keys ]; then
      echo ">>> SSH: key-only daemon listening on :22"
    else
      echo "!!! SSH daemon is running, but SSH_PUBLIC_KEY/PUBLIC_KEY/authorized_keys is empty"
    fi
  else
    echo "!!! SSH daemon failed to start; use the instance console to diagnose it"
  fi
else
  echo "!!! SSH server is missing from this image build"
fi

# Boot-status snapshot for the landing page; rewritten at each milestone.
# Holds the API key once generated -> keep it root-only.
STATUS_FILE="${STATUS_FILE:-/tmp/model-turnkey-boot-status.json}"
EP_URL="${PUBLIC_ENDPOINT%/}" TLS_STATE="not configured" HTTPS_HOSTPORT="" CERT_PATH="" KEY_PATH="" OFFLOAD_STATE="off"
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" != "1" ]; then
  [ -n "$EP_URL" ] || EP_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"
  TLS_STATE="https:// managed by Runpod proxy"
fi
status_update() {
  printf '{"phase":"%s","endpoint":"%s","tls":"%s","https_hostport":"%s","cert":"%s","keyfile":"%s","offload":"%s","api_key":"%s"}\n' \
    "$1" "$EP_URL" "$TLS_STATE" "$HTTPS_HOSTPORT" "$CERT_PATH" "$KEY_PATH" "$OFFLOAD_STATE" "${VLLM_API_KEY:-}" > "$STATUS_FILE"
  chmod 600 "$STATUS_FILE" 2>/dev/null || true
}

# Landing page (:1111, dual-protocol TLS+plain, or HTTPS behind Runpod proxy).
# Started before the weight download so status is visible from minute one.
# Vast needs OPEN_BUTTON_PORT=1111 + '-p 1111:1111'; Runpod needs 1111/http.
if [ "${LANDING_PAGE:-1}" != "0" ] && [ -f /opt/landing.py ]; then
  if [ "$PLATFORM" = "runpod" ]; then
    LANDING_TRUST_PROXY_HTTPS=1
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
    export LANDING_TRUST_PROXY_HTTPS OPEN_BUTTON_TOKEN
  fi
  status_update booting
  MODEL_DIR="$MODEL_DIR" STATUS_FILE="$STATUS_FILE" \
    MODEL_DISPLAY_NAME="$MODEL_DISPLAY_NAME" \
    LANDING_MODEL_ID="${SERVED_MODEL_NAME%% *}" \
    MODEL_PROFILE="$MODEL_PROFILE" \
    MODEL_CONTEXT_WINDOW="$MAX_MODEL_LEN" \
    MODEL_OUTPUT_LIMIT="$MODEL_OUTPUT_LIMIT" \
    MODEL_DOWNLOAD_GIB="$MODEL_DOWNLOAD_GIB" \
    LANDING_FEATURES="$LANDING_FEATURES" \
    LANDING_TRUST_PROXY_HTTPS="${LANDING_TRUST_PROXY_HTTPS:-0}" \
    OPEN_BUTTON_TOKEN="${OPEN_BUTTON_TOKEN:-}" python3 /opt/landing.py &
  echo ">>> Landing page (Open button) live on :1111"
  if [ "$PLATFORM" = "runpod" ]; then
    echo ">>> Runpod dashboard: https://${RUNPOD_POD_ID}-1111.proxy.runpod.net/?token=${OPEN_BUTTON_TOKEN}"
  fi
fi

# The marker records the checkpoint ID so two profiles cannot accidentally mix
# shards in the same directory. Legacy GLM markers were empty and remain valid
# for the default GLM profile.
DOWNLOAD_MARKER="$MODEL_DIR/.download-complete"
DOWNLOAD_INTENT="$MODEL_DIR/.download-model"
DOWNLOADED_MODEL=""
[ -f "$DOWNLOAD_MARKER" ] && DOWNLOADED_MODEL="$(cat "$DOWNLOAD_MARKER")"
INTENDED_MODEL=""
[ -f "$DOWNLOAD_INTENT" ] && INTENDED_MODEL="$(cat "$DOWNLOAD_INTENT")"
if [ -n "$INTENDED_MODEL" ] && [ "$INTENDED_MODEL" != "$MODEL_ID" ]; then
  echo "FATAL: $MODEL_DIR has a partial/complete download for $INTENDED_MODEL"
  echo "FATAL: choose a different MODEL_DIR; checkpoints must not share a directory"
  exit 1
fi
if [ -f "$DOWNLOAD_MARKER" ] &&
   { [ "$DOWNLOADED_MODEL" = "$MODEL_ID" ] ||
     { [ -z "$DOWNLOADED_MODEL" ] && [ "$MODEL_PROFILE" = "glm52-exl3" ]; }; }; then
  :
else
  if [ -n "$DOWNLOADED_MODEL" ] && [ "$DOWNLOADED_MODEL" != "$MODEL_ID" ]; then
    echo "FATAL: $MODEL_DIR contains checkpoint $DOWNLOADED_MODEL, not $MODEL_ID"
    echo "FATAL: choose a different MODEL_DIR; checkpoints must not share a directory"
    exit 1
  fi
  status_update downloading-weights
  if [ "$MODEL_DOWNLOAD_GIB" != "0" ]; then
    echo ">>> Downloading $MODEL_ID (~${MODEL_DOWNLOAD_GIB} GiB) to $MODEL_DIR"
  else
    echo ">>> Downloading $MODEL_ID to $MODEL_DIR"
  fi
  [ -n "${HF_TOKEN:-}" ] && echo ">>> (HF_TOKEN detected: authenticated download)" || echo ">>> (set HF_TOKEN env for higher rate limits)"
  mkdir -p "$MODEL_DIR"
  printf '%s\n' "$MODEL_ID" > "$DOWNLOAD_INTENT"
  MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" HF_XET_HIGH_PERFORMANCE=1 python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ['MODEL_ID'], local_dir=os.environ['MODEL_DIR'],
                  max_workers=int(os.environ.get('MODEL_DOWNLOAD_WORKERS', '16')))"
  printf '%s\n' "$MODEL_ID" > "$DOWNLOAD_MARKER"
  echo ">>> Weights ready."
fi

if [ "$MODEL_PROFILE" = "qwen36-27b-nvfp4" ]; then
  MODEL_DIR="$MODEL_DIR" MTP_TOKENS="$MTP_TOKENS" python3 - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
config = json.loads((model_dir / "config.json").read_text())
architectures = config.get("architectures") or []
if "Qwen3_5ForConditionalGeneration" not in architectures:
    raise SystemExit(
        "FATAL: qwen36-27b-nvfp4 requires Qwen3_5ForConditionalGeneration; "
        f"checkpoint declares {architectures!r}"
    )
quant = config.get("quantization_config") or {}
method = str(quant.get("quant_method", "")).lower()
algorithm = str(quant.get("quant_algo", "")).upper()
if method != "modelopt" or algorithm not in {
    "MIXED_PRECISION",
    "NVFP4",
    "W4A16_NVFP4",
}:
    raise SystemExit(
        "FATAL: qwen36-27b-nvfp4 requires a ModelOpt NVFP4-compatible "
        f"checkpoint; found quant_method={method!r}, quant_algo={algorithm!r}"
    )
if int(os.environ["MTP_TOKENS"]) > 0:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(
            "FATAL: Qwen MTP was enabled but model.safetensors.index.json is missing"
        )
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    if not any(name.startswith("mtp.") for name in weight_map):
        raise SystemExit(
            "FATAL: Qwen MTP was enabled but the checkpoint has no mtp.* weights"
        )
print(
    f">>> Qwen checkpoint verified: {architectures[0]}, "
    f"ModelOpt {algorithm}"
)
PY
fi

if [ "$MODEL_PROFILE" = "glm52-exl3" ]; then
# MTP78 trellis draft (default ON) — the MTP draft layer quantized to 3.0bpw EXL3
# (all-256-expert, full-corpus calibrated), validated at BF16 acceptance PARITY
# while freeing ~3.8 GB/GPU for KV cache.
#
# MTP78_MODE=graft (default) does in-place surgery on layer 78 of the target.
# It is the default because it is the ONLY mode with long-context correctness
# evidence behind it: rental needle 6/6 on fp8, and benchmarks/vast-45582113-matrix
# armC 3/3 at 150K/190K/250K. Both ran VLLM_EXL3_TRELLIS_MIN_M=4.
#
#   MTP78_MODE=override  SEPARATE draft dir via --speculative-config. Leaves the
#     target byte-identical and measured slightly better on acceptance (MAL 3.548 /
#     84.9% vs 3.517 / 83.9%), but see the warning below -- it does not currently
#     boot on this image with a correct trellis window.
#   MTP78_MODE=off       stock BF16 draft
# MTP78_TRELLIS=0 is still honoured as a synonym for off.
#
# WARNING, measured 2026-07-26 on AIBeast. In override mode the rank-sliced draft is
# CUDA-graph captured at m=3 and dies with
#   "EXL3 eager parity path entered during CUDA graph capture (m=3)"
# m=3 is invariant: it does NOT follow num_speculative_tokens (MTP_TOKENS=4 still
# reported m=3) and does NOT follow cudagraph_capture_sizes (a list starting at 8
# still reported m=3). Do NOT make that crash go away by lowering
# VLLM_EXL3_TRELLIS_MIN_M to 1. That was tried and it converts a loud boot failure
# into SILENT CORRUPTION -- m=1..3 move off the eager PARITY path onto a trellis
# kernel nothing has verified there. Result: 32K needle 0/2 and 370K needle 0/5,
# output pure garbage, while a short-prompt quality gate still passed 6/6.
# Until the draft's capture size is controllable, graft is the mode to use.
MTP78_MODE="${MTP78_MODE:-graft}"
[ "${MTP78_TRELLIS:-1}" = "0" ] && MTP78_MODE=off
MTP78_DRAFT_DIR="$MODEL_DIR/.mtp78-draft"
# DRAFT_MODEL may be supplied from the environment to point --speculative-config at
# an EXTERNAL draft checkpoint, independently of MTP78_MODE. This is how a non-EXL3
# draft (e.g. the NVFP4 MTP draft from lukealonso/GLM-5.2-NVFP4, model-mtp.safetensors,
# 5.6 GiB) is used: it recovers most of the KV that the BF16 fallback draft gives up,
# while still avoiding _apply_rank_sliced -- so it does not hit the v20 speculator
# capture bug that makes an EXL3 rank-sliced draft unbootable (see the warning above).
DRAFT_MODEL="${DRAFT_MODEL:-}"
if [ -n "$DRAFT_MODEL" ]; then
  echo ">>> Draft: external checkpoint $DRAFT_MODEL (MTP78_MODE=$MTP78_MODE ignored for draft selection)"
  MTP78_MODE=off
fi

fetch_mtp78_overlay() {
  [ -d "$MODEL_DIR/.mtp78-overlay/3bpw-keep0" ] && return 0
  echo ">>> MTP78: downloading 3bpw-keep0 trellis overlay (~3.7 GB)"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('malaiwah/GLM-5.2-EXL3-TR3-MTP78', allow_patterns=['3bpw-keep0/*'], local_dir='$MODEL_DIR/.mtp78-overlay', max_workers=8)"
}

if [ "$MTP78_MODE" = "override" ]; then
  status_update preparing-mtp78
  if python3 /opt/scripts/patch_deepseek_mtp.py; then
    if fetch_mtp78_overlay && python3 /opt/scripts/build_mtp78_draft.py \
         "$MODEL_DIR" "$MODEL_DIR/.mtp78-overlay/3bpw-keep0" "$MTP78_DRAFT_DIR"; then
      DRAFT_MODEL="$MTP78_DRAFT_DIR"
      echo ">>> MTP78: trellis draft via --speculative-config override (target untouched)"
    else
      echo "!!! MTP78: draft build failed — falling back to the in-checkpoint BF16 draft"
    fi
  else
    echo "!!! MTP78: vLLM patch anchor missing in this image — keeping BF16 draft"
  fi
elif [ "$MTP78_MODE" = "graft" ]; then
  if [ ! -f "$MODEL_DIR/.mtp78-grafted" ]; then
    status_update grafting-mtp78
    fetch_mtp78_overlay
    if python3 /opt/scripts/patch_deepseek_mtp.py; then
      if python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" "$MODEL_DIR/.mtp78-overlay/3bpw-keep0"; then
        echo ">>> MTP78: trellis draft active (grafted in place)"
      else
        echo "!!! MTP78 graft failed — reverting to BF16 draft"
        python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" --revert || true
      fi
    else
      echo "!!! MTP78: vLLM patch anchor missing in this image — keeping BF16 draft"
    fi
  else
    python3 /opt/scripts/patch_deepseek_mtp.py || true  # image may have been re-pulled
    echo ">>> MTP78: trellis draft already grafted"
  fi
else
  echo ">>> MTP78 disabled — stock BF16 draft"
fi
# A checkpoint grafted by an earlier boot must be reverted before override mode
# can work, otherwise the target still carries the trellis layer 78.
if [ "$MTP78_MODE" != "graft" ] && [ -f "$MODEL_DIR/.mtp78-grafted" ]; then
  echo ">>> Reverting a previous in-place graft (MTP78_MODE=$MTP78_MODE)"
  python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" --revert || true
fi
else
  MTP78_MODE=off
  DRAFT_MODEL=""
  echo ">>> GLM-specific MTP78 graft: skipped for profile $MODEL_PROFILE"
fi

# Vision (default ON): MoonViT-3d tower (Kimi-K2.6, frozen) + Baseten's trained
# 49.5M-param PatchMerger projector, bolted onto our EXL3 text backbone. Both
# vision parts are BF16 and checkpoint-agnostic — only ~1 GB, no text weight is
# touched. VISION=0 serves pure text.
# The EXL3 loader is multimodal-aware (it collapses `language_model.` prefixes),
# so the rank-sliced text weights still load under the Glm5v wrapper.
VISION="${VISION:-$PROFILE_MULTIMODAL}"
if [ "$MODEL_PROFILE" != "glm52-exl3" ]; then
  VISION=0
fi
VISION_REPO="${VISION_REPO:-chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid}"
if [ "$VISION" = "1" ]; then
  # Gate on what config.json ACTUALLY says, not just the marker file. An MTP78 graft
  # REVERT rewrites config.json back to its pre-vision form, which silently strips the
  # Glm5v wrapper while leaving .vision-enabled in place -- the installer then reported
  # "already installed", skipped re-wrapping, and the server answered image requests
  # with "GLM-5.2 is not a multimodal model" (400). Measured on AIBeast 2026-07-26.
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
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$VISION_REPO', local_dir='$MODEL_DIR/.vision',
  allow_patterns=['vision_tower.safetensors','mm_projector.safetensors','config.json',
                  'configuration_glm5v.py','kimi_k25_processor.py','kimi_k25_vision_processing.py',
                  'media_utils.py','preprocessor_config.json','chat_template.jinja',
                  'plugins/**'], max_workers=8)" || vision_ok=0
    if [ "$vision_ok" = "1" ]; then
      cp "$MODEL_DIR/.vision"/vision_tower.safetensors "$MODEL_DIR/.vision"/mm_projector.safetensors "$MODEL_DIR/" || vision_ok=0
      for f in configuration_glm5v.py kimi_k25_processor.py kimi_k25_vision_processing.py media_utils.py preprocessor_config.json; do
        [ -f "$MODEL_DIR/.vision/$f" ] && cp "$MODEL_DIR/.vision/$f" "$MODEL_DIR/"
      done
      # The vision chat template is MANDATORY: the text-only template never emits
      # <|begin_of_image|><|image|><|end_of_image|>, so the multimodal processor
      # fails with "Failed to apply prompt replacement for mm_items['vision_chunk']".
      # Keep the text-only one for VISION=0 restore.
      if [ -f "$MODEL_DIR/.vision/chat_template.jinja" ]; then
        [ -f "$MODEL_DIR/chat_template.jinja" ] && [ ! -f "$MODEL_DIR/chat_template.jinja.text-only" ] \
          && cp "$MODEL_DIR/chat_template.jinja" "$MODEL_DIR/chat_template.jinja.text-only"
        cp "$MODEL_DIR/.vision/chat_template.jinja" "$MODEL_DIR/chat_template.jinja"
      else
        vision_ok=0
      fi
      pip install -q "$MODEL_DIR/.vision/plugins/glm5v_nf3" || vision_ok=0
    fi
    if [ "$vision_ok" = "1" ] && python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" "$MODEL_DIR/.vision" && python3 /opt/scripts/index_add_vision.py "$MODEL_DIR"; then
      echo ">>> Vision: ENABLED (image input active; VISION=0 to disable)"
    else
      echo "!!! Vision install failed — falling back to text-only"
      python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" --revert || true
      python3 /opt/scripts/index_add_vision.py "$MODEL_DIR" --revert || true
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
      pip install -q "$MODEL_DIR/.vision/plugins/glm5v_nf3" 2>/dev/null || true
      echo ">>> Vision: already installed (plugin re-registered in this container)"
    else
      echo "!!! Vision: marker present but $MODEL_DIR/.vision/plugins/glm5v_nf3 is missing."
      echo "!!! The Glm5v arch will not register. Remove $MODEL_DIR/.vision-enabled to reinstall,"
      echo "!!! or set VISION=0 to serve text-only."
    fi
  fi
elif [ -f "$MODEL_DIR/.vision-enabled" ]; then
  echo ">>> VISION=0: reverting to text-only config"
  python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" --revert
  python3 /opt/scripts/index_add_vision.py "$MODEL_DIR" --revert || true
  [ -f "$MODEL_DIR/chat_template.jinja.text-only" ] && mv "$MODEL_DIR/chat_template.jinja.text-only" "$MODEL_DIR/chat_template.jinja"
fi

# DRAM KV offload: OFFLOAD_FRACTION of the instance's RAM allocation. The GLM
# profile defaults to 0.70; Qwen/custom default to 0. Sized from
# min(cgroup limit, MemTotal) —
# inside a container /proc/meminfo shows the whole host's RAM, but a partial
# rental (e.g. 4 of 8 GPUs) only gets a slice of it.
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-$PROFILE_OFFLOAD_FRACTION}"
KVT_ARGS=()
if [ "$OFFLOAD_FRACTION" != "0" ]; then
  MEM_BYTES=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') * 1024 ))
  CG_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
  if [ "$CG_LIMIT" != "max" ] && [ "$CG_LIMIT" -lt "$MEM_BYTES" ] 2>/dev/null; then
    MEM_BYTES=$CG_LIMIT
  fi
  OFF_BYTES=$(python3 -c "print(int($MEM_BYTES*$OFFLOAD_FRACTION))")
  MEMLOCK_KB=$(ulimit -l)
  if [ "$MEMLOCK_KB" != "unlimited" ] && [ "$MEMLOCK_KB" -lt "$((OFF_BYTES / 1024))" ] 2>/dev/null; then
    echo "!!! WARNING: memlock ulimit (${MEMLOCK_KB} KB) is below the $((OFF_BYTES/1073741824)) GiB KV pool to pin."
    if [ "${OFFLOAD_IGNORE_MEMLOCK:-1}" = "1" ]; then
      # Measured 2026-07-26 on an owned box: a 128 GiB offload tier runs fine with
      # memlock capped at ~31 GiB (kv_offload_total_bytes climbs normally), because
      # the connector does not mlock the whole tier up front. Vast's Docker Options
      # field accepts only ports, environment variables and hostname, so a --ulimit
      # setting there is ignored; gating on that unavailable control would disable a
      # working feature outright.
      echo "!!! Proceeding anyway (warn-and-proceed is the default; OFFLOAD_IGNORE_MEMLOCK=0 to"
      echo "!!! disable offload instead). This degrades rather than fails: the connector does not"
      echo "!!! mlock the tier up front (measured: a 125 GiB tier offloading normally under a"
      echo "!!! 31 GiB memlock), and kv_load_failure_policy=recompute means any KV that cannot be"
      echo "!!! brought back is simply recomputed. Verify kv_offload_* metrics climb."
    else
      echo "!!! Vast does not accept --ulimit in template Docker Options. Set"
      echo "!!! OFFLOAD_IGNORE_MEMLOCK=1 only after verifying kv_offload_* metrics at this limit."
      echo "!!! Continuing WITHOUT DRAM offload."
      OFFLOAD_FRACTION=0
    fi
  fi
fi
if [ "$OFFLOAD_FRACTION" != "0" ]; then
  OFFLOAD_STATE="$((OFF_BYTES/1073741824)) GiB pinned DRAM"
  echo ">>> DRAM KV offload: $((OFF_BYTES/1073741824)) GiB (${OFFLOAD_FRACTION} of instance RAM allocation)"
  # kv_load_failure_policy=recompute: a KV block that cannot be fetched back from
  # the DRAM tier is recomputed instead of failing the request. The vLLM default
  # is "fail", which would turn any offload hiccup into a terminal error — not an
  # acceptable trade for a cache tier that is a pure optimisation.
  KVT_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF_BYTES}}")
  # OffloadingConnector rejects expandable_segments (VMM can remap pinned KV pages)
  export PYTORCH_CUDA_ALLOC_CONF=""
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# GLM vision serve args: Qwen and custom profiles use their checkpoint's native
# multimodal implementation instead of this GLM-specific graft marker.
VISION_ARGS=()
if [ "$MODEL_PROFILE" = "glm52-exl3" ] &&
   [ "${VISION:-1}" = "1" ] && [ -f "$MODEL_DIR/.vision-enabled" ]; then
  VISION_ARGS=(--limit-mm-per-prompt "{\"vision_chunk\":${VISION_CHUNKS:-8}}" --trust-remote-code)
fi

export CUDA_DEVICE_MAX_CONNECTIONS=32 CUTE_DSL_ARCH=sm_120a OMP_NUM_THREADS=16
export ONLINE_QUANT=none
export SAFETENSORS_FAST_GPU=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple
export TORCH_CUDA_ARCH_LIST=12.0a FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=2400
if [ "$MODEL_PROFILE" = "glm52-exl3" ]; then
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
else
  # The base image carries GLM defaults in its OCI environment. They must not
  # leak into Qwen/custom profiles, especially the GLM MLA scale file, disabled
  # ModelOpt kernel, and custom multi-GPU NCCL preload.
  unset VLLM_EXL3_EXT_PATH VLLM_EXL3_ABI_SHIM VLLM_NVFP4_MLA_SCALES_FILE
  unset VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND
  unset VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS
  unset VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER
  unset VLLM_DISABLE_SHARED_EXPERTS_STREAM VLLM_DISABLED_KERNELS
  unset NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH VLLM_NCCL_SO_PATH LD_PRELOAD
fi
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

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

# TLS via Let's Encrypt DNS-01 (optional): set ACME_DOMAIN + ACME_DNS_PROVIDER
# (lego provider name, e.g. cloudflare, duckdns) + the provider's cred envs
# (e.g. CLOUDFLARE_DNS_API_TOKEN, or DUCKDNS_TOKEN). Certs persist on the
# volume and are reused while valid >7 days, else re-issued at boot.
# Turnkey auto-DNS (deSEC): set DESEC_TOKEN + DESEC_DOMAIN (your *.dedyn.io zone).
# A per-instance name — stable across reboots (keyed to the provider instance
# ID) so DNS
# records don't pile up in the zone and the LE cert can be reused — is
# registered at startup and pointed at this instance.
current_public_ip() {
  # Provider IP variables can be startup snapshots. Prefer a live external
  # observation each boot and retain them as a no-egress fallback.
  local live_ip=""
  live_ip="$(curl -fsS -m 10 https://api.ipify.org 2>/dev/null || true)"
  printf '%s' "${live_ip:-${PUBLIC_IPADDR:-${RUNPOD_PUBLIC_IP:-}}}"
}
PUBLIC_IP_CURRENT="$(current_public_ip)"

# Runpod's HTTP proxy terminates TLS and forwards plain HTTP to the container.
# App-level TLS is only appropriate when the API is separately exposed as a
# direct TCP port.
ACME_DIRECT=1
if [ "$PLATFORM" = "runpod" ] && [ "${RUNPOD_DIRECT_TLS:-0}" != "1" ]; then
  ACME_DIRECT=0
  if [ -n "${ACME_DOMAIN:-}${DESEC_TOKEN:-}${DESEC_DOMAIN:-}" ]; then
    echo "!!! Runpod proxy mode ignores ACME/deSEC settings because the proxy already terminates TLS."
    echo "!!! Set RUNPOD_DIRECT_TLS=1 and PUBLIC_ENDPOINT=https://host:mapped-port for direct TCP TLS."
  fi
fi

if [ "$ACME_DIRECT" = "1" ] && [ -n "${DESEC_TOKEN:-}" ] && [ -n "${DESEC_DOMAIN:-}" ] && [ -z "${ACME_DOMAIN:-}" ]; then
  SUB="model-${INSTANCE_ID:-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
  MYIP="$PUBLIC_IP_CURRENT"
  if [ -z "$MYIP" ]; then
    echo "!!! Could not determine public IP; skipping deSEC auto-DNS"
  else
    # ttl 3600 = deSEC's account minimum; lower values are rejected with HTTP 400
    echo ">>> Registering ${SUB}.${DESEC_DOMAIN} -> ${MYIP} via deSEC"
    curl -sf -X PUT "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/" \
      -H "Authorization: Token ${DESEC_TOKEN}" -H "Content-Type: application/json" \
      -d "[{\"subname\":\"${SUB}\",\"type\":\"A\",\"ttl\":3600,\"records\":[\"${MYIP}\"]}]" >/dev/null \
      && export ACME_DOMAIN="${SUB}.${DESEC_DOMAIN}" ACME_DNS_PROVIDER=desec DESEC_TOKEN \
      && echo ">>> Registered. Endpoint will be: https://${ACME_DOMAIN}:${VAST_TCP_PORT_8000:-${RUNPOD_TCP_PORT_8000:-<mapped-port>}}/v1" \
      || echo "!!! deSEC registration failed (HTTP error — check DESEC_TOKEN/DESEC_DOMAIN); continuing without auto-DNS"
  fi
fi

TLS_ARGS=()
if [ "$ACME_DIRECT" = "1" ] && [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ] && command -v lego >/dev/null; then
  CRT="/workspace/.lego/certificates/${ACME_DOMAIN}.crt"
  KEY="/workspace/.lego/certificates/${ACME_DOMAIN}.key"
  # /workspace persists across reboots: reuse a cert with >7 days left instead of
  # re-issuing every boot (LE duplicate-cert limit is 5/week; a reboot loop burns it)
  if [ -f "$CRT" ] && [ -f "$KEY" ] && openssl x509 -checkend 604800 -noout -in "$CRT" >/dev/null 2>&1; then
    echo ">>> Reusing persisted cert for $ACME_DOMAIN (>7 days validity left)"
  else
    echo ">>> Issuing LetsEncrypt cert for $ACME_DOMAIN via DNS-01 ($ACME_DNS_PROVIDER)"
    # DESEC_PROPAGATION_TIMEOUT: LE multi-perspective validation checks from
    # several vantage points; deSEC anycast can lag >60s — without the wait,
    # issuance fails with 'NXDOMAIN during secondary validation' (QC-run find).
    # 2 attempts: transient DNS propagation failures are common on first boot.
    for _try in 1 2; do
      DESEC_PROPAGATION_TIMEOUT="${DESEC_PROPAGATION_TIMEOUT:-300}" \
      lego --accept-tos --email "${ACME_EMAIL:-admin@$ACME_DOMAIN}" \
        --dns "$ACME_DNS_PROVIDER" --domains "$ACME_DOMAIN" \
        --path /workspace/.lego run && break
      echo "!!! ACME attempt $_try failed$( [ $_try = 2 ] && echo '; continuing WITHOUT TLS' || echo '; retrying in 30s')"
      sleep 30
    done
  fi
  if [ -f "$CRT" ] && [ -f "$KEY" ]; then
    TLS_ARGS=(--ssl-certfile "$CRT" --ssl-keyfile "$KEY")
    echo ">>> TLS enabled: https://$ACME_DOMAIN:${VAST_TCP_PORT_8000:-${RUNPOD_TCP_PORT_8000:-<mapped-port>}}/v1"
  fi
fi

# Feed the final endpoint/TLS picture to the landing page's status file
LOCAL_HEALTH_URL="http://localhost:${PORT:-8000}/health"
if [ ${#TLS_ARGS[@]} -gt 0 ]; then
  [ -n "$EP_URL" ] || EP_URL="https://${ACME_DOMAIN}:${VAST_TCP_PORT_8000:-${RUNPOD_TCP_PORT_8000:-8000}}"
  TLS_STATE="https://${ACME_DOMAIN}"
  if [ "$PLATFORM" != "runpod" ]; then
    HTTPS_HOSTPORT="${ACME_DOMAIN}:${VAST_TCP_PORT_1111:-1111}"
  fi
  CERT_PATH="$CRT" KEY_PATH="$KEY"
  LOCAL_HEALTH_URL="https://localhost:${PORT:-8000}/health"
elif [ "$PLATFORM" = "runpod" ]; then
  [ -n "$EP_URL" ] || EP_URL="https://${RUNPOD_POD_ID}-${PORT:-8000}.proxy.runpod.net"
  TLS_STATE="https:// managed by Runpod proxy"
else
  [ -n "$EP_URL" ] || EP_URL="http://${PUBLIC_IP_CURRENT:-localhost}:${VAST_TCP_PORT_8000:-8000}"
fi

if [ "$PLATFORM" = "runpod" ] && [ ${#TLS_ARGS[@]} -eq 0 ]; then
  echo ">>> Runpod proxy endpoint: ${EP_URL}/v1"
  echo "!!! Runpod's HTTP proxy has a 100-second connection limit; use the SSH tunnel"
  echo "!!! documented in README.md for long generations and large-context requests."
elif [ "$PLATFORM" = "runpod" ]; then
  echo ">>> Runpod direct-TCP TLS endpoint: ${EP_URL}/v1"
fi

# Egress hygiene: no telemetry; offline mode once weights are local
# Keep the torch.compile / AOT cache on the persistent volume. It defaults to
# /root/.cache/vllm inside the container, so a replaced container recompiles the
# backbone and the eagle head from scratch — ~100 s of every boot, for ~1.6 GB of
# artifacts. VLLM_DISABLE_COMPILE_CACHE is NOT set here (that guard belongs to the
# gilded-gnosis fork's cache bug); verify decode tok/s after enabling this.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MODEL_DIR/.vllm-cache}"
mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true
# Some upstream images still export the removed VLLM_CACHE_DIR alias. Recent
# vLLM warns for every unknown VLLM_* variable, so keep only the supported
# persistent-cache setting above.
unset VLLM_CACHE_DIR

export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
echo ">>> Listening sockets at boot (expect landing :1111, SSH :22, then vLLM :${PORT:-8000}):"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | head -8 || true

if [ "$MODEL_PROFILE" = "glm52-exl3" ]; then
# EXL3 trellis window vs speculative capture. The MTP draft is CUDA-graph captured
# at m=MTP_TOKENS. If that falls below VLLM_EXL3_TRELLIS_MIN_M the EXL3 kernel
# hands off to its eager parity path, which cannot be captured, and the engine dies:
#   RuntimeError: EXL3 eager parity path entered during CUDA graph capture (m=3);
#                 capture sizes must lie inside the Trellis window [4, 32]
# The default MIN_M=4 is therefore only safe by coincidence -- it happens to equal
# the 1+3 the TARGET captures at. With a separate draft (MTP78_MODE=override) the
# draft captures at m=3 and the boot fails. Widen the window instead of clamping
# MTP: MIN_M=1 is the configuration validated on this checkpoint (needle 6/6, fp8).
# There is deliberately NO auto-lower of VLLM_EXL3_TRELLIS_MIN_M here. An earlier
# revision lowered it to 1 whenever MTP_TOKENS < MIN_M, to clear the override-mode
# capture crash. That was wrong twice over: the stated mechanism was false (m does
# not track MTP_TOKENS), and lowering the window silently corrupts output instead of
# failing loudly. It also fired in GRAFT mode, where MIN_M=4 is correct and required,
# so it would have poisoned the one configuration that works. Leave MIN_M at 4.
# Concurrency vs the trellis/capture window. At decode the engine runs
#   m = MAX_NUM_SEQS * (1 + MTP_TOKENS)
# query tokens per step. m must stay inside BOTH the CUDA-graph capture window
# and the EXL3 trellis window [MIN_M, MAX_M], or decode silently falls off the
# captured trellis fast path onto the eager path once concurrency rises -- a
# cliff that no boot-time error announces and that only shows up as lost tok/s
# under real multi-stream load.
#
# This is why MAX_NUM_SEQS defaults to 8 and not 32: 8*(1+3)=32 exactly fills the
# default window. The old default of 32 gave m=128, four times outside it.
# To genuinely serve more streams, raise all three together, e.g. 16 seqs:
#   -e MAX_NUM_SEQS=16 \
#   -e CUDAGRAPH_CAPTURE_SIZES=8,16,24,32,40,48,56,64 \
#   -e MAX_CUDAGRAPH_CAPTURE_SIZE=64 -e TUNE_VLLM_EXL3_TRELLIS_MAX_M=64
# (more captured sizes cost capture time and some VRAM, so measure rather than
# assume it wins).
_DECODE_M=$(( ${MAX_NUM_SEQS:-8} * (1 + MTP_TOKENS) ))
_CAP_MAX=${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}
_TRELLIS_MAX=${VLLM_EXL3_TRELLIS_MAX_M:-32}
if [ "$_DECODE_M" -gt "$_CAP_MAX" ] || [ "$_DECODE_M" -gt "$_TRELLIS_MAX" ]; then
  echo "!!! WARNING: MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} x (1+MTP_TOKENS=$MTP_TOKENS) = $_DECODE_M query tokens/step,"
  echo "!!!          but the cudagraph capture window tops out at $_CAP_MAX and the EXL3 trellis"
  echo "!!!          window at $_TRELLIS_MAX. Decode will leave the captured trellis path once"
  echo "!!!          enough streams are concurrent. Raise CUDAGRAPH_CAPTURE_SIZES,"
  echo "!!!          MAX_CUDAGRAPH_CAPTURE_SIZE and TUNE_VLLM_EXL3_TRELLIS_MAX_M to >= $_DECODE_M,"
  echo "!!!          or lower MAX_NUM_SEQS to $(( _CAP_MAX / (1 + MTP_TOKENS) ))."
fi
fi

SPEC_ARGS=()
# DRAFT_QUANTIZATION: the draft's own quantization method. Without it the draft
# INHERITS the target's --quantization (exl3 here), so a non-EXL3 draft is loaded
# through the EXL3 path and dies in _apply_rank_sliced during speculator capture
# with the m=3 trellis-window error -- which looks identical to the v20 EXL3-draft
# bug but has a completely different cause. Measured on AIBeast 2026-07-26 with the
# NVFP4 draft from lukealonso/GLM-5.2-NVFP4 (quant_algo NVFP4, quant_method modelopt):
# vLLM's method name for it is "modelopt_fp4".
_SPEC_QUANT=""
if [ -n "${DRAFT_QUANTIZATION:-}" ]; then
  _SPEC_QUANT=",\"quantization\":\"${DRAFT_QUANTIZATION}\""
  echo ">>> Draft quantization: ${DRAFT_QUANTIZATION} (overrides the target's --quantization)"
fi
if [ "$MODEL_PROFILE" = "glm52-exl3" ] && [ "$MTP_TOKENS" != "0" ]; then
  if [ -n "$DRAFT_MODEL" ]; then
    SPEC_ARGS=(--speculative-config "{\"model\":\"$DRAFT_MODEL\",\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS,\"moe_backend\":\"triton\",\"draft_sample_method\":\"probabilistic\"${_SPEC_QUANT}}")
  else
    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS,\"moe_backend\":\"triton\",\"draft_sample_method\":\"probabilistic\"}")
  fi
elif [ "$MODEL_PROFILE" = "qwen36-27b-nvfp4" ] && [ "$MTP_TOKENS" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
  echo ">>> Qwen MTP: $MTP_TOKENS speculative token(s) (opt-in; profile default is off)"
elif [ "$MODEL_PROFILE" = "custom" ] && [ -n "${SPECULATIVE_CONFIG:-}" ]; then
  SPEC_ARGS=(--speculative-config "$SPECULATIVE_CONFIG")
fi

status_update starting-engine

# Best-effort: surface readiness + endpoint into the vast.ai dashboard label
if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
  ( until curl -skf "$LOCAL_HEALTH_URL" >/dev/null 2>&1; do sleep 20; done
    curl -s -X PUT "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
      -H "Authorization: Bearer ${CONTAINER_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"label\": \"${MODEL_DISPLAY_NAME} READY ${EP_URL}/v1\"}" >/dev/null 2>&1 || true
  ) &
fi

# Supervisor: on a rental, an engine crash must not leave the box idle burning
# money. Restart vLLM (up to SUPERVISOR_MAX_RESTARTS, default 5) with backoff;
# a crash-loop that exceeds the budget exits so the instance is visibly failed
# rather than thrashing. SUPERVISOR=0 disables (single exec, legacy behaviour).
serve_once() {
# --served-model-name accepts several aliases; SERVED_MODEL_NAME is split on
# whitespace so a drop-in replacement can answer to the names existing clients
# already use without touching those clients.
read -r -a SERVED_NAMES <<< "$SERVED_MODEL_NAME"
# AUTH=none -> omit --api-key entirely (passing an empty one still enforces auth).
# NB: `[ test ] && ARR=(...)` returns non-zero when the test is false, which
# under `set -e` kills serve_once outright — precisely in the AUTH=none and
# GPU_BLOCKS_OVERRIDE=0 cases these branches exist to support. Use if/then.
AUTH_ARGS=()
if [ "${AUTH:-key}" != "none" ]; then
  AUTH_ARGS=(--api-key "$VLLM_API_KEY")
fi
# Pool sizing: GLM pins the pool so its validated 512K KV headroom is
# predictable. Qwen/custom default to zero so vLLM sizes the pool normally.
BLOCKS_ARGS=()
if [ "$GPU_BLOCKS_OVERRIDE" != "0" ]; then
  BLOCKS_ARGS=(--num-gpu-blocks-override "$GPU_BLOCKS_OVERRIDE")
fi
case "$MODEL_PROFILE" in
  glm52-exl3)
    vllm serve "$MODEL_DIR" \
      --served-model-name "${SERVED_NAMES[@]}" \
      --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --decode-context-parallel-size 4 \
      --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 64 \
      --quantization exl3 --kv-cache-dtype fp8 \
      --attention-backend B12X_MLA_SPARSE --moe-backend b12x --load-format safetensors \
      --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES:-4,8,12,16,20,24,28,32}],\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_allreduce_rms\":true}}" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}" \
      --enable-chunked-prefill --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
      --enable-prompt-tokens-details --enable-force-include-usage \
      --no-async-scheduling \
      --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
      ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
      --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
      ${BLOCKS_ARGS[@]+"${BLOCKS_ARGS[@]}"} ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
      "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}"
    ;;
  qwen36-27b-nvfp4)
    QWEN_MODALITY_ARGS=()
    if [ "$MULTIMODAL" != "1" ]; then
      QWEN_MODALITY_ARGS=(--language-model-only)
    fi
    QWEN_KV_ARGS=()
    if [ -n "${KV_CACHE_DTYPE:-}" ]; then
      QWEN_KV_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    vllm serve "$MODEL_DIR" \
      --served-model-name "${SERVED_NAMES[@]}" \
      --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --quantization modelopt --load-format safetensors \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-chunked-prefill \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      --enable-prompt-tokens-details --enable-force-include-usage \
      ${QWEN_MODALITY_ARGS[@]+"${QWEN_MODALITY_ARGS[@]}"} \
      ${QWEN_KV_ARGS[@]+"${QWEN_KV_ARGS[@]}"} \
      ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
      "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}"
    ;;
  custom)
    CUSTOM_QUANT_ARGS=()
    if [ -n "${QUANTIZATION:-}" ]; then
      CUSTOM_QUANT_ARGS=(--quantization "$QUANTIZATION")
    fi
    CUSTOM_KV_ARGS=()
    if [ -n "${KV_CACHE_DTYPE:-}" ]; then
      CUSTOM_KV_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    CUSTOM_REASONING_ARGS=()
    if [ -n "${REASONING_PARSER:-}" ]; then
      CUSTOM_REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")
    fi
    CUSTOM_TOOL_ARGS=()
    if [ -n "${TOOL_CALL_PARSER:-}" ]; then
      CUSTOM_TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
    fi
    CUSTOM_MODALITY_ARGS=()
    if [ "$MULTIMODAL" != "1" ]; then
      CUSTOM_MODALITY_ARGS=(--language-model-only)
    fi
    vllm serve "$MODEL_DIR" \
      --served-model-name "${SERVED_NAMES[@]}" \
      --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-chunked-prefill \
      ${CUSTOM_QUANT_ARGS[@]+"${CUSTOM_QUANT_ARGS[@]}"} \
      ${CUSTOM_KV_ARGS[@]+"${CUSTOM_KV_ARGS[@]}"} \
      ${CUSTOM_REASONING_ARGS[@]+"${CUSTOM_REASONING_ARGS[@]}"} \
      ${CUSTOM_TOOL_ARGS[@]+"${CUSTOM_TOOL_ARGS[@]}"} \
      ${CUSTOM_MODALITY_ARGS[@]+"${CUSTOM_MODALITY_ARGS[@]}"} \
      ${BLOCKS_ARGS[@]+"${BLOCKS_ARGS[@]}"} \
      ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
      "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}"
    ;;
esac
}

if [ "${SUPERVISOR:-1}" = "0" ]; then
  serve_once
  exit $?
fi

# Supervisor v2 (QC-hardened). v1 relied on the foreground child exiting and on
# pattern-based cleanup; a real crash leaves VLLM:: workers alive holding all
# VRAM, so the parent's exit is not a reliable death signal and name-pattern
# kills miss the survivors. v2: run the server in its OWN SESSION (setsid), so
# the whole tree can be killed by process group, and detect death by polling
# both the process and /health.
kill_server_tree() {
  [ -n "${SRV_PID:-}" ] && kill -9 -- "-$SRV_PID" 2>/dev/null   # negative PID = whole group
  pkill -9 -f 'VLLM:[:]' 2>/dev/null
  pkill -9 -f 'EngineCor[e]' 2>/dev/null
  pkill -9 -f 'vllm serv[e]' 2>/dev/null
  for _ in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-0}" -lt 3000 ] 2>/dev/null && break
    sleep 5
  done
}

MAXR="${SUPERVISOR_MAX_RESTARTS:-5}"
# Job control: gives every background job its own process group, so the
# supervisor can reap the whole vLLM tree without spawning a detached bash.
set -m
for attempt in $(seq 0 "$MAXR"); do
  if [ "$attempt" -gt 0 ]; then
    status_update restarting
    echo "!!! vLLM died — restart $attempt/$MAXR in $((attempt*15))s" >&2
    sleep $((attempt*15))
    kill_server_tree
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
  # Death = the session leader is gone. (Health is not a liveness proxy: the
  # engine legitimately takes ~15 min of JIT/cudagraph work before it answers.)
  while kill -0 "$SRV_PID" 2>/dev/null; do sleep 10; done
  echo "!!! vLLM exited (attempt $attempt)" >&2
  kill_server_tree
done
echo "!!! vLLM crash-looped past $MAXR restarts — giving up (destroy or debug the instance)" >&2
exit 1
