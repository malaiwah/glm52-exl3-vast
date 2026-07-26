#!/bin/bash
# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 Blackwell (96GB), TP4/DCP4,
# 512K context, fp8 KV (correct on stock drivers — see evidence gists in labels),
# MTP speculative decode, DRAM KV offload auto-sized to a fraction of the
# instance's RAM allocation (cgroup-aware).
# All logs go to stdout (vast.ai console). SSH per vast standards works alongside.
#
# SELF-SERVICE CONFIG (docs/self-service-config.md). Every tunable is resolved
# from three layers — built-in defaults < startup environment < the JSON state
# file the landing page writes on the volume — re-resolved on every vLLM start,
# so the user can change the deployment from the :1111 page without rebuilding
# the image or destroying the rental. A candidate config that fails to come up,
# or that comes up and fails the long-context probe, is rolled back to the last
# known-good one automatically, with its config and log preserved for analysis.
set -e

echo "=== GLM-5.2 EXL3 turnkey ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -4 || true
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 4 ] || { echo "FATAL: need 4 GPUs, found $NGPU"; exit 1; }

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
SCRIPTS_DIR="${SCRIPTS_DIR:-/opt/scripts}"
MODEL_ROOT="${MODEL_ROOT:-/workspace}"
export GLM_STATE_DIR="${GLM_STATE_DIR:-$MODEL_ROOT/.glm-config}"
export GLM_RUNTIME_DIR="${GLM_RUNTIME_DIR:-/tmp/glm-runtime}"
GLM_LOG_DIR="$GLM_STATE_DIR/logs"
SERVE_LOG="$GLM_LOG_DIR/serve-current.log"
RESTART_FLAG="$GLM_RUNTIME_DIR/restart-request"
VERIFY_FILE="$GLM_RUNTIME_DIR/verify.json"
VERIFY_LAST="$GLM_STATE_DIR/verify-last.json"
CONFIG_ENV="$GLM_RUNTIME_DIR/config.env"
mkdir -p "$GLM_STATE_DIR" "$GLM_RUNTIME_DIR" "$GLM_LOG_DIR" "$GLM_STATE_DIR/failures"
rm -f "$RESTART_FLAG" "$VERIFY_FILE"

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
  python3 "$SCRIPTS_DIR/config_cli.py" show || true
  # MODEL_DIR follows the variant unless the template pinned it.
  if [ -n "$MODEL_DIR_PINNED" ]; then
    MODEL_DIR="$MODEL_DIR_PINNED"
  else
    MODEL_DIR="$MODEL_ROOT/${MODEL_DIRNAME:-GLM-5.2-EXL3-TR3-3.0bpw}"
  fi
  export MODEL_DIR
  return 0
}

apply_config

# Boot-status snapshot for the landing page; rewritten at each milestone.
# Holds the API key once generated -> keep it root-only.
STATUS_FILE="${STATUS_FILE:-/tmp/glm-boot-status.json}"
EP_URL="" TLS_STATE="not configured" HTTPS_HOSTPORT="" CERT_PATH="" KEY_PATH="" OFFLOAD_STATE="off"
status_update() {
  printf '{"phase":"%s","endpoint":"%s","tls":"%s","https_hostport":"%s","cert":"%s","keyfile":"%s","offload":"%s","api_key":"%s","model_dir":"%s"}\n' \
    "$1" "$EP_URL" "$TLS_STATE" "$HTTPS_HOSTPORT" "$CERT_PATH" "$KEY_PATH" "$OFFLOAD_STATE" "${VLLM_API_KEY:-}" "$MODEL_DIR" > "$STATUS_FILE"
  chmod 600 "$STATUS_FILE" 2>/dev/null || true
}

# Landing page for the vast "Open" button (:1111, dual-protocol TLS+plain).
# Started before the weight download so status is visible from minute one.
# Needs OPEN_BUTTON_PORT=1111 env + '-p 1111:1111' in the template.
# It also hosts the config editor, so it gets the state/runtime dirs and the
# scripts dir (for the shared knob registry in glm_config.py).
if [ "${LANDING_PAGE:-1}" != "0" ] && [ -f /opt/landing.py ]; then
  status_update booting
  MODEL_DIR="$MODEL_DIR" STATUS_FILE="$STATUS_FILE" GLM_SCRIPTS_DIR="$SCRIPTS_DIR" \
    GLM_STATE_DIR="$GLM_STATE_DIR" GLM_RUNTIME_DIR="$GLM_RUNTIME_DIR" \
    LANDING_API_PORT="${PORT:-8000}" python3 /opt/landing.py &
  echo ">>> Landing page (Open button) live on :1111"
fi

fetch_weights() {
  # Gate on a completion marker, not config.json: small files land early in the
  # parallel download, so config.json existing does not mean the shards made it.
  # snapshot_download resumes/verifies incrementally, so re-running is safe.
  if [ -f "$MODEL_DIR/.download-complete" ]; then
    return 0
  fi
  status_update downloading-weights
  echo ">>> Downloading ${MODEL_REPO:-brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw} weights to $MODEL_DIR (resumes if interrupted)"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo ">>> (HF_TOKEN detected: authenticated download)"
  else
    echo ">>> (set HF_TOKEN env for higher rate limits)"
  fi
  # HF_HUB_OFFLINE is exported later in the boot; a variant switch at runtime has
  # to be able to reach the hub again, so clear it for the download only.
  HF_HUB_OFFLINE=0 HF_XET_HIGH_PERFORMANCE=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_REPO:-brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw}', local_dir='$MODEL_DIR', max_workers=16)"
  touch "$MODEL_DIR/.download-complete"
  echo ">>> Weights ready."
  return 0
}

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
# MTP78_TRELLIS=0 is still honoured as a synonym for off (it is folded into the
# MTP_DRAFT knob by the config resolver, which also accepts MTP78_MODE directly).
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
# The config layer refuses MTP_DRAFT=tr3-override on the v20 base for this reason.
MTP78_DRAFT_DIR=""

fetch_mtp78_overlay() {
  [ -d "$MODEL_DIR/.mtp78-overlay/3bpw-keep0" ] && return 0
  echo ">>> MTP78: downloading 3bpw-keep0 trellis overlay (~3.7 GB)"
  HF_HUB_OFFLINE=0 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('malaiwah/GLM-5.2-EXL3-TR3-MTP78', allow_patterns=['3bpw-keep0/*'], local_dir='$MODEL_DIR/.mtp78-overlay', max_workers=8)"
}

prepare_mtp78() {
  MTP78_DRAFT_DIR="$MODEL_DIR/.mtp78-draft"
  # DRAFT_MODEL may be supplied from the environment (or the state file) to point
  # --speculative-config at an EXTERNAL draft checkpoint, independently of
  # MTP78_MODE. This is how a non-EXL3 draft (e.g. the NVFP4 MTP draft from
  # lukealonso/GLM-5.2-NVFP4, model-mtp.safetensors, 5.6 GiB) is used: it recovers
  # most of the KV that the BF16 fallback draft gives up, while still avoiding
  # _apply_rank_sliced -- so it does not hit the v20 speculator capture bug that
  # makes an EXL3 rank-sliced draft unbootable (see the warning above).
  if [ -n "${DRAFT_MODEL:-}" ]; then
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
  return 0
}

# Vision (default ON): MoonViT-3d tower (Kimi-K2.6, frozen) + Baseten's trained
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
prepare_vision() {
  VISION_REPO="${VISION_REPO:-chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid}"
  if [ "${VISION:-1}" = "1" ]; then
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
      HF_HUB_OFFLINE=0 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$VISION_REPO', local_dir='$MODEL_DIR/.vision',
  allow_patterns=['vision_tower.safetensors','mm_projector.safetensors','config.json',
                  'configuration_glm5v.py','kimi_k25_processor.py','kimi_k25_vision_processing.py',
                  'media_utils.py','preprocessor_config.json','chat_template.jinja',
                  'plugins/**'], max_workers=8)" || vision_ok=0
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
  fetch_weights
  prepare_mtp78
  prepare_vision
  python3 "$SCRIPTS_DIR/reconcile_checkpoint.py" "$MODEL_DIR" --vision "${VISION:-1}" || \
    echo "!!! reconcile: failed — the checkpoint config may not match the weights on disk"
  return 0
}

# DRAM KV offload: OFFLOAD_FRACTION of the instance's RAM allocation (default
# 0.70); OFFLOAD_FRACTION=0 disables. Sized from min(cgroup limit, MemTotal) —
# inside a container /proc/meminfo shows the whole host's RAM, but a partial
# rental (e.g. 4 of 8 GPUs) only gets a slice of it.
KVT_ARGS=()
compute_offload() {
  KVT_ARGS=()
  local off_fraction="${OFFLOAD_FRACTION:-0.70}"
  if [ "$off_fraction" != "0" ]; then
    MEM_BYTES=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') * 1024 ))
    CG_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
    if [ "$CG_LIMIT" != "max" ] && [ "$CG_LIMIT" -lt "$MEM_BYTES" ] 2>/dev/null; then
      MEM_BYTES=$CG_LIMIT
    fi
    OFF_BYTES=$(python3 -c "print(int($MEM_BYTES*$off_fraction))")
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
    OFFLOAD_STATE="$((OFF_BYTES/1073741824)) GiB pinned DRAM"
    echo ">>> DRAM KV offload: $((OFF_BYTES/1073741824)) GiB (${off_fraction} of instance RAM allocation)"
    # kv_load_failure_policy=recompute: a KV block that cannot be fetched back from
    # the DRAM tier is recomputed instead of failing the request. The vLLM default
    # is "fail", which would turn any offload hiccup into a terminal error — not an
    # acceptable trade for a cache tier that is a pure optimisation.
    KVT_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF_BYTES}}")
    # OffloadingConnector rejects expandable_segments (VMM can remap pinned KV pages)
    export PYTORCH_CUDA_ALLOC_CONF=""
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
  if [ "${VISION:-1}" = "1" ] && [ -f "$MODEL_DIR/.vision-enabled" ]; then
    VISION_ARGS=(--limit-mm-per-prompt "{\"vision_chunk\":${VISION_CHUNKS:-8}}" --trust-remote-code)
  fi
  return 0
}

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
# NB: the two trellis-window variables above are ALSO self-service knobs. The
# authoritative block runs once, at boot; apply_config re-exports the resolved
# values before every vLLM start, so a state-file change wins over both.
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
# A per-instance name — stable across reboots (keyed to CONTAINER_ID) so DNS
# records don't pile up in the zone and the LE cert can be reused — is
# registered at startup and pointed at this instance.
if [ -n "${DESEC_TOKEN:-}" ] && [ -n "${DESEC_DOMAIN:-}" ] && [ -z "${ACME_DOMAIN:-}" ]; then
  SUB="glm-${CONTAINER_ID:-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
  MYIP="${PUBLIC_IPADDR:-$(curl -s -m 10 https://api.ipify.org)}"
  if [ -z "$MYIP" ]; then
    echo "!!! Could not determine public IP; skipping deSEC auto-DNS"
  else
    # ttl 3600 = deSEC's account minimum; lower values are rejected with HTTP 400
    echo ">>> Registering ${SUB}.${DESEC_DOMAIN} -> ${MYIP} via deSEC"
    curl -sf -X PUT "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/" \
      -H "Authorization: Token ${DESEC_TOKEN}" -H "Content-Type: application/json" \
      -d "[{\"subname\":\"${SUB}\",\"type\":\"A\",\"ttl\":3600,\"records\":[\"${MYIP}\"]}]" >/dev/null \
      && export ACME_DOMAIN="${SUB}.${DESEC_DOMAIN}" ACME_DNS_PROVIDER=desec DESEC_TOKEN \
      && echo ">>> Registered. Endpoint will be: https://${ACME_DOMAIN}:${VAST_TCP_PORT_8000:-<mapped-port>}/v1" \
      || echo "!!! deSEC registration failed (HTTP error — check DESEC_TOKEN/DESEC_DOMAIN); continuing without auto-DNS"
  fi
fi

TLS_ARGS=()
if [ -n "${ACME_DOMAIN:-}" ] && [ -n "${ACME_DNS_PROVIDER:-}" ] && command -v lego >/dev/null; then
  CRT="/workspace/.lego/certificates/${ACME_DOMAIN}.crt"
  KEY="/workspace/.lego/certificates/${ACME_DOMAIN}.key"
  # /workspace persists across reboots: reuse a cert with >7 days left instead of
  # re-issuing every boot (LE duplicate-cert limit is 5/week; a reboot loop burns it)
  if [ -f "$CRT" ] && openssl x509 -checkend 604800 -noout -in "$CRT" >/dev/null 2>&1; then
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
  [ -f "$CRT" ] && TLS_ARGS=(--ssl-certfile "$CRT" --ssl-keyfile "$KEY") && echo ">>> TLS enabled: https://$ACME_DOMAIN:${VAST_TCP_PORT_8000:-<mapped-port>}/v1"
fi

# Feed the final endpoint/TLS picture to the landing page's status file
if [ ${#TLS_ARGS[@]} -gt 0 ]; then
  EP_URL="https://${ACME_DOMAIN}:${VAST_TCP_PORT_8000:-8000}"
  TLS_STATE="https://${ACME_DOMAIN}"
  HTTPS_HOSTPORT="${ACME_DOMAIN}:${VAST_TCP_PORT_1111:-1111}"
  CERT_PATH="$CRT" KEY_PATH="$KEY"
  LOCAL_SCHEME="https"
else
  EP_URL="http://${PUBLIC_IPADDR:-localhost}:${VAST_TCP_PORT_8000:-8000}"
  LOCAL_SCHEME="http"
fi
LOCAL_BASE="$LOCAL_SCHEME://localhost:${PORT:-8000}"

# Egress hygiene: no telemetry; offline mode once weights are local
# Keep the torch.compile / AOT cache on the persistent volume. It defaults to
# /root/.cache/vllm inside the container, so a replaced container recompiles the
# backbone and the eagle head from scratch — ~100 s of every boot, for ~1.6 GB of
# artifacts. VLLM_DISABLE_COMPILE_CACHE is NOT set here (that guard belongs to the
# gilded-gnosis fork's cache bug); verify decode tok/s after enabling this.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MODEL_DIR/.vllm-cache}"
mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true

export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
echo ">>> Listening sockets at boot (expect only vllm on ${PORT:-8000} + vast ssh):"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | head -8 || true

# EXL3 trellis window vs speculative capture. The MTP draft is CUDA-graph captured
# at m=MTP_TOKENS. If that falls below VLLM_EXL3_TRELLIS_MIN_M the EXL3 kernel
# hands off to its eager parity path, which cannot be captured, and the engine dies:
#   RuntimeError: EXL3 eager parity path entered during CUDA graph capture (m=3);
#                 capture sizes must lie inside the Trellis window [4, 32]
# The default MIN_M=4 is therefore only safe by coincidence -- it happens to equal
# the 1+3 the TARGET captures at. With a separate draft (MTP78_MODE=override) the
# draft captures at m=3 and the boot fails.
# There is deliberately NO auto-lower of VLLM_EXL3_TRELLIS_MIN_M here. An earlier
# revision lowered it to 1 whenever MTP_TOKENS < MIN_M, to clear the override-mode
# capture crash. That was wrong twice over: the stated mechanism was false (m does
# not track MTP_TOKENS), and lowering the window silently corrupts output instead of
# failing loudly. It also fired in GRAFT mode, where MIN_M=4 is correct and required,
# so it would have poisoned the one configuration that works. Leave MIN_M at 4 —
# the config layer marks it non-editable and rejects any state file that moves it.
#
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
  _SPEC_QUANT=""
  if [ -n "${DRAFT_QUANTIZATION:-}" ]; then
    _SPEC_QUANT=",\"quantization\":\"${DRAFT_QUANTIZATION}\""
    echo ">>> Draft quantization: ${DRAFT_QUANTIZATION} (overrides the target's --quantization)"
  fi
  if [ "${MTP_TOKENS:-3}" != "0" ]; then
    if [ -n "${DRAFT_MODEL:-}" ]; then
      SPEC_ARGS=(--speculative-config "{\"model\":\"$DRAFT_MODEL\",\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-3},\"moe_backend\":\"triton\",\"draft_sample_method\":\"probabilistic\"${_SPEC_QUANT}}")
    else
      SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-3},\"moe_backend\":\"triton\",\"draft_sample_method\":\"probabilistic\"}")
    fi
  fi
  return 0
}

# Best-effort: surface readiness + endpoint into the vast.ai dashboard label
if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
  ( EP="${ACME_DOMAIN:+https://$ACME_DOMAIN}"; EP="${EP:-http://${PUBLIC_IPADDR:-?}}"
    PORTPART="${VAST_TCP_PORT_8000:+:$VAST_TCP_PORT_8000}"
    until curl -sf "http://localhost:${PORT:-8000}/health" >/dev/null 2>&1; do sleep 20; done
    curl -s -X PUT "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/"       -H "Authorization: Bearer ${CONTAINER_API_KEY}" -H "Content-Type: application/json"       -d "{\"label\": \"GLM-5.2 READY ${EP}${PORTPART}/v1\"}" >/dev/null 2>&1 || true
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
read -r -a SERVED_NAMES <<< "${SERVED_MODEL_NAME:-GLM-5.2}"
# AUTH=none -> omit --api-key entirely (passing an empty one still enforces auth).
# NB: `[ test ] && ARR=(...)` returns non-zero when the test is false, which
# under `set -e` kills serve_once outright — precisely in the AUTH=none and
# GPU_BLOCKS_OVERRIDE=0 cases these branches exist to support. Use if/then.
AUTH_ARGS=()
if [ "${AUTH:-key}" != "none" ]; then
  AUTH_ARGS=(--api-key "$VLLM_API_KEY")
fi
# Pool sizing: the override pins the pool at 512K so the KV headroom the trellis
# draft frees is predictable rather than absorbed. Set GPU_BLOCKS_OVERRIDE=0 to
# drop the flag and let vLLM use all available KV (bigger pool, more concurrency).
BLOCKS_ARGS=()
if [ "${GPU_BLOCKS_OVERRIDE:-2048}" != "0" ]; then
  BLOCKS_ARGS=(--num-gpu-blocks-override "${GPU_BLOCKS_OVERRIDE:-2048}")
fi
# stdout still goes to the vast console; the tee copy is what a rollback
# preserves as the failed boot's error log. Process substitution (not a pipe) so
# that $! stays the vllm process — a pipeline's exit would otherwise wait on tee,
# which surviving VLLM:: workers can hold open long after the engine is dead.
vllm serve "$MODEL_DIR" \
  --served-model-name "${SERVED_NAMES[@]}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --trust-remote-code \
  --tensor-parallel-size 4 --decode-context-parallel-size "${DCP:-4}" \
  --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 64 \
  --quantization "${QUANTIZATION:-exl3}" --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --attention-backend B12X_MLA_SPARSE --moe-backend b12x --load-format safetensors \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES:-4,8,12,16,20,24,28,32}],\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_allreduce_rms\":true}}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}" \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-3072}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}" \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-prompt-tokens-details --enable-force-include-usage \
  --no-async-scheduling \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  ${BLOCKS_ARGS[@]+"${BLOCKS_ARGS[@]}"} ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
  "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}" \
  > >(tee -a "$SERVE_LOG") 2>&1
}

if [ "${SUPERVISOR:-1}" = "0" ]; then
  prepare_checkpoint
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
# A short-prompt health check is NOT evidence of correctness: nvfp4 KV without
# calibrated scales, a lowered trellis MIN_M and vision on the EXL3 target all
# pass /health and a two-line completion while producing garbage past ~32K
# tokens. So every start is followed by a probe that includes a long-context
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
  read -r -a _names <<< "${SERVED_MODEL_NAME:-GLM-5.2}"
  python3 "$SCRIPTS_DIR/verify_serving.py" \
    --base-url "$LOCAL_BASE" --api-key "${VLLM_API_KEY:-}" --model "${_names[0]}" \
    --out "$VERIFY_FILE" --pid "$SRV_PID" \
    --max-model-len "${MAX_MODEL_LEN:-524288}" \
    --needle-tokens "${VERIFY_NEEDLE_TOKENS:-32768}" \
    --timeout "${VERIFY_HEALTH_TIMEOUT_S:-2400}" \
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
  read -r -a _names <<< "${SERVED_MODEL_NAME:-GLM-5.2}"
  echo ">>> Self-analysis: asking the running model why $fdir failed"
  python3 "$SCRIPTS_DIR/analyze_failure.py" --dir "$fdir" \
    --base-url "$LOCAL_BASE" --api-key "${VLLM_API_KEY:-}" --model "${_names[0]}" \
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
  prepare_checkpoint
  compute_offload
  compute_vision_args
  warn_capture_window
  build_spec_args
  rm -f "$VERIFY_FILE"
  : > "$SERVE_LOG"

  status_update starting-engine
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
    if [ -f "$RESTART_FLAG" ]; then
      reason="requested"
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      reason="died"
      break
    fi
    if [ "$verified" = "0" ] && [ -f "$VERIFY_FILE" ]; then
      verified=1
      mv "$VERIFY_FILE" "$VERIFY_LAST" || true
      if verdict_ok "$VERIFY_LAST"; then
        echo ">>> Verified: $(verdict_reason "$VERIFY_LAST")"
        python3 "$SCRIPTS_DIR/config_cli.py" mark-good --log "$SERVE_LOG" || true
        status_update serving
        attempt=0
        good=1
        maybe_self_analyze
      else
        echo "!!! VERIFICATION FAILED: $(verdict_reason "$VERIFY_LAST")" >&2
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
