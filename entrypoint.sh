#!/bin/bash
# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 Blackwell (96GB), TP4/DCP4,
# 512K context, fp8 KV (correct on stock drivers — see evidence gists in labels),
# MTP speculative decode, DRAM KV offload auto-sized to a fraction of the
# instance's RAM allocation (cgroup-aware).
# All logs go to stdout (vast.ai console). SSH per vast standards works alongside.
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

MODEL_DIR="${MODEL_DIR:-/workspace/GLM-5.2-EXL3-TR3-3.0bpw}"

# Boot-status snapshot for the landing page; rewritten at each milestone.
# Holds the API key once generated -> keep it root-only.
STATUS_FILE="${STATUS_FILE:-/tmp/glm-boot-status.json}"
EP_URL="" TLS_STATE="not configured" HTTPS_HOSTPORT="" CERT_PATH="" KEY_PATH="" OFFLOAD_STATE="off"
status_update() {
  printf '{"phase":"%s","endpoint":"%s","tls":"%s","https_hostport":"%s","cert":"%s","keyfile":"%s","offload":"%s","api_key":"%s"}\n' \
    "$1" "$EP_URL" "$TLS_STATE" "$HTTPS_HOSTPORT" "$CERT_PATH" "$KEY_PATH" "$OFFLOAD_STATE" "${VLLM_API_KEY:-}" > "$STATUS_FILE"
  chmod 600 "$STATUS_FILE" 2>/dev/null || true
}

# Landing page for the vast "Open" button (:1111, dual-protocol TLS+plain).
# Started before the weight download so status is visible from minute one.
# Needs OPEN_BUTTON_PORT=1111 env + '-p 1111:1111' in the template.
if [ "${LANDING_PAGE:-1}" != "0" ] && [ -f /opt/landing.py ]; then
  status_update booting
  MODEL_DIR="$MODEL_DIR" STATUS_FILE="$STATUS_FILE" python3 /opt/landing.py &
  echo ">>> Landing page (Open button) live on :1111"
fi

# Gate on a completion marker, not config.json: small files land early in the
# parallel download, so config.json existing does not mean the shards made it.
# snapshot_download resumes/verifies incrementally, so re-running is safe.
if [ ! -f "$MODEL_DIR/.download-complete" ]; then
  status_update downloading-weights
  echo ">>> Downloading EXL3 weights (~332 GB) to $MODEL_DIR (resumes if interrupted)"
  [ -n "${HF_TOKEN:-}" ] && echo ">>> (HF_TOKEN detected: authenticated download)" || echo ">>> (set HF_TOKEN env for higher rate limits)"
  HF_XET_HIGH_PERFORMANCE=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw', local_dir='$MODEL_DIR', max_workers=16)"
  touch "$MODEL_DIR/.download-complete"
  echo ">>> Weights ready."
fi

# MTP78 trellis draft overlay (default ON): the MTP draft layer quantized to
# 3.0bpw EXL3 (all-256-expert, full-corpus calibrated) — validated at BF16
# acceptance PARITY (MAL 3.06 vs 3.05) while freeing ~3.8 GB/GPU for KV cache.
# MTP78_TRELLIS=0 reverts to the stock BF16 draft. Needs the deepseek_mtp
# loader patch (voipmonitor/vllm#11) applied to the image's vLLM at boot.
MTP78_TRELLIS="${MTP78_TRELLIS:-1}"
if [ "$MTP78_TRELLIS" = "1" ]; then
  if [ ! -f "$MODEL_DIR/.mtp78-grafted" ]; then
    status_update grafting-mtp78
    echo ">>> MTP78: downloading 3bpw-keep0 trellis overlay (~3.7 GB)"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('malaiwah/GLM-5.2-EXL3-TR3-MTP78', allow_patterns=['3bpw-keep0/*'], local_dir='$MODEL_DIR/.mtp78-overlay', max_workers=8)"
    if python3 /opt/scripts/patch_deepseek_mtp.py; then
      if python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" "$MODEL_DIR/.mtp78-overlay/3bpw-keep0"; then
        echo ">>> MTP78: trellis draft active (BF16-parity acceptance, +KV headroom)"
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
elif [ -f "$MODEL_DIR/.mtp78-grafted" ]; then
  echo ">>> MTP78_TRELLIS=0: reverting graft to stock BF16 draft"
  python3 /opt/scripts/graft_mtp78.py "$MODEL_DIR" --revert
fi

# Vision (default ON): MoonViT-3d tower (Kimi-K2.6, frozen) + Baseten's trained
# 49.5M-param PatchMerger projector, bolted onto our EXL3 text backbone. Both
# vision parts are BF16 and checkpoint-agnostic — only ~1 GB, no text weight is
# touched. VISION=0 serves pure text.
# The EXL3 loader is multimodal-aware (it collapses `language_model.` prefixes),
# so the rank-sliced text weights still load under the Glm5v wrapper.
VISION="${VISION:-1}"
VISION_REPO="${VISION_REPO:-chronarion/GLM-5.2-Vision-MXFP8-NVFP4-NF3-Hybrid}"
if [ "$VISION" = "1" ]; then
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
    echo ">>> Vision: already installed"
  fi
elif [ -f "$MODEL_DIR/.vision-enabled" ]; then
  echo ">>> VISION=0: reverting to text-only config"
  python3 /opt/scripts/build_vision_config.py "$MODEL_DIR" --revert
  python3 /opt/scripts/index_add_vision.py "$MODEL_DIR" --revert || true
  [ -f "$MODEL_DIR/chat_template.jinja.text-only" ] && mv "$MODEL_DIR/chat_template.jinja.text-only" "$MODEL_DIR/chat_template.jinja"
fi

# DRAM KV offload: OFFLOAD_FRACTION of the instance's RAM allocation (default
# 0.70); OFFLOAD_FRACTION=0 disables. Sized from min(cgroup limit, MemTotal) —
# inside a container /proc/meminfo shows the whole host's RAM, but a partial
# rental (e.g. 4 of 8 GPUs) only gets a slice of it.
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-0.70}"
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
    if [ "${OFFLOAD_IGNORE_MEMLOCK:-0}" = "1" ]; then
      # Measured 2026-07-26 on an owned box: a 128 GiB offload tier runs fine with
      # memlock capped at ~31 GiB (kv_offload_total_bytes climbs normally), because
      # the connector does not mlock the whole tier up front. Rootless podman also
      # cannot raise memlock past the user's hard limit, so '--ulimit memlock=-1:-1'
      # is a no-op there and the check would disable a working feature outright.
      # Opt-in, because on an unknown rental host the conservative default is right.
      echo "!!! OFFLOAD_IGNORE_MEMLOCK=1 — proceeding anyway (verify kv_offload_* metrics climb)."
    else
      echo "!!! Add '--ulimit memlock=-1:-1' to the template Docker options to enable offload,"
      echo "!!! or set OFFLOAD_IGNORE_MEMLOCK=1 if you have verified offload works at this limit."
      echo "!!! Continuing WITHOUT DRAM offload."
      OFFLOAD_FRACTION=0
    fi
  fi
fi
if [ "$OFFLOAD_FRACTION" != "0" ]; then
  OFFLOAD_STATE="$((OFF_BYTES/1073741824)) GiB pinned DRAM"
  echo ">>> DRAM KV offload: $((OFF_BYTES/1073741824)) GiB (${OFFLOAD_FRACTION} of instance RAM allocation)"
  KVT_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF_BYTES}}")
  # OffloadingConnector rejects expandable_segments (VMM can remap pinned KV pages)
  export PYTORCH_CUDA_ALLOC_CONF=""
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# Vision serve args: bound the encoder reserve (chronarion's guidance) and cap
# media per request so a screenshot burst can't blow the profile.
VISION_ARGS=()
if [ "${VISION:-1}" = "1" ] && [ -f "$MODEL_DIR/.vision-enabled" ]; then
  VISION_ARGS=(--limit-mm-per-prompt "{\"vision_chunk\":${VISION_CHUNKS:-8}}" --trust-remote-code)
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
else
  EP_URL="http://${PUBLIC_IPADDR:-localhost}:${VAST_TCP_PORT_8000:-8000}"
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

status_update starting-engine

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
vllm serve "$MODEL_DIR" \
  --served-model-name "${SERVED_NAMES[@]}" \
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
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-prompt-tokens-details --enable-force-include-usage \
  --no-async-scheduling \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  ${BLOCKS_ARGS[@]+"${BLOCKS_ARGS[@]}"} ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
  "${TLS_ARGS[@]}" "${SPEC_ARGS[@]}" "${KVT_ARGS[@]}"
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
