#!/usr/bin/env bash
# Fleet deployment for JarvisLabs: one persistent filesystem, cheap instances
# that fill it, GPU instances that serve from it warm, and an optional LiteLLM
# router that keeps sessions on the replica whose prefix cache already holds
# them.
#
# The pattern (see docs/jarvislabs-fleet.md for the full runbook):
#
#   jl filesystem create --name glm53-weights --storage 500 --region IN1
#   jarvislabs_fleet.sh populate        # 1-GPU spot fills the FS, then dies
#   jarvislabs_fleet.sh serve           # 4-GPU spot boots from the warm FS
#   jarvislabs_fleet.sh router          # CPU VM LiteLLM with prefix affinity
#   jarvislabs_fleet.sh scale           # second serve instance + router pick-up
#
# Weights live ONLY on the filesystem; each instance keeps its config state,
# caches and tokens on its own disk, so replicas never write to the shared
# volume. JarvisLabs filesystems mount at /home/jl_fs and may be attached to
# any number of your instances at once.
#
# Requires: jl (authenticated), ssh, python3. All costs are spot pricing.
set -Eeuo pipefail

FLEET_STATE="${FLEET_STATE:-$HOME/.jarvis-fleet.json}"
FS_NAME="${FS_NAME:-glm53-weights}"
FS_SIZE_GB="${FS_SIZE_GB:-500}"
FS_REGION="${FS_REGION:-IN1}"
POPULATOR_GPU="${POPULATOR_GPU:-RTX-PRO6000}"
SERVE_GPU="${SERVE_GPU:-RTX-PRO6000}"
SERVE_GPUS="${SERVE_GPUS:-4}"
ROUTER_VCPUS="${ROUTER_VCPUS:-2}"
ROUTER_RAM_GB="${ROUTER_RAM_GB:-8}"
ROUTER_PORT="${ROUTER_PORT:-4000}"
WEIGHTS_SUBDIR="GLM-5.3-EXL3-TR3-3.42bpw"
MODEL_REPO="davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw"
MODEL_REVISION="99c6f951333d2b38f1efefa533c7afadf0d376e3"
FS_MOUNT="/home/jl_fs"
QUICKSTART_REF="cdfbbe41c2913b161f20e3e42ce897b705d66667"
QUICKSTART_URL="https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/$QUICKSTART_REF/scripts/jarvislabs_quickstart.sh"

log()   { printf '>>> %s\n' "$*" >&2; }
fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 2; }

require_jl() { command -v jl >/dev/null 2>&1 || fatal "jl CLI not found; run 'jl setup' first."; }

# --- fleet state --------------------------------------------------------------
state_load() { [[ -r "$FLEET_STATE" ]] && cat "$FLEET_STATE" || echo '{}'; }
state_set() { # state_set <key> <value>
  local tmp
  tmp=$(mktemp)
  state_load | python3 -c "
import json, sys
doc = json.load(sys.stdin)
doc[sys.argv[1]] = sys.argv[2]
print(json.dumps(doc, indent=2))
" "$1" "$2" >"$tmp" && mv "$tmp" "$FLEET_STATE"
}
state_get() { state_load | python3 -c "
import json, sys
print(json.load(sys.stdin).get(sys.argv[1], ''))
" "$1"; }

fs_id() {
  local id
  id=$(jl filesystem list --json 2>/dev/null | python3 -c "
import json, sys
for row in json.load(sys.stdin):
    if row.get('fs_name') == '$FS_NAME':
        print(row.get('fs_id')); break
" 2>/dev/null || true)
  [[ -n "$id" ]] || fatal "filesystem '$FS_NAME' not found; run: jl filesystem create --name $FS_NAME --storage $FS_SIZE_GB --region $FS_REGION"
  echo "$id"
}

wait_ssh() { # wait_ssh <ip> [timeout]
  local ip="$1" deadline=$((SECONDS + ${2:-180}))
  while ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$ip" true 2>/dev/null; do
    (( SECONDS < deadline )) || fatal "ssh to $ip never became ready"
    sleep 5
  done
}

ROUTER_USER="ubuntu"
wait_ssh_router() { # CPU VMs log in as the ubuntu user, not root
  local ip="$1" deadline=$((SECONDS + 300))
  while ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ROUTER_USER@$ip" true 2>/dev/null; do
    (( SECONDS < deadline )) || fatal "ssh to $ROUTER_USER@$ip never became ready"
    sleep 5
  done
}

create_instance() { # create_instance <name> <extra jl create args...>; prints machine_id
  local name="$1"; shift
  local out
  out=$(jl create --name "$name" --yes --json "$@") ||
    fatal "jl create failed for $name: $out"
  printf '%s' "$out" | python3 -c "
import json, sys
doc = json.load(sys.stdin)
print(doc['machine_id'])
print(doc.get('public_ip', ''), file=sys.stderr)
"
}


instance_ip() { # instance_ip <machine_id>: jl get can answer before the
                # public IP is assigned; poll until it appears (~120s max)
                # so wait_ssh never starts from an empty address.
  local ip deadline=$((SECONDS + 120))
  while :; do
    ip=$(jl get "$1" --json 2>/dev/null | python3 -c "
import json, sys
doc = json.load(sys.stdin)
print(doc.get('public_ip') or '')
" 2>/dev/null) || ip=""
    if [[ -n "$ip" ]]; then
      printf '%s\n' "$ip"
      return 0
    fi
    (( SECONDS < deadline )) || { log "no public IP for $1 after 120s"; return 1; }
    sleep 5
  done
}
# --- populate -----------------------------------------------------------------
cmd_populate() {
  require_jl
  local id
  id=$(fs_id)
  log "creating 1-GPU $POPULATOR_GPU spot instance to fill the filesystem (attach $id)"
  local t0=$SECONDS
  local mid
  mid=$(create_instance glm53-populator --gpu "$POPULATOR_GPU" --num-gpus 1 \
        --spot --region "$FS_REGION" --fs-id "$id")
  state_set populator "$mid"
  local ip
  ip=$(instance_ip "$mid")
  wait_ssh "$ip"
  log "populator $mid up ($ip); downloading $MODEL_REPO@$MODEL_REVISION (resumable)"
  local remote_cmd
  remote_cmd=$(printf 'REPO=%q REV=%q DIR=%q REF=%q bash -s' \
    "$MODEL_REPO" "$MODEL_REVISION" "$FS_MOUNT/$WEIGHTS_SUBDIR" "$QUICKSTART_REF")
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "root@$ip" "$remote_cmd" <<'REMOTE'
set -e
pip install -q hf_transfer 2>/dev/null || pip install -q hf_transfer
mkdir -p "$DIR"
t0=$SECONDS
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - "$REPO" "$REV" "$DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=16)
PYEOF
echo "POPULATE_SECONDS=$((SECONDS - t0))"
du -sh "$DIR"
# Prime the appliance image onto the filesystem in the same session: the
# runner's prepare honors an existing unpacked tree, so every replica skips
# the registry fetch (~3 minutes per boot).
export DEBIAN_FRONTEND=noninteractive
command -v skopeo >/dev/null 2>&1 || apt-get update -qq && apt-get install -y -qq skopeo rsync jq >/dev/null
curl -fsSL --connect-timeout 10 --max-time 60 --retry 10 --retry-delay 2 --retry-all-errors \
  https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/$REF/scripts/jarvislabs_container_rootfs.sh \
  -o /root/rootfs.sh
IMG="ghcr.io/malaiwah/glm52-exl3-vast:latest"
DIGEST=$(skopeo inspect --override-os linux --override-arch amd64 "docker://$IMG" --format '{{.Digest}}')
TURNKEY_IMAGE="$IMG@$DIGEST" TURNKEY_ROOT=/home/jl_fs/.image/qual \
  TURNKEY_WORKSPACE=/home/turnkey/workspace TURNKEY_GRAFT=0 bash /root/rootfs.sh prepare
du -sh /home/jl_fs/.image/qual
REMOTE
  jl destroy "$mid" --yes >/dev/null
  state_set populator ""
  log "filesystem filled in $((SECONDS - t0))s total (instance destroyed; data persists)"
}

# --- serve --------------------------------------------------------------------
cmd_serve() { # cmd_serve <name>
  require_jl
  local name="${1:-glm53-serve-a}"
  local id
  id=$(fs_id)
  local t0=$SECONDS
  local mid
  mid=$(create_instance "$name" --gpu "$SERVE_GPU" --num-gpus "$SERVE_GPUS" \
        --spot --region "$FS_REGION" --fs-id "$id" --http-ports 8000,1111)
  state_set "$name" "$mid"
  local ip
  ip=$(instance_ip "$mid")
  wait_ssh "$ip"
  log "serve instance $mid ($name) up ($ip); grafting appliance with weights pinned to the filesystem"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "root@$ip" "
    curl -fsSL '$QUICKSTART_URL' -o /root/quickstart.sh
    export MODEL_DIR='$FS_MOUNT/$WEIGHTS_SUBDIR'
    export GLM_STATE_DIR=/home/turnkey/workspace/.glm-config
    export TURNKEY_WORKSPACE=/home/turnkey/workspace
    # Reuse the unpacked appliance image from the shared filesystem: the
    # first boot pays the registry fetch, later instances skip it.
    export TURNKEY_ROOT='$FS_MOUNT/.image/qual'
    # One shared quantization cache: its content is a pure function of the
    # weights revision, the quantization algorithm and the GPU architecture,
    # so every replica reads the same bytes. Cold boots are serialized (the
    # manager creates one slot per cycle), so no two replicas write it at
    # once; a slot relaunch after any boot finds it complete.
    mkdir -p '$FS_MOUNT/.runtimes'
    export VLLM_EXL3_ONLINE_CACHE_DIR='$FS_MOUNT/.runtimes/exl3-online'
    export VLLM_EXL3_ONLINE_CACHE_MODE=readwrite
    # AIBeast's selected runtime (maintenance glm53-optimization-20260908):
    # prefill fairness at a 60% compute share, with the tuned batching shape
    # measured for mixed interactive/long-prefill traffic.
    export PREFILL_FAIRNESS_ENGINE=compute_share
    export PREFILL_COMPUTE_SHARE=0.6
    export MAX_NUM_SEQS=12
    export MAX_NUM_BATCHED_TOKENS=3072
    export VLLM_EXL3_PREFILL_CAPACITY=2048
    export GPU_MEMORY_UTILIZATION=0.95
    export PREFIX_CACHE_BACKEND=lmcache
    export LMCACHE_L1_MAX_GB=125
    export PREFIX_CACHE_DISK_GB=384
    # 12 seqs x (1 + 3 MTP) = 48 decode tokens per step: the capture and
    # trellis windows must be raised together or decode silently leaves the
    # captured fast path under concurrency (glm_config rule concurrency-window).
    export CUDAGRAPH_CAPTURE_SIZES=4,8,12,16,20,24,28,32,36,40,44,48
    export MAX_CUDAGRAPH_CAPTURE_SIZE=48
    export VLLM_EXL3_TRELLIS_MAX_M=48
    bash /root/quickstart.sh
  " 2>&1 | tee /tmp/fleet-"$name".log | tail -40
  log "$name serving; total time-to-serve $((SECONDS - t0))s (see /tmp/fleet-$name.log for the endpoint block)"
}

# --- router -------------------------------------------------------------------
router_config() { # prints a litellm config.yaml covering every serve instance
  local state name mid ip reply api_url api_key cfg=""
  state=$(state_load)
  while read -r name mid; do
    [[ -n "$mid" ]] || continue
    ip=$(instance_ip "$mid") || continue
    # Ask the instance for its own verified proxy URL and API key: it knows
    # its MACHINE_ID/DNS env, and the probe proves the URL before we route
    # production traffic at it.
    # </dev/null: ssh must not consume the while-read loop's stdin.
    reply=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no "root@$ip" '
      key=$(cat /home/turnkey/workspace/.glm-config/.vllm-api-key 2>/dev/null || cat /home/jl_fs/.vllm-api-key 2>/dev/null)
      host=${DNS#https://}
      for i in 0 1 2 3 4 5; do
        u="https://$(hostname | cut -c1-6)${MACHINE_ID}${i}.${host}"
        code=$(curl --max-time 10 -s -o /dev/null -w "%{http_code}" \
          "$u/v1/models" -H "Authorization: Bearer $key" 2>/dev/null || true)
        [ "$code" = "200" ] && echo "$u $key" && exit 0
      done
      exit 1
    ' </dev/null) || { log "skipping $name: no verified proxy URL"; continue; }
    read -r api_url api_key <<<"$reply"
    # $() strips the trailing newline, so append one explicitly: entries must
    # not glue to the following YAML mapping.
    cfg+="$(printf '  - model_name: GLM-5.3\n    litellm_params:\n      model: openai/GLM-5.3\n      api_base: %s/v1\n      api_key: %s\n' "$api_url" "$api_key")"$'\n'
  done < <(printf '%s' "$state" | python3 -c "
import json, sys
for k, v in sorted(json.load(sys.stdin).items()):
    if k.startswith('glm53-serve-') and v: print(k, v)
")
  [[ -n "$cfg" ]] || fatal "no serve instances in $FLEET_STATE; run 'serve' first"
  printf 'model_list:\n%s' "$cfg"
  # LiteLLM session stickiness: requests from the same client key (or with a
  # session id) keep landing on the replica whose prefix cache already holds
  # their conversation; the shuffle only spreads NEW sessions.
  printf 'router_settings:\n'
  printf '  routing_strategy: simple-shuffle\n'
  printf '  model_group_affinity_config:\n'
  printf '    GLM-5.3:\n'
  printf '      - deployment_affinity\n'
  printf '      - session_affinity\n'
  printf '  deployment_affinity_ttl_seconds: 3600\n'
  # The router's public port must not be an open relay: every request needs
  # the master key (persisted on the router, printed by cmd_router).
  printf 'general_settings:\n  master_key: os.environ/LITELLM_MASTER_KEY\n'
}

router_tls() { # router_tls <ip>: with DESEC_TOKEN + DESEC_DOMAIN exported,
               # register glm53-router.<zone> -> <ip> and issue a Let's
               # Encrypt certificate via lego DNS-01 (the appliance's
               # deSEC path, reused on the router VM). No-op otherwise.
  [[ -n "${DESEC_TOKEN:-}" && -n "${DESEC_DOMAIN:-}" ]] || return 0
  local ip="$1" sub="glm53-router"
  log "TLS: registering ${sub}.${DESEC_DOMAIN} -> $ip and issuing a certificate"
  local remote_cmd
  remote_cmd=$(printf 'IP=%q ZONE=%q TOKEN=%q EMAIL=%q bash -s' \
    "$ip" "$DESEC_DOMAIN" "$DESEC_TOKEN" "${ACME_EMAIL:?ACME_EMAIL must be set for certificate issuance}")
  scp -q "$(dirname "${BASH_SOURCE[0]}")/desec_acme_guard.py" \
    "$ROUTER_USER@$ip:router/desec_acme_guard.py" 2>/dev/null || \
    log "TLS: could not ship desec_acme_guard.py; lego runs without the guard"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ROUTER_USER@$ip" "$remote_cmd" <<'REMOTE' ||
set -e
mkdir -p "$HOME/router/bin"
if [ ! -x "$HOME/router/bin/lego" ]; then
  curl -fsSL "https://github.com/go-acme/lego/releases/download/v5.4.1/lego_5.4.1_linux_amd64.tar.gz" \
    | tar -xz -C "$HOME/router/bin" lego
  curl -fsSL "https://github.com/go-acme/lego/releases/download/v5.4.1/lego_5.4.1_checksums.txt" \
    | grep linux_amd64.tar.gz | sha256sum -c --status - || { echo BAD-LEGO-CHECKSUM; exit 1; }
fi
# A record for the router name (ttl 3600 is deSEC's account minimum). The
# token travels in a 0600 header file, never argv.
hdr=$(umask 077 && mktemp)
printf 'Authorization: Token %s\n' "$TOKEN" > "$hdr"
curl -sf -X PUT "https://desec.io/api/v1/domains/$ZONE/rrsets/" \
  -H @"$hdr" -H "Content-Type: application/json" \
  -d "[{\"subname\":\"glm53-router\",\"type\":\"A\",\"ttl\":3600,\"records\":[\"$IP\"]}]" >/dev/null
rm -f "$hdr"
# The deSEC guard (same one the appliance runs beside lego) repairs the
# transient authoritative split observed in live DNS-01 issuances.
pip3 install -q --break-system-packages dnspython 2>/dev/null || pip3 install -q dnspython
python3 "$HOME/router/desec_acme_guard.py" --zone "$ZONE" \
  --domain "glm53-router.$ZONE" --timeout 300 &
guard=$!
DESEC_TOKEN="$TOKEN" "$HOME/router/bin/lego" --path "$HOME/router/lego" \
  --server https://acme-v02.api.letsencrypt.org/directory \
  --email "$EMAIL" --dns desec --domains "glm53-router.$ZONE" \
  --accept-tos --dns.propagation-wait 45s run
kill $guard 2>/dev/null || true
ls "$HOME/router/lego/certificates/"
REMOTE
  log "TLS: issuance failed; continuing with plain HTTP on :$ROUTER_PORT"
}

cmd_router() {
  require_jl
  local t0=$SECONDS
  local mid
  mid=$(create_instance glm53-router --cpu --vm --vcpus "$ROUTER_VCPUS" \
        --ram "$ROUTER_RAM_GB" --region "$FS_REGION")
  state_set router "$mid"
  local ip
  ip=$(instance_ip "$mid")
  wait_ssh_router "$ip"
  log "router $mid up ($ip); installing LiteLLM"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ROUTER_USER@$ip" '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v pip3 >/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip >/dev/null
    fi
    pip3 install -q "litellm[proxy]==1.100.0" --break-system-packages 2>/dev/null \
      || sudo pip3 install -q "litellm[proxy]==1.100.0"
    mkdir -p "$HOME/router"
    if [ ! -s "$HOME/router/master-key" ]; then
      python3 -c "import secrets; print(\"sk-router-\" + secrets.token_hex(24))" > "$HOME/router/master-key"
    fi
  '
  router_tls "$ip"
  # Re-generate the config from live fleet state and ship it, then start.
  regenerate_router_config "$ip"
  router_restart "$ip"
  local rkey
  rkey=$(ssh -o BatchMode=yes "$ROUTER_USER@$ip" 'cat $HOME/router/master-key')
  # CPU VMs have no JarvisLabs HTTPS proxy; probe the public IP directly.
  # With a certificate issued, litellm serves TLS on 443 under the
  # glm53-router.<zone> name; otherwise plain HTTP on ROUTER_PORT.
  local probe url
  if ssh -o BatchMode=yes "$ROUTER_USER@$ip" 'ls $HOME/router/lego/certificates/glm53-router.*.crt >/dev/null 2>&1'; then
    url="https://glm53-router.${DESEC_DOMAIN:-}:${ROUTER_TLS_PORT:-443}/v1"
  else
    url="http://$ip:$ROUTER_PORT/v1"
  fi
  probe=$(curl --max-time 15 -s -o /dev/null -w '%{http_code}' "${url%/v1}/health/liveliness" 2>/dev/null || true)
  if [[ "$probe" == "200" ]]; then
    log "router ready in $((SECONDS - t0))s at $url (publicly reachable)"
    log "router master key (send as 'Authorization: Bearer <key>'): $rkey"
  else
    log "router ready in $((SECONDS - t0))s but the probe failed (${probe:-no answer}): $url"
    log "master key: $rkey"
    log "if the port is unreachable, use an SSH tunnel: ssh -L $ROUTER_PORT:localhost:$ROUTER_PORT $ROUTER_USER@$ip"
  fi
}

regenerate_router_config() { # regenerate_router_config <router-ip>: write
                            # atomically so litellm never reads a half-written
                            # config, and never echo the api_key values.
  local ip="$1" cfg
  cfg=$(router_config) || fatal "could not build router config"
  printf '%s\n' "$cfg" | ssh -o BatchMode=yes "$ROUTER_USER@$ip" \
    'cat > $HOME/router/config.yaml.new && mv $HOME/router/config.yaml.new $HOME/router/config.yaml'
  log "router config updated:"
  printf '%s\n' "$cfg" | sed 's/^\( *api_key:\).*/\1 <redacted>/' >&2
}

router_restart() { # router_restart <ip>: (re)start litellm under systemd so it
                   # survives ssh sessions; the VM has no user lingering.
  ssh -o BatchMode=yes "$ROUTER_USER@$1" '
    set -e
    mkdir -p $HOME/router
    if [ ! -s "$HOME/router/master-key" ]; then
      python3 -c "import secrets; print(\"sk-router-\" + secrets.token_hex(24))" > "$HOME/router/master-key"
    fi
    sudo systemctl stop litellm-router.service 2>/dev/null || true
    tls_args=""
    port='"$ROUTER_PORT"'
    crt=$HOME/router/lego/certificates/glm53-router.*.crt
    key=$HOME/router/lego/certificates/glm53-router.*.key
    cap_props=""
    if ls $crt >/dev/null 2>&1 && ls $key >/dev/null 2>&1; then
      # Serve the issued certificate on the TLS port. Binding 443 as a
      # non-root unit needs the ambient bind capability.
      tls_args="--ssl_certfile_path $(ls $crt | head -1) --ssl_keyfile_path $(ls $key | head -1)"
      port='"${ROUTER_TLS_PORT:-443}"'
      cap_props="--property=AmbientCapabilities=CAP_NET_BIND_SERVICE --property=CapabilityBoundingSet=CAP_NET_BIND_SERVICE"
    fi
    sudo systemd-run --uid='"$ROUTER_USER"' --unit=litellm-router --collect \
      --working-directory=$HOME/router --setenv=HOME=$HOME \
      --setenv=LITELLM_MASTER_KEY=$(cat $HOME/router/master-key) \
      $cap_props \
      $(command -v litellm) --config $HOME/router/config.yaml \
      --port $port --host 0.0.0.0 $tls_args
    sleep 15
    curl -sk --max-time 10 https://127.0.0.1:$port/health/liveliness \
      || curl -s --max-time 10 http://127.0.0.1:$port/health/liveliness
    echo "ROUTER_LIVE"
    # Final gate: a unit that crashed after start fails the whole restart.
    systemctl is-active --quiet litellm-router
  '
}

# --- scale / status / teardown -------------------------------------------------
cmd_scale() {
  # First unused glm53-serve-<letter> slot in the local state (a..z) instead
  # of a hardcoded name: repeated scale calls each get their own slot, and
  # the letters never collide with the manager's glm53-serve-<N> slots.
  local letter
  letter=$(state_load | python3 -c "
import json, sys
taken = {k for k, v in json.load(sys.stdin).items() if v}
for c in 'abcdefghijklmnopqrstuvwxyz':
    if 'glm53-serve-' + c not in taken:
        print(c)
        break
")
  [[ -n "$letter" ]] || fatal "no free glm53-serve-<letter> slot in a..z"
  cmd_serve "glm53-serve-$letter"
  local rip
  rip=$(instance_ip "$(state_get router)")
  if [[ -n "$rip" ]]; then
    regenerate_router_config "$rip"
    router_restart "$rip"
  else
    log "no router running; start one with: $0 router"
  fi
}

cmd_manager() { # install the autonomous fleet manager on the router VM
  require_jl
  local rip
  rip=$(instance_ip "$(state_get router)")
  [[ -n "$rip" ]] || fatal "no router VM in $FLEET_STATE; run '$0 router' first."
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # The fleet-wide replica key: minted once, persisted in the local state.
  local fleet_key
  fleet_key=$(state_get fleet_key)
  if [[ -z "$fleet_key" ]]; then
    fleet_key="sk-fleet-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    state_set fleet_key "$fleet_key"
  fi

  # Register the autonomous serve script (secret baked in at registration).
  # Idempotent: update the existing glm53-fleet-serve registration in place
  # when there is one (jarvislabs ids go stale otherwise), add it otherwise.
  # The rendered file holds the fleet key, so trap it away on every path.
  local rendered script_id
  rendered=$(mktemp)
  chmod 600 "$rendered"
  trap "rm -f '$rendered'" EXIT
  sed "s|\${GLM_FLEET_KEY:?GLM_FLEET_KEY must be set by the registration step}|$fleet_key|g" \
    "$script_dir/fleet_serve_script.sh" > "$rendered"
  grep -q "$fleet_key" "$rendered" || fatal "fleet key substitution did not match"
  script_id=$(jl scripts list --json 2>/dev/null | python3 -c "
import json, sys
doc = json.load(sys.stdin)
rows = doc if isinstance(doc, list) else doc.get('scripts', [])
for row in rows:
    if row.get('script_name') == 'glm53-fleet-serve':
        print(row.get('script_id') or row.get('id') or '')
        break
")
  if [[ -n "$script_id" ]]; then
    jl scripts update "$script_id" "$rendered" >/dev/null ||
      fatal "could not update startup script $script_id"
    log "startup script updated in place (id $script_id)"
  else
    jl scripts add "$rendered" --name glm53-fleet-serve >/dev/null ||
      fatal "could not register the startup script"
    # The add response's JSON shape is not stable across CLI versions;
    # resolve the id by name from the authoritative list instead.
    script_id=$(jl scripts list --json 2>/dev/null | python3 -c "
import json, sys
doc = json.load(sys.stdin)
rows = doc if isinstance(doc, list) else doc.get('scripts', [])
for row in rows:
    if row.get('script_name') == 'glm53-fleet-serve':
        print(row.get('script_id') or row.get('id') or '')
        break
")
    [[ -n "$script_id" ]] || fatal "registered but could not resolve the new script id"
    log "startup script registered (id $script_id)"
  fi

  # Ship the manager + config; the VM needs its own jl CLI and credentials.
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ROUTER_USER@$rip" '
    set -e
    mkdir -p "$HOME/fleet-manager" "$HOME/.config/jl"
    if ! command -v jl >/dev/null 2>&1; then
      pip3 install -q --user --break-system-packages jarvislabs 2>/dev/null || pip3 install -q jarvislabs
    fi
  '
  scp -q ~/.config/jl/config.toml "$ROUTER_USER@$rip:.config/jl/config.toml"
  scp -q "$script_dir/fleet_manager.py" "$ROUTER_USER@$rip:fleet-manager/"
  local id
  id=$(fs_id)
  # shellcheck disable=SC2087  # local expansion is the point: the fleet
  # key, filesystem id and script id are workstation-side values.
  ssh -o BatchMode=yes "$ROUTER_USER@$rip" "python3 -" <<EOF
import json, os
cfg = {
    "fleet_key": "$fleet_key",
    "fs_id": "$id",
    "script_id": "$script_id",
    "min_replicas": ${MIN_REPLICAS:-1},
    "max_replicas": ${MAX_REPLICAS:-3},
    "on_demand_min": ${ON_DEMAND_MIN:-0},
    "on_demand_kind": "${ON_DEMAND_KIND:-container}",
    "gpu": "$SERVE_GPU",
    "num_gpus": $SERVE_GPUS,
    "region": "$FS_REGION",
    "poll_seconds": 30,
    "cooldown_seconds": 300,
    "scale_up_waiting": 2,
    "scale_down_idle_seconds": 900,
    "scale_down_concurrency": 4,
    "unhealthy_grace_seconds": 300,
    "capacity_per_replica": 12,
    "router_api": ${ROUTER_API:-False},
    "router_url": "${ROUTER_URL:-https://glm53-router.${DESEC_DOMAIN:-malaiwah.dedyn.io}}",
}
# Hot-reload mode needs the master key for the admin API; it lives on the
# router VM only, so read it there. Absent key disables API mode safely.
_mk = os.path.expanduser("~/router/master-key")
if cfg["router_api"] and os.path.exists(_mk):
    cfg["master_key"] = open(_mk).read().strip()
elif cfg["router_api"]:
    cfg["router_api"] = False
    print("WARNING: ROUTER_API requested but ~/router/master-key is absent;"
          " falling back to config-rewrite mode")
path = os.path.expanduser("~/fleet-manager/fleet.json")
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
os.chmod(path, 0o600)
print("config written")
EOF
  ssh -o BatchMode=yes "$ROUTER_USER@$rip" '
    sudo systemctl stop fleet-manager.service 2>/dev/null || true
    sudo systemd-run --uid='"$ROUTER_USER"' --unit=fleet-manager --collect \
      --setenv=HOME=$HOME --working-directory=$HOME/fleet-manager \
      --setenv=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin \
      /usr/bin/python3 $HOME/fleet-manager/fleet_manager.py
    sleep 3
    systemctl is-active fleet-manager
  '
  log "fleet manager running (min=${MIN_REPLICAS:-1} max=${MAX_REPLICAS:-3});"
  log "watch it with: ssh $ROUTER_USER@$rip journalctl -u fleet-manager -f"
}

cmd_status() {
  require_jl
  jl list
  echo
  jl filesystem list
}

cmd_teardown() { # cmd_teardown [--all]: destroy replica slots + state-file
                 # instances. The router VM is destroyed only with --all: its
                 # LE certificate, master key and fleet.json exist nowhere
                 # else, and Let's Encrypt allows 5 duplicate certs/week.
  require_jl
  # Destroy every LIVE glm53-serve-* slot first, enumerated from the API —
  # the local state file does not know about slots another workstation (or
  # the manager) created.
  local mid
  while read -r mid; do
    [[ -n "$mid" ]] || continue
    log "destroying serve slot $mid"
    jl destroy "$mid" --yes >/dev/null 2>&1 || log "could not destroy $mid"
  done < <(jl list --json 2>/dev/null | python3 -c "
import json, sys
for row in json.load(sys.stdin):
    if str(row.get('name') or '').startswith('glm53-serve-'):
        print(row.get('machine_id') or row.get('mid') or '')
")
  # Then the state file's instances. Skip 'fleet_key' (a secret, not a
  # machine id) and 'router' (kept unless --all).
  local state name
  state=$(state_load)
  while read -r name mid; do
    [[ -n "$mid" ]] || continue
    log "destroying $name ($mid)"
    jl destroy "$mid" --yes >/dev/null 2>&1 || log "could not destroy $mid"
  done < <(printf '%s' "$state" | python3 -c "
import json, sys
for k, v in sorted(json.load(sys.stdin).items()):
    if k not in ('fleet_key', 'router') and v:
        print(k, v)
")
  local rmid
  rmid=$(state_get router)
  if [[ "${1:-}" == "--all" && -n "$rmid" ]]; then
    log "destroying router ($rmid)"
    jl destroy "$rmid" --yes >/dev/null 2>&1 || log "could not destroy $rmid"
  else
    log "router kept (the LE certificate, master key and fleet.json live only on it; use 'teardown --all' to destroy it too)"
  fi
  echo '{}' > "$FLEET_STATE"
  log "fleet instances destroyed; filesystem kept (jl filesystem remove to delete it)"
}

main() {
  case "${1:-}" in
    populate) shift; cmd_populate "$@" ;;
    serve)    shift; cmd_serve "${1:-}" ;;
    router)   shift; cmd_router "$@" ;;
    scale)    shift; cmd_scale "$@" ;;
    manager)  shift; cmd_manager "$@" ;;
    status)   shift; cmd_status "$@" ;;
    teardown) shift; cmd_teardown "$@" ;;
    *) cat >&2 <<USAGE
Usage: jarvislabs_fleet.sh <command>

  populate   fill the persistent filesystem with GLM-5.3 weights (1-GPU spot,
             auto-destroyed when done)
  serve      launch a 4-GPU spot instance that boots from the warm filesystem
             (repeatable; each instance is independent)
  router     launch the CPU-VM LiteLLM router with prefix-cache affinity
  scale      second serve instance + router pick-up
  status     list fleet instances and the filesystem
  teardown   destroy live serve slots + state-file instances; the router VM is
             kept (its LE certificate, master key and fleet.json exist only
             there). 'teardown --all' also destroys the router. The
             filesystem is always kept

Environment: FS_NAME, FS_SIZE_GB, POPULATOR_GPU, SERVE_GPU, SERVE_GPUS, ROUTER_PORT
USAGE
      exit 2 ;;
  esac
}

main "$@"
