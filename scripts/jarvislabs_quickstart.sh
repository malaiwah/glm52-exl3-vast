#!/usr/bin/env bash
# One-command GLM-5.3 deployment for a JarvisLabs GPU container instance.
#
#   curl -fsSL https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/<commit>/scripts/jarvislabs_quickstart.sh | bash
#
# JarvisLabs container instances have no Docker/Podman daemon, so this wraps
# scripts/jarvislabs_container_rootfs.sh (skopeo fetch, umoci unpack, graft,
# fail-closed provenance verification) and adds only what a first-time human
# needs: :latest is resolved to its immutable digest before anything runs,
# the /workspace layout is prepared, the appliance launches in the background,
# and the script waits until the endpoint is healthy, then prints the endpoint,
# the generated API key, and the dashboard token.
#
# Everything defaults sensibly; every step is overridable through environment:
#   TURNKEY_IMAGE (tag or repo@sha256:digest), MODEL_PROFILE, TURNKEY_ROOT,
#   TURNKEY_WORKSPACE, HF_TOKEN (optional; the default checkpoint is public).
#
# The whole script executes through main() at the bottom so a truncated
# `curl | bash` transfer cannot execute a destructive prefix.
set -Eeuo pipefail

DEFAULT_IMAGE="docker.io/malaiwah/glm52-exl3-vast:latest"
RUNNER_URL_DEFAULT="https://raw.githubusercontent.com/malaiwah/glm52-exl3-vast/main/scripts/jarvislabs_container_rootfs.sh"

log()   { printf '>>> %s\n' "$*"; }
fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 2; }

on_its_own_host() {
  [[ -f /.dockerenv || -f /run/.containerenv ]]
}

require_gpu_container() {
  on_its_own_host || fatal "this quickstart is for JarvisLabs GPU container instances (root inside the instance)."
  [[ "$(id -u)" == 0 ]] || fatal "run as root inside the GPU container."
  command -v nvidia-smi >/dev/null 2>&1 || fatal "no nvidia-smi: this is not a GPU instance."
}

require_runner() {
  # The runner is fetched from a pinned revision when the caller supplies one,
  # never silently from a moving branch.
  RUNNER="${TURNKEY_RUNNER:-}"
  if [[ -z "$RUNNER" ]]; then
    if [[ -f scripts/jarvislabs_container_rootfs.sh ]]; then
      RUNNER="scripts/jarvislabs_container_rootfs.sh"
    else
      log "fetching the rootfs runner"
      curl --fail --show-error --location --retry 4 --retry-all-errors \
        -o /root/rootfs.sh "$RUNNER_URL_DEFAULT"
      RUNNER="/root/rootfs.sh"
    fi
  fi
  [[ -f "$RUNNER" ]] || fatal "rootfs runner not found: $RUNNER"
  bash -n "$RUNNER" || fatal "fetched runner does not parse; refusing to execute it."
}

resolve_image() {
  command -v skopeo >/dev/null 2>&1 || {
    log "installing skopeo"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq skopeo
  }
  local reference="${TURNKEY_IMAGE:-$DEFAULT_IMAGE}"
  if [[ "$reference" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
    IMAGE="$reference"
    log "using pinned image $IMAGE"
    return 0
  fi
  log "resolving $reference to its immutable digest"
  local digest
  digest="$(skopeo inspect --override-os linux --override-arch amd64 \
    "docker://$reference" --format '{{.Digest}}')" ||
    fatal "cannot resolve $reference (registry unreachable or unknown tag)."
  IMAGE="${reference%%:*}@${digest}"
  log "pinned image: $IMAGE"
}

prepare_workspace_layout() {
  # The graft requires /workspace to be a symlink to TURNKEY_WORKSPACE (which
  # must live on the persistent volume). A fresh JarvisLabs container ships an
  # empty real /workspace directory; relocate its contents when present.
  local target="${TURNKEY_WORKSPACE:-/home/turnkey/workspace}"
  [[ "$target" == "$(realpath -m -- "$target")" ]] ||
    fatal "TURNKEY_WORKSPACE must be a canonical absolute path."
  case "$target" in /home/*|/mnt/*|/srv/*|/tmp/*|/var/tmp/*) ;;
    *) fatal "TURNKEY_WORKSPACE must live on the persistent volume (/home/...)." ;;
  esac
  mkdir -p "$target"
  if [[ -L /workspace ]]; then
    [[ "$(realpath -m -- /workspace)" == "$target" ]] ||
      fatal "/workspace already links to $(realpath -- /workspace), not $target; set TURNKEY_WORKSPACE to match."
    return 0
  fi
  if [[ -d /workspace ]]; then
    if [[ -n "$(ls -A /workspace 2>/dev/null)" ]]; then
      log "moving existing /workspace contents into $target"
      findmnt -rn -o TARGET | grep -qx "/workspace" &&
        fatal "/workspace is a live mount; relocate it manually before running the quickstart."
      mv /workspace/* "$target"/ 2>/dev/null || true
      mv /workspace/.[!.]* "$target"/ 2>/dev/null || true
      [[ -z "$(ls -A /workspace 2>/dev/null)" ]] ||
        fatal "/workspace is not empty after relocation; resolve it manually."
    fi
    rmdir /workspace || fatal "cannot remove the empty /workspace directory; resolve it manually."
    log "removed the provider's empty /workspace (contents preserved in $target)"
  fi
  [[ -e /workspace || -L /workspace ]] || ln -s "$target" /workspace
}

launch_and_wait() {
  local root="${TURNKEY_ROOT:-/home/turnkey/qual}"
  export TURNKEY_IMAGE="$IMAGE"
  export TURNKEY_ROOT="$root"
  export TURNKEY_WORKSPACE="${TURNKEY_WORKSPACE:-/home/turnkey/workspace}"
  export TURNKEY_GRAFT=1

  require_fresh_container
  log "stage 1/4: fetching and unpacking the image (a few minutes)"
  bash "$RUNNER" prepare
  relocate_conflicting_opt_trees
  log "stage 2/4: installing the appliance over this container and verifying sources"
  bash "$RUNNER" graft
  log "stage 3/4: checking the resolved configuration (no GPU involved)"
  MODEL_PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}" SSHD=0 \
    bash "$RUNNER" smoke | tail -40
  log "stage 4/4: launching the appliance in the background"
  local log_file=/home/turnkey/serve.log
  mint_landing_token
  MODEL_PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}" SSHD=0 \
    OPEN_BUTTON_TOKEN="${OPEN_BUTTON_TOKEN:-}" \
    setsid nohup bash "$RUNNER" run >"$log_file" 2>&1 < /dev/null &
  log "boot log: $log_file"

  local waited=0 health=""
  log "waiting for the endpoint (weights download + model load; typically 25-45 minutes on a fresh instance)"
  while (( waited < 3600 )); do
    health="$(curl --max-time 5 -s -o /dev/null -w '%{http_code}' \
      "http://127.0.0.1:${PORT:-8000}/health" 2>/dev/null || true)"
    [[ "$health" == "200" ]] && break
    # Give the launch chain (setsid -> runner -> exec entrypoint) a grace
    # period before liveness can be declared failed; the process name only
    # exists after the exec, and an eager check false-positives.
    if (( waited >= 180 )) && ! pgrep -f 'model-turnkey-entry.sh' >/dev/null 2>&1; then
      tail -20 "$log_file" >&2 || true
      fatal "the appliance exited before serving; inspect $log_file."
    fi
    sleep 30; waited=$(( waited + 30 ))
    (( waited % 300 == 0 )) && log "still booting... ${waited}s elapsed (last status: ${health:-none})"
  done
  [[ "$health" == "200" ]] || fatal "endpoint did not become healthy within 60 minutes; inspect $log_file."

  print_summary
}

require_fresh_container() {
  # A graft is one-way: the container's OS layer is replaced. After an image
  # update there is no in-place upgrade path; say so in plain words.
  if [[ -f /.turnkey-grafted || -f /.turnkey-graft-rootfs ]]; then
    fatal "this container already carries an appliance graft (possibly of an older image). Create a fresh instance (jl destroy + jl create) and re-run."
  fi
}

relocate_conflicting_opt_trees() {
  # JarvisLabs pytorch containers ship a provider CUDA stack under /opt
  # (notably /opt/nvidia). The graft preflight refuses to replace a real
  # provider directory, by design; the appliance brings its own CUDA user
  # space, so move colliding provider trees aside instead of failing.
  local rootfs="${TURNKEY_ROOT:-/home/turnkey/qual}/bundle/rootfs" entry name
  [[ -d "$rootfs/opt" ]] || return 0
  for entry in "$rootfs"/opt/*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    if [[ -e "/opt/$name" && ! -L "/opt/$name" ]]; then
      [[ "$name" == "nvidia" ]] || fatal "/opt/$name is a real provider directory the appliance also ships; relocate it manually and re-run."
      mv "/opt/$name" "/root/provider-opt-nvidia-backup"
      log "relocated the provider's /opt/nvidia to /root/provider-opt-nvidia-backup (the appliance ships its own)"
    fi
  done
  return 0
}

mint_landing_token() {
  # The appliance auto-mints a dashboard token only when it detects the
  # provider by hostname (jl-vm-* VMs). JarvisLabs container hostnames do not
  # match, so mint and persist the token here instead; the runner forwards
  # OPEN_BUTTON_TOKEN to the appliance.
  local file="${TURNKEY_WORKSPACE:-/home/turnkey/workspace}/.model-turnkey-landing-token"
  if [[ -z "${OPEN_BUTTON_TOKEN:-}" ]]; then
    if [[ -s "$file" ]]; then
      OPEN_BUTTON_TOKEN="$(cat "$file")"
    else
      OPEN_BUTTON_TOKEN="lp-$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
      ( umask 077; printf '%s\n' "$OPEN_BUTTON_TOKEN" > "$file" )
    fi
    export OPEN_BUTTON_TOKEN
    log "dashboard token ready (persisted at $file)"
  fi
}

print_summary() {
  local port="${PORT:-8000}"
  local keyfile key token models
  keyfile="$(for f in "${TURNKEY_WORKSPACE:-/home/turnkey/workspace}/.vllm-api-key" /workspace/.vllm-api-key; do [[ -r "$f" ]] && { printf '%s' "$f"; break; }; done || true)"
  key=""
  [[ -n "$keyfile" ]] && key="$(cat "$keyfile")"
  local ws="${TURNKEY_WORKSPACE:-/home/turnkey/workspace}"
  token=""
  [[ -r "$ws/.model-turnkey-landing-token" ]] && token="$(cat "$ws/.model-turnkey-landing-token")"
  models="$(curl --max-time 10 -s -H "Authorization: Bearer $key" \
    "http://127.0.0.1:${port}/v1/models" 2>/dev/null | \
    python3 -c 'import json,sys
try:
    print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception:
    pass' || true)"
  # JarvisLabs exposes http-ports only through its HTTPS proxy, never as raw
  # TCP on the public IP. The proxy hostnames embed the first six hostname
  # characters, the machine id, and a per-port index; discover and VERIFY the
  # URLs instead of guessing the index policy.
  local api_url="" dash_url="" i u code
  if [[ -n "${MACHINE_ID:-}" && -n "${DNS:-}" ]]; then
    local host="${DNS#https://}"
    for i in 0 1 2 3 4 5; do
      u="https://$(hostname | cut -c1-6)${MACHINE_ID}${i}.${host}"
      code="$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' \
        "$u/v1/models" -H "Authorization: Bearer $key" 2>/dev/null || true)"
      [[ "$code" == "200" ]] && api_url="$u" && break
    done
    for i in 0 1 2 3 4 5; do
      u="https://$(hostname | cut -c1-6)${MACHINE_ID}${i}.${host}"
      body="$(curl --max-time 10 -s "$u/" 2>/dev/null || true)"
      [[ "$body" == *"GLM-5.3 turnkey"* ]] && dash_url="$u" && break
    done
  fi
  local api_line="" dash_note="" dash_line="" fallback_note=""
  if [[ -n "$api_url" ]]; then
    api_line="$api_url/v1 (verified with your API key)"
    dash_note="verified; token persisted at $ws/.model-turnkey-landing-token"
  else
    api_line="not found: JarvisLabs containers expose ports only via their HTTPS"
    fallback_note="proxy. Open the JarvisLabs dashboard for this instance, or tunnel:
  ssh -L 8000:localhost:8000 root@$(curl --max-time 5 -s https://api.ipify.org 2>/dev/null || echo '<public-ip>')"
  fi
  if [[ -n "$dash_url" ]]; then
    dash_line="$dash_url/?token=$token ($dash_note)"
  else
    dash_line="(dashboard token at $ws/.model-turnkey-landing-token; proxy URL in the JarvisLabs dashboard)"
  fi
  cat <<SUMMARY

==================================================================
 GLM-5.3 is serving.
==================================================================
 Endpoint (from your machine)    : $api_line
 Endpoint (inside the instance)  : http://127.0.0.1:${port}/v1
 API key                         : ${key:-(see ${ws}/.vllm-api-key)}
                                    send it as "Authorization: Bearer <key>"
 Model name                      : ${models:-GLM-5.3 (see /v1/models)}
 Dashboard (from your machine)   : $dash_line
$fallback_note
 Boot log                        : /home/turnkey/serve.log
 Weights / state                 : $ws (persistent volume)

 Try it from your machine:
   curl ${api_url:-http://127.0.0.1:${port}}/v1/chat/completions \\
     -H "Authorization: Bearer $key" \\
     -H 'Content-Type: application/json' \\
     -d '{"model":"GLM-5.3","messages":[{"role":"user","content":"Say hi"}]}'

 Remember: spot instances can be reclaimed at any time, and billing
 continues while the instance exists. When you are done:
   jl destroy ${MACHINE_ID:-<machine-id>}
==================================================================
SUMMARY
}

main() {
  require_gpu_container
  require_runner
  resolve_image
  prepare_workspace_layout
  launch_and_wait
}

main "$@"
