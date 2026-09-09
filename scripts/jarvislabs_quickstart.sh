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

  log "stage 1/4: fetching and unpacking the image (a few minutes)"
  bash "$RUNNER" prepare
  log "stage 2/4: installing the appliance over this container and verifying sources"
  bash "$RUNNER" graft
  log "stage 3/4: checking the resolved configuration (no GPU involved)"
  MODEL_PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}" SSHD=0 \
    bash "$RUNNER" smoke | tail -40
  log "stage 4/4: launching the appliance in the background"
  local log_file=/home/turnkey/serve.log
  MODEL_PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}" SSHD=0 \
    setsid nohup bash "$RUNNER" run >"$log_file" 2>&1 < /dev/null &
  log "boot log: $log_file"

  local waited=0 health=""
  log "waiting for the endpoint (weights download + model load; typically 25-45 minutes on a fresh instance)"
  while (( waited < 3600 )); do
    health="$(curl --max-time 5 -s -o /dev/null -w '%{http_code}' \
      "http://127.0.0.1:${PORT:-8000}/health" 2>/dev/null || true)"
    [[ "$health" == "200" ]] && break
    if ! kill -0 "$(pgrep -f 'model-turnkey-entry.sh' | head -1)" 2>/dev/null; then
      tail -20 "$log_file" >&2 || true
      fatal "the appliance exited before serving; inspect $log_file."
    fi
    sleep 30; waited=$(( waited + 30 ))
    (( waited % 300 == 0 )) && log "still booting... ${waited}s elapsed (last status: ${health:-none})"
  done
  [[ "$health" == "200" ]] || fatal "endpoint did not become healthy within 60 minutes; inspect $log_file."

  print_summary
}

print_summary() {
  local port="${PORT:-8000}"
  local keyfile key token models
  keyfile="$(for f in /home/turnkey/.vllm-api-key /workspace/../.vllm-api-key; do [[ -r "$f" ]] && { printf '%s' "$f"; break; }; done || true)"
  key=""
  [[ -n "$keyfile" && -r "$keyfile" ]] && key="$(cat "$keyfile")"
  token=""
  [[ -r /workspace/.model-turnkey-landing-token ]] && token="$(cat /workspace/.model-turnkey-landing-token)"
  models="$(curl --max-time 10 -s -H "Authorization: Bearer $key" \
    "http://127.0.0.1:${port}/v1/models" | tr ',' '\n' | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ' || true)"
  cat <<SUMMARY

==================================================================
 GLM-5.3 is serving.
==================================================================
 Endpoint (inside this instance) : http://127.0.0.1:${port}/v1
 Endpoint (from your machine)    : your JarvisLabs proxy URL for port ${port}
                                    (dashboard -> this instance -> port ${port})
 API key                         : ${key:-(see /home/turnkey/.vllm-api-key)}
                                    send it as "Authorization: Bearer <key>"
 Model name                      : ${models:-GLM-3 (see /v1/models)}
 Dashboard (port 1111)           : append ?token=${token:-(cat /workspace/.model-turnkey-landing-token)}
                                    to your port-1111 proxy URL
 Boot log                        : /home/turnkey/serve.log
 Weights / state                 : /home/turnkey/workspace (persistent volume)

 Try it:
   curl http://127.0.0.1:${port}/v1/chat/completions \\
     -H "Authorization: Bearer $key" \\
     -H 'Content-Type: application/json' \\
     -d '{"model":"GLM-5.3","messages":[{"role":"user","content":"Say hi"}]}'

 Remember: spot instances can be reclaimed at any time, and billing
 continues while the instance exists. When you are done:
   jl destroy <machine-id>
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
