#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/model"
printf '{}\n' >"$tmp/model/config.json"
printf 'complete\n' >"$tmp/download-complete"
cat >"$tmp/bin/podman" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$PODMAN_TEST_LOG"
case "$1 $2" in
  'container exists')
    if [[ "$3" == rollback ]]; then exit "${ROLLBACK_EXISTS_STATUS:-1}"; fi
    exit "${EXISTS_STATUS:-1}" ;;
  'image exists') exit "${IMAGE_EXISTS_STATUS:-0}" ;;
esac
case "$1" in
  run)
    printf '%s\n' "$@" >"$PODMAN_TEST_ARGS"
    printf 'fake-container-id\n' ;;
  stop) exit "${STOP_STATUS:-0}" ;;
  rename) ;;
  *) echo "unexpected operation: $*" >&2; exit 2 ;;
esac
SH
cat >"$tmp/bin/nvidia-ctk" <<'SH'
#!/usr/bin/env bash
printf 'discovery\n' >>"$GPU_DISCOVERY_LOG"
printf '%s\n' nvidia.com/gpu=0 nvidia.com/gpu=1 nvidia.com/gpu=2 nvidia.com/gpu=3
SH
chmod +x "$tmp/bin/podman" "$tmp/bin/nvidia-ctk"
export PATH="$tmp/bin:$PATH" MODEL_DIR_HOST="$tmp/model"
export DOWNLOAD_MARKER_HOST="$tmp/download-complete" NAME=candidate
export PODMAN_TEST_ARGS="$tmp/args" PODMAN_TEST_LOG="$tmp/log"
export GPU_DISCOVERY_LOG="$tmp/gpu-log"

launch() { bash "$repo/scripts/run-local-podman.sh" >"$tmp/stdout" 2>"$tmp/stderr"; }
reject() {
  if launch; then echo 'unsafe launch unexpectedly succeeded' >&2; exit 1; fi
}
no_mutation() {
  if grep -Eq '^(stop|rm|rename|run) ' "$tmp/log"; then
    echo 'unexpected container mutation' >&2; cat "$tmp/log" >&2; exit 1
  fi
}
reset_log() { : >"$tmp/log"; rm -f "$tmp/args" "$tmp/gpu-log"; }

# Default collisions must not stop, remove, rename, or replace the container.
for smoke in 0 1; do
  reset_log
  CONFIG_SMOKE="$smoke" EXISTS_STATUS=0 reject
  no_mutation
  if [[ "$smoke" == 1 ]]; then test ! -e "$tmp/gpu-log"; fi
done
# Smoke must never replace even when the real-launch opt-in is present.
reset_log
CONFIG_SMOKE=1 EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback reject
no_mutation
# Podman failure must not be mistaken for an available name.
reset_log
CONFIG_SMOKE=1 EXISTS_STATUS=125 reject
no_mutation
# Missing image may not be implicitly pulled, nor stop an existing candidate.
reset_log
EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback IMAGE_EXISTS_STATUS=1 reject
no_mutation
# GPU-free smoke works without discovering devices, and never downloads.
reset_log
CONFIG_SMOKE=1 launch
test ! -e "$tmp/gpu-log"
grep -Fxq -- '--pull=never' "$tmp/args"
if grep -Eq -- '^--(gpus|device|replace|privileged)(=|$)' "$tmp/args"; then exit 1; fi
# Podman 4.9-compatible CDI selection preserves physical TP rank order.
reset_log
CONFIG_SMOKE=0 GPU_DEVICE_MODE=cdi GPU_DEVICES=2,1,0,3 launch
grep -Fxq 'CUDA_VISIBLE_DEVICES=2,1,0,3' "$tmp/args"
for gpu in 2 1 0 3; do grep -Fxq "nvidia.com/gpu=$gpu" "$tmp/args"; done
if grep -Eq -- '^--(gpus|replace|privileged)(=|$)' "$tmp/args"; then exit 1; fi
# A preflight checks dependencies without changing any container.
reset_log
LAUNCH_PREFLIGHT=1 GPU_DEVICE_MODE=cdi launch
no_mutation
# A replacement is allowed only with an unused rollback identity.
reset_log
EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback ROLLBACK_EXISTS_STATUS=0 reject
no_mutation
reset_log
EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback ROLLBACK_EXISTS_STATUS=125 reject
no_mutation
reset_log
EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback STOP_STATUS=1 reject
if grep -Eq '^(rm|rename|run) ' "$tmp/log"; then exit 1; fi
reset_log
EXISTS_STATUS=0 REPLACE_EXISTING=1 ROLLBACK_NAME=rollback launch
grep -Fxq 'stop -t 120 candidate' "$tmp/log"
grep -Fxq 'rename candidate rollback' "$tmp/log"
if grep -Eq '^rm |--replace' "$tmp/log"; then exit 1; fi

echo 'local Podman safety and device behavior: PASS'
