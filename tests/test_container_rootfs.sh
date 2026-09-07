#!/usr/bin/env bash
# Fail-closed contract of the container-only rootfs runner.
#
# The runner unpacks and chroots a published image on hosts that cannot run a
# container runtime. Everything it refuses here is refused before it touches
# the network, the registry, or a GPU.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/jarvislabs_container_rootfs.sh"
DIGEST=ghcr.io/malaiwah/glm52-exl3-vast@sha256:200b1841453b6a46c91f0b7a2866589cda7e52625b29590bb7a69fe90948e6c9
FAILURES=0

check() {
  local name="$1" cond="$2" detail="${3:-}"
  if [[ "$cond" == 0 ]]; then
    printf '  ok   %s\n' "$name"
  else
    printf '  FAIL %s %s\n' "$name" "$detail"
    FAILURES=$((FAILURES + 1))
  fi
}

run() { env -i PATH="$PATH" HOME="$HOME" "$@" 2>&1; }

echo "=== refusals happen before any registry access ==="

out="$(run bash "$SCRIPT" frobnicate || true)"
check "an unknown subcommand is rejected" \
  "$([[ "$out" == *"usage:"* ]] && echo 0 || echo 1)" "$out"

out="$(run bash "$SCRIPT" prepare || true)"
check "a missing image reference is rejected" \
  "$([[ "$out" == *TURNKEY_IMAGE* ]] && echo 0 || echo 1)" "$out"

out="$(run TURNKEY_IMAGE=ghcr.io/malaiwah/glm52-exl3-vast:latest bash "$SCRIPT" prepare || true)"
check "a floating tag is rejected in favour of a digest" \
  "$([[ "$out" == *"repository@sha256:digest"* ]] && echo 0 || echo 1)" "$out"

if [[ "$(id -u)" != 0 ]]; then
  out="$(run TURNKEY_IMAGE="$DIGEST" bash "$SCRIPT" prepare || true)"
  check "a non-root invocation is rejected" \
    "$([[ "$out" == *"run this as root"* ]] && echo 0 || echo 1)" "$out"
else
  echo "  skip a non-root invocation is rejected (running as root)"
fi

echo "=== the pinned unpack toolchain is explicit ==="

source_text="$(cat "$SCRIPT")"
check "umoci is pinned by version and hash" \
  "$([[ "$source_text" == *"UMOCI_SHA256=b51c267ec394499e42c6fde47f240b7b7dba57ea49df0b5acd304378b82a3b71"* ]] && echo 0 || echo 1)"
check "the unpacked tree is verified against the image's own provenance" \
  "$([[ "$source_text" == *"unpacked rootfs differs from the built image"* ]] && echo 0 || echo 1)"

if [[ "$FAILURES" != 0 ]]; then
  printf '%s check(s) failed\n' "$FAILURES" >&2
  exit 1
fi
echo "all container rootfs contract checks passed"
