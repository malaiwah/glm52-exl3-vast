#!/usr/bin/env bash
# Fail-closed runner behavior, using only temporary fixtures (no host writes).
# Fixture programs expand variables in their child shell, not in this test.
# shellcheck disable=SC2016
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/scripts/jarvislabs_container_rootfs.sh"
DIGEST=ghcr.io/malaiwah/glm52-exl3-vast@sha256:200b1841453b6a46c91f0b7a2866589cda7e52625b29590bb7a69fe90948e6c9
FAILURES=0
TMP="$(mktemp -d)"
trap 'rm -rf -- "$TMP"' EXIT

check_status() {
  local name="$1" expected="$2" status=0
  shift 2
  "$@" >"$TMP/output" 2>&1 || status=$?
  if [[ "$status" == "$expected" ]]; then
    printf '  ok   %s\n' "$name"
  else
    printf '  FAIL %s (expected %s, got %s)\n' "$name" "$expected" "$status"
    cat "$TMP/output"
    FAILURES=$((FAILURES + 1))
  fi
}

run() { env -i PATH="$PATH" HOME="$HOME" "$@"; }

check_status 'unknown subcommand is rejected' 2 run bash "$SCRIPT" frobnicate
check_status 'missing image reference is rejected' 1 run bash "$SCRIPT" prepare
check_status 'floating tag is rejected' 2 run TURNKEY_IMAGE=example/model:latest bash "$SCRIPT" prepare
check_status 'filesystem root is rejected before mutation' 2 run TURNKEY_IMAGE="$DIGEST" TURNKEY_ROOT=/ bash "$SCRIPT" prepare
if [[ "$(id -u)" != 0 ]]; then
  check_status 'non-root invocation is rejected' 2 run TURNKEY_IMAGE="$DIGEST" bash "$SCRIPT" prepare
fi

# Source functions without invoking main. Each fixture runs in its own shell so
# fatal exits, producer failures and errexit semantics are exercised normally.
fixture() {
  env -i PATH="$PATH" HOME="$HOME" SCRIPT="$SCRIPT" TMP="$TMP" DIGEST="$DIGEST" \
    bash -c 'source "$SCRIPT"; ROOT="$TMP/appliance"; ROOTFS="$ROOT/bundle/rootfs"; IMAGE="$DIGEST"; eval "$1"' bash "$1"
}

mkdir -p "$TMP/appliance/bundle/rootfs" "$TMP/user-data" "$TMP/other"
printf 'keep\n' >"$TMP/user-data/payload"
ln -s "$TMP/user-data" "$TMP/alias"
ln -s "$TMP/appliance/workspace" "$TMP/matching"

check_status 'canonical dedicated root is accepted' 0 fixture 'validate_root'
check_status 'symlink root is refused' 2 fixture 'ROOT="$TMP/alias"; validate_root'
check_status 'noncanonical root is refused' 2 fixture 'ROOT="$TMP/appliance/../other"; validate_root'
check_status 'system directory is refused' 2 fixture 'ROOT=/usr/local; validate_root'
check_status 'graft requires explicit opt-in' 2 fixture 'TURNKEY_GRAFT=0; container_marker_present() { return 0; }; require_container_graft'
check_status 'bare host is refused even with opt-in' 2 fixture 'TURNKEY_GRAFT=1; container_marker_present() { return 1; }; require_container_graft'
check_status 'opted-in container passes guard' 0 fixture 'TURNKEY_GRAFT=1; container_marker_present() { return 0; }; require_container_graft'
check_status 'existing directory is preserved by refusal' 2 fixture 'require_matching_link "$TMP/user-data" "$ROOT/workspace"'
check_status 'unrelated symlink is preserved by refusal' 2 fixture 'require_matching_link "$TMP/alias" "$ROOT/workspace"'
check_status 'matching symlink is accepted' 0 fixture 'require_matching_link "$TMP/matching" "$ROOT/workspace"'
check_status 'absent target is accepted' 0 fixture 'require_matching_link "$TMP/absent" "$ROOT/workspace"'
check_status 'user data survived conflict checks' 0 cmp "$TMP/user-data/payload" <(printf 'keep\n')
check_status 'unrelated symlink survived conflict check' 0 test -L "$TMP/alias"
check_status 'purge outside unpack trees is refused' 2 fixture 'purge "$TMP/user-data"'
check_status 'mounted subtree is refused' 2 fixture 'findmnt() { printf "%s\n" "$ROOTFS/dev"; }; purge "$ROOT/bundle"'
check_status 'graft in-use rootfs is refused even without mounts' 2 fixture 'findmnt() { printf "/\n"; }; graft_present() { return 0; }; purge "$ROOT/bundle"'
check_status 'stop refuses graft process shutdown before unmount' 2 fixture 'require_root() { :; }; graft_present() { return 0; }; unmount_runtime() { exit 91; }; TURNKEY_ROOT="$ROOT"; main stop'
check_status 'mount inspection failure is refused' 2 fixture 'findmnt() { return 1; }; require_unmounted_tree "$ROOT/bundle"'
check_status 'sibling mount does not block a tree' 0 fixture 'findmnt() { printf "%s\n" "$ROOT/bundle-other/dev"; }; require_unmounted_tree "$ROOT/bundle"'
check_status 'runtime workspace alias is refused' 2 fixture 'WORKSPACE=/workspace; validate_workspace'
check_status 'workspace inside unpack tree is refused' 2 fixture 'WORKSPACE="$ROOTFS/checkpoint"; validate_workspace'
check_status 'dedicated workspace is accepted' 0 fixture 'WORKSPACE="$ROOT/workspace"; validate_workspace'

ln -s "$TMP/user-data" "$TMP/appliance/oci"
check_status 'symlink unpack tree is refused' 2 fixture 'purge "$ROOT/oci"'
rm "$TMP/appliance/oci"
printf '%s\n' "$DIGEST" >"$TMP/appliance/bundle/rootfs/.unpacked"
check_status 'matching image reuses unpack without tools or writes' 0 fixture 'skopeo() { exit 91; }; purge() { exit 92; }; fetch_and_unpack'
check_status 'different image cannot replace existing rootfs' 2 fixture 'IMAGE="example/model@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; fetch_and_unpack'
check_status 'unpacked image cannot be purged' 2 fixture 'findmnt() { printf "/\n"; }; purge "$ROOT/bundle"'
check_status 'entry refuses an unverified image' 2 fixture 'IMAGE=wrong; enter'
check_status 'entry stops on verification failure before launch' 43 fixture 'graft_present() { return 1; }; mounts_available() { return 0; }; verify_rootfs() { return 43; }; chroot() { exit 91; }; enter'

printf '{broken\n' >"$TMP/appliance/bundle/config.json"
check_status 'image parser failure propagates through loader' 2 fixture 'declare -a env_args=(); load_env image_env'
check_status 'partial producer output cannot hide failure' 2 fixture 'image_env() { printf "PATH=/usr/bin\n"; return 1; }; declare -a env_args=(); load_env image_env'
check_status 'appliance producer propagates image failure' 2 fixture 'image_env() { printf "PATH=/usr/bin\n"; return 1; }; declare -a env_args=(); load_env appliance_env'

mkdir -p "$TMP/driver-a" "$TMP/driver-b"
printf 'first driver\n' >"$TMP/driver-a/libcuda.so.1"
printf 'different driver\n' >"$TMP/driver-b/libcuda.so.1"
check_status 'conflicting driver basenames are rejected' 2 fixture 'require_unmounted_tree() { :; }; ldconfig() { printf "libcuda.so.1 => %s\n" "$TMP/driver-a/libcuda.so.1" "$TMP/driver-b/libcuda.so.1"; }; inject_driver'
check_status 'driver copy errors are not suppressed' 57 fixture 'require_unmounted_tree() { :; }; ldconfig() { printf "libcuda.so.1 => %s\n" "$TMP/driver-a/libcuda.so.1"; }; cp() { return 57; }; inject_driver'

if [[ "$FAILURES" != 0 ]]; then
  printf '%s check(s) failed\n' "$FAILURES" >&2
  exit 1
fi
echo 'all container rootfs contract checks passed'
