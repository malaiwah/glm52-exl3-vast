#!/bin/bash
# Recover the PyTorch FileBaton sentinel for the SparkInfer PCIe DMA extension.
#
# A compiler killed during image replacement can leave this zero-byte `lock`
# on a persistent /cache volume. FileBaton treats existence as ownership, so a
# later container waits forever even though no process can ever remove it.
#
# This helper is intentionally narrow:
#   * only sparkinfer_pcie_dma_ext/lock is considered;
#   * .ninja_lock and every other extension are untouched;
#   * a process whose cwd or command line references the extension gets a
#     bounded grace period;
#   * owner detection scans THIS container's /proc only, so it is
#     same-PID-namespace only: a compiler in another container that shares this
#     persistent /cache lives in a separate PID namespace, is invisible here,
#     and its LIVE lock would look ownerless. The cross-container safety is a
#     minimum sentinel-age gate (EXT_LOCK_MIN_AGE_S, default 1800s): an
#     apparently-ownerless sentinel younger than that is presumed live and left
#     in place this pass, so a fresh cross-namespace lock is never quarantined;
#   * an ownerless sentinel is renamed, never deleted.
set -u

extensions_root="${1:-${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}}"
extension_dir="$extensions_root/sparkinfer_pcie_dma_ext"
sentinel="$extension_dir/lock"
wait_seconds="${TORCH_EXTENSION_LOCK_WAIT_S:-30}"
preflight_wait_seconds="${TORCH_EXTENSION_PREFLIGHT_WAIT_S:-10}"
# Cross-container safety: an apparently-ownerless sentinel younger than this is
# presumed to be a live FileBaton held by a compiler in another PID namespace
# (a second container on the same persistent /cache) and is left untouched. The
# default is the longest plausible build.
min_age_seconds="${EXT_LOCK_MIN_AGE_S:-1800}"

case "$wait_seconds" in
  ""|*[!0-9]*) wait_seconds=30 ;;
esac
case "$preflight_wait_seconds" in
  ""|*[!0-9]*) preflight_wait_seconds=10 ;;
esac
case "$min_age_seconds" in
  ""|*[!0-9]*) min_age_seconds=1800 ;;
esac

[ "${RECOVER_STALE_EXTENSION_LOCKS:-1}" = "1" ] || exit 0
[ -e "$sentinel" ] || exit 0

compiler_pids() {
  local proc_dir proc_pid proc_cwd proc_cmd
  for proc_dir in /proc/[0-9]*; do
    [ -d "$proc_dir" ] || continue
    proc_pid="${proc_dir##*/}"
    [ "$proc_pid" = "$$" ] && continue
    proc_cwd="$(readlink "$proc_dir/cwd" 2>/dev/null || true)"
    case "$proc_cwd" in
      "$extension_dir"|"$extension_dir"/*)
        printf '%s ' "$proc_pid"
        continue
        ;;
    esac
    proc_cmd="$(tr '\0' ' ' <"$proc_dir/cmdline" 2>/dev/null || true)"
    case "$proc_cmd" in
      *"$extension_dir"*) printf '%s ' "$proc_pid" ;;
    esac
  done
}

# Serialize recovery when two appliances were accidentally pointed at the same
# persistent cache. util-linux/flock is installed by the appliance image.
preflight_file="$extensions_root/.turnkey-extension-lock-recovery"
mkdir -p "$extensions_root" 2>/dev/null || true
exec 8>"$preflight_file" || {
  echo "!!! Extension-lock recovery could not open $preflight_file; leaving $sentinel untouched." >&2
  exit 75
}
if command -v flock >/dev/null 2>&1; then
  if ! flock -w "$preflight_wait_seconds" 8; then
    echo "!!! Extension-lock recovery preflight is busy; leaving $sentinel untouched." >&2
    exit 75
  fi
else
  echo "!!! Extension-lock recovery has no flock utility; continuing with process checks only." >&2
fi

# Recheck after taking the appliance-level lock: another preflight may already
# have recovered the sentinel while this process was waiting.
[ -e "$sentinel" ] || exit 0

elapsed=0
owners="$(compiler_pids)"
while [ -n "$owners" ] && [ "$elapsed" -lt "$wait_seconds" ]; do
  echo ">>> Extension lock is active (PIDs:${owners}); waiting for its compiler (${elapsed}/${wait_seconds}s)."
  sleep 2
  elapsed=$((elapsed + 2))
  [ -e "$sentinel" ] || exit 0
  owners="$(compiler_pids)"
done

if [ -n "$owners" ]; then
  echo "!!! Extension lock still has a live compiler after ${wait_seconds}s (PIDs:${owners})." >&2
  echo "!!! Refusing to move $sentinel; the supervisor will retry instead." >&2
  exit 75
fi

# No owner is visible in THIS PID namespace. That does NOT prove the lock is
# stale: a compiler in another container sharing this persistent /cache lives in
# a separate PID namespace and is invisible above. Require the sentinel to be
# older than the minimum age before quarantining it, so a fresh cross-namespace
# lock is presumed live and left in place (the supervisor retries; a genuinely
# stale sentinel ages past the threshold and is recovered on a later pass).
now_epoch="$(date +%s 2>/dev/null || echo 0)"
sentinel_mtime="$(stat -c %Y -- "$sentinel" 2>/dev/null || echo 0)"
if [ "$sentinel_mtime" -gt 0 ]; then
  age=$((now_epoch - sentinel_mtime))
  if [ "$age" -lt "$min_age_seconds" ]; then
    echo "!!! Apparently-ownerless extension lock $sentinel is only ${age}s old (< ${min_age_seconds}s min age)." >&2
    echo "!!! A compiler in another PID namespace (a second container on this /cache) may still" >&2
    echo "!!! hold it, so it is left in place this pass; the supervisor will retry." >&2
    exit 75
  fi
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
quarantine="$extension_dir/lock.stale-$stamp-$$"
if mv -- "$sentinel" "$quarantine"; then
  echo ">>> Recovered ownerless SparkInfer extension lock:"
  echo ">>>   $sentinel"
  echo ">>>   quarantined as $quarantine"
  exit 0
fi

# A concurrent recovery may have won between the final check and mv.
[ ! -e "$sentinel" ] && exit 0
echo "!!! Could not quarantine ownerless extension lock $sentinel." >&2
exit 75
