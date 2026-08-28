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
#   * an ownerless sentinel is renamed, never deleted.
set -u

extensions_root="${1:-${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}}"
extension_dir="$extensions_root/sparkinfer_pcie_dma_ext"
sentinel="$extension_dir/lock"
wait_seconds="${TORCH_EXTENSION_LOCK_WAIT_S:-30}"
preflight_wait_seconds="${TORCH_EXTENSION_PREFLIGHT_WAIT_S:-10}"

case "$wait_seconds" in
  ""|*[!0-9]*) wait_seconds=30 ;;
esac
case "$preflight_wait_seconds" in
  ""|*[!0-9]*) preflight_wait_seconds=10 ;;
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
