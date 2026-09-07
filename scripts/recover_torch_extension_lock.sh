#!/bin/bash
# FileBaton's empty sentinel has no PID, host, or namespace identity. Neither
# /proc visibility nor age proves that a shared-cache build is dead. Never
# rename/delete it: the caller can select a fresh private extension cache.
set -u
extensions_root="${1:-${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}}"
sentinel="$extensions_root/sparkinfer_pcie_dma_ext/lock"
[ "${RECOVER_STALE_EXTENSION_LOCKS:-1}" = "1" ] || exit 0
[ -e "$sentinel" ] || [ -L "$sentinel" ] || exit 0
echo "!!! Extension lock has unverifiable ownership: $sentinel; leaving it untouched." >&2
exit 75
