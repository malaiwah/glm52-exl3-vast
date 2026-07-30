#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"

cat >"$tmp/bin/lmcache" <<'SH'
#!/usr/bin/env bash
printf '%q ' "$@" >"$LMCACHE_TEST_SERVER_ARGS"
printf '\n' >>"$LMCACHE_TEST_SERVER_ARGS"
echo 'LMCache ZMQ cache server is running'
if [[ "${LMCACHE_TEST_EXIT_AFTER_READY:-0}" == 1 ]]; then
  sleep 1
  exit 23
fi
trap 'exit 0' INT TERM
while true; do sleep 1 & wait $! || true; done
SH
cat >"$tmp/bin/curl" <<'SH'
#!/usr/bin/env bash
[[ "${LMCACHE_TEST_HTTP_READY:-0}" == 1 ]]
SH
cat >"$tmp/model-server" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$LMCACHE_TEST_MODEL_ARGS"
printf '%s\n' "${PYTORCH_CUDA_ALLOC_CONF-<unset>}" >"$LMCACHE_TEST_MODEL_ENV"
sleep "${LMCACHE_TEST_MODEL_SLEEP:-0}"
SH
chmod +x "$tmp/bin/lmcache" "$tmp/bin/curl" "$tmp/model-server"

run_mode() {
  local mode="$1" port="$2"
  PATH="$tmp/bin:$PATH" \
  LMCACHE_MODE="$mode" PORT="$port" GLM_GPU_COUNT=4 DCP=2 \
  LMCACHE_L1_GB=2 LMCACHE_L2_GB=7 LMCACHE_L2_PATH="$tmp/l2" \
  LMCACHE_LOG="$tmp/$mode.log" \
  LMCACHE_TEST_SERVER_ARGS="$tmp/$mode-server.args" \
  LMCACHE_TEST_MODEL_ARGS="$tmp/$mode-model.args" \
  LMCACHE_TEST_MODEL_ENV="$tmp/$mode-model.env" \
  "$BASH" "$repo/scripts/glm52_lmcache_wrapper.sh" \
    "$tmp/model-server" --model-arg value
}

run_mode ram 8002
grep -Fq -- '--port 5557' "$tmp/ram-server.args"
grep -Fq -- '--http-port 8091' "$tmp/ram-server.args"
grep -Fq -- '--prometheus-port 9092' "$tmp/ram-server.args"
grep -Fq -- '--l1-size-gb 2' "$tmp/ram-server.args"
grep -Fq -- '--max-gpu-workers 4' "$tmp/ram-server.args"
grep -Fq -- '--chunk-size 512' "$tmp/ram-server.args"
if grep -Fq -- '--l2-adapter' "$tmp/ram-server.args"; then
  echo "RAM-only mode unexpectedly configured an L2 adapter" >&2
  exit 1
fi
grep -Fq -- '--kv-transfer-config' "$tmp/ram-model.args"
grep -Fq -- '"lmcache.mp.port":5557' "$tmp/ram-model.args"
grep -Fxq 'expandable_segments:False' "$tmp/ram-model.env"

run_mode disk 8003
grep -Fq -- '--l2-adapter' "$tmp/disk-server.args"
grep -Fq -- 'fs_native' "$tmp/disk-server.args"
grep -Fq -- 'max_capacity_gb' "$tmp/disk-server.args"

LMCACHE_MODE=off \
LMCACHE_TEST_MODEL_ARGS="$tmp/off-model.args" \
LMCACHE_TEST_MODEL_ENV="$tmp/off-model.env" \
"$BASH" "$repo/scripts/glm52_lmcache_wrapper.sh" "$tmp/model-server" untouched
grep -Fxq untouched "$tmp/off-model.args"
grep -Fxq '<unset>' "$tmp/off-model.env"

if PATH="$tmp/bin:$PATH" LMCACHE_MODE=ram PORT=8004 \
  LMCACHE_L1_GB=2 LMCACHE_LOG="$tmp/crash.log" \
  LMCACHE_TEST_EXIT_AFTER_READY=1 LMCACHE_TEST_MODEL_SLEEP=5 \
  LMCACHE_TEST_SERVER_ARGS="$tmp/crash-server.args" \
  LMCACHE_TEST_MODEL_ARGS="$tmp/crash-model.args" \
  LMCACHE_TEST_MODEL_ENV="$tmp/crash-model.env" \
  "$BASH" "$repo/scripts/glm52_lmcache_wrapper.sh" \
    "$tmp/model-server" 2>"$tmp/crash.stderr"; then
  echo "wrapper unexpectedly survived LMCache failure" >&2
  exit 1
fi
grep -Fq 'LMCache exited while vLLM was running' "$tmp/crash.stderr"

echo "turnkey LMCache wrapper: PASS"
