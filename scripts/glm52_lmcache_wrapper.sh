#!/usr/bin/env bash
# Bounded r11 LMCache MP wrapper for the turnkey GLM-5.2 server.
set -Eeuo pipefail

[[ "$#" -gt 0 ]] || { echo "FATAL: LMCache wrapper needs a server command" >&2; exit 2; }
mode="${LMCACHE_MODE:-off}"
mode="$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]')"
case "$mode" in
  off|0) exec "$@" ;;
  ram|memory|1) mode=ram ;;
  disk|ram-disk|memory-disk) mode=disk ;;
  *) echo "FATAL: LMCACHE_MODE must be off, ram, or disk" >&2; exit 2 ;;
esac
command -v lmcache >/dev/null || {
  echo "FATAL: LMCACHE_MODE=$mode but this image has no lmcache CLI" >&2
  exit 2
}

service_port="${PORT:-8000}"
offset=$((service_port >= 8000 ? service_port - 8000 : 0))
host="${LMCACHE_HOST:-127.0.0.1}"
port="${LMCACHE_PORT:-$((5555 + offset))}"
http_port="${LMCACHE_HTTP_PORT:-$((8089 + offset))}"
metrics_port="${LMCACHE_PROMETHEUS_PORT:-$((9090 + offset))}"
chunk="${LMCACHE_CHUNK_SIZE:-512}"
case "${DCP:-1}:$chunk" in
  3:512|6:512) chunk=384 ;;
esac
l1_gb="${LMCACHE_L1_GB:-24}"
log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"

args=(server --host "$host" --port "$port" --chunk-size "$chunk"
  --max-gpu-workers "${LMCACHE_MAX_GPU_WORKERS:-${TP:-1}}"
  --max-cpu-workers "${LMCACHE_MAX_CPU_WORKERS:-4}"
  --l1-size-gb "$l1_gb" --l1-init-size-gb "${LMCACHE_L1_INIT_GB:-$l1_gb}"
  --l1-write-ttl-seconds "${LMCACHE_L1_WRITE_TTL:-600}"
  --l1-read-ttl-seconds "${LMCACHE_L1_READ_TTL:-300}"
  --eviction-policy LRU --eviction-trigger-watermark 0.90
  --eviction-ratio 0.10 --l2-store-policy default --l2-prefetch-policy retain
  --http-port "$http_port" --prometheus-port "$metrics_port")

l2=disabled
if [[ "$mode" == disk ]]; then
  l2="${LMCACHE_L2_PATH:?LMCACHE_L2_PATH is required in disk mode}"
  mkdir -p "$l2"
  l2_json="$(python3 - "$l2" "${LMCACHE_L2_WORKERS:-4}" \
      "${LMCACHE_L2_GB:-256}" <<'PY'
import json, sys
print(json.dumps({"type":"fs_native","base_path":sys.argv[1],
 "num_workers":int(sys.argv[2]),"use_odirect":False,
 "max_capacity_gb":float(sys.argv[3])}, separators=(",",":")))
PY
)"
  args+=(--l2-adapter "$l2_json")
fi

transfer="$(python3 - "$host" "$port" <<'PY'
import json, sys
print(json.dumps({"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
 "kv_connector_extra_config":{"lmcache.mp.host":"tcp://"+sys.argv[1],
 "lmcache.mp.port":int(sys.argv[2]),"lmcache.mp.mq_timeout":60,
 "lmcache.mp.heartbeat_interval":5}}, separators=(",",":")))
PY
)"

allocator="${PYTORCH_CUDA_ALLOC_CONF:-}"
if [[ -z "$allocator" ]]; then
  allocator="expandable_segments:False"
else
  allocator="${allocator//expandable_segments:True/expandable_segments:False}"
fi
export PYTORCH_CUDA_ALLOC_CONF="$allocator"
export LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
rm -f "$log"
lmcache "${args[@]}" >"$log" 2>&1 &
cache_pid=$!
model_pid=""
stop_children() {
  [[ -z "$model_pid" ]] || kill -TERM "$model_pid" 2>/dev/null || true
  kill -TERM "$cache_pid" 2>/dev/null || true
}
trap stop_children INT TERM HUP

ready=0
for _ in $(seq 1 "${LMCACHE_START_TIMEOUT:-120}"); do
  kill -0 "$cache_pid" 2>/dev/null || break
  if curl -fsS --max-time 1 "http://127.0.0.1:${http_port}/healthcheck" \
       >/dev/null 2>&1 ||
     grep -Fq "${LMCACHE_READY_LOG_TEXT:-LMCache ZMQ cache server is running}" "$log"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "FATAL: LMCache did not become ready; log follows" >&2
  sed -n '1,240p' "$log" >&2
  stop_children
  exit 1
fi
echo ">>> LMCache ready: mode=$mode L1=${l1_gb}GiB chunk=$chunk L2=$l2 metrics=:${metrics_port}"

"$@" --kv-transfer-config "$transfer" &
model_pid=$!
set +e
finished=""
wait -n -p finished "$cache_pid" "$model_pid"
rc=$?
set -e
if [[ "$finished" == "$cache_pid" ]]; then
  echo "FATAL: LMCache exited while vLLM was running" >&2
  sed -n '1,240p' "$log" >&2
  stop_children
  wait "$model_pid" 2>/dev/null || true
  exit 1
fi
stop_children
wait "$cache_pid" 2>/dev/null || true
exit "$rc"
