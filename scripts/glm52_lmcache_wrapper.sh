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
case "$host" in
  127.0.0.1|localhost) ;;
  *)
    echo "FATAL: the appliance LMCache service must remain loopback-only: $host" >&2
    exit 2
    ;;
esac
port="${LMCACHE_PORT:-$((5555 + offset))}"
http_port="${LMCACHE_HTTP_PORT:-$((8089 + offset))}"
metrics_port="${LMCACHE_PROMETHEUS_PORT:-$((9090 + offset))}"
chunk="${LMCACHE_CHUNK_SIZE:-512}"
case "${DCP:-1}:$chunk" in
  3:512|6:512) chunk=384 ;;
esac
l1_gb="${LMCACHE_L1_GB:-24}"
l1_init_gb="${LMCACHE_L1_INIT_GB:-}"
if [[ -z "$l1_init_gb" ]]; then
  # LMCache grows L1 lazily, but passing the full capacity as its initial arena
  # defeats that design. A 125 GiB flagship tier then competes with NFS model
  # page-in before it contains a single prefix. Retain LMCache's 20 GiB
  # upstream default as the ceiling while allowing smaller tiers to initialize
  # at their complete size.
  if [[ "$l1_gb" =~ ^[0-9]+$ ]] && (( l1_gb > 20 )); then
    l1_init_gb=20
  else
    l1_init_gb="$l1_gb"
  fi
fi
log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"
gpu_workers="${LMCACHE_MAX_GPU_WORKERS:-${TENSOR_PARALLEL_SIZE:-${GLM_GPU_COUNT:-${TP:-1}}}}"
[[ "$gpu_workers" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: LMCache GPU worker count must be a positive integer: $gpu_workers" >&2
  exit 2
}

args=(server --host "$host" --port "$port" --chunk-size "$chunk"
  --max-gpu-workers "$gpu_workers"
  --max-cpu-workers "${LMCACHE_MAX_CPU_WORKERS:-4}"
  --l1-size-gb "$l1_gb" --l1-init-size-gb "$l1_init_gb"
  --l1-write-ttl-seconds "${LMCACHE_L1_WRITE_TTL:-600}"
  --l1-read-ttl-seconds "${LMCACHE_L1_READ_TTL:-300}"
  --eviction-policy LRU --eviction-trigger-watermark 0.90
  --eviction-ratio 0.10 --l2-store-policy default --l2-prefetch-policy retain
  --http-host "$host" --http-port "$http_port"
  --prometheus-port "$metrics_port")

l2=disabled
if [[ "$mode" == disk ]]; then
  l2="${LMCACHE_L2_PATH:?LMCACHE_L2_PATH is required in disk mode}"
  mkdir -p "$l2"
  l2_json="$(python3 - "$l2" "${LMCACHE_L2_WORKERS:-4}" \
      "${LMCACHE_L2_GB:-256}" "${LMCACHE_L2_EVICTION_POLICY:-LRU}" \
      "${LMCACHE_L2_EVICTION_TRIGGER_WATERMARK:-0.90}" \
      "${LMCACHE_L2_EVICTION_RATIO:-0.10}" <<'PY'
import json, sys
print(json.dumps({"type":"fs_native","base_path":sys.argv[1],
 "num_workers":int(sys.argv[2]),"use_odirect":False,
 "max_capacity_gb":float(sys.argv[3]),"eviction":{
  "eviction_policy":sys.argv[4],"trigger_watermark":float(sys.argv[5]),
  "eviction_ratio":float(sys.argv[6])}}, separators=(",",":")))
PY
)"
  args+=(--l2-adapter "$l2_json")
fi

transfer="$(python3 - "$host" "$port" \
    "${LMCACHE_RETRIEVE_TIMEOUT_SECONDS:-180}" <<'PY'
import json, sys
print(json.dumps({"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
 "kv_connector_extra_config":{"lmcache.mp.host":"tcp://"+sys.argv[1],
 "lmcache.mp.port":int(sys.argv[2]),"lmcache.mp.mq_timeout":60,
 "lmcache.mp.heartbeat_interval":5,
 "lmcache.mp.retrieve_timeout":float(sys.argv[3])}}, separators=(",",":")))
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
secret_unsets=()
while IFS='=' read -r name _value; do
  case "$name" in
    *_TOKEN|*_API_KEY|*_SECRET|*_PASSWORD|*_PRIVATE_KEY|\
    AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)
      secret_unsets+=(-u "$name")
      ;;
  esac
done < <(env)
env "${secret_unsets[@]}" lmcache "${args[@]}" >"$log" 2>&1 &
unset secret_unsets name _value
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
echo ">>> LMCache ready: mode=$mode L1=${l1_gb}GiB (initial ${l1_init_gb}GiB, lazy growth) chunk=$chunk GPU-workers=$gpu_workers L2=$l2 status=http://127.0.0.1:${http_port}/status"

"$@" --kv-transfer-config "$transfer" &
model_pid=$!
# Bash 3.2 (still the system shell on macOS) has neither `wait -n` nor
# `wait -p`. Polling the two supervised children keeps the wrapper portable
# without weakening the important invariant: an LMCache crash must tear down
# vLLM, while an ordinary vLLM exit must drain LMCache.
finished=""
while [[ -z "$finished" ]]; do
  if ! kill -0 "$cache_pid" 2>/dev/null; then
    finished=cache
    break
  fi
  if ! kill -0 "$model_pid" 2>/dev/null; then
    finished=model
    break
  fi
  sleep 0.1
done
if [[ "$finished" == cache ]]; then
  echo "FATAL: LMCache exited while vLLM was running" >&2
  sed -n '1,240p' "$log" >&2
  stop_children
  wait "$model_pid" 2>/dev/null || true
  exit 1
fi
set +e
wait "$model_pid"
rc=$?
set -e
stop_children
wait "$cache_pid" 2>/dev/null || true
exit "$rc"
