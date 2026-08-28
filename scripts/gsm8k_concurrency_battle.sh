#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id required}"
repo=/home/mbelleau/sglang_qwen35/llm-inference-bench
venv=/mnt/fast/build/r31-memory-stack-20260808/evidence/decode-stress/.venv
root="/mnt/fast/build/r31-memory-stack-20260808/evidence/gsm8k-concurrency/${run_id}"
bench="$repo/llm_decode_bench.py"
mkdir -p "$root"

resolve_container() {
  local host_pid
  host_pid=$(sudo ss -ltnpH 'sport = :8000' 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -1)
  if [[ -z "$host_pid" ]]; then
    return 1
  fi
  grep -oE '[0-9a-f]{64}' "/proc/$host_pid/cgroup" | head -1
}

snapshot() {
  local dest="$1"
  local cid
  cid=$(resolve_container)
  {
    date -u +timestamp_utc=%Y-%m-%dT%H:%M:%SZ
    printf 'container_id=%s\n' "$cid"
    podman inspect --format 'name={{.Name}} image={{.ImageName}} started={{.State.StartedAt}} status={{.State.Status}} restart_count={{.RestartCount}} oom_killed={{.State.OOMKilled}}' "$cid"
    printf 'health_http='
    curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 http://127.0.0.1:8000/health
    printf 'models_http='
    curl -sS -o "$dest.models.json" -w '%{http_code}\n' --max-time 10 http://127.0.0.1:8000/v1/models
    nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.free,utilization.gpu,power.draw,power.limit,temperature.gpu,fan.speed,clocks.sm,clocks.mem,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader,nounits
  } > "$dest.txt" 2>&1
  curl -sS --max-time 15 http://127.0.0.1:8000/metrics > "$dest.prom" 2> "$dest.metrics.err" || true
}

start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
commit=$(git -C "$repo" rev-parse HEAD)
cid=$(resolve_container)
{
  printf 'start_utc=%s\n' "$start_utc"
  printf 'repo=https://github.com/local-inference-lab/llm-inference-bench\n'
  printf 'commit=%s\n' "$commit"
  printf 'benchmark_version='
  grep -m1 '^VERSION' "$bench" || true
  printf 'dataset_selection=stride_select_items(items,30); item index floor(i*1319/30), i=0..29\n'
  printf 'dataset_indices='
  python3 -c 'print(",".join(str((i*1319)//30) for i in range(30)))'
  printf 'container_id=%s\n' "$cid"
  podman inspect --format 'name={{.Name}} image={{.ImageName}} started={{.State.StartedAt}} status={{.State.Status}} restart_count={{.RestartCount}} oom_killed={{.State.OOMKilled}}' "$cid"
  podman inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$cid" | grep -E '^(MODEL|SERVED_MODEL_NAME|TP|DCP|MTP|MAX_NUM_SEQS|MAX_MODEL_LEN|MAX_BATCHED_TOKENS|GRAPH|KV_CACHE|KV_CACHE_MEMORY_BYTES|ONLINE_QUANT|ASYNC|VLLM_USE_V1|VLLM_EXL3|CUDA_VISIBLE_DEVICES|LMCache|LMCACHE)=' | sort || true
  curl -sS --max-time 10 http://127.0.0.1:8000/v1/models
} > "$root/metadata.txt" 2>&1

snapshot "$root/before-all"

for c in $(seq 1 12); do
  step=$(printf 'c%02d' "$c")
  step_dir="$root/$step"
  mkdir -p "$step_dir"
  step_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  snapshot "$step_dir/before"
  command=(
    "$venv/bin/python" "$bench"
    --host 127.0.0.1 --port 8000 --model local-primary
    --test-profile gsm8k
    --profile-runs 30
    --profile-concurrency "$c"
    --max-tokens 32768
    --completion-stats-temperature 0
    --completion-stats-no-prefill-scout
    --completion-stats-save-text
    --display-mode plain
    --hw-monitor-interval 0.5
    --output "$step_dir/results.json"
  )
  printf '%q ' "${command[@]}" > "$step_dir/command.txt"
  printf '\n' >> "$step_dir/command.txt"
  printf 'start_utc=%s\n' "$step_start" > "$step_dir/timing.txt"
  set +e
  "${command[@]}" > "$step_dir/raw.log" 2>&1
  rc=$?
  set -e
  step_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf 'end_utc=%s\nexit_code=%s\n' "$step_end" "$rc" >> "$step_dir/timing.txt"
  snapshot "$step_dir/after"
  cid=$(resolve_container)
  podman logs --since "$step_start" "$cid" 2>&1 | gzip -c > "$step_dir/server.log.gz" || true
  zgrep -Ei 'out of memory|cuda oom|allocation fail|enginecore.*(error|dead|exit)|worker.*(error|dead|exit)|traceback|preempt|connector.*(fail|error)|cache.*(fail|error)|compile|warning' "$step_dir/server.log.gz" > "$step_dir/server-alerts.txt" || true
  if [[ -s "$step_dir/results.json" ]]; then
    "$venv/bin/python" - "$step_dir/results.json" > "$step_dir/summary.txt" <<'PY'
import json, math, statistics, sys
d=json.load(open(sys.argv[1]))
print('top_keys='+','.join(d))
cs=d.get('completion_stats') or d.get('completion_stats_results') or {}
print(json.dumps(cs, ensure_ascii=False, sort_keys=True)[:20000])
PY
  fi
  printf 'completed %s exit=%s at %s\n' "$step" "$rc" "$step_end" | tee -a "$root/progress.log"
  if [[ "$rc" -ne 0 ]]; then
    printf 'stopping after failed %s\n' "$step" | tee -a "$root/progress.log"
    exit "$rc"
  fi
done

snapshot "$root/after-all"
date -u +end_utc=%Y-%m-%dT%H:%M:%SZ > "$root/completed.txt"
