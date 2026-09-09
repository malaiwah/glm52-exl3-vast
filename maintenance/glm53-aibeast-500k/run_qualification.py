#!/usr/bin/env python3
"""Run paid GPU qualification ONLY on the isolated, staged candidate.

Requires explicit RUN_GPU_QUALIFICATION=1. Restarts/kills only candidate cache
and engine processes, rotates only candidate-derived L2, never production.
The deadline proof executes installed adapter methods on CPU with deterministic
future/clock controls; service fail-stop and recovery are separately GPU-live.
No claim is made that a 180-second GPU DMA hang was injected.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import re
import sys
import threading
import time
import uuid

import candidate as deployment
sys.path.insert(0, str(deployment.REPO / "scripts"))
import benchmark_serving as bench
import verify_serving as verify


def overlap_probe(base, model, nonce):
    """Measure HTTP-observed decode progress during a fresh long prefill."""
    short, short_tokens = bench.make_prompt(base, "", model, 1024, nonce + "-decode")
    long, long_tokens = bench.make_prompt(base, "", model, 131072, nonce + "-prefill")
    decoding = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decode_future = pool.submit(
            bench.stream_completion, base, "", model, short, short_tokens, 8192,
            temperature=1.0, on_first_text=decoding.set, trace=True)
        if not decoding.wait(300):
            return {"ok": False, "error": "decode produced no first text before overlap",
                    "decode": decode_future.result()}
        prefill_future = pool.submit(
            bench.stream_completion, base, "", model, long, long_tokens, 1,
            temperature=1.0, trace=True)
        prefill, decode = prefill_future.result(), decode_future.result()
    prefill_times = prefill.get("timeline", {})
    decode_times = decode.get("timeline", {})
    begin, end = prefill_times.get("started_at", 0), prefill_times.get("first_text_at", 0)
    progress = [stamp for stamp in decode_times.get("text_chunk_times", [])
                if begin < stamp < end]
    return {
        "ok": bool(prefill.get("ok") and decode.get("ok") and long_tokens >= 128000
                   and decode_times.get("first_text_at", float("inf")) < begin
                   and len(progress) >= 2),
        "prefill": prefill, "decode": decode, "decode_chunks_during_prefill": len(progress),
        "scope": "HTTP request-to-first-text interval and text-chunk progress, not GPU kernel timing",
    }


def cold_log_clean(text):
    return not re.search(
        r"out of memory|restarting vLLM|vLLM exited|VERIFICATION FAILED|crash-looped",
        text, re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-eviction-prefixes", type=int, default=512,
                        help="fail closed if L2 pressure does not evict within this many unique 131K prefixes")
    args = parser.parse_args()
    deployment.require(os.environ.get("RUN_GPU_QUALIFICATION") == "1",
                       "explicit RUN_GPU_QUALIFICATION=1 is required; this performs GPU workloads and candidate fault injection")
    os.umask(0o077)
    root, env = deployment.settings()
    stage_hash = deployment.staged(root, env)
    deployment.verify_model(root, env, stage_hash)
    deployment.require(not deployment.inspect(deployment.SPEC["production_container"])["State"]["Running"],
                       "production must remain stopped during the separately authorized maintenance window")
    container = deployment.candidate(root, env)
    deployment.require(container["State"]["Running"], "start the staged candidate first")
    name, model = env["NAME"], env["SERVED_MODEL_NAME"]
    base = f"http://127.0.0.1:{env['PORT']}"
    run_id = uuid.uuid4().hex
    out = root / "evidence" / run_id
    out.mkdir()
    (root / "qualification.json").unlink(missing_ok=True)
    diagnostics = None
    with (out / "gpu-initial.log").open("w") as log:
        subprocess.run(["nvidia-smi"], stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = {}
    cache = {"stage_sha256": stage_hash, "complete": False, "checks": {}}
    load = {"stage_sha256": stage_hash, "complete": False, "checks": {}}
    common = ["--base-url", base, "--model", model]

    def save(label, doc):
        path = out / (label + ".json")
        path.write_text(json.dumps(doc, indent=2) + "\n")
        return {"path": path.name, "sha256": deployment.digest(path)}

    def collect_diagnostics():
        with (out / "container.log").open("w") as log:
            subprocess.run(["podman", "logs", name], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (out / "gpu-final.log").open("w") as log:
            subprocess.run(["nvidia-smi"], stdout=log, stderr=subprocess.STDOUT, check=True)
        return {label: {"path": filename, "sha256": deployment.digest(out / filename)}
                for label, filename in (("container", "container.log"), ("gpu_initial", "gpu-initial.log"),
                                        ("gpu_final", "gpu-final.log"))}

    def check(report, label, passed, doc, scope):
        report["checks"][label] = {"ok": bool(passed), "scope": scope, "artifact": save(label, doc)}
        save("cache", cache)
        save("load", load)
        deployment.require(passed, f"qualification failed: {label}; receipts retained in {out}")

    def script(label, script_name, *extra):
        command = [sys.executable, str(deployment.REPO / "scripts" / script_name), *common,
                   *extra, "--out", str(out / (label + ".json"))]
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = out / (label + ".json")
        deployment.require(path.is_file(), f"{label} produced no report; inspect {out}")
        doc = json.loads(path.read_text())
        deployment.require(result.returncode == 0 and doc.get("complete") is True and doc.get("ok") is True,
                           f"{label} failed; inspect {out}")
        return doc

    def healthy():
        try:
            bench.json_request(base + "/health", timeout=5)
            return True
        except Exception:
            return False

    def wait_ready():
        deadline = time.monotonic() + 90 * 60
        while time.monotonic() < deadline:
            deployment.require(deployment.candidate(root, env)["State"]["Running"], "candidate exited during qualification")
            if healthy():
                return
            time.sleep(5)
        raise ValueError("candidate did not become healthy within 90 minutes")

    def restart():
        deployment.candidate(root, env)
        deployment.podman("stop", "-t", "120", name)
        deployment.podman("start", name)
        wait_ready()

    def inventory():
        # Ignore transient bookkeeping/small files; committed KV objects are
        # large. Retain file identities, not cache contents or prompts.
        result = {}
        for path in (root / "lmcache" / "l2").rglob("*"):
            try:
                size = path.stat().st_size
                if path.is_file() and not path.name.startswith(".") and size >= 131072:
                    result[str(path.relative_to(root / "lmcache" / "l2"))] = size
            except FileNotFoundError:
                pass
        return result

    def runtime_processes(kill_cache=False):
        code = '''import json, os, pathlib, signal
rows=[]; cache=[]; engines=[]
for entry in pathlib.Path('/proc').iterdir():
 if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
 try: args=(entry/'cmdline').read_bytes().decode(errors='replace').strip('\\0').split('\\0')
 except (OSError,ProcessLookupError): continue
 if not args: continue
 pid=int(entry.name)
 if any(pathlib.Path(arg).name=='lmcache' for arg in args[:2]) and 'server' in args: cache.append(pid)
 if args[0].startswith('VLLM::') or ('python' in pathlib.Path(args[0]).name and any('vllm.entrypoints' in arg for arg in args)): engines.append(pid)
 rows.append({'pid':pid,'argv':args})
if KILL:
 assert len(cache)==1, 'expected exactly one candidate LMCache server, got '+str(cache)
 assert engines, 'cannot identify candidate vLLM process group'
 os.kill(cache[0],signal.SIGKILL)
print(json.dumps({'cache':cache,'engines':engines,'processes':rows}))
'''
        code = "KILL=" + repr(kill_cache) + "\n" + code
        return json.loads(deployment.podman("exec", name, "python3", "-c", code, capture=True))

    try:
        wait_ready()
        start = json.loads((root / "start.json").read_text())
        deployment.require(start["stage_sha256"] == stage_hash and start["container_id"] == container["Id"],
                           "no matching cold-start receipt; launch a fresh candidate stage")
        deployment.require(env.get("VERIFY") == "0",
                           "cold qualification requires the automatic startup sampler disabled")
        before_processes = runtime_processes()
        before_metrics = bench.get_metrics(base, "")
        before_log = deployment.podman("logs", name, capture=True)
        no_requests = (
            bench.metric(before_metrics, "vllm:request_success_total") == 0
            and "vllm:num_requests_running" in before_metrics
            and bench.metric(before_metrics, "vllm:num_requests_running") == 0
            and bench.metric(before_metrics, "vllm:num_requests_waiting") == 0
        )
        sampled = verify.stochastic_sampling_probe(base, "", model)
        after_processes = runtime_processes()
        after_log = deployment.podman("logs", name, capture=True)
        cold_log_path = out / "cold-start.log"
        cold_log_path.write_text(before_log + "\n--- first sampler ---\n" + after_log)
        initial_generation = (
            bool(before_processes["engines"])
            and set(before_processes["engines"]) == set(after_processes["engines"])
            and deployment.candidate(root, env).get("RestartCount", 0) == 0
            and cold_log_clean(before_log) and cold_log_clean(after_log)
        )
        check(load, "cold_jit",
              start["compile_cache_empty"] and no_requests and initial_generation
              and sampled.get("ok") is True,
              {"start": start, "first_temperature1_sample": sampled,
               "before": before_processes, "after": after_processes,
               "metrics_before": before_metrics, "initial_generation": initial_generation,
               "no_previous_or_concurrent_requests": no_requests,
               "log": {"path": cold_log_path.name, "sha256": deployment.digest(cold_log_path)}},
              "GPU-live first API sampler; automatic verifier disabled; original engine generation and complete cold logs")
        reports["features"] = save("features", script("features", "feature_suite.py"))
        concurrency = script("concurrency", "benchmark_serving.py", "--concurrency", "1,4,8",
                             "--requests-per-level", "8", "--output-tokens", "512", "--temperature", "1.0")
        for count in (1, 4, 8):
            rows = [row for row in concurrency["concurrency"] if row.get("concurrency") == count]
            check(load, f"c{count}", len(rows) == 1 and rows[0].get("failed") == 0, rows,
                  "GPU-live concurrent temperature1 512-output-token requests")
        overlap = overlap_probe(base, model, "overlap-" + run_id)
        check(load, "long_prefill_decode_overlap", overlap["ok"], overlap,
              "GPU-live cold 128K prefill overlapping already active decode; both stream timelines retained")
        reports["needles"] = save("needles", script("needles", "needle_matrix.py",
            "--max-model-len", env["MAX_MODEL_LEN"], "--sizes",
            "32768,131072,262144,393216,500000,516096", "--seeds", "20260726,20260727"))
        seed = "maintenance-" + run_id
        offload = script("offload", "offload_prefix_benchmark.py", "--prefix-tokens", "131072",
                         "--eviction-prefixes", "5", "--prompt-seed", seed)
        reports["offload"] = save("offload", offload)
        time.sleep(10)
        durable_before = inventory()
        deployment.require(durable_before, "no committed L2 objects observed; cannot prove persistent reload")
        restart()
        reload = script("restart-offload", "offload_prefix_benchmark.py", "--prefix-tokens", "131072",
                        "--eviction-prefixes", "5", "--prompt-seed", seed)
        check(cache, "nvme_l2_hit", reload.get("initial_external_hit_observed") is True,
              {"durable_objects_before": durable_before, "after_restart": reload},
              "GPU-live exact same prefix after engine and LMCache restart; fresh RAM/GPU imply durable L2 origin")
        check(cache, "restart_recovery", reload.get("requests_ok") is True, reload,
              "GPU-live candidate restart and real cached/recomputed requests")

        # Rotate only this candidate's derived L2 while its service is stopped.
        # No unlink, model mutation, or production cache access occurs.
        deployment.podman("stop", "-t", "120", name)
        l2 = root / "lmcache" / "l2"
        preserved = root / ("lmcache-before-miss-" + run_id)
        l2.rename(preserved)
        l2.mkdir()
        deployment.podman("start", name)
        wait_ready()
        miss = script("miss-offload", "offload_prefix_benchmark.py", "--prefix-tokens", "131072",
                      "--eviction-prefixes", "5", "--prompt-seed", seed)
        check(cache, "cache_miss_recompute", miss["cold"]["request"].get("ok") is True
              and miss["cold"]["metrics"].get("external_prefix_hit_tokens") == 0,
              {"preserved_l2": str(preserved), "empty_l2_restart": miss},
              "GPU-live repeated prefix recomputes successfully with both process caches fresh and candidate L2 empty")

        # Exercise the real 384GiB tier, not an unrelated tiny-cache profile.
        baseline = inventory()
        peak = sum(baseline.values())
        cap = int(env["PREFIX_CACHE_DISK_GB"]) * 1024**3
        pressure = {"cap_bytes": cap, "steps": [], "removed_objects": []}
        for index in range(args.max_eviction_prefixes):
            prompt, tokens = bench.make_prompt(base, "", model, 131072, f"l2-pressure-{run_id}-{index}")
            result = bench.stream_completion(base, "", model, prompt, tokens, 32)
            deployment.require(result.get("ok") is True, "L2 pressure request failed")
            time.sleep(2)
            current = inventory()
            total = sum(current.values())
            peak = max(peak, total)
            removed = sorted(set(baseline) - set(current))
            pressure["steps"].append({"index": index, "bytes": total, "request": result})
            save("l2-pressure-progress", pressure)
            deployment.require(total <= cap, "L2 committed object bytes exceeded configured capacity")
            if removed and peak >= int(cap * 0.8):
                pressure["removed_objects"] = removed
                break
            baseline.update(current)
        check(cache, "l2_eviction", bool(pressure["removed_objects"]), pressure,
              "GPU-live unique-prefix pressure at production candidate cap; committed key churn and byte-bound observed")

        # Execute the actual installed adapter methods, never the vendored copy.
        harness = (deployment.REPO / "tests" / "test_scopedlmcache_retrieve.py").read_text()
        code = '''import hashlib,json,os,pathlib,sys,unittest
sys.path.insert(0,'/opt/scripts')
import patch_scopedlmcache_retrieve as installer
path=installer.default_target()
installer.patch(path,verify_only=True)
os.environ['LMCACHE_ADAPTER_TEST_SOURCE']=str(path)
ns={'__name__':'deadline_qualification','__file__':'/opt/tests/test_scopedlmcache_retrieve.py'}
exec(compile(sys.stdin.read(),ns['__file__'],'exec'),ns)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ns['RetrieveDeadlineTests']))
print(json.dumps({'ok':result.wasSuccessful(),'tests_run':result.testsRun,'installed_path':str(path),'installed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'installed adapter CPU controlled futures/clock; NOT a real GPU DMA hang'}))
sys.exit(0 if result.wasSuccessful() else 1)
'''
        fault = subprocess.run(["podman", "exec", "-i", name, "python3", "-c", code],
                               input=harness, text=True, capture_output=True)
        (out / "installed-deadline.log").write_text(fault.stdout + fault.stderr)
        cpu = json.loads(fault.stdout.splitlines()[-1]) if fault.stdout.strip() else {"ok": False}
        check(cache, "retrieve_deadline_fail_stop", fault.returncode == 0 and cpu.get("ok") is True, cpu,
              "installed adapter CPU proof of180s ownership retention/deadline; real GPU stall not injected")

        # Distinct real service boundary proof, not mislabeled as a DMA timeout.
        before = runtime_processes(kill_cache=True)
        observed_down = False
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not healthy():
                observed_down = True
                break
            time.sleep(1)
        deployment.require(observed_down, "cache crash did not fail-stop the serving boundary within180s")
        state = deployment.inspect(name)
        if not state["State"]["Running"]:
            deployment.podman("start", name)
        wait_ready()
        after = runtime_processes()
        recovery = verify.stochastic_sampling_probe(base, "", model)
        replaced = bool(after["engines"]) and not (set(before["engines"]) & set(after["engines"]))
        check(cache, "group_failure_recovery", replaced and recovery.get("ok") is True,
              {"before": before, "after": after, "health_unavailable_observed": observed_down,
               "recovery": recovery}, "GPU-live LMCache SIGKILL -> service fail-stop -> replaced vLLM group and real sampler")
        final = verify.short_probe(base, "", model)
        check(load, "post_stress_health", final.get("ok") is True and healthy(), final,
              "GPU-live request and health after full context/cache/fault matrix")
        cache["complete"] = load["complete"] = True
        reports["cache"] = save("cache", cache)
        reports["load"] = save("load", load)
        diagnostics = collect_diagnostics()
        bundle = {"stage_sha256": stage_hash, "container_id": container["Id"], "reports": reports,
                  "diagnostics": diagnostics}
        save("evidence", bundle)
        deployment.qualify(root, env, out / "evidence.json")
    finally:
        save("cache", cache)
        save("load", load)
        if diagnostics is None:
            collect_diagnostics()
        print(f"Qualification receipts: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        sys.exit(4)
