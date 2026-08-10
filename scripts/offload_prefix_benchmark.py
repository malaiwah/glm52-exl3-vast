#!/usr/bin/env python3
"""Measure DRAM KV offload as a reusable-prefix tier.

This is deliberately not a KV-capacity benchmark.  It measures the appliance's
intended agentic-workload path:

1. cold prefill a large, unique prefix;
2. repeat it while it is still resident in the GPU prefix cache;
3. prefill enough other unique documents to evict the first prefix from GPU;
4. repeat the first prefix and observe an external-cache/DRAM load.

The JSON result records request latency plus vLLM's native prefix-cache and
``kv_offload_*`` counters.  It never records prompts or API keys.
"""
import argparse
import datetime
import json
import sys
import time
import uuid

import benchmark_serving as bench


OFFLOAD_COUNTERS = {
    "kv_offload_allocation_failure": (
        "vllm:kv_offload_allocation_failure_total",
        "vllm:kv_offload_allocation_failure"),
    "kv_offload_load_bytes": (
        "vllm:kv_offload_load_bytes_total", "vllm:kv_offload_load_bytes"),
    "kv_offload_load_count": (
        "vllm:kv_offload_load_size_count",),
    "kv_offload_load_time": (
        "vllm:kv_offload_load_time_total", "vllm:kv_offload_load_time"),
    "kv_offload_store_bytes": (
        "vllm:kv_offload_store_bytes_total", "vllm:kv_offload_store_bytes"),
    "kv_offload_store_count": (
        "vllm:kv_offload_store_size_count",),
    "kv_offload_store_time": (
        "vllm:kv_offload_store_time_total", "vllm:kv_offload_store_time"),
    "kv_offload_stores_skipped": (
        "vllm:kv_offload_stores_skipped_total",
        "vllm:kv_offload_stores_skipped"),
}

OFFLOAD_GAUGES = (
    "vllm:kv_offload_cpu_allocation_size",
    "vllm:kv_offload_cpu_cache_read_usage_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_write_usage_perc",
    "vllm:kv_offload_total_bytes",
)

def metric_summary(before, after):
    out = {
        "prefix_query_tokens": round(bench.metric_delta(
            before, after, "vllm:prefix_cache_queries_total")),
        "gpu_prefix_hit_tokens": round(bench.metric_delta(
            before, after, "vllm:prefix_cache_hits_total")),
        "external_prefix_query_tokens": round(bench.metric_delta(
            before, after, "vllm:external_prefix_cache_queries_total")),
        "external_prefix_hit_tokens": round(bench.metric_delta(
            before, after, "vllm:external_prefix_cache_hits_total")),
    }
    for label, names in OFFLOAD_COUNTERS.items():
        out[label] = round(bench.metric_delta(before, after, *names), 6)
    for name in OFFLOAD_GAUGES:
        if name in after:
            out[name.removeprefix("vllm:")] = round(after[name], 6)
    return out


def connector_hit_evidence(metrics):
    external = bool(metrics.get("external_prefix_hit_tokens", 0) > 0)
    native_bytes = bool(metrics.get("kv_offload_load_bytes", 0) > 0)
    return {
        "external_hit_observed": external,
        "native_load_bytes_observed": native_bytes,
        # A *confirmed DRAM* hit requires the DRAM-specific native signal
        # (kv_offload_load_bytes). A bare external hit can be served from
        # persistent NVMe L2 (e.g. an LMCache reload after restart), so it is
        # proof of an external hit only, not of a DRAM reload.
        "dram_hit_observed": native_bytes,
    }


def connector_hit_summary(initial_metrics, reload_metrics):
    """Accept either a post-eviction hit or a persistent hit after restart."""
    initial = connector_hit_evidence(initial_metrics)
    reload = connector_hit_evidence(reload_metrics)
    return {
        **reload,
        "initial_external_hit_observed": initial["external_hit_observed"],
        "any_external_hit_observed": bool(
            reload["external_hit_observed"]
            or initial["external_hit_observed"]),
    }


def run_request(base, key, model, prompt, prompt_tokens, output_tokens, timeout,
                insecure):
    before = bench.get_metrics(base, key, insecure)
    started = time.perf_counter()
    result = bench.stream_completion(
        base, key, model, prompt, prompt_tokens, output_tokens, timeout, insecure)
    elapsed = time.perf_counter() - started
    after = bench.get_metrics(base, key, insecure)
    return {
        "request": result,
        "wall_s": round(elapsed, 6),
        "metrics": metric_summary(before, after),
    }


def write_result(path, doc):
    bench.write_result(path, doc)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--prefix-tokens", type=int, default=131072)
    parser.add_argument("--eviction-prefixes", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument(
        "--prompt-seed", default="",
        help=("stable prompt identity for a restart/NVMe-reuse A/B; leave "
              "empty for a fresh UUID"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    key = bench.read_key(args.api_key_file)
    model = args.model or bench.discover_model(base, key, args.insecure)
    run_id = args.prompt_seed or uuid.uuid4().hex
    target_prompt, actual = bench.make_prompt(
        base, key, model, args.prefix_tokens, f"offload-target-{run_id}",
        args.insecure)
    evictors = [
        bench.make_prompt(
            base, key, model, args.prefix_tokens,
            f"offload-evict-{run_id}-{index}", args.insecure)
        for index in range(args.eviction_prefixes)
    ]

    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": model,
        "settings": {
            "requested_prefix_tokens": args.prefix_tokens,
            "actual_target_prefix_tokens": actual,
            "eviction_prefixes": args.eviction_prefixes,
            "output_tokens": args.output_tokens,
            "prompt_seed": args.prompt_seed,
        },
        "cold": {},
        "gpu_hot": {},
        "eviction": [],
        "dram_reload": {},
        "complete": False,
        "ok": False,
    }
    write_result(args.out, doc)
    try:
        doc["cold"] = run_request(
            base, key, model, target_prompt, actual, args.output_tokens,
            args.timeout, args.insecure)
        write_result(args.out, doc)
        doc["gpu_hot"] = run_request(
            base, key, model, target_prompt, actual, args.output_tokens,
            args.timeout, args.insecure)
        write_result(args.out, doc)
        for prompt, tokens in evictors:
            doc["eviction"].append(run_request(
                base, key, model, prompt, tokens, args.output_tokens,
                args.timeout, args.insecure))
            write_result(args.out, doc)
        doc["dram_reload"] = run_request(
            base, key, model, target_prompt, actual, args.output_tokens,
            args.timeout, args.insecure)
        doc["complete"] = True
        stages = [doc["cold"], doc["gpu_hot"], *doc["eviction"],
                  doc["dram_reload"]]
        doc["ok"] = all(
            stage.get("request", {}).get("ok") for stage in stages)
        # A healthy request matrix is distinct from proving an external hit.
        # vLLM's connector-level counter applies to both the native
        # OffloadingConnector and LMCache. Native additionally exports
        # kv_offload_load_bytes; LMCache deliberately does not, so requiring
        # that native-only counter produces a false negative despite a
        # chunk-aligned external hit.
        reload_metrics = doc["dram_reload"].get("metrics", {})
        # A deterministic prompt can deliberately be warm in persistent L2 on
        # the first request after an engine restart. Keep that distinct from
        # the in-process eviction/reload signal while accepting either as
        # proof that the external tier actually served tokens.
        doc.update(connector_hit_summary(
            doc["cold"].get("metrics", {}), reload_metrics))
        write_result(args.out, doc)
    except Exception as error:
        doc["fatal_error"] = f"{type(error).__name__}: {error}"
        write_result(args.out, doc)
        raise
    return 0 if doc["ok"] and doc["any_external_hit_observed"] else 2


if __name__ == "__main__":
    sys.exit(main())
