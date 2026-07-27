#!/usr/bin/env python3
"""Dependency-free serving benchmark for the turnkey appliance.

The harness is intentionally small enough to run inside the production image.
It measures uncached prefill, streamed TTFT/TPOT/ITL, aggregate output
throughput, speculative-decoding acceptance, preemptions, and a concurrency
sweep. Results are written as JSON and never contain the API key.

Example:

    python3 /opt/scripts/benchmark_serving.py \
      --api-key-file /workspace/.vllm-api-key \
      --input-tokens 1024 --output-tokens 512 \
      --concurrency 1,2,4,8 --requests-per-level 8 \
      --prefill-tokens 1024,8192,32768,131072 \
      --out /workspace/benchmarks/baseline.json
"""
import argparse
import concurrent.futures
import datetime
import json
import math
import os
import statistics
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid


def request(url, payload=None, key="", timeout=600, stream=False, insecure=False):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET")
    ctx = ssl._create_unverified_context() if insecure and url.startswith("https") else None
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def json_request(url, payload=None, key="", timeout=600, insecure=False):
    with request(url, payload, key, timeout, insecure=insecure) as response:
        body = response.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def read_key(path):
    if path:
        with open(path) as handle:
            return handle.read().strip()
    return os.environ.get("VLLM_API_KEY", "")


def discover_model(base, key, insecure=False):
    doc = json_request(base + "/v1/models", key=key, insecure=insecure)
    models = doc.get("data") if isinstance(doc, dict) else None
    if not models:
        raise RuntimeError("/v1/models returned no served model")
    return models[0]["id"]


def parse_metrics(text):
    """Sum Prometheus samples by their exact base metric name."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[-1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def get_metrics(base, key, insecure=False):
    with request(base + "/metrics", key=key, timeout=60, insecure=insecure) as response:
        return parse_metrics(response.read().decode("utf-8", "replace"))


def metric(metrics, *names):
    for name in names:
        if name in metrics:
            return metrics[name]
    return 0.0


def metric_delta(before, after, *names):
    return max(0.0, metric(after, *names) - metric(before, *names))


def spec_summary(before, after):
    drafts = metric_delta(
        before, after, "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_drafts_total")
    draft_tokens = metric_delta(
        before, after, "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_delta(
        before, after, "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_total")
    return {
        "drafts": round(drafts),
        "draft_tokens": round(draft_tokens),
        "accepted_tokens": round(accepted),
        "mean_acceptance_length": (
            round(1.0 + accepted / drafts, 4) if drafts else None),
        "draft_token_acceptance_rate": (
            round(accepted / draft_tokens, 4) if draft_tokens else None),
    }


def count_tokens(base, key, model, text, insecure=False):
    doc = json_request(
        base + "/tokenize", {"model": model, "prompt": text}, key=key,
        timeout=180, insecure=insecure)
    if not isinstance(doc, dict) or not isinstance(doc.get("count"), int):
        raise RuntimeError("/tokenize did not return an exact count")
    return doc["count"]


FILLER = (
    "Telemetry batch {i} stayed nominal; the operator reviewed pressure, "
    "temperature, and checksum fields before closing the record."
)


def make_prompt(base, key, model, target_tokens, nonce, insecure=False):
    """Build an uncached prompt within roughly 3% of the requested token count."""
    target_tokens = max(32, int(target_tokens))
    nlines = max(2, target_tokens // 24)
    for _ in range(4):
        lines = [f"Benchmark document {nonce}; all records are independent."]
        lines.extend(FILLER.format(i=i) for i in range(nlines))
        lines.append(
            "Continue with a deterministic sequence of short lowercase words. "
            "Do not summarize the document.")
        text = "\n".join(lines)
        actual = count_tokens(base, key, model, text, insecure)
        if abs(actual - target_tokens) / target_tokens <= 0.03:
            return text, actual
        nlines = max(1, round(nlines * target_tokens / max(actual, 1)))
    return text, actual


def stream_completion(base, key, model, prompt, prompt_tokens, output_tokens,
                      timeout=1800, insecure=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(output_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first = None
    token_events = []
    usage = {}
    pieces = []
    try:
        with request(base + "/v1/chat/completions", payload, key, timeout,
                     stream=True, insecure=insecure) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    packet = json.loads(data)
                except ValueError:
                    continue
                if packet.get("usage"):
                    usage = packet["usage"]
                choices = packet.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece is None:
                    piece = delta.get("reasoning_content", delta.get("reasoning"))
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    token_events.append(now)
                    pieces.append(str(piece))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:1000]
        return {"ok": False, "error": f"HTTP {error.code}: {detail}"}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    ended = time.perf_counter()
    if first is None:
        return {"ok": False, "error": "stream returned no text-bearing deltas"}
    observed_output = usage.get("completion_tokens")
    if not isinstance(observed_output, int):
        observed_output = len(token_events)
    observed_prompt = usage.get("prompt_tokens")
    if not isinstance(observed_prompt, int):
        observed_prompt = prompt_tokens
    itls = [b - a for a, b in zip(token_events, token_events[1:])]
    elapsed = ended - started
    ttft = first - started
    tpot = ((elapsed - ttft) / (observed_output - 1)
            if observed_output > 1 else 0.0)
    return {
        "ok": True,
        "prompt_tokens": observed_prompt,
        "output_tokens": observed_output,
        "elapsed_s": round(elapsed, 6),
        "ttft_ms": round(ttft * 1000, 3),
        "tpot_ms": round(tpot * 1000, 3),
        "mean_itl_ms": round(statistics.mean(itls) * 1000, 3) if itls else 0.0,
        "output_head": "".join(pieces)[:120],
    }


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_requests(results, wall_s, before, after, concurrency):
    good = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    prompt_tokens = sum(item["prompt_tokens"] for item in good)
    output_tokens = sum(item["output_tokens"] for item in good)
    ttft = [item["ttft_ms"] for item in good]
    tpot = [item["tpot_ms"] for item in good]
    itl = [item["mean_itl_ms"] for item in good]
    prompt_metric = metric_delta(
        before, after, "vllm:prompt_tokens_total", "vllm:prompt_tokens")
    generation_metric = metric_delta(
        before, after, "vllm:generation_tokens_total", "vllm:generation_tokens")
    preemptions = metric_delta(
        before, after, "vllm:num_preemptions_total", "vllm:num_preemptions")
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "completed": len(good),
        "failed": len(failed),
        "duration_s": round(wall_s, 4),
        "request_throughput_rps": round(len(good) / wall_s, 4) if wall_s else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_throughput_tok_s": round(prompt_tokens / wall_s, 3) if wall_s else 0.0,
        "output_throughput_tok_s": round(output_tokens / wall_s, 3) if wall_s else 0.0,
        "mean_ttft_ms": round(statistics.mean(ttft), 3) if ttft else 0.0,
        "p50_ttft_ms": round(percentile(ttft, 50), 3),
        "p95_ttft_ms": round(percentile(ttft, 95), 3),
        "mean_tpot_ms": round(statistics.mean(tpot), 3) if tpot else 0.0,
        "p50_tpot_ms": round(percentile(tpot, 50), 3),
        "p95_tpot_ms": round(percentile(tpot, 95), 3),
        "mean_itl_ms": round(statistics.mean(itl), 3) if itl else 0.0,
        "metric_prompt_tokens": round(prompt_metric),
        "metric_generation_tokens": round(generation_metric),
        "preemptions": round(preemptions),
        "speculative": spec_summary(before, after),
        "errors": [item.get("error", "") for item in failed][:10],
        "requests_detail": good,
    }


def run_level(base, key, model, input_tokens, output_tokens, concurrency,
              request_count, timeout, insecure=False):
    prompts = []
    run_id = uuid.uuid4().hex
    for index in range(request_count):
        prompt, actual = make_prompt(
            base, key, model, input_tokens, f"{run_id}-{index}", insecure)
        prompts.append((prompt, actual))
    before = get_metrics(base, key, insecure)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_completion, base, key, model, prompt, actual,
                        output_tokens, timeout, insecure)
            for prompt, actual in prompts
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    after = get_metrics(base, key, insecure)
    return summarize_requests(results, wall_s, before, after, concurrency)


def write_result(path, doc):
    if not path:
        json.dump(doc, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(doc, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def parse_ints(text):
    return [int(value) for value in text.split(",") if value.strip()]


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--prefill-tokens", default="1024,8192,32768")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    key = read_key(args.api_key_file)
    model = args.model or discover_model(base, key, args.insecure)
    metadata = {}
    for item in args.metadata:
        name, sep, value = item.partition("=")
        if not sep:
            parser.error("--metadata values must be KEY=VALUE")
        metadata[name] = value
    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": model,
        "settings": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": parse_ints(args.concurrency),
            "requests_per_level": args.requests_per_level,
            "prefill_tokens": parse_ints(args.prefill_tokens),
            "warmup": args.warmup,
        },
        "metadata": metadata,
        "prefill": [],
        "concurrency": [],
    }
    for index in range(args.warmup):
        run_level(base, key, model, 256, 32, 1, 1, args.timeout, args.insecure)
    for tokens in parse_ints(args.prefill_tokens):
        result = run_level(
            base, key, model, tokens, 1, 1, 1, args.timeout, args.insecure)
        result["target_prompt_tokens"] = tokens
        doc["prefill"].append(result)
    for concurrency in parse_ints(args.concurrency):
        count = max(concurrency, args.requests_per_level)
        doc["concurrency"].append(run_level(
            base, key, model, args.input_tokens, args.output_tokens,
            concurrency, count, args.timeout, args.insecure))
    doc["ok"] = all(row["failed"] == 0 for row in
                    doc["prefill"] + doc["concurrency"])
    write_result(args.out, doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
