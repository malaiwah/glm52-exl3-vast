#!/usr/bin/env python3
"""Run reproducible needle-in-a-haystack probes across lengths and seeds.

This extends the boot-time 32K correctness gate without making every appliance
startup wait for a full 512K prefill. It is intended for release qualification:

    python3 /opt/scripts/needle_matrix.py \
      --api-key-file /workspace/.vllm-api-key \
      --sizes 32768,131072,262144,393216,490000 \
      --depths 0.01,0.1,0.25,0.5,0.75,0.9,0.99 \
      --seeds 20260726,20260727 \
      --out /workspace/benchmarks/needles.json
"""
import argparse
import datetime
import json
import os
import sys

import verify_serving as verify


def parse_numbers(text, cast):
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("at least one value is required")
    return values


def read_key(path):
    if path:
        with open(path) as handle:
            return handle.read().strip()
    return os.environ.get("VLLM_API_KEY", "")


def discover_model(base, key):
    doc = verify._req(base + "/v1/models", key=key, timeout=60)
    models = doc.get("data") if isinstance(doc, dict) else None
    if not models:
        raise RuntimeError("/v1/models returned no served model")
    return models[0]["id"]


def capped_sizes(sizes, max_model_len, reserve):
    ceiling = max(0, max_model_len - reserve)
    return sorted(set(min(size, ceiling) for size in sizes if min(size, ceiling) >= 8192))


def write_result(path, doc):
    blob = json.dumps(doc, indent=1) + "\n"
    if not path:
        sys.stdout.write(blob)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(blob)
    os.replace(tmp, path)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-model-len", type=int, default=524288)
    parser.add_argument("--reserve-tokens", type=int, default=4096)
    parser.add_argument("--sizes", default="32768,131072,262144,393216,490000")
    parser.add_argument("--depths", default="0.01,0.1,0.25,0.5,0.75,0.9,0.99")
    parser.add_argument("--seeds", default="20260726")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    key = read_key(args.api_key_file)
    model = args.model or discover_model(base, key)
    try:
        sizes = capped_sizes(
            parse_numbers(args.sizes, int), args.max_model_len, args.reserve_tokens)
        depths = parse_numbers(args.depths, float)
        seeds = parse_numbers(args.seeds, int)
    except ValueError as error:
        parser.error(str(error))
    if not sizes:
        parser.error("no requested size leaves at least 8192 tokens after the reserve")
    if any(depth <= 0 or depth >= 1 for depth in depths):
        parser.error("depths must be greater than 0 and less than 1")

    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": model,
        "max_model_len": args.max_model_len,
        "reserve_tokens": args.reserve_tokens,
        "requested_sizes": parse_numbers(args.sizes, int),
        "tested_sizes": sizes,
        "depths": depths,
        "seeds": seeds,
        "probes": [],
    }
    for size in sizes:
        for seed in seeds:
            try:
                result = verify.needle_probe(
                    base, key, model, size, depths, args.timeout, seed)
            except Exception as error:
                result = {
                    "attempted": True,
                    "target_tokens": size,
                    "seed": seed,
                    "ok": False,
                    "detail": f"{type(error).__name__}: {error}",
                }
            doc["probes"].append(result)
            # Persist every expensive probe. An interruption should not discard
            # the successful portion of a multi-hour qualification matrix.
            doc["ok"] = all(item.get("ok") for item in doc["probes"])
            write_result(args.out, doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
