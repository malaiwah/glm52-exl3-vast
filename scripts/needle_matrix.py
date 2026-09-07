#!/usr/bin/env python3
"""Run a tokenizer-measured context ladder and post-stress serving checks.

The default ladder reaches 520192 prompt-body tokens within a 524288-token
model envelope. It does not claim that the entire envelope is usable as input:
chat formatting and generation need the reserved tokens. Run against an
isolated candidate; this sends real long-prefill and stochastic decode traffic.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import uuid

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
    blob = json.dumps(doc, indent=1, allow_nan=False) + "\n"
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
    parser.add_argument("--sizes", default="32768,131072,262144,393216,500000,520192")
    parser.add_argument("--depths", default="0.01,0.1,0.25,0.5,0.75,0.9,0.99")
    parser.add_argument("--seeds", default="20260726")
    parser.add_argument(
        "--prompt-seed", default="",
        help="repeatable prompt identity; reruns may hit prefix cache (default: fresh UUID)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    try:
        requested = parse_numbers(args.sizes, int)
        if args.reserve_tokens < 4096 or args.timeout < 1:
            raise ValueError("reserve must be at least 4096 and timeout must be positive")
        if any(size < 8192 for size in requested):
            raise ValueError("requested sizes must be at least 8192")
        sizes = capped_sizes(requested, args.max_model_len, args.reserve_tokens)
        depths = parse_numbers(args.depths, float)
        seeds = parse_numbers(args.seeds, int)
    except ValueError as error:
        parser.error(str(error))
    if not sizes:
        parser.error("no requested size leaves at least 8192 tokens after the reserve")
    if any(not math.isfinite(depth) or depth <= 0 or depth >= 1 for depth in depths):
        parser.error("depths must be finite, greater than 0 and less than 1")
    if len(depths) > len(verify.CITIES):
        parser.error("more depths requested than distinct needle cities available")

    base = args.base_url.rstrip("/")
    key = read_key(args.api_key_file)
    model = args.model or discover_model(base, key)
    run_id = args.prompt_seed or uuid.uuid4().hex
    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": model,
        "max_model_len": args.max_model_len,
        "reserve_tokens": args.reserve_tokens,
        "requested_sizes": requested,
        "tested_sizes": sizes,
        "requested_sizes_fit": max(requested) <= args.max_model_len - args.reserve_tokens,
        "depths": depths,
        "seeds": seeds,
        "prompt_identity": run_id,
        "prompt_identity_policy": "caller-supplied" if args.prompt_seed else "fresh-uuid",
        "cache_regime": "not measured; unique trial prefixes, repeat identity may reuse cache",
        "token_count_scope": "tokenized haystack body, excludes chat template and retrieval question",
        "probes": [],
        "post_stress": {},
        "measured_max_tokens": 0,
        "complete": False,
        "ok": False,
    }

    def checkpoint():
        if args.out:
            write_result(args.out, doc)

    checkpoint()
    try:
        for size in sizes:
            for seed in seeds:
                identity = f"{run_id}:{size}:{seed}"
                trial_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                try:
                    result = verify.needle_probe(
                        base, key, model, size, depths, args.timeout, trial_seed)
                    tokens = result.get("tokens")
                    measured = (result.get("tokens_exact") is True
                                and type(tokens) is int and tokens > 0)
                    result["length_verified"] = bool(
                        measured and abs(tokens - size) / size <= 0.01)
                    result["ok"] = bool(result.get("ok") and result["length_verified"])
                    if measured:
                        doc["measured_max_tokens"] = max(doc["measured_max_tokens"], tokens)
                except Exception as error:
                    result = {
                        "attempted": True, "target_tokens": size, "seed": trial_seed,
                        "ok": False, "detail": f"{type(error).__name__}: {error}",
                    }
                result["matrix_seed"] = seed
                doc["probes"].append(result)
                checkpoint()
        for name, probe in (("short_prompt", verify.short_probe),
                            ("stochastic_sampling", verify.stochastic_sampling_probe)):
            try:
                doc["post_stress"][name] = probe(base, key, model)
            except Exception as error:
                doc["post_stress"][name] = {
                    "ok": False, "detail": f"{type(error).__name__}: {error}"}
            checkpoint()
        doc["complete"] = True
        doc["ok"] = bool(doc["requested_sizes_fit"]
                         and all(item.get("ok") for item in doc["probes"])
                         and all(item.get("ok") for item in doc["post_stress"].values()))
    except BaseException as error:
        doc["fatal_error"] = f"{type(error).__name__}: {error}"
        doc["ok"] = False
        write_result(args.out, doc)
        raise
    write_result(args.out, doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
