#!/usr/bin/env python3
"""MTP acceptance test (review finding 7): serve the exported SIQ checkpoint
with its co-trained MTP head as the speculative drafter and measure real
acceptance from the engine's spec-decode counters.

The trainer co-trains a 1-layer MTP block (MTP_W); export writes it as
num_nextn_predict_layers=1. Greedy decode over license-flavored prompts
(on-distribution for this corpus), then report:
  acceptance rate     = accepted / draft tokens
  mean accepted len   = 1 + accepted / drafts   (bounded by 1+K)

Usage: fruit_serve_mtp.py <checkpoint> <kv_dtype>
Env: K (num_speculative_tokens, default 1), NTOK (tokens/prompt, 256).
"""
import os
import sys
import time


def main() -> None:
    ckpt, kv = sys.argv[1], sys.argv[2]
    k = int(os.environ.get("K", "1"))
    ntok = int(os.environ.get("NTOK", "256"))
    # co-resident daemons (e.g. the TTS server) can hold VRAM; 0.75 still
    # leaves ~13 GB headroom over the 5B model at 2k ctx
    util = float(os.environ.get("GPU_UTIL", "0.75"))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    last = None
    for method in ("mtp", "glm4_moe_mtp", "deepseek_mtp"):
        try:
            llm = LLM(model=ckpt, kv_cache_dtype=kv, max_model_len=2048,
                      max_num_seqs=4, max_num_batched_tokens=2048,
                      trust_remote_code=False, enforce_eager=True,
                      gpu_memory_utilization=util,
                      disable_log_stats=False,
                      attention_backend="B12X_MLA_SPARSE",
                      speculative_config={"method": method,
                                          "num_speculative_tokens": k})
            print(f"[mtp] engine up, method={method} k={k}", flush=True)
            break
        except Exception as e:
            last = e
            print(f"[mtp] method={method} failed: {type(e).__name__}: {e}",
                  flush=True)
    else:
        raise SystemExit(f"no MTP method booted: {last}")

    sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    t0 = time.time()
    prompts = ("MIT License\n\nCopyright",
               "Permission is hereby granted, free of charge,",
               "Redistribution and use in source and binary forms",
               "The license terms are:")
    for p in prompts:
        out = llm.generate([p], sp, use_tqdm=False)
        print(f"[gen] {p[:40]!r} -> {out[0].outputs[0].text[:70]!r}",
              flush=True)
    dt = time.time() - t0
    print(f"[decode] {ntok * len(prompts) / dt:.1f} tok/s with MTP k={k} "
          f"(CC1)", flush=True)

    vals = {}
    try:
        from vllm.v1.metrics.reader import Counter, Vector
        for m in llm.get_metrics():
            if "spec_decode" not in m.name:
                continue
            v = m.value if isinstance(m, Counter) else \
                (m.values if isinstance(m, Vector) else None)
            if v is not None:
                vals[m.name] = v
                print(f"[metric] {m.name} = {v}", flush=True)
    except Exception as e:
        print(f"[mtp] metrics reader unavailable: {e}", flush=True)
    drafts = vals.get("vllm:spec_decode_num_drafts")
    draft_t = vals.get("vllm:spec_decode_num_draft_tokens")
    acc = vals.get("vllm:spec_decode_num_accepted_tokens")
    if draft_t and acc is not None:
        print(f"[mtp] acceptance rate = {acc / draft_t:.3f} "
              f"({acc}/{draft_t} draft tokens)", flush=True)
    if drafts and acc is not None:
        print(f"[mtp] mean accepted len = {1 + acc / drafts:.3f} "
              f"(ceiling {1 + k})", flush=True)
    print(f"FRUIT-MTP-OK kv={kv} k={k}", flush=True)


if __name__ == "__main__":
    main()
