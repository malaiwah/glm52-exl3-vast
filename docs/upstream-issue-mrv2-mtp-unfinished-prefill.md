# [perf][MRV2/MTP] Hydrate draft KV but skip discarded logits/proposals for unfinished chunked prefills

## Summary

On the MRV2 MTP path, unfinished chunked-prefill requests still run target
vocabulary projection/sampling and MTP proposal generation even though the
scheduler emits no token and explicitly discards their draft proposals.

Upstream vLLM PR #49171 optimizes the common no-scheduled-drafts target path,
but states that speculative decoding is unchanged. It does not cover mixed
decode/prefill batches, and it does not avoid dead work in the MTP drafter.

Reviewed source:
`local-inference-lab/vllm@e2666d9a65f41fc376607531453cbd57c4c71016`

Deployment motivating this report:

- GLM-5.2 EXL3-TR3
- MRV2
- TP4/DCP4 over PCIe
- MTP3 probabilistic
- `max_num_batched_tokens=3072`
- up to 12 concurrent requests
- approximately 154K vocabulary

This is a static finding; no performance gain is claimed until a matched GPU
A/B.

## Evidence

### Target

`vllm/v1/worker/gpu/model_runner.py:1225-1271` assigns one bonus-logit row per
request. When any request has scheduled drafts:

```python
total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
num_logits = num_draft_tokens_per_req + num_bonus_tokens
```

Thus unfinished-prefill rows in a mixed batch still receive a target LM-head
row.

`model_runner.py:1460-1497` then unconditionally executes `compute_logits()`
and the regular or rejection sampler.

Only afterward, `vllm/v1/worker/gpu/input_batch.py:447-493`, sets
`num_sampled=0` when `seq_len < prefill_len`.

### MTP drafter

`model_runner.py:2007-2041` invokes the speculator for every batch.

`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:445-483` runs the
required MTP prefill forward but also computes and samples the first draft token
for every request.

At MTP3, `speculator.py:485-510` runs two additional draft decode/sample steps.

The scheduler then explicitly discards these results:

```python
if request.is_prefill_chunk:
    # Ignore draft tokens for prefill chunks.
    ...
    continue
```

(`vllm/v1/core/sched/scheduler.py:2111-2131`)

## Required distinction

The MTP prefill forward itself is not waste: it hydrates the MTP KV cache. The
optimization must skip only output projection, sampling and speculative
lookahead for rows that remain unfinished.

The final prefill chunk must continue to produce the target token and MTP
drafts.

## Proposed work

1. Adapt the compact target-logits machinery from upstream PR #49171.
2. Extend the speculative branch to use a per-request bonus count:
   - one when `num_computed + num_scheduled >= prefill_len`;
   - zero otherwise.
3. Add a safe all-unfinished MTP fast path:
   - run draft prefill forward and write draft KV;
   - skip draft `compute_logits`, sampling and the remaining MTP decode steps;
   - return no draft proposals.
4. Treat mixed-batch MTP row compaction as a follow-up if the simpler fast path
   proves valuable.

## Correctness risks

- Do not skip or alter MTP KV hydration.
- `combine_sampled_and_draft_tokens` currently assumes a uniform per-request
  bonus count.
- Verify zero-width rows in rejection sampling.
- Mask online speculative statistics for unfinished rows.
- Do not publish stale verification-capacity/confidence state.
- Preserve prompt-logprobs, LoRA, grammar and async-output behavior.
- Warm every new CUDA-graph/Triton variant at startup.

## Test plan

### Unit/static

- Pure unfinished-prefill batch: zero target logits rows.
- Mixed decode + unfinished prefill: zero rows only for unfinished requests.
- Final prefill chunk: exactly one target bonus row and normal MTP proposal.
- MTP hydrate-only: draft KV after the chunk matches the baseline path.
- Scheduler receives no drafts for unfinished requests.
- No negative or stale speculative statistics.
- Prompt logprobs, LoRA, structured output and async scheduling regressions.

### GPU A/B

Run matched unique-prefix prompts at C1/C4/C8/C12 with 8K, 64K, 128K and
near-500K inputs, batch budget 3072:

- PP and TTFT.
- TG and MAL after prefill completes.
- Target logits/sampler duration.
- MTP prefill-forward, draft-logits and draft-decode duration.
- PCIe collective bytes.
- Peak and steady VRAM.
- Five-depth needles, second seed and degeneration gate.

At approximately hidden=6,144 and vocab=154K, C12/MTP3 can discard up to 12
target rows plus 36 draft rows per unfinished step: roughly 90.7 GFLOP globally
in LM-head projections, about 14.1 MiB of gathered BF16 logits before FP32
sampling temporaries, plus 24 one-token draft forwards. This is meaningful
small-M and launch/collective work, but small relative to the 3,072-token
target-plus-MTP prefill. An honest GLM expectation is approximately 1-4% at
concurrent long prefill; 5% is an optimistic upper bound pending profiler
evidence.

## Related work

- Upstream vLLM PR #49171: target compaction for the common path.
- local-inference-lab/vllm#206: separate MTP greedy-local-argmax and memory
  findings; not a duplicate.
