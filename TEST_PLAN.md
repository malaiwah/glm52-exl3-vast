# Cost-controlled appliance test plan

This plan has two cost tiers. A sub-1B Qwen representative validates provider,
UI and OpenAI-API plumbing on one GPU. A single four-GPU flagship pass per
provider then validates the immutable GLM image, loader, 520K envelope,
performance and power without repeating every failure experiment. Rentals are
sequential and are deleted as soon as their evidence is copied.

## Guardrails

- Publish the test image under its Git commit SHA; never replace `latest` from
  a manual workflow dispatch.
- Use one on-demand GPU for provider/UI smoke tests and exactly four cards only
  for the final GLM profile. Record the hourly price before accepting either.
- Require an RTX 5090 or RTX PRO 6000 Blackwell because the pinned CUDA image
  and custom kernels target `sm120+`. Do not substitute an RTX 4090/Ada GPU
  merely because the smoke-test model fits. Set an automatic provider
  termination deadline where supported.
- Use a 60 GB local disk on Vast and a 50 GB container disk plus 20 GB
  `/workspace` volume on Runpod for the smoke profile. Allocate at least
  450 GB for the GLM checkpoint and evidence.
- Use `Qwen/Qwen3.5-0.8B` through the `custom` profile with an 8K context. Its
  small download keeps the live test short while exercising the Qwen3.5
  architecture supported by the pinned vLLM runtime.
- Store API keys only in process environment variables. Never put credentials
  in manifests, logs, test artifacts, commits, or shell history.
- Record every created instance/Pod ID immediately. Terminate rather than stop
  after the final check so storage billing also ends.
- For the SOUL composite test, launch with `SOUL_AUTONOMY_MAX_LEVEL=3` and
  `TERMINATE_ENABLED=1`. Exercise levels 1, 2, then 3 early; leave level 3
  selected for the remaining workload and through the start of teardown.

## Coverage matrix

| area | local / build | Vast live | Runpod live |
|---|---:|---:|---:|
| Bash, Python, workflow, JSON, Docker build | yes | image pull | image pull |
| GLM, Qwen 27B, and custom profile resolution | yes | custom | custom |
| Qwen 27B architecture/quant/MTP metadata guard | yes | yes | — |
| Provider detection and generated endpoint | — | yes | yes |
| GPU count/name guard | fixtures | yes | yes |
| Key-only SSH and port forwarding | config lint | yes | yes |
| Weight download and checkpoint-specific marker | fixtures | yes | yes |
| Persistent API key, dashboard token, compile/model cache | — | restart | restart |
| Dashboard boot status and model-neutral snippets | render tests | yes | yes |
| Dashboard token rejection and accepted view | handler tests | yes | yes |
| API authentication (`401` without key, success with key) | — | yes | yes |
| `/health`, `/v1/models`, chat, streaming, usage details | — | yes | yes |
| Qwen reasoning and automatic tool-call parser | — | yes, full 27B | yes |
| Qwen text-only mode | — | yes, full 27B | yes |
| Native Qwen vision mode | — | one provider | one provider if time remains |
| Qwen MTP speculative decoding | config | eager MTP2 passed; compiled path rejected | one provider if time remains |
| GLM v31 InstantTensor cold boot and AOT-cache reuse | config | yes | yes |
| GLM feature suite at production scale, including strict JSON with thinking | harness | yes | yes |
| GLM cold prefill and sustained C1/C2/C4/C8 decode | harness | yes | yes |
| Per-phase GPU power and power-limit telemetry | harness | yes | yes |
| Exact ~517K five-depth needle and degeneration gate | harness | yes | yes |
| Supervisor recovery after terminating the engine child | — | yes | yes |
| Vast readiness label | — | yes | — |
| Runpod HTTPS proxy and dashboard URL | — | — | yes |
| deSEC RRset create/update/delete and authoritative propagation | — | yes | yes |
| DNS-01 issuance, challenge cleanup, and trusted certificate | yes | yes | yes |
| Hybrid Runpod networking (`1111/http`, `8000/http` fallback, `8443/tcp` TLS) | config | — | yes |
| Runpod long-request SSH-tunnel route | — | — | yes |
| Appliance-initiated typed teardown and no remaining billable resource | tests | yes | yes |

## Live launch profile

```text
MODEL_PROFILE=custom
MODEL_ID=Qwen/Qwen3.5-0.8B
MODEL_DOWNLOAD_GIB=2
MODEL_DISPLAY_NAME=Qwen3.5-0.8B Smoke
SERVED_MODEL_NAME=qwen-smoke
MAX_MODEL_LEN=8192
MODEL_OUTPUT_LIMIT=1024
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=2048
MULTIMODAL=0
REASONING_PARSER=qwen3
TOOL_CALL_PARSER=qwen3_coder
LANDING_PAGE=1
```

The initial phase is text-only with speculation disabled. After the baseline
passes, restart once with:

```text
MULTIMODAL=1
SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2}
```

If combining vision and MTP obscures a failure, test them separately on the
same already-rented machine.

## Acceptance checks

1. Image reaches the `ready` phase without inherited GLM backends or scale
   files appearing in the effective environment.
2. The unauthenticated model request is rejected and the authenticated request
   lists only `qwen-smoke`.
3. A non-streaming chat returns the requested sentinel; a streaming chat
   produces deltas and a final usage chunk.
4. Reasoning content is separated when emitted, and a forced tool request
   produces a valid OpenAI tool-call structure.
5. The dashboard rejects a missing/bad token and renders the correct model,
   profile, context, endpoint, and client snippets with the valid token.
6. SSH accepts the account key, rejects password authentication, and can carry
   a working tunnel to port 8000.
7. Killing the vLLM child produces a supervised restart and the health endpoint
   returns without changing the API key.
8. A provider stop/start preserves `/workspace`, skips the weight download,
   and preserves the API/dashboard keys.
9. Vision identifies a simple public test image; MTP boots and completes a
   deterministic short prompt without a crash.
10. After all evidence has been copied, leave SOUL at level 3 and use the
    landing page's token-gated **Terminate instance** flow. Type the exact
    provider id, acknowledge destruction, select session erase, and let the
    appliance stop SOUL/the engine, erase bounded session state, and issue the
    provider destroy call. Verify externally that the instance/Pod is gone,
    no billable storage remains, and its temporary DNS RRset was deleted.

## Explicit residual tests

The real 27B Qwen NVFP4 one-card matrix is complete on Vast RTX 5090: 192K
context, detailed native vision, full API features, throughput, speculation
controls, and near-maximum retrieval passed. Repeating the final performance
rows on Runpod remains a provider-comparison task, not a profile blocker. GLM
vision is still a separate opt-in feature profile: the current EXL3 graft
passes detailed short screenshot extraction but fails the mandatory 32K text
gate, so it is not allowed to borrow the flagship text profile's 520K claim.

## Flagship GLM-5.2 qualification

The production pass reuses one 4x RTX PRO 6000 Blackwell rental and downloads
both GLM variants once. Provider plumbing is not repeated: the economical
matrix above already gates Vast, Runpod, DNS, TLS, proxying, SSH, restart,
persistence, and teardown. The expensive pass concentrates on model-specific
correctness, memory, and performance.

Execute in this order so a failure cannot contaminate later conclusions:

1. Record immutable image/checkpoint revisions, GPU/NUMA/PCIe topology, P2P
   capabilities, driver, CUDA, vLLM, B12X, free disk, and wall-clock startup
   phases. Treat a mixed-root rental as a correctness and relative A/B host,
   not an absolute AIBeast performance proxy.
2. Establish one stable loader, target, MTP, KV, batch, workspace, graph, and
   pool baseline. Require three consecutive uncached 32K retrieval probes
   before changing more than one parameter at a time.
3. Sweep prefill chunk/workspace, lossless PCIe-DMA crossover, DCP query split,
   CKV prefetch depth, MTP depth/proposal method, and MTP-off control. Reject
   any arm that boots but later OOMs, degenerates, or loses retrieval.
4. On the selected arm, run authenticated discovery/tokenization, ordinary
   and thinking chat, SSE usage, multi-turn with optional preserved thinking,
   automatic tool call and tool-result continuation. Strict JSON-schema output
   must pass both with thinking disabled and across the thinking-to-answer
   boundary; run concurrent requests and reject any HTTP failure, invalid
   schema, or genuine committed-token FSM failure. Forced-tool mode and vision
   remain separate diagnostics rather than flagship text-profile release gates.
5. Measure unique-prefix prefill at 1K/8K/32K and aggregate decode at
   C1/C2/C4/C8. Record TTFT, output throughput, failures, preemptions, mean
   speculative acceptance length, per-position acceptance, and the exact
   request shape. Periodic vLLM logger buckets are diagnostic only.
6. Run seeded needles at 32K and near maximum context, including depths near
   both ends. A clean short needle does not qualify 512K. Preserve partial
   results atomically so a late failure does not erase hours of evidence.
7. Only after that stable baseline, cold-start the InstantTensor loader at
   least three times. Keep it opt-in unless all starts, warmups, and first
   uncached requests pass; loader speed cannot compensate for a race.
8. Compare the EXL3 and MadeBy561 variants on the same host. Compare the final
   image read-only against the owned AIBeast v19 daily-driver control, then
   repeat the winning v20 image on AIBeast before claiming absolute production
   throughput.
9. Destroy the rental, delete its DNS record and temporary credentials, verify
   zero Vast/Runpod resources, and retain only bounded, credential-free JSON
   evidence.

Release goals on an all-NODE 4x96 GB host are at least 2,500 prompt tokens/s,
100 C1 output tokens/s, useful aggregate scaling through C8, one usable
512K–520K solo session, and clean maximum-context retrieval. A rental with
`SYS` GPU paths can validate the configuration but cannot fail the absolute
throughput goals.
