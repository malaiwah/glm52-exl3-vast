# Cost-controlled appliance test plan

Reusable qualification protocol for every release. Results are recorded in
[TEST_RESULTS.md](TEST_RESULTS.md); per-release GLM model qualification
details are in the `docs/glm52-rXX-*.md` files. Next release: **full non-Flash
GLM-5.3 3.42bpw 520K candidate**, not yet GPU-qualified. Earlier r26/r28
results retain their historical model and image scope.

This plan has two cost tiers. A sub-1B Qwen representative validates provider,
UI and OpenAI-API plumbing on one GPU. A separate authorized four-GPU full-model
pass validates the immutable candidate image, loader, preserved 520,192-token
total envelope, performance and power. Do not substitute Flash or GLM-5.2
evidence for the full GLM-5.3 gate. Rentals are sequential and deleted only
after their evidence is copied; existing owned-host production is not a rental
to stop or destroy.

## Full GLM-5.3 candidate maintenance gate

The release intent is `MODEL_PROFILE=glm53-3.42bpw-500k`, variant
`exl3-tr3-glm53-3.42bpw-500k`, family `glm52`, checkpoint
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@99c6f951333d2b38f1efefa533c7afadf0d376e3`.
Its 81 weight LFS identities match evaluated `8bef807a0fcdd180e984a26b50e731cdba9a8ff2`;
the changed template still needs the parser/continuation gate.

The candidate image is now rooted on local-inference-lab's Gilded Gnosis v20
r34 (`docker.io/voipmonitor/vllm@sha256:820181fb…`, CUDA 13.2.1 / Torch
2.12.0+cu132 / NCCL 2.30.4) plus the reviewed r34 maintenance sources and this
repository's fail-closed overlays. That is the same runtime family currently
serving production, so its cold-boot, cache and recovery evidence must still be
re-measured for GLM-5.3 weights, but a CUDA/Torch/NCCL platform change is no
longer part of this gate. The earlier VerdictAI-based candidate is superseded;
its published digest remains the GLM-5.3-Flash lineage only.

Use [maintenance/glm53-aibeast-500k](maintenance/glm53-aibeast-500k/) to stage
an immutable candidate independently. Podman 4.9 uses inventoried manual
devices, GPU order 2,1,0,3, separate port/name/state/cache, and restart disabled.
Staging and CPU inspection do not authorize a GPU window or production stop.
The runner must refuse to start while production occupies the GPUs. Preserve
the old image digest, launch environment, state and cache; rollback starts
that preserved service rather than rewriting it to the new defaults.

The runnable sequence is `candidate.py stage`, `verify-model`, `preflight`
and `config-smoke`; `start` is permitted only after an operator has deliberately
stopped production in the authorized window. Then run:

```bash
RUN_GPU_QUALIFICATION=1 uv run --no-project --python 3.12 maintenance/glm53-aibeast-500k/candidate.py run-qualification
```

The runner produces receipts and invokes the evidence gate; do not fabricate
or manually fill success JSON. Its real-GPU matrix includes features,
C1/C4/C8, >=500K retrieval, cache transfer/restart/miss and 384-GiB L2 pressure,
and deliberate LMCache/engine-group failure recovery. L2 eviction pressure is
expensive and bounded at 512 requests of roughly 131K tokens; absence of
observed eviction fails rather than silently skipping that gate.

1. After image build, run the provenance generator with the serving interpreter
   and final installed overlay/cache files. Extract `/opt/runtime-provenance.json`,
   compare critical hashes with the files actually inside the image, and record
   the separately obtained pushed digest. Missing required critical source,
   NCCL or ExLlama native files fail admission. No source bind mounts in the gate.
2. Confirm TP4/DCP4, native MTP3, online K6, dynamic NVFP4/FP8 RoPE,
   4,518,907,904 KV bytes/GPU, scheduler 2048, prefill arena 1024, C8 and GMU 0.93.
   Confirm fraction-derived LMCache RAM is capped at 125 GiB with 20 GiB lazy
   initialization. DRAM/NVMe cache sizes do not count as active-context capacity.
3. Record cold-boot and first-use physical memory/allocator data, including
   top-p temperature-1 sampling, all 512 requested output tokens, strict JSON
   and tool paths. Require healthy API and zero engine restarts.
   The isolated stage sets `VERIFY=0`; the runner must own the first API sample,
   require zero previous/in-flight requests, unchanged engine generation and
   cold logs free of startup failures/retries. A later warmed success cannot
   certify a failed cold attempt.
   The AIBeast profile deliberately uses `AUTH=none` and `LANDING_PAGE=0` on a
   trusted host network; record that exposure and verify its firewall/tunnel
   boundary. Test authenticated API/dashboard behavior in the separate
   provider/authenticated profile, not as a false claim about this stage.
4. Exercise streaming and nonstreaming literal tool delimiters, stripped stop
   text, null content, multi-tool continuation and reordered result IDs using
   the new template and installed parser, not a standalone regex surrogate.
5. Run tokenizer-exact unique-prefix multi-depth retrieval at increasing
   lengths, ending with **at least 500,000 actual input tokens plus useful
   output** within 520,192 total tokens. Record actual API usage, output facts,
   health and error/preemption evidence over multiple seeds. A configured
   context value, logical pool count or available-memory estimate cannot pass.
6. Exercise C1/C4/C8, long cache-miss prefill concurrent with active decodes,
   cancellation, sampler paths and multi-turn tool workloads. Record longest
   output gap and prefill progress; do not infer eight full-length concurrent
   requests from C8 short-prompt throughput.
7. Validate cold/warm DRAM and L2 prefix retrieval, miss/recompute, observed
   L2 eviction, and deliberate cache/engine-group failure recovery. Confirm
   external-hit tokens and actual transfers, then restart the same image with
   persistent candidate caches and repeat the boundary gate. The runner's
   deadline-ownership proof uses CPU-controlled futures with the actual
   installed adapter; it is not a 180-second live GPU DMA-hang injection.
   Keep that scoped proof distinct from real cache-transfer/recovery receipts.
8. Re-run core correctness and log audit after stress and a second boot of the
   **published digest**. Only complete exact-stage receipts permit official
   release/`latest` promotion. Reboot enablement and production cutover remain
   explicit later operator actions. Keep rollback usable throughout.

No 750K capacity promise is part of this candidate. Additional context or
different KV formats require a separately scoped measured ladder.

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
  600 GB for the full GLM checkpoint, image and compile caches; use at least
  650 GB for a JarvisLabs full VM. Optional bounded L2 needs additional local
  space; do not count the same disk margin twice.
- Use `Qwen/Qwen3.5-0.8B` through the `custom` profile with an 8K context. Its
  small download keeps the live test short while exercising the Qwen3.5
  architecture supported by the pinned vLLM runtime.
- Store API keys only in process environment variables. Never put credentials
  in manifests, logs, test artifacts, commits, or shell history.
- Record every created instance/Pod/VM ID immediately. Terminate or destroy
  rather than stop/pause after the final check so storage billing also ends.
- For the SOUL composite test, launch with `SOUL_AUTONOMY_MAX_LEVEL=3` and
  `TERMINATE_ENABLED=1`. Exercise levels 1, 2, then 3 early; leave level 3
  selected for the remaining workload and through the start of teardown.

## Coverage matrix

| area | local / build | Vast live | Runpod live | JarvisLabs live |
|---|---:|---:|---:|---:|
| Bash, Python, workflow, JSON, Docker build | yes | image pull | image pull | image pull |
| GLM, Qwen 27B, and custom profile resolution | yes | custom | custom | flagship |
| Qwen 27B architecture/quant/MTP metadata guard | yes | yes | — | — |
| Provider detection and generated endpoint | — | yes | yes | yes |
| GPU count/name guard | fixtures | yes | yes | yes |
| Key-only SSH and port forwarding | config lint | yes | yes | yes |
| Weight download and checkpoint-specific marker | fixtures | yes | yes | yes |
| Persistent API key, dashboard token, compile/model cache | — | restart | restart | restart |
| Dashboard boot status and model-neutral snippets | render tests | yes | yes | yes |
| Dashboard token rejection and accepted view | handler tests | yes | yes | yes |
| API authentication (`401` without key, success with key) | — | yes | yes | yes |
| `/health`, `/v1/models`, chat, streaming, usage details | — | yes | yes | yes |
| Qwen reasoning and automatic tool-call parser | — | yes, full 27B | yes | — |
| Qwen text-only mode | — | yes, full 27B | yes | — |
| Native Qwen vision mode | — | one provider | one provider if time remains | — |
| Qwen MTP speculative decoding | config | eager MTP2 passed; compiled path rejected | one provider if time remains | — |
| GLM InstantTensor cold boot and AOT-cache reuse | config | yes | yes | yes |
| GLM feature suite at production scale, including strict JSON with thinking | harness | yes | yes | yes |
| GLM cold prefill and sustained C1/C2/C4/C8 decode | harness | yes | yes | yes |
| Per-phase GPU power and power-limit telemetry | harness | yes | yes | yes |
| Exact ~517K five-depth needle and degeneration gate | harness | yes | yes | yes |
| Supervisor recovery after terminating the engine child | — | yes | yes | yes |
| Vast readiness label | — | yes | — | — |
| Runpod HTTPS proxy and dashboard URL | — | — | yes | — |
| JarvisLabs direct-IP VM bootstrap and NCCL fallback | shell lint | — | — | yes |
| deSEC RRset create/update/delete and authoritative propagation | — | yes | yes | yes |
| DNS-01 issuance, challenge cleanup, and trusted certificate | yes | yes | yes | yes |
| Hybrid Runpod networking (`1111/http`, `8000/http` fallback, `8443/tcp` TLS) | config | — | yes | — |
| Direct TLS plus SSH-tunnel fallback | config | yes | yes | yes |
| Appliance-initiated typed teardown and no remaining billable resource | tests | yes | yes | yes |

## Live launch profile

```text
MODEL_PROFILE=custom
MODEL_ID=Qwen/Qwen3.5-0.8B
MODEL_DOWNLOAD_GIB=2
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
   zero Vast/Runpod/JarvisLabs resources, and retain only bounded,
   credential-free JSON evidence.

Release goals on an all-NODE 4x96 GB host are at least 2,500 prompt tokens/s,
100 C1 output tokens/s, useful aggregate scaling through C8, one usable
512K–520K solo session, and clean maximum-context retrieval. A rental with
`SYS` GPU paths can validate the configuration but cannot fail the absolute
throughput goals.
