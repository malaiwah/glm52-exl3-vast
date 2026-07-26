# Cost-controlled appliance test plan

This plan validates one immutable image on Vast.ai and Runpod without paying
to load the production GLM checkpoint or downloading the 27B Qwen checkpoint
twice. Live rentals are sequential, use one GPU each, and are destroyed as
soon as their evidence is collected.

## Guardrails

- Publish the test image under its Git commit SHA; never replace `latest` from
  a manual workflow dispatch.
- Use one on-demand GPU at a time and record the provider's hourly price before
  accepting the rental.
- Prefer an RTX 5090 because the pinned CUDA image targets Blackwell. Set an
  automatic provider termination deadline where supported.
- Use a 60 GB local disk on Vast and a 50 GB container disk plus 20 GB
  `/workspace` volume on Runpod.
- Use `Qwen/Qwen3.5-0.8B` through the `custom` profile with an 8K context. Its
  small download keeps the live test short while exercising the Qwen3.5
  architecture supported by the pinned vLLM runtime.
- Store API keys only in process environment variables. Never put credentials
  in manifests, logs, test artifacts, commits, or shell history.
- Record every created instance/Pod ID immediately. Terminate rather than stop
  after the final check so storage billing also ends.

## Coverage matrix

| area | local / build | Vast live | Runpod live |
|---|---:|---:|---:|
| Bash, Python, workflow, JSON, Docker build | yes | image pull | image pull |
| GLM, Qwen 27B, and custom profile resolution | yes | custom | custom |
| Qwen 27B architecture/quant/MTP metadata guard | yes | — | — |
| Provider detection and generated endpoint | — | yes | yes |
| GPU count/name guard | fixtures | yes | yes |
| Key-only SSH and port forwarding | config lint | yes | yes |
| Weight download and checkpoint-specific marker | fixtures | yes | yes |
| Persistent API key, dashboard token, compile/model cache | — | restart | restart |
| Dashboard boot status and model-neutral snippets | render tests | yes | yes |
| Dashboard token rejection and accepted view | handler tests | yes | yes |
| API authentication (`401` without key, success with key) | — | yes | yes |
| `/health`, `/v1/models`, chat, streaming, usage details | — | yes | yes |
| Qwen reasoning and automatic tool-call parser | — | yes | yes |
| Qwen text-only mode | — | yes | yes |
| Native Qwen vision mode | — | one provider | one provider if time remains |
| `qwen3_next_mtp` speculative decoding | config | one provider | one provider if time remains |
| Supervisor recovery after terminating the engine child | — | yes | yes |
| Vast readiness label | — | yes | — |
| Runpod HTTPS proxy and dashboard URL | — | — | yes |
| Direct TLS/ACME configuration validation | yes | no DNS credential | proxy TLS |
| Runpod long-request SSH-tunnel route | — | — | yes |
| Teardown and no remaining billable resource | — | yes | yes |

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
SPECULATIVE_CONFIG={"method":"qwen3_next_mtp","num_speculative_tokens":2}
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
10. The provider resource is terminated and no test instance/Pod or attached
    billable storage remains.

## Explicit residual tests

The one-GPU budget cannot validate GLM TP4/DCP4, the EXL3/MTP78 graft,
GLM-specific vision graft, DRAM KV offload at production scale, or 512K
correctness. The real 27B NVFP4 profile also requires a separate performance
and memory qualification before its conservative 32K default is raised. These
are release qualification tests, not economical provider-integration smoke
tests.
