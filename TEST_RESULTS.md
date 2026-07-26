# Appliance test results — 2026-07-26

This report records the cost-controlled execution of `TEST_PLAN.md`. Provider
credentials and generated appliance tokens were kept in process-local
environment variables and are not included here.

## Artifacts

- Branch: `codex/appliance-live-test`
- Final runtime-image commit:
  `4e35fdf5c1815c118a9d721e7e40a2b23ac42c93`
- Final immutable image:
  `ghcr.io/malaiwah/glm52-exl3-vast:4e35fdf5c1815c118a9d721e7e40a2b23ac42c93`
- Final image digest:
  `sha256:8b5e957a8328039762616944eafebddd681c1acc35f6b33fa041f10852fc05d9`
- Registry payload: 46 layers, 12,493,859,286 compressed bytes (11.6 GiB);
  approximately 30 GB unpacked.
- Final build:
  <https://github.com/malaiwah/glm52-exl3-vast/actions/runs/30224142616>

The final build passed lint and image publication with SHA-pinned Node 24
actions, least-privilege permissions, per-ref concurrency, timeouts, maximum
provenance, and an SBOM. Manual dispatch published the immutable SHA tag and
did not replace `latest`.

## Local and browser checks

Passed:

- ShellCheck, Actionlint, Python bytecode compilation, JSON parsing, and
  whitespace/error checks.
- Profile-only resolution for GLM-5.2, Qwen3.6-27B NVFP4, and a custom model.
- Secure/insecure landing-page render conditions and JavaScript syntax.
- Desktop and 390-pixel mobile browser rendering with no horizontal overflow.
- Token-gated dashboard behavior.
- Two-turn mocked streaming chat.
- GLM `reasoning_content` and current vLLM/Qwen `reasoning` deltas.
- Reasoning-only Qwen output becomes a visible, non-empty assistant turn.
- `Preserve thinking` defaults off, can be enabled, and does not duplicate a
  reasoning-only response in later context.
- Clear resets both the visible conversation and request history.

## Vast.ai live execution

The first candidate was terminated when its image pull remained incomplete
for more than 17 minutes. A faster RTX 5090 host then started a cached image in
about 51 seconds; the 0.8B checkpoint downloaded in about 6 seconds and vLLM
became ready after its normal model/JIT initialization.

Passed on the fast host:

- Docker-entrypoint launch mode and one-GPU custom profile.
- Provider detection, mapped API/dashboard/SSH ports, and READY label.
- GPU guard, checkpoint download marker, and model-neutral landing content.
- Dashboard missing/wrong-token rejection and valid-token rendering.
- API `401` without its key and success with the persisted key.
- `/health`, `/v1/models`, `/metrics`, non-streaming chat, SSE streaming,
  final usage chunk, Qwen reasoning parser, and forced tool-call parsing.
- Key-only SSH and an authenticated API request through an SSH tunnel.
- Engine-child termination followed by recovery and acceptance of the same
  persisted API key.
- Provider stop/start with the same checkpoint marker and byte size, no
  weight re-download, and the same API key.

Live evidence exposed two issues that were fixed afterward:

1. Current vLLM uses `reasoning`; the UI only read the legacy
   `reasoning_content` field.
2. The upstream image exports removed `VLLM_CACHE_DIR`; the entrypoint now
   unsets it while retaining supported `VLLM_CACHE_ROOT`.

Vast credit moved from approximately `$21.8431` to `$21.6435`, for about
`$0.20` total including the deliberately aborted slow-host attempts. The final
Vast API check returned zero instances.

### deSEC dynamic DNS and direct TLS follow-up

The supplied DNS token exposed one zone. A unique documentation-address RRset
was created using the appliance's atomic bulk `PUT`, resolved from deSEC's
authoritative nameserver, updated, deleted, and confirmed absent through both
the API and authoritative DNS.

A second short Vast RTX 5090 run exercised the appliance path itself:

- `model-<instance-id>` registered to the observed public IPv4 address.
- lego created the DNS-01 TXT challenge, Let's Encrypt validated it, and lego
  removed the TXT RRset afterward.
- The issued certificate had the generated hostname as both CN and SAN, a
  trusted Let's Encrypt chain, and a 90-day validity window.
- The direct mapped HTTPS endpoint passed `/health`, authenticated
  `/v1/models`, and a Qwen chat completion.
- The token-gated dashboard accepted TLS on its mapped port; an unauthenticated
  request correctly returned 403.
- The Vast label became
  `Qwen3.5-0.8B-Dynamic-DNS-Smoke READY https://.../v1`, making readiness and
  the final endpoint visible in the console.

The follow-up used about `$0.06` of Vast credit. The rental was destroyed, its
A record was deleted, the record returned API 404, and the final Vast inventory
again returned zero instances.

## Runpod live execution

Static REST/template validation passed, and authenticated Runpod inventory and
GPU-stock queries worked.

Placement results:

- Two Community create attempts returned Runpod's machine-resource HTTP 500
  before a Pod ID was allocated; inventory remained empty.
- Secure RTX 4090 stock reported `High` at `$0.69/hour`.
- Two Secure Pods were allocated sequentially. Both remained in provider image
  provisioning with `runtime: null`, no public IP/port mappings, and proxy
  HTTP 404. They were terminated at 228 and 437 seconds respectively.
- One attempt used the provenance/SBOM image index and one used the prior
  single-platform immutable image manifest. Neither reached the container
  entrypoint, so the manifest form did not explain the delay.

Estimated Runpod compute exposure was about `$0.13` before storage rounding.
The final Runpod REST inventory returned zero Pods.

Because Runpod never started the container, Runpod-specific runtime checks
(proxy TLS, generated URLs, SSH, restart persistence, vision, MTP, and
supervisor recovery) are **blocked by cold-image provisioning**, not passed.
They must not be inferred from the successful Vast run or the static template
checks.

## Coverage summary

| Area | Result |
|---|---|
| Build, profiles, manifests, local UI | Passed |
| Vast provider integration and baseline appliance features | Passed |
| deSEC API lifecycle, DNS-01 cleanup, and trusted direct TLS | Passed on Vast |
| Qwen reasoning/tool parser with small live model | Passed on Vast |
| UI reasoning compatibility and multi-turn behavior | Fixed and browser-tested |
| Runpod template/API schema and placement handling | Passed |
| Runpod container/runtime compatibility | Blocked before entrypoint |
| Small-model vision and `qwen3_next_mtp` | Not reached |
| GLM TP4/DCP4, EXL3/MTP78, 512K, production vision/offload | Explicit release qualification gap |
| Final provider resources | Vast: 0; Runpod: 0 |

## Next economical live pass

Do not repeat a blind Runpod rental. First arrange one of:

1. a Runpod machine known to cache the pinned base image;
2. a Runpod-supported registry/cache path that can deliver the approximately
   11.6 GiB compressed appliance image within the chosen cold-start budget; or
3. a deliberately slimmed base image that preserves the required vLLM fork
   and Blackwell kernels.

Then reuse the same 0.8B profile and execute only the still-uncovered Runpod,
vision, and MTP rows from `TEST_PLAN.md`.
