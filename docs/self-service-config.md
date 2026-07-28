# Self-service configuration

A user rents the template, the image is locked in, and they get working
defaults. Everything after that — model variant, draft type, speculation depth,
context length, KV dtype, concurrency, DRAM offload, vision — is changeable
from the landing page on `:1111`, applied by restarting vLLM only. No image
rebuild, no re-rent, no SSH.

The parts:

| file | role |
|---|---|
| `scripts/glm_config.py` | knob registry, three-layer resolution, pre-validation matrix, failure signatures. Imported by everything. |
| `scripts/config_cli.py` | the shell-facing side: `env`, `show`, `validate`, `mark-good`, `should-rollback`, `rollback`, `pending-analysis`. |
| `scripts/verify_serving.py` | health + short prompts + **long-context needle probe**. Decides whether a config is good. |
| `scripts/analyze_failure.py` | asks the running model to explain the config that failed. |
| `scripts/soul_controller.py` | optional embedded Nanobot monitoring, incident interpretation, and journal. At levels 1–3 it replaces the duplicate direct rollback analyzer call. |
| `scripts/soul_config.py` | independent SOUL configuration, ceiling, redaction, atomic state, journal pagination, and audit helpers. |
| `scripts/reconcile_checkpoint.py` | makes `config.json` + the weight index match the tensors actually on disk. |
| `landing.py` | `/config` editor, `/config/apply`, `/config/import`, `/config/export`, `/config/reset`, `/config/status`. |
| `entrypoint.sh` | resolves the config before **every** vLLM start; runs the trial / verify / rollback state machine. |

---

## 1. Inheritance model

Lowest to highest precedence:

```
built-in defaults  <  startup environment  <  JSON state file on the volume
   glm_config.py         template env, frozen           $GLM_STATE_DIR/config.json
                         at container start             written by the landing page
```

**Why the file wins over env.** The environment comes from the rental template.
Once the instance exists, the user cannot change it without destroying and
re-renting. The landing page is the only post-launch control surface, so it has
to be able to override what the template set — otherwise the template's
`MAX_NUM_SEQS=32` would be unfixable from inside the box.

**Why the env layer is frozen at boot.** `config_cli.py snapshot-env` runs
before the entrypoint exports anything of its own and writes
`$GLM_RUNTIME_DIR/startup-env.json`. Every later resolution reads that snapshot,
not the live environment. Without this, the entrypoint exporting
`MAX_MODEL_LEN` for the serve command would make itself look like the user's
env layer on the next restart, and the layering would quietly become
self-referential.

**The state file is a diff, not a snapshot.** `glm_config.minimize()` stores
only the knobs whose value differs from `defaults + env`. Writing all of them
would freeze the instance against its own template (a knob the user never
touched would start winning over the operator's env), and it makes an exported
config portable to an instance launched with different template env.

Legacy spellings still feed the env layer: `MTP78_MODE`, `MTP78_TRELLIS=0`,
`DRAFT_MODEL`, `TUNE_VLLM_EXL3_TRELLIS_MAX_M`. Nothing that worked as an env var
before stopped working.

### Where state lives

| path | contents | lifetime |
|---|---|---|
| `$GLM_STATE_DIR` (`/workspace/.glm-config`) | `config.json`, `known-good.json`, `apply-state.json`, `verify-last.json`, `checkpoint-baseline.json`, `failures/`, `logs/` | the volume — survives container replacement |
| `$GLM_RUNTIME_DIR` (`/tmp/glm-runtime`) | `startup-env.json`, `config.env`, `verify.json`, `restart-request` | the container |
| `$GLM_STATE_DIR/soul` | optional SOUL config, status, JSONL journal, incidents, evidence, snapshots, Nanobot workspace/sessions and logs | the volume — fully selected by secure erase |

The split is deliberate. A restart flag or a verify verdict that survived a
container swap would fire once more against a configuration that was never
applied in that container.

### State file schema

```json
{
  "values":     { "MTP_TOKENS": 5, "MAX_NUM_SEQS": 5, "VISION": false },
  "written_at": "2026-07-26T15:06:17Z"
}
```

`values` holds knob keys from the registry only; unknown keys are ignored with
a note and out-of-range values fall back to the layer below with a note. A
state file that is not valid JSON is ignored entirely rather than
bricking the boot — the note appears in the boot log and on `/config`.

`known-good.json` stores `{ts, values, effective, sources, verify}`. Restoring
it writes `values` back to the state file, so the restored configuration is
reproduced through the same resolution path rather than pinned.

---

## 2. The knobs, and what each trades off

Every knob in `glm_config.KNOBS` carries a `rationale` string; the editor shows
it under the field, and `analyze_failure.py` feeds the rationale of the changed
knobs to the model. Summary of the trade each one makes:

| knob | trades |
|---|---|
| `MODEL_VARIANT` | EXL3-TR3 (provider default), the measured MadeBy561 MXFP8/NVFP4/NF3 hybrid, the legacy experimental NVFP4 checkpoint, or a family-specific development target. A variant may supply a coherent draft/memory preset; switching costs a full re-download. |
| `MTP_DRAFT` | `native` uses the checkpoint's own draft (rank-sliced TR3 in the pinned Brandon revision; serialized NVFP4 experts in MadeBy561). `tr3-graft`, `tr3-override`, external `nvfp4`, legacy `bf16`, and `off` remain controlled compatibility/experiment paths. |
| `MTP_TOKENS` | Depth of speculation. With the cheap trellis draft MTP-5 beats MTP-3 (53.4 vs 51.5 tok/s); with the 19.3 GB BF16 draft it lost 22%. Also widens the decode query width. |
| `MTP_DRAFT_SAMPLE_METHOD` | The measured GLM profiles use `probabilistic`; `greedy` remains explicitly overridable for matched A/B work. |
| `DCP` | The balanced Brandon default uses DCP2 for ordinary prefill/decode while retaining a verified ~517K request. DCP4 is the maximum-context variant; DCP1 prioritizes low-concurrency decode but cannot expose the same context envelope. |
| `DCP_CKV_PREFETCH_DEPTH`, `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | Topology overlap and the context crossover for query splitting. `auto`/`-1` retain calibration; the MadeBy561 profile pins the measured 0/8,192 shape. |
| `F8_DMA`, `PCIE_DMA_MIN_BYTES`, `PCIE_CALIBRATION` | Collective wire format and byte crossover. The family stays lossless/automatic; the MadeBy561 profile pins the 521K-qualified FP8 ring/393,216-byte shape. |
| `KV_CACHE_DTYPE` | calibrated `nvfp4_ds_mla` (GLM default; cross-provider-qualified on v31 and re-gated at each base refresh) vs fp8 (~1.7x bytes/token); models without calibrated MLA scales are refused. |
| `MAX_MODEL_LEN` | Longest request, and a hard startup gate against available KV. |
| `GPU_BLOCKS_OVERRIDE` | 0 auto-profiles the largest safe pool. The portable DCP2/GMU-0.976 release shape exposed 523,264 logical tokens on Runpod; the value varies with usable VRAM, graphs, driver and loader. A positive value pins a reproducible smaller pool. |
| `OFFLOAD_FRACTION` | Host DRAM used as an L2 prefix cache after GPU eviction—not extra active-context capacity. It is passed as one aggregate native-connector budget (vLLM derives the TP worker slices), with `recompute` on miss. At 50% of a 251 GiB host, an evicted 133,504-token prefix reloaded in 0.69s instead of a 52.47s recompute; 50% also retained ~51 GiB host headroom. |
| `VISION` | Image input vs long-context correctness on EXL3 (see rule 6) and ~1.99 GiB/GPU on the final v20 qualification shape. |
| `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` | Concurrency and prefill chunk against the capture window and against VRAM headroom. |
| `MAX_CUDAGRAPH_CAPTURE_SIZE`, `CUDAGRAPH_CAPTURE_SIZES`, `VLLM_EXL3_TRELLIS_MAX_M` | The three ceilings that must move together to serve more streams. |

---

## 3. Validation matrix

Measured on 4x RTX PRO 6000 on 2026-07-26 unless stated. `error` blocks an
apply; `warn` is shown and allowed. Rule ids are the `id` field in the
findings, so the UI, the boot log and this table agree.

### `concurrency-window` — error
`MAX_NUM_SEQS * (1 + MTP_TOKENS)` must be `<= min(MAX_CUDAGRAPH_CAPTURE_SIZE,
VLLM_EXL3_TRELLIS_MAX_M)`, both 64 in the balanced MTP-5 profile. Exceeding it does **not** error
at boot; decode silently leaves the captured trellis fast path under
concurrency and loses throughput. Raising concurrency requires raising
`CUDAGRAPH_CAPTURE_SIZES`, `MAX_CUDAGRAPH_CAPTURE_SIZE` and
`VLLM_EXL3_TRELLIS_MAX_M` together. This is why `MAX_NUM_SEQS` defaults to 8:
`8 * (1+3) = 32` exactly fills the window.

### Role-aware Trellis minimum — fixed in v29
The old base treated target and draft layers as if they shared one minimum.
That made the MTP draft's m=1..3 graph shapes fail unless a global
`VLLM_EXL3_TRELLIS_MIN_M=1` workaround was used. v29 stamps the draft role at
construction: target layers retain m=4 and draft layers advertise a capturable
m=1 floor. The appliance therefore leaves the variable absent. Advanced
experiments can still set `TUNE_VLLM_EXL3_TRELLIS_MIN_M`, but it is not a
self-service profile knob.

### `draft-quant-inherit` — error
A draft inherits the target's `--quantization` unless its `SpeculativeConfig`
carries its own. An NVFP4 draft against an EXL3 target must therefore set
`DRAFT_QUANTIZATION=modelopt_fp4`, or it is loaded through the EXL3 path and
throws the identical `m=3` error as the rule above. Same symptom, different
cause; a config trap, not a bug. Selecting `MTP_DRAFT=nvfp4` fills it in
automatically, and clearing it is an error.

### `kv-nvfp4-uncalibrated` — error
KV dtype `nvfp4_ds_mla` requires model-calibrated MLA outer scales. v29 ships
the GLM-5.2 scale file used by the EXL3 profile, so calibrated NVFP4 KV is its
default. Uncalibrated earlier experiments silently degenerated at long context
while short checks passed; other families/variants remain blocked unless their
registry entry explicitly declares equivalent calibration.

### `vision-long-context` — warn (blunt)
`VISION=1` on the EXL3-TR3 checkpoint **corrupts long-context output**: measured
32K needle 0/3 with degenerate text on the final v20 turnkey image; mean MTP
acceptance collapsed to 1.25–1.50. The same process described a 5120×2880
screenshot with 17/18 exact details, answered an exact multi-turn follow-up
that reused the image, and passed text-only chat. Marked EXPERIMENTAL /
known-broken-at-long-context rather than blocked because vision is genuinely
useful for short-prompt image work. Vision is opt-in; the long-context probe
fails this configuration by design, so it is reported as unverified rather
than healthy. This rule applies to both the balanced and max-context EXL3-TR3
variants.

The warning is not inferred solely from the EXL3 quantizer. A separate
MadeBy561 MTP-off run passed 3/3 32K text retrieval but only 1/18 fields on the
same detailed image; an upstream-style wrapper mapper/prefix correction reached
2/18 without changing PP/TG. The failure therefore spans the two compositions
tested, although simpler upstream image tests may still pass. No general GLM
vision profile is promoted until one composition passes both detailed image
and long-context gates.

### `vision-kv-pressure` — warn
The current graft adds about 1.99 GiB/GPU. DCP4 at utilization 0.975 exposed
564,736 KV tokens and served the short vision suite; 0.98 exposed 610,560
tokens but left 37.12 MiB free and OOMed on the first 48 MiB verification
allocation. Boot-time KV admission is not runtime headroom. The warning above
384K recommends no more than 0.975 on that exact tested shape and requires
cold memory plus retrieval requalification for any other shape.

### `instanttensor-context-margin` — warn
On the exact TP4/DCP2 EXL3 profile, InstantTensor loaded about 0.04 GiB/GPU
more resident model memory than safetensors. `MAX_MODEL_LEN=524288` at
utilization 0.978 then failed KV admission twice (9.04 GiB needed, 9.03 GiB
available). That earlier v31 shape passed `MAX_MODEL_LEN=520192`; GG v20-r5
adds safe retained-CUDA-graph accounting and exposes 514,944 KV tokens at the
same utilization. The current `MAX_MODEL_LEN=513536` default passed cold and
cache-reused boots, the required feature suite, two independent ~510.5K
five-depth retrievals, and a 507,902 + 4,096 token request. The loader reduced
target+draft load from 60.5–62.6 seconds to 32.4–33.1 seconds. It remains the
balanced EXL3 default because safetensors failed three near-max runtime
attempts at the same 514,432-token boundary. This warning protects larger
overrides rather than disabling safe graph accounting or treating retries as
success.

### `pool-smaller-than-context` — error
`GPU_BLOCKS_OVERRIDE * 256 < MAX_MODEL_LEN` cannot start: vLLM refuses when it
cannot fit one max-length request. Reducing `MAX_MODEL_LEN` is the standard
remedy for the whole family of `To serve at least one request with the model's
max seq len ... larger than the available KV cache memory` failures; so is
dropping the pin (`GPU_BLOCKS_OVERRIDE=0`), turning vision off, or using a
smaller draft.

Also encoded: `tr3-draft-needs-exl3` (error), `dcp-divides-tp` (error),
`capture-sizes-max`, `gpu-util-high`, `dcp-reduces-pool`, `variant-untested`,
`spec-off`, `draft-model-overrides`, `chunk-bigger-than-context` (warnings).

### Failure signatures

`glm_config.SIGNATURES` maps log patterns to explanations (KV validation, EXL3
capture, OOM, unregistered architecture, the layer-78 `w2_weight` KeyError,
`ONLINE_QUANT`, the multimodal prompt-replacement failure). They are attached to
a preserved failure and handed to the model as context for its analysis.

---

## 4. Apply and rollback state machine

```
                    ┌────────────────────────────────────────────┐
                    │ steady: known-good config, verified         │
                    └───┬────────────────────────────────────────┘
   user clicks Apply    │  pre-validation (errors block: nothing written)
                        ▼
        write state file + apply-state=trial + touch restart-request
                        │
                        ▼
   supervisor sees the flag ──► kill vLLM ──► apply_config ──► prepare_checkpoint
                        │                       (re-resolved from scratch)
                        ▼
                   start vLLM + verifier
                        │
        ┌───────────────┼────────────────────────────┐
        ▼               ▼                            ▼
   engine dies    verify FAILS                   verify OK
        │               │                            │
        └───────┬───────┘                            ▼
                ▼                        mark-good, apply-state=steady,
        should-rollback?                 queue self-analysis if a failure
        ├── yes ──► preserve failed config+log+diff,                is pending
        │           restore known-good, restart  ──► (trial again)
        └── no  ──► ordinary crash: restart budget (SUPERVISOR_MAX_RESTARTS)
                    or, if it was only the probe that failed and there is
                    nothing better to run, keep serving and report
                    apply-state=degraded / "UNVERIFIED" — never "healthy"
```

Design points:

- **`should-rollback` is a real question.** Rolling back to the same
  configuration is a crash loop, not a rollback. The supervisor only rolls back
  when the running config actually differs from the known-good one (or when
  there is no known-good but there IS a state file to drop). Otherwise the
  restart budget applies, exactly as before this feature existed.
- **Restart budget resets on a config change.** A rollback or an applied change
  sets `attempt=0`, so the 5-restart crash budget still means "this config
  crash-looped", not "this instance has restarted 5 times in its life".
- **A failed probe does not necessarily kill a running engine.** If there is
  nothing better to fall back to, the engine keeps serving and the landing page
  says UNVERIFIED. Killing a partially-working endpoint on a rental to reach an
  identical one is not an improvement.
- **Pre-validation also runs at boot.** A hand-edited state file that fails
  validation is rolled back before the engine is started, not after.
- **What is preserved on rollback:** `failures/<UTC>/config.json` (the failed
  values and their resolution), `error.log` (tail of the failed boot),
  `diff.txt`, `meta.json` (reason + matched signatures). Ten failures are kept.

### Verification: why a short prompt proves nothing

`verify_serving.py` runs three phases: `/health`, three short prompts, then a
**needle probe** — one long haystack with access codes at fractional depths,
scored by retrieval, in a single prefill. All three of the config-level
corruption modes this appliance has met (nvfp4 KV without scales, trellis
`MIN_M` lowered, vision on EXL3) pass `/health` and short prompts perfectly and
fail past ~32K tokens. The verdict therefore carries `long_context_verified`
separately from `ok`, and the landing page has a **Correctness** card distinct
from the **Engine** (liveness) card. With `VERIFY_LONG_CONTEXT=0` or a context
budget too small to probe, the page says "short prompts only — long context
UNVERIFIED"; it never says healthy.

The probe defaults to `VERIFY_NEEDLE_TOKENS=32768` — big enough to catch every
corruption mode observed so far (they all show at 32K), small enough to cost
seconds. Token counts come from vLLM's `/tokenize` when available, else a
chars/3.7 estimate. `VERIFY=0` disables verification entirely; nothing then
rolls back and the UI says so.

### Self-analysis

Once the known-good config is serving again, the supervisor calls
`analyze_failure.py` against the local endpoint with: the config diff, the
rollback reason, the matched failure signatures, the rationale text of every
changed knob, the tail of the failed boot log, and the tail of the working boot
log. The answer is written to `failures/<ts>/analysis.md` and rendered on
`/config`. It is capped at 3 attempts per failure, runs with thinking disabled,
and is labelled on the page as a summary of the logs rather than an authority.

---

## 5. Checkpoint reconciliation (the revert-bug fix)

Two existing revert paths restore a config **snapshot**:
`graft_mtp78.py --revert` restores `config.json.orig`, and
`build_vision_config.py --revert` restores `config.json.text-only`. A snapshot is
only correct if nothing else changed in between. Snapshot the config while layer
78 is grafted, revert the graft later, and the restored file still says
`hybrid_tr3_tail.moe_layers = [3, 78]` while the layer-78 weights are BF16:

```
KeyError: model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight
```

The same shape of bug strips the `Glm5v` wrapper while `.vision-enabled` stays,
and image requests come back "GLM-5.2 is not a multimodal model".

`reconcile_checkpoint.py` runs at the end of **every** checkpoint transition and
derives from observation instead of memory:

| derived field | derived from |
|---|---|
| `hybrid_tr3_tail.moe_layers` | are the layer-78 expert tensors per-expert (BF16) or packed (trellis)? |
| `quantization_config.ignore` | same question — layer-78 patterns are dropped iff the weights are trellis |
| `architectures` / wrapper | are the vision shards on disk **and** wanted? |
| `model.safetensors.index.json` | rebuilt from the safetensors headers of the shards that exist |
| `chat_template.jinja` | the vision template iff the wrapper is active |
| `.mtp78-grafted`, `.vision-enabled` | markers describe reality; they never decide it |

The pristine values the graft overwrites (`moe_layers`, the layer-78 `ignore`
patterns) are captured once into `checkpoint-baseline.json`, preferring a source
that does **not** already say 78 — taking the grafted value as the baseline
would re-create the very bug. It is idempotent and reads only safetensors
headers, so it costs milliseconds.

---

## 6. HTTP surface

All paths take `?token=<OPEN_BUTTON_TOKEN>`; POSTs also carry the token in the
body. The config editor requires a non-empty `OPEN_BUTTON_TOKEN` — without one
it is not exposed at all. Unlike `/chat` it does not additionally require TLS
(a rental without a DNS provider has no cert, and the feature would then exist
only for the minority who set up deSEC); over plain HTTP the page says so.

| method | path | does |
|---|---|---|
| GET | `/config` | editor: value, source badge (`default`/`env`/`file`), rationale, live pre-validation, apply state, preserved failures + analyses |
| POST | `/config/apply` | coerce → validate → (errors: write nothing) → write state file, `apply-state=trial`, request restart |
| POST | `/config/import` | same, from a pasted export |
| POST | `/config/reset` | delete the state file, restart on env + defaults |
| POST | `/config/restart` | restart with the current config |
| GET | `/config/export` | `glm52-config.json` download (`values` + informational `effective`/`sources`) |
| GET | `/config/status` | JSON: apply state, last verdict, correctness headline, boot phase |

---

## 7. Operational notes

- An apply costs one engine restart (minutes of load/JIT). Changing
  `MTP_DRAFT`, `VISION` or `MODEL_VARIANT` can add a download on top; the
  boot-status phases (`grafting-mtp78`, `installing-vision`,
  `downloading-weights`, `applying-config`, `rolling-back`) show which.
- The API key, the TLS cert and the DNS registration are **not** re-done on a
  config restart; only the engine is restarted.
- `logs/serve-current.log` is truncated at `SERVE_LOG_MAX_MB` (64) once the
  config is verified — the boot log has been copied to `logs/last-good.log` by
  then, so nothing diagnostic is lost.
- New env vars introduced by this feature, all optional: `MODEL_ROOT`,
  `GLM_STATE_DIR`, `GLM_RUNTIME_DIR`, `SCRIPTS_DIR`, `VERIFY`,
  `VERIFY_LONG_CONTEXT`, `VERIFY_NEEDLE_TOKENS`, `VERIFY_HEALTH_TIMEOUT_S`,
  `SERVE_LOG_MAX_MB`, `LANDING_PORT`.

## 8. Live qualification status

This layer is no longer justified only by static substitutions:

- The custom/Qwen path, provider detection, configuration UI, authentication,
  restart persistence, supervisor recovery, Runpod proxy, direct TLS and DNS
  lifecycle were exercised on live Vast and Runpod Blackwell rentals.
- Both the 303 GB rank-sliced EXL3 checkpoint and the 341 GB MadeBy561 hybrid
  have booted through the resolved GLM arguments on four RTX PRO 6000
  Blackwell GPUs. The hybrid's native quantizer and serialized NVFP4 MTP experts,
  calibrated KV, explicit pool, chunk/workspace controls, and C1–C8 serving
  have been measured rather than inferred.
- The real GLM-5.2 model answered three fresh 32K probes (15/15 needles) and
  both lossless and FP8-ring five-depth probes above 521K. Exact matrices are
  recorded in `TEST_RESULTS.md`, not hidden behind `/health`.
- Checkpoint reconciliation observed the current per-expert rank-sliced
  layer-78 tensors on the real EXL3 snapshot. Unit fixtures retain packed,
  rank-sliced, BF16, stale-marker, vision, and dry-run coverage.
- Cached-weight engine restart and verifier timing is measured in situ:
  6m31s to health and 6m49s to verified on AIBeast with populated AOT caches;
  the fully warmed repeat was verified in 5m00s and the 32K gate itself takes
  17–19s there.

Absolute v20 throughput is confirmed on the all-NODE AIBeast host at 280 W/card:
2,701 tok/s at 8K, 1,987 at 66K, 121.6 tok/s C1 and 269.7 aggregate at C8.
Remaining claims stay narrow: InstantTensor is promoted only for the exact
balanced EXL3 profile and the current 513,536-token GG r5 envelope; other
variants retain their own loader choice. GLM vision remains a separate
short-context experiment rather than part of the text flagship envelope.
