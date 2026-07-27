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
a note, out-of-range values fall back to the layer below with a note, and
non-editable keys (`VLLM_EXL3_TRELLIS_MIN_M`, `BASE_GENERATION`) are ignored
outright. A state file that is not valid JSON is ignored entirely rather than
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
| `MODEL_VARIANT` | EXL3-TR3 (measured; smallest weights, biggest pool) vs NVFP4 (faster decode, calibrated KV scales, ~30% less pool, **untested here**). Switching costs a full re-download. |
| `MTP_DRAFT` | `tr3-graft` (3.7 GB, +3.8 GB/GPU of KV, the only draft with long-context evidence) / `tr3-override` (best acceptance, **unbootable on v20**) / `nvfp4` (external dir, needs `DRAFT_QUANTIZATION`) / `bf16` (19.3 GB, always works) / `off` (~30% slower decode). |
| `MTP_TOKENS` | Depth of speculation. With the cheap trellis draft MTP-5 beats MTP-3 (53.4 vs 51.5 tok/s); with the 19.3 GB BF16 draft it lost 22%. Also widens the decode query width. |
| `DCP` | KV sharded across ranks (4) vs replicated (1). DCP=4 is what makes 512K fit. |
| `KV_CACHE_DTYPE` | fp8 (safe, ~1.7x bytes/token) vs nvfp4 (needs calibrated MLA outer scales; silent long-context corruption without them). |
| `MAX_MODEL_LEN` | Longest request, and a hard startup gate against available KV. |
| `GPU_BLOCKS_OVERRIDE` | Predictable 512K pool vs "take everything" (~697K tokens measured). |
| `OFFLOAD_FRACTION` | Host RAM for a pinned KV tier; pure cache, `recompute` on miss. |
| `VISION` | Image input vs long-context correctness on EXL3 (see rule 6) and ~1.31 GiB/GPU + ~13% text decode. |
| `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` | Concurrency and prefill chunk against the capture window and against VRAM headroom. |
| `MAX_CUDAGRAPH_CAPTURE_SIZE`, `CUDAGRAPH_CAPTURE_SIZES`, `VLLM_EXL3_TRELLIS_MAX_M` | The three ceilings that must move together to serve more streams. |
| `VLLM_EXL3_TRELLIS_MIN_M` | **Locked at 4.** Displayed with the reason, not editable. |

---

## 3. Validation matrix

Measured on 4x RTX PRO 6000 on 2026-07-26 unless stated. `error` blocks an
apply; `warn` is shown and allowed. Rule ids are the `id` field in the
findings, so the UI, the boot log and this table agree.

### `concurrency-window` — error
`MAX_NUM_SEQS * (1 + MTP_TOKENS)` must be `<= min(MAX_CUDAGRAPH_CAPTURE_SIZE,
VLLM_EXL3_TRELLIS_MAX_M)`, both 32 by default. Exceeding it does **not** error
at boot; decode silently leaves the captured trellis fast path under
concurrency and loses throughput. Raising concurrency requires raising
`CUDAGRAPH_CAPTURE_SIZES`, `MAX_CUDAGRAPH_CAPTURE_SIZE` and
`VLLM_EXL3_TRELLIS_MAX_M` together. This is why `MAX_NUM_SEQS` defaults to 8:
`8 * (1+3) = 32` exactly fills the window.

### `trellis-min-m` — error
`VLLM_EXL3_TRELLIS_MIN_M` must stay 4. Lowering it to 1 makes an EXL3 capture
crash disappear but **silently corrupts output**: measured 32K needle 0/2 and
370K 0/5, pure garbage, while short-prompt checks still passed 6/6. The knob is
non-editable in the UI and a state file that carries a different value is
rejected.

### `capture-below-trellis-min` — error
Any entry in `CUDAGRAPH_CAPTURE_SIZES` below the trellis minimum puts capture on
the eager parity path and the worker dies with `EXL3 eager parity path entered
during CUDA graph capture (m=N)`. The fix is a bigger capture size, never a
smaller window.

### `tr3-draft-on-v20` — error
On the **GG v20 base only**, an EXL3 rank-sliced MTP draft (`MTP_DRAFT=tr3-override`)
cannot be CUDA-graph captured: `SpeculatorCudaGraphManager` captures at m
outside the trellis window `[4,32]` and the engine dies with `EXL3 eager parity
path entered during CUDA graph capture (m=3)`. m=3 is invariant — it follows
neither `num_speculative_tokens` nor `cudagraph_capture_sizes`. Valid on the
pre-v20 base; `BASE_GENERATION` is a read-only knob so the draft type is
validated against the image it will actually run on. `bf16` drafts work on both.

### `draft-quant-inherit` — error
A draft inherits the target's `--quantization` unless its `SpeculativeConfig`
carries its own. An NVFP4 draft against an EXL3 target must therefore set
`DRAFT_QUANTIZATION=modelopt_fp4`, or it is loaded through the EXL3 path and
throws the identical `m=3` error as the rule above. Same symptom, different
cause; a config trap, not a bug. Selecting `MTP_DRAFT=nvfp4` fills it in
automatically, and clearing it is an error.

### `kv-nvfp4-uncalibrated` — error
KV dtype `nvfp4_ds_mla` requires per-checkpoint calibrated MLA outer scales. The
EXL3 checkpoint has none: with nvfp4 KV, long context silently degenerates
(needle 0/6 at 505K, output like `".,, while.,, and and while,,,,"`) while
GSM8K, vision and structured output all pass. fp8 is the safe default; it costs
~1.7x KV bytes/token. Allowed on the NVFP4 variant, which does have them.

### `vision-long-context` — warn (blunt)
`VISION=1` on the EXL3-TR3 checkpoint **corrupts long-context output**: measured
32K needle 0/2 with degenerate text on both the v20 and pre-v20 bases, while
short-prompt and vision smoke tests passed 6/6. Marked EXPERIMENTAL /
known-broken-at-long-context rather than blocked, because vision is the shipped
default and is genuinely useful for short-prompt image work. The long-context
probe fails this configuration by design, so it will be reported as unverified
rather than healthy.

### `vision-kv-pressure` — warn
Vision raises memory pressure enough that `MAX_MODEL_LEN` 384K has failed KV
validation (needed 5.7 GiB, had 3.97 GiB). Warned above 384K.

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

## 8. Not verified without a container

Listed honestly, because none of it can be proven by reading:

- No vLLM was started. The serve arguments derived from the new knobs
  (`--quantization`, `--kv-cache-dtype`, `--decode-context-parallel-size`) are
  string substitutions of values the previous version hard-coded, but the
  NVFP4 variant path in particular is **derived, not measured**: repo name,
  quantization method and the `hf-overrides` block are assumptions.
- The needle probe's prompt has never been answered by GLM-5.2 here — it was
  exercised against a fake OpenAI server. Its retrieval threshold (all codes
  must be found) may need tuning against the real model's phrasing.
- `prepare_checkpoint()` re-running on a live restart (graft/vision transitions
  without a container replacement) is exercised only by the reconciler's unit
  test against a synthetic checkpoint, not against the 332 GB real one.
- The supervisor state machine was driven end-to-end with stubbed
  `serve_once`/`start_verifier` (see the report), not with a real engine, so
  the timing of `/health` against a 15-minute JIT boot is untested in situ.
