# Repository Guidelines

## Project Overview

This repository builds a Docker appliance for serving GLM-5.2, Qwen3.6-27B, and compatible custom checkpoints through vLLM on Vast.ai, Runpod, or JarvisLabs. It provides an authenticated OpenAI-compatible API, persistent model/cache storage, a management dashboard, optional TLS and SSH, automatic verification/rollback, provider-aware termination, and an opt-in Nanobot-based diagnostic controller (SOUL).

The checked-in profiles are coherent configurations, not model-name aliases. Do not change only `MODEL_ID`/`MODEL_DIR` for a model that needs different quantization, topology, parser, attention, speculation, or vision settings; add or update a family/profile in `scripts/glm_config.py`.

## Architecture & Data Flow

1. `entrypoint.sh` runs as container PID 1. It reads `MODEL_PROFILE` (`glm52-exl3` | `qwen36-27b-nvfp4` | `custom`; unknown → `FATAL` exit 1), detects visible GPUs with `scripts/gpu_detect.py` (intersecting `nvidia-smi` with `NVIDIA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, exporting `GLM_GPU_COUNT`), identifies the provider through `scripts/provider.py`, and enforces a driver/CUDA floor gate (`MIN_NVIDIA_DRIVER_VERSION=590.48.01`, `MIN_NVIDIA_CUDA_VERSION=13.2`; skipped under `CONFIG_SMOKE=1`/`NGPU=0`/`ALLOW_UNSUPPORTED_NVIDIA_DRIVER=1`).
2. `scripts/config_cli.py` and `scripts/glm_config.py` resolve configuration. `glm_config.resolve()` precedence (lowest → highest): `default` < `family` < `detected` (GPU count → `TENSOR_PARALLEL_SIZE`/`DCP`) < `variant` < `env` (startup env frozen once via `snapshot-env`) < `file` (`$GLM_STATE_DIR/config.json`). The **state file is the highest resolution layer**. `config_cli.py env` prints `export K=V` plus the `FAMILY_SERVE_ARGS` and `PROFILE_RUNTIME_ENV` bash arrays. At runtime, `entrypoint.sh` then applies the family env block, `PROFILE_RUNTIME_ENV`, and finally `TUNE_<NAME>=value` overrides (`apply_tuning_overrides()`) as the last layer for controlled experiments — `TUNE_*` is a runtime override, not a `resolve()` layer.
3. Checkpoint transition helpers modify or reconcile model metadata when required. `scripts/reconcile_checkpoint.py` derives checkpoint configuration from the files actually on disk rather than trusting stale snapshots (fixes graft/vision-revert stale-config bugs); `build_vision_config.py`/`index_add_vision.py` merge the GLM vision wrapper; `graft_mtp78.py`/`build_mtp78_draft.py`/`build_nvfp4_mtp_draft.py` build MTP78 draft overlays.
4. vLLM starts on port 8000 (`serve_once` builds the argv from generic flags + `FAMILY_SERVE_ARGS` + `VISION_ARGS` + `BLOCKS_ARGS` + `AUTH_ARGS` + `TLS_ARGS` + `SPEC_ARGS` + `KVT_ARGS`; `VLLM_API_KEY` passed via env, not argv, to avoid `/proc/cmdline` leak). `scripts/verify_serving.py` gates health with short inference probes (arithmetic/factual/instruction), a structured-output probe, and a tokenizer-calibrated long-context needle probe. Short-pass + long-fail is the known silent-corruption signature. Passing settings become known-good; failures are preserved under the state directory, rolled back, and may be analyzed by `scripts/analyze_failure.py` (or queued to SOUL incident analysis).
5. `landing.py` serves the token-gated management UI on port 1111 (dual-protocol TLS+plain on one listener via first-byte peek; `OPEN_BUTTON_TOKEN` HMAC gating). Config edits write to persistent state and a runtime restart request tells `entrypoint.sh` to re-resolve and restart vLLM. Browser-side scrapes vLLM `/metrics`; SSE chat streaming.
6. Optional `scripts/soul_controller.py` runs an async diagnostic loop as a sibling (not child) of the vLLM supervisor, probes health/metrics/GPU state, and appends redacted journal records. It is advisory; startup verification and rollback remain authoritative.
7. Provider self-termination goes through a three-gate ratcheted flow: termination switches (`terminate_worker.run()`), typed confirmation matching `instance_id`, provider `ready()`; `secure_erase.py` optionally destroys session evidence (keys/SSH/logs/state; NOT public weights) before `provider.py` destroys the paid instance via an armed `HttpTransport(allow_destructive=True)`.

Keep persistent and ephemeral state separate:

- `$GLM_STATE_DIR` (default `/workspace/.glm-config`, on the volume, survives restarts): `config.json` (user state, a diff not a snapshot), `known-good.json`, `apply-state.json`, `failures/<ts>/` (preserved failed configs+logs, pruned to 10), `logs/`, `checkpoint-baseline.json`, `verify-last.json`, `soul/` (config/status/journal/incidents/evidence/snapshots/workspace/logs).
- `$GLM_RUNTIME_DIR` (default `/tmp/glm-runtime`, per-container, NOT on volume): `startup-env.json`, `config.env`, `restart-request`, `verify.json`, `terminate-in-progress`, `engine-stopped`, `boot-notes.log`, `terminate.json`, `terminate-switches.json`, `soul/`.
- `/tmp/glm-boot-status.json` (`STATUS_FILE`): the landing page's single status source (phase, endpoint, api_key, served_model, gpus, provider, cert/keyfile paths); written atomically (staged+mv, umask 077, chmod 600). A sanitized `$STATUS_FILE.soul` copy (api_key stripped, `chgrp soul`, 640) is published alongside for the deprivileged SOUL controller.

Configuration is data-driven. `FAMILIES`, `VARIANTS`, `KNOBS`, `DRAFTS`, and validation rules in `scripts/glm_config.py` are the source of truth. The shell entrypoint and dashboard consume those registries; do not introduce a second hard-coded model/config table. `FORBIDDEN_STATE_KEYS` (`TERMINATE_ENABLED`/`TERMINATE_LOCKED`) cannot be set from the state file (startup-only switches).

## Key Directories

- `scripts/`: runtime orchestration helpers, configuration/state logic, provider operations, serving verification, checkpoint transforms, vLLM/SparkInfer patches, field-review tooling, SOUL controller/config, and qualification benchmarks. 36 `.py` tools + 7 shell helpers.
- `tests/`: flat, dependency-free Python test modules (20 `.py`) plus two Bash tests. Tests are runnable directly and are designed not to require a GPU or network.
- `docs/`: design and operator references — model families, self-service config, SOUL, termination/erase, configuration reference, benchmarks, plus per-release GLM qualification evidence (`glm52-r14/r17/r20/r25/r26/r28-*`, `glm52-3.25-offload-qualification.md`), MTP78, and Qwen OMP guide.
- `soul/`: immutable SOUL role and trust-boundary instructions (`SOUL.md`) copied read-only into the image.
- `patches/field-review-r26/`: the current field-review ledger copied into the image; older ledgers (`r14`/`r17`/`r20`/`r25`) retained.
- `.github/workflows/`: CI lint, test, profile-smoke, template validation, and Docker build/publish workflow.
- `docs/field-review-results/`: preserved Vast field-review PR-validation campaign with counter-validation and raw artifacts.

## Development Commands

Run from the repository root. CI uses these exact checks:

```bash
shellcheck entrypoint.sh scripts/run-local-podman.sh scripts/bench-glm52-kld-tp4.sh scripts/recover_torch_extension_lock.sh scripts/jarvislabs_vm_bootstrap.sh scripts/glm52_lmcache_wrapper.sh scripts/acme_retry.sh
python3 -m py_compile landing.py scripts/*.py
python3 tests/test_field_review_patches.py
python3 tests/test_termination.py
python3 tests/test_families.py
python3 tests/test_r28_base_gate.py
python3 tests/test_vllm_serial_spec_warning_patch.py
python3 tests/test_exl3_mixk_patch.py
python3 tests/test_desec_acme_guard.py
python3 tests/test_knob_wiring.py
python3 tests/test_gpu_detect.py
python3 tests/test_extension_lock_recovery.py
bash tests/test_lmcache_wrapper.sh
bash tests/test_acme_retry.sh
python3 tests/test_serving_tools.py
python3 tests/test_nvfp4_mtp_draft.py
python3 tests/test_feature_suite.py
python3 tests/test_structured_output_patch.py
python3 tests/test_exl3_parity_abi_patch.py
python3 tests/test_field_review_log_audit.py
python3 tests/test_checkpoint_reconcile.py
python3 tests/test_soul.py
```

Exercise profile resolution without a GPU, downloads, or engine startup:

```bash
CONFIG_SMOKE=1 SCRIPTS_DIR=scripts MODEL_PROFILE=glm52-exl3 bash entrypoint.sh
CONFIG_SMOKE=1 SCRIPTS_DIR=scripts MODEL_PROFILE=qwen36-27b-nvfp4 bash entrypoint.sh
CONFIG_SMOKE=1 SCRIPTS_DIR=scripts MODEL_PROFILE=custom MODEL_ID=Qwen/Qwen3.5-0.8B bash entrypoint.sh
```

Validate provider templates and build the image:

```bash
python3 -m json.tool runpod-template.json >/dev/null
python3 -m json.tool runpod-template-qwen36.json >/dev/null
docker build -t model-turnkey:dev .
```

Regenerate the SOUL dependency lock only after editing `requirements-soul.in`:

```bash
uv pip compile requirements-soul.in --generate-hashes --python-version 3.11 --universal -o requirements-soul.lock
```

Commit both dependency files together. The Docker build installs the lock with `pip --require-hashes`.

## Code Conventions & Common Patterns

- Python follows PEP 8 naming: `snake_case` functions, `UPPER_CASE` constants, and `PascalCase` classes. Shell-facing values are usually uppercase environment variables. There is no configured Black, Ruff, Flake8, or mypy step; preserve nearby formatting.
- Prefer stdlib-only code for boot-critical modules (`glm_config.py`, `soul_config.py`). The vLLM environment and isolated `/opt/nanobot-venv` must not contaminate each other (the Dockerfile freeze-diffs the system/vLLM Python env before vs. after the SOUL venv install and fails the build on any diff, proving the venv install leaks nothing into the system env).
- Represent config and operation results as plain dictionaries. There is no DI framework: inject environment mappings, paths, transports, subprocess functions, or constructor arguments so logic remains testable without hardware/network access.
- Scope family-specific knobs and validations. GLM-only backend, DCP, EXL3, graft, or vision behavior must not leak into Qwen/custom profiles. `validate()` measures failures and errors block apply; inapplicable knobs are refused (`err('knob-inapplicable')`), not ignored. Update registry metadata and `tests/test_knob_wiring.py` expectations when adding a knob.
- Use atomic JSON writes for durable state and append-only JSONL (`O_APPEND`+`fsync`) for journals. Derive checkpoint state from observable files, and keep operations idempotent/reversible where existing tools do so.
- Handle external failures defensively: bounded timeouts, useful status/result dictionaries, safe defaults for missing/corrupt state, and graceful UI degradation. Do not use broad exception handling to hide configuration or security errors.
- Redact tokens, API keys, private-key material, and credentials before logging, journaling, persisting diagnostics, or returning HTTP content. `landing.py` and `scripts/soul_controller.py` reuse `scripts/soul_config.py`'s `redact()`/`SECRET_PATTERNS`; `scripts/provider.py` does not depend on `soul_config` and instead defines its own narrower `redact(url)` for query-string credentials. Match the existing redactor for the module you're editing rather than adding a third one. Keys travel in Bearer headers / `0600` header files / subprocess env, never argv or `/proc/cmdline`.
- Async work is concentrated in SOUL: `SoulController.run()` is an `asyncio` loop calling `one_cycle()` (~5s); use bounded tasks/timeouts and `asyncio.to_thread` for blocking subprocess work. `landing.py` uses `ThreadingHTTPServer` with `_config_lock`/`_terminate_lock` serializing apply/terminate. Respect SIGTERM/SIGINT and close managed agents cleanly.
- Destructive provider/erase actions must remain opt-in and multiply gated: explicit enablement/confirmation, an armed transport (`allow_destructive=True`), dry-run support (`TERMINATE_DRY_RUN`), and injectable/sandboxed paths (`ERASE_CONFINE_TO`) in tests. Termination switches use a tighten-only ratchet — `tighten()` is the only mutator; there is no loosen/unlock API.
- In `entrypoint.sh`, preserve Bash arrays and quoting for generated vLLM arguments. Background jobs use `set -m` (own process group for group-kill). Use `CONFIG_SMOKE=1` after changing config resolution or serve-argument wiring.

## Important Files

- `entrypoint.sh`: PID-1 supervisor and end-to-end boot/restart/rollback lifecycle (MODEL_PROFILE switch, GPU/provider detection, config apply, TLS/ACME, weight prep, `serve_once`, verify/rollback/restart budget `SUPERVISOR_MAX_RESTARTS=5`, `on_term` trap, `status_update`).
- `landing.py`: dashboard, config/termination endpoints, status rendering, and dual-protocol port 1111 server.
- `scripts/glm_config.py`: authoritative model families, variants, knobs, drafts, derived values, validation matrix, termination switches, log-pattern `SIGNATURES`, and state/runtime path helpers.
- `scripts/config_cli.py`: shell interface to config resolution, validation, known-good state, rollback, and restart flags. Exit codes 0/2/3.
- `scripts/verify_serving.py`: post-start correctness gate; `/health` alone is not a successful boot verdict.
- `scripts/reconcile_checkpoint.py`: authoritative checkpoint metadata reconciliation after model transitions.
- `scripts/provider.py`, `scripts/terminate_worker.py`, `scripts/secure_erase.py`: provider detection and safety-critical destruction workflow.
- `scripts/soul_config.py`, `scripts/soul_controller.py`: diagnostic configuration, redaction, durable journal, and async controller.
- `scripts/verify_r17_base.py`, `scripts/verify_r26_base.py`, `scripts/verify_r28_base.py`: immutable native-source contract gates (fail closed); `verify_r28_base.py` is the current gate.
- `scripts/apply_field_review_patches.py`, `scripts/field_review_log_audit.py`: fail-closed field-review patch applier and runtime-log audit.
- `Dockerfile`: pinned vLLM/CUDA base (`voipmonitor/vllm@sha256:501e10…`), isolated SOUL environment, copied runtime files, sha256-gated patch application, exposed ports, and entrypoint.
- `requirements-soul.in`, `requirements-soul.lock`: direct (`nanobot-ai==0.3.0`) and hash-locked SOUL dependencies.
- `runpod-template.json`, `runpod-template-qwen36.json`: checked-in Pod profiles; keep their ports, volume mount, and model profile coherent with runtime behavior.
- `scripts/jarvislabs_vm_bootstrap.sh`: full-VM launcher for JarvisLabs, whose managed template catalog does not accept a user Docker image.
- `.github/workflows/build.yml`: canonical local QA command list and image build/publish behavior.
- `TEST_PLAN.md`, `TEST_RESULTS.md`: manual paid-provider qualification matrix and observed hardware results; do not treat local tests as GPU qualification.
- `README.md`, `CHANGELOG.md`: top-level operator reference and release lineage.

## Runtime/Tooling Preferences

- Production target: Linux Docker with NVIDIA Blackwell (`sm120+`); supported examples are RTX 5090 and RTX PRO 6000 Blackwell. RTX 4090 (`sm89`) is not a supported target for this pinned image.
- Python 3.11 is the dependency-lock target. There is no `pyproject.toml`, package build, Node/Bun runtime, or JavaScript package manager.
- Use `uv` only to compile `requirements-soul.in`; installation is pip-compatible and hash-verified. Do not hand-edit `requirements-soul.lock`.
- ShellCheck is the shell linter. Python QA is bytecode compilation plus direct stdlib tests; no repository-wide formatter or type checker is configured.
- Runtime ports are 8000 (OpenAI API), 1111 (dashboard), 8443 (direct TLS path; Runpod socat proxy to 127.0.0.1:8000), and 22 (optional key-only SSH).
- Use Docker/profile smoke mode for runtime checks. Local macOS execution can validate pure Python and shell/config behavior, but not CUDA, vLLM kernels, GPU topology, or provider lifecycle.

## Testing & QA

Tests use the Python standard library (`unittest`, `unittest.mock`, and some procedural `check()` harnesses), not pytest. Follow the flat `tests/test_*.py` layout and run the affected module directly. Two frameworks coexist: `unittest.TestCase` classes (the majority, 16 modules) and procedural `check(name, cond, detail)` harnesses with module-level `test_*` functions plus a custom `main()` (`test_families.py`, `test_knob_wiring.py`, `test_gpu_detect.py`, `test_termination.py`). Use `tempfile.TemporaryDirectory`, injected environment/path objects, stub HTTP transports, and mocked subprocesses; tests must not contact Vast.ai, Runpod, or JarvisLabs, mutate real checkpoints, erase host files, or require `nvidia-smi`.

For every change:

- Config registry/serve args: run `test_families.py`, `test_knob_wiring.py`, and all three `CONFIG_SMOKE` profiles.
- GPU visibility: run `test_gpu_detect.py`.
- Provider termination or erase: run `test_termination.py` and preserve all arming/sandbox guards.
- Checkpoint transforms/reconciliation: run `test_checkpoint_reconcile.py` with temporary fixtures.
- Dashboard/OpenAI features: run `test_feature_suite.py`; manually exercise the changed HTTP path when behavior changes.
- SOUL state, probes, redaction, or retention: run `test_soul.py`.
- Benchmark or long-context tooling: run `test_serving_tools.py`.
- vLLM/SparkInfer patches: run the corresponding `test_*_patch.py` and `test_r28_base_gate.py` (the current immutable-source gate; `test_r17`/`test_r26` remain for the older bases and are intentionally not in CI).

There is no configured coverage threshold; coverage is structural/invariant-focused (pinning exact serve argv, ledger/digest pins, knob-to-consumer wiring, fail-closed patch idempotency, switch-ratchet monotonicity, secret redaction, landing-page security headers). CI is GPU-free and validates syntax, pure logic, stubbed integrations, profile rendering, templates, and image construction. Changes to kernels, memory/topology defaults, long-context correctness, TLS/provider behavior, or production profiles also require the cost-controlled live workflow in `TEST_PLAN.md`; record observed results in `TEST_RESULTS.md` rather than claiming qualification from local tests.
