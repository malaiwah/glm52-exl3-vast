# Repository Guidelines

## Project Overview

This repository builds a Docker appliance for serving GLM-5.2, Qwen3.6-27B, and compatible custom checkpoints through vLLM on Vast.ai or Runpod. It provides an authenticated OpenAI-compatible API, persistent model/cache storage, a management dashboard, optional TLS and SSH, automatic verification/rollback, provider-aware termination, and an opt-in Nanobot-based diagnostic controller (SOUL).

The checked-in profiles are coherent configurations, not model-name aliases. Do not change only `MODEL_ID`/`MODEL_DIR` for a model that needs different quantization, topology, parser, attention, speculation, or vision settings; add or update a family/profile in `scripts/glm_config.py`.

## Architecture & Data Flow

1. `entrypoint.sh` runs as container PID 1. It detects visible GPUs with `scripts/gpu_detect.py`, identifies the provider through `scripts/provider.py`, and snapshots the startup environment.
2. `scripts/config_cli.py` and `scripts/glm_config.py` resolve configuration in precedence order: built-in family/variant defaults < startup environment < persistent JSON overrides. The resolved shell environment drives the family-specific `vllm serve` arguments.
3. Checkpoint transition helpers modify or reconcile model metadata when required. `scripts/reconcile_checkpoint.py` derives checkpoint configuration from the files actually on disk rather than trusting stale snapshots.
4. vLLM starts on port 8000. `scripts/verify_serving.py` gates health with short inference checks and a long-context needle probe. Passing settings become known-good; failures are preserved under the state directory, rolled back, and may be analyzed by `scripts/analyze_failure.py`.
5. `landing.py` serves the token-gated management UI on port 1111 (plain HTTP and TLS on one listener). Config edits are written to persistent state and a runtime restart request tells `entrypoint.sh` to re-resolve and restart vLLM.
6. Optional `scripts/soul_controller.py` runs an async diagnostic loop, probes health/metrics/GPU state, and appends redacted journal records. It is advisory; startup verification and rollback remain authoritative.

Keep persistent and ephemeral state separate:

- `$GLM_STATE_DIR` (normally `/workspace/.glm-config`): config, known-good state, verification results, failures, SOUL journal/evidence.
- `$GLM_RUNTIME_DIR` (normally `/tmp/glm-runtime`): startup snapshot, resolved `config.env`, restart/termination flags, and other per-container state.
- `/tmp/glm-boot-status.json`: live boot phase consumed by the landing page.

Configuration is data-driven. `FAMILIES`, `VARIANTS`, `KNOBS`, and validation rules in `scripts/glm_config.py` are the source of truth. The shell entrypoint and dashboard consume those registries; do not introduce a second hard-coded model/config table.

## Key Directories

- `scripts/`: runtime orchestration helpers, configuration/state logic, provider operations, serving verification, checkpoint transforms, and qualification tools.
- `tests/`: flat, dependency-free Python test modules plus small fixtures. Tests are runnable directly and are designed not to require a GPU or network.
- `docs/`: design and operator references for model families, self-service config, SOUL, and termination/erase behavior.
- `soul/`: immutable SOUL role and trust-boundary instructions copied into the image.
- `.github/workflows/`: CI lint, test, profile-smoke, template validation, and Docker build/publish workflow.

## Development Commands

Run from the repository root. CI uses these exact checks:

```bash
shellcheck entrypoint.sh
python3 -m py_compile landing.py scripts/*.py
python3 tests/test_termination.py
python3 tests/test_families.py
python3 tests/test_knob_wiring.py
python3 tests/test_gpu_detect.py
python3 tests/test_serving_tools.py
python3 tests/test_feature_suite.py
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
- Prefer stdlib-only code for boot-critical modules. The vLLM environment and isolated `/opt/nanobot-venv` must not contaminate each other.
- Represent config and operation results as plain dictionaries. There is no DI framework: inject environment mappings, paths, transports, subprocess functions, or constructor arguments so logic remains testable without hardware/network access.
- Scope family-specific knobs and validations. GLM-only backend, DCP, EXL3, graft, or vision behavior must not leak into Qwen/custom profiles. Update registry metadata and `tests/test_knob_wiring.py` expectations when adding a knob.
- Use atomic JSON writes for durable state and append-only JSONL for journals. Derive checkpoint state from observable files, and keep operations idempotent/reversible where existing tools do so.
- Handle external failures defensively: bounded timeouts, useful status/result dictionaries, safe defaults for missing/corrupt state, and graceful UI degradation. Do not use broad exception handling to hide configuration or security errors.
- Redact tokens, API keys, private-key material, and credentials before logging, journaling, persisting diagnostics, or returning HTTP content. `landing.py` and `scripts/soul_controller.py` reuse `scripts/soul_config.py`'s `redact()`/`SECRET_PATTERNS`; `scripts/provider.py` does not depend on `soul_config` and instead defines its own narrower `redact(url)` for query-string credentials. Match the existing redactor for the module you're editing rather than adding a third one.
- Async work is concentrated in SOUL: use bounded tasks/timeouts and `asyncio.to_thread` for blocking subprocess work. Respect SIGTERM/SIGINT and close managed agents cleanly.
- Destructive provider/erase actions must remain opt-in and multiply gated: explicit enablement/confirmation, an armed transport, dry-run support, and injectable/sandboxed paths in tests.
- In `entrypoint.sh`, preserve Bash arrays and quoting for generated vLLM arguments. Use `CONFIG_SMOKE=1` after changing config resolution or serve-argument wiring.

## Important Files

- `entrypoint.sh`: PID-1 supervisor and end-to-end boot/restart/rollback lifecycle.
- `landing.py`: dashboard, config/termination endpoints, status rendering, and dual-protocol port 1111 server.
- `scripts/glm_config.py`: authoritative model families, variants, knobs, derived values, and validation matrix.
- `scripts/config_cli.py`: shell interface to config resolution, validation, known-good state, rollback, and restart flags.
- `scripts/verify_serving.py`: post-start correctness gate; `/health` alone is not a successful boot verdict.
- `scripts/reconcile_checkpoint.py`: authoritative checkpoint metadata reconciliation after model transitions.
- `scripts/provider.py`, `scripts/terminate_worker.py`, `scripts/secure_erase.py`: provider detection and safety-critical destruction workflow.
- `scripts/soul_config.py`, `scripts/soul_controller.py`: diagnostic configuration, redaction, durable journal, and async controller.
- `Dockerfile`: pinned vLLM/CUDA base, isolated SOUL environment, copied runtime files, exposed ports, and entrypoint.
- `requirements-soul.in`, `requirements-soul.lock`: direct and hash-locked SOUL dependencies.
- `runpod-template.json`, `runpod-template-qwen36.json`: checked-in Pod profiles; keep their ports, volume mount, and model profile coherent with runtime behavior.
- `.github/workflows/build.yml`: canonical local QA command list and image build/publish behavior.
- `TEST_PLAN.md`, `TEST_RESULTS.md`: manual paid-provider qualification matrix and observed hardware results; do not treat local tests as GPU qualification.

## Runtime/Tooling Preferences

- Production target: Linux Docker with NVIDIA Blackwell (`sm120+`); supported examples are RTX 5090 and RTX PRO 6000 Blackwell. RTX 4090 (`sm89`) is not a supported target for this pinned image.
- Python 3.11 is the dependency-lock target. There is no `pyproject.toml`, package build, Node/Bun runtime, or JavaScript package manager.
- Use `uv` only to compile `requirements-soul.in`; installation is pip-compatible and hash-verified. Do not hand-edit `requirements-soul.lock`.
- ShellCheck is the shell linter. Python QA is bytecode compilation plus direct stdlib tests; no repository-wide formatter or type checker is configured.
- Runtime ports are 8000 (OpenAI API), 1111 (dashboard), 8443 (direct TLS path), and 22 (optional key-only SSH).
- Use Docker/profile smoke mode for runtime checks. Local macOS execution can validate pure Python and shell/config behavior, but not CUDA, vLLM kernels, GPU topology, or provider lifecycle.

## Testing & QA

Tests use the Python standard library (`unittest`, `unittest.mock`, and some procedural `check()` harnesses), not pytest. Follow the flat `tests/test_*.py` layout and run the affected module directly. Use `tempfile.TemporaryDirectory`, injected environment/path objects, stub HTTP transports, and mocked subprocesses; tests must not contact Vast.ai/Runpod, mutate real checkpoints, erase host files, or require `nvidia-smi`.

For every change:

- Config registry/serve args: run `test_families.py`, `test_knob_wiring.py`, and all three `CONFIG_SMOKE` profiles.
- GPU visibility: run `test_gpu_detect.py`.
- Provider termination or erase: run `test_termination.py` and preserve all arming/sandbox guards.
- Checkpoint transforms/reconciliation: run `test_checkpoint_reconcile.py` with temporary fixtures.
- Dashboard/OpenAI features: run `test_feature_suite.py`; manually exercise the changed HTTP path when behavior changes.
- SOUL state, probes, redaction, or retention: run `test_soul.py`.
- Benchmark or long-context tooling: run `test_serving_tools.py`.

There is no configured coverage threshold. CI is GPU-free and validates syntax, pure logic, stubbed integrations, profile rendering, templates, and image construction. Changes to kernels, memory/topology defaults, long-context correctness, TLS/provider behavior, or production profiles also require the cost-controlled live workflow in `TEST_PLAN.md`; record observed results in `TEST_RESULTS.md` rather than claiming qualification from local tests.