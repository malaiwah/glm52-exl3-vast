# Configuration reference

The complete environment-knob table, extracted from the README. Values resolve
as `defaults < family < variant < startup env < state file`; see
[self-service-config.md](self-service-config.md) and
[model-families.md](model-families.md).

## Environment variables

| env | default | why you'd change it |
|---|---|---|
| `MODEL_PROFILE` | `glm53-3.42bpw-500k` | full 3.42bpw reduced-workspace **candidate** preserving 520,192 tokens; >=500K spot retrieval passed, full matrix open. The older full `glm53-3.42bpw` retains its historical 393,216 limit, not refreshed-image qualification; 3.25bpw, GLM-5.2 and custom are explicit alternatives. Flash `glm53-k6`/`glm53-k8` are refused on this base |
| `MODEL_ID` | custom profile only | select a checkpoint for `MODEL_PROFILE=custom`; named GLM profiles own their immutable model revisions and cannot safely be changed by substituting only a model ID |
| `MODEL_DIR` | profile-specific path under `/workspace` | point at complete weights; the completion marker must match the pinned model repository and revision |
| `SERVED_MODEL_NAME` | profile name | whitespace-separated aliases, so existing clients keep working; this is also the name every dashboard page displays |
| `TENSOR_PARALLEL_SIZE` | 4 GLM / 1 Qwen | match a supported profile topology |
| `MAX_MODEL_LEN` | profile-specific | primary full 3.42bpw candidate 520192 total tokens; historical full 3.42 393216; optional 3.25 experiment 524288. Include templating/output; one 501086-token haystack retrieval does not complete the boundary matrix |
| `MULTIMODAL` | n/a GLM / 1 Qwen | Qwen `0` saves vision VRAM with `--language-model-only`; GLM vision remains controlled by `VISION` (default 0) |
| `MM_MAX_PIXELS` | n/a GLM / 8388608 Qwen | cap native image processing near a 4K working image; the 5K detail gate passed at this value |
| `QUANTIZATION` | custom profile only | vLLM quantizer name such as `modelopt` |
| `REASONING_PARSER` / `TOOL_CALL_PARSER` | custom profile only | model-specific OpenAI response parsers |
| `REASONING_EFFORT_DEFAULT` | `high` | full GLM chat-template default; `low`, `high`, or `max`. Individual requests can override it |
| `PREFILL_SCHEDULE_INTERVAL` | `1` | full GLM prefill admission cadence. `1` preserves unthrottled behavior; evaluate `2` or higher for concurrent decode responsiveness using the TP cadence backport before changing production |
| `TRUST_REMOTE_CODE` | `0` | explicit opt-in for reviewed custom checkpoint Python. Native GLM/Qwen profiles do not grant checkpoint-code execution by default |
| `AUTH` | `key` | `none` serves unauthenticated on a trusted LAN |
| `MIN_NVIDIA_DRIVER_VERSION` | `590.48.01` | lower bound of the qualified driver/CUDA pair; the gate runs before model download |
| `MIN_NVIDIA_CUDA_VERSION` | `13.2` | reported CUDA capability paired with the driver floor; prevents an r590/CUDA 13.1 host from passing |
| `ALLOW_UNSUPPORTED_NVIDIA_DRIVER` | `0` | bypass both admission floors only for a separately qualified compatibility stack |
| `GPU_BLOCKS_OVERRIDE` | 0 | auto-profile the largest safe KV pool; a positive value pins vLLM blocks, not tokens. On this MLA stack the reported logical capacity is `blocks × 64 × DCP` (for example, DCP4 needs 2,048 blocks—not 8,192—for exactly 524,288 tokens). Re-verify this relationship after an engine/topology change. |
| `KV_CACHE_MEMORY_BYTES` | primary full 3.42 candidate 4518907904 / historical full 3.42 3415867392 / otherwise profile-specific | per-GPU fixed KV pool, not free VRAM; supersedes GMU for KV sizing. Do not combine with `GPU_BLOCKS_OVERRIDE`; first-use sampler/workspace needs separate headroom |
| `OFFLOAD_FRACTION` | 0.5 GLM / 0 Qwen | aggregate host-DRAM prefix-cache budget (not active-context capacity); `0.5` was measured historically on a 256 GiB host; native vLLM derives the TP worker slices. Refreshed GLM-5.3 LMCache DRAM retrieval after GPU pressure is measured below, not disk L2/restart qualification |
| `OFFLOAD_IGNORE_MEMLOCK` | `1` | proceed when the memlock ulimit is below the tier size (see below); `0` disables offload instead |
| `PREFIX_CACHE_BACKEND` | `lmcache` GLM / `native` other profiles | `lmcache` is the r13-qualified supervised DCP-aware process; `native` keeps the in-process OffloadingConnector rollback control. Both use `OFFLOAD_FRACTION` for aggregate DRAM and neither enlarges active context. |
| `LMCACHE_L1_MAX_GB` | `0` (no extra ceiling) | optional aggregate LMCache DRAM ceiling in GiB, applied to the fraction-derived budget before memlock handling; AIBeast candidate caps at 125 GiB. It never increases the fraction budget |
| `LMCACHE_L1_INIT_GB` | min(20, configured L1) | initial LMCache DRAM arena; the remaining configured tier grows lazily. Raise only if first-hit allocation latency matters more than model page-in and host-memory headroom. |
| `PREFIX_CACHE_DISK_GB` | `0` | positive values enable LMCache's native filesystem L2 with this hard GiB limit under `<MODEL_ROOT>/.lmcache`; derived prompt KV may be sensitive, so prefer encrypted local NVMe and enable best-effort secure termination |
| `LMCACHE_L2_EVICTION_POLICY` | `LRU` | L2 disk-tier eviction policy; `LRU` evicts least-recently-used entries when the watermark is hit |
| `LMCACHE_L2_EVICTION_TRIGGER_WATERMARK` | `0.90` | L2 usage fraction that triggers eviction (0.90 = evict when 90% full); without this the L2 fills without garbage collection |
| `LMCACHE_L2_EVICTION_RATIO` | `0.10` | fraction of cached entries to evict per trigger (0.10 = evict 10% of entries) |
| `LMCACHE_RETRIEVE_TIMEOUT_SECONDS` | `180` | retrieve deadline consumed by the installed adapter; completed failures may recompute, but timeout/health loss while DMA ownership is unresolved fail-stops the worker rather than recycling writable GPU pages |
| `MTP_DRAFT` | full GLM-5.3 `native` | native EXL3/TR3 MTP3, not BF16; `off` disables speculation. Full GLM-5.3 rejects GLM-5.2 graft/override paths. Legacy `MTP78_TRELLIS=0` selects `native`, not a dtype |
| `MTP_DRAFT_SAMPLE_METHOD` | `probabilistic` GLM | measured MTP-5 proposal mode; `greedy` remains available for controlled A/B tests |
| `F8_DMA` | `0` family / `ring` MadeBy561 | compressed PCIe collective mode; the hybrid override passed the 521K five-depth gate |
| `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | `-1` family / `8192` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured crossover |
| `PCIE_DMA_MIN_BYTES` | `-1` family / `393216` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured byte crossover |
| `OPEN_BUTTON_TOKEN` | provider-specific | required to expose the `:1111` config editor; Vast supplies it and Runpod/JarvisLabs get a persisted generated token when one is not set |
| `SOUL_AUTONOMY_LEVEL` | `0` | enable the embedded diagnostic SOUL: Observe `1` (no shell), Investigate `2` (bounded read-only shell), or Verify `3` (idle-only canary and conditional long-context probe) |
| `SOUL_AUTONOMY_MAX_LEVEL` | `3` | startup-only ceiling for landing-page overrides; invalid values fail closed to `0` |
| `SOUL_HEARTBEAT_INTERVAL_S` / `SOUL_JOURNAL_INTERVAL_S` | `300` / `3600` | deterministic snapshot and blog-style journal cadence; changing these does not restart vLLM |
| `VERIFY` | `1` | `0` disables the post-start correctness probe entirely (the page then reports "unverified" and nothing rolls back) |
| `VERIFY_LONG_CONTEXT` | `1` | `0` keeps deterministic short prompts, strict structured output, and the bounded temperature-1 512-token sampling gate, but skips long-context retrieval |
| `VERIFY_NEEDLE_TOKENS` | `32768` | size of the long-context retrieval probe |
| `VERIFY_HEALTH_TIMEOUT_S` | `3600` | health wait after vLLM launch; accommodates first local NFS/cachefilesd page-in. Model download occurs before this timer. |
| `GLM_STATE_DIR` | `<volume>/.glm-config` | where the config state file, known-good config, failures and logs live |
| `MODEL_FAMILY` / `MODEL_VARIANT` | selected by `MODEL_PROFILE` | primary candidate `glm52` / `exl3-tr3-glm53-3.42bpw-500k`; the family names architecture, not checkpoint version. The Flash `glm53` family is withdrawn from this image |
| `SSHD` | `auto` | `auto` starts the bundled key-only sshd when a provider injects a public key and nothing is already listening; `0` never starts it and `1` always tries |
| `CONFIG_SMOKE` | `0` | `1` resolves the config, prints the argv and exits without downloading or touching a GPU |
| `TERMINATE_ENABLED` | `0` | `1` exposes the terminate control on the landing page (startup env only) |
| `TERMINATE_LOCKED` | `0` | `1` hard-locks termination for the life of the container (startup env only) |
| `TERMINATE_PROVIDER` | (auto) | force `vastai`, `runpod`, or `jarvislabs` when detection fails |
| `RUNPOD_TERMINATE_API_KEY` | (unset) | RunPod account API key. Use when the injected pod-scoped key lacks delete permission, is missing/altered, or the target is another pod |
| `JARVISLABS_MACHINE_ID` / `JARVISLABS_REGION` | launcher-provided | numeric VM id and `IN1`, `IN2`, or `EU1`; identify the VM and select its lifecycle backend |
| `JARVISLABS_TERMINATE_API_KEY` | (unset) | opt-in JarvisLabs account key for appliance self-destroy; unlike Vast's injected key, it is not scoped to one VM |
| `TERMINATE_DRY_RUN` | `0` | `1` prepares the destroy request and does not send it |
| `TERMINATE_PROBE` | `1` | `0` skips the read-only credential pre-check |

For the exact candidate checkpoint/template revision, memory profile and GPU
promotion gates, see [README](../README.md#glm-53-full-model-342bpw-500k-candidate-to-official-release).
State-file values override new variant defaults: stage a new state/cache root
and preserve the old deployment for rollback. Known-invalid startup
configurations are refused before launch; an untested variant warning is not
qualification. `CONFIG_SMOKE=1` resolves the CPU-side contract only.

### AIBeast maintenance trial selector

The public `MODEL_PROFILE=glm53-3.42bpw-500k` keeps the exercised rental-floor
defaults. The separate maintenance entry overrides those defaults:
**`MAINTENANCE_TRIAL=parity` is the default first trial**, at the user's
explicit request to try current GLM-5.2 resources first. This is not a change
to production configuration. The
[live baseline](../maintenance/glm53-aibeast-500k/production-baseline.json)
records both production `Config.Env` and actual serving argv.

| environment | parity first trial | explicit `rental-floor` |
|---|---|---|
| `MAX_NUM_SEQS` | 12 | 8 |
| `MAX_NUM_BATCHED_TOKENS` | 3072 | 2048 |
| `VLLM_EXL3_PREFILL_CAPACITY` | 3072 | 1024 |
| `GPU_MEMORY_UTILIZATION` | 0.95 | 0.93 |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 48 | 32 |
| `CUDAGRAPH_CAPTURE_SIZES` | 4,8,12,16,20,24,28,32,36,40,44,48 | 4,8,12,16,20,24,28,32 |
| `VLLM_EXL3_TRELLIS_MAX_M` | 48 | 32 |
| `LMCACHE_L1_INIT_GB` | 125 | 20 |

Both trials retain full 3.42bpw at the pinned revision, 520,192 total tokens,
KV 4,518,907,904 bytes/GPU, TP4/DCP4, interleave 64, native probabilistic MTP3,
online K6, dynamic NVFP4/FP8 RoPE, LMCache RAM ceiling 125 GiB and disk 384 GiB.
No architecture claim requires reduced workspaces; no token/precision reduction
or lower-bit substitution is authorized.

Only after a recorded parity failure or insufficient measured margin may the
operator select `MAINTENANCE_TRIAL=rental-floor`. Unknown selectors are rejected;
there is no automatic fallback. The parity default name is retained, while
rental-floor adds `-rental-floor`, separating stage manifest, state and caches.
The selector is recorded in stage identity/environment; do not reuse a stage
to change trials. Retain the selected trial for subsequent qualification
commands; boot-check and rollback use the saved stage.

The old `200b1841…` image rejects 3072 scheduler/prefill values. **Build and
record a new immutable image digest before parity staging**; the prior digest's
GPU evidence is floor-only, and the new image's parity GPU gate is not yet run.
Safe staging/CPU inspection, after setting `IMAGE` to that new digest:

```bash
: "${IMAGE:?Set IMAGE to the new parity-capable image digest}"
export IMAGE
MAINTENANCE_TRIAL=parity uv run --no-project --python 3.12 maintenance/glm53-aibeast-500k/candidate.py stage
MAINTENANCE_TRIAL=parity uv run --no-project --python 3.12 maintenance/glm53-aibeast-500k/candidate.py config-smoke
```

Run from the repository root. After a justified failure only, use explicit
`MAINTENANCE_TRIAL=rental-floor` for a fresh, separately named stage and smoke.
These commands do not stop production. **The maintenance window has not
opened; no production stop/restart is authorized.**

The [current spot receipts](../TEST_RESULTS.md#jarvislabs-spot-container-proof-2026-09-08)
belong to image `200b1841…` / source `e9623135…`, not later checkout changes.
Gilded r34 plus necessary patches does not inherit all 27 Verdict fixes.
Flash is omitted because its runtime is missing, not because a port is proven
impossible; separate Verdict and historical 393K qualifications do not transfer.
The spot rootfs graft matched 19 critical source, seven module-initializer and
seven image-recorded native-library hashes. Provider NCCL 2.23.4 files remained
disclosed; inspected live processes loaded image NCCL 2.30.4. This is not full
filesystem or OCI isolation/security equivalence. No AIBeast production restart
or image promotion was performed. Predecessor 483634 resumed as **500157**,
which was **confirmed Paused after evidence capture**, preserving old storage
that remained billable while paused. Subsequently, at the user's explicit
request, destruction succeeded; `jl list` returned `[]` and all six resource
counts were zero. The [new destroy receipt](../maintenance/glm53-aibeast-500k/evidence-spot-20260908/destroy.json)
does not rewrite the historical pause evidence.

Operator compatibility corrections ([issue 47](https://github.com/malaiwah/glm52-exl3-vast/issues/47)):
`MODEL_DISPLAY_NAME`, `MODEL_DOWNLOAD_WORKERS` and `ALLOW_UNSUPPORTED_GPU`
are not consumed configuration knobs. Use `SERVED_MODEL_NAME` for dashboard
naming; download concurrency is fixed as described below; driver bypass is
`ALLOW_UNSUPPORTED_NVIDIA_DRIVER`, not a GPU-architecture override.

## DRAM prefix-cache offload and memlock

This tier does not enlarge the
GPU KV pool or make a larger active request fit. It preserves evicted prefix
KV in host DRAM so repeated system prompts, repositories, tool histories and
other large agentic prefixes can be restored instead of recomputed.

Vast accepts only ports, environment variables
and hostname in its template Docker Options, so a `--ulimit memlock=...` entry
there is ignored. Fortunately, gating offload on memlock is measurably a false
gate: a 125 GiB tier offloads normally under a 31 GiB limit because the
connector does not mlock the tier up front. The default is therefore
warn-and-proceed. Completed cache misses and reported load failures can be
recomputed instead of failing a request. This does not make unresolved DMA
safe: LMCache retrieve timeout or health loss with outstanding writes stops
the worker rather than reusing its pages. Set `OFFLOAD_IGNORE_MEMLOCK=0` for
conservative disable-instead behaviour.

This is therefore **possible on Vast, but host-dependent rather than
provider-guaranteed**. The appliance sizes from the container's actual cgroup
memory limit (not the offer headline), and disables the tier when that budget
is unusable. Select a high-RAM offer and confirm the boot log's resolved
aggregate/per-worker capacity. Qwen keeps this off by default until its hybrid
attention/Gated-DeltaNet connector path passes the same external-hit
qualification as GLM.

`OFFLOAD_FRACTION` is an aggregate host-RAM budget. In the pinned native vLLM
connector, `cpu_bytes_to_use` already accounts for the complete TP world and
derives each worker's physical slice; dividing the value by TP again makes the
real cache four times smaller on TP4. The appliance passes the aggregate value
and reports both the total and estimated per-worker slice at boot.

Historical GLM-5.2 evidence, **not refreshed GLM-5.3 qualification**:
the corrected TP4 implementation was exercised on a 251 GiB AIBeast host with
`OFFLOAD_FRACTION=0.5` (125 GiB aggregate). A cold 133,731-token prefix took
52.47 seconds. After five different ~133K prompts forced it completely out of
GPU cache, the same prefix returned from DRAM in 0.69 seconds: 133,504 external
prefix-hit tokens and 9.89 GB loaded CPU-to-GPU across four workers, with zero
allocation failures. That is about **76x lower TTFT than recomputation** for
this agentic-prefix shape. The preallocated tier left about 51 GiB of host RAM
available. Although the configurator permits larger fractions, 50% is the
recommended ceiling on a 256 GiB host; 70% would leave too little operating
margin on this machine.

The refreshed spot run's 131,409-token prefix took 62.4 s then 1.3 s with
native GPU prefix reuse only. After an independent 501,098-token pressure
probe, the original prefix returned correctly in **3.115 s**, with **115,200
external-cache hit tokens** and 15,872 native-hit tokens added. This measures
**LMCache DRAM retrieval after GPU pressure**, with some GPU-resident reuse,
not a pure DRAM-only request or complete GPU eviction. Disk L2, restart
persistence and fault recovery remain unproved; see the
[cache receipts](../TEST_RESULTS.md#jarvislabs-spot-container-proof-2026-09-08).

## Checkpoint downloads

Checkpoint downloads use `huggingface_hub.snapshot_download` with the bundled
`hf-xet` transport and `HF_XET_HIGH_PERFORMANCE=1`, with the file-level worker
count fixed at 16 (`max_workers=16`). Hugging Face's adaptive Xet concurrency
remains the default for each file; advanced deployments can pass through
`HF_XET_FIXED_DOWNLOAD_CONCURRENCY` after measuring their route. An `HF_TOKEN`
authenticates the request and can avoid anonymous rate limits, but does not by
itself guarantee that a particular host-to-CAS route will be fast. See Hugging
Face's [model-download guidance](https://huggingface.co/docs/hub/models-downloading)
and [Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

Supply `HF_TOKEN` using a silent prompt (for example,
`read -rsp "HF token: " HF_TOKEN; export HF_TOKEN; echo`) or a
permission-restricted credential file. Pass the environment variable name,
not its value, to container launchers. Never include literal token values in
command arguments, shell history, documentation or captured evidence.
