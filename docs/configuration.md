# Configuration reference

The complete environment-knob table, extracted from the README. Values resolve
as `defaults < family < startup env < state file`; see
[self-service-config.md](self-service-config.md) and
[model-families.md](model-families.md).

## Environment variables

| env | default | why you'd change it |
|---|---|---|
| `MODEL_PROFILE` | `glm52-exl3` | select `qwen36-27b-nvfp4` for low-cost testing, or `custom` with `MODEL_ID` |
| `MODEL_ID` | profile checkpoint | use a compatible alternate checkpoint without changing its profile defaults |
| `MODEL_DIR` | profile-specific path under `/workspace` | point at weights you already have; the download marker is tied to `MODEL_ID` |
| `MODEL_DISPLAY_NAME` | profile name | dashboard and provider label |
| `SERVED_MODEL_NAME` | profile name | whitespace-separated aliases, so existing clients keep working |
| `TENSOR_PARALLEL_SIZE` | 4 GLM / 1 Qwen | match a supported profile topology |
| `MAX_MODEL_LEN` | 524288 GLM / 196608 Qwen | change a qualified scheduler/memory envelope only with a fresh near-maximum retrieval and vision gate |
| `MULTIMODAL` | n/a GLM / 1 Qwen | Qwen `0` saves vision VRAM with `--language-model-only`; GLM vision remains controlled by `VISION` (default 0) |
| `MM_MAX_PIXELS` | n/a GLM / 8388608 Qwen | cap native image processing near a 4K working image; the 5K detail gate passed at this value |
| `QUANTIZATION` | custom profile only | vLLM quantizer name such as `modelopt` |
| `REASONING_PARSER` / `TOOL_CALL_PARSER` | custom profile only | model-specific OpenAI response parsers |
| `AUTH` | `key` | `none` serves unauthenticated on a trusted LAN |
| `ALLOW_UNSUPPORTED_GPU` | `0` | bypass the profile GPU-name check; the required visible GPU count still applies |
| `MIN_NVIDIA_DRIVER_VERSION` | `590.48.01` | lower bound of the qualified driver/CUDA pair; the gate runs before model download |
| `MIN_NVIDIA_CUDA_VERSION` | `13.2` | reported CUDA capability paired with the driver floor; prevents an r590/CUDA 13.1 host from passing |
| `ALLOW_UNSUPPORTED_NVIDIA_DRIVER` | `0` | bypass both admission floors only for a separately qualified compatibility stack |
| `GPU_BLOCKS_OVERRIDE` | 0 | auto-profile the largest safe KV pool; a positive value pins vLLM blocks, not tokens. On this MLA stack the reported logical capacity is `blocks × 64 × DCP` (for example, DCP4 needs 2,048 blocks—not 8,192—for exactly 524,288 tokens). Re-verify this relationship after an engine/topology change. |
| `KV_CACHE_MEMORY_BYTES` | 0 | positive per-GPU byte count fixes KV memory and supersedes GMU for KV sizing; use the profiler's printed value when eager/speculative workspace must not be consumed by auto-KV, and do not combine it with `GPU_BLOCKS_OVERRIDE` |
| `OFFLOAD_FRACTION` | 0.5 GLM / 0 Qwen | host DRAM used as an aggregate L2 prefix cache (not active-context capacity); `0.5` is the measured agentic-workload setting on a 256 GiB host and native vLLM derives the TP worker slices |
| `OFFLOAD_IGNORE_MEMLOCK` | `1` | proceed when the memlock ulimit is below the tier size (see below); `0` disables offload instead |
| `PREFIX_CACHE_BACKEND` | `lmcache` GLM / `native` other profiles | `lmcache` is the r13-qualified supervised DCP-aware process; `native` keeps the in-process OffloadingConnector rollback control. Both use `OFFLOAD_FRACTION` for aggregate DRAM and neither enlarges active context. |
| `LMCACHE_L1_INIT_GB` | min(20, configured L1) | initial LMCache DRAM arena; the remaining configured tier grows lazily. Raise only if first-hit allocation latency matters more than model page-in and host-memory headroom. |
| `PREFIX_CACHE_DISK_GB` | `0` | positive values enable LMCache's native filesystem L2 with this hard GiB limit under `<MODEL_ROOT>/.lmcache`; derived prompt KV may be sensitive, so prefer encrypted local NVMe and enable best-effort secure termination |
| `MTP78_MODE` | `off` (native) | the current Brandon revision contains a native rank-sliced TR3 draft; `graft` and `override` remain experimental compatibility paths. MadeBy561's native draft uses serialized NVFP4 experts. Prefer the `MTP_DRAFT` knob on the config page. |
| `MTP_DRAFT_SAMPLE_METHOD` | `probabilistic` GLM | measured MTP-5 proposal mode; `greedy` remains available for controlled A/B tests |
| `F8_DMA` | `0` family / `ring` MadeBy561 | compressed PCIe collective mode; the hybrid override passed the 521K five-depth gate |
| `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | `-1` family / `8192` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured crossover |
| `PCIE_DMA_MIN_BYTES` | `-1` family / `393216` MadeBy561 | `-1` keeps topology calibration; the hybrid pins its measured byte crossover |
| `OPEN_BUTTON_TOKEN` | provider-specific | required to expose the `:1111` config editor; Vast supplies it and Runpod/JarvisLabs get a persisted generated token when one is not set |
| `SOUL_AUTONOMY_LEVEL` | `0` | enable the embedded diagnostic SOUL: Observe `1` (no shell), Investigate `2` (bounded read-only shell), or Verify `3` (idle-only canary and conditional long-context probe) |
| `SOUL_AUTONOMY_MAX_LEVEL` | `3` | startup-only ceiling for landing-page overrides; invalid values fail closed to `0` |
| `SOUL_HEARTBEAT_INTERVAL_S` / `SOUL_JOURNAL_INTERVAL_S` | `300` / `3600` | deterministic snapshot and blog-style journal cadence; changing these does not restart vLLM |
| `VERIFY` | `1` | `0` disables the post-start correctness probe entirely (the page then reports "unverified" and nothing rolls back) |
| `VERIFY_LONG_CONTEXT` | `1` | `0` keeps the short-prompt checks only — read the warning above before using it |
| `VERIFY_NEEDLE_TOKENS` | `32768` | size of the long-context retrieval probe |
| `VERIFY_HEALTH_TIMEOUT_S` | `3600` | health wait after vLLM launch; accommodates first local NFS/cachefilesd page-in. Model download occurs before this timer. |
| `GLM_STATE_DIR` | `<volume>/.glm-config` | where the config state file, known-good config, failures and logs live |
| `MODEL_FAMILY` / `MODEL_VARIANT` | selected by `MODEL_PROFILE` | `glm52`/`exl3-tr3`, `qwen36`/`qwen36-nvfp4`, or `custom`/`custom`; the config page can switch these without rebuilding |
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
warn-and-proceed, and it degrades rather than fails —
`kv_load_failure_policy=recompute` means any KV block that cannot be fetched
back is recomputed instead of erroring the request. Set
`OFFLOAD_IGNORE_MEMLOCK=0` for conservative disable-instead behaviour.

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

The corrected TP4 implementation was exercised on a 251 GiB AIBeast host with
`OFFLOAD_FRACTION=0.5` (125 GiB aggregate). A cold 133,731-token prefix took
52.47 seconds. After five different ~133K prompts forced it completely out of
GPU cache, the same prefix returned from DRAM in 0.69 seconds: 133,504 external
prefix-hit tokens and 9.89 GB loaded CPU-to-GPU across four workers, with zero
allocation failures. That is about **76x lower TTFT than recomputation** for
this agentic-prefix shape. The preallocated tier left about 51 GiB of host RAM
available. Although the configurator permits larger fractions, 50% is the
recommended ceiling on a 256 GiB host; 70% would leave too little operating
margin on this machine.

## Checkpoint downloads

Checkpoint downloads use `huggingface_hub.snapshot_download` with the bundled
`hf-xet` transport and `HF_XET_HIGH_PERFORMANCE=1`. `MODEL_DOWNLOAD_WORKERS`
defaults to 16 concurrent files. Hugging Face's adaptive Xet concurrency remains
the default for each file; advanced deployments can pass through
`HF_XET_FIXED_DOWNLOAD_CONCURRENCY` after measuring their route. An `HF_TOKEN`
authenticates the request and can avoid anonymous rate limits, but does not by
itself guarantee that a particular host-to-CAS route will be fast. See Hugging
Face's [model-download guidance](https://huggingface.co/docs/hub/models-downloading)
and [Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).
