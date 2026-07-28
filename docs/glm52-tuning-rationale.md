# GLM-5.2 production-profile deviation ledger

This is the challenge record for the `glm52-exl3` balanced profile, not a list
of folklore flags. The parameter decisions were isolated on the July 27 v31
control (vLLM `0c79e41`, SparkInfer `c3828fd`) using four RTX PRO 6000
Blackwell cards at 280 W, driver 595.71.05 and CUDA 13.2, then exercised on
Vast and Runpod. The release candidate pins GG v20-r5
(`voipmonitor/vllm@sha256:7b230b…`, vLLM integration tree `936ed48`,
SparkInfer tree `f532ec9`); the base refresh remains a requalification
boundary rather than retroactively changing the control's evidence.

The restored safetensors control started on its first attempt, loaded the
target in 57.63 seconds and the draft in 0.78 seconds, directly loaded both
persistent AOT artifacts in 1.10 seconds total, exposed 530,304 KV tokens,
and passed 3/3 short plus 3/3 cold 32K retrieval. It had no startup failure.
Its exact log is preserved at:

`/mnt/vault/llm/vllm+lmcache/turnkey-qualification/20260727/final-safetensors-control-log-audit.log`

## How to read the decisions

- **retain** means the setting is required by the model/checkpoint or has
  direct appliance evidence.
- **pin** means it currently equals the fork default but is kept explicit so
  an actively changing base image cannot silently change a qualified profile.
- **auto** means the machine probe should decide.
- **remove** means no current source consumer or duplicate behavior was found.
- **experiment** means plausible but not promoted without an exact A/B and
  retrieval/degeneration gate.

## API, model, memory, and scheduler parameters

| parameter | upstream/fork default | why this profile differs | benefit / requirement | cost or risk | verdict |
|---|---|---|---|---|---|
| model + tokenizer path | repository ID | immutable local bind | avoids a 309 GiB re-download and leaves shared weights untouched | host must provide a complete checkpoint and marker | retain |
| `trust_remote_code` | off | GLM/EXL3 and optional Glm5v plugin need registered model code | checkpoint can load the required architecture | remote code is privileged; revisions/image are pinned | retain |
| model alias `GLM-5.2`, `local-primary` | path-derived | stable OpenAI-compatible names | clients survive local path/provider changes | aliases must stay documented | retain |
| default `reasoning_effort=high` | model/API dependent | exposes GLM thinking by default while the UI can turn it off | useful reasoning without client-specific template code | more tokens and latency | retain, client-overridable |
| `glm45` reasoning parser | none | converts GLM reasoning into the OpenAI reasoning field | thinking is visible instead of hiding the answer | parser changes are compatibility-sensitive | retain |
| `glm47` tool parser + auto tool choice | off | GLM native tool syntax | tool calls and agentic loops work | parser bugs can affect malformed calls | retain; feature-suite gated |
| prompt-token details + forced stream usage | off | appliance observability and UI accounting | exact PP tests and streamed usage | tiny response/CPU overhead | retain |
| quantization `exl3` | inferred/none | target is an EXL3 rank-sliced checkpoint | makes the 753B model fit four 96 GB cards with measured quality | custom kernel/source layer; SM120-only qualification | retain |
| `MAX_MODEL_LEN=520192` | checkpoint advertises 1,048,576 | target requirement is usable 512K–520K on TP4; 524,288 failed InstantTensor admission | exact ~517K requests pass with output reserve | not the checkpoint maximum | retain |
| RoPE clamp to 520,192 | checkpoint table is 1,048,576 | served and allocated envelopes must agree | avoids building/using unnecessary tables outside admission | changing context requires regeneration/requalification | retain |
| `use_index_cache=true` + GLM sparse-layer pattern | checkpoint/runtime dependent | GLM has learned sparse-attention layers | required for the intended DSA path and long-context performance | incorrect pattern corrupts attention routing | retain; checkpoint-coupled |
| load format `instanttensor` | auto | five first-attempt boots and two exact ~517K retrievals passed | target+draft load 32–33 s versus 60–63 s; avoids the safetensors near-max fragmentation/OOM observed three times | exact 524,288 admission failed twice; profile must stay at 520,192 | retain as balanced default |
| safetensors | auto/fallback | operator-selectable generic loader | conventional loader and valid for shorter gates | failed three times at the 514,432-token boundary on this near-max shape; not full-envelope-qualified | fallback only |
| attention backend `B12X_MLA_SPARSE` | auto | SM120 GLM sparse MLA implementation | required for the measured PP/TG/context envelope | custom fork and kernels | retain |
| MoE backend `b12x` | auto | Blackwell routed-expert path | contributes to >100 tok/s C1 with MTP | hardware-specific and actively developed | retain |
| TP4 | 1/detected | four local cards and rank-sliced target | model fit and full-card utilization | inter-card traffic | retain, derived from detected GPU count |
| DCP2 | 1 | balanced production topology | 513,536+ logical KV capacity in the independent issue-33 study, long-prefill gains, stock-like decode, and 520K admission here | DCP1 decodes faster; DCP4 exposes more context but slows ordinary work | retain balanced profile |
| DCP `a2a` transport | backend default | measured DCP2 route | correct sharded-query exchange on this fork | collective overhead at small batches | retain |
| KV dtype `nvfp4_ds_mla` | auto | calibrated GLM scale file is supplied | enough KV capacity for 520K and good decode | quality depends on exact scales; never use uncalibrated | retain |
| GPU utilization 0.976 | 0.9 | cross-provider admission point with graph profiling | preserves the 520,192-token request envelope while leaving runtime workspace on cards that expose less than AIBeast's usable VRAM | a smaller KV surplus than 0.978; driver changes still require a cold boot | retain |
| explicit block override | auto | zero lets the corrected profiler size it | InstantTensor exposes 523,776 tokens and retains runtime headroom | an exact 4,064-block pin did not rescue safetensors and increased fragmentation | retain auto |
| prefix caching | off/model dependent | agentic workloads repeat large histories | avoids re-prefill when a prefix remains resident | KV bookkeeping/memory | retain |
| DRAM offload 50% | off | L2 prefix cache, not active-GPU-KV expansion | 125.5 GiB aggregate host tier can restore large evicted prefixes faster than PP | experimental connector; host memlock is only ~31 GiB per worker and must be raised for fully pinned pages | retain with explicit warning |
| offload failure policy `recompute` | connector default varies | cache is an optimization | page/load failure does not kill inference | a miss pays full PP | retain |
| maximum sequences 8 | 256 | product concurrency target is C1–C8 | aggregate generation rises through C8 | eight maximum-length sessions cannot coexist; graph/trellis window must cover `8×(1+MTP)` | retain |
| batched tokens 3,072 | model-dependent | matched issue-33 and local performance control | strong long PP while keeping activation memory compatible with 520K | vLLM emits a generic speculative-slot warning | retain; 4,096 failed admission |
| chunked prefill | on for long contexts | explicit for clarity | bounds activation memory and interleaves work | chunk boundaries add scheduling overhead | retain |
| async scheduling | often off | v20 EXL3 release constraint is speculative/CKV lifetime safety | avoids unqualified lifetime races | may leave throughput on the table | retain off |
| graph capture sizes 4..64 step 4 | auto-derived | MTP5 × C8 requires up to 48 query tokens/step | keeps all supported concurrency in captured trellis path | 0.20 GiB/GPU plus 11 s capture | retain |
| maximum graph size 64 | auto | matches the capture list and trellis maximum | avoids eager fallback through C8/MTP5 | larger graph sets consume memory/start time | retain |
| compilation custom ops `all` | defaults | lets fork kernels remain opaque/fused under Inductor | selected B12X/EXL3 paths compile correctly | makes source-version compatibility important | pin |
| fused all-reduce + RMS pass | off/automatic | PCIe Blackwell path | reduces a communication boundary | topology-specific; correctness gated | retain |
| external rank-sliced TR3 draft | no speculation | leaves target byte-identical and saves draft VRAM versus BF16 | restores KV capacity while retaining parity acceptance | extra immutable artifact and compile key | retain |
| MTP tokens 5 | 1/disabled | local MTP5 fit and capture window are verified | high MAL and >100 tok/s C1; direct depth is still bounded | more draft/verify work and query slots | retain |
| probabilistic draft sampling | greedy | selected issue-33/profile behavior | preserves sampling distribution and measured high acceptance at temperature 1 | stores/uses draft logits; more GPU memory | retain |
| standard rejection sampling | standard | no divergence today | distribution-preserving, established control | may accept fewer smooth near-miss drafts than block verification | retain control |
| block verification | opt-in | matched GLM temperature-one A/B plus exact long-context gate | passed the exact ~517K 5/5 retrieval; C1 improved 14.4% and C8 1.5% | C2 fell 7.5% and C4 10.9%; benefit is output/concurrency-sensitive | experiment only |
| structured-output backend `auto` | auto | no forced backend | tool calling remains independent and usable | schema compilation can add latency; GLM reasoning+MTP edge cases exist upstream | retain optional |
| vision | off | both tested compositions failed one mandatory gate | preserves full text quality/context | no default image capability | retain off |

## Runtime environment deviations

These values are read by the pinned vLLM/SparkInfer source or by CUDA/NCCL.
Rows whose value equals the current source default are still called out.

| environment | source default | selected value and reason | downside | verdict |
|---|---|---|---|---|
| `CUDA_DEVICE_MAX_CONNECTIONS` | CUDA default/driver-selected | `32`, matching the v20 overlap stack | may hurt another topology | retain, topology re-test |
| `CUTE_DSL_ARCH` | detected | `sm_120a`, exact RTX PRO 6000 target | not portable to pre-Blackwell | retain |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | broad/detected | `12.0a` / `12.0f` to keep compiled artifacts architecture-specific | artifacts cannot be reused on another architecture | retain |
| `OMP_NUM_THREADS` | process default | `16`, bounded CPU helper parallelism | may be non-optimal on smaller rental CPUs | retain, host-derived later |
| `SAFETENSORS_FAST_GPU` | off | `1`, direct GPU-oriented loader | loader-specific memory behavior | retain for safetensors |
| `PYTORCH_CUDA_ALLOC_CONF` | allocator default | empty while offload is enabled because VMM expandable segments can remap pinned KV pages | loses expandable-segment fragmentation protection | retain with offload; restore when offload is off |
| `NCCL_IB_DISABLE` | auto | `1`, AIBeast/rental is local PCIe without IB | disables a real IB fabric if present | retain for single-node provider profiles only |
| `NCCL_P2P_LEVEL` | auto | `SYS`, permits cross-root PCIe P2P | may choose a slower path on unusual hosts | retain; topology probe required |
| `NCCL_PROTO` | auto | `LL,LL128,Simple`, leaves all relevant protocols available | prevents future NCCL policy from using a new protocol | pin for tested image |
| local NCCL preload/path variables | system NCCL discovery | pinned local-inference NCCL 2.30.4 library | ABI/version coupling | retain with image digest |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 s | 2,400 s for 300+ GiB cold starts and graph compilation | slower failure reporting | retain; supervisor still reports phases |
| `VLLM_USE_FLASHINFER_SAMPLER` | true | `1`, same as fork default | none beyond FlashInfer dependency | pin |
| `VLLM_USE_AOT_COMPILE` | false | `1` | persistent compiled model reloads in ~1.1 s instead of recompiling | stale/corrupt caches were a historical risk; fingerprint and retrieval gate are mandatory | retain |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | false | `1` | reconstructs 77 target artifacts and two draft artifacts directly | disk usage and strict software/config fingerprint | retain |
| compile/autotune cache directories | user cache | immutable-checkpoint-external persistent volume | reuse survives container replacement and cannot mutate weights | consumes local SSD; invalidate on fingerprint change | retain |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | true | `1`, same as fork default | accounts graph high-water before KV allocation | reports less KV than older optimistic behavior | pin for safety |
| `VLLM_MEMORY_PROFILE_INCLUDE_ATTN` | false | `1` | includes persistent DCP/attention resources before sizing KV | lower advertised KV is possible | retain for reliable high-GMU boot |
| `VLLM_USE_V2_MODEL_RUNNER` | auto | `1`, release-qualified execution path | new runner changes rapidly | retain with pinned image |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | false | `0`, same as default | full qualified graph behavior | less flexible dynamic escape | pin |
| `VLLM_EXL3_EXT_PATH` / `VLLM_EXL3_ABI_SHIM` | unset | pinned extension directory and Torch 2.12 shim | exact ABI coupling | retain; required |
| `VLLM_EXL3_TRELLIS_BLOCK_M` | 8 | `8`, validated kernel geometry | none versus current default | pin |
| `VLLM_EXL3_PREFILL_BLOCK_M` | 64 | `64`, issue-33/EXL3 planned prefill | another GPU might prefer a different tile | pin measured geometry |
| `VLLM_EXL3_PREFILL_CHUNK` | 128 | `1`, selected DCP2 issue-33 route | more planner/launch granularity; surprising versus source default | retain only because profile evidence includes it; isolate in future A/B |
| `VLLM_EXL3_TRELLIS_MAX_M` | 32 | `64` | covers C8×MTP5 query geometry without eager fallback | larger arena/compile surface | retain |
| `VLLM_NVFP4_MLA_SCALES_FILE` | unset | GLM-calibrated outer-scale JSON | wrong file silently harms quality | retain and checksum |
| `VLLM_DISABLED_KERNELS` | empty | disable `MarlinFP8ScaledMMLinearKernel` | removes a generic fallback | retain; checkpoint path is not Marlin |
| `VLLM_USE_B12X_FP8_GEMM` | false | `1`, SM120 dense/shared path | custom kernel risk | retain with quality/perf gates |
| `VLLM_USE_B12X_SPARSE_INDEXER` | false | `1`, learned GLM DSA indexer | selector changes can alter deep retrieval | retain; cold maximum needles mandatory |
| `VLLM_USE_B12X_MOE` | false | `1`, routed-expert kernel | hardware-specific | retain |
| `VLLM_USE_B12X_MHC` / `B12X_MHC_MAX_TOKENS` | false / launcher-specific | `1` / 16,384, captured v20 fusion envelope | extra custom fusion surface | retain pending isolated upstream study |
| `VLLM_USE_B12X_WO_PROJECTION` | false | `1`, Blackwell output-projection path | custom path | retain with exact-output gate |
| `B12X_MOE_FORCE_A16` | off | `1`, avoids lossy activation quantization in MoE | can be slower and use more bandwidth than A8 | retain for quality |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | true | `1`, same as fork default | none versus default | pin |
| shared-expert stream disable | false | `0`, explicitly re-enables stream | issue-33 profile retained owner/stream behavior | stream/lifetime complexity | retain measured profile |
| shared-expert stream threshold | 256 | `16` | starts overlap at smaller batches, matching issue-33 | can add stream overhead at small work | retain measured profile |
| multi-stream GEMM threshold | 1,024 | `1,024`, current default | documents measured crossover | none versus default | pin |
| `SPARKINFER_FUSED_INDEXER` | enabled | `1`, current default | exact fast indexer | selector is quality-sensitive | pin |
| `SPARKINFER_INDEXER_DIRECT_K` | enabled | `1`, current default | avoids compatibility fallback | actively developed path | pin |
| `SPARKINFER_INDEXER_STREAM_SCORER` | implementation default | `1`, issue-33 path | extra stream coordination | retain measured profile |
| `SPARKINFER_MLA_PREFILL_STRATEGY` | auto | `auto` | lets supported GLM shapes select the correct prefill kernel | behavior may change with source revision | retain with pinned revision |
| `SPARKINFER_MLA_SM120_NUM_SPLITS` | heuristic when 0/unset | `0` | preserves shape-aware heuristic | not fully reproducible across future code | pin source revision |
| `SPARKINFER_MLA_SM120_PREFILL_MG` | enabled | `1`, current default and required by tested long PP path | unsupported shapes must fall back/fail clearly | pin |
| paged-index supertile K | 32,768 | 32,768 in vLLM and SparkInfer | keeps both sides on identical page geometry | larger/smaller context distributions might prefer another value | pin |
| `SPARKINFER_W4A16_SMALL_M_DIRECT` | enabled | `1`, current default | direct small-M decode path | kernel-specific | pin |
| `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE` | `auto` | `1`, selected issue-33 MTP route | can be wrong for another MTP/attention shape | retain profile-specific |
| `VLLM_B12X_MLA_SPEC_DECODE_MAX_Q` | 8 | `8`, current default | bounds fast speculative decode Q | none versus default | pin |
| `VLLM_USE_B12X_DCP_A2A` | false | `1` | activates the B12X DCP exchange implementation | custom collectives | retain |
| DCP A2A max / large backend | 0 / `ag_rs` | 16 / `ag_rs` | A2A for small Q, AG/RS above crossover | topology-sensitive | retain pending direct crossover A/B |
| `VLLM_DCP_GLOBAL_TOPK` | true | `1`, same as default | exact global sparse candidates | collective cost | pin |
| `VLLM_DCP_SHARD_DRAFT` | unset | `1` | draft follows DCP geometry instead of replication | communication/implementation complexity | retain; MTP quality gated |
| `VLLM_DCP_INDEXER_SHARDS` | 0 | `0`, fully sharded | maximum KV capacity | partial replication can improve decode ~6–9% at ~5% KV cost upstream | retain for 520K requirement |
| `VLLM_DCP_TOPK_OWNER_MERGE` | false | `0`, same as default and issue-33 winner | avoids the slower/mixed owner merge route | gives up a potential topology-specific win | pin |
| `VLLM_DCP_QUERY_SPLIT` | false | `1` | materially improves long prefill | overhead on short prompts | retain |
| query-split minimum context | 0 | 8,192, measured crossover | avoids short-prompt regression while retaining long PP | machine-specific | retain/profile pin |
| full CKV gather | false | `1` | largest measured long-prefill win | consumes KV/workspace and collective bandwidth | retain |
| CKV gather min | 16 | 512 | avoids gather overhead on small prompts | delays fast path | retain profile pin |
| CKV gather max | 524,288 | 140,000 | measured +7.1% at 64K and +3.9% at 128K in the independent DCP2 study; prior 16,384 was too low | costs about 13,824 KV tokens in that stack | retain |
| CKV prefetch depth | 1 | `0` | AIBeast calibration found no overlap gain; DCP2 borrowed workspace is unsupported | gives up possible overlap on a better topology | retain `auto` machinery, selected 0 here |
| CKV prefetch workspace | 1,024 MiB | 1,024 MiB | same as source default; needed if another topology selects prefetch | reserved only when active | pin |
| project-before-merge / gather-in-workspace | false / false | false / false for DCP2 | the TP4/DCP2 16/32/32 head contract rejects borrowed workspace | misses DCP4 fast workspace path | retain topology guard |
| project threshold | 1,024 | 1,024 | same as default | none | pin |
| `VLLM_ENABLE_PCIE_ALLREDUCE` | false | `1` | PCIe-oriented collective path materially improves DCP/TP traffic | output-lifetime bugs existed in earlier revisions | retain only with fixed pinned stack |
| PCIe all-reduce backend | `cpp` | `b12x` | selected Blackwell path | custom code and topology sensitivity | retain |
| one-shot AR max | 84 KiB | 64 KiB | measured control cutoff | may leave 64–84 KiB on slower fallback | retain profile pin |
| fused add/RMS max | 84 KiB | 84 KiB | current default | none versus default | pin |
| C++ AR first-stage cutoff | backend-derived | 56 KiB | measured local NCCL/C++ crossover | topology-specific | retain |
| ignore-cutoff max rows | backend-derived | 0 | do not bypass the measured cutoff | may miss a shape-specific shortcut | retain |
| PCIe DMA wire dtype | lossless | `0` for vLLM/B12X/SparkInfer | avoids the known quality risk of compressed collectives | more PCIe bytes | retain lossless |
| PCIe DMA minimum bytes | 6 MiB | calibrated 25,165,824 bytes on this GPU order | uses DMA only beyond its measured crossover | calibration costs startup once and is fingerprint-specific | auto |
| allow cross-NUMA one-shot | true | `1`, same as default; host is one NUMA node | keeps all GPU pairs eligible | bad on some multi-socket hosts | pin current topology |
| one-shot single channel | false | `0`, same as default | retains parallel channels | more resources | pin |

## Removed after the audit

| setting | finding | action |
|---|---|---|
| `VLLM_USE_B12X_PCIE_DMA` | no consumer in the pinned vLLM/SparkInfer source; real transport selectors are `VLLM_ENABLE_PCIE_ALLREDUCE`, backend, DMA dtype, and threshold | removed |
| `VLLM_RTX6K_FUSED_ALLREDUCE_ADD` | no source consumer | removed |
| `VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER` | no source consumer | removed |
| second vision `--trust-remote-code` | generic GLM arguments already pass it | removed |

The remaining “unknown vLLM environment” log lines are registration cosmetics
for variables that do have current source consumers. Upstream draft PR
`local-inference-lab/vllm#186` registers them without disabling unknown-variable
detection. EXL3 registrations are already covered by upstream PR #139.

## Warnings that do not justify changing the profile

- The alternating `204.8 / 0 / 204.8` PP logger is bucket accounting for
  2,048-token chunks, not half-speed execution. End-to-end unique prompt tokens
  divided by prefill time remain the benchmark.
- `num_speculative_tokens > 1` and the 3,072 scheduled-token warning are generic.
  This exact MTP5 shape passes retrieval, quality, and C1–C8 performance. A
  4,096 A/B is still worthwhile; the warning alone is not evidence.
- native P2P atomics are unavailable across these PCIe devices. Symmetric-memory
  collectives fall back; the selected B12X/NCCL paths remain active.
- the OffloadingConnector API is marked experimental. That is a reliability
  caveat, not an indication that the measured DRAM cache is inactive.
- `temperature=1.0, top_p=0.95` comes from the model's generation config.
  Because `do_sample=false` in that file, the earlier Transformers warning was
  benign; production verbosity is back at its default.
- three Triton kernels compiled during the first 32K verifier request. The
  compiled artifacts persist, and subsequent logs show SparkInfer disk-cache
  hits. Verification intentionally pays this cold-shape cost before a profile
  is marked known-good.

## Next decision experiments

Only changes with a plausible path to a product requirement advance:

1. MTP3 versus MTP5 is rechecked under sustained real agent traffic after the
   controlled decode artifact is captured. Compare accepted and drafted
   counter deltas, normalized draft acceptance `(MAL-1)/depth`, aggregate
   throughput, latency, and errors; raw MAL alone is not comparable because
   MTP5 has a higher ceiling.
2. Query-split/collective crossovers are recalibrated only after the planned
   driver/OS refresh; cached results include driver, GPU order, affinity, and
   software revision.
3. Partial indexer replication is not a balanced-profile candidate while the
   520K requirement needs the roughly 5% KV capacity it consumes.
4. New sparse-indexer selection policies from SparkInfer PRs #82/#84 are not
   applied universally. The current Brandon control already passed the
   five-depth 521K gate; a checkpoint-specific policy requires a frozen cold
   discriminator showing a real failure and a no-regression ladder.

The proposed 4,096-token batch was tested and rejected before benchmarking.
It created a correctly distinct compile key and compiled normally, but raised
the EXL3 target/draft runtime from 77.66 to 77.83 GiB/GPU and enlarged each
Trellis arena from the 3,072 shape to 1,374.2 MiB. Available KV fell from
9.15 to 7.57 GiB: 8.97 GiB was required for 520,192 tokens, and vLLM estimated
only 438,656 usable tokens. This is a direct product-requirement failure, not a
small throughput trade. The failed log and inspect are preserved beside the
control as `batch4096-admission-failure.*`.
