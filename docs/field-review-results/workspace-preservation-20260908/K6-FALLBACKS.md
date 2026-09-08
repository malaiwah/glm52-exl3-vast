# K6 Bring-up Fallback Log

Tracking every fallback/workaround used during GLM-5.3 Flash K6 bring-up.
Each must be fixed properly before the K6 appliance can be declared "ultimate".

## Current resolution (2026-08-29)

The hard blocker recorded below is resolved. The checkpoint now serves at
TP4/DCP4 with B12X, 520,192-token request capacity, NVFP4 DS-MLA KV, chunked
prefill, and prefix caching. The clean production verifier passed all short,
structured-output, and 32K three-needle checks.

The production process reuses
`/home/turnkey/.triton-warningfix-20260829-v8`. It runs with the default vLLM
JIT warning mode, `jit_monitor_verbose=False`, and no `sitecustomize` callsite
tracer. Startup and post-verification logs are warning- and error-free.

### Exact production mount manifest

`$VLLM` below is `/opt/infernal-invocation/vllm/vllm`; `$B12X` is
`/opt/infernal-invocation/b12x/b12x`. Every file overlay is read-only.

| Host source | Container destination | Purpose |
|-------------|-----------------------|---------|
| `/home/turnkey` | `/home/turnkey` | Checkpoint, launch script, caches, and patch sources |
| `patches/exl3_patched.py` | `$VLLM/model_executor/layers/quantization/exl3.py` | K6 EXL3 load/dispatch and route warmup |
| `patches/w4a16_kernel_k6.py` | `$B12X/moe/_shared/kernels/w4a16/kernel.py` | K6 W4A16 planning and kernels |
| `patches/b12x_fused_moe_impl_image.py` | `$B12X/moe/fused_moe/_impl.py` | Fused-MoE route preparation |
| `patches/tilelang_kernels_mhc_serial.py` | `$VLLM/model_executor/kernels/mhc/tilelang_kernels.py` | Serial mHC TileLang coverage |
| `patches/dcp_utils_layout.py` | `$VLLM/v1/attention/ops/dcp_utils.py` | Direct DCP A2A layout fallback |
| `patches/torchao_utils_enum.py` | `/usr/local/lib/python3.12/dist-packages/torchao/utils.py` | Avoid obsolete Enum pytree registration |
| `patches/envs_b12x_glm_nope.py` | `$VLLM/envs.py` | Remove the stale unknown B12X environment binding |
| `patches/kda_no_block_ptr.py` | `$VLLM/third_party/flash_linear_attention/ops/kda.py` | Current Triton pointer API |
| `patches/solve_tril_no_block_ptr.py` | `$VLLM/third_party/flash_linear_attention/ops/solve_tril.py` | Current Triton pointer API |
| `patches/chunk_delta_h_no_block_ptr.py` | `$VLLM/third_party/flash_linear_attention/ops/chunk_delta_h.py` | Current Triton pointer API |
| `patches/rotary_common_image.py` | `$VLLM/model_executor/layers/rotary_embedding/common.py` | Restore bundled rotary dispatch |
| `patches/deepseek_v4_mhc_warmup.py` | `$VLLM/model_executor/warmup/deepseek_v4_mhc_warmup.py` | mHC runtime-shape warmup |
| `patches/generation_config_vllm.json` | `/home/turnkey/GLM-5.3-Flash-TR3-6bpw/generation_config.json` | Use vLLM generation defaults |
| `patches/pcie_oneshot_device_barrier.py` | `$B12X/comm/pcie/pcie_oneshot.py` | Device-bound close barriers |
| `patches/mla_attention_sparse_info.py` | `$VLLM/model_executor/layers/attention/mla_attention.py` | Correct sparse-only prefill notice level |
| `patches/glm5next_model_router_once.py` | `$VLLM/models/glm5next/nvidia/model.py` | Apply the MoE router gate once |
| `patches/route_pack_image_exact.py` | `$B12X/moe/_shared/kernels/w4a16/route_pack.py` | Runtime route bounds and stable scalar signatures |
| `patches/sparse_mla_triton_warmup.py` | `$VLLM/model_executor/warmup/sparse_mla_triton_warmup.py` | In-process sparse metadata warmup |
| `patches/indexer.py` | `$VLLM/v1/attention/backends/mla/indexer.py` | Stable metadata compile keys and pooled ratio coverage |
| `patches/kpool_compress.py` | `$VLLM/models/glm5next/nvidia/ops/kpool_compress.py` | Runtime K-pool tail token count |
| `patches/glm5_kpool_warmup.py` | `$VLLM/model_executor/warmup/glm5_kpool_warmup.py` | Initialized-cache K-pool pointer variants |
| `patches/kernel_warmup_image_exact.py` | `$VLLM/model_executor/warmup/kernel_warmup.py` | Integrate route, metadata, and K-pool warmups |

`patches/sitecustomize_callsites.py` is intentionally not mounted in
production. The full launch remains `/home/turnkey/k6-serve-b12x.sh`.

## Fallback 1: EXL3 bits validation patched (K4-only → K4/K6/K8)
- **File**: `exl3.py` line 1130, `expected["bits"]` hardcoded to 4
- **Patch**: Accept `(4, 6, 8)` tuple; comparison uses `in` instead of `==`
- **Also**: Line 1269 `"bits": 4` → `"bits": hf_config.quantization_config.get("bits", 4)` (was `config` → NameError, fixed to `hf_config`)
- **Also**: Line 1285 log message "K4 experts" → "EXL3 experts"
- **Status**: Needs upstream PR to `local-inference-lab/vllm` — the bits validation should accept any valid EXL3 bitrate (3-8) for the `glm53_routed_experts_only` scope.

## Fallback 2: TP size check bypassed for unsliced_tp_stream checkpoints
- **File**: `exl3.py` line 2758, `checkpoint_tp != layer.exl3_tp_size` raised ValueError
- **Patch**: Skip TP check when `source_layout == "unsliced_tp_stream"` (checkpoint is topology-neutral by design)
- **Status**: Correct fix — the K6 checkpoint stores experts unsliced, so any TP size is valid. Needs upstream PR.

## Fallback 3: B12X MoE disabled, using triton backend (FAILED)
- **File**: `k6-serve.sh`, `--moe-backend b12x` → `--moe-backend triton`, `VLLM_USE_B12X_MOE=1` → `0`
- **Reason**: B12X W4A16 kernel has no tile config for K6 expert shapes: `ValueError: no valid W4A16 tile config for M/N/K=3072/1024/4096, moe_block_size=128`
- **Root cause**: B12X only supports K4 trellis. K6 needs B12X PR #245 (generalize cooperative fused Trellis decode to K2-K6/MCG).
- **Result**: Same crash even with triton backend — the EXL3 quant method's `_apply_rank_sliced()` calls B12X `plan_tp_moe_scratch` → `compile_w4a16_fused_moe` → `_select_tile_config` directly, regardless of the `--moe-backend` flag. The MoE backend controls dispatch; the EXL3 quant layer has its own B12X code path.
- **Resolution**: This was a hard blocker for the initial image. The current runtime includes the K2–K6/MCG tile work from B12X #245; expert inference, memory profiling, and graph capture now complete.

## Fallback 4: load-format safetensors (not fastsafetensors)
- **File**: `k6-serve.sh`, `--load-format fastsafetensors` → `--load-format safetensors`
- **Reason**: fastsafetensors loader crashed with `TypeError: Input tensor data type is not supported` during NCCL broadcast of K6 trellis tensors at TP4
- **Status**: Investigate whether fastsafetensors can handle K6 dtypes with a patch, or if this is a fundamental limitation.

## Image being used
- `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2` (Brandon v84)
- This is the "bring-up base" per HANDOFF.md two-base strategy
- NOT the ultimate appliance — it's a diagnostic milestone
- The final appliance will be built from our turnkey image + GLM-5.3 patches

## Checkpoint
- `malaiwah/GLM-5.3-Flash-TR3-6bpw` (237 GB, 120 shards)
- Architecture: `Glm5NextForConditionalGeneration`
- Quantization: EXL3 K6 (routed experts only, MCG codebook, BF16 non-routed)
- Published `serving_reader_qualified=false` remains unchanged pending the formal bound receipt; live TP4/DCP4 diagnostic verification passes.

## Resolution summary

- EXL3 K6 validation and dynamic bits metadata are patched.
- Unsliced TP-stream checkpoints are accepted at TP4.
- All 120 weight shards load successfully.
- B12X K6 expert inference is active; the failed Triton-backend detour is not
  used.
- KV-cache sizing and CUDA graph capture complete at GMU 0.93.
- `safetensors` remains the qualified load format; `fastsafetensors` is not
  claimed.
- Follow-up warning fixes are upstream as B12X #256/#257, vLLM #54345,
  and TorchAO #4849.
