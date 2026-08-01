#!/usr/bin/env python3
"""Single source of truth for the self-service config: knob registry, the
three-layer inheritance model, and the pre-validation matrix.

Imported by landing.py (editor UI + apply) and driven from entrypoint.sh
through config_cli.py. Stdlib only — it has to run before anything is
installed and inside the landing page process.

INHERITANCE (lowest to highest precedence)
    1. built-in defaults          (this file)
    2. startup environment        (env of the container at PID-1 start,
                                   snapshotted once so a mid-life env change
                                   cannot silently reorder the layers)
    3. state file on the volume   ($STATE_DIR/config.json, written by the
                                   landing page, survives restarts)

The file wins over env deliberately: env comes from the rental template, which
the user cannot edit without destroying and re-renting the instance. The
landing page is the only way to change anything after launch, so it must be
able to override what the template set.

Every rule in VALIDATIONS below was measured on real hardware; the `why`
strings are the operator-facing explanation and are quoted verbatim in
docs/self-service-config.md.
"""
import datetime
import json
import os
import re

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/GLM-5.2-EXL3-TR3-3.0bpw")


def state_dir() -> str:
    """Persistent config dir, next to the model dir (i.e. on the volume)."""
    return os.environ.get("GLM_STATE_DIR") or os.path.join(
        os.path.dirname(MODEL_DIR.rstrip("/")) or "/workspace", ".glm-config")


def runtime_dir() -> str:
    """Per-container scratch. Must NOT be on the volume: a stale restart flag
    or verify verdict surviving a container replacement would fire once more
    against a config that was never applied."""
    return os.environ.get("GLM_RUNTIME_DIR", "/tmp/glm-runtime")


def p_state() -> str:        return os.path.join(state_dir(), "config.json")
def p_known_good() -> str:   return os.path.join(state_dir(), "known-good.json")
def p_apply_state() -> str:  return os.path.join(state_dir(), "apply-state.json")
def p_failures() -> str:     return os.path.join(state_dir(), "failures")
def p_logs() -> str:         return os.path.join(state_dir(), "logs")
def p_baseline() -> str:     return os.path.join(state_dir(), "checkpoint-baseline.json")
def p_startup_env() -> str:  return os.path.join(runtime_dir(), "startup-env.json")
def p_restart_flag() -> str: return os.path.join(runtime_dir(), "restart-request")
def p_verify() -> str:       return os.path.join(runtime_dir(), "verify.json")
def p_verify_last() -> str:  return os.path.join(state_dir(), "verify-last.json")
def p_config_env() -> str:   return os.path.join(runtime_dir(), "config.env")


# --------------------------------------------------------------------------
# model families
# --------------------------------------------------------------------------
# A FAMILY owns everything that is true about a model architecture rather than
# about this deployment: which checkpoints exist, which vLLM flags the engine
# needs, which knobs even mean anything, and which of the measured failure rules
# apply. The serve line used to hard-code GLM-5.2's answers to all of that —
# including `--tensor-parallel-size 4` as a literal, which made the image
# unstartable on a 1-GPU host.
#
# Split of responsibility, deliberately not clean-looking:
#   * CLI ARGS live here, because the landing page has to be able to show what a
#     family implies and to reject knobs that cannot apply to it.
#   * The ENV BLOCK stays in entrypoint.sh, guarded by family. That block is the
#     "each one is a measured decision" list, it has to run AFTER the image's own
#     ENV to override it, and the TUNE_ mechanism is defined in terms of it.
#     Moving it here would have made a second source of truth for the one thing
#     in this repo with the most expensive history behind it.
#
# `tested` means: someone booted it and measured the result. Only glm52 is.

FAMILIES = {
    "glm52": {
        "label": "GLM-5.2 753B (MLA + EXL3/NVFP4) — the measured default",
        "tested": True,
        "env_block": "glm52",
        "default_variant": "exl3-tr3",
        "kv_dtypes": ["fp8", "nvfp4_ds_mla"],
        "spec_method": "mtp",
        # knobs that only make sense here; everything else is generic
        "own_knobs": ("MTP_DRAFT", "DRAFT_MODEL", "DRAFT_QUANTIZATION", "DCP",
                      "MTP_DRAFT_SAMPLE_METHOD",
                      "MTP_REJECTION_SAMPLE_METHOD",
                      "KV_SCALE_MODE",
                      "VLLM_EXL3_TRELLIS_MAX_M",
                      "DCP_PREFILL_WORKSPACE_MIB", "DCP_CKV_PREFETCH_DEPTH",
                      "DCP_CKV_GATHER_MAX_TOKENS",
                      "DCP_KV_CACHE_INTERLEAVE_SIZE",
                      "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS",
                      "B12X_PCIE_DMA", "F8_DMA", "PCIE_CALIBRATION",
                      "PCIE_DMA_MIN_BYTES",
                      "CLAMP_ROPE_TABLES", "VISION", "VISION_CHUNKS"),
        "defaults": {
            "MAX_MODEL_LEN": 524288, "MODEL_OUTPUT_LIMIT": 131072, "MTP_TOKENS": 3,
            "MAX_NUM_SEQS": 8, "MAX_NUM_BATCHED_TOKENS": 3072,
            "DCP_PREFILL_WORKSPACE_MIB": 1024,
            "DCP_CKV_PREFETCH_DEPTH": "auto",
            "DCP_CKV_GATHER_MAX_TOKENS": 140000,
            "DCP_KV_CACHE_INTERLEAVE_SIZE": 64,
            "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": -1,
            "B12X_PCIE_DMA": True,
            "F8_DMA": "0",
            "PCIE_CALIBRATION": "auto",
            "PCIE_DMA_MIN_BYTES": -1,
            "CLAMP_ROPE_TABLES": True,
            "GPU_MEMORY_UTILIZATION": 0.96, "GPU_BLOCKS_OVERRIDE": 0,
            "OFFLOAD_FRACTION": 0, "VISION": False,
            "KV_CACHE_DTYPE": "nvfp4_ds_mla",
            "KV_SCALE_MODE": "static-calibrated",
            "LOAD_FORMAT": "safetensors",
            "MTP_DRAFT_SAMPLE_METHOD": "greedy",
            "SERVED_MODEL_NAME": "GLM-5.2",
        },
        # %(...)s are substituted from the resolved knobs
        "serve_args": [
            "--decode-context-parallel-size", "%(DCP)s",
            "--dcp-comm-backend", "a2a",
            "--dcp-kv-cache-interleave-size", "%(DCP_KV_CACHE_INTERLEAVE_SIZE)s",
            "--attention-backend", "B12X_MLA_SPARSE",
            "--moe-backend", "b12x",
            "--load-format", "%(LOAD_FORMAT)s",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "glm47",
            "--reasoning-parser", "glm45",
            "--default-chat-template-kwargs", '{"reasoning_effort":"high"}',
            "--hf-overrides",
            '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSS'
            'FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}',
        ],
        "compilation_config": (
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'
            '[%(CUDAGRAPH_CAPTURE_SIZES)s],"custom_ops":["all"],'
            '"pass_config":{"fuse_allreduce_rms":true}}'),
        "notes": ("MLA attention with a sparse indexer, an EXL3 or NVFP4 checkpoint, "
                  "DCP-sharded KV, and the MTP78 speculative-draft apparatus. Every "
                  "default here has a measurement behind it; see README.md."),
    },
    "qwen36": {
        "label": "Qwen3.6 27B NVFP4 Vision — 1x RTX 5090 production profile",
        "tested": True,
        "env_block": "generic",
        "default_variant": "qwen36-nvfp4",
        # nvfp4_ds_mla is an MLA KV layout; it does not exist for this family.
        "kv_dtypes": ["auto", "fp8"],
        "spec_method": "mtp",
        "own_knobs": ("MULTIMODAL", "MM_MAX_PIXELS"),
        "defaults": {
            "TENSOR_PARALLEL_SIZE": 1,
            "MAX_MODEL_LEN": 196608,
            "MODEL_OUTPUT_LIMIT": 8192,
            "MTP_TOKENS": 0,
            "MAX_NUM_SEQS": 4,
            "MAX_NUM_BATCHED_TOKENS": 4096,
            "GPU_MEMORY_UTILIZATION": 0.90,
            "GPU_BLOCKS_OVERRIDE": 0,
            "OFFLOAD_FRACTION": 0,
            "KV_CACHE_DTYPE": "auto",
            "LOAD_FORMAT": "safetensors",
            "SERVED_MODEL_NAME": "Qwen3.6-27B",
            "MULTIMODAL": True,
            "MM_MAX_PIXELS": 8388608,
        },
        "serve_args": [
            "--load-format", "%(LOAD_FORMAT)s",
            "--mm-processor-kwargs", '{"max_pixels":%(MM_MAX_PIXELS)s}',
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ],
        "notes": (
            "Vision-enabled ModelOpt NVFP4 checkpoint qualified on one 32 GB RTX "
            "5090 at a 196,608-token request limit, 8,388,608-pixel image cap, "
            "and checkpoint-selected FP8 KV. The appliance path, compiled default, "
            "thinking-content normalization, structured output, tools, detailed "
            "vision, near-maximum retrieval and preserved multi-turn reasoning were "
            "live-qualified with the full 27B checkpoint. Native MTP is "
            "slower and reduces usable context on this image; it remains "
            "opt-in. It uses vLLM's current 'mtp' method; GG v20-r9 forces that "
            "path to eager mode around a FlashInfer frozen-query-shape crash. "
            "GLM-only MLA/DCP/EXL3 settings are removed rather than inherited "
            "from the base image."),
    },
    "custom": {
        "label": "Custom checkpoint — conservative generic vLLM profile",
        "tested": False,
        "env_block": "generic",
        "default_variant": "custom",
        "kv_dtypes": ["auto", "fp8"],
        "spec_method": "mtp",
        "own_knobs": ("MODEL_ID", "QUANTIZATION", "REASONING_PARSER",
                      "TOOL_CALL_PARSER", "MULTIMODAL"),
        "defaults": {
            "TENSOR_PARALLEL_SIZE": 1,
            "MAX_MODEL_LEN": 32768,
            "MODEL_OUTPUT_LIMIT": 8192,
            "MTP_TOKENS": 0,
            "MAX_NUM_SEQS": 4,
            "MAX_NUM_BATCHED_TOKENS": 4096,
            "GPU_MEMORY_UTILIZATION": 0.90,
            "GPU_BLOCKS_OVERRIDE": 0,
            "OFFLOAD_FRACTION": 0,
            "KV_CACHE_DTYPE": "auto",
            "LOAD_FORMAT": "safetensors",
            "SERVED_MODEL_NAME": "Local-model",
            "MODEL_ID": "",
            "QUANTIZATION": "",
            "REASONING_PARSER": "",
            "TOOL_CALL_PARSER": "",
            "MULTIMODAL": True,
        },
        "serve_args": [],
        "notes": (
            "Generic profile for another Hugging Face checkpoint. It deliberately "
            "does not guess architecture-specific parsers, quantization, speculative "
            "decoding, or GLM kernel settings."),
    },
}


def family(name=None) -> dict:
    return FAMILIES.get(name or "glm52", FAMILIES["glm52"])


# --------------------------------------------------------------------------
# model variants
# --------------------------------------------------------------------------
# `kv_scales_calibrated` is the ONLY thing that makes nvfp4 KV admissible: the
# nvfp4 MLA path needs per-checkpoint calibrated outer scales and silently
# degenerates at long context without them (see VALIDATIONS/kv-nvfp4).

VARIANTS = {
    "exl3-tr3": {
        "family": "glm52",
        "label": "EXL3-TR3 3.0bpw — balanced DCP2/MTP5 (default, measured)",
        "repo": "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw",
        "revision": "9297b9f1d53af5c67cffa01e30cc071a1ff7144b",
        "dirname": "GLM-5.2-EXL3-TR3-3.0bpw",
        "quantization": "exl3",
        "native_mtp_format": "exl3-tr3",
        # The pinned July 27 checkpoint serializes layer 78 as the same
        # rank-sliced EXL3/TR3 format as layers 3..77. A separate draft is no
        # longer needed; the old external arm remains selectable for A/Bs.
        "default_draft": "native",
        "defaults": {
            "DCP": "2",
            # GG v20-r9's dynamic-token KV path and exact indexer fold expose
            # 543,488 KV tokens on the qualified 96 GiB TP4/DCP2 shape at
            # GMU 0.955. The full binary 512 Ki-token request envelope passed
            # a 521,275-token five-depth retrieval with runtime headroom.
            "MAX_MODEL_LEN": 524288,
            "MAX_NUM_BATCHED_TOKENS": 3072,
            "MAX_NUM_SEQS": 8,
            # GG v20-r9's complete paired ABI stores dynamic outer scales per
            # token and FP8 RoPE in the 368-byte record. On AIBeast it matched
            # the BF16 reference at mean KLD 0.116770 in two independent runs,
            # then passed two exact 510,533-token five-depth retrievals and the
            # final 521,275-token full-envelope gate.
            "KV_SCALE_MODE": "dynamic-token",
            "MTP_TOKENS": 5,
            "MTP_DRAFT_SAMPLE_METHOD": "probabilistic",
            "DCP_PREFILL_WORKSPACE_MIB": 1024,
            "DCP_CKV_PREFETCH_DEPTH": "0",
            "DCP_CKV_GATHER_MAX_TOKENS": 140000,
            "DCP_KV_CACHE_INTERLEAVE_SIZE": "1",
            "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": 8192,
            # r11 safetensors at 0.957 exposed 542,208 logical KV tokens on its
            # cold qualification boot and completed an exact 522,360-token
            # five-depth request. The same image warm-started with its AOT
            # cache, exposed 553,472 tokens, and showed no corruption or the
            # historical low-throughput cache failure.
            "GPU_MEMORY_UTILIZATION": 0.957,
            "GPU_BLOCKS_OVERRIDE": 0,
            "LOAD_FORMAT": "safetensors",
            "OFFLOAD_FRACTION": 0.5,
            "PREFIX_CACHE_BACKEND": "lmcache",
            "F8_DMA": "0",
            "PCIE_CALIBRATION": "auto",
            "PCIE_DMA_MIN_BYTES": -1,
            "MAX_CUDAGRAPH_CAPTURE_SIZE": 64,
            "CUDAGRAPH_CAPTURE_SIZES":
                "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64",
            "VLLM_EXL3_TRELLIS_MAX_M": 64,
        },
        # Applied after the family block and before explicit TUNE_* overrides.
        # These are the published issue-33 route choices independently
        # reproduced on AIBeast with the Brandon target + EXL3 MTP78.
        "runtime_env": {
            "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE": "1",
            "VLLM_DCP_TOPK_OWNER_MERGE": "0",
            "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "0",
            "VLLM_EXL3_PREFILL_BLOCK_M": "64",
            # r11 interprets this as a row count, not the old boolean-like
            # enable flag. Its EXL3 backend default is 128; setting 1 cannot
            # cover the target parity window and fails during memory profiling.
            "VLLM_EXL3_PREFILL_CHUNK": "128",
            "VLLM_EXL3_TRELLIS_BLOCK_M": "8",
            "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": "1024",
            "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "16",
            "VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA": "1",
            "VLLM_PCIE_ONESHOT_SINGLE_CHANNEL": "0",
            "SPARKINFER_FUSED_INDEXER": "1",
            "SPARKINFER_INDEXER_DIRECT_K": "1",
            "SPARKINFER_INDEXER_STREAM_SCORER": "1",
            "SPARKINFER_MLA_PREFILL_STRATEGY": "auto",
            "SPARKINFER_MLA_SM120_NUM_SPLITS": "0",
            "SPARKINFER_MLA_SM120_PREFILL_MG": "1",
            "SPARKINFER_PAGED_INDEX_SUPERTILE_K": "32768",
            "SPARKINFER_W4A16_SMALL_M_DIRECT": "1",
            "B12X_PAGED_INDEX_SUPERTILE_K": "32768",
        },
        # v29 carries the checkpoint-calibrated GLM-5.2 MLA outer scales at
        # VLLM_NVFP4_MLA_SCALES_FILE. The scales are model-wide, not tied to
        # the target weight quantization.
        "kv_scales_calibrated": True,
        "download_gib": 309,
        "tested": True,
    },
    "exl3-tr3-max-context": {
        "family": "glm52",
        "label": "EXL3-TR3 3.0bpw — DCP4 maximum-context (measured)",
        "repo": "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw",
        "revision": "9297b9f1d53af5c67cffa01e30cc071a1ff7144b",
        "dirname": "GLM-5.2-EXL3-TR3-3.0bpw",
        "quantization": "exl3",
        "native_mtp_format": "exl3-tr3",
        "default_draft": "native",
        "defaults": {
            "DCP": "4",
            "MAX_MODEL_LEN": 524288,
            "MAX_NUM_BATCHED_TOKENS": 4096,
            "MAX_NUM_SEQS": 8,
            "MTP_TOKENS": 5,
            "MTP_DRAFT_SAMPLE_METHOD": "probabilistic",
            "DCP_PREFILL_WORKSPACE_MIB": 1024,
            "DCP_CKV_PREFETCH_DEPTH": "auto",
            "DCP_CKV_GATHER_MAX_TOKENS": 140000,
            "DCP_KV_CACHE_INTERLEAVE_SIZE": "64",
            "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": -1,
            "GPU_MEMORY_UTILIZATION": 0.98,
            "GPU_BLOCKS_OVERRIDE": 0,
            "OFFLOAD_FRACTION": 0.5,
            "F8_DMA": "0",
            "PCIE_CALIBRATION": "auto",
            "PCIE_DMA_MIN_BYTES": -1,
            "MAX_CUDAGRAPH_CAPTURE_SIZE": 64,
            "CUDAGRAPH_CAPTURE_SIZES":
                "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64",
            "VLLM_EXL3_TRELLIS_MAX_M": 64,
        },
        "kv_scales_calibrated": True,
        "download_gib": 309,
        "tested": True,
    },
    "nvfp4": {
        "family": "glm52",
        "label": "NVFP4 hybrid (EXPERIMENTAL in this template)",
        "repo": "lukealonso/GLM-5.2-NVFP4",
        "dirname": "GLM-5.2-NVFP4",
        "quantization": "modelopt_fp4",
        "kv_scales_calibrated": True,
        "download_gib": 400,
        "tested": False,
    },
    "madeby561-hybrid": {
        "family": "glm52",
        "label": "MadeBy561 MXFP8/NVFP4/NF3 hybrid (512K MTP3 profile)",
        "repo": "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid",
        # The release-bundle commits add audited profiles, KLD receipts and the
        # calibrated NVFP4 MLA scale sidecar. The 184 weight shards remain
        # byte-identical to immutable weight revision 68babde2.
        "revision": "66f3623dd8fefb5ca8046706912d5d31c8d196af",
        "dirname": "GLM-5.2-MXFP8-NVFP4-NF3-Hybrid",
        "quantization": "nvfp4_nf3_hybrid",
        "quantization_config": (
            '{"linear":{"weight":"mxfp8"},'
            '"shared_experts":{"weight":"mxfp8"},'
            '"ignore":["re:^model\\\\.layers\\\\.0\\\\.",'
            '"re:.*\\\\.self_attn\\\\.indexer\\\\.",'
            '"re:.*\\\\.mlp\\\\.gate$",'
            '"model.layers.78.eh_proj","lm_head"]}'
        ),
        # The checkpoint's native layer-78 experts are already serialized
        # NVFP4 (U8 weights + FP8/FP32 scales). Do not call this "bf16": only
        # some dense/attention tensors remain BF16 and eligible ones receive
        # the checkpoint's MXFP8 load-time overlay.
        "default_draft": "native",
        # v20's automatic pool sizing consumed the memory that the hybrid MTP
        # proposal needs on its first real request. Pinning exactly one 512K
        # request preserves runtime headroom. A 3072-token chunk with a 512 MiB
        # workspace passed three uncached 32K prefills plus C1/C2/C4/C8, but
        # OOMed immediately at 520K. Keep the v20 NF3 default chunk at 2048;
        # the smaller borrowed DCP workspace retains additional room for the
        # maximum-context runtime path.
        "defaults": {
            "MAX_NUM_BATCHED_TOKENS": 2048,
            "DCP_PREFILL_WORKSPACE_MIB": 512,
            "GPU_MEMORY_UTILIZATION": 0.98,
            "GPU_BLOCKS_OVERRIDE": 2048,
            "MTP_DRAFT_SAMPLE_METHOD": "probabilistic",
            "DCP_CKV_PREFETCH_DEPTH": "0",
            "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": 8192,
            "F8_DMA": "ring",
            "PCIE_CALIBRATION": "off",
            "PCIE_DMA_MIN_BYTES": 393216,
        },
        "kv_scales_calibrated": True,
        "download_gib": 341,
        "tested": True,
    },
    "qwen36-nvfp4": {
        "family": "qwen36",
        "label": "Qwen3.6-27B ModelOpt NVFP4",
        "repo": "nvidia/Qwen3.6-27B-NVFP4",
        "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        "dirname": "Qwen3.6-27B-NVFP4",
        "quantization": "modelopt",
        "kv_scales_calibrated": False,
        "download_gib": 21,
        "tested": True,
    },
    "custom": {
        "family": "custom",
        "label": "Custom Hugging Face checkpoint",
        "repo": "",
        "dirname": "custom",
        "quantization": "",
        "kv_scales_calibrated": False,
        "download_gib": 0,
        "tested": False,
    },
}

# The higher-fidelity mixed-K checkpoint needs DCP4 to fit its 315.9 GiB
# payload, native MTP3 and one full binary-512K request on four 96 GiB cards.
# Its exact r11 gate deliberately pins the KV pool: the auto pool can consume
# first-request workspace.  The final offload qualification also lowers the
# prefill chunk and the exact-fold workspace budget so LMCache's four CUDA
# transfer contexts do not consume the last all-reduce/output-conversion
# margin. vLLM #222 natively splits the reusable arena below the 2,048-row
# scheduler chunk when capacity is more important than peak throughput. The
# r17 shape-aware path runs one serial K3/K4 block-64 plan per slice: 1,024 and
# 1,536 rows both measured about 10-11% below the unsliced arm at 3K/32K/128K.
# Since 1,024 recovered about 665 MiB/rank while 1,536 recovered only 332 MiB
# without improving PP, the qualified profile keeps the 1,024-row trade.
VARIANTS["exl3-tr3-3.25bpw"] = {
    "family": "glm52",
    "label": "EXL3-TR3 mixed 3.25bpw — higher-fidelity 512K",
    "repo": "willfalco/GLM-5.2-EXL3-TR3-3.25bpw",
    "revision": "d7d79c2d14599dfce7a5d12b85f7ad73f40e623d",
    "dirname": "GLM-5.2-EXL3-TR3-3.25bpw",
    "quantization": "exl3",
    "native_mtp_format": "exl3-tr3",
    "default_draft": "native",
    "defaults": dict(VARIANTS["exl3-tr3"]["defaults"]),
    "runtime_env": dict(VARIANTS["exl3-tr3"]["runtime_env"]),
    "kv_scales_calibrated": True,
    "download_gib": 316,
    "tested": True,
}
VARIANTS["exl3-tr3-3.25bpw"]["defaults"].update({
    "DCP": "4",
    "MAX_MODEL_LEN": 524288,
    "MAX_NUM_BATCHED_TOKENS": 2048,
    "VLLM_EXL3_PREFILL_CAPACITY": 1024,
    "MAX_NUM_SEQS": 8,
    "MTP_TOKENS": 3,
    "DCP_CKV_PREFETCH_DEPTH": "0",
    "DCP_CKV_GATHER_MAX_TOKENS": 140000,
    "DCP_KV_CACHE_INTERLEAVE_SIZE": "64",
    "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": -1,
    "GPU_MEMORY_UTILIZATION": 0.957,
    # 2,048 * 64 * DCP4 = exactly 524,288 logical tokens.
    "GPU_BLOCKS_OVERRIDE": 2048,
    "OFFLOAD_FRACTION": 0.5,
    "PREFIX_CACHE_BACKEND": "lmcache",
    "PREFIX_CACHE_DISK_GB": 0,
    "PCIE_CALIBRATION": "off",
    "MAX_CUDAGRAPH_CAPTURE_SIZE": 32,
    "CUDAGRAPH_CAPTURE_SIZES": "4,8,12,16,20,24,28,32",
    "VLLM_EXL3_TRELLIS_MAX_M": 32,
})
VARIANTS["exl3-tr3-3.25bpw"]["runtime_env"].update({
    # Mixed-K loading replaces thousands of per-expert safetensor allocations
    # with tier-contiguous slabs. Expandable segments plus the patch's
    # post-layer cache release avoid an otherwise deterministic preparation
    # OOM on four 96 GB cards.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SAFETENSORS_FAST_GPU": "1",
    # DCP4 selected routes from the qualified mixed-K launch.
    "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE": "0",
    "VLLM_DCP_TOPK_OWNER_MERGE": "1",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    # At 3,072 tokens and the source-default 256 MiB budget, the first 128K
    # request could OOM after a successful boot.  A 2,048-token chunk plus the
    # exact streaming-carry path above 64 MiB passed two 128K requests and
    # 520,001/524,012-token boundary prefills with 125 GiB LMCache DRAM and a
    # bounded 512 GiB filesystem tier.
    "SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB": "64",
})

# MTP draft types -> the three env knobs the serve path actually consumes.
# `tr3-graft`   in-place surgery on layer 78 of the target (the ONLY draft with
#               long-context evidence: needle 6/6 fp8, armC 3/3 at 150/190/250K)
# `tr3-override` separate rank-sliced EXL3 draft dir (--speculative-config model)
# `nvfp4`       external NVFP4 draft; MUST carry its own quantization (rule 4)
DRAFTS = {
    "off":           {"label": "off (no speculative decode)", "mtp78_mode": "off"},
    "native":        {"label": "Native in-checkpoint draft", "mtp78_mode": "off"},
    # Kept as a backwards-compatible spelling only. Current EXL3 checkpoints
    # serialize routed layer-78 experts as Trellis, not BF16.
    "bf16":          {"label": "Native in-checkpoint draft (legacy “bf16” spelling)",
                      "mtp78_mode": "off"},
    "tr3-graft":     {"label": "EXL3 3bpw trellis, grafted into the target", "mtp78_mode": "graft"},
    "tr3-override":  {"label": "EXL3 3bpw trellis, separate draft dir", "mtp78_mode": "override"},
    "nvfp4":         {"label": "External NVFP4 draft dir", "mtp78_mode": "off",
                      "draft_quantization": "modelopt_fp4"},
}

# --------------------------------------------------------------------------
# knob registry
# --------------------------------------------------------------------------
# key        state-file / primary env name
# aliases    additional env names accepted for the env layer (legacy spellings)
# type       bool | int | float | choice | csv | str
# editable   False -> shown read-only in the UI (a locked, load-bearing value)
# scope      "engine"     restart of vLLM is enough
#            "checkpoint" needs on-disk preparation (graft/vision) before serve
#            "download"   needs a (large) weight download

KNOBS = [
    dict(key="MODEL_FAMILY", type="choice", default="glm52", choices=list(FAMILIES),
         group="Model", scope="download", label="Model family",
         rationale=(
             "Which model architecture this instance serves. The family decides the "
             "engine flags, which knobs exist, and which of the measured failure rules "
             "apply — MLA-only features like DCP and the EXL3 trellis are refused "
             "outside GLM rather than silently ignored. 'glm52' is the family every "
             "production number in this README was measured on. 'qwen36' is the "
             "one-GPU NVFP4 development profile; 'custom' accepts another Hugging "
             "Face checkpoint without guessing architecture-specific flags. Changing "
             "family triggers a fresh download.")),

    dict(key="TENSOR_PARALLEL_SIZE", type="int", default=4, min=1, max=16,
         group="Parallelism", scope="engine", label="Tensor parallel size",
         rationale=(
             "How many GPUs the model's weights are sharded across. DEFAULTS TO THE "
             "NUMBER OF GPUs THIS CONTAINER CAN ACTUALLY SEE — not to 4, which is what "
             "the serve line hard-coded and what made the image unstartable on any "
             "other shape of host. Detection intersects nvidia-smi with "
             "NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES (including CUDA's rule "
             "that enumeration stops at the first invalid entry), and a provider's "
             "advertised count is treated as corroboration, not truth. Setting it "
             "explicitly overrides detection; it can never exceed the visible count.")),

    dict(key="MODEL_VARIANT", type="choice", default="exl3-tr3",
         choices=list(VARIANTS), group="Model", scope="download",
         label="Model variant",
         rationale=(
             "Which checkpoint is served. EXL3-TR3 3.0bpw is what this template "
             "measures and ships: ~77 GiB/rank, which is what buys the 512K pool on "
             "96 GB cards, at AIME/HMMT/GPQA parity with BF16. The NVFP4 hybrid is "
             "~6% faster to decode and is the only variant with calibrated MLA outer "
             "scales (so the only one where nvfp4 KV is safe), but it is BIGGER on "
             "disk, gives up ~30% of the KV pool, and this template's serve args for "
             "it are derived, not measured. Switching variants triggers a fresh "
             "multi-hundred-GB download.")),

    dict(key="MODEL_ID", families=("custom",), type="str", default="",
         group="Model", scope="download", label="Hugging Face model ID",
         rationale=(
             "Repository to download for the custom profile, for example "
             "'Qwen/Qwen3.5-0.8B'. Required when MODEL_FAMILY=custom.")),

    dict(key="QUANTIZATION", families=("custom",), type="str", default="",
         group="Model", scope="engine", label="vLLM quantization method",
         rationale=(
             "Optional --quantization value for a custom checkpoint. Leave empty "
             "when the checkpoint auto-detects its quantization.")),

    dict(key="REASONING_PARSER", families=("custom",), type="str", default="",
         group="Serving", scope="engine", label="Reasoning parser",
         rationale=(
             "Optional vLLM reasoning parser for the custom checkpoint. It is not "
             "guessed because a wrong parser can hide or corrupt response content.")),

    dict(key="TOOL_CALL_PARSER", families=("custom",), type="str", default="",
         group="Serving", scope="engine", label="Tool-call parser",
         rationale=(
             "Optional vLLM tool-call parser. Supplying one also enables automatic "
             "tool choice; leave empty for models without a verified parser.")),

    dict(key="MULTIMODAL", families=("qwen36", "custom"), type="bool", default=True,
         group="Multimodal", scope="engine", label="Load native multimodal encoder",
         rationale=(
             "When off, passes --language-model-only to save VRAM. The qualified "
             "Qwen RTX 5090 profile defaults on and uses the checkpoint's native "
             "vision encoder. GLM's grafted vision path remains controlled by "
             "VISION.")),

    dict(key="MM_MAX_PIXELS", families=("qwen36",), type="int",
         default=8388608, min=262144, max=16777216,
         group="Multimodal", scope="engine", label="Max image pixels",
         rationale=(
             "Caps the pixels sent to Qwen's native vision tower. The checkpoint "
             "default is 16,777,216, which processes a 5120x2880 screenshot at all "
             "14.7 MP and OOMed a 32 GB RTX 5090 even after lowering GMU to 0.93. "
             "8,388,608 is approximately a 4K working image and passed two "
             "deterministic 5120x2880 dashboard gates, including one after a "
             "192K prefill. Raising it requires additional transient VRAM and a "
             "fresh detailed-vision test.")),

    dict(key="MAX_MODEL_LEN", type="int", default=524288, min=4096, max=1048576,
         group="Model", scope="engine", label="Max context length",
         rationale=(
             "Longest single request the engine will accept, and the length vLLM "
             "must be able to fit in KV before it will start at all. Lowering it is "
             "the standard remedy when boot fails with 'To serve at least one request "
             "with the model's max seq len ... larger than the available KV cache "
             "memory' — it is a startup gate, not a throughput knob. Vision has its "
             "own memory profile and must re-pass the startup/needle gate.")),

    dict(key="CLAMP_ROPE_TABLES", families=("glm52",), type="bool", default=True,
         group="Memory", scope="engine", label="Clamp RoPE tables to served context",
         rationale=(
             "Overrides the checkpoint's max_position_embeddings with MAX_MODEL_LEN. "
             "The MadeBy561 checkpoint advertises 1,048,576 positions while the "
             "turnkey profile serves 524,288; the known-good v19 launcher clamps it "
             "to avoid allocating unused target and draft RoPE tables. Disable "
             "only for a controlled A/B against the checkpoint-native allocation.")),

    dict(key="GPU_BLOCKS_OVERRIDE", type="int", default=0, min=0, max=65536,
         group="Model", scope="engine", label="KV pool pin (--num-gpu-blocks-override)",
         rationale=(
             "0 lets vLLM profile all available KV and is the GLM release default "
             "(roughly 1.0-1.1M tokens on the v29 4x96 GB profile). A positive value "
             "pins a reproducible smaller pool. On the measured MLA stack logical "
             "capacity is blocks x 64 x DCP, so 2048 blocks means 524,288 tokens "
             "at TP4/DCP4; do not multiply the requested block count by DCP a "
             "second time. A pin must still fit MAX_MODEL_LEN.")),

    dict(key="KV_CACHE_DTYPE", type="choice", default="fp8",
         choices=["auto", "fp8", "nvfp4_ds_mla"], group="Model", scope="engine",
         label="KV cache dtype",
         rationale=(
             "nvfp4 KV costs substantially fewer bytes/token than fp8 but requires "
             "model-calibrated MLA outer scales. v29 ships those scales for GLM-5.2, "
             "so the EXL3 profile may use it; variants without declared calibration "
             "are refused. Earlier uncalibrated runs silently degenerated while short "
             "checks passed, so release qualification still requires cold long-context "
             "retrieval.")),

    dict(key="KV_SCALE_MODE", families=("glm52",), type="choice",
         default="static-calibrated",
         choices=["static-calibrated", "dynamic-token"],
         group="Model", scope="engine", label="NVFP4 MLA scale mode",
         rationale=(
             "static-calibrated uses the pinned GLM-5.2 per-layer scale artifact "
             "and the established BF16-RoPE 432-byte record. dynamic-token opts "
             "into GG v20-r9's paired 368-byte FP8-RoPE record and stores one "
             "outer scale per token. The flagship EXL3 profile uses dynamic "
             "mode after two identical KLD runs, two exact 510,533-token "
             "retrievals, and a 521,275-token full-envelope pass on GG v20-r9. "
             "Other variants retain "
             "static-calibrated until independently qualified.")),

    dict(key="MTP_DRAFT", families=("glm52",), type="choice", default="native", choices=list(DRAFTS),
         group="Speculative decoding", scope="checkpoint", label="MTP draft type",
         aliases=[],
         rationale=(
             "Which draft model proposes the speculative tokens. The pinned Brandon "
             "checkpoint now stores layer-78 routed experts natively as EXL3/TR3; "
             "attention, shared experts, eh_proj and norms remain BF16. Native is "
             "therefore the no-mutation default. Earlier Brandon revisions carried "
             "a 19.3 GB BF16 layer 78; the graft modes were built for those revisions "
             "and remain available only for controlled compatibility experiments. "
             "'tr3-override' keeps the target byte-identical and measured slightly "
             "better on acceptance on the older checkpoint; v29 supports its "
             "separately rank-sliced draft graphs. MadeBy561's native path instead "
             "uses serialized NVFP4 experts plus an MXFP8 dense overlay. 'nvfp4' is "
             "a separate Luke draft dir that avoids "
             "the rank-sliced path (and must declare its own quantization); it "
             "regressed the measured MadeBy561 native path and is not interchangeable "
             "with it. 'bf16' remains only as a legacy spelling for native. 'off' "
             "disables speculation (~30% slower decode).")),

    dict(key="MTP_TOKENS", type="int", default=3, min=0, max=8,
         group="Speculative decoding", scope="engine", label="Speculation depth",
         rationale=(
             "How many tokens the draft proposes per step. 3 is the shipped default. "
             "The optimum moves with the draft's cost: with the older Brandon "
             "revision's 19.3 GB BF16 draft, "
             "MTP-5 lost ~22%; with the 3.7 GB trellis draft it WINS (GSM8K n=30: "
             "MTP-2 42.9, MTP-3 51.5, MTP-5 53.4 tok/s). Raising it also raises the "
             "decode query width m = MAX_NUM_SEQS x (1 + MTP_TOKENS), which has to "
             "stay inside the capture/trellis window. 0 disables speculation.")),

    dict(key="MTP_DRAFT_SAMPLE_METHOD", families=("glm52",), type="choice",
         default="greedy", choices=["greedy", "probabilistic"],
         group="Speculative decoding", scope="engine",
         label="Draft sampling method",
         rationale=(
             "How the MTP draft proposes candidates. The v29 EXL3 release and its "
             "long-context qualification use greedy drafting, which is the turnkey "
             "default. Probabilistic drafting remains available for controlled "
             "quality/acceptance experiments.")),

    dict(key="MTP_REJECTION_SAMPLE_METHOD", families=("glm52",), type="choice",
         default="standard", choices=["standard", "block"],
         group="Speculative decoding", scope="engine",
         label="Draft verification method",
         rationale=(
             "How target probabilities accept or reject speculative draft tokens. "
             "Standard is vLLM's established distribution-preserving default and the "
             "qualified turnkey control. Block verification passed the same exact "
             "~517K five-depth retrieval gate, but a matched two-run temperature-one "
             "GLM A/B was workload-sensitive: C1 improved 14.4%, C2 fell 7.5%, C4 "
             "fell 10.9%, and C8 improved 1.5%. Keep it experimental because it did "
             "not produce a consistent end-to-end gain across the supported "
             "concurrency envelope.")),

    dict(key="DRAFT_MODEL", families=("glm52",), type="str", default="", group="Speculative decoding",
         scope="checkpoint", label="External draft dir (advanced)",
         rationale=(
             "Path to an external draft checkpoint for --speculative-config. Leave "
             "empty unless you are running a draft this template did not build — "
             "MTP draft type already fills this in for the modes that need it. An "
             "external draft is how you get a non-EXL3 draft without the rank-sliced "
             "capture bug.")),

    dict(key="DRAFT_QUANTIZATION", families=("glm52",), type="str", default="", group="Speculative decoding",
         scope="engine", label="Draft quantization method",
         rationale=(
             "A draft INHERITS the target's --quantization unless its speculative "
             "config carries its own. So an NVFP4 draft against an EXL3 target is "
             "loaded through the EXL3 path and dies in _apply_rank_sliced with the "
             "same 'eager parity path entered during CUDA graph capture (m=3)' error "
             "as the v20 rank-sliced bug — identical symptom, completely different "
             "cause. This is a config trap, not a bug. vLLM's name for the NVFP4 "
             "modelopt draft is 'modelopt_fp4'. Left empty, the draft type sets it.")),

    dict(key="DCP", families=("glm52",), type="choice", default="4",
         choices=["1", "2", "4", "8"],
         group="Parallelism", scope="engine", label="Decode context parallel",
         rationale=(
             "Shards the KV cache across GPUs at decode (--decode-context-parallel-size). "
             "It defaults to the tensor-parallel size, i.e. every rank holds a slice "
             "of the KV, which is what makes a 512K pool fit at all. Lowering it "
             "replicates KV instead, roughly dividing the usable context by the same "
             "factor, and is only interesting when debugging a suspected DCP/a2a "
             "issue. Must divide the tensor-parallel size.")),

    dict(key="MAX_NUM_SEQS", type="int", default=8, min=1, max=256,
         group="Concurrency", scope="engine", label="Max concurrent sequences",
         rationale=(
             "Concurrency ceiling. It defaults to 8 and not 32 because decode runs "
             "m = MAX_NUM_SEQS x (1 + MTP_TOKENS) query tokens per step, and 8 x 4 = 32 "
             "exactly fills the default CUDA-graph capture and EXL3 trellis windows. "
             "Past that the engine does NOT error — decode silently leaves the captured "
             "trellis fast path once enough streams are concurrent and loses throughput "
             "with nothing in the log to say so. Raise it together with the capture "
             "window knobs, then measure.")),

    dict(key="MAX_NUM_BATCHED_TOKENS", type="int", default=3072, min=512, max=65536,
         group="Concurrency", scope="engine", label="Max batched tokens",
         rationale=(
             "Chunked-prefill chunk size: the upper bound on tokens scheduled in one "
             "engine step. Bigger chunks prefill long prompts faster and cost KV "
             "headroom and decode latency (a decode step queued behind a big prefill "
             "chunk waits for it). 3072 is the shipped balance for a 512K-context "
             "single-stream workload.")),

    dict(key="VLLM_EXL3_PREFILL_CAPACITY", families=("glm52",), type="int",
         default=3072, min=1, max=65536,
         group="Memory", scope="engine", label="EXL3 prefill arena rows",
         rationale=(
             "Maximum rows allocated for one EXL3 prefill dispatch. It may be smaller "
             "than MAX_NUM_BATCHED_TOKENS: vLLM then slices a scheduler chunk through "
             "the same arena without dropping its short tail. On the retained 4x RTX "
             "PRO 6000 Blackwell host, 3,072/1,536/1,024/512 rows used "
             "759.8/399.7/279.7/159.7 MiB per target rank. In the matched MTP3 gate, "
             "1,024 versus 2,048 returned 73,472 logical KV tokens with a 0.29% "
             "prefill-throughput delta. Capacity must not exceed "
             "MAX_NUM_BATCHED_TOKENS. Lower values trade extra launches for memory "
             "headroom and require a long-prefill correctness and performance A/B.")),

    dict(key="DCP_PREFILL_WORKSPACE_MIB", families=("glm52",), type="int",
         default=1024, min=256, max=16384,
         group="Memory", scope="engine", label="DCP full-CKV workspace (MiB/GPU)",
         rationale=(
             "Scratch space per GPU for the fast transient full-CKV prefill path. "
             "In the current v29 runtime, 1,024 MiB covers 16,384 logical context "
             "tokens; longer prompts fall back to the lower-memory borrowed-workspace "
             "path. Raising this may improve long-prefill speed, especially across "
             "weak PCIe topologies, but every added MiB comes directly out of the KV "
             "pool. Keep enough KV for one MAX_MODEL_LEN request and confirm with the "
             "long-context gate. This is an empirical A/B control, not a promise that "
             "the largest value is fastest.")),

    dict(key="DCP_CKV_PREFETCH_DEPTH", families=("glm52",), type="choice",
         default="auto", choices=["auto", "0", "1"],
         group="Performance", scope="engine", label="DCP CKV prefetch overlap",
         rationale=(
             "auto asks the base image's PCIe calibrator whether overlapping the "
             "next CKV gather helps on this exact GPU/NUMA topology. A value of 1 "
             "can win on a single-root NODE layout and lose badly when one rank "
             "crosses CPU sockets; 0 disables the overlap without disabling the "
             "full-CKV path. The calibrated decision is cached on the model volume.")),

    dict(key="DCP_CKV_GATHER_MAX_TOKENS", families=("glm52",), type="int",
         default=140000, min=512, max=1048576,
         group="Performance", scope="engine", label="Full-CKV gather ceiling",
         rationale=(
             "Largest logical context covered by the transient full-CKV gather path. "
             "The old 16,384 ceiling forced ordinary 64K/128K prefills onto a slower "
             "distributed path. The public issue-33 study measured +15.8%/+12.6% "
             "at 64K/128K under DCP2 and +49.1%/+50.0% under DCP4 when raised to "
             "140,000. This appliance independently passed a five-depth 521K gate "
             "with the 140K DCP2 shape. Larger values consume transient workspace "
             "and must preserve enough auto-profiled KV for MAX_MODEL_LEN.")),

    dict(key="DCP_KV_CACHE_INTERLEAVE_SIZE", families=("glm52",), type="choice",
         default="64", choices=["1", "2", "4", "8", "16", "32", "64"],
         group="Performance", scope="engine", label="DCP KV interleave",
         rationale=(
             "Interleave granularity inside each 64-token KV block. The historical "
             "turnkey value is 64; the public issue-33 DCP2 production study selected "
             "1 after finding 16 mixed. Treat changes as topology-specific and compare "
             "prefill, C1-C8 decode, KV capacity and maximum-context retrieval.")),

    dict(key="DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS", families=("glm52",), type="int",
         default=-1, min=-1, max=1048576,
         group="Performance", scope="engine",
         label="DCP query-split minimum context",
         rationale=(
             "-1 accepts the topology calibrator's result. A non-negative value "
             "overrides the context-length crossover after calibration. The measured "
             "MadeBy561 profile uses 8,192: shorter prompts avoid split overhead while "
             "long prefills use the DCP query-split path.")),

    dict(key="B12X_PCIE_DMA", families=("glm52",), type="bool", default=True,
         group="Performance", scope="engine", label="B12X PCIe DMA transport",
         rationale=(
             "Enables SparkInfer/B12X's PCIe DMA path for collective payloads. "
             "The v20 image calibrates the lossless BF16 crossover before loading "
             "the model and retains NCCL below that size. Disable only for an A/B "
             "control or when diagnosing a platform-specific transport failure.")),

    dict(key="F8_DMA", families=("glm52",), type="choice", default="0",
         choices=["0", "ag", "ring", "a2a", "i8", "i8_ring", "i8_a2a",
                  "mx", "mx_ring", "mx_a2a"],
         group="Performance", scope="engine", label="Compressed DMA wire mode",
         rationale=(
             "0 is the lossless release default and is eligible for automatic PCIe "
             "crossover calibration. ag/ring/a2a and the i8/mx variants compress "
             "collective traffic and can materially improve a constrained PCIe "
             "topology, but they are explicit quality/performance experiments. "
             "Any nonzero mode must re-pass maximum-context needles and degeneration "
             "checks before it should be used as a local profile default.")),

    dict(key="PCIE_CALIBRATION", families=("glm52",), type="choice",
         default="auto", choices=["auto", "off", "force"],
         group="Performance", scope="engine", label="Topology calibration",
         rationale=(
             "auto reuses a cached v20 PCIe microbenchmark for this GPU topology; "
             "force measures again, and off uses conservative fixed fallbacks. It "
             "selects the lossless DMA size crossover, query-split crossover, and "
             "whether CKV prefetch overlap is beneficial before model weights load.")),

    dict(key="PCIE_DMA_MIN_BYTES", families=("glm52",), type="int",
         default=-1, min=-1, max=67108864,
         group="Performance", scope="engine", label="PCIe DMA crossover (bytes)",
         rationale=(
             "-1 accepts the topology calibrator's result. A non-negative value "
             "overrides its payload-size crossover after calibration. The measured "
             "MadeBy561 ring profile uses 393,216 bytes; change it only with an "
             "isolated throughput and maximum-context correctness A/B.")),

    dict(key="MAX_CUDAGRAPH_CAPTURE_SIZE", type="int", default=32, min=1, max=512,
         group="Concurrency", scope="engine", label="Max CUDA-graph capture size",
         rationale=(
             "Largest batch width that gets a captured CUDA graph. Decode above this "
             "runs eager: correct, but slower, and on EXL3 it also leaves the trellis "
             "fast path. Must be >= MAX_NUM_SEQS x (1 + MTP_TOKENS) for the concurrency "
             "you actually intend to serve. More captured sizes cost capture time at "
             "boot and some VRAM.")),

    dict(key="CUDAGRAPH_CAPTURE_SIZES", type="csv", default="4,8,12,16,20,24,28,32",
         group="Concurrency", scope="engine", label="CUDA-graph capture sizes",
         rationale=(
             "The exact batch widths captured. Every width you actually decode at "
             "should be in this list, and the largest entry should match "
             "MAX_CUDAGRAPH_CAPTURE_SIZE. Note the smallest entry must stay inside the "
             "EXL3 trellis window [4, 32]: a capture below 4 falls into the eager "
             "parity path and the engine dies during capture.")),

    dict(key="VLLM_EXL3_TRELLIS_MAX_M", families=("glm52",), type="int", default=32, min=4, max=512,
         aliases=["TUNE_VLLM_EXL3_TRELLIS_MAX_M"],
         group="Concurrency", scope="engine", label="EXL3 trellis window max (m)",
         rationale=(
             "Upper edge of the EXL3 trellis kernel window. Decode query widths above "
             "it fall off the trellis fast path. Raising concurrency means raising this "
             "together with MAX_CUDAGRAPH_CAPTURE_SIZE and CUDAGRAPH_CAPTURE_SIZES — "
             "raising one alone just moves which of the two ceilings you hit.")),

    dict(key="GPU_MEMORY_UTILIZATION", type="float", default=0.90, min=0.50, max=0.99,
         group="Memory", scope="engine", label="GPU memory utilization",
         rationale=(
             "Fraction of each card vLLM may claim. Higher means more KV; too high and "
             "the memory profile OOMs during capture or KV validation fails outright. "
             "The GLM v29 profile uses 0.96 with calibrated NVFP4 KV and an auto-sized "
             "pool; other families provide their own conservative default. "
             "Memory-profile draws vary by a few hundred MiB run to run, so a value "
             "that boots 2 times in 3 is not a value that boots.")),

    dict(key="KV_CACHE_MEMORY_BYTES", type="int", default=0, min=0,
         max=1000000000000,
         group="Memory", scope="engine", label="Fixed KV cache bytes (0 = auto)",
         rationale=(
             "Advanced escape hatch for separating KV capacity from model/runtime "
             "headroom. 0 lets vLLM infer KV from GPU_MEMORY_UTILIZATION. A positive "
             "per-GPU byte count emits --kv-cache-memory-bytes; vLLM then ignores "
             "GPU_MEMORY_UTILIZATION for KV sizing. This is useful when an eager or "
             "speculative path needs transient workspace that an auto-filled KV pool "
             "would consume. Copy the reproducible value printed by vLLM's startup "
             "memory profiler, then leave a measured safety margin.")),

    dict(key="LOAD_FORMAT", type="choice", default="safetensors",
         choices=["safetensors", "instanttensor"],
         group="Memory", scope="engine", label="Weight loader",
         rationale=(
             "safetensors is the r11 production default. The immutable r11 image "
             "loaded all 81 target shards in 91.36s from a warm local store, passed "
             "the OpenAI feature suite and an exact 522,360-token five-depth "
             "retrieval, then reused its on-disk AOT cache without corruption or a "
             "throughput collapse. InstantTensor remains an explicit experiment: it "
             "can load faster when it works, but prior qualification found "
             "intermittent startup stalls and memory-shape failures. Loader changes "
             "remain a cold-boot, performance and near-maximum-correctness boundary.")),

    dict(key="MODEL_OUTPUT_LIMIT", type="int", default=131072, min=256, max=262144,
         group="Serving", scope="engine", label="Client output limit",
         rationale=(
             "Maximum generation length advertised by the landing-page client "
             "snippets and used by quick chat. This does not enlarge the model's "
             "context window; prompt plus output must still fit MAX_MODEL_LEN.")),

    dict(key="OFFLOAD_FRACTION", type="float", default=0.0, min=0.0, max=0.95,
         group="Memory", scope="engine", label="DRAM KV offload fraction",
         rationale=(
             "Fraction of the instance's RAM allocation used as an L2 CPU prefix "
             "cache after GPU eviction; it does not enlarge active GPU KV capacity "
             "(0 disables). Current native vLLM "
             "interprets cpu_bytes_to_use as one aggregate budget and derives each TP "
             "worker's physical slice from it; the appliance must not divide the "
             "connector value by TP a second time. Sized from "
             "min(cgroup limit, MemTotal), so a partial rental does not oversize it. "
             "It is a pure cache tier: kv_load_failure_policy="
             "recompute means a block that cannot be fetched back is recomputed rather "
             "than failing the request. Costs host RAM and nothing else; the win is on "
             "prefix-cache hits across requests. Live TP4 validation at 0.5 restored "
             "an evicted 133,504-token prefix in 0.69s instead of 52.47s recomputation "
             "and retained about 51 GiB available on a 251 GiB host. Larger fractions "
             "need equivalent host-headroom validation.")),

    dict(key="PREFIX_CACHE_BACKEND", type="choice", default="native",
         choices=["native", "lmcache"], group="Memory", scope="engine",
         label="External prefix-cache backend",
         rationale=(
             "native uses vLLM's in-process OffloadingConnector. lmcache uses GG "
             "v20-r11's DCP-aware LMCache MP "
             "connector. Both treat OFFLOAD_FRACTION as aggregate host-DRAM L1 "
             "capacity and neither enlarges active GPU KV. LMCache adds independent "
             "health/metrics, durable L2 support, prefetch and restart-safe cache "
             "management, but also adds a supervised process and connector boundary. "
             "The r11 GLM profile promotes LMCache after a 125 GiB four-worker DRAM "
             "gate, complete feature and 522,360-token retrieval suites, and a "
             "restart-persistent bounded NVMe test. Native remains the rollback "
             "control.")),

    dict(key="PREFIX_CACHE_DISK_GB", type="int", default=0, min=0, max=8192,
         group="Memory", scope="engine", label="LMCache NVMe limit (GiB)",
         rationale=(
             "0 keeps the external prefix tier in DRAM only. A positive value enables "
             "LMCache's native filesystem L2 with this hard capacity limit under the "
             "persistent model root. NVMe is slower than DRAM but can retain many "
             "large agentic prefixes without recomputing them. It stores derived "
             "prompt KV and therefore can contain sensitive session material: use "
             "local encrypted storage where possible, enable secure termination, and "
             "treat erase as best effort because SSD wear levelling and provider "
             "snapshots are outside the appliance's control.")),

    dict(key="VISION", families=("glm52",), type="bool", default=False, group="Multimodal", scope="checkpoint",
         label="Vision (image input) — EXPERIMENTAL on EXL3",
         rationale=(
             "Bolts the MoonViT-3d tower + PatchMerger projector (~890 MB, BF16, no "
             "text weight touched) onto the text backbone. QUALIFICATION IS "
             "COMPOSITION-SPECIFIC: our EXL3/TR3 composition returned 17/18 exact "
             "details from a 5120x2880 screenshot but failed the 32K text gate; the "
             "tested MadeBy561 composition passed the 32K text gate but extracted "
             "only 1-2/18 dashboard details. Other published Baseten-tower merges "
             "report successful mixed image/retrieval tests, so this is not a claim "
             "that the tower itself is broken. The current graft adds ~1.99 GiB/GPU. "
             "Keep it opt-in until this exact checkpoint passes both detailed-image "
             "and long-context gates.")),

    dict(key="VISION_CHUNKS", families=("glm52",), type="int", default=8, min=1, max=64,
         group="Multimodal", scope="engine", label="Max vision chunks per request",
         rationale=(
             "Caps media items per request (--limit-mm-per-prompt) so a screenshot "
             "burst cannot blow the memory profile. Each image costs ~151 tokens and "
             "images above ~2560 px hit the same ~4250-token patch cap anyway.")),

    dict(key="SERVED_MODEL_NAME", type="str", default="GLM-5.2", group="Serving",
         scope="engine", label="Served model name(s)",
         rationale=(
             "Whitespace-separated aliases the endpoint answers to. Add the name your "
             "existing clients already use (e.g. 'GLM-5.2 local-primary') and they need "
             "no reconfiguration to point at this instance.")),

]

KNOB_BY_KEY = {k["key"]: k for k in KNOBS}
EDITABLE = [k for k in KNOBS if k.get("editable", True)]


# --------------------------------------------------------------------------
# coercion
# --------------------------------------------------------------------------

class ConfigError(ValueError):
    pass


def coerce(knob: dict, raw):
    """Turn a JSON/env/form value into the knob's declared type, or raise."""
    key, typ = knob["key"], knob["type"]
    if raw is None:
        return knob["default"]
    if typ == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        raise ConfigError(f"{key}: expected a boolean, got {raw!r}")
    if typ == "int":
        try:
            v = int(str(raw).strip())
        except (TypeError, ValueError):
            raise ConfigError(f"{key}: expected an integer, got {raw!r}")
        if "min" in knob and v < knob["min"]:
            raise ConfigError(f"{key}: {v} is below the minimum {knob['min']}")
        if "max" in knob and v > knob["max"]:
            raise ConfigError(f"{key}: {v} is above the maximum {knob['max']}")
        return v
    if typ == "float":
        try:
            v = float(str(raw).strip())
        except (TypeError, ValueError):
            raise ConfigError(f"{key}: expected a number, got {raw!r}")
        if "min" in knob and v < knob["min"]:
            raise ConfigError(f"{key}: {v} is below the minimum {knob['min']}")
        if "max" in knob and v > knob["max"]:
            raise ConfigError(f"{key}: {v} is above the maximum {knob['max']}")
        return v
    if typ == "choice":
        v = str(raw).strip()
        if v not in knob["choices"]:
            raise ConfigError(f"{key}: {v!r} is not one of {', '.join(knob['choices'])}")
        return v
    if typ == "csv":
        v = re.sub(r"\s+", "", str(raw))
        if not v:
            raise ConfigError(f"{key}: must not be empty")
        if not re.fullmatch(r"\d+(,\d+)*", v):
            raise ConfigError(f"{key}: expected a comma-separated list of integers, got {raw!r}")
        return v
    v = str(raw).strip()
    if re.search(r"[\r\n\x00]", v):
        raise ConfigError(f"{key}: control characters are not allowed")
    return v


def to_text(knob: dict, value) -> str:
    if knob["type"] == "bool":
        return "1" if value else "0"
    if knob["type"] == "float":
        return f"{value:g}"
    return str(value)


# --------------------------------------------------------------------------
# the three layers
# --------------------------------------------------------------------------

def utcnow_iso() -> str:
    """Timezone-aware UTC, formatted with a trailing Z. (datetime.utcnow() is
    deprecated from 3.12 and prints a warning into the vast console on every
    call.)"""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcnow_stamp() -> str:
    """Sortable UTC stamp used for failure directory names."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {} if default is None else default


def write_json_atomic(path, obj, mode=0o600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def env_layer(env=None, invalid=None) -> dict:
    """Knob values present in the (startup) environment, incl. legacy spellings.

    A malformed value falls back to the default, but it must not do so
    silently: pass `invalid` (a list) to collect one message per rejected
    value so the snapshot/boot log can say the knob was ignored."""
    env = os.environ if env is None else env
    out = {}
    for knob in KNOBS:
        for name in [knob["key"]] + list(knob.get("aliases", [])):
            if env.get(name, "") != "":
                try:
                    out[knob["key"]] = coerce(knob, env[name])
                except ConfigError as e:
                    # a bad env value falls back to the default — loudly
                    if invalid is not None:
                        invalid.append(str(e))
                break
    # legacy draft spellings: MTP78_MODE / MTP78_TRELLIS / DRAFT_MODEL predate
    # the MTP_DRAFT knob and are still documented in the README.
    if "MTP_DRAFT" not in out:
        legacy = None
        mode = env.get("MTP78_MODE", "")
        if mode == "graft":
            legacy = "tr3-graft"
        elif mode == "override":
            legacy = "tr3-override"
        elif mode == "off":
            legacy = "native"
        # Order matters and matches the old entrypoint: MTP78_TRELLIS=0 was
        # applied AFTER MTP78_MODE and won over it.
        if env.get("MTP78_TRELLIS", "") == "0":
            legacy = "native"
        if env.get("DRAFT_MODEL", ""):
            legacy = "nvfp4" if env.get("DRAFT_QUANTIZATION", "") == "modelopt_fp4" else legacy
        if legacy:
            out["MTP_DRAFT"] = legacy
    # GLM_GPU_COUNT is exported by gpu_detect.py at boot; not a KNOB, but
    # detected_gpu_count() needs it from the startup snapshot.
    gpu_count = env.get("GLM_GPU_COUNT", "")
    if str(gpu_count).strip():
        out["GLM_GPU_COUNT"] = gpu_count
    return out


def snapshot_startup_env(env=None) -> dict:
    """Freeze the env layer for the life of the container (called once by the
    entrypoint). Everything after that reads the snapshot, so the layering can
    never be perturbed by something the entrypoint exports later."""
    invalid = []
    layer = env_layer(env, invalid=invalid)
    doc = dict(layer)
    if invalid:
        # Persist the rejections alongside the frozen layer so resolve() can
        # report them on every later read; a template typo must not vanish.
        doc["__invalid_env__"] = invalid
    write_json_atomic(p_startup_env(), doc, mode=0o644)
    return layer


def load_startup_env() -> dict:
    snap = read_json(p_startup_env())
    return snap if isinstance(snap, dict) else {}


def load_state_file() -> dict:
    """User state file. Returns {} when absent; raises on unusable content so a
    caller can report it instead of silently serving defaults."""
    path = p_state()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ConfigError("state file is not a JSON object")
    values = doc.get("values", doc)
    if not isinstance(values, dict):
        raise ConfigError("state file 'values' is not a JSON object")
    check_forbidden(values)
    return values


# --------------------------------------------------------------------------
# termination switches — startup environment ONLY, and one-way at runtime
# --------------------------------------------------------------------------
# These are not knobs. A knob is something the landing page may change; these
# two decide whether the landing page may destroy the instance at all, so the
# landing page must not be able to grant itself either of them.
#
#   TERMINATE_ENABLED  kill switch      — is the terminate control available?
#   TERMINATE_LOCKED   anti-kill switch — hard lock; termination is refused
#                                         regardless of TERMINATE_ENABLED
#
# Three structural properties, in the order they matter:
#
#  1. NOT IN THE STATE FILE. `check_forbidden` REJECTS a state file that even
#     mentions them, naming the key. Not "ignored" — rejected, because a file
#     that contains them is evidence of an escalation attempt and the rest of
#     its contents have not earned any trust either.
#  2. RATCHET. `tighten()` is the only mutator and it computes
#         enabled := enabled AND requested_enabled
#         locked  := locked  OR  requested_locked
#     so loosening is not "disallowed", it is unrepresentable. There is no
#     `loosen()` to audit.
#  3. PER-CONTAINER. The ratchet state lives in the runtime dir, not on the
#     volume, so it is re-derived from the startup environment on every
#     container start. Loosening therefore requires editing the template
#     environment and restarting — i.e. provider-dashboard access, which is
#     exactly the authority a landing-page user may not have.

FORBIDDEN_STATE_KEYS = ("TERMINATE_ENABLED", "TERMINATE_LOCKED")


def check_forbidden(values: dict):
    """Raise if a state file tries to set a startup-only switch."""
    for key in FORBIDDEN_STATE_KEYS:
        if key in values:
            raise ConfigError(
                f"state file rejected: '{key}' is a startup-environment control and "
                "can never be set from the state file or the landing page. "
                "Termination switches only loosen by restarting the container with a "
                "different environment, which needs provider-dashboard access. "
                "Remove the key and try again.")


_PID1_ENV = None


def pid1_env() -> dict:
    """The environment of PID 1, read from /proc/1/environ.

    MEASURED ON A LIVE RUNPOD POD: RunPod injects RUNPOD_POD_ID, RUNPOD_API_KEY
    and friends into the container's MAIN process only. An SSH login shell — or
    anything else started from a new session — gets a FRESH environment and sees
    none of them. A helper that reads os.environ and concludes "no provider
    detected" would be wrong in exactly the situation where a user has SSH'd in
    to sort out a stuck instance.

    The entrypoint itself is PID 1, so it never needed this; every helper that
    might be invoked from another session does. Cached: PID 1's environment
    cannot change under us."""
    global _PID1_ENV
    if _PID1_ENV is None:
        out = {}
        try:
            with open("/proc/1/environ", "rb") as f:
                for item in f.read().split(b"\0"):
                    if b"=" in item:
                        k, v = item.split(b"=", 1)
                        out[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
        except OSError:
            out = {}
        _PID1_ENV = out
    return _PID1_ENV


def effective_env(env=None) -> dict:
    """PID 1's environment, with this process's own environment layered on top.

    An explicit dict is returned untouched — tests and callers that mean a
    specific environment get exactly that. Otherwise the process's own non-empty
    variables win (so an operator can still override something in their shell),
    and anything the shell is missing falls back to what the container was
    actually started with."""
    if env is not None:
        return env
    merged = dict(pid1_env())
    for k, v in os.environ.items():
        if v != "":
            merged[k] = v
    return merged


def env_flag(env, name, default=False):
    v = (env.get(name) or "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "on")


def p_switches() -> str:
    return os.path.join(runtime_dir(), "terminate-switches.json")


def switches_from_env(env=None) -> dict:
    """The startup-environment truth. Defaults are argued in
    docs/termination-and-erase.md; the short version is that every provider
    dashboard can already terminate an instance, so the in-container control is
    a convenience that must be opted into, while the lock is opt-in because a
    locked instance that nobody can unlock is its own failure mode."""
    env = effective_env(env)
    return {"enabled": env_flag(env, "TERMINATE_ENABLED", False),
            "locked": env_flag(env, "TERMINATE_LOCKED", False),
            "source": "startup environment"}


def init_switches(env=None) -> dict:
    """Called once per container start by the entrypoint."""
    st = switches_from_env(env)
    write_json_atomic(p_switches(), st, mode=0o644)
    return st


def read_switches(env=None) -> dict:
    """Current effective switches. Falls back to the environment when the
    runtime file is missing — and the fallback is the RESTRICTIVE direction by
    construction, since a missing file means a fresh container."""
    st = read_json(p_switches())
    if not isinstance(st, dict) or "enabled" not in st or "locked" not in st:
        return switches_from_env(env)
    return {"enabled": bool(st.get("enabled")), "locked": bool(st.get("locked")),
            "source": st.get("source", "runtime")}


def tighten(enabled=None, locked=None, reason="") -> dict:
    """The ONLY mutator. AND/OR with the current state, so it cannot loosen."""
    cur = read_switches()
    new = {
        "enabled": bool(cur["enabled"] and (True if enabled is None else bool(enabled))),
        "locked": bool(cur["locked"] or (False if locked is None else bool(locked))),
        "source": "tightened at runtime" + (f": {reason}" if reason else ""),
    }
    write_json_atomic(p_switches(), new, mode=0o644)
    return new


def termination_allowed(env=None):
    """-> (allowed, reason). `reason` is shown to the user verbatim."""
    st = read_switches(env)
    if st["locked"]:
        return False, (
            "TERMINATION IS LOCKED on this instance (anti-kill switch). This cannot be "
            "unlocked from this page, by any token, for any reason — the lock only "
            "clears when the container is restarted with TERMINATE_LOCKED unset in its "
            "startup environment, which requires access to the provider dashboard or "
            "the template that launched it. Terminate from the provider dashboard "
            "instead if that is what you want.")
    if not st["enabled"]:
        return False, (
            "The in-container terminate control is DISABLED (kill switch off, the "
            "default). It can only be enabled by launching the instance with "
            "TERMINATE_ENABLED=1 in its environment — not from this page. You can "
            "always terminate from your provider's dashboard, which is the path that "
            "actually stops billing.")
    return True, "termination is available on this instance"


def applies_to(knob, family_name) -> bool:
    """Is this knob meaningful for the selected family?

    A knob scoped to a family it is not in is INAPPLICABLE — not "ignored". DCP,
    the EXL3 trellis window and the vision wrapper describe MLA/GLM machinery
    that does not exist in a dense Qwen; offering them would invite invalid
    architecture-specific combinations that this validation layer exists to
    prevent."""
    fams = knob.get("families")
    return not fams or (family_name or "glm52") in fams


def detected_gpu_count(env=None):
    """What the entrypoint measured, passed down as GLM_GPU_COUNT.

    The entrypoint runs gpu_detect.py once at boot (it is the only thing allowed
    to shell out to nvidia-smi) and exports the result. Resolution reads that,
    so nothing in the config layer depends on being able to see a GPU."""
    env = effective_env(env)
    try:
        n = int(env.get("GLM_GPU_COUNT", ""))
    except ValueError:
        return None
    return n if n > 0 else None


def resolve_family(state_values, env_values) -> str:
    """The family has to be settled before anything else can be resolved,
    because it supplies a defaults layer of its own."""
    knob = KNOB_BY_KEY["MODEL_FAMILY"]
    name = knob["default"]
    for layer in (env_values, state_values):
        if layer and "MODEL_FAMILY" in layer:
            try:
                name = coerce(knob, layer["MODEL_FAMILY"])
            except ConfigError:
                pass
    return name


def resolve(state_values=None, env_values=None):
    """-> (effective {key: value}, sources {key: ...}, notes[])

    Sources, lowest to highest: 'default' < 'family' < 'env' < 'file'.
    The family layer sits between the built-in defaults and the environment: it
    is what "GLM-5.2 wants TP=4 and Qwen3.6 wants TP=1" means, and the operator
    must still be able to override it from the template or the page.
    Inapplicable knobs are reported with source 'n/a' and keep a value only so
    that nothing downstream has to special-case a missing key.

    Never raises: an unusable state file degrades to env+defaults and says so in
    `notes`, because bricking the instance over a bad file is worse than
    ignoring it."""
    notes = []
    if env_values is None:
        env_values = load_startup_env()
    if isinstance(env_values, dict):
        for bad in env_values.get("__invalid_env__", []):
            notes.append(f"startup env value ignored ({bad}); the default applies")
    if state_values is None:
        try:
            state_values = load_state_file()
        except (ConfigError, ValueError, OSError) as e:
            notes.append(f"state file ignored ({e}); falling back to env + defaults")
            state_values = {}

    fam_name = resolve_family(state_values, env_values)
    fam = family(fam_name)
    fam_defaults = fam.get("defaults", {})

    ngpu = detected_gpu_count(env_values if env_values else None)
    effective, sources = {}, {}
    for knob in KNOBS:
        key = knob["key"]
        value, src = knob["default"], "default"
        if key == "MODEL_VARIANT":
            value = fam.get("default_variant", value)
            src = "family"
        if key in fam_defaults:
            try:
                value, src = coerce(knob, fam_defaults[key]), "family"
            except ConfigError as e:
                notes.append(f"family default {e}")
        # DETECTED layer: above the family, below the environment. The hardware
        # in front of us outranks a preset's guess about hardware, and both are
        # outranked by someone saying what they want.
        if ngpu and key == "TENSOR_PARALLEL_SIZE":
            value, src = min(ngpu, knob["max"]), "detected"
        if ngpu and key == "DCP":
            # DCP defaults to the TP size: one KV shard per rank. It has to
            # divide TP, and TP is the detected count at this point.
            value, src = str(min(ngpu, 8)), "detected"
        if key in env_values:
            try:
                value, src = coerce(knob, env_values[key]), "env"
            except ConfigError as e:
                notes.append(f"env {e}")
        if key in state_values:
            if not knob.get("editable", True):
                notes.append(f"{key} is locked; the state file's value is ignored")
            else:
                try:
                    value, src = coerce(knob, state_values[key]), "file"
                except ConfigError as e:
                    notes.append(f"state file {e}")
        if not applies_to(knob, fam_name):
            src = "n/a"
        effective[key], sources[key] = value, src
    unknown = [k for k in state_values if k not in KNOB_BY_KEY]
    if unknown:
        notes.append("state file has unknown keys, ignored: " + ", ".join(sorted(unknown)))
    # A checkpoint variant may need a coherent measured memory shape, not just
    # a different repository. Apply that layer above family defaults but below
    # explicit template/state choices so selecting a variant is safe while
    # every individual knob remains overridable.
    variant = VARIANTS.get(effective.get("MODEL_VARIANT"), {})
    for key, raw in variant.get("defaults", {}).items():
        # Hardware detection supplies a safe topology-shaped default (not an
        # operator override). A measured variant may intentionally choose DCP2
        # on TP4, so it outranks "detected"; explicit env/file choices still win.
        if key not in KNOB_BY_KEY or sources.get(key) not in (
                "default", "family", "detected"):
            continue
        try:
            effective[key] = coerce(KNOB_BY_KEY[key], raw)
            sources[key] = "variant"
        except ConfigError as e:
            notes.append(f"variant default {e}")
    # A variant may also carry a stock draft with a different representation
    # from the family's default.
    if (fam_name == "glm52" and variant.get("default_draft")
            and sources.get("MTP_DRAFT") in ("default", "family")):
        effective["MTP_DRAFT"] = variant["default_draft"]
        sources["MTP_DRAFT"] = "variant"
    # The detected DCP must follow the *resolved* TP, not the raw GPU count:
    # an operator may deliberately under-subscribe TP on a bigger host, and a
    # raw count (e.g. 6) may not even be a legal DCP choice. This runs after
    # every layer so an explicit env/file/variant DCP is never touched.
    if sources.get("DCP") == "detected":
        try:
            tp_res = int(effective.get("TENSOR_PARALLEL_SIZE") or 1)
        except (TypeError, ValueError):
            tp_res = 1
        legal = sorted((int(c) for c in KNOB_BY_KEY["DCP"]["choices"]),
                       reverse=True)
        effective["DCP"] = str(next(
            (c for c in legal if c <= tp_res and tp_res % c == 0), 1))
    return effective, sources, notes


def minimize(values: dict, dropped=None) -> dict:
    """Keep only the values that actually OVERRIDE the layers below.

    The state file is a diff, not a snapshot. Writing every knob would freeze
    the instance against its own template: a value the user never touched would
    start winning over the env the operator set at launch, and a later change to
    the defaults could never reach the box. It also keeps an exported config
    portable between instances with different templates."""
    env_layer = load_startup_env()
    fam_name = resolve_family(values, env_layer)
    # Two baselines, and the difference between them matters.
    #  * `base` is what the SELECTED family would give with no state file, so a
    #    knob the user left at that family's own default is not pinned into the
    #    file — the family stays free to change it later.
    #  * `unfiled` is what we would get with no state file AT ALL, which is the
    #    only thing MODEL_FAMILY can be compared against. Comparing the family
    #    to `base` is circular (base was built FROM it), and the earlier version
    #    did exactly that: the selected family always equalled its own baseline,
    #    was therefore never written, and every apply silently reverted to the
    #    previous family — taking its validation rules with it.
    base, _s1, _n1 = resolve(state_values={"MODEL_FAMILY": fam_name})
    unfiled, _s2, _n2 = resolve(state_values={})
    out = {}
    if fam_name != unfiled["MODEL_FAMILY"]:
        out["MODEL_FAMILY"] = fam_name
    for knob in KNOBS:
        key = knob["key"]
        if key == "MODEL_FAMILY" or key not in values or not knob.get("editable", True):
            continue
        try:
            value = coerce(knob, values[key])
        except ConfigError:
            continue
        # A knob the selected family does not have is dropped rather than
        # written: switching family would otherwise leave a stale MTP_DRAFT or
        # DCP in the file, which the next validation would reject and the user
        # would have no obvious way to clear.
        if not applies_to(knob, fam_name):
            if dropped is not None and value != knob["default"]:
                dropped.append(key)
            continue
        if value != base[key]:
            out[key] = value
    return out


# --------------------------------------------------------------------------
# derived values (what the serve path actually consumes)
# --------------------------------------------------------------------------

def family_serve_args(cfg: dict):
    """The family's own vLLM flags, with %(KNOB)s placeholders filled in.

    Returned as a list so the caller can quote each element exactly once; these
    values contain JSON with spaces and braces."""
    fam = family(cfg.get("MODEL_FAMILY"))
    subs = {k: to_text(KNOB_BY_KEY[k], v) for k, v in cfg.items() if k in KNOB_BY_KEY}
    args = [a % subs if "%(" in a else a for a in fam.get("serve_args", [])]
    if cfg.get("MODEL_FAMILY") == "glm52" and cfg.get("CLAMP_ROPE_TABLES", True):
        # Keep this conditional rather than formatting a sentinel into the JSON:
        # disabling the experiment must omit the override entirely so the checkpoint's
        # native value is used, exactly matching the pre-clamp rental control.
        try:
            index = args.index("--hf-overrides") + 1
            overrides = json.loads(args[index])
            overrides["max_position_embeddings"] = cfg["MAX_MODEL_LEN"]
            args[index] = json.dumps(overrides, separators=(",", ":"))
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    variant = VARIANTS.get(cfg.get("MODEL_VARIANT"), {})
    if variant.get("quantization"):
        args = ["--quantization", variant["quantization"]] + args
    if variant.get("quantization_config"):
        args += ["--quantization-config", variant["quantization_config"]]
    if cfg.get("KV_CACHE_MEMORY_BYTES", 0) > 0:
        args += ["--kv-cache-memory-bytes", str(cfg["KV_CACHE_MEMORY_BYTES"])]
    if cfg.get("MODEL_FAMILY") in ("qwen36", "custom") and not cfg.get("MULTIMODAL"):
        args += ["--language-model-only"]
    if (
        cfg.get("MODEL_FAMILY") == "qwen36"
        and cfg.get("MTP_TOKENS", 0) > 0
    ):
        # GG v20-r9 / FlashInfer freezes q_len_per_req=1 in the compiled
        # decode wrapper, while Qwen's native MTP2 draft needs q_len=3. The
        # first request otherwise kills the engine. Eager mode is slower but
        # is the compatibility path until upstream owns one wrapper per query
        # length.
        args += ["--enforce-eager"]
    if cfg.get("MODEL_FAMILY") == "custom":
        if cfg.get("QUANTIZATION"):
            args = ["--quantization", cfg["QUANTIZATION"]] + args
        if cfg.get("REASONING_PARSER"):
            args += ["--reasoning-parser", cfg["REASONING_PARSER"]]
        if cfg.get("TOOL_CALL_PARSER"):
            args += ["--enable-auto-tool-choice", "--tool-call-parser",
                     cfg["TOOL_CALL_PARSER"]]
    comp = fam.get("compilation_config")
    if comp:
        args += ["--compilation-config", comp % subs]
    return args


def derive(cfg: dict) -> dict:
    """Expand the knobs into the env the entrypoint's serve path reads."""
    fam_name = cfg.get("MODEL_FAMILY", "glm52")
    fam = family(fam_name)
    variant = VARIANTS.get(cfg["MODEL_VARIANT"], VARIANTS["exl3-tr3"])
    draft = DRAFTS.get(cfg["MTP_DRAFT"], DRAFTS["tr3-graft"])
    out = dict(cfg)
    if fam_name == "custom":
        model_id = cfg.get("MODEL_ID", "")
        dirname = re.sub(r"[^A-Za-z0-9._-]", "-", model_id).strip("-") or "custom"
        out["MODEL_REPO"] = model_id
        out["MODEL_DIRNAME"] = os.path.join("models", dirname)
        out["QUANTIZATION"] = cfg.get("QUANTIZATION", "")
    else:
        out["MODEL_REPO"] = variant["repo"]
        out["MODEL_REVISION"] = variant.get("revision", "")
        out["MODEL_DIRNAME"] = variant["dirname"]
        out["QUANTIZATION"] = variant["quantization"]
        out["NATIVE_MTP_FORMAT"] = variant.get("native_mtp_format", "")
    out["FAMILY_ENV_BLOCK"] = fam.get("env_block", "generic")
    out["SPEC_METHOD"] = fam.get("spec_method", "mtp")
    out["FAMILY_SERVE_ARGS"] = family_serve_args(cfg)
    out["PROFILE_RUNTIME_ENV"] = [
        f"{key}={value}" for key, value in variant.get("runtime_env", {}).items()
    ]
    # The MTP78 draft apparatus is GLM-only; outside it there is no graft, no
    # overlay and no external draft dir, just a speculation depth.
    if fam_name != "glm52":
        out["MTP78_MODE"] = "off"
        out["DRAFT_MODEL"] = ""
        out["DRAFT_QUANTIZATION"] = ""
        out["VISION"] = False
        return out
    out["MTP78_MODE"] = draft["mtp78_mode"]
    # rule draft-inherits-quant: a non-EXL3 draft against an EXL3 target must
    # declare its own quantization or it is loaded through the EXL3 path.
    if not cfg.get("DRAFT_QUANTIZATION") and draft.get("draft_quantization"):
        out["DRAFT_QUANTIZATION"] = draft["draft_quantization"]
    if cfg["MTP_DRAFT"] == "nvfp4" and not cfg.get("DRAFT_MODEL"):
        out["DRAFT_MODEL"] = os.path.join(
            os.path.dirname(MODEL_DIR.rstrip("/")) or "/workspace", "GLM-5.2-NVFP4-mtp")
    if cfg["MTP_TOKENS"] == 0:
        out["MTP78_MODE"] = "off"
    return out


# --------------------------------------------------------------------------
# pre-validation matrix
# --------------------------------------------------------------------------
# Every entry: (id, level, keys, message). level "error" blocks an apply;
# level "warn" is shown but allowed. Measured 2026-07-26 unless stated.

def validate(cfg: dict, context=None):
    """-> list of {id, level, keys, message}. `context` may carry
    {'vision_available': bool} for checks that depend on the volume."""
    ctx = context or {}
    out = []

    def err(rid, keys, msg):
        out.append({"id": rid, "level": "error", "keys": keys, "message": msg})

    def warn(rid, keys, msg):
        out.append({"id": rid, "level": "warn", "keys": keys, "message": msg})

    variant = VARIANTS.get(cfg["MODEL_VARIANT"], VARIANTS["exl3-tr3"])
    draft = cfg["MTP_DRAFT"]
    seqs, toks = cfg["MAX_NUM_SEQS"], cfg["MTP_TOKENS"]
    fam_name = cfg.get("MODEL_FAMILY", "glm52")
    fam = family(fam_name)
    is_glm = fam_name == "glm52"

    if fam_name == "qwen36" and toks > 0:
        warn(
            "qwen-mtp-eager",
            ["MTP_TOKENS"],
            "Qwen native MTP currently forces --enforce-eager on GG v20-r9. "
            "The compiled FlashInfer path freezes q_len_per_req=1 and crashes "
            "when MTP2 requires q_len=3. On one RTX 5090, eager MTP2 reduced "
            "the measured KV ceiling from 227,161 to about 195,200 tokens and "
            "decode from 65/110/190 to 46/81/101 tok/s at C1/C2/C4 despite "
            "74-79% draft acceptance. Keep MTP_TOKENS=0 for the qualified "
            "compiled default.",
        )

    # 0. family coherence ---------------------------------------------------
    if not fam.get("tested"):
        warn("family-untested", ["MODEL_FAMILY"],
             f"{fam['label']}: this exact full-size checkpoint/profile has not completed "
             "the production-scale qualification matrix. Treat a clean boot as the "
             "beginning of model-specific validation rather than the end of it.")
    if fam_name == "custom" and not cfg.get("MODEL_ID"):
        err("custom-model-required", ["MODEL_ID"],
            "MODEL_FAMILY=custom requires MODEL_ID=<Hugging Face repo>; refusing "
            "to guess or silently download GLM-5.2.")
    if variant.get("family") and variant["family"] != fam_name:
        err("variant-family-mismatch", ["MODEL_VARIANT", "MODEL_FAMILY"],
            f"model variant '{cfg['MODEL_VARIANT']}' belongs to the "
            f"{variant['family']} family, not {fam_name}. Pick one of: "
            + ", ".join(k for k, v in VARIANTS.items() if v.get("family") == fam_name))
    if not str(cfg.get("SERVED_MODEL_NAME", "")).strip():
        err("served-name-empty", ["SERVED_MODEL_NAME"],
            "SERVED_MODEL_NAME cannot be blank: it is the id clients pass in "
            "the OpenAI 'model' field and the name every dashboard page "
            "displays. Set at least one alias (the profile default is fine).")
    if cfg["KV_CACHE_DTYPE"] not in fam.get("kv_dtypes", []):
        err("kv-dtype-family", ["KV_CACHE_DTYPE", "MODEL_FAMILY"],
            f"KV dtype '{cfg['KV_CACHE_DTYPE']}' is not available for {fam['label']}. "
            f"This family supports: {', '.join(fam.get('kv_dtypes', []))}. "
            "(nvfp4_ds_mla in particular is an MLA KV layout — it has no meaning "
            "outside an MLA model.)")
    ngpu = ctx.get("gpu_count")
    tp = cfg["TENSOR_PARALLEL_SIZE"]
    if ngpu is not None and tp > ngpu:
        err("tp-exceeds-gpus", ["TENSOR_PARALLEL_SIZE"],
            f"tensor-parallel size {tp} needs {tp} GPUs and this host has {ngpu}. The "
            "engine cannot start. Lower it, or rent a host with more GPUs.")
    if is_glm and tp < 4:
        err("glm-needs-four-ranks", ["TENSOR_PARALLEL_SIZE", "MODEL_FAMILY"],
            f"the GLM-5.2 753B checkpoint is ~308 GiB of weights (~77 GiB/rank at "
            f"TP=4). At TP={tp} it cannot fit: even on 96 GB cards that is "
            f"{308 // max(tp, 1)} GiB per rank before any KV cache. Use 4 or more "
            "GPUs, or switch to a family sized for this host (MODEL_FAMILY=qwen36).")
    if is_glm and tp != 4:
        warn("tp-off-measured", ["TENSOR_PARALLEL_SIZE"],
             f"every measured GLM-5.2 number in this repo — the 835,584-token pool, "
             f"121.3 tok/s decode, needle 5/5 at 490K — comes from TP=4 on 4x RTX PRO "
             f"6000. At TP={tp} none of them should be assumed to hold: the KV pool "
             "pin (GPU_BLOCKS_OVERRIDE), the DCP shard count and the memory profile "
             "all move with the rank count. The CUDA-graph/trellis window arithmetic "
             "(MAX_NUM_SEQS x (1+MTP_TOKENS)) is a per-kernel batch width and does NOT "
             "move with TP — that one rule still applies unchanged.")
    if is_glm and tp != 4 and cfg["GPU_BLOCKS_OVERRIDE"]:
        warn("pool-pin-off-measured", ["GPU_BLOCKS_OVERRIDE", "TENSOR_PARALLEL_SIZE"],
             f"the KV pool is pinned at {cfg['GPU_BLOCKS_OVERRIDE']} blocks, a value "
             "chosen to give exactly 512K tokens at TP=4/DCP=4. At a different rank "
             "count that pin no longer means 512K; set GPU_BLOCKS_OVERRIDE=0 to let "
             "vLLM size the pool for the hardware it actually has.")

    # 1. concurrency vs the capture + trellis windows -----------------------
    # GLM/EXL3-SCOPED. The trellis window belongs to the EXL3 kernel; applying
    # these rules to a dense Qwen would reject configurations that are fine.
    m = seqs * (1 + toks)
    cap_max = cfg["MAX_CUDAGRAPH_CAPTURE_SIZE"]
    trellis_max = cfg["VLLM_EXL3_TRELLIS_MAX_M"]
    # The trellis window belongs to the EXL3 kernel; on a non-EXL3 GLM target
    # only the CUDA-graph capture window bounds the decode width. The width
    # rule applies with speculation off too: m = MAX_NUM_SEQS is still a
    # per-kernel batch width that silently leaves the captured path.
    if variant.get("quantization") == "exl3":
        window = min(cap_max, trellis_max)
    else:
        window = cap_max
    if is_glm and m > window:
        err("concurrency-window", ["MAX_NUM_SEQS", "MTP_TOKENS",
                                   "MAX_CUDAGRAPH_CAPTURE_SIZE", "VLLM_EXL3_TRELLIS_MAX_M"],
            f"MAX_NUM_SEQS x (1 + MTP_TOKENS) = {seqs} x {1 + toks} = {m} query tokens per "
            f"decode step, but the CUDA-graph capture window tops out at {cap_max} and the "
            f"EXL3 trellis window at {trellis_max}. Exceeding it does NOT error at boot: "
            "decode silently leaves the captured trellis fast path under concurrency and "
            "loses throughput, with nothing in the log to say so. Raise "
            "CUDAGRAPH_CAPTURE_SIZES, MAX_CUDAGRAPH_CAPTURE_SIZE and "
            f"VLLM_EXL3_TRELLIS_MAX_M together to >= {m}, or lower MAX_NUM_SEQS to "
            f"{max(1, window // (1 + toks))}.")
    sizes = [int(x) for x in str(cfg["CUDAGRAPH_CAPTURE_SIZES"]).split(",") if x]
    if sizes:
        if max(sizes) != cap_max:
            warn("capture-sizes-max", ["CUDAGRAPH_CAPTURE_SIZES", "MAX_CUDAGRAPH_CAPTURE_SIZE"],
                 f"the largest captured size ({max(sizes)}) does not equal "
                 f"MAX_CUDAGRAPH_CAPTURE_SIZE ({cap_max}); decode above {max(sizes)} runs "
                 "eager whatever the max says.")
    # 2. rank-sliced EXL3 draft compatibility --------------------------------
    # v29 stamps the draft role at construction and gives draft layers a
    # capturable m=1 floor while target layers keep m=4. The former v20 blanket
    # rejection and global MIN_M workaround are both obsolete.
    if (is_glm and draft.startswith("tr3-")
            and variant.get("quantization") != "exl3"):
        err("tr3-draft-needs-exl3", ["MTP_DRAFT", "MODEL_VARIANT"],
            "the TR3 trellis drafts are layer-78 overlays built for the EXL3-TR3 "
            f"checkpoint; they have no meaning on the {variant['label']} target. Use the "
            "native in-checkpoint draft or an external nvfp4 draft dir.")

    # 3. a draft inherits the target's quantization --------------------------
    # derive() fills DRAFT_QUANTIZATION from the draft registry when the knob
    # is left empty (the documented behavior), so the rule must judge the
    # value that will actually be served, not the raw knob.
    effective_draft_quant = (cfg.get("DRAFT_QUANTIZATION")
                             or DRAFTS.get(draft, {}).get("draft_quantization", ""))
    if is_glm and draft == "nvfp4" and effective_draft_quant != "modelopt_fp4":
        err("draft-quant-inherit", ["DRAFT_QUANTIZATION", "MTP_DRAFT"],
            "a draft inherits the target's --quantization unless its speculative config "
            "carries its own, so an NVFP4 draft against an EXL3 target is loaded through "
            "the EXL3 path and throws the identical 'capture (m=3)' error as the v20 "
            "rank-sliced bug. Set DRAFT_QUANTIZATION=modelopt_fp4. This is a config trap, "
            "not a bug.")
    if is_glm and re.search(r'["\\]', str(cfg.get("DRAFT_MODEL") or "")):
        err("draft-model-quoting", ["DRAFT_MODEL"],
            "DRAFT_MODEL may not contain double quotes or backslashes: the "
            "path is embedded verbatim in the --speculative-config JSON "
            "literal and would corrupt it at engine start.")
    if is_glm and cfg.get("DRAFT_MODEL") and draft in (
            "tr3-graft", "native", "bf16", "off"):
        warn("draft-model-overrides", ["DRAFT_MODEL", "MTP_DRAFT"],
             "an explicit DRAFT_MODEL takes over draft selection; the MTP draft type is "
             "then only used to decide what happens to the target checkpoint.")

    # 5. nvfp4 KV needs calibrated MLA outer scales --------------------------
    if is_glm and cfg["KV_CACHE_DTYPE"] not in ("fp8", "auto") \
            and not variant["kv_scales_calibrated"]:
        err("kv-nvfp4-uncalibrated", ["KV_CACHE_DTYPE", "MODEL_VARIANT"],
            f"KV dtype {cfg['KV_CACHE_DTYPE']} requires per-checkpoint calibrated MLA "
            f"outer scales and the {variant['label']} profile declares none. Earlier "
            "uncalibrated NVFP4 runs silently degenerated at long context while short "
            "quality and feature checks passed. Use fp8 for this variant, or provide "
            "and qualify model-specific calibrated scales.")
    if is_glm and cfg["KV_SCALE_MODE"] == "dynamic-token" \
            and cfg["KV_CACHE_DTYPE"] != "nvfp4_ds_mla":
        err("kv-dynamic-dtype", ["KV_SCALE_MODE", "KV_CACHE_DTYPE"],
            "Dynamic-token MLA scaling requires KV_CACHE_DTYPE=nvfp4_ds_mla. "
            "Use static-calibrated for fp8/auto KV, or select the NVFP4 MLA dtype.")

    # 6. vision on an EXL3 target -------------------------------------------
    if (is_glm and cfg["VISION"]
            and cfg["MODEL_VARIANT"].startswith("exl3-tr3")):
        warn("vision-long-context", ["VISION"],
             "EXPERIMENTAL / THIS COMPOSITION FAILED LONG CONTEXT: VISION=1 on the "
             "tested EXL3-TR3 checkpoint corrupts long-context output. The final v20 turnkey image "
             "measured 32K needle 0/3 with degenerate text and mean MTP acceptance "
             "collapsing to 1.25-1.50, while a 5120x2880 screenshot returned 17/18 "
             "exact details and its multi-turn follow-up was exact. This finding is "
             "not generalized to the Baseten tower or other published quant/wrapper "
             "compositions. Safe here only for short-prompt image work; the "
             "long-context probe will fail this configuration by design.")
    if is_glm and cfg["VISION"] and cfg["MAX_MODEL_LEN"] > 393216:
        warn("vision-kv-pressure", ["VISION", "MAX_MODEL_LEN"],
             f"the current vision graft adds ~1.99 GiB/GPU. On the tested DCP4 "
             f"EXL3-TR3 max-context shape, MAX_MODEL_LEN={cfg['MAX_MODEL_LEN']} and "
             f"GPU_MEMORY_UTILIZATION=0.975 exposed 564,736 KV tokens and served short "
             f"requests, while 0.98 exposed 610,560 tokens but left only 37.12 MiB free "
             f"and OOMed on the first 48 MiB verification allocation. Capacity at boot "
             f"is not runtime headroom, and neither result makes vision long-context "
             f"correct. Use <=0.975 on that exact shape; otherwise reduce context or "
             f"re-qualify memory and retrieval from a cold start.")
    if is_glm and cfg["VISION"] and ctx.get("vision_available") is False:
        warn("vision-unavailable", ["VISION"],
             "vision assets are not on the volume yet; enabling it triggers a ~1 GB "
             "download during the restart.")

    # 7. memory / pool sanity ------------------------------------------------
    if cfg["PREFIX_CACHE_DISK_GB"] and cfg["PREFIX_CACHE_BACKEND"] != "lmcache":
        err("disk-cache-needs-lmcache",
            ["PREFIX_CACHE_DISK_GB", "PREFIX_CACHE_BACKEND"],
            "The native OffloadingConnector has no filesystem tier. Select "
            "PREFIX_CACHE_BACKEND=lmcache, or set PREFIX_CACHE_DISK_GB=0.")
    if cfg["PREFIX_CACHE_DISK_GB"] and not cfg["OFFLOAD_FRACTION"]:
        err("disk-cache-needs-l1",
            ["PREFIX_CACHE_DISK_GB", "OFFLOAD_FRACTION"],
            "LMCache's bounded filesystem L2 is supervised by its DRAM L1 manager. "
            "Set OFFLOAD_FRACTION above zero before enabling the disk tier.")
    if cfg["PREFIX_CACHE_DISK_GB"]:
        warn("disk-cache-sensitive",
             ["PREFIX_CACHE_DISK_GB", "PREFIX_CACHE_BACKEND"],
             f"LMCache may retain up to {cfg['PREFIX_CACHE_DISK_GB']} GiB of derived "
             "prompt KV on persistent storage. This can contain session material. "
             "Use encrypted local NVMe where possible and start the appliance with "
             "TERMINATE_ENABLED=1 so its optional secure-erase flow is available. "
             "Erasure remains best effort on SSDs and provider-managed storage.")
    fixed_kv = cfg.get("KV_CACHE_MEMORY_BYTES", 0)
    if fixed_kv and cfg["GPU_BLOCKS_OVERRIDE"]:
        err("fixed-kv-block-conflict",
            ["KV_CACHE_MEMORY_BYTES", "GPU_BLOCKS_OVERRIDE"],
            "Choose one reproducible KV-pool control. --kv-cache-memory-bytes "
            "already fixes the per-GPU pool; combining it with a block-count "
            "override is ambiguous and has not been qualified.")
    elif fixed_kv:
        warn("fixed-kv-cache",
             ["KV_CACHE_MEMORY_BYTES", "GPU_MEMORY_UTILIZATION"],
             f"KV_CACHE_MEMORY_BYTES={fixed_kv} fixes the per-GPU KV allocation. "
             "vLLM ignores GPU_MEMORY_UTILIZATION for KV sizing when this is set. "
             "The fixed pool can preserve transient runtime workspace, but it also "
             "caps context/concurrency and is hardware/model specific; verify its "
             "reported token capacity and run the near-maximum correctness gate.")
    qualified_r11_512k = (
        is_glm
        and cfg["MODEL_VARIANT"] == "exl3-tr3"
        and cfg["LOAD_FORMAT"] == "safetensors"
        and cfg["DCP"] == "2"
        and cfg["MAX_MODEL_LEN"] == 524288
        and cfg["MAX_NUM_BATCHED_TOKENS"] == 3072
        and cfg["MAX_NUM_SEQS"] == 8
        and cfg["DCP_CKV_GATHER_MAX_TOKENS"] == 140000
        and cfg["GPU_MEMORY_UTILIZATION"] == 0.957
        and cfg["KV_CACHE_DTYPE"] == "nvfp4_ds_mla"
        and cfg["KV_SCALE_MODE"] == "dynamic-token"
        and cfg["PREFIX_CACHE_BACKEND"] == "lmcache"
        and cfg["MTP_TOKENS"] == 5
        and cfg["MAX_CUDAGRAPH_CAPTURE_SIZE"] == 64
        and cfg["VLLM_EXL3_TRELLIS_MAX_M"] == 64
        and not cfg["VISION"]
        and not cfg["GPU_BLOCKS_OVERRIDE"]
        and not fixed_kv
    )
    if (is_glm and cfg["MODEL_VARIANT"].startswith("exl3-tr3")
            and cfg["LOAD_FORMAT"] == "instanttensor"
            and cfg["MAX_MODEL_LEN"] >= 524288
            and cfg["GPU_MEMORY_UTILIZATION"] <= 0.978):
        warn("instanttensor-context-margin",
             ["LOAD_FORMAT", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION"],
             "an earlier TP4/DCP2 EXL3 shape at MAX_MODEL_LEN=524288 and "
             "GPU_MEMORY_UTILIZATION=0.978 failed KV admission on both InstantTensor "
             "attempts: 9.04 GiB was needed and 9.03 GiB remained. InstantTensor "
             "loaded ~0.04 GiB/GPU more resident model memory than safetensors. "
             "The r11 production profile therefore uses safetensors. Re-qualify "
             "this override from a cold start and at maximum context; do not infer "
             "reliability from one successful InstantTensor boot.")
    if (cfg["GPU_MEMORY_UTILIZATION"] > 0.95
            and not cfg["GPU_BLOCKS_OVERRIDE"] and not fixed_kv):
        if qualified_r11_512k:
            warn("gpu-util-high", ["GPU_MEMORY_UTILIZATION"],
                 f"{cfg['GPU_MEMORY_UTILIZATION']} is high by generic vLLM standards, "
                 "but the GG v20-r11 safetensors/DCP2/LMCache profile at 0.957 "
                 "passed a cold boot, a same-image AOT-cache restart, the complete "
                 "OpenAI feature suite, and an exact 522,360-token five-depth "
                 "retrieval on AIBeast. It exposed 542,208 logical KV tokens on "
                 "the cold qualification boot. "
                 "A new driver, GPU SKU, loader, graph shape, or vision setting is still "
                 "a cold-boot and near-maximum retrieval requalification boundary.")
        else:
            warn("gpu-util-high", ["GPU_MEMORY_UTILIZATION"],
                 f"{cfg['GPU_MEMORY_UTILIZATION']} leaves very little headroom. The "
                 "memory profile varies by a few hundred MiB between boots, so a value "
                 "that boots two times in three is not a value that boots — it is a "
                 "rollback waiting to happen.")
    if (is_glm and cfg["MODEL_VARIANT"] == "madeby561-hybrid"
            and cfg["MAX_MODEL_LEN"] >= 524288
            and not cfg["GPU_BLOCKS_OVERRIDE"]):
        warn("hybrid-512k-needs-pool-pin",
             ["MODEL_VARIANT", "MAX_MODEL_LEN", "GPU_BLOCKS_OVERRIDE"],
             "the v20 hybrid target can pass startup admission with an automatically "
             "sized pool and still OOM in the first MTP proposal: the pool consumes "
             "the proposal's transient allocation. The measured 512K profile pins "
             "2048 blocks. Keep that pin, or re-run the 32K and near-maximum needle "
             "gates before trusting this configuration.")
    if (is_glm
            and variant.get("quantization") == "exl3"
            and cfg["VLLM_EXL3_PREFILL_CAPACITY"]
            > cfg["MAX_NUM_BATCHED_TOKENS"]):
        err("exl3-prefill-capacity-above-scheduler",
            ["VLLM_EXL3_PREFILL_CAPACITY", "MAX_NUM_BATCHED_TOKENS"],
            "VLLM_EXL3_PREFILL_CAPACITY is the reusable EXL3 arena inside one "
            "scheduler chunk and cannot exceed MAX_NUM_BATCHED_TOKENS. Lower the "
            "arena capacity or raise the scheduler chunk; values are rejected "
            "rather than silently clamped so the measured memory shape remains "
            "reproducible.")
    if (is_glm and cfg["MODEL_VARIANT"] == "exl3-tr3-3.25bpw"
            and cfg["MAX_MODEL_LEN"] >= 524288
            and cfg["GPU_BLOCKS_OVERRIDE"] >= 2048
            and cfg["OFFLOAD_FRACTION"] > 0
            and cfg["MAX_NUM_BATCHED_TOKENS"] > 2048):
        err("mixed-325-offload-headroom",
            ["MODEL_VARIANT", "MAX_MODEL_LEN", "GPU_BLOCKS_OVERRIDE",
             "OFFLOAD_FRACTION", "MAX_NUM_BATCHED_TOKENS"],
            "the qualified 3.25bpw DCP4/MTP3 512K offload profile uses a "
            "2,048-token prefill chunk and a 64 MiB sparse-fold budget. The older "
            "3,072-token shape OOMed its first 128K request after a successful "
            "boot. Keep MAX_NUM_BATCHED_TOKENS<=2048 for full-context offload, "
            "or re-run the cold 128K and near-maximum request gates.")
    # DCP multiplies logical KV capacity only on the MLA/DCP (GLM) path; for
    # other families a block is 64 tokens, full stop — using the inapplicable
    # GLM DCP value here would wave through a pool 4x too small.
    dcp_mult = int(cfg.get("DCP", "1") or "1") if is_glm else 1
    logical_tokens_per_block = 64 * dcp_mult
    if (cfg["GPU_BLOCKS_OVERRIDE"]
            and cfg["GPU_BLOCKS_OVERRIDE"] * logical_tokens_per_block
            < cfg["MAX_MODEL_LEN"]):
        err("pool-smaller-than-context", ["GPU_BLOCKS_OVERRIDE", "MAX_MODEL_LEN"],
            f"the pinned KV pool is {cfg['GPU_BLOCKS_OVERRIDE']} blocks x 64 tokens "
            f"x DCP={dcp_mult} = "
            f"{cfg['GPU_BLOCKS_OVERRIDE'] * logical_tokens_per_block} tokens, "
            f"smaller than MAX_MODEL_LEN "
            f"({cfg['MAX_MODEL_LEN']}). vLLM refuses to start when it cannot fit one "
            "max-length request: 'To serve at least one request with the model's max seq "
            "len ... larger than the available KV cache memory'. Raise the override "
            f"to >= {-(-cfg['MAX_MODEL_LEN'] // logical_tokens_per_block)} blocks, "
            "set it to 0 to let vLLM size "
            "the pool, or lower MAX_MODEL_LEN.")
    if cfg["MAX_NUM_BATCHED_TOKENS"] > cfg["MAX_MODEL_LEN"]:
        warn("chunk-bigger-than-context", ["MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN"],
             "the prefill chunk is larger than the whole context window; it will simply "
             "be clamped.")
    qualified_madeby_ring = (
        cfg.get("MODEL_VARIANT") == "madeby561-hybrid"
        and cfg["F8_DMA"] == "ring"
        and cfg["MAX_MODEL_LEN"] == 524288
        and cfg["MAX_NUM_BATCHED_TOKENS"] == 2048
        and cfg["DCP_PREFILL_WORKSPACE_MIB"] == 512
        and cfg["GPU_BLOCKS_OVERRIDE"] == 2048
        and str(cfg["DCP_CKV_PREFETCH_DEPTH"]) == "0"
        and cfg["DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS"] == 8192
        and cfg["PCIE_DMA_MIN_BYTES"] == 393216
    )
    if is_glm and cfg["F8_DMA"] != "0" and not qualified_madeby_ring:
        warn("compressed-dma-needs-retrieval",
             ["F8_DMA", "MAX_MODEL_LEN"],
             f"F8_DMA={cfg['F8_DMA']} compresses collective traffic. It may improve "
             "PCIe-limited throughput, but it is a quality-affecting path rather than "
             "the lossless auto-calibrated default. Re-run cold near-maximum retrieval "
             "and degeneration checks on this exact shape before promoting it. The "
             "unmodified MadeBy561 ring preset is exempt because it passed the "
             "five-depth 521K gate on the final v20 turnkey image.")

    # 8. parallelism (DCP is an MLA-path feature: GLM only) -------------------
    if is_glm and int(cfg["TENSOR_PARALLEL_SIZE"]) % int(cfg["DCP"]) != 0:
        err("dcp-divides-tp", ["DCP", "TENSOR_PARALLEL_SIZE"],
            f"decode context parallel size {cfg['DCP']} must divide the tensor-parallel "
            f"size ({cfg['TENSOR_PARALLEL_SIZE']}).")
    if is_glm and int(cfg["DCP"]) == 1 and cfg["MAX_MODEL_LEN"] > 262144:
        warn("dcp-reduces-pool", ["DCP", "MAX_MODEL_LEN"],
             f"DCP={cfg['DCP']} replicates KV instead of sharding it across ranks, "
             "cutting the usable pool by about 4x on TP4. "
             f"MAX_MODEL_LEN {cfg['MAX_MODEL_LEN']} is unlikely to fit.")

    # 9. variant ------------------------------------------------------------
    if not variant["tested"]:
        warn("variant-untested", ["MODEL_VARIANT"],
             f"{variant['label']}: this template's serve arguments for it are derived, "
             "not measured. Expect to iterate, and expect a "
             f"~{variant['download_gib']} GiB download on the next restart.")
    if is_glm and toks == 0:
        warn("spec-off", ["MTP_TOKENS"],
             "speculative decoding disabled; expect roughly 30% lower decode throughput.")

    # 10. a persisted knob that this family does not have ---------------------
    # Same principle as the termination switches: a state file that carries
    # something meaningless for the running configuration is not quietly
    # ignored, it is reported with the key named. The landing page drops these
    # on apply, so reaching this rule means the file was edited by hand.
    for key in ctx.get("state_keys") or ():
        knob = KNOB_BY_KEY.get(key)
        if knob and not applies_to(knob, fam_name):
            err("knob-inapplicable", [key, "MODEL_FAMILY"],
                f"'{key}' has no meaning for {fam['label']} and is set in the state "
                f"file. {knob['label']} belongs to the "
                f"{', '.join(knob['families'])} family. Remove it, or switch family.")
    return out


def errors(findings):
    return [f for f in findings if f["level"] == "error"]


# --------------------------------------------------------------------------
# known boot-failure signatures (used by rollback reporting + self-analysis)
# --------------------------------------------------------------------------

SIGNATURES = [
    (r"larger than the available KV cache memory|No available memory for the cache blocks",
     "KV validation failed: the engine could not fit one request of MAX_MODEL_LEN in the "
     "KV cache it was left with. Reducing MAX_MODEL_LEN is the standard remedy; lowering "
     "GPU_BLOCKS_OVERRIDE (or setting it to 0), turning VISION off, or a smaller draft "
     "also free KV."),
    (r"eager parity path entered during CUDA graph capture",
     "An EXL3 kernel was asked to capture a CUDA graph at a batch width outside the "
     "Trellis window. v29 normally classifies target and draft roles independently; "
     "a global VLLM_EXL3_TRELLIS_MIN_M override can defeat that. A non-EXL3 draft can "
     "also reach this path when it inherits the target's --quantization; set "
     "DRAFT_QUANTIZATION for such drafts."),
    (r"CUDA out of memory|torch\.OutOfMemoryError",
     "Out of VRAM. Lower GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MAX_NUM_SEQS or "
     "MAX_NUM_BATCHED_TOKENS, or use a smaller draft."),
    (r"are not supported for now|Model architectures \[.*\] are not supported",
     "The architecture in config.json is not registered in this container — typically "
     "the Glm5v vision wrapper without the vision plugin installed. Turning VISION off "
     "(or letting the boot reinstall the plugin) resolves it."),
    (r"KeyError: 'model\.layers\.78\..*w2_weight'|routed_experts\.w2_weight",
     "Layer 78 is described as a trellis MoE in config.json while the weights on disk are "
     "BF16 (or the reverse). This is a derived-config/weights mismatch, not a knob: the "
     "checkpoint reconciler rewrites hybrid_tr3_tail.moe_layers and "
     "quantization_config.ignore from the tensors actually present."),
    (r"quantization_config is only supported when",
     "An online quantization overlay is fighting the checkpoint's own quantization "
     "config; ONLINE_QUANT=none is required for EXL3."),
    (r"Failed to apply prompt replacement for mm_items",
     "The multimodal processor did not find the image placeholders, i.e. the text-only "
     "chat template is installed while the vision wrapper is active."),
]


def match_signatures(log_text: str):
    hits = []
    for pattern, explanation in SIGNATURES:
        if re.search(pattern, log_text):
            hits.append(explanation)
    return hits


# --------------------------------------------------------------------------
# diffs + apply state
# --------------------------------------------------------------------------

def diff(old: dict, new: dict):
    """-> list of (key, old, new) over the knob registry."""
    rows = []
    for knob in KNOBS:
        key = knob["key"]
        a, b = old.get(key), new.get(key)
        if a != b:
            rows.append((key, a, b))
    return rows


def diff_text(old: dict, new: dict) -> str:
    rows = diff(old, new)
    if not rows:
        return "(no differences)"
    return "\n".join(f"{k}: {a!r} -> {b!r}" for k, a, b in rows)


def apply_state() -> dict:
    st = read_json(p_apply_state())
    if not isinstance(st, dict) or "mode" not in st:
        return {"mode": "steady", "since": None, "detail": ""}
    return st


def set_apply_state(mode: str, **kw):
    st = {"mode": mode}
    st.update(kw)
    write_json_atomic(p_apply_state(), st, mode=0o644)
    return st
