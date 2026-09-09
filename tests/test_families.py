#!/usr/bin/env python3
"""Tests for the model-family preset layer.

    python3 tests/test_families.py

Nothing here starts an engine or touches a GPU: it exercises the resolver, the
family-scoped validation matrix, and the exact argv the serve line would be
given. These tests prove profile plumbing and preserve the measured GLM path;
full-checkpoint runtime qualification remains a separate live test.
"""
import json
import os
import shutil
import sys
import subprocess
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import glm_config as gc  # noqa: E402
import config_cli  # noqa: E402

FAILURES, PASSED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILURES).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILURES[-1] = (name, detail)
        raise AssertionError(detail)


def section(t):
    print(f"\n=== {t} ===")


def resolved(family=None, gpus=None, **state):
    values = dict(state)
    if family:
        values["MODEL_FAMILY"] = family
    env = {"GLM_GPU_COUNT": str(gpus)} if gpus else {}
    return gc.resolve(state_values=values, env_values=env)


def ids(findings):
    return {f["id"] for f in findings}


def errs(findings):
    return {f["id"] for f in findings if f["level"] == "error"}


# --------------------------------------------------------------------------

def test_glm_release_defaults():
    section("the GLM GG v20-r13 qualification defaults")
    eff, src, _ = resolved(gpus=4)
    check("family defaults to glm52", eff["MODEL_FAMILY"] == "glm52")
    check("TP is 4 on a 4-GPU box, and it is DETECTED not assumed",
          eff["TENSOR_PARALLEL_SIZE"] == 4 and src["TENSOR_PARALLEL_SIZE"] == "detected",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    check("the balanced profile selects DCP2 on TP4",
          eff["DCP"] == "2" and src["DCP"] == "variant")
    check("context serves the qualified binary 512K envelope",
          eff["MAX_MODEL_LEN"] == 524288)
    check("the lightweight trellis draft uses MTP-5", eff["MTP_TOKENS"] == 5)
    check("calibrated NVFP4 MLA KV", eff["KV_CACHE_DTYPE"] == "nvfp4_ds_mla")
    check("the r9-qualified dynamic scale ABI is the flagship default",
          eff["KV_SCALE_MODE"] == "dynamic-token")
    check("KV pool is auto-profiled", eff["GPU_BLOCKS_OVERRIDE"] == 0)
    check("the qualified safetensors runtime margin uses GMU 0.957",
          eff["GPU_MEMORY_UTILIZATION"] == 0.957)
    memory_finding = next(
        f for f in gc.validate(eff) if f["id"] == "gpu-util-high")
    check("the measured default gets the qualified high-utilization warning",
          "522,360" in memory_finding["message"]
          and "safetensors" in memory_finding["message"],
          memory_finding["message"])
    check("the agentic profile reserves half of host DRAM as L2 prefix cache",
          eff["OFFLOAD_FRACTION"] == 0.5)
    check("the qualified profile selects supervised LMCache",
          eff["PREFIX_CACHE_BACKEND"] == "lmcache")
    check("vision is an opt-in feature profile", eff["VISION"] is False)
    check("the measured MTP5 profile uses probabilistic drafting",
          eff["MTP_DRAFT_SAMPLE_METHOD"] == "probabilistic")
    check("the qualified profile keeps standard rejection sampling",
          eff["MTP_REJECTION_SAMPLE_METHOD"] == "standard")
    check("lossless B12X PCIe DMA is on with the measured DCP2 prefetch policy",
          eff["B12X_PCIE_DMA"] is True and eff["F8_DMA"] == "0"
          and eff["PCIE_CALIBRATION"] == "auto"
          and eff["DCP_CKV_PREFETCH_DEPTH"] == "0")
    check("full-CKV gather covers the measured 128K prefill matrix",
          eff["DCP_CKV_GATHER_MAX_TOKENS"] == 140000)
    check("served name GLM-5.2", eff["SERVED_MODEL_NAME"] == "GLM-5.2")

    args = gc.family_serve_args(eff)
    line = " ".join(args)
    for flag in ("--quantization exl3",
                 "--decode-context-parallel-size 2",
                 "--dcp-comm-backend a2a",
                 "--dcp-kv-cache-interleave-size 1",
                 "--attention-backend B12X_MLA_SPARSE",
                 "--moe-backend b12x",
                 "--load-format safetensors",
                 "--enable-auto-tool-choice",
                 "--tool-call-parser glm47",
                 "--reasoning-parser glm45"):
        check(f"serve args still carry {flag}", flag in line, line[:200])
    check("hf_overrides still carries the sparse-indexer pattern",
          "index_topk_pattern" in line and "FFFSSS" in line)
    check("hf_overrides still sets use_index_cache", '"use_index_cache":true' in line)
    check("GLM RoPE tables are clamped to the served context",
          '"max_position_embeddings":524288' in line)
    unclamped, _, _ = resolved(gpus=4, CLAMP_ROPE_TABLES=False)
    check("the pre-clamp control can omit the RoPE override for a clean A/B",
          "max_position_embeddings" not in " ".join(
              gc.family_serve_args(unclamped)))
    check("compilation config still has custom_ops + fuse_allreduce_rms",
          '"custom_ops":["all"]' in line and '"fuse_allreduce_rms":true' in line)
    check("capture sizes are substituted from the knob",
          '"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]'
          in line, line[-250:])
    d = gc.derive(eff)
    check("the GLM env block is selected", d["FAMILY_ENV_BLOCK"] == "glm52")
    check("spec method is mtp", d["SPEC_METHOD"] == "mtp")
    check("the pinned checkpoint uses its native rank-sliced TR3 MTP78",
          d["MTP78_MODE"] == "off" and eff["MTP_DRAFT"] == "native")
    profile_env = dict(item.split("=", 1) for item in d["PROFILE_RUNTIME_ENV"])
    for key, expected in (
            ("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "1"),
            ("VLLM_DCP_TOPK_OWNER_MERGE", "0"),
            ("VLLM_EXL3_PREFILL_CHUNK", "128"),
            ("VLLM_EXL3_PREFILL_BLOCK_M", "64"),
            ("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "16"),
            ("VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "1024")):
        check(f"the balanced route bundle sets {key}",
              profile_env.get(key) == expected, str(profile_env))
    check("repo unchanged", d["MODEL_REPO"] == "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw")
    check("checkpoint revision is immutable",
          d["MODEL_REVISION"] == "9297b9f1d53af5c67cffa01e30cc071a1ff7144b")




def test_glm_max_context_profile():
    section("the Brandon DCP4 maximum-context profile")
    eff, src, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-max-context")
    for key, expected in (
            ("DCP", "4"),
            ("MAX_MODEL_LEN", 524288),
            ("MAX_NUM_BATCHED_TOKENS", 4096),
            ("MTP_TOKENS", 5),
            ("DCP_CKV_GATHER_MAX_TOKENS", 140000),
            ("DCP_KV_CACHE_INTERLEAVE_SIZE", "64"),
            ("GPU_MEMORY_UTILIZATION", 0.98),
            ("GPU_BLOCKS_OVERRIDE", 0),
            ("OFFLOAD_FRACTION", 0.5),
            ("MAX_CUDAGRAPH_CAPTURE_SIZE", 64),
            ("VLLM_EXL3_TRELLIS_MAX_M", 64)):
        check(f"the maximum-context profile sets {key}",
              eff[key] == expected and src[key] == "variant",
              f"{eff[key]} / {src[key]}")
    check("the maximum-context profile uses the pinned checkpoint-native draft",
          eff["MTP_DRAFT"] == "native")
    check("the maximum-context profile is retrieval-qualified",
          gc.VARIANTS["exl3-tr3-max-context"]["tested"] is True)
    check("the profile has no balanced DCP2 route overrides",
          gc.derive(eff)["PROFILE_RUNTIME_ENV"] == [])
    check("the profile is accepted without a DCP capacity warning",
          "dcp-reduces-pool" not in ids(gc.validate(eff)))
    line = " ".join(gc.family_serve_args(eff))
    check("the profile serves with DCP4 and interleave64",
          "--decode-context-parallel-size 4" in line
          and "--dcp-kv-cache-interleave-size 64" in line)


def test_madeby561_hybrid():
    section("the MadeBy561 v20 qualified profile")
    eff, src, _ = resolved(gpus=4, MODEL_VARIANT="madeby561-hybrid")
    check("the variant selects the published hybrid checkpoint",
          gc.derive(eff)["MODEL_REPO"]
          == "madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid")
    check("the variant pins the audited release bundle",
          gc.derive(eff)["MODEL_REVISION"]
          == "66f3623dd8fefb5ca8046706912d5d31c8d196af")
    check("the variant chooses its native serialized NVFP4 MTP draft by default",
          eff["MTP_DRAFT"] == "native" and src["MTP_DRAFT"] == "variant",
          f"{eff['MTP_DRAFT']} / {src['MTP_DRAFT']}")
    for key, expected in (
            ("MAX_NUM_BATCHED_TOKENS", 2048),
            ("DCP_PREFILL_WORKSPACE_MIB", 512),
            ("GPU_MEMORY_UTILIZATION", 0.98),
            ("GPU_BLOCKS_OVERRIDE", 2048),
            ("MTP_DRAFT_SAMPLE_METHOD", "probabilistic"),
            ("DCP_CKV_PREFETCH_DEPTH", "0"),
            ("DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS", 8192),
            ("F8_DMA", "ring"),
            ("PCIE_CALIBRATION", "off"),
            ("PCIE_DMA_MIN_BYTES", 393216)):
        check(f"the measured hybrid memory shape sets {key}",
              eff[key] == expected and src[key] == "variant",
              f"{eff[key]} / {src[key]}")
    check("the maximum-context ring profile is flagged tested",
          gc.VARIANTS["madeby561-hybrid"]["tested"] is True)
    check("the exact 521K-qualified ring shape does not warn new users",
          "compressed-dma-needs-retrieval" not in ids(gc.validate(eff)))
    check("MTP3 remains enabled", eff["MTP_TOKENS"] == 3)
    check("standard rejection sampling remains the production control",
          eff["MTP_REJECTION_SAMPLE_METHOD"] == "standard")
    line = " ".join(gc.family_serve_args(eff))
    check("the native hybrid quantizer is selected",
          "--quantization nvfp4_nf3_hybrid" in line, line[:300])
    check("the published MXFP8 dense/shared-expert overlay is selected",
          "--quantization-config" in line
          and '"shared_experts":{"weight":"mxfp8"}' in line, line[-500:])
    check("the native draft needs no graft or external quantizer",
          gc.derive(eff)["MTP78_MODE"] == "off"
          and gc.derive(eff)["DRAFT_QUANTIZATION"] == "")
    check("the old MTP78_MODE=off spelling migrates to neutral native terminology",
          gc.env_layer({"MTP78_MODE": "off"})["MTP_DRAFT"] == "native")
    check("the explicit bf16 spelling remains accepted for old templates",
          gc.env_layer({"MTP_DRAFT": "bf16"})["MTP_DRAFT"] == "bf16")
    explicit, explicit_src, _ = resolved(
        gpus=4, MODEL_VARIANT="madeby561-hybrid", MTP_DRAFT="off")
    check("an explicit draft choice still wins",
          explicit["MTP_DRAFT"] == "off"
          and explicit_src["MTP_DRAFT"] == "file")
    overridden, overridden_src, _ = resolved(
        gpus=4, MODEL_VARIANT="madeby561-hybrid",
        MAX_NUM_BATCHED_TOKENS=2048, GPU_MEMORY_UTILIZATION=0.97,
        GPU_BLOCKS_OVERRIDE=2304)
    check("explicit memory choices still win over the variant profile",
          overridden["MAX_NUM_BATCHED_TOKENS"] == 2048
          and overridden["GPU_MEMORY_UTILIZATION"] == 0.97
          and overridden["GPU_BLOCKS_OVERRIDE"] == 2304
          and all(overridden_src[k] == "file" for k in (
              "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION",
              "GPU_BLOCKS_OVERRIDE")))


def test_higher_fidelity_exl3_candidate():
    section("the higher-fidelity EXL3 3.25bpw candidate")
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.25bpw")
    derived = gc.derive(eff)
    check("the candidate pins willfalco's immutable checkpoint",
          derived["MODEL_REPO"] ==
          "willfalco/GLM-5.2-EXL3-TR3-3.25bpw"
          and derived["MODEL_REVISION"] ==
          "d7d79c2d14599dfce7a5d12b85f7ad73f40e623d")
    check("layer 78 remains the checkpoint-native rank-sliced Trellis draft",
          eff["MTP_DRAFT"] == "native"
          and derived["NATIVE_MTP_FORMAT"] == "exl3-tr3"
          and derived["MTP78_MODE"] == "off")
    profile_env = dict(item.split("=", 1)
                       for item in derived["PROFILE_RUNTIME_ENV"])
    check("mixed-K preparation uses the qualified allocator contract",
          profile_env["PYTORCH_CUDA_ALLOC_CONF"] ==
          "expandable_segments:True"
          and profile_env["SAFETENSORS_FAST_GPU"] == "1")
    check("the qualified mixed-K profile selects DCP4 routes",
          eff["DCP"] == "4"
          and eff["DCP_KV_CACHE_INTERLEAVE_SIZE"] == "64"
          and profile_env["VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE"] == "0"
          and profile_env["VLLM_DCP_TOPK_OWNER_MERGE"] == "1"
          and profile_env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1")
    check("the profile pins exactly one binary-512K request",
          eff["MAX_MODEL_LEN"] == 524288
          and eff["GPU_BLOCKS_OVERRIDE"] == 2048)
    check("MTP3 and C8 remain inside the measured capture window",
          eff["MTP_TOKENS"] == 3
          and eff["MAX_NUM_SEQS"] == 8
          and eff["MAX_CUDAGRAPH_CAPTURE_SIZE"] == 32
          and eff["VLLM_EXL3_TRELLIS_MAX_M"] == 32)
    check("the full-context profile selects the qualified 125 GiB LMCache tier",
          eff["MAX_NUM_BATCHED_TOKENS"] == 2048
          and eff["VLLM_EXL3_PREFILL_CAPACITY"] == 1024
          and eff["OFFLOAD_FRACTION"] == 0.5
          and eff["PREFIX_CACHE_BACKEND"] == "lmcache"
          and eff["PREFIX_CACHE_DISK_GB"] == 0)
    check("the 522K/API/performance gate promotes the variant",
          "variant-untested" not in ids(gc.validate(eff)))
    unsafe_chunk, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-3.25bpw",
        MAX_NUM_BATCHED_TOKENS=3072)
    check("the configurator refuses the measured 3,072-token offload OOM shape",
          "mixed-325-offload-headroom" in ids(gc.validate(unsafe_chunk)))
    impossible_arena, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-3.25bpw",
        VLLM_EXL3_PREFILL_CAPACITY=3072)
    check("the configurator rejects an EXL3 arena above the scheduler chunk",
          "exl3-prefill-capacity-above-scheduler"
          in ids(gc.validate(impossible_arena)))


def test_r20_336_online_quant_candidate():
    section("the r20-qualified 3.36bpw online-K6 profile")
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.36bpw")
    derived = gc.derive(eff)
    check("the profile pins the complete immutable 3.36bpw checkpoint",
          derived["MODEL_REPO"] ==
          "willfalco/GLM-5.2-EXL3-TR3-3.36bpw"
          and derived["MODEL_REVISION"] ==
          "8d9aa923a17502675ca23737349b67f2e66bb69d")
    line = " ".join(gc.family_serve_args(eff))
    runtime = dict(item.split("=", 1)
                   for item in derived["PROFILE_RUNTIME_ENV"])
    check("the measured profile promotes online K6 and exact 512K GPU KV",
          eff["ONLINE_QUANT"] == "exl3-b6"
          and eff["GPU_BLOCKS_OVERRIDE"] == 2048
          and eff["MAX_MODEL_LEN"] == 524288
          and eff["MAX_NUM_BATCHED_TOKENS"] == 3072
          and eff["VLLM_EXL3_PREFILL_CAPACITY"] == 3072
          and "--quantization-config" in line)
    check("the measured LMCache and transport posture is explicit",
          eff["PREFIX_CACHE_BACKEND"] == "lmcache"
          and eff["OFFLOAD_FRACTION"] == 0.5
          and eff["PREFIX_CACHE_DISK_GB"] == 0
          and eff["PCIE_DMA_MIN_BYTES"] == -1
          and runtime["NCCL_BUFFSIZE"] == "1048576"
          and runtime["VLLM_DCP_A2A_MAX_TOKENS"] == "48"
          and runtime["VLLM_DCP_TOPK_OWNER_MERGE"] == "0"
          and runtime["VLLM_EXL3_PREFILL_BLOCK_M"] == "32"
          and runtime["PYTORCH_CUDA_ALLOC_CONF"] ==
              "expandable_segments:False")
    k6, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.36bpw",
                        ONLINE_QUANT="exl3-b6")
    k6_line = " ".join(gc.family_serve_args(k6))
    k6_env = dict(item.split("=", 1)
                  for item in gc.derive(k6)["PROFILE_RUNTIME_ENV"])
    check("K6 uses the exact r20 helper online overlay",
          "--quantization exl3" in k6_line
          and '"shared_experts":{"weight":"mxfp8"}' in k6_line
          and k6_env["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == "6"
          and k6_env["VLLM_EXL3_ONLINE_CACHE_MODE"] == "readwrite"
          and k6_env["VLLM_B12X_ABSORB_BMM"] == "0")
    mxfp8, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.36bpw",
                           ONLINE_QUANT="mxfp8")
    mx_line = " ".join(gc.family_serve_args(mxfp8))
    mx_env = dict(item.split("=", 1)
                  for item in gc.derive(mxfp8)["PROFILE_RUNTIME_ENV"])
    check("MXFP8 preserves the published conservative tensor exclusions",
          '"linear":{"weight":"mxfp8"}' in mx_line
          and "q_a_proj" in mx_line and "kv_a_proj_with_mqa" in mx_line
          and "lm_head" in mx_line
          and mx_env["VLLM_B12X_ABSORB_BMM"] == "1")
    hybrid, _, _ = resolved(gpus=4, MODEL_VARIANT="madeby561-hybrid",
                            ONLINE_QUANT="mxfp8")
    check("a competing online overlay is refused on a non-EXL3 quantizer",
          "online-quant-needs-exl3" in errs(gc.validate(hybrid)))

    unpinned, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-3.36bpw",
        GPU_BLOCKS_OVERRIDE=0)
    check("removing the pool pin reports the observed first-request OOM",
          "mixed-336-offload-needs-pool-pin"
          in ids(gc.validate(unpinned)))


def test_r28_342_shared_h_profile():
    section("the r28-qualified shared-H 3.42bpw profile")
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.42bpw")
    derived = gc.derive(eff)
    check("the profile pins the immutable 3.42bpw checkpoint",
          derived["MODEL_REPO"] ==
          "willfalco/GLM-5.2-EXL3-TR3-3.42bpw"
          and derived["MODEL_REVISION"] ==
          "a350292cb2038f2c31732569a711a89e5d72fd46")
    check("the profile pins the workload-safe 520,192-token context",
          eff["GPU_BLOCKS_OVERRIDE"] == 0
          and eff["GPU_MEMORY_UTILIZATION"] == 0.93
          and eff["MAX_MODEL_LEN"] == 520192
          and eff["KV_CACHE_MEMORY_BYTES"] == 4518907904)
    check("the profile retains K6, dynamic NVFP4 and production LMCache",
          eff["ONLINE_QUANT"] == "exl3-b6"
          and eff["KV_CACHE_DTYPE"] == "nvfp4_ds_mla"
          and eff["MTP_DRAFT_SAMPLE_METHOD"] == "probabilistic"
          and eff["PREFIX_CACHE_BACKEND"] == "lmcache"
          and eff["PREFIX_CACHE_DISK_GB"] == 0)

def test_glm53_342_dsa_profile():
    section("the qualified GLM-5.3 glm_moe_dsa 3.42bpw profile")
    variant = "exl3-tr3-glm53-3.42bpw"
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT=variant)
    derived = gc.derive(eff)
    check("GLM-5.3 reuses the architecture-compatible GLM-5.2 runtime family",
          eff["MODEL_FAMILY"] == "glm52"
          and gc.VARIANTS[variant]["family"] == "glm52")
    check("the profile pins davidsyoung's immutable checkpoint",
          derived["MODEL_REPO"] ==
          "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw"
          and derived["MODEL_REVISION"] ==
          "8bef807a0fcdd180e984a26b50e731cdba9a8ff2")
    check("the profile pins the live-qualified 393,216-token memory boundary",
          eff["MAX_MODEL_LEN"] == 393216
          and eff["GPU_MEMORY_UTILIZATION"] == 0.93
          and eff["GPU_BLOCKS_OVERRIDE"] == 0
          and eff["KV_CACHE_MEMORY_BYTES"] == 3415867392)
    check("the profile retains mixed-K online K6, native MTP3, and dynamic KV",
          eff["ONLINE_QUANT"] == "exl3-b6"
          and eff["MTP_DRAFT"] == "native"
          and eff["MTP_TOKENS"] == 3
          and eff["MTP_DRAFT_SAMPLE_METHOD"] == "probabilistic"
          and eff["KV_CACHE_DTYPE"] == "nvfp4_ds_mla"
          and eff["KV_SCALE_MODE"] == "dynamic-token")
    check("the profile retains the measured C8 scheduler shape",
          eff["MAX_NUM_SEQS"] == 8
          and eff["MAX_NUM_BATCHED_TOKENS"] == 3072)
    check("the profile serves under the GLM-5.3 model name",
          eff["SERVED_MODEL_NAME"] == "GLM-5.3")
    args = " ".join(gc.family_serve_args(eff))
    check("the profile keeps the measured DCP4 sparse-MLA serve path",
          "--decode-context-parallel-size 4" in args
          and "--attention-backend B12X_MLA_SPARSE" in args
          and "--moe-backend b12x" in args)
    findings = gc.validate(eff)
    check("the profile is live-qualified",
          gc.VARIANTS[variant]["tested"]
          and "variant-untested" not in ids(findings), str(findings))
    check("the profile refuses unsafe inherited memory and GLM-5.2 grafts",
          derived["MTP_GRAFT_COMPATIBLE"] == "0"
          and "mtp-graft-incompatible" in errs(gc.validate(
              {**eff, "MTP_DRAFT": "tr3-graft"}))
          and "glm53-qualified-envelope" in errs(gc.validate(
              {**eff, "MAX_MODEL_LEN": 520192}))
          and "glm53-qualified-envelope" in errs(gc.validate(
              {**eff, "KV_CACHE_MEMORY_BYTES": 0})))
    external_draft = {
        **eff,
        "MTP_DRAFT": "tr3-graft",
        "DRAFT_MODEL": "/models/complete-external-draft",
    }
    external_findings = gc.validate(external_draft)
    check("a complete external draft overrides incompatible graft preparation",
          "mtp-graft-incompatible" not in errs(external_findings)
          and "draft-model-overrides" in ids(external_findings)
          and gc.derive(external_draft)["MTP78_MODE"] == "off",
          str(external_findings))



def test_known_good_replays_across_profile_env_change():
    section("known-good rollback replay across container profile changes")
    old_env = {
        "GLM_GPU_COUNT": "4",
        "MODEL_FAMILY": "glm52",
        "MODEL_VARIANT": "exl3-tr3",
    }
    new_env = {
        "GLM_GPU_COUNT": "4",
        "MODEL_FAMILY": "glm52",
        "MODEL_VARIANT": "exl3-tr3-glm53-3.42bpw",
    }
    old_effective, _sources, _notes = gc.resolve(
        state_values={}, env_values=old_env)
    original = gc.load_startup_env
    gc.load_startup_env = lambda: dict(new_env)
    try:
        replay, error = config_cli._known_good_replay({
            "values": {},
            "effective": old_effective,
        })
    finally:
        gc.load_startup_env = original
    restored, _sources, _notes = gc.resolve(
        state_values=replay or {}, env_values=new_env)
    check("an env-only known-good becomes a replayable state diff",
          not error
          and replay.get("MODEL_VARIANT") == "exl3-tr3"
          and not gc.diff(old_effective, restored),
          error or gc.diff_text(old_effective, restored))

    qwen_old_env = {
        "GLM_GPU_COUNT": "1",
        "MODEL_FAMILY": "qwen36",
        "MODEL_VARIANT": "qwen36-27b-nvfp4",
    }
    qwen_new_env = {**qwen_old_env, "GLM_GPU_COUNT": "4"}
    qwen_effective, _sources, _notes = gc.resolve(
        state_values={}, env_values=qwen_old_env)
    gc.load_startup_env = lambda: dict(qwen_new_env)
    try:
        qwen_replay, qwen_error = config_cli._known_good_replay({
            "values": {},
            "effective": qwen_effective,
        })
    finally:
        gc.load_startup_env = original
    qwen_restored, _sources, _notes = gc.resolve(
        state_values=qwen_replay or {}, env_values=qwen_new_env)
    check("known-good exact replay ignores target-family inapplicable knobs",
          not qwen_error
          and qwen_effective["DCP"] == "1"
          and qwen_restored["DCP"] == "4"
          and not gc.applies_to(gc.KNOB_BY_KEY["DCP"], "qwen36"),
          qwen_error)

    gc.write_json_atomic(gc.p_state(), {
        "values": {},
        "written_at": gc.utcnow_iso(),
    }, mode=0o600)
    gc.write_json_atomic(gc.p_known_good(), {
        "ts": "old-profile",
        "values": {},
        "effective": old_effective,
    }, mode=0o600)
    for path in (gc.p_restart_flag(),):
        try:
            os.remove(path)
        except OSError:
            pass
    gc.load_startup_env = lambda: dict(new_env)
    original_now = config_cli._now
    config_cli._now = lambda: "rollback-replay-test"
    try:
        rc = config_cli._cmd_rollback_locked(
            SimpleNamespace(log="", reason="profile changed"))
        persisted = gc.load_state_file()
    finally:
        gc.load_startup_env = original
        config_cli._now = original_now
    restored, _sources, _notes = gc.resolve(
        state_values=persisted, env_values=new_env)
    check("rollback persists the replay diff and requests one restart",
          rc == 0
          and persisted.get("MODEL_VARIANT") == "exl3-tr3"
          and not gc.diff(old_effective, restored)
          and os.path.exists(gc.p_restart_flag()),
          f"rc={rc}; {gc.diff_text(old_effective, restored)}")

    one_gpu_env = dict(new_env)
    one_gpu_env["GLM_GPU_COUNT"] = "1"
    try:
        os.remove(gc.p_restart_flag())
    except OSError:
        pass
    state_before_refusal = gc.load_state_file()
    gc.load_startup_env = lambda: dict(one_gpu_env)
    config_cli._now = lambda: "rollback-hardware-refusal"
    try:
        refused_replay, refusal = config_cli._known_good_replay({
            "values": {},
            "effective": old_effective,
        })
        refused_rc = config_cli._cmd_rollback_locked(
            SimpleNamespace(log="", reason="host topology changed"))
        state_after_refusal = gc.load_state_file()
    finally:
        gc.load_startup_env = original
        config_cli._now = original_now
    check("rollback refuses a known-good topology that exceeds current GPUs",
          refused_replay is None
          and "tp-exceeds-gpus" in refusal
          and refused_rc == 3
          and state_after_refusal == state_before_refusal
          and not os.path.exists(gc.p_restart_flag()),
          f"error={refusal!r}; rc={refused_rc}; state={state_after_refusal!r}")
    for path in (gc.p_state(), gc.p_known_good(), gc.p_restart_flag()):
        try:
            os.remove(path)
        except OSError:
            pass


def test_flash_is_refused_not_substituted():
    section("GLM-5.3-Flash is refused, with the image that serves it named")
    check("only the Gilded-servable families are selectable",
          list(gc.FAMILIES) == ["glm52", "qwen36", "custom"],
          str(list(gc.FAMILIES)))
    check("no Flash checkpoint variant is selectable",
          not [n for n in gc.VARIANTS if n.startswith("glm53-")],
          str(list(gc.VARIANTS)))
    check("the full GLM-5.3/GLM-5.2 profiles and the other families remain",
          {"exl3-tr3", "exl3-tr3-3.25bpw", "exl3-tr3-3.36bpw",
           "exl3-tr3-3.42bpw", "exl3-tr3-glm53-3.42bpw",
           "exl3-tr3-glm53-3.25bpw", "exl3-tr3-glm53-3.42bpw-500k",
           "qwen36-nvfp4", "custom"} <= set(gc.VARIANTS),
          str(sorted(gc.VARIANTS)))
    check("no knob is still scoped to the withdrawn Flash family",
          not [k["key"] for k in gc.KNOBS
               if "glm53" in (k.get("families") or ())],
          str([k["key"] for k in gc.KNOBS
               if "glm53" in (k.get("families") or ())]))
    # The refusal must not be a ConfigError: every ConfigError in the resolver
    # degrades to the layer below, which is precisely the silent substitution
    # this rule exists to prevent.
    check("the refusal is not the exception type the resolver swallows",
          not issubclass(gc.FlashProfileUnavailable, gc.ConfigError))
    for key, value in (("MODEL_FAMILY", "glm53"),
                       ("MODEL_VARIANT", "glm53-k6"),
                       ("MODEL_VARIANT", "glm53-k8")):
        for layer in ("state_values", "env_values"):
            kwargs = {"state_values": {}, "env_values": {}}
            kwargs[layer] = {key: value}
            message = ""
            try:
                gc.resolve(**kwargs)
            except gc.FlashProfileUnavailable as exc:
                message = str(exc)
            check(f"{key}={value} from the {layer} layer fails closed",
                  bool(message), "resolve() returned a substitute instead")
            check(f"and the {key} refusal names what is missing and where "
                  "Flash is served",
                  key in message and value in message
                  and "GLM5Next" in message
                  and "Gilded Gnosis" in message
                  and gc.FLASH_IMAGE in message, message)
    check("the refusal points at the full GLM-5.3 profile as the alternative",
          "glm53-3.42bpw-500k" in gc.flash_unavailable_message("glm53-k6"))
    check("a Flash selector that is not requested changes nothing",
          gc.resolve(state_values={}, env_values={})[0]["MODEL_FAMILY"]
          == "glm52")


def test_qwen_preset():
    section("the Qwen preset")
    eff, src, _ = resolved("qwen36", gpus=1)
    check("TP is the detected GPU count, so a 1-GPU host just works",
          eff["TENSOR_PARALLEL_SIZE"] == 1 and src["TENSOR_PARALLEL_SIZE"] == "detected",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    eff8, src8, _ = resolved("qwen36", gpus=8)
    check("and on an 8-GPU host it uses all 8 rather than leaving 7 idle",
          eff8["TENSOR_PARALLEL_SIZE"] == 8)
    check("the qualified RTX 5090 profile exposes a 192 Ki-token context",
          eff["MAX_MODEL_LEN"] == 196608)
    check("MTP is opt-in for the production profile", eff["MTP_TOKENS"] == 0)
    check("the qualified GPU utilization is selected",
          eff["GPU_MEMORY_UTILIZATION"] == 0.90)
    check("KV dtype is auto, not the MLA fp8 default", eff["KV_CACHE_DTYPE"] == "auto")
    check("the KV pool pin is dropped", eff["GPU_BLOCKS_OVERRIDE"] == 0)
    check("variant follows the family", eff["MODEL_VARIANT"] == "qwen36-nvfp4")

    d = gc.derive(eff)
    line = " ".join(d["FAMILY_SERVE_ARGS"])
    check("repo is the NVFP4 development checkpoint",
          d["MODEL_REPO"] == "nvidia/Qwen3.6-27B-NVFP4")
    check("Qwen checkpoint revision is immutable",
          d["MODEL_REVISION"] ==
          "0893e1606ff3d5f97a441f405d5fc541a6bdf404")
    check("ModelOpt quantization is selected", "--quantization modelopt" in line, line)
    for flag in ("--decode-context-parallel-size", "--dcp-comm-backend",
                 "--dcp-kv-cache-interleave-size", "B12X_MLA_SPARSE",
                 "--moe-backend", "index_topk_pattern", "use_index_cache",
                 "glm47", "glm45", "reasoning_effort", "custom_ops",
                 "fuse_allreduce_rms"):
        check(f"no GLM-ism leaks through: {flag}", flag not in line, line[:200])
    check("the card's reasoning parser is set", "--reasoning-parser qwen3" in line)
    check("the qualified Qwen tool parser is selected",
          "--tool-call-parser qwen3_coder" in line and
          "--enable-auto-tool-choice" in line)
    check("the production profile enables its native vision path",
          "--language-model-only" not in line)
    check("the profile caps native vision near a 4K working image",
          '--mm-processor-kwargs {"max_pixels":8388608}' in line, line)
    check("the default compiled Qwen profile does not force eager mode",
          "--enforce-eager" not in line)
    mtp_eff, _, _ = resolved("qwen36", gpus=1, MTP_TOKENS=2)
    mtp_line = " ".join(gc.family_serve_args(mtp_eff))
    check("Qwen MTP2 uses the live-qualified eager compatibility path",
          "--enforce-eager" in mtp_line, mtp_line)
    mtp_findings = gc.validate(mtp_eff)
    check("the configurator explains Qwen MTP's eager-mode tradeoff",
          "qwen-mtp-eager" in ids(mtp_findings),
          str(ids(mtp_findings)))
    fixed_eff, _, _ = resolved(
        "qwen36", gpus=1, KV_CACHE_MEMORY_BYTES=5_500_000_000)
    fixed_args = gc.family_serve_args(fixed_eff)
    check("a fixed KV byte budget reaches vLLM as two intact argv entries",
          ["--kv-cache-memory-bytes", "5500000000"] ==
          fixed_args[fixed_args.index("--kv-cache-memory-bytes"):
                     fixed_args.index("--kv-cache-memory-bytes") + 2],
          str(fixed_args))
    fixed_findings = gc.validate(fixed_eff)
    check("the configurator explains that fixed KV supersedes GMU sizing",
          "fixed-kv-cache" in ids(fixed_findings), str(ids(fixed_findings)))
    conflict_eff, _, _ = resolved(
        "qwen36", gpus=1, KV_CACHE_MEMORY_BYTES=5_500_000_000,
        GPU_BLOCKS_OVERRIDE=100)
    check("fixed KV bytes and a block-count override are rejected together",
          "fixed-kv-block-conflict" in errs(gc.validate(conflict_eff)))
    check("the generic env block is selected", d["FAMILY_ENV_BLOCK"] == "generic")
    check("spec method uses vLLM's current mtp name", d["SPEC_METHOD"] == "mtp")
    check("the MTP78 apparatus is forced off", d["MTP78_MODE"] == "off")
    check("no draft dir or draft quantization is derived",
          d["DRAFT_MODEL"] == "" and d["DRAFT_QUANTIZATION"] == "")
    check("vision is forced off", d["VISION"] is False)
    check("the full-size text profile is flagged live-qualified",
          gc.family("qwen36")["tested"] is True
          and gc.VARIANTS["qwen36-nvfp4"]["tested"] is True)


def test_custom_profile():
    section("the custom profile")
    missing, _, _ = resolved("custom", gpus=1)
    check("a custom profile without MODEL_ID is refused",
          "custom-model-required" in errs(gc.validate(missing)))
    eff, _, _ = resolved(
        "custom", gpus=1, MODEL_ID="Qwen/Qwen3.5-0.8B",
        QUANTIZATION="modelopt", REASONING_PARSER="qwen3",
        TOOL_CALL_PARSER="qwen3_coder", MULTIMODAL=False)
    d = gc.derive(eff)
    line = " ".join(d["FAMILY_SERVE_ARGS"])
    check("the requested repository is preserved",
          d["MODEL_REPO"] == "Qwen/Qwen3.5-0.8B")
    check("its directory is isolated under models/",
          d["MODEL_DIRNAME"] == "models/Qwen-Qwen3.5-0.8B")
    check("custom quantization is wired", "--quantization modelopt" in line, line)
    check("custom reasoning is wired", "--reasoning-parser qwen3" in line, line)
    check("custom tools are wired",
          "--enable-auto-tool-choice" in line and
          "--tool-call-parser qwen3_coder" in line, line)
    check("text-only mode is wired", "--language-model-only" in line, line)
    check("GLM flags stay absent",
          not any(x in line for x in ("B12X_MLA_SPARSE", "--dcp-comm-backend",
                                      "--quantization exl3")), line)


def test_inapplicable_knobs():
    section("knobs that cannot apply are refused, not ignored")
    eff, src, _ = resolved("qwen36")
    for key in ("MTP_DRAFT", "DRAFT_MODEL", "DRAFT_QUANTIZATION", "DCP",
                "DCP_CKV_GATHER_MAX_TOKENS", "DCP_KV_CACHE_INTERLEAVE_SIZE",
                "MTP_DRAFT_SAMPLE_METHOD", "MTP_REJECTION_SAMPLE_METHOD",
                "VLLM_EXL3_TRELLIS_MAX_M",
                "VISION", "VISION_CHUNKS"):
        check(f"{key} is marked n/a on qwen36", src[key] == "n/a", src[key])
        check(f"{key} is applicable on glm52",
              gc.applies_to(gc.KNOB_BY_KEY[key], "glm52"))

    # a hand-edited state file that carries one is an ERROR naming the key
    f = gc.validate(eff, {"state_keys": ["DCP", "MTP_TOKENS"]})
    check("a persisted inapplicable knob is an error",
          "knob-inapplicable" in errs(f), str(errs(f)))
    msg = [x["message"] for x in f if x["id"] == "knob-inapplicable"][0]
    check("and the message names the key", "'DCP'" in msg, msg)
    check("an applicable knob in the same file is not flagged",
          len([x for x in f if x["id"] == "knob-inapplicable"]) == 1)

    # REGRESSION: minimize() must PERSIST the chosen family. Comparing the
    # family against a baseline built from that same family is circular — it
    # always matched, was never written, and every apply silently fell back to
    # the previous family, taking its validation rules with it.
    out = gc.minimize({"MODEL_FAMILY": "qwen36", "MTP_TOKENS": 4})
    check("the selected family is written to the state file",
          out.get("MODEL_FAMILY") == "qwen36", str(out))
    check("a knob left at the new family's own default is NOT pinned",
          "MAX_MODEL_LEN" not in gc.minimize({"MODEL_FAMILY": "qwen36",
                                              "MAX_MODEL_LEN": 196608}),
          str(gc.minimize({"MODEL_FAMILY": "qwen36", "MAX_MODEL_LEN": 196608})))
    check("but an override of it is",
          gc.minimize({"MODEL_FAMILY": "qwen36",
                       "MAX_MODEL_LEN": 131072}).get("MAX_MODEL_LEN") == 131072)
    check("the default family is not written at all",
          "MODEL_FAMILY" not in gc.minimize({"MODEL_FAMILY": "glm52",
                                             "MTP_TOKENS": 5}))

    # minimize() drops them instead of writing them
    dropped = []
    out = gc.minimize({"MODEL_FAMILY": "qwen36", "DCP": "1", "MTP_TOKENS": 4}, dropped)
    check("minimize drops the inapplicable knob", "DCP" not in out, str(out))
    check("and reports it as dropped rather than silently", dropped == ["DCP"], str(dropped))
    check("while keeping the applicable one", out.get("MTP_TOKENS") == 4, str(out))


def test_rules_are_family_scoped():
    section("the measured GLM rules do not fire on another family")
    # concurrency window: 32 seqs x (1+2) = 96, far outside the trellis window
    eff, _, _ = resolved("qwen36", MAX_NUM_SEQS=32)
    f = gc.validate(eff)
    check("the trellis concurrency rule does not apply to a dense model",
          "concurrency-window" not in ids(f), str(ids(f)))
    check("nor the vision long-context rule", "vision-long-context" not in ids(f))
    check("nor the DCP rules",
          not {"dcp-divides-tp", "dcp-reduces-pool"} & ids(f), str(ids(f)))

    # ... and still fire on GLM
    eff, _, _ = resolved(MAX_NUM_SEQS=32)
    f = gc.validate(eff)
    check("the concurrency rule still fires on GLM",
          "concurrency-window" in errs(f), str(errs(f)))
    eff, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-max-context", VISION=True)
    check("the max-context EXL3 profile cannot evade the vision quality warning",
          "vision-long-context" in ids(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4)
    check("the exact qualified r11 safetensors shape clears the old loader warning",
          "instanttensor-context-margin" not in ids(gc.validate(eff)))
    eff, _, _ = resolved(
        gpus=4, LOAD_FORMAT="instanttensor",
        MAX_MODEL_LEN=524288, MAX_NUM_BATCHED_TOKENS=3200)
    check("changing the qualified 524288 scheduler shape requires requalification",
          "instanttensor-context-margin" in ids(gc.validate(eff)))
    too_small, _, _ = resolved(
        gpus=4, DCP="2", MAX_MODEL_LEN=524288,
        GPU_BLOCKS_OVERRIDE=4095)
    exact, _, _ = resolved(
        gpus=4, DCP="2", MAX_MODEL_LEN=524288,
        GPU_BLOCKS_OVERRIDE=4096)
    check("the pool-pin validator accounts for DCP2 logical capacity",
          "pool-smaller-than-context" in errs(gc.validate(too_small))
          and "pool-smaller-than-context" not in errs(gc.validate(exact)))
    eff, _, _ = resolved(MTP_DRAFT="tr3-override")
    check("v29 accepts the separately rank-sliced EXL3 draft",
          "tr3-draft-on-v20" not in ids(gc.validate(eff)))
    eff, _, _ = resolved(F8_DMA="ring")
    check("compressed DMA demands a new retrieval gate",
          "compressed-dma-needs-retrieval" in ids(gc.validate(eff)))


def test_family_coherence_rules():
    section("family coherence")
    eff, _, _ = resolved("qwen36", MODEL_VARIANT="exl3-tr3")
    check("a GLM variant on the Qwen family is an error",
          "variant-family-mismatch" in errs(gc.validate(eff)))
    eff, _, _ = resolved("qwen36", KV_CACHE_DTYPE="nvfp4_ds_mla")
    f = gc.validate(eff)
    check("an MLA KV layout on a non-MLA family is an error",
          "kv-dtype-family" in errs(f))
    msg = [x["message"] for x in f if x["id"] == "kv-dtype-family"][0]
    check("and explains that nvfp4_ds_mla is an MLA layout", "MLA KV layout" in msg, msg)
    eff, _, _ = resolved(KV_CACHE_DTYPE="fp8", KV_SCALE_MODE="dynamic-token")
    check("dynamic per-token scaling cannot be paired with fp8 KV",
          "kv-dynamic-dtype" in errs(gc.validate(eff)))
    eff, _, _ = resolved(
        KV_CACHE_DTYPE="nvfp4_ds_mla", KV_SCALE_MODE="dynamic-token")
    check("the complete dynamic NVFP4 mode is accepted as an experiment",
          "kv-dynamic-dtype" not in errs(gc.validate(eff)))
    eff, _, _ = resolved("qwen36")
    check("the live-qualified Qwen family no longer emits the untested warning",
          "family-untested" not in ids(gc.validate(eff)))
    custom, _, _ = resolved("custom")
    check("an untested family still warns",
          "family-untested" in ids(gc.validate(custom)))


def test_gpu_count_gate():
    section("tensor parallel vs the GPUs that exist")
    eff, _, _ = resolved()
    f = gc.validate(eff, {"gpu_count": 1})
    check("GLM at TP=4 on a 1-GPU host is refused", "tp-exceeds-gpus" in errs(f))
    msg = [x["message"] for x in f if x["id"] == "tp-exceeds-gpus"][0]
    check("and the message says how many are needed vs found",
          "needs 4 GPUs and this host has 1" in msg, msg)
    eff, _, _ = resolved("qwen36", gpus=1)
    check("Qwen at TP=1 on a 1-GPU host is fine",
          "tp-exceeds-gpus" not in errs(gc.validate(eff, {"gpu_count": 1})))
    check("and GLM on a 1-GPU host is refused for a better reason than the gate",
          "glm-needs-four-ranks" in errs(gc.validate(resolved(gpus=1)[0],
                                                     {"gpu_count": 1})))
    check("GLM at TP=8 is expressible, with the measured numbers disclaimed",
          "tp-off-measured" in ids(gc.validate(resolved(TENSOR_PARALLEL_SIZE=8)[0])))
    msg = [x["message"] for x in gc.validate(resolved(TENSOR_PARALLEL_SIZE=8)[0])
           if x["id"] == "tp-off-measured"][0]
    check("and that warning says which numbers do NOT move with TP",
          "does NOT move with TP" in msg, msg)
    eff, _, _ = resolved("qwen36", TENSOR_PARALLEL_SIZE=4)
    check("Qwen can still be told to use 4 GPUs",
          not errs(gc.validate(eff, {"gpu_count": 4})), str(errs(gc.validate(eff))))




def test_env_layer_still_wins_over_family():
    section("layering: default < family < variant < env < file")
    env = {"TENSOR_PARALLEL_SIZE": 2, "GLM_GPU_COUNT": "8"}
    eff, src, _ = gc.resolve(state_values={"MODEL_FAMILY": "qwen36"}, env_values=env)
    check("the template env overrides detection",
          eff["TENSOR_PARALLEL_SIZE"] == 2 and src["TENSOR_PARALLEL_SIZE"] == "env",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    eff, src, _ = gc.resolve(state_values={"MODEL_FAMILY": "qwen36",
                                           "TENSOR_PARALLEL_SIZE": 3},
                             env_values=env)
    check("and the state file overrides the env",
          eff["TENSOR_PARALLEL_SIZE"] == 3 and src["TENSOR_PARALLEL_SIZE"] == "file")
    eff, src, _ = gc.resolve(state_values={"MODEL_FAMILY": "qwen36"},
                             env_values={"GLM_GPU_COUNT": "2"})
    check("with neither, the DETECTED count stands",
          eff["TENSOR_PARALLEL_SIZE"] == 2 and src["TENSOR_PARALLEL_SIZE"] == "detected",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    eff, src, _ = gc.resolve(state_values={}, env_values={})
    check("with no detection at all it falls back to the built-in default, and the "
          "entrypoint refuses to boot on 0 GPUs rather than guessing",
          src["TENSOR_PARALLEL_SIZE"] == "default")


def test_shipped_variants_validate_clean():
    section("every shipped tested variant passes its own pre-validation")
    # The regression that motivated this test: madeby561-hybrid's 2048-token
    # scheduler chunk sat below the registry-default 3072-row EXL3 arena, and
    # the EXL3-arena rule (wrongly scoped to all of GLM) refused the
    # documented qualified profile at every apply — and no test asserted the
    # shipped variants resolve error-free.
    for name, variant in gc.VARIANTS.items():
        if not variant.get("tested"):
            continue
        fam = variant.get("family", "glm52")
        gpus = 1 if fam == "qwen36" else 4
        eff, _, _ = resolved(family=fam, gpus=gpus, MODEL_VARIANT=name)
        found = errs(gc.validate(eff, {"gpu_count": gpus}))
        check(f"{name} resolves with zero error-level findings",
              not found, f"{name}: {sorted(found)}")


def test_validation_scoping_fixes():
    section("validation rules stay scoped to the kernel/family they bound")
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-3.25bpw",
                         VLLM_EXL3_PREFILL_CAPACITY=4096)
    check("the EXL3 arena rule still fires on an EXL3 variant",
          "exl3-prefill-capacity-above-scheduler" in errs(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="madeby561-hybrid",
                         VLLM_EXL3_PREFILL_CAPACITY=4096)
    check("the EXL3 arena rule is silent for the NVFP4/NF3 hybrid",
          "exl3-prefill-capacity-above-scheduler"
          not in ids(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, MTP_TOKENS=0, MAX_NUM_SEQS=128)
    check("the decode-width window still applies with MTP off",
          "concurrency-window" in errs(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, MODEL_VARIANT="madeby561-hybrid",
                         MAX_NUM_SEQS=8, VLLM_EXL3_TRELLIS_MAX_M=8)
    check("the trellis window does not bound the hybrid decode width",
          "concurrency-window" not in ids(gc.validate(eff)))
    eff, _, _ = resolved(family="qwen36", gpus=1,
                         GPU_BLOCKS_OVERRIDE=1024, MAX_MODEL_LEN=196608)
    check("a pinned Qwen pool smaller than the context is refused",
          "pool-smaller-than-context" in errs(gc.validate(eff)))
    eff, src, _ = resolved(gpus=8, MODEL_VARIANT="madeby561-hybrid",
                           TENSOR_PARALLEL_SIZE=4)
    check("detected DCP follows the resolved TP, not the raw GPU count",
          eff["DCP"] == "4"
          and "dcp-divides-tp" not in errs(gc.validate(eff, {"gpu_count": 8})),
          f"DCP={eff['DCP']}")
    eff, _, _ = resolved(gpus=6)
    check("a 6-GPU host resolves a legal DCP choice that divides TP",
          eff["DCP"] == "2", f"DCP={eff['DCP']}")
    eff, _, _ = resolved(gpus=4, SERVED_MODEL_NAME="  ")
    check("a blank SERVED_MODEL_NAME is refused",
          "served-name-empty" in errs(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, MTP_DRAFT="nvfp4")
    check("the nvfp4 draft's registry autofill satisfies the quant rule",
          "draft-quant-inherit" not in ids(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, MTP_DRAFT="nvfp4", DRAFT_QUANTIZATION="exl3")
    check("an explicit wrong draft quantization is still refused",
          "draft-quant-inherit" in errs(gc.validate(eff)))
    eff, _, _ = resolved(gpus=4, DRAFT_MODEL='/x/"y')
    check("a quote in DRAFT_MODEL is refused before it corrupts the JSON",
          "draft-model-quoting" in errs(gc.validate(eff)))


def test_full_glm53_candidate_boundaries():
    section("full GLM-5.3 preserves AIBeast's context without claiming qualification")
    eff, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-glm53-3.42bpw-500k")
    findings = gc.validate(eff, {"gpu_count": 4})
    check("the complete candidate is admissible but unqualified",
          not errs(findings) and "variant-untested" in ids(findings), findings)
    check("the served model retains 3.42bpw and AIBeast's full request budget",
          gc.derive(eff)["MODEL_REPO"] ==
          "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw"
          and eff["MAX_MODEL_LEN"] == 520192
          and eff["SERVED_MODEL_NAME"] == "GLM-5.3")
    check("the generic profile retains the exercised rental floor",
          eff["MAX_NUM_SEQS"] == 8
          and eff["MAX_NUM_BATCHED_TOKENS"] == 2048
          and eff["VLLM_EXL3_PREFILL_CAPACITY"] == 1024
          and eff["GPU_MEMORY_UTILIZATION"] == 0.93
          and eff["MAX_CUDAGRAPH_CAPTURE_SIZE"] == 32
          and eff["VLLM_EXL3_TRELLIS_MAX_M"] == 32)
    parity, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-glm53-3.42bpw-500k",
        MAX_NUM_SEQS=12, MAX_NUM_BATCHED_TOKENS=3072,
        VLLM_EXL3_PREFILL_CAPACITY=3072, GPU_MEMORY_UTILIZATION=0.95,
        MAX_CUDAGRAPH_CAPTURE_SIZE=48,
        CUDAGRAPH_CAPTURE_SIZES="4,8,12,16,20,24,28,32,36,40,44,48",
        VLLM_EXL3_TRELLIS_MAX_M=48)
    findings = gc.validate(parity, {"gpu_count": 4})
    check("live GLM-5.2 resource parity is admissible, not GPU-qualified",
          not errs(findings) and "variant-untested" in ids(findings), findings)
    for key, value in {
        "MAX_NUM_SEQS": 13, "MAX_NUM_BATCHED_TOKENS": 3073,
        "VLLM_EXL3_PREFILL_CAPACITY": 3073,
        "MAX_CUDAGRAPH_CAPTURE_SIZE": 52, "VLLM_EXL3_TRELLIS_MAX_M": 52,
        "CUDAGRAPH_CAPTURE_SIZES": "4,8,12,16,20,24,28,32",
    }.items():
        check(f"parity rejects an oversized or incoherent {key}",
              "glm53-candidate-envelope" in errs(gc.validate({**parity, key: value})))
    smaller, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-glm53-3.25bpw")
    for key, value in {"MAX_NUM_BATCHED_TOKENS": 3072,
                       "VLLM_EXL3_PREFILL_CAPACITY": 3072}.items():
        check(f"the 3.25bpw candidate retains its original {key} ceiling",
              "glm53-candidate-envelope" in errs(gc.validate({**smaller, key: value})))
    check("a larger unmeasured envelope is not silently accepted",
          "glm53-candidate-envelope" in errs(gc.validate(
              {**eff, "MAX_MODEL_LEN": 655360}, {"gpu_count": 4})))
    check("an inherited smaller KV pool cannot masquerade as 512K support",
          "glm53-candidate-envelope" in errs(gc.validate(
              {**eff, "KV_CACHE_MEMORY_BYTES": 3415867392})))
    check("the old GLM-5.2 graft cannot modify the full GLM-5.3 checkpoint",
          "mtp-graft-incompatible" in errs(gc.validate(
              {**eff, "MTP_DRAFT": "tr3-graft"})))
    low, _, _ = resolved(
        gpus=4, MODEL_VARIANT="exl3-tr3-glm53-3.42bpw-500k",
        REASONING_EFFORT_DEFAULT="low")
    args = gc.family_serve_args(low)
    kwargs = json.loads(args[args.index("--default-chat-template-kwargs") + 1])
    check("operator reasoning effort reaches the actual serving contract",
          kwargs == {"reasoning_effort": "low"})


def test_prefill_fairness():
    section("opt-in measured-service fairness")
    for variant in gc.VARIANTS:
        family = gc.VARIANTS[variant].get("family", "glm52")
        cfg, _, _ = resolved(family=family, MODEL_VARIANT=variant)
        args = gc.family_serve_args(cfg)
        check(f"{variant} does not implicitly enable fairness",
              "--fairness-engine" not in args and "--prefill-compute-share" not in args)

    baseline, _, _ = resolved(gpus=4, MODEL_VARIANT="exl3-tr3-glm53-3.42bpw-500k")
    enabled = {**baseline, "PREFILL_FAIRNESS_ENGINE": "compute_share"}
    args = gc.family_serve_args(enabled)
    fairness_index = args.index("--fairness-engine")
    check("enabling fairness adds only its two CLI controls",
          args[fairness_index:fairness_index + 4] ==
          ["--fairness-engine", "compute_share", "--prefill-compute-share", "0.6"]
          and args[:fairness_index] + args[fairness_index + 4:]
          == gc.family_serve_args(baseline))
    check("compute_share accepts the existing MTP and TP4/DCP4 contract",
          not errs(gc.validate(enabled, {"gpu_count": 4})))
    check("compute_share also accepts speculation disabled",
          not errs(gc.validate({**enabled, "MTP_TOKENS": 0}, {"gpu_count": 4})))
    check("cadence cannot throttle an enabled fairness controller",
          "fairness-cadence" in errs(gc.validate(
              {**enabled, "PREFILL_SCHEDULE_INTERVAL": 2})))
    check("fairness off preserves cadence scheduling",
          "fairness-cadence" not in errs(gc.validate(
              {**baseline, "PREFILL_SCHEDULE_INTERVAL": 2})))
    qwen, _, _ = resolved(family="qwen36", PREFILL_FAIRNESS_ENGINE="compute_share")
    check("non-GLM fairness is refused rather than silently ignored",
          "fairness-family" in errs(gc.validate(qwen)))

    env = gc.env_layer({"PREFILL_FAIRNESS_ENGINE": "compute_share",
                        "PREFILL_COMPUTE_SHARE": "0.7"})
    cfg, _, _ = gc.resolve(state_values={}, env_values=env)
    args = gc.family_serve_args(cfg)
    check("startup environment controls the emitted share",
          args[args.index("--prefill-compute-share") + 1] == "0.7")
    cfg, _, _ = gc.resolve(state_values={"PREFILL_COMPUTE_SHARE": 0.4}, env_values=env)
    args = gc.family_serve_args(cfg)
    check("persisted share overrides the startup environment",
          args[args.index("--prefill-compute-share") + 1] == "0.4")
    cfg, _, _ = gc.resolve(state_values={"PREFILL_FAIRNESS_ENGINE": "off"}, env_values=env)
    args = gc.family_serve_args(cfg)
    check("persisted off removes both flags even with a configured share",
          "--fairness-engine" not in args and "--prefill-compute-share" not in args)
    for value in (0, 1, -0.1, 1.1, "nan", "inf", "not-a-number"):
        for layer in ("env", "state"):
            invalid = {"PREFILL_COMPUTE_SHARE": value}
            try:
                gc.resolve(state_values=invalid if layer == "state" else {},
                           env_values=gc.env_layer(invalid) if layer == "env" else {})
            except gc.ConfigError:
                rejected = True
            else:
                rejected = False
            check(f"invalid {layer} share {value!r} cannot silently become 0.6", rejected)
    cfg, _, _ = gc.resolve(state_values={"PREFILL_COMPUTE_SHARE": 0.3},
                          env_values=gc.env_layer({"PREFILL_COMPUTE_SHARE": "nan"}))
    check("a valid state override repairs malformed lower-precedence input",
          cfg["PREFILL_COMPUTE_SHARE"] == 0.3)
    invalid_env = gc.env_layer({"PREFILL_FAIRNESS_ENGINE": "typo",
                               "PREFILL_COMPUTE_SHARE": "nan"})
    try:
        gc.resolve(state_values={}, env_values=invalid_env)
    except gc.ConfigError:
        rejected = True
    else:
        rejected = False
    check("unknown fairness selector is not silently treated as off", rejected)
    original = gc.load_startup_env
    gc.load_startup_env = lambda: dict(invalid_env)
    try:
        repair = gc.minimize({"PREFILL_FAIRNESS_ENGINE": "off",
                              "PREFILL_COMPUTE_SHARE": 0.6})
        restored, _, _ = gc.resolve(state_values=repair)
        check("state minimization preserves default-valued repairs over invalid env",
              repair == {"PREFILL_FAIRNESS_ENGINE": "off", "PREFILL_COMPUTE_SHARE": 0.6}
              and "--fairness-engine" not in gc.family_serve_args(restored))
    finally:
        gc.load_startup_env = original


def test_fairness_cli():
    section("fairness CLI export and persisted disable")
    with tempfile.TemporaryDirectory(prefix="glm-fairness-cli-") as tmp:
        env = {**os.environ, "GLM_STATE_DIR": os.path.join(tmp, "state"),
               "GLM_RUNTIME_DIR": os.path.join(tmp, "runtime")}
        env.pop("GLM_CONFIG_ATTEMPT", None)
        os.makedirs(env["GLM_RUNTIME_DIR"])
        snapshot = os.path.join(env["GLM_RUNTIME_DIR"], "startup-env.json")
        state = os.path.join(env["GLM_STATE_DIR"], "config.json")
        gc.write_json_atomic(snapshot, {
            "MODEL_FAMILY": "glm52", "GLM_GPU_COUNT": "4",
            "MODEL_VARIANT": "exl3-tr3-glm53-3.42bpw-500k",
            "PREFILL_FAIRNESS_ENGINE": "compute_share", "PREFILL_COMPUTE_SHARE": 0.6})
        cli = [sys.executable, os.path.join(REPO, "scripts/config_cli.py")]
        for values, expected in (({}, True), ({"PREFILL_FAIRNESS_ENGINE": "off"}, False)):
            gc.write_json_atomic(state, {"values": values})
            exported = subprocess.run(cli + ["env"], env=env, text=True, capture_output=True)
            check("CLI resolves valid fairness state", exported.returncode == 0, exported.stderr)
            shell = subprocess.run(["bash"], input=exported.stdout +
                                   '\nprintf "%s\\n" "${FAMILY_SERVE_ARGS[@]}"\n',
                                   env=env, text=True, capture_output=True)
            args = shell.stdout.splitlines()
            check("shell consumes the same enabled/disabled fairness argv",
                  shell.returncode == 0
                  and ("--fairness-engine" in args) == expected
                  and ("--prefill-compute-share" in args) == expected)
        gc.write_json_atomic(state, {"values": {"PREFILL_SCHEDULE_INTERVAL": 2}})
        check("CLI rejects cadence conflict before engine startup",
              subprocess.run(cli + ["validate", "--quiet"], env=env,
                             capture_output=True).returncode == 2)
        gc.write_json_atomic(state, {"values": {"PREFILL_COMPUTE_SHARE": "nan"}})
        check("CLI refuses malformed share rather than exporting fallback argv",
              subprocess.run(cli + ["env"], env=env, capture_output=True).returncode != 0)


def run(test):
    """Run one test_* function. A failing check() raises AssertionError
    after recording the failure; swallow it here so the rest of the
    suite still runs and main() still returns 1 on any failure."""
    try:
        test()
    except AssertionError:
        pass


def main():
    tmp = tempfile.mkdtemp(prefix="glm-fam-test-")
    saved = dict(os.environ)
    os.environ["GLM_STATE_DIR"] = os.path.join(tmp, "state")
    os.environ["GLM_RUNTIME_DIR"] = os.path.join(tmp, "run")
    os.makedirs(os.environ["GLM_STATE_DIR"], exist_ok=True)
    os.makedirs(os.environ["GLM_RUNTIME_DIR"], exist_ok=True)
    try:
        run(test_glm_release_defaults)
        run(test_glm_max_context_profile)
        run(test_madeby561_hybrid)
        run(test_higher_fidelity_exl3_candidate)
        run(test_r20_336_online_quant_candidate)
        run(test_r28_342_shared_h_profile)
        run(test_glm53_342_dsa_profile)
        run(test_full_glm53_candidate_boundaries)
        run(test_prefill_fairness)
        run(test_fairness_cli)
        run(test_known_good_replays_across_profile_env_change)
        run(test_flash_is_refused_not_substituted)
        run(test_qwen_preset)
        run(test_custom_profile)
        run(test_inapplicable_knobs)
        run(test_rules_are_family_scoped)
        run(test_family_coherence_rules)
        run(test_gpu_count_gate)
        run(test_env_layer_still_wins_over_family)
        run(test_shipped_variants_validate_clean)
        run(test_validation_scoping_fixes)
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} passed, {len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
