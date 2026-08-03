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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import glm_config as gc  # noqa: E402

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


def test_glm_release_integration():
    section("the GG v20-r20 runtime integration matches the measured launch contract")
    entry = open(os.path.join(REPO, "entrypoint.sh")).read()
    dockerfile = open(os.path.join(REPO, "Dockerfile")).read()
    acme_retry = open(os.path.join(REPO, "scripts", "acme_retry.sh")).read()
    config_cli = open(os.path.join(REPO, "scripts", "config_cli.py")).read()
    local_runner = open(os.path.join(REPO, "scripts", "run-local-podman.sh")).read()
    kld_runner = open(
        os.path.join(REPO, "scripts", "bench-glm52-kld-tp4.sh")).read()
    runpod = json.load(open(os.path.join(REPO, "runpod-template.json")))
    check("the base image is the pinned GG v20-r20 manifest",
          "sha256:40c891fd3fd573a92708e8a4bfa028ec91127a92491504c59006cf9735b20560"
          in dockerfile
          and "verify_r20_base.py" in dockerfile
          and "apply_field_review_patches.py" not in dockerfile
          and "field-review-r14" not in dockerfile)
    check("static NVFP4 scaling selects and verifies the reviewed artifact",
          "KV_SCALE_MODE:-static-calibrated" in entry
          and "VLLM_NVFP4_MLA_SCALES_FILE" in entry
          and "efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2"
          in entry and
          "/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json"
          in dockerfile)
    check("dynamic-token NVFP4 selects the complete paired r9 ABI",
          "dynamic-token)" in entry
          and "export KV_FP8_ROPE=1" in entry
          and "export VLLM_NVFP4_MLA_DYNAMIC_SCALE=1" in entry
          and "unset VLLM_NVFP4_MLA_SCALES_FILE" in entry)
    check("adaptive exact indexer folding stays bounded",
          'SPARKINFER_INDEXER_TWO_LEVEL_FOLD:-auto' in entry
          and 'SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB:-256' in entry)
    check("deSEC DNS-01 runs the authoritative convergence guard",
          'acme_retry.sh" --once' in entry
          and 'acme_retry.sh" --retry' in entry
          and 'desec_acme_guard.py"' in acme_retry
          and '--zone "$DESEC_DOMAIN" --domain "$ACME_DOMAIN"' in acme_retry
          and 'guard_pid=$!' in acme_retry
          and '--dns.propagation-wait "${DESEC_LEGO_PROPAGATION_WAIT:-45s}"'
          in acme_retry
          and 'rrsets/_acme-challenge.${sub}/TXT/' in acme_retry
          and 'timeout --signal=TERM --kill-after=10' in acme_retry
          and "dnspython==2.8.0" in dockerfile
          and "lego_v4.35.2_linux_amd64.tar.gz" in dockerfile
          and "ee5be4bf457de8e3efa86a51651c75c87f0ee0e4e9f3ae14f6034d68365770f3"
          in dockerfile
          and "COPY scripts/ /opt/scripts/" in dockerfile)
    check("VMs without CUDA peer access fall back before custom collectives",
          'nvidia-smi topo -p2p r' in entry
          and "^[[:space:]]*GPU[0-9]+[[:space:]]" in entry
          and '_p2p_matrix_rows' in entry
          and 'export B12X_PCIE_DMA=0' in entry
          and 'export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_USE_B12X_DCP_A2A=0'
          in entry
          and 'using NCCL/SHM collectives' in entry)
    check("JarvisLabs gets a persisted generated dashboard token and launch URL",
          '[ "$PLATFORM" = "runpod" ] || [ "$PLATFORM" = "jarvislabs" ]'
          in entry
          and "OPEN_BUTTON_TOKEN_FILE" in entry
          and 'https://${ACME_DOMAIN}:${PUBLIC_DASHBOARD_PORT}/?token=${OPEN_BUTTON_TOKEN}'
          in entry
          and "Use an SSH tunnel, then open:" in entry)
    check("the target/draft trellis minimum is role-aware",
          'if [ "${MTP_TOKENS:-3}" = "0" ]; then' in entry and
          "export VLLM_EXL3_TRELLIS_MIN_M=1" in entry and
          "unset VLLM_EXL3_TRELLIS_MIN_M" in entry)
    check("the validated Trellis ceiling survives the runtime env refresh",
          'VLLM_EXL3_TRELLIS_MAX_M="${VLLM_EXL3_TRELLIS_MAX_M:-32}"'
          in entry and
          "export VLLM_EXL3_TRELLIS_MAX_M=32" not in entry)
    for setting in (
            "VLLM_DCP_QUERY_SPLIT=1",
            "VLLM_DCP_TOPK_OWNER_MERGE=1",
            'VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-auto}"',
            "VLLM_DCP_PROJECT_BEFORE_MERGE=1",
            "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1"):
        check(f"prefill policy includes {setting}", setting in entry)
    check("the DCP prefill workspace is topology-safe",
          'if [ "${DCP:-4}" = "4" ]; then' in entry
          and "VLLM_DCP_PROJECT_BEFORE_MERGE=0" in entry
          and "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0" in entry)
    check("the DCP prefill workspace is self-service",
          'VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB="${DCP_PREFILL_WORKSPACE_MIB:-1024}"'
          in entry)
    check("safetensors remains the architecture-level fallback",
          gc.family("glm52")["defaults"]["LOAD_FORMAT"] == "safetensors")
    check("the balanced profile promotes the near-max-qualified loader",
          resolved(gpus=4)[0]["LOAD_FORMAT"] == "safetensors")
    check("local shared checkpoints have an explicit immutable mode",
          'if [ "${MODEL_READ_ONLY:-0}" = "1" ]; then' in entry
          and "Checkpoint is read-only; mutation steps skipped." in entry)
    check("a prepared vision derivative is allowed read-only",
          '[ ! -f "$MODEL_DIR/.vision-enabled" ]' in entry
          and '--vision "${VISION:-0}" --dry-run --quiet' in entry)
    check("vision plugins install from a writable copy for read-only checkpoints",
          "install_vision_plugin()" in entry
          and 'cp -a "$plugin_dir/." "$install_dir/"' in entry
          and "--no-build-isolation --no-deps" in entry
          and "read-only derivative plugin registered" in entry)
    check("a missing vision plugin fails before the vLLM restart loop",
          "FATAL: Vision marker is present" in entry
          and "FATAL: read-only vision derivative cannot register" in entry)
    check("immutable targets default all compile caches away from the checkpoint",
          'export VLLM_CACHE_ROOT="/cache/$CACHE_NAMESPACE/vllm"' in entry
          and 'export TRITON_CACHE_DIR="/cache/$CACHE_NAMESPACE/triton"' in entry
          and 'export TORCH_EXTENSIONS_DIR="/cache/$CACHE_NAMESPACE/torch_extensions"'
          in entry
          and 'export TORCHINDUCTOR_CACHE_DIR="/cache/$CACHE_NAMESPACE/torchinductor"'
          in entry)
    check("persistent compile caches are isolated at each runtime fingerprint",
          'CACHE_NAMESPACE="${LOCAL_INFERENCE_CACHE_FINGERPRINT:-turnkey-unversioned}-turnkey-r20-native1"'
          in entry
          and '$MODEL_DIR/.vllm-cache/$CACHE_NAMESPACE/vllm' in entry
          and '/cache/$CACHE_NAMESPACE/torch_extensions' in entry)
    check("the r20 native EXL3 gate and compatibility fallback are cache-versioned",
          "verify_r20_base.py" in dockerfile
          and "patch_exl3_parity_abi.py" in dockerfile
          and "patch_exl3_mixk.py" in dockerfile
          and "ad8b9b1d202c65d68f4f3cdcb8c6b1dac0670216f03dfdde4429416b089baae6"
          in dockerfile
          and dockerfile.index("patch_exl3_parity_abi.py")
          < dockerfile.rindex("patch_exl3_mixk.py")
          and "-turnkey-r20-native1" in entry)
    check("the local Podman runner does not bypass cache fingerprinting",
          "-e VLLM_CACHE_ROOT=/cache/vllm" not in local_runner
          and "-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions" not in local_runner)
    check("the local Podman runner preserves the requested CUDA rank order",
          '-e CUDA_VISIBLE_DEVICES="$GPU_DEVICES"' in local_runner)
    check("auto profile sentinels do not leak into the calibration helper",
          "env -u DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS" in entry
          and "-u PCIE_DMA_MIN_BYTES" in entry)
    for key in ("OFFLOAD_FRACTION", "OFFLOAD_IGNORE_MEMLOCK",
                "KV_CACHE_MEMORY_BYTES",
                "LMCACHE_L1_INIT_GB",
                "CUDAGRAPH_CAPTURE_SIZES", "MAX_CUDAGRAPH_CAPTURE_SIZE",
                "VLLM_EXL3_TRELLIS_MAX_M", "VLLM_EXL3_PREFILL_CAPACITY",
                "DCP_CKV_GATHER_MAX_TOKENS",
                "DCP_KV_CACHE_INTERLEAVE_SIZE",
                "MTP_REJECTION_SAMPLE_METHOD", "KV_SCALE_MODE"):
        check(f"the local runner forwards an explicit {key}",
              key in local_runner and
              'config_env+=(-e "$config_name=${!config_name}")' in local_runner)
    check("the local runner can explicitly override detected TP and DCP",
          "TENSOR_PARALLEL_SIZE DCP MAX_MODEL_LEN" in local_runner)
    check("the local runner does not mask the selected variant with old defaults",
          'MAX_NUM_BATCHED_TOKENS:-2048' not in local_runner
          and 'GPU_BLOCKS_OVERRIDE:-2048' not in local_runner
          and 'F8_DMA:-ring' not in local_runner)
    check("the local runner forwards arbitrary validated TUNE_ overrides",
          "TUNE_[A-Z0-9_]*" in local_runner
          and '"${tuning_env[@]}"' in local_runner)
    check("the local runner forwards its documented GPU-free config smoke",
          '-e CONFIG_SMOKE="${CONFIG_SMOKE:-0}"' in local_runner
          and 'restart_policy=no' in local_runner
          and '--restart="$restart_policy"' in local_runner)
    check("the local runner mounts bounded persistent LMCache below the "
          "secure-erase-aware workspace",
          'LMCACHE_DISK_HOST="${LMCACHE_DISK_HOST:-}"' in local_runner
          and '$LMCACHE_DISK_HOST:/workspace/.lmcache:rw' in local_runner
          and "LMCACHE_DISK_HOST must be an absolute path" in local_runner)
    check("the local runner gives vLLM and LMCache a graceful replacement",
          'podman container exists "$NAME"' in local_runner
          and 'podman stop -t "${STOP_TIMEOUT:-120}" "$NAME"' in local_runner
          and 'podman rm -f "$NAME"' in local_runner)
    check("a native local run clears any draft path baked into the image",
          'draft_env=(-e DRAFT_MODEL=)' in local_runner
          and 'draft_env=(-e DRAFT_MODEL="$DRAFT_MODEL_CONTAINER")'
          in local_runner)
    check("the local runner can compose a read-only vision derivative",
          'VISION_ASSET_HOST="${VISION_ASSET_HOST:-}"' in local_runner
          and '$MODEL_DIR_CONTAINER/.vision:ro' in local_runner
          and '$MODEL_DIR_CONTAINER/.vision-enabled:ro' in local_runner
          and "set both VISION_ASSET_HOST and VISION_MARKER_HOST" in local_runner)
    check("the local runner can supply an immutable shared tokenizer",
          'TOKENIZER_JSON_HOST="${TOKENIZER_JSON_HOST:-}"' in local_runner
          and '$MODEL_DIR_CONTAINER/tokenizer.json:ro' in local_runner
          and "tokenizer serialization does not exist" in local_runner)
    check("absolute shared-store checkpoint symlinks remain resolvable",
          'SHARED_MODEL_STORE_HOST="${SHARED_MODEL_STORE_HOST:-}"' in local_runner
          and 'SHARED_MODEL_STORE_CONTAINER="${SHARED_MODEL_STORE_CONTAINER:-$SHARED_MODEL_STORE_HOST}"'
          in local_runner
          and "SHARED_MODEL_STORE_CONTAINER must be an absolute path" in local_runner)
    check("Hugging Face snapshot symlinks retain their sibling blob store",
          '*/models--*/snapshots/*)' in local_runner
          and 'hf_model_root="${MODEL_DIR_HOST%%/snapshots/*}"' in local_runner
          and 'MODEL_MOUNT_CONTAINER=/models/hf-checkpoint' in local_runner
          and '$MODEL_MOUNT_HOST:$MODEL_MOUNT_CONTAINER:ro' in local_runner)
    check("the KLD runner also preserves Hugging Face snapshot symlinks",
          '*/models--*/snapshots/*)' in kld_runner
          and 'MODEL_MOUNT_ROOT="$hf_model_root"' in kld_runner
          and 'MODEL_CONTAINER_ROOT="/models/hf-checkpoint/$hf_snapshot_relative"'
          in kld_runner
          and '$MODEL_MOUNT_ROOT:$MODEL_MOUNT_CONTAINER:ro' in kld_runner)
    check("the KLD protocol can retain DCP4 for a mixed checkpoint that needs it",
          'KLD_DCP="${KLD_DCP:-1}"' in kld_runner
          and '"dcp=$KLD_DCP"' in kld_runner
          and '\\"decode_context_parallel_size\\":$KLD_DCP' in kld_runner)
    check("an external draft is compatible with an immutable target",
          '[ "${MTP78_MODE:-off}" != "off" ] && [ -z "${DRAFT_MODEL:-}" ]'
          in entry)
    check("InstantTensor is an explicit opt-in",
          gc.KNOB_BY_KEY["LOAD_FORMAT"]["choices"] == ["safetensors", "instanttensor"])
    check("the CLI explains the variant inheritance layer",
          "default < family < variant < env < state file" in config_cli)
    check("offload reports a per-worker physical estimate",
          "OFF_BYTES=$((OFF_TOTAL_BYTES / OFFLOAD_RANKS))" in entry)
    check("the native connector receives the aggregate offload budget",
          '"cpu_bytes_to_use\\":$OFF_TOTAL_BYTES' in entry
          and '"cpu_bytes_to_use\\":$OFF_BYTES' not in entry)
    check("the Runpod GLM template inherits the measured variant defaults",
          runpod["env"]["VISION"] == "0"
          and "OFFLOAD_FRACTION" not in runpod["env"]
          and "MTP_DRAFT_SAMPLE_METHOD" not in runpod["env"]
          and "MTP78_MODE" not in runpod["env"])
    for setting in ("VLLM_PCIE_DMA_FP8",
                    "SPARKINFER_PCIE_DMA_FP8", "PCIE_CALIBRATION_ONLY=1",
                    "VLLM_USE_MEGA_AOT_ARTIFACT=1",
                    "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:--1",
                    "PCIE_DMA_MIN_BYTES:--1"):
        check(f"the v20 transport path includes {setting}", setting in entry)
    for obsolete in ("VLLM_USE_B12X_PCIE_DMA",
                     "VLLM_RTX6K_FUSED_ALLREDUCE_ADD",
                     "VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER"):
        check(f"the appliance removes inherited legacy knob {obsolete}",
              f"unset {obsolete}" in entry
              and f"export {obsolete}" not in entry)
    check("the vision argument fragment does not duplicate trust-remote-code",
          'VISION_ARGS=(--limit-mm-per-prompt "{\\"vision_chunk\\":${VISION_CHUNKS:-8}}")'
          in entry)


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
          and profile_env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
          and profile_env["SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB"] == "64")
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
          and eff["PREFIX_CACHE_DISK_GB"] == 512
          and eff["PCIE_DMA_MIN_BYTES"] == 6291456
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


def test_long_context_gate_is_family_independent():
    section("the long-context gate belongs to every family")
    import verify_serving
    src = open(os.path.join(REPO, "scripts", "verify_serving.py")).read()
    check("the verifier has no family branch",
          "MODEL_FAMILY" not in src and "FAMILIES" not in src,
          "verify_serving.py must stay model-agnostic")
    check("it does not import the config registry at all",
          "import glm_config" not in src)
    check("it still refuses to call short prompts sufficient",
          "long_context_verified" in src)
    check("and the needle probe is the thing that sets it",
          "needle_probe" in src and "long_context_verified" in src)
    # the probe budget follows MAX_MODEL_LEN, which is a family default
    for fam, expected in (("glm52", 524288), ("qwen36", 196608)):
        eff, _, _ = resolved(fam)
        check(f"{fam} probes against its own context ceiling ({expected})",
              eff["MAX_MODEL_LEN"] == expected)
    check("the verifier's degeneracy detector is generic",
          "def degenerate" in src and "punctuation" in src)


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
        run(test_glm_release_integration)
        run(test_glm_max_context_profile)
        run(test_madeby561_hybrid)
        run(test_higher_fidelity_exl3_candidate)
        run(test_r20_336_online_quant_candidate)
        run(test_qwen_preset)
        run(test_custom_profile)
        run(test_inapplicable_knobs)
        run(test_rules_are_family_scoped)
        run(test_family_coherence_rules)
        run(test_gpu_count_gate)
        run(test_long_context_gate_is_family_independent)
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
