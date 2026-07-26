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
                      "VLLM_EXL3_TRELLIS_MAX_M", "VLLM_EXL3_TRELLIS_MIN_M",
                      "VISION", "VISION_CHUNKS", "BASE_GENERATION"),
        "defaults": {
            "TENSOR_PARALLEL_SIZE": 4, "MAX_MODEL_LEN": 524288, "MTP_TOKENS": 3,
            "MAX_NUM_SEQS": 8, "MAX_NUM_BATCHED_TOKENS": 3072,
            "GPU_MEMORY_UTILIZATION": 0.93, "GPU_BLOCKS_OVERRIDE": 2048,
            "KV_CACHE_DTYPE": "fp8", "SERVED_MODEL_NAME": "GLM-5.2",
        },
        # %(...)s are substituted from the resolved knobs
        "serve_args": [
            "--decode-context-parallel-size", "%(DCP)s",
            "--dcp-comm-backend", "a2a",
            "--dcp-kv-cache-interleave-size", "64",
            "--attention-backend", "B12X_MLA_SPARSE",
            "--moe-backend", "b12x",
            "--load-format", "safetensors",
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
        "label": "Qwen3.6 27B (dense, hybrid Gated DeltaNet) — UNTESTED here",
        "tested": False,
        "env_block": "generic",
        "default_variant": "qwen36-bf16",
        # nvfp4_ds_mla is an MLA KV layout; it does not exist for this family.
        "kv_dtypes": ["auto", "fp8"],
        # The model card's own vLLM line uses
        #   --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
        # so speculation applies here too — with a different method and without
        # any of the MTP78 draft machinery.
        "spec_method": "qwen3_next_mtp",
        "own_knobs": (),
        "defaults": {
            # 1, not 4: 27B at BF16 is ~55.6 GB and fits one 96 GB card, and the
            # whole point of this preset is that the image can start on a 1-GPU
            # host at all.
            "TENSOR_PARALLEL_SIZE": 1,
            "MAX_MODEL_LEN": 262144,        # native; 1M needs YaRN rope scaling
            "MTP_TOKENS": 2,                # the model card's number
            "MAX_NUM_SEQS": 8,
            "MAX_NUM_BATCHED_TOKENS": 3072,
            "GPU_MEMORY_UTILIZATION": 0.90,
            "GPU_BLOCKS_OVERRIDE": 0,       # let vLLM size the pool
            "KV_CACHE_DTYPE": "auto",
            "SERVED_MODEL_NAME": "Qwen3.6-27B",
        },
        "serve_args": [
            "--reasoning-parser", "qwen3",
        ],
        "compilation_config": (
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'
            '[%(CUDAGRAPH_CAPTURE_SIZES)s]}'),
        "notes": (
            "UNVALIDATED PRESET. Qwen3.6-27B is a DENSE 27B model with hybrid "
            "attention (16 x (3 x Gated DeltaNet -> FFN) + 1 x Gated Attention -> FFN), "
            "262,144 native context extensible to ~1M with YaRN. It is not MLA, so "
            "DCP, the sparse-indexer hf_overrides and the EXL3 trellis have no meaning "
            "here and are refused rather than silently ignored. Facts from the model "
            "card (huggingface.co/Qwen/Qwen3.6-27B); the serve line is derived from "
            "the card's own vLLM example. NOBODY HAS BOOTED THIS IMAGE WITH IT: "
            "whether this vLLM fork implements Gated DeltaNet and qwen3_next_mtp is "
            "unknown. No tool-call parser is set — the card does not name one, and "
            "guessing wrong makes --enable-auto-tool-choice fail at startup."),
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
        "label": "EXL3-TR3 3.0bpw (default, measured)",
        "repo": "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw",
        "dirname": "GLM-5.2-EXL3-TR3-3.0bpw",
        "quantization": "exl3",
        "kv_scales_calibrated": False,
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
    "qwen36-bf16": {
        "family": "qwen36",
        "label": "Qwen3.6-27B BF16 (UNVALIDATED preset)",
        "repo": "Qwen/Qwen3.6-27B",
        "dirname": "Qwen3.6-27B",
        # No --quantization at all: the upstream checkpoint is BF16, and naming a
        # quantization method the checkpoint does not carry is a boot failure.
        "quantization": "",
        "kv_scales_calibrated": False,
        "download_gib": 56,
        "tested": False,
    },
}

# MTP draft types -> the three env knobs the serve path actually consumes.
# `tr3-graft`   in-place surgery on layer 78 of the target (the ONLY draft with
#               long-context evidence: needle 6/6 fp8, armC 3/3 at 150/190/250K)
# `tr3-override` separate rank-sliced EXL3 draft dir (--speculative-config model)
# `nvfp4`       external NVFP4 draft; MUST carry its own quantization (rule 4)
DRAFTS = {
    "off":           {"label": "off (no speculative decode)", "mtp78_mode": "off"},
    "bf16":          {"label": "BF16 in-checkpoint draft (stock, 19.3 GB)", "mtp78_mode": "off"},
    "tr3-graft":     {"label": "EXL3 3bpw trellis, grafted into the target", "mtp78_mode": "graft"},
    "tr3-override":  {"label": "EXL3 3bpw trellis, separate draft dir", "mtp78_mode": "override"},
    "nvfp4":         {"label": "External NVFP4 draft dir", "mtp78_mode": "off",
                      "draft_quantization": "modelopt_fp4"},
}

# Base image generations. The v20 speculator cannot CUDA-graph-capture a
# rank-sliced EXL3 draft (rule tr3-on-v20).
BASE_GENERATIONS = ("v20", "pre-v20")


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
             "number in this README was measured on. 'qwen36' is a dense 27B preset "
             "that nobody has booted with this image: it exists so the template can "
             "run on one GPU and be iterated on, not because it is known to work. "
             "Changing family triggers a fresh download.")),

    dict(key="TENSOR_PARALLEL_SIZE", type="int", default=4, min=1, max=8,
         group="Parallelism", scope="engine", label="Tensor parallel size",
         rationale=(
             "How many GPUs the model's weights are sharded across. This was a "
             "hard-coded 4 in the serve line, which meant the image could not start on "
             "a 1-GPU host at all — it aborted at the GPU count check before reaching "
             "vLLM. GLM-5.2 needs 4; Qwen3.6-27B at BF16 is ~56 GB and fits one 96 GB "
             "card, so the Qwen preset defaults to 1. It must be <= the number of "
             "visible GPUs, and the instance is refused at boot if it is not.")),

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

    dict(key="MAX_MODEL_LEN", type="int", default=524288, min=4096, max=1048576,
         group="Model", scope="engine", label="Max context length",
         rationale=(
             "Longest single request the engine will accept, and the length vLLM "
             "must be able to fit in KV before it will start at all. Lowering it is "
             "the standard remedy when boot fails with 'To serve at least one request "
             "with the model's max seq len ... larger than the available KV cache "
             "memory' — it is a startup gate, not a throughput knob. With vision "
             "resident, fp8 KV sizes the usable ceiling nearer 384-420K than 512K.")),

    dict(key="GPU_BLOCKS_OVERRIDE", type="int", default=2048, min=0, max=65536,
         group="Model", scope="engine", label="KV pool pin (--num-gpu-blocks-override)",
         rationale=(
             "Pins the KV pool at exactly 512K tokens (2048 blocks x 256) so the "
             "headroom the trellis draft frees stays predictable instead of being "
             "absorbed. Set 0 to drop the pin and let vLLM take all available KV — "
             "measured ~697K tokens of pool on the EXL3+MTP78 stack, i.e. much more "
             "concurrency margin, at the cost of a pool size that moves whenever "
             "anything else about memory changes.")),

    dict(key="KV_CACHE_DTYPE", type="choice", default="fp8",
         choices=["auto", "fp8", "nvfp4_ds_mla"], group="Model", scope="engine",
         label="KV cache dtype",
         rationale=(
             "fp8 is the safe default and costs ~1.7x the bytes per token that nvfp4 "
             "would. nvfp4 KV requires per-checkpoint calibrated MLA outer scales; "
             "the EXL3 checkpoint has none, and without them long context silently "
             "degenerates — measured needle 0/6 at 505K with degenerate text while "
             "GSM8K, vision and structured output all still passed. Nothing short of "
             "a long-context retrieval test catches it.")),

    dict(key="MTP_DRAFT", families=("glm52",), type="choice", default="tr3-graft", choices=list(DRAFTS),
         group="Speculative decoding", scope="checkpoint", label="MTP draft type",
         aliases=[],
         rationale=(
             "Which draft model proposes the speculative tokens. All four drafts "
             "accept near-identically (MAL ~3.52, ~84%); what differs is VRAM and "
             "whether they boot. 'tr3-graft' (default) swaps layer 78 of the target "
             "for a 3.0bpw EXL3 trellis version in place: 19.3 GB -> 3.7 GB, ~3.8 "
             "GB/GPU straight into KV, and it is the only draft with long-context "
             "evidence behind it. 'tr3-override' keeps the target byte-identical and "
             "measured slightly better on acceptance, but a rank-sliced EXL3 draft "
             "cannot be CUDA-graph captured on the v20 base. 'nvfp4' is an external "
             "draft dir that avoids the rank-sliced path entirely (and must declare "
             "its own quantization). 'bf16' is the stock in-checkpoint draft: always "
             "works, costs 19.3 GB. 'off' disables speculation (~30% slower decode).")),

    dict(key="MTP_TOKENS", type="int", default=3, min=0, max=8,
         group="Speculative decoding", scope="engine", label="Speculation depth",
         rationale=(
             "How many tokens the draft proposes per step. 3 is the shipped default. "
             "The optimum moves with the draft's cost: with the 19.3 GB BF16 draft, "
             "MTP-5 lost ~22%; with the 3.7 GB trellis draft it WINS (GSM8K n=30: "
             "MTP-2 42.9, MTP-3 51.5, MTP-5 53.4 tok/s). Raising it also raises the "
             "decode query width m = MAX_NUM_SEQS x (1 + MTP_TOKENS), which has to "
             "stay inside the capture/trellis window. 0 disables speculation.")),

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

    dict(key="DCP", families=("glm52",), type="choice", default="4", choices=["1", "2", "4"],
         group="Parallelism", scope="engine", label="Decode context parallel",
         rationale=(
             "Shards the KV cache across GPUs at decode (--decode-context-parallel-size). "
             "DCP=4 is what makes a 512K pool fit at all: each rank holds a quarter of "
             "the KV. Lowering it replicates KV instead, roughly dividing the usable "
             "context by the same factor, and is only interesting when debugging a "
             "suspected DCP/a2a issue. Must divide the tensor-parallel size (4).")),

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

    dict(key="VLLM_EXL3_TRELLIS_MIN_M", families=("glm52",), type="int", default=4, min=4, max=4,
         editable=False, group="Concurrency", scope="engine",
         label="EXL3 trellis window min (m) — LOCKED",
         rationale=(
             "LOCKED AT 4. Lowering it to 1 makes an EXL3 capture crash disappear and "
             "SILENTLY CORRUPTS OUTPUT: measured 32K needle 0/2 and 370K needle 0/5, "
             "pure garbage, while a short-prompt quality gate still passed 6/6. m=1..3 "
             "move off the eager parity path onto a trellis kernel nothing has verified "
             "there. There is deliberately no control here that lowers it.")),

    dict(key="GPU_MEMORY_UTILIZATION", type="float", default=0.93, min=0.50, max=0.99,
         group="Memory", scope="engine", label="GPU memory utilization",
         rationale=(
             "Fraction of each card vLLM may claim. Higher means more KV; too high and "
             "the memory profile OOMs during capture or KV validation fails outright. "
             "0.93 is the value the 512K fp8 pool was validated at on 96 GB cards. "
             "Memory-profile draws vary by a few hundred MiB run to run, so a value "
             "that boots 2 times in 3 is not a value that boots.")),

    dict(key="OFFLOAD_FRACTION", type="float", default=0.70, min=0.0, max=0.95,
         group="Memory", scope="engine", label="DRAM KV offload fraction",
         rationale=(
             "Fraction of the instance's RAM allocation used as a pinned CPU KV tier "
             "(0 disables). Sized from min(cgroup limit, MemTotal), so a partial rental "
             "does not oversize it. It is a pure cache tier: kv_load_failure_policy="
             "recompute means a block that cannot be fetched back is recomputed rather "
             "than failing the request. Costs host RAM and nothing else; the win is on "
             "prefix-cache hits across requests.")),

    dict(key="VISION", families=("glm52",), type="bool", default=True, group="Multimodal", scope="checkpoint",
         label="Vision (image input) — EXPERIMENTAL on EXL3",
         rationale=(
             "Bolts the MoonViT-3d tower + PatchMerger projector (~890 MB, BF16, no "
             "text weight touched) onto the text backbone. KNOWN BROKEN AT LONG "
             "CONTEXT ON THE EXL3 TARGET: measured 32K needle 0/2 with degenerate text "
             "on both the v20 and pre-v20 bases, while short-prompt and vision smoke "
             "tests passed 6/6. It also costs ~1.31 GiB/GPU and ~13% of text decode, "
             "enough memory pressure that MAX_MODEL_LEN 384K can fail KV validation "
             "(needed 5.7 GiB, had 3.97). Use it for short-prompt image work only, and "
             "turn it off for anything that depends on long-context retrieval.")),

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

    dict(key="BASE_GENERATION", families=("glm52",), type="choice", default="v20", choices=list(BASE_GENERATIONS),
         editable=False, group="Serving", scope="engine",
         label="Base image generation (read-only)",
         rationale=(
             "Which fork generation the image was built from. It is baked into the "
             "image, not chosen at runtime, and it decides whether a rank-sliced EXL3 "
             "draft can be captured at all: on v20 the SpeculatorCudaGraphManager "
             "captures it at m=3, outside the trellis window [4,32], and the engine "
             "dies. Shown here because the draft type must be validated against it.")),
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


def env_layer(env=None) -> dict:
    """Knob values present in the (startup) environment, incl. legacy spellings."""
    env = os.environ if env is None else env
    out = {}
    for knob in KNOBS:
        for name in [knob["key"]] + list(knob.get("aliases", [])):
            if env.get(name, "") != "":
                try:
                    out[knob["key"]] = coerce(knob, env[name])
                except ConfigError:
                    pass          # a bad env value falls back to the default
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
            legacy = "bf16"
        # Order matters and matches the old entrypoint: MTP78_TRELLIS=0 was
        # applied AFTER MTP78_MODE and won over it.
        if env.get("MTP78_TRELLIS", "") == "0":
            legacy = "bf16"
        if env.get("DRAFT_MODEL", ""):
            legacy = "nvfp4" if env.get("DRAFT_QUANTIZATION", "") == "modelopt_fp4" else legacy
        if legacy:
            out["MTP_DRAFT"] = legacy
    return out


def snapshot_startup_env(env=None) -> dict:
    """Freeze the env layer for the life of the container (called once by the
    entrypoint). Everything after that reads the snapshot, so the layering can
    never be perturbed by something the entrypoint exports later."""
    layer = env_layer(env)
    write_json_atomic(p_startup_env(), layer, mode=0o644)
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
    that does not exist in a dense Qwen; offering them would invite exactly the
    class of unbootable configuration (the m=3 capture crash) that this whole
    validation layer exists to prevent."""
    fams = knob.get("families")
    return not fams or (family_name or "glm52") in fams


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
    if state_values is None:
        try:
            state_values = load_state_file()
        except (ConfigError, ValueError, OSError) as e:
            notes.append(f"state file ignored ({e}); falling back to env + defaults")
            state_values = {}

    fam_name = resolve_family(state_values, env_values)
    fam = family(fam_name)
    fam_defaults = fam.get("defaults", {})

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
    variant = VARIANTS.get(cfg.get("MODEL_VARIANT"), {})
    if variant.get("quantization"):
        args = ["--quantization", variant["quantization"]] + args
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
    out["MODEL_REPO"] = variant["repo"]
    out["MODEL_DIRNAME"] = variant["dirname"]
    out["QUANTIZATION"] = variant["quantization"]
    out["FAMILY_ENV_BLOCK"] = fam.get("env_block", "generic")
    out["SPEC_METHOD"] = fam.get("spec_method", "mtp")
    out["FAMILY_SERVE_ARGS"] = family_serve_args(cfg)
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

    # 0. family coherence ---------------------------------------------------
    if not fam.get("tested"):
        warn("family-untested", ["MODEL_FAMILY"],
             f"{fam['label']}: NOBODY HAS BOOTED THIS IMAGE WITH THIS FAMILY. The serve "
             "arguments are derived from the model card, not measured, and it is unknown "
             "whether this vLLM build implements the architecture at all. Expect to "
             "iterate, and treat a clean boot as the beginning of validation rather than "
             "the end of it.")
    if variant.get("family") and variant["family"] != fam_name:
        err("variant-family-mismatch", ["MODEL_VARIANT", "MODEL_FAMILY"],
            f"model variant '{cfg['MODEL_VARIANT']}' belongs to the "
            f"{variant['family']} family, not {fam_name}. Pick one of: "
            + ", ".join(k for k, v in VARIANTS.items() if v.get("family") == fam_name))
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
    if is_glm and tp != 4:
        warn("tp-off-measured", ["TENSOR_PARALLEL_SIZE"],
             f"every GLM-5.2 number in this repo was measured at TP=4; {tp} is "
             "unexplored territory for this family (and the 753B weights do not fit in "
             "fewer than 4 x 96 GB anyway).")

    # 1. concurrency vs the capture + trellis windows -----------------------
    # GLM/EXL3-SCOPED. The trellis window belongs to the EXL3 kernel; applying
    # these rules to a dense Qwen would reject configurations that are fine.
    m = seqs * (1 + toks)
    cap_max = cfg["MAX_CUDAGRAPH_CAPTURE_SIZE"]
    trellis_max = cfg["VLLM_EXL3_TRELLIS_MAX_M"]
    window = min(cap_max, trellis_max)
    if is_glm and toks > 0 and m > window:
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
        if is_glm and min(sizes) < cfg["VLLM_EXL3_TRELLIS_MIN_M"]:
            err("capture-below-trellis-min", ["CUDAGRAPH_CAPTURE_SIZES"],
                f"a capture size of {min(sizes)} is below the EXL3 trellis window minimum "
                f"({cfg['VLLM_EXL3_TRELLIS_MIN_M']}). Capture then falls into the eager "
                "parity path and the worker dies with 'EXL3 eager parity path entered "
                f"during CUDA graph capture (m={min(sizes)})'. The fix is to raise the "
                "capture size, never to lower the trellis minimum.")

    # 2. the locked trellis minimum -----------------------------------------
    if is_glm and cfg["VLLM_EXL3_TRELLIS_MIN_M"] != 4:
        err("trellis-min-m", ["VLLM_EXL3_TRELLIS_MIN_M"],
            "VLLM_EXL3_TRELLIS_MIN_M must stay 4. Lowering it to 1 makes an EXL3 capture "
            "crash disappear but SILENTLY CORRUPTS OUTPUT: measured 32K needle 0/2 and "
            "370K needle 0/5, pure garbage, while short-prompt checks still passed 6/6.")

    # 3. rank-sliced EXL3 draft vs the base generation -----------------------
    if is_glm and draft == "tr3-override" and cfg["BASE_GENERATION"] == "v20":
        err("tr3-draft-on-v20", ["MTP_DRAFT"],
            "an EXL3 rank-sliced MTP draft (TR3-trellis, separate draft dir) cannot be "
            "CUDA-graph captured on the GG v20 base: SpeculatorCudaGraphManager captures "
            "at m outside the trellis window [4,32] and the engine dies with 'EXL3 eager "
            "parity path entered during CUDA graph capture (m=3)'. m=3 is invariant — it "
            "follows neither MTP_TOKENS nor cudagraph_capture_sizes. Use 'tr3-graft' "
            "(same weights, grafted into the target, and the only draft with long-context "
            "evidence) or a bf16/nvfp4 draft.")

    if is_glm and draft.startswith("tr3-") and cfg["MODEL_VARIANT"] != "exl3-tr3":
        err("tr3-draft-needs-exl3", ["MTP_DRAFT", "MODEL_VARIANT"],
            "the TR3 trellis drafts are layer-78 overlays built for the EXL3-TR3 "
            f"checkpoint; they have no meaning on the {variant['label']} target. Use the "
            "bf16 in-checkpoint draft or an external nvfp4 draft dir.")

    # 4. a draft inherits the target's quantization --------------------------
    if is_glm and draft == "nvfp4" and cfg.get("DRAFT_QUANTIZATION") != "modelopt_fp4":
        err("draft-quant-inherit", ["DRAFT_QUANTIZATION", "MTP_DRAFT"],
            "a draft inherits the target's --quantization unless its speculative config "
            "carries its own, so an NVFP4 draft against an EXL3 target is loaded through "
            "the EXL3 path and throws the identical 'capture (m=3)' error as the v20 "
            "rank-sliced bug. Set DRAFT_QUANTIZATION=modelopt_fp4. This is a config trap, "
            "not a bug.")
    if is_glm and cfg.get("DRAFT_MODEL") and draft in ("tr3-graft", "bf16", "off"):
        warn("draft-model-overrides", ["DRAFT_MODEL", "MTP_DRAFT"],
             "an explicit DRAFT_MODEL takes over draft selection; the MTP draft type is "
             "then only used to decide what happens to the target checkpoint.")

    # 5. nvfp4 KV needs calibrated MLA outer scales --------------------------
    if is_glm and cfg["KV_CACHE_DTYPE"] not in ("fp8", "auto") \
            and not variant["kv_scales_calibrated"]:
        err("kv-nvfp4-uncalibrated", ["KV_CACHE_DTYPE", "MODEL_VARIANT"],
            f"KV dtype {cfg['KV_CACHE_DTYPE']} requires per-checkpoint calibrated MLA "
            f"outer scales and the {variant['label']} checkpoint has none. With nvfp4 KV "
            "long context silently degenerates — measured needle 0/6 at 505K with "
            "degenerate output — while GSM8K, vision and structured output all pass. fp8 "
            "is the safe default; it costs ~1.7x the KV bytes per token.")

    # 6. vision on an EXL3 target -------------------------------------------
    if is_glm and cfg["VISION"] and cfg["MODEL_VARIANT"] == "exl3-tr3":
        warn("vision-long-context", ["VISION"],
             "EXPERIMENTAL / KNOWN BROKEN AT LONG CONTEXT: VISION=1 on the EXL3-TR3 "
             "checkpoint corrupts long-context output — measured 32K needle 0/2 with "
             "degenerate text on BOTH the v20 and pre-v20 bases, while short-prompt and "
             "vision smoke tests passed 6/6. Safe only for short-prompt image work. The "
             "long-context probe will fail this configuration by design.")
    if is_glm and cfg["VISION"] and cfg["MAX_MODEL_LEN"] > 393216:
        warn("vision-kv-pressure", ["VISION", "MAX_MODEL_LEN"],
             f"vision costs ~1.31 GiB/GPU and raises memory pressure enough that even "
             f"MAX_MODEL_LEN 384K has failed KV validation (needed 5.7 GiB, had 3.97). At "
             f"{cfg['MAX_MODEL_LEN']} expect 'To serve at least one request with the "
             "model's max seq len ... larger than the available KV cache memory' at boot. "
             "Reducing MAX_MODEL_LEN is the standard remedy.")
    if is_glm and cfg["VISION"] and ctx.get("vision_available") is False:
        warn("vision-unavailable", ["VISION"],
             "vision assets are not on the volume yet; enabling it triggers a ~1 GB "
             "download during the restart.")

    # 7. memory / pool sanity ------------------------------------------------
    if cfg["GPU_MEMORY_UTILIZATION"] > 0.95:
        warn("gpu-util-high", ["GPU_MEMORY_UTILIZATION"],
             f"{cfg['GPU_MEMORY_UTILIZATION']} leaves very little headroom. The memory "
             "profile varies by a few hundred MiB between boots, so a value that boots "
             "two times in three is not a value that boots — it is a rollback waiting to "
             "happen.")
    if cfg["GPU_BLOCKS_OVERRIDE"] and cfg["GPU_BLOCKS_OVERRIDE"] * 256 < cfg["MAX_MODEL_LEN"]:
        err("pool-smaller-than-context", ["GPU_BLOCKS_OVERRIDE", "MAX_MODEL_LEN"],
            f"the pinned KV pool is {cfg['GPU_BLOCKS_OVERRIDE']} blocks = "
            f"{cfg['GPU_BLOCKS_OVERRIDE'] * 256} tokens, smaller than MAX_MODEL_LEN "
            f"({cfg['MAX_MODEL_LEN']}). vLLM refuses to start when it cannot fit one "
            "max-length request: 'To serve at least one request with the model's max seq "
            "len ... larger than the available KV cache memory'. Raise the override "
            f"to >= {-(-cfg['MAX_MODEL_LEN'] // 256)} blocks, set it to 0 to let vLLM size "
            "the pool, or lower MAX_MODEL_LEN.")
    if cfg["MAX_NUM_BATCHED_TOKENS"] > cfg["MAX_MODEL_LEN"]:
        warn("chunk-bigger-than-context", ["MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN"],
             "the prefill chunk is larger than the whole context window; it will simply "
             "be clamped.")

    # 8. parallelism (DCP is an MLA-path feature: GLM only) -------------------
    if is_glm and int(cfg["TENSOR_PARALLEL_SIZE"]) % int(cfg["DCP"]) != 0:
        err("dcp-divides-tp", ["DCP", "TENSOR_PARALLEL_SIZE"],
            f"decode context parallel size {cfg['DCP']} must divide the tensor-parallel "
            f"size ({cfg['TENSOR_PARALLEL_SIZE']}).")
    if is_glm and int(cfg["DCP"]) < 4 and cfg["MAX_MODEL_LEN"] > 262144:
        warn("dcp-reduces-pool", ["DCP", "MAX_MODEL_LEN"],
             f"DCP={cfg['DCP']} replicates KV instead of sharding it across the 4 ranks, "
             f"cutting the usable pool by about {4 // int(cfg['DCP'])}x. "
             f"MAX_MODEL_LEN {cfg['MAX_MODEL_LEN']} is unlikely to fit.")

    # 9. variant ------------------------------------------------------------
    if not variant["tested"]:
        warn("variant-untested", ["MODEL_VARIANT"],
             f"{variant['label']}: this template's serve arguments for it are derived, "
             "not measured. Expect to iterate, and expect a "
             f"~{variant['download_gib']} GiB download on the next restart.")
    if toks == 0:
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
     "trellis window [4,32]. On the v20 base this is what a rank-sliced EXL3 draft "
     "(MTP draft type 'tr3-override') always does, and it is also what a non-EXL3 draft "
     "does when it inherits the target's --quantization (set DRAFT_QUANTIZATION). Do NOT "
     "widen the window downwards: lowering VLLM_EXL3_TRELLIS_MIN_M silently corrupts "
     "output."),
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
