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

def test_glm_unchanged():
    section("the GLM path is exactly what it was")
    eff, src, _ = resolved(gpus=4)
    check("family defaults to glm52", eff["MODEL_FAMILY"] == "glm52")
    check("TP is 4 on a 4-GPU box, and it is DETECTED not assumed",
          eff["TENSOR_PARALLEL_SIZE"] == 4 and src["TENSOR_PARALLEL_SIZE"] == "detected",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    check("DCP follows TP", eff["DCP"] == "4" and src["DCP"] == "detected")
    check("context is 512K", eff["MAX_MODEL_LEN"] == 524288)
    check("MTP-3", eff["MTP_TOKENS"] == 3)
    check("fp8 KV", eff["KV_CACHE_DTYPE"] == "fp8")
    check("pool pinned at 2048 blocks", eff["GPU_BLOCKS_OVERRIDE"] == 2048)
    check("util 0.93", eff["GPU_MEMORY_UTILIZATION"] == 0.93)
    check("vision on", eff["VISION"] is True)
    check("served name GLM-5.2", eff["SERVED_MODEL_NAME"] == "GLM-5.2")

    args = gc.family_serve_args(eff)
    line = " ".join(args)
    for flag in ("--quantization exl3",
                 "--decode-context-parallel-size 4",
                 "--dcp-comm-backend a2a",
                 "--dcp-kv-cache-interleave-size 64",
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
    check("compilation config still has custom_ops + fuse_allreduce_rms",
          '"custom_ops":["all"]' in line and '"fuse_allreduce_rms":true' in line)
    check("capture sizes are substituted from the knob",
          '"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]' in line, line[-200:])
    d = gc.derive(eff)
    check("the GLM env block is selected", d["FAMILY_ENV_BLOCK"] == "glm52")
    check("spec method is mtp", d["SPEC_METHOD"] == "mtp")
    check("the MTP78 graft is still the default draft", d["MTP78_MODE"] == "graft")
    check("repo unchanged", d["MODEL_REPO"] == "brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw")


def test_qwen_preset():
    section("the Qwen preset")
    eff, src, _ = resolved("qwen36", gpus=1)
    check("TP is the detected GPU count, so a 1-GPU host just works",
          eff["TENSOR_PARALLEL_SIZE"] == 1 and src["TENSOR_PARALLEL_SIZE"] == "detected",
          f"{eff['TENSOR_PARALLEL_SIZE']} / {src['TENSOR_PARALLEL_SIZE']}")
    eff8, src8, _ = resolved("qwen36", gpus=8)
    check("and on an 8-GPU host it uses all 8 rather than leaving 7 idle",
          eff8["TENSOR_PARALLEL_SIZE"] == 8)
    check("economical profile defaults to a 32K context", eff["MAX_MODEL_LEN"] == 32768)
    check("MTP is opt-in for the development profile", eff["MTP_TOKENS"] == 0)
    check("KV dtype is auto, not the MLA fp8 default", eff["KV_CACHE_DTYPE"] == "auto")
    check("the KV pool pin is dropped", eff["GPU_BLOCKS_OVERRIDE"] == 0)
    check("variant follows the family", eff["MODEL_VARIANT"] == "qwen36-nvfp4")

    d = gc.derive(eff)
    line = " ".join(d["FAMILY_SERVE_ARGS"])
    check("repo is the NVFP4 development checkpoint",
          d["MODEL_REPO"] == "nvidia/Qwen3.6-27B-NVFP4")
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
    check("the economical profile is text-only by default",
          "--language-model-only" in line)
    check("the generic env block is selected", d["FAMILY_ENV_BLOCK"] == "generic")
    check("spec method uses vLLM's current mtp name", d["SPEC_METHOD"] == "mtp")
    check("the MTP78 apparatus is forced off", d["MTP78_MODE"] == "off")
    check("no draft dir or draft quantization is derived",
          d["DRAFT_MODEL"] == "" and d["DRAFT_QUANTIZATION"] == "")
    check("vision is forced off", d["VISION"] is False)
    check("the family is flagged untested", gc.family("qwen36")["tested"] is False)


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
                "VLLM_EXL3_TRELLIS_MAX_M", "VLLM_EXL3_TRELLIS_MIN_M",
                "VISION", "VISION_CHUNKS", "BASE_GENERATION"):
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
                                              "MAX_MODEL_LEN": 32768}),
          str(gc.minimize({"MODEL_FAMILY": "qwen36", "MAX_MODEL_LEN": 32768})))
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
    check("nor does the trellis minimum rule", "trellis-min-m" not in ids(f))
    check("nor the EXL3 draft/base rule", "tr3-draft-on-v20" not in ids(f))
    check("nor the vision long-context rule", "vision-long-context" not in ids(f))
    check("nor the DCP rules",
          not {"dcp-divides-tp", "dcp-reduces-pool"} & ids(f), str(ids(f)))

    # ... and still fire on GLM
    eff, _, _ = resolved(MAX_NUM_SEQS=32)
    f = gc.validate(eff)
    check("the concurrency rule still fires on GLM",
          "concurrency-window" in errs(f), str(errs(f)))
    # The trellis minimum is doubly protected: the knob is non-editable AND
    # range-locked to 4, so a state file cannot even get a 1 into the resolver.
    eff, _, _ = resolved(VLLM_EXL3_TRELLIS_MIN_M=1)
    check("a state file cannot lower the trellis minimum at all",
          eff["VLLM_EXL3_TRELLIS_MIN_M"] == 4, str(eff["VLLM_EXL3_TRELLIS_MIN_M"]))
    # ... and if something ever did, the rule still catches it on GLM only.
    forced = dict(eff, VLLM_EXL3_TRELLIS_MIN_M=1)
    check("the rule still fires on GLM when the value is forced",
          "trellis-min-m" in errs(gc.validate(forced)))
    forced_q = dict(resolved("qwen36")[0], VLLM_EXL3_TRELLIS_MIN_M=1)
    check("but not on a family without a trellis",
          "trellis-min-m" not in ids(gc.validate(forced_q)))
    eff, _, _ = resolved(MTP_DRAFT="tr3-override")
    check("the v20 rank-sliced draft rule still fires on GLM",
          "tr3-draft-on-v20" in errs(gc.validate(eff)))


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
    eff, _, _ = resolved("qwen36")
    check("an untested family always warns", "family-untested" in ids(gc.validate(eff)))


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
    for fam, expected in (("glm52", 524288), ("qwen36", 32768)):
        eff, _, _ = resolved(fam)
        check(f"{fam} probes against its own context ceiling ({expected})",
              eff["MAX_MODEL_LEN"] == expected)
    check("the verifier's degeneracy detector is generic",
          "def degenerate" in src and "punctuation" in src)


def test_env_layer_still_wins_over_family():
    section("layering: default < family < env < file")
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


def main():
    tmp = tempfile.mkdtemp(prefix="glm-fam-test-")
    saved = dict(os.environ)
    os.environ["GLM_STATE_DIR"] = os.path.join(tmp, "state")
    os.environ["GLM_RUNTIME_DIR"] = os.path.join(tmp, "run")
    os.makedirs(os.environ["GLM_STATE_DIR"], exist_ok=True)
    os.makedirs(os.environ["GLM_RUNTIME_DIR"], exist_ok=True)
    try:
        test_glm_unchanged()
        test_qwen_preset()
        test_custom_profile()
        test_inapplicable_knobs()
        test_rules_are_family_scoped()
        test_family_coherence_rules()
        test_gpu_count_gate()
        test_long_context_gate_is_family_independent()
        test_env_layer_still_wins_over_family()
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
