#!/usr/bin/env python3
"""Is every knob in the registry actually CONSUMED by the thing it claims to
control?

    python3 tests/test_knob_wiring.py

The other suites assert the registry is internally consistent. That is not the
same as the entrypoint obeying it, and the difference has cost real money: a
knob can exist, validate, render in the UI, resolve correctly, appear in
config.env — and be read by nobody. `CUDAGRAPH_CAPTURE_SIZES` was that once.

So this walks the registry and requires each knob to be traceable to a consumer:
a `$KNOB` reference in entrypoint.sh, a `%(KNOB)s` placeholder in a family's
serve args, or an explicit entry in INDIRECT below saying who reads it and how.
Adding a knob without wiring it fails here rather than on a rented GPU.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import glm_config as gc  # noqa: E402

# Knobs consumed somewhere other than the entrypoint's own text. Each needs a
# reason, and the reason is checked where it can be.
INDIRECT = {
    "MODEL_FAMILY": ("read by glm_config.resolve/derive to pick the family; the "
                     "entrypoint consumes the derived FAMILY_ENV_BLOCK, "
                     "FAMILY_SERVE_ARGS, SPEC_METHOD and MODEL_REPO"),
    "MODEL_VARIANT": ("read by glm_config.derive to produce MODEL_REPO, "
                      "MODEL_DIRNAME and the family's --quantization"),
    "MODEL_ID": ("read by glm_config.derive to produce MODEL_REPO and "
                 "MODEL_DIRNAME for the custom family"),
    "QUANTIZATION": ("read by glm_config.family_serve_args for the custom "
                     "profile's --quantization flag"),
    "REASONING_PARSER": ("read by glm_config.family_serve_args for the custom "
                         "profile's --reasoning-parser flag"),
    "TOOL_CALL_PARSER": ("read by glm_config.family_serve_args for the custom "
                         "profile's auto-tool-choice flags"),
    "MULTIMODAL": ("read by glm_config.family_serve_args to add or omit "
                   "--language-model-only"),
    "MODEL_OUTPUT_LIMIT": ("read by landing.deployment for generated client "
                           "configuration and quick-chat max_tokens"),
    "MTP_DRAFT": ("read by glm_config.derive, which turns it into MTP78_MODE, "
                  "DRAFT_MODEL and DRAFT_QUANTIZATION — the three values the "
                  "entrypoint actually consumes"),
    "BASE_GENERATION": ("read-only; consumed by the validation matrix "
                        "(tr3-draft-on-v20), never by the serve path"),
    "VLLM_EXL3_TRELLIS_MIN_M": ("exported as an engine env var by config.env and "
                                "read by the EXL3 kernel, not by this script"),
    "VLLM_EXL3_TRELLIS_MAX_M": ("exported as an engine env var by config.env and "
                                "read by the EXL3 kernel, not by this script"),
}

FAILURES, PASSED = [], []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append((name, detail))
        print(f"  FAIL {name}  {detail}")


def main():
    entry = open(os.path.join(REPO, "entrypoint.sh")).read()
    landing = open(os.path.join(REPO, "landing.py")).read()

    # every placeholder any family substitutes into its serve args
    placeholders = set()
    for fam in gc.FAMILIES.values():
        blob = " ".join(fam.get("serve_args", [])) + (fam.get("compilation_config") or "")
        placeholders |= set(re.findall(r"%\((\w+)\)s", blob))

    print("=== every knob has a consumer ===")
    for knob in gc.KNOBS:
        key = knob["key"]
        in_entry = bool(re.search(r"\$\{?" + re.escape(key) + r"[:}\s]", entry))
        in_family = key in placeholders
        indirect = key in INDIRECT
        where = ("entrypoint.sh" if in_entry else
                 "family serve args" if in_family else
                 "indirect: " + INDIRECT.get(key, "") if indirect else "NOWHERE")
        check(f"{key} is consumed ({where[:60]})",
              in_entry or in_family or indirect,
              "declared in the registry but read by nothing")

    print("\n=== family serve-arg placeholders resolve to real knobs ===")
    for p in sorted(placeholders):
        check(f"%({p})s is a registry knob", p in gc.KNOB_BY_KEY,
              "placeholder names a knob that does not exist")

    print("\n=== the derived values the entrypoint reads are all emitted ===")
    # QUANTIZATION is intentionally absent: it reaches vLLM through
    # FAMILY_SERVE_ARGS only, and exporting it too would duplicate the flag.
    derived_keys = ("MODEL_REPO", "MODEL_DIRNAME", "MTP78_MODE",
                    "DRAFT_MODEL", "DRAFT_QUANTIZATION", "FAMILY_ENV_BLOCK",
                    "SPEC_METHOD")
    eff, _s, _n = gc.resolve(state_values={})
    derived = gc.derive(eff)
    for k in derived_keys:
        used = bool(re.search(r"\$\{?" + re.escape(k) + r"[:}\s]", entry))
        check(f"{k} is emitted by derive() and read by the entrypoint",
              k in derived and used,
              f"in derive: {k in derived}, referenced in entrypoint: {used}")
    check("FAMILY_SERVE_ARGS is emitted as an array and expanded in the serve line",
          "FAMILY_SERVE_ARGS" in derived
          and "${FAMILY_SERVE_ARGS[@]" in entry)

    print("\n=== knobs the landing page must not silently hardcode ===")
    # The header used to advertise "512K context / MTP-3 / 4x RTX PRO 6000"
    # regardless of what was actually resolved, which is how two non-defects got
    # diagnosed as defects on a live pod.
    check("no hardcoded 512K context in the landing page",
          "512K context" not in landing, "the header must be family-derived")
    check("no hardcoded MTP-3 in the landing page", "MTP-3" not in landing)
    check("no hardcoded GPU model in the landing page",
          "RTX PRO 6000" not in landing)
    check("the weights total is not a hardcoded constant",
          "WEIGHTS_TOTAL_GIB = 309" not in landing,
          "download size must come from the resolved variant")
    check("the landing page reads the resolved config for its header",
          "resolve()" in landing and "family(" in landing)
    check("chat accepts both GLM and Qwen reasoning delta fields",
          "d.reasoning_content??d.reasoning" in landing)
    check("chat sends the accumulated multi-turn history",
          "messages:wireMessages()" in landing and 'MODEL=$model_js, msgs=[]' in landing)
    check("preserve-thinking is available and defaults off",
          'id=preserve>' in landing and 'id=preserve checked' not in landing)
    check("reasoning-only output remains visible assistant content",
          "const stopped=ctrl&&ctrl.signal.aborted,finalText=answer||reasoning" in landing
          and "if(!answer&&reasoning){out.textContent=reasoning" in landing)
    check("Runpod proxy HTTPS is treated as a secure landing connection",
          "LANDING_TRUST_PROXY_HTTPS" in landing
          and "or TRUST_PROXY_HTTPS" in landing)
    check("chat JavaScript values are safely escaped for inline scripts",
          "model_js=js_literal" in landing and "key_js=js_literal" in landing
          and "ep_js=js_literal" in landing and '.replace("<", "\\\\u003c")' in landing)
    check("credential-bearing pages are never cached or framed",
          'self.send_header("Cache-Control", "no-store")' in landing
          and 'self.send_header("X-Frame-Options", "DENY")' in landing)

    print(f"\n{len(PASSED)} passed, {len(FAILURES)} failed")
    for name, detail in FAILURES:
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
