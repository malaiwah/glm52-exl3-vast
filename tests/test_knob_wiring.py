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
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import glm_config as gc  # noqa: E402

# Knobs consumed somewhere other than the entrypoint's own text.  Each entry is
# machine-checkable so the claim is verified, not trusted:
#   "derives": [KEY, ...]  each KEY must be emitted by gc.derive() AND referenced
#                           in entrypoint.sh (the entrypoint reads the derived value)
#   "serve_args": True      the knob is consumed by gc.family_serve_args() (the
#                           function source is inspected for the key name)
#   "runtime_env": True     gc.derive() emits the knob into PROFILE_RUNTIME_ENV
#   "landing": True         the knob is read by landing.py
#   "config_env": True       the knob is exported via config_cli.py's env command
INDIRECT = {
    "MODEL_FAMILY": {
        "derives": ["FAMILY_ENV_BLOCK", "FAMILY_SERVE_ARGS", "SPEC_METHOD",
                    "MODEL_REPO"],
        "why": "read by glm_config.resolve/derive to pick the family; the "
               "entrypoint consumes the derived values listed above",
    },
    "MODEL_VARIANT": {
        "derives": ["MODEL_REPO", "MODEL_DIRNAME"],
        "why": "read by glm_config.derive to produce MODEL_REPO and "
               "MODEL_DIRNAME (the variant's --quantization reaches vLLM "
               "through FAMILY_SERVE_ARGS, not the entrypoint's own text)",
    },
    "MODEL_ID": {
        "derives": ["MODEL_REPO", "MODEL_DIRNAME"],
        "why": "read by glm_config.derive to produce MODEL_REPO and "
               "MODEL_DIRNAME for the custom family",
    },
    "MTP_DRAFT": {
        "derives": ["MTP78_MODE", "DRAFT_MODEL", "DRAFT_QUANTIZATION"],
        "why": "read by glm_config.derive, which turns it into MTP78_MODE, "
               "DRAFT_MODEL and DRAFT_QUANTIZATION — the three values the "
               "entrypoint actually consumes",
    },
    "QUANTIZATION": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args for the custom "
               "profile's --quantization flag",
    },
    "REASONING_PARSER": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args for the custom "
               "profile's --reasoning-parser flag",
    },
    "TOOL_CALL_PARSER": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args for the custom "
               "profile's auto-tool-choice flags",
    },
    "MULTIMODAL": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args to add or omit "
               "--language-model-only",
    },
    "CLAMP_ROPE_TABLES": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args to add the served "
               "MAX_MODEL_LEN to GLM hf_overrides",
    },
    "MODEL_OUTPUT_LIMIT": {
        "landing": True,
        "why": "read by landing.deployment for generated client "
               "configuration and quick-chat max_tokens",
    },
    "KV_CACHE_MEMORY_BYTES": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args to add the optional "
               "--kv-cache-memory-bytes override",
    },
    "ONLINE_QUANT": {
        "serve_args": True,
        "why": "read by glm_config.family_serve_args for the r20 online "
               "quantization config; derive also emits its runtime cache env",
    },
    "VLLM_PROMPT_LOGPROBS_CHUNK_SIZE": {
        "runtime_env": True,
        "why": "read by glm_config.derive and emitted into the "
               "PROFILE_RUNTIME_ENV array consumed by entrypoint.sh",
    },
    "VLLM_EXL3_TRELLIS_MAX_M": {
        "entrypoint": True,
        "why": "exported and consumed by entrypoint.sh as an EXL3 kernel env "
               "var (the config layer validates it; the entrypoint wires it to "
               "the kernel through the serve env block)",
    },
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
                 "indirect: " + INDIRECT.get(key, {}).get("why", "") if indirect
                 else "NOWHERE")
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
    derived_keys = ("MODEL_REPO", "MODEL_REVISION", "MODEL_DIRNAME", "MTP78_MODE",
                    "DRAFT_MODEL", "DRAFT_QUANTIZATION", "FAMILY_ENV_BLOCK",
                    "SPEC_METHOD")
    # This is a structural wiring audit, so do not inherit a startup snapshot
    # left in /tmp by an earlier integration test (or a local appliance run).
    # In particular, a custom-family snapshot legitimately has no pinned model
    # revision and would make this check depend on test execution order.
    eff, _s, _n = gc.resolve(state_values={}, env_values={})
    derived = gc.derive(eff)
    for k in derived_keys:
        used = bool(re.search(r"\$\{?" + re.escape(k) + r"[:}\s]", entry))
        check(f"{k} is emitted by derive() and read by the entrypoint",
              k in derived and used,
              f"in derive: {k in derived}, referenced in entrypoint: {used}")
    check("FAMILY_SERVE_ARGS is emitted as an array and expanded in the serve line",
          "FAMILY_SERVE_ARGS" in derived
          and "${FAMILY_SERVE_ARGS[@]" in entry)

    print("\n=== indirect knobs: their claims are verified, not trusted ===")
    # Each INDIRECT entry now carries a machine-checkable spec.  The free-text
    # explanations that used to be here could rot silently — a derived key
    # renamed or an entrypoint rewrite would leave the claim stale.  Now we
    # assert the claim against the real source.
    serve_args_src = inspect.getsource(gc.family_serve_args)
    derive_src = inspect.getsource(gc.derive)
    config_cli = open(os.path.join(REPO, "scripts", "config_cli.py")).read()
    for knob, spec in sorted(INDIRECT.items()):
        for dk in spec.get("derives", []):
            check(f"{knob} derives {dk}: emitted by derive()",
                  dk in derived, f"{dk} not in derive() output keys")
            check(f"{knob} derives {dk}: referenced in entrypoint.sh",
                  bool(re.search(r"\$\{?" + re.escape(dk) + r"[:}\s=\[@]", entry)),
                  f"{dk} not referenced in entrypoint.sh")
        if spec.get("serve_args"):
            check(f"{knob} is consumed by family_serve_args()",
                  knob in serve_args_src,
                  f"{knob} not found in family_serve_args source")
        if spec.get("runtime_env"):
            runtime_env = dict(
                item.split("=", 1) for item in derived["PROFILE_RUNTIME_ENV"]
            )
            check(f"{knob} is emitted through PROFILE_RUNTIME_ENV",
                  knob in derive_src and knob in runtime_env,
                  f"{knob} not emitted by derive()")
        if spec.get("landing"):
            check(f"{knob} is read by landing.py",
                  knob in landing, f"{knob} not found in landing.py")
        if spec.get("config_env"):
            check(f"{knob} is exported by config_cli.py",
                  knob in config_cli, f"{knob} not found in config_cli.py")
        if spec.get("entrypoint"):
            check(f"{knob} is consumed by entrypoint.sh",
                  bool(re.search(r"\$\{?" + re.escape(knob) + r"[:}\s=\[@]", entry)),
                  f"{knob} not found in entrypoint.sh")

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
    check("quick chat prepends a presentation system prompt",
          'role:"system",content:CHAT_SYSTEM_PROMPT' in landing and
          "GitHub-flavored Markdown" in landing)
    check("quick chat safely renders Markdown responses",
          "function renderMarkdown(source)" in landing and
          "function escapeHtml(s)" in landing and
          'rel="noopener noreferrer"' in landing)
    check("quick chat trims leading model whitespace",
          "answer.trimStart()" in landing and
          "reasoning.trimStart()" in landing and
          "cleanAnswer=answer.trim()" in landing)
    check("expanded thinking has a scrollable multi-line body",
          'tbody.className="thinkbody"' in landing and
          ".thinkbody{" in landing and "white-space:pre-wrap" in landing)
    check("thinking expands while streaming and collapses only when finished",
          "think.open=true;reasoning+=thought" in landing and
          "think.open=false;" in landing and
          "if(think.open){think.open=false" not in landing)
    check("thinking follows the newest streamed chunk",
          "tbody.scrollTop=tbody.scrollHeight" in landing)
    check("empty answer bubble stays hidden during reasoning",
          'const out=el("div","msg bot","");out.hidden=true' in landing and
          "if(d.content){answer+=d.content;out.hidden=false" in landing)
    check("primary send button becomes the streaming stop control",
          'sendBtn.textContent=active?"■ Stop":"Send"' in landing and
          'sendBtn.classList.toggle("stopmode",active)' in landing and
          "function toggleGeneration(){if(ctrl)ctrl.abort();else send()}" in landing and
          "sendBtn.onclick=toggleGeneration" in landing and
          "id=stop" not in landing)
    check("stopping during reasoning does not promote thinking to the answer",
          'const finalText=cleanAnswer||(!stopped?cleanReasoning:"")' in landing and
          "if(stopped&&!cleanAnswer)" in landing and
          'out.textContent=stopped?"Generation stopped."' in landing)
    check("quick chat advertises its send keyboard shortcut",
          '<kbd>Ctrl</kbd> + <kbd>Enter</kbd> sends' in landing and
          "aria-describedby=sendhint" in landing and
          'e.ctrlKey&&e.key==="Enter"' in landing)
    check("preserve-thinking is available and defaults off",
          'id=preserve>' in landing and 'id=preserve checked' not in landing)
    check("reasoning-only output remains visible assistant content",
          'const finalText=cleanAnswer||(!stopped?cleanReasoning:"")' in landing
          and "if(!cleanAnswer&&cleanReasoning){out.hidden=false;" in landing
          and "out.innerHTML=renderMarkdown(cleanReasoning);think.remove()" in landing)
    check("Runpod proxy trust is per-request (X-Forwarded-Proto), not a blanket flag",
          "LANDING_TRUST_PROXY_HTTPS" in landing
          and "X-Forwarded-Proto" in landing
          and "or TRUST_PROXY_HTTPS" not in landing)
    check("chat JavaScript values are safely escaped for inline scripts",
          "model_js=js_literal" in landing and "key_js=js_literal" in landing
          and "ep_js=js_literal" in landing and '.replace("<", "\\\\u003c")' in landing)
    check("credential-bearing pages are never cached or framed",
          'self.send_header("Cache-Control", "no-store")' in landing
          and 'self.send_header("X-Frame-Options", "DENY")' in landing)
    check("the credential-bearing landing TLS listener requires TLS 1.2+",
          "ctx.minimum_version = ssl.TLSVersion.TLSv1_2" in landing)
    check("current OMP config path and YAML shape are rendered",
          "~/.omp/agent/models.yml" in landing
          and "~/.pi/agent/models.json" not in landing
          and "supportsTools: true" in landing
          and "maxTokens: $output" in landing)
    check("client harnesses link official product pages and installers",
          all(text in landing for text in (
              "https://omp.sh/",
              "curl -fsSL https://omp.sh/install | sh",
              "https://www.anthropic.com/claude-code",
              "curl -fsSL https://claude.ai/install.sh | bash",
              "https://developers.openai.com/codex/",
              "curl -fsSL https://chatgpt.com/codex/install.sh | sh",
              "https://opencode.ai/",
              "curl -fsSL https://opencode.ai/install | bash",
              "function copyInstall(b)",
          )))
    check("client provider ids are model and provider neutral",
          '"glm-vast"' not in landing and "model_provider = \"glm-vast\"" not in landing)
    check("boot highlights are timestamped at the source",
          "printf '[%s] %s\\n'" in entry
          and "date -u '+%Y-%m-%dT%H:%M:%SZ'" in entry)

    print(f"\n{len(PASSED)} passed, {len(FAILURES)} failed")
    for name, detail in FAILURES:
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
