#!/usr/bin/env python3
"""Self-analysis: ask the model that IS running why the config that isn't failed.

Runs after a rollback, once the known-good config is serving again. Feeds the
failed boot log, the working boot log and the config diff to the local endpoint
and writes a plain-language explanation into the failure directory, which the
landing page renders.

    analyze_failure.py --dir /workspace/.glm-config/failures/2026...Z \
        --base-url http://localhost:8000 --api-key sk-... --model served-model

Deliberately conservative: at most MAX_ATTEMPTS tries per failure, thinking
disabled (this is a summarisation job on a box whose GPUs belong to the user's
workload), and every prompt is capped so a 40 MB log cannot blow the context.
"""
import argparse
import json
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc  # noqa: E402

MAX_ATTEMPTS = 3
FAILED_LOG_CHARS = 14000
GOOD_LOG_CHARS = 3000

SYSTEM = (
    "You are the diagnostic assistant embedded in a multi-model inference appliance. "
    "A user changed the serving configuration, the engine failed to come up or failed "
    "its correctness probe, and the appliance automatically rolled back to the previous "
    "working configuration. Explain, in plain language, what went wrong and what to do "
    "differently. Be concrete about which setting caused it. If the evidence does not "
    "identify a cause, say so plainly instead of guessing."
)

TEMPLATE = """## What changed
{diff}

## Why the change was rejected
{reason}

## Rules this appliance already knows (matched against the failed log)
{signatures}

## Rationale for the settings that changed
{rationales}

## Tail of the FAILED boot log
```
{failed_log}
```

## Tail of the boot log of the WORKING configuration
```
{good_log}
```

Write at most 250 words, in this shape:
1. One sentence: what happened.
2. Which setting is responsible, and the mechanism.
3. What to try instead (concrete values).
Do not invent log lines. Do not recommend lowering VLLM_EXL3_TRELLIS_MIN_M.
"""


def _tail_chars(path, n):
    try:
        with open(path, "rb") as f:
            size = os.path.getsize(path)
            if size > n:
                f.seek(size - n)
                f.readline()
            return f.read().decode("utf-8", "replace")
    except OSError:
        return "(no log captured)"


def ask(base, key, model, prompt, timeout=600):
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
               "max_tokens": 800, "temperature": 0.2,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + key} if key else {})})
    ctx = ssl._create_unverified_context() if base.startswith("https") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        doc = json.loads(r.read().decode("utf-8", "replace"))
    return ((doc.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="model")
    args = ap.parse_args(argv)

    fdir = args.dir
    meta_path = os.path.join(fdir, "meta.json")
    meta = gc.read_json(meta_path)
    if os.path.exists(os.path.join(fdir, "analysis.md")):
        return 0
    attempts = int(meta.get("analysis_attempts", 0))
    if attempts >= MAX_ATTEMPTS:
        return 3
    meta["analysis_attempts"] = attempts + 1
    gc.write_json_atomic(meta_path, meta, mode=0o644)

    try:
        with open(os.path.join(fdir, "diff.txt")) as f:
            diff = f.read().strip() or "(no differences recorded)"
    except OSError:
        diff = "(no differences recorded)"
    changed = [line.split(":")[0] for line in diff.splitlines() if ":" in line]
    rationales = "\n".join(
        f"- {k}: {gc.KNOB_BY_KEY[k]['rationale']}" for k in changed if k in gc.KNOB_BY_KEY
    ) or "(none of the changed keys is a known knob)"
    signatures = "\n".join(f"- {s}" for s in meta.get("signatures") or []) or "(no known signature matched)"

    prompt = TEMPLATE.format(
        diff=diff,
        reason=meta.get("reason") or "(not recorded)",
        signatures=signatures,
        rationales=rationales,
        failed_log=_tail_chars(os.path.join(fdir, "error.log"), FAILED_LOG_CHARS),
        good_log=_tail_chars(os.path.join(gc.p_logs(), "last-good.log"), GOOD_LOG_CHARS))

    try:
        answer = ask(args.base_url, args.api_key, args.model, prompt)
    except Exception as e:
        print(f"!!! self-analysis: endpoint call failed ({e}); will retry", file=sys.stderr)
        return 1
    if not answer:
        print("!!! self-analysis: empty answer; will retry", file=sys.stderr)
        return 1

    header = (f"# Why the {meta.get('ts', '?')} configuration was rolled back\n\n"
              f"*Generated by the running model at "
              f"{gc.utcnow_iso()}. It is a summary of the logs "
              f"below, not an authority — verify before acting.*\n\n")
    body = header + answer + "\n\n---\n\n## Config diff\n\n```\n" + diff + "\n```\n"
    tmp = os.path.join(fdir, "analysis.md.tmp")
    with open(tmp, "w") as f:
        f.write(body)
    os.replace(tmp, os.path.join(fdir, "analysis.md"))
    meta["analysis"] = "analysis.md"
    gc.write_json_atomic(meta_path, meta, mode=0o644)
    print(f">>> self-analysis written to {fdir}/analysis.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
