#!/usr/bin/env python3
"""Is the engine actually serving CORRECTLY? — health + short prompt + a
long-context needle probe.

THE DESIGN LESSON THIS FILE EXISTS FOR: a short-prompt health check is NOT
evidence of correctness. Measured on 2026-07-26, several configurations passed
every short-prompt check — including vision smoke tests — while producing pure
garbage past ~32K tokens:

  * nvfp4 KV without calibrated MLA outer scales : needle 0/6 at 505K, GSM8K fine
  * uncalibrated nvfp4 MLA KV on earlier runtimes : long retrieval failed, short passed
  * vision on the EXL3-TR3 target                : needle 0/3 at 32K; 5K image passed

So nothing here reports "healthy" on the strength of /health or a two-line
completion. A verdict carries `long_context_verified` separately, and the
landing page renders that distinction rather than collapsing it.

    verify_serving.py --base-url http://localhost:8000 --api-key sk-... \
        --model served-model --out /tmp/model-runtime/verify.json \
        --max-model-len 524288 --needle-tokens 32768 [--pid 1234]

Exit 0 when the verdict is ok, 1 otherwise. The verdict JSON is always written.
"""
import argparse
import datetime
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import http.client
import socket

FILLER = [
    "The maintenance log for bay {i} recorded nominal pressure and no operator action.",
    "Shift {i} closed with the coolant loop within tolerance and the seals unchanged.",
    "Inventory pass {i} reconciled without discrepancy against the previous manifest.",
    "Telemetry window {i} showed steady draw with no excursions worth escalating.",
    "Review {i} confirmed the checklist was completed in order and countersigned.",
]
CITIES = ["Kyoto", "Reykjavik", "Valparaiso", "Ouagadougou", "Trondheim",
          "Montevideo", "Samarkand", "Wellington"]

# Bounded retry for transient HTTP failures so a single dropped connection or
# timeout does not roll back a known-good config. Non-transient failures
# (e.g. 4xx auth errors) are never retried.
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_S = 5.0


def _is_transient(exc):
    """True for failures worth retrying: connection/timeout errors and 5xx.

    urllib.error.HTTPError is a subclass of URLError, so the HTTPError branch
    must come first; a 4xx response is a definitive failure, not transient.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError,
                        socket.timeout, OSError, http.client.HTTPException)):
        return True
    return False


def _req(url, payload=None, key="", timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    # When PID 1 has installed the /etc/hosts alias mapping the certificate
    # hostname to loopback (GLM_INTERNAL_TLS_VERIFY=1), verify the cert like any
    # client: this is the authoritative correctness gate, so it should not blind
    # itself to a bad certificate. Loopback/self-hosted TLS without that alias
    # (or plain HTTP) keeps the tolerant path.
    ctx = None
    if url.startswith("https"):
        ctx = (ssl.create_default_context()
               if os.environ.get("GLM_INTERNAL_TLS_VERIFY") == "1"
               else ssl._create_unverified_context())
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        body = r.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def wait_health(base, key, timeout, pid=None, poll=10):
    """Block until /health answers. Returns (ok, detail). A dead server PID
    short-circuits: the supervisor handles that case, not us."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pid and not _alive(pid):
            return False, f"server process {pid} exited before /health answered"
        try:
            _req(base + "/health", key=key, timeout=5)
            return True, "healthy"
        except Exception:
            time.sleep(poll)
    return False, f"/health did not answer within {timeout}s"


def discover_model(base, key):
    doc = _req(base + "/v1/models", key=key, timeout=60)
    models = doc.get("data") if isinstance(doc, dict) else None
    if not models or not models[0].get("id"):
        raise RuntimeError("/v1/models returned no served model")
    return models[0]["id"]


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def complete(base, key, model, prompt, max_tokens=64, timeout=300,
             enable_thinking=False):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    doc = _req(base + "/v1/chat/completions", payload, key=key, timeout=timeout)
    choice = (doc.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            f"completion exhausted its {max_tokens}-token output budget")
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def structured_output_probe(base, key, model, timeout=300):
    """Exercise the reasoning→grammar boundary used by agent clients."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                "Think briefly, then return integer 42 as the answer. "
                "Follow the supplied schema exactly."
            ),
        }],
        # Qwen3.6 has empirically needed up to ~820 tokens to cross the
        # reasoning boundary for this probe; leave room for the JSON answer.
        "max_tokens": 1024,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    error = None
    doc = None
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            doc = _req(
                base + "/v1/chat/completions",
                payload,
                key=key,
                timeout=timeout,
            )
            error = None
            break
        except Exception as exc:
            error = exc
            if attempt < TRANSIENT_RETRIES and _is_transient(exc):
                time.sleep(TRANSIENT_BACKOFF_S)
                continue
            break
    if error is not None:
        return {"ok": False, "detail": f"request failed: {error}"}
    msg = ((doc.get("choices") or [{}])[0].get("message") or {})
    try:
        parsed = json.loads(msg.get("content") or "")
    except (TypeError, ValueError):
        parsed = None
    ok = parsed == {"answer": 42}
    return {
        "ok": ok,
        "detail": (
            "thinking + strict JSON schema passed"
            if ok
            else f"unexpected structured document: {parsed!r}"
        ),
        "reasoning_field": bool(
            msg.get("reasoning_content") or msg.get("reasoning")
        ),
    }


DEGENERATE_RE = re.compile(r"^[\s\W_]+$")


def degenerate(text: str) -> str:
    """-> '' when the text looks like language, else why it does not.

    The observed failure mode is not an error, it is text like
    '".,, while.,, and and while,,,,"' — so the detector looks for punctuation
    soup and for one short phrase eating the whole answer."""
    t = text.strip()
    if len(t) < 2:
        return "empty answer"
    if DEGENERATE_RE.match(t):
        return "answer contains no word characters"
    nonword = sum(1 for c in t if not (c.isalnum() or c.isspace()))
    if nonword / max(len(t), 1) > 0.45:
        return f"{100 * nonword / len(t):.0f}% of the answer is punctuation"
    words = t.split()
    if len(words) >= 12:
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        top = max(set(grams), key=grams.count)
        if grams.count(top) / len(grams) > 0.25:
            return f"the phrase {top!r} repeats over a quarter of the answer"
    return ""


def count_tokens(base, key, model, text):
    """Exact count via vLLM's /tokenize; ~3.7 chars/token when unavailable."""
    try:
        doc = _req(base + "/tokenize", {"model": model, "prompt": text}, key=key, timeout=60)
        if isinstance(doc, dict) and isinstance(doc.get("count"), int):
            return doc["count"], True
    except Exception:
        pass
    return int(len(text) / 3.7), False


def build_haystack(target_tokens, depths, seed=20260726):
    """Filler with codes buried at fractional depths. Returns (text, [(city, code)])."""
    rng = random.Random(seed)
    needles = []
    if len(depths) > len(CITIES):
        sys.stderr.write(
            f"warning: {len(depths)} depths requested but only {len(CITIES)} "
            f"city names available; assigning suffixed names to keep needles "
            f"distinct.\n")
    for index, d in enumerate(depths):
        # Each needle must address a unique city so the retrieval question
        # never asks for the same name twice. When there are more depths than
        # city names, append a numeric suffix (e.g. "Kyoto-2") to keep the
        # labels distinct instead of silently colliding.
        base_city = CITIES[index % len(CITIES)]
        city = base_city if index < len(CITIES) else f"{base_city}-{index // len(CITIES) + 1}"
        code = "%s-%04d" % ("".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)),
                            rng.randrange(1000, 9999))
        needles.append((city, code, d))
    # ~18 tokens per filler line is a starting guess; the caller re-measures.
    nlines = max(64, int(target_tokens / 18))
    # Keep a per-run marker in the first cache block. Otherwise a matrix of
    # related probes can accidentally measure prefix-cache reuse instead of a
    # fresh long prefill.
    lines = [f"Independent retrieval trial {seed}; do not use another trial's codes."]
    lines.extend(FILLER[i % len(FILLER)].format(i=i) for i in range(1, nlines))
    for city, code, d in needles:
        pos = min(nlines - 1, max(1, int(nlines * d)))
        lines[pos] = f"IMPORTANT: the access code for {city} is {code}."
    body = "\n".join(lines)
    return body, [(c, k) for c, k, _ in needles]


def _needle_probe_once(base, key, model, target_tokens, depths, timeout,
                       seed, max_tokens):
    """Single needle-probe attempt; may raise on a transient failure."""
    started = time.perf_counter()
    build_target = target_tokens
    text, needles = build_haystack(build_target, depths, seed)
    actual, exact = count_tokens(base, key, model, text)
    # Calibrate against the served tokenizer. Filler token density varies by
    # tokenizer, and a loose estimate can silently turn a claimed 520K probe
    # into a materially shorter request. Two proportional corrections normally
    # land within one percent while keeping the expensive inference count at one.
    for _ in range(2):
        if not actual or abs(actual - target_tokens) / target_tokens <= 0.01:
            break
        lines = text.count("\n") + 1
        per_line = actual / max(lines, 1)
        build_target = int(target_tokens * 18 / max(per_line, 1e-6))
        text, needles = build_haystack(build_target, depths, seed)
        actual, exact = count_tokens(base, key, model, text)
    asked = ", ".join(c for c, _ in needles)
    prompt = (text + "\n\n---\nThe document above contains access codes. "
              f"List the access code for each of these cities: {asked}. "
              "Answer with one 'City: CODE' pair per line and nothing else.")
    answer = complete(base, key, model, prompt, max_tokens=max_tokens,
                      timeout=timeout)
    found = [code for _city, code in needles if code in answer]
    return {
        "attempted": True,
        "seed": seed,
        "target_tokens": target_tokens,
        "tokens": actual,
        "tokens_exact": exact,
        "depths": depths,
        "found": len(found),
        "total": len(needles),
        "degenerate": degenerate(answer),
        "answer_head": answer[:400],
        "ok": len(found) == len(needles) and not degenerate(answer),
        "duration_s": round(time.perf_counter() - started, 3),
    }


def needle_probe(base, key, model, target_tokens, depths, timeout=600,
                 seed=20260726, max_tokens=256):
    """One prefill, all depths — the harness shape that found the 490K result.

    A single transient network failure is retried once so a dropped connection
    during a long, expensive prefill does not by itself fail a good config.
    Non-transient failures (e.g. a 4xx auth error) propagate immediately; the
    caller decides how to record them.
    """
    last_error = None
    for attempt in range(2):
        try:
            return _needle_probe_once(base, key, model, target_tokens, depths,
                                      timeout, seed, max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt == 0 and _is_transient(exc):
                time.sleep(TRANSIENT_BACKOFF_S)
                continue
            break
    raise last_error


SHORT_CHECKS = [
    ("arithmetic", "What is 17 * 23? Reply with only the number.", r"\b391\b"),
    ("factual", "What is the capital of France? Reply with only the city name.", r"(?i)paris"),
    ("instruction", "Reply with exactly the word BANANA and nothing else.", r"(?i)banana"),
]


def short_probe(base, key, model, timeout=180):
    checks = []
    for name, prompt, expect in SHORT_CHECKS:
        # Retry transient failures (connection/timeout) a bounded number of
        # times so a single blip does not fail a good config. Non-transient
        # failures (e.g. 4xx auth errors) fail immediately — retrying those
        # would only mask a genuinely broken configuration.
        answer = None
        error = None
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                answer = complete(
                    base, key, model, prompt, max_tokens=256,
                    timeout=timeout, enable_thinking=True)
                error = None
                break
            except Exception as e:
                error = e
                if attempt < TRANSIENT_RETRIES and _is_transient(e):
                    time.sleep(TRANSIENT_BACKOFF_S)
                    continue
                break
        if error is not None:
            checks.append({"name": name, "ok": False,
                           "detail": f"request failed: {error}"})
            continue
        bad = degenerate(answer)
        ok = bool(re.search(expect, answer)) and not bad
        checks.append({"name": name, "ok": ok,
                       "detail": bad or f"answered {answer[:80]!r}"})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-env", default="",
                    help="read the API key from this environment variable")
    ap.add_argument(
        "--model", default="",
        help="served model alias; defaults to the first entry from /v1/models")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-model-len", type=int, default=524288)
    ap.add_argument("--needle-tokens", type=int, default=32768)
    ap.add_argument("--depths", default="0.15,0.55,0.9")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds to wait for /health")
    ap.add_argument("--pid", default="")
    ap.add_argument("--skip-long-context", action="store_true")
    ap.add_argument("--needle-timeout", type=int, default=600,
                    help="per-request timeout for the needle probe, seconds")
    ap.add_argument(
        "--needle-max-tokens", type=int, default=256,
        help="maximum completion tokens for the needle answer")
    args = ap.parse_args(argv)
    if args.needle_max_tokens < 1:
        ap.error("--needle-max-tokens must be positive")
    if args.api_key_env:
        args.api_key = os.environ.get(args.api_key_env, "")

    base = args.base_url.rstrip("/")
    started = time.time()
    verdict = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "health": False,
        "short_prompt": {"ok": False, "checks": []},
        "structured_output": {"ok": False, "detail": "not attempted"},
        "long_context": {"attempted": False, "ok": False,
                         "detail": "not attempted"},
        "long_context_verified": False,
        "ok": False,
        "reason": "",
    }

    ok, detail = wait_health(base, args.api_key, args.timeout, args.pid)
    verdict["health"] = ok
    if not ok:
        verdict["reason"] = detail
        return _finish(verdict, args, started)

    if not args.model:
        try:
            args.model = discover_model(base, args.api_key)
        except Exception as e:
            verdict["reason"] = f"could not discover served model: {e}"
            return _finish(verdict, args, started)

    verdict["short_prompt"] = short_probe(base, args.api_key, args.model)
    if not verdict["short_prompt"]["ok"]:
        verdict["reason"] = "short-prompt checks failed: " + "; ".join(
            f"{c['name']}: {c['detail']}" for c in verdict["short_prompt"]["checks"]
            if not c["ok"])
        return _finish(verdict, args, started)

    verdict["structured_output"] = structured_output_probe(
        base, args.api_key, args.model
    )
    if not verdict["structured_output"]["ok"]:
        verdict["reason"] = (
            "thinking + structured-output check failed: "
            + verdict["structured_output"]["detail"]
        )
        return _finish(verdict, args, started)

    reserve = max(4096, args.needle_max_tokens + 2048)
    budget = min(args.needle_tokens, max(0, args.max_model_len - reserve))
    if args.skip_long_context or budget < 8192:
        verdict["long_context"] = {
            "attempted": False, "ok": False,
            "detail": ("long-context probe disabled by configuration"
                       if args.skip_long_context else
                       f"context budget {budget} is too small for a meaningful probe")}
        verdict["ok"] = True
        verdict["reason"] = ("serving; SHORT PROMPTS ONLY — long-context correctness is "
                             "NOT verified")
        return _finish(verdict, args, started)

    depths = [float(d) for d in args.depths.split(",") if d.strip()]
    try:
        lc = needle_probe(
            base, args.api_key, args.model, budget, depths,
            timeout=args.needle_timeout, max_tokens=args.needle_max_tokens)
    except Exception as e:
        lc = {"attempted": True, "ok": False, "detail": f"probe failed: {e}"}
    verdict["long_context"] = lc
    verdict["long_context_verified"] = bool(lc.get("ok"))
    verdict["ok"] = bool(lc.get("ok"))
    if not lc.get("ok"):
        if "detail" in lc and "found" not in lc:
            verdict["reason"] = (
                "LONG-CONTEXT PROBE COULD NOT COMPLETE: " + lc["detail"] +
                ". This is an execution/admission failure, not evidence of silent "
                "output corruption; inspect the engine log for the root cause.")
        else:
            verdict["reason"] = (
                "LONG-CONTEXT PROBE FAILED: retrieved {f}/{t} codes from a ~{n} token "
                "context{d}. Short prompts passed, which is exactly the signature of "
                "the known silent-corruption class (for example NVFP4 MLA KV without "
                "model-calibrated scales, or an unqualified vision/runtime "
                "combination)."
            ).format(f=lc.get("found", 0), t=lc.get("total", 0),
                     n=lc.get("tokens", 0),
                     d=" — " + lc["degenerate"] if lc.get("degenerate") else "")
    else:
        verdict["reason"] = "serving; long-context retrieval verified"
    return _finish(verdict, args, started)


def _finish(verdict, args, started):
    verdict["duration_s"] = round(time.time() - started, 1)
    blob = json.dumps(verdict, indent=1) + "\n"
    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(blob)
        os.replace(tmp, args.out)
    else:
        sys.stdout.write(blob)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
