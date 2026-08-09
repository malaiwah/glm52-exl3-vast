#!/usr/bin/env python3
"""Compact OpenAI-compatible feature qualification for a served model.

The suite keeps generations short so it can accompany every candidate without
distorting rental cost. It covers auth, tokenization, ordinary and thinking
chat, streaming, multi-turn (including preserved reasoning), outbound and
round-trip tool calling, optional vision, and release-gating structured JSON
both with and without thinking because agent clients rely on both shapes.
"""
import argparse
import base64
import binascii
import datetime
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib


def http_request(url, payload=None, key="", timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def json_request(url, payload=None, key="", timeout=120):
    with http_request(url, payload, key, timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def message(doc):
    return ((doc.get("choices") or [{}])[0].get("message") or {})


def visible_text(msg):
    """The client-visible answer: `content` ONLY.

    The suite exists to catch a miswired reasoning parser, and that failure
    mode routes everything into `reasoning_content` while `content` stays
    empty. Falling back to the reasoning fields here let five release-gating
    checks pass on responses every real client would see as empty — the exact
    lie verify_serving.py's complete() already refuses to tell. Reasoning
    presence is asserted separately where it matters."""
    return str(msg.get("content") or "").strip()


def reasoning_text(msg):
    """The reasoning channel, for checks that assert on thinking itself."""
    return str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()


def chat(base, key, model, messages, **extra):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 96,
    }
    payload.update(extra)
    return json_request(base + "/v1/chat/completions", payload, key)


def structured_answer(base, key, model, thinking):
    return message(chat(
        base, key, model,
        [{"role": "user", "content":
          "Think briefly, then return integer 42 as the answer. "
          "Follow the supplied schema exactly."}],
        # Qwen3.6 can spend ~820 tokens reaching the reasoning boundary even
        # for this tiny schema. Keep enough room for the constrained answer so
        # the qualification measures grammar correctness rather than truncation.
        max_tokens=1024 if thinking else 96,
        response_format={
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
        chat_template_kwargs={"enable_thinking": thinking}))


def parse_structured_answer(msg):
    try:
        doc = json.loads(msg.get("content") or "")
    except (TypeError, ValueError):
        return {}, False
    return doc, doc == {"answer": 42}


def stream_chat(base, key, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content":
                      "Reply with exactly STREAM-OK and nothing else."}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    pieces, usage = [], {}
    with http_request(base + "/v1/chat/completions", payload, key) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                packet = json.loads(data)
            except ValueError:
                continue
            usage = packet.get("usage") or usage
            choices = packet.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                # Client-visible stream only: reasoning deltas must not let
                # the streaming gate pass when `content` never arrives.
                pieces.append(str(delta.get("content") or ""))
    return "".join(pieces), usage


def red_png_data_uri():
    """A dependency-free 32x32 red PNG for a deterministic vision smoke."""
    width = height = 32
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def record(checks, name, ok, detail="", required=True, **data):
    checks.append({
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "detail": str(detail)[:400],
        **data,
    })


def release_ok(checks):
    """Optional capabilities are reported without gating the appliance."""
    return all(item["ok"] for item in checks if item.get("required", True))


def run(base, key, model, vision, auth_enabled=None, checkpoint=None):
    """Run every feature check, isolating failures so one bad endpoint call
    cannot discard the rest of the run.

    ``checkpoint`` is invoked with the current checks list after each record so
    partial results survive an interruption, matching benchmark_serving.py's
    incremental-persistence discipline.
    """
    checks = []
    if auth_enabled is None:
        auth_enabled = bool(key)

    def _record(name, ok, detail="", required=True, **data):
        record(checks, name, ok, detail, required=required, **data)
        if checkpoint is not None:
            checkpoint(checks)

    try:
        json_request(base + "/v1/models", key="definitely-wrong")
        _record("auth-rejects-bad-key", not auth_enabled,
                ("bad key was accepted as expected because authentication is disabled"
                 if not auth_enabled else "bad key was accepted"))
    except urllib.error.HTTPError as error:
        _record("auth-rejects-bad-key",
                auth_enabled and error.code in (401, 403),
                f"HTTP {error.code}; authentication configured={auth_enabled}")
    except Exception as error:
        _record("auth-rejects-bad-key", False,
                f"{type(error).__name__}: {error}")

    try:
        tokenized = json_request(
            base + "/tokenize", {"model": model, "prompt": "one two three"}, key)
        _record("tokenize", isinstance(tokenized.get("count"), int),
                f"count={tokenized.get('count')}")
    except Exception as error:
        _record("tokenize", False, f"{type(error).__name__}: {error}")

    try:
        basic = message(chat(
            base, key, model,
            [{"role": "user", "content":
              "Reply with exactly BANANA-OK and nothing else."}],
            chat_template_kwargs={"enable_thinking": False}))
        basic_text = visible_text(basic)
        _record("chat-no-thinking", "BANANA-OK" in basic_text, basic_text)
    except Exception as error:
        _record("chat-no-thinking", False, f"{type(error).__name__}: {error}")

    try:
        thinking = message(chat(
            base, key, model,
            [{"role": "user", "content":
              "Think briefly, then state the result of 19 * 23."}],
            max_tokens=512, chat_template_kwargs={"enable_thinking": True}))
        thinking_text = visible_text(thinking)
        reasoning = thinking.get("reasoning_content") or thinking.get("reasoning") or ""
        _record("chat-thinking-visible", "437" in thinking_text,
                thinking_text, reasoning_field=bool(reasoning))
    except Exception as error:
        _record("chat-thinking-visible", False, f"{type(error).__name__}: {error}")

    try:
        stream_text, usage = stream_chat(base, key, model)
        _record("streaming-with-usage", "STREAM-OK" in stream_text
                and isinstance(usage.get("completion_tokens"), int),
                stream_text, completion_tokens=usage.get("completion_tokens"))
    except Exception as error:
        _record("streaming-with-usage", False, f"{type(error).__name__}: {error}")

    try:
        turn1 = message(chat(
            base, key, model,
            [{"role": "user", "content":
              "Remember the code COBALT-731. Acknowledge briefly."}],
            chat_template_kwargs={"enable_thinking": True}))
        assistant_turn = {"role": "assistant", "content": turn1.get("content") or ""}
        preserved = turn1.get("reasoning_content") or turn1.get("reasoning")
        if preserved:
            assistant_turn["reasoning_content"] = preserved
        turn2 = message(chat(
            base, key, model,
            [{"role": "user", "content":
              "Remember the code COBALT-731. Acknowledge briefly."},
             assistant_turn,
             {"role": "user", "content": "What code did I ask you to remember?"}],
            chat_template_kwargs={"enable_thinking": False}))
        turn2_text = visible_text(turn2)
        _record("multi-turn-preserve-thinking", "COBALT-731" in turn2_text,
                turn2_text, preserved_reasoning=bool(preserved))
    except Exception as error:
        _record("multi-turn-preserve-thinking", False,
                f"{type(error).__name__}: {error}")

    try:
        structured = structured_answer(base, key, model, thinking=False)
        _structured_doc, structured_ok = parse_structured_answer(structured)
        _record("structured-json", structured_ok,
                structured.get("content") or "")
    except Exception as error:
        _record("structured-json", False, f"{type(error).__name__}: {error}")

    try:
        structured_thinking = structured_answer(base, key, model, thinking=True)
        _thinking_doc, thinking_ok = parse_structured_answer(structured_thinking)
        _record("structured-json-thinking", thinking_ok,
                structured_thinking.get("content") or "",
                reasoning_field=bool(
                    structured_thinking.get("reasoning_content")
                    or structured_thinking.get("reasoning")))
    except Exception as error:
        _record("structured-json-thinking", False,
                f"{type(error).__name__}: {error}")

    calls = []
    try:
        tool = message(chat(
            base, key, model,
            [{"role": "user", "content":
              "Call get_weather exactly once for Montreal; do not call it more than once."}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            tool_choice="auto",
            chat_template_kwargs={"enable_thinking": False}))
        calls = tool.get("tool_calls") or []
        _record("tool-call", bool(calls)
                and calls[0].get("function", {}).get("name") == "get_weather",
                json.dumps(calls)[:400])
        signatures = {
            (call.get("function", {}).get("name"),
             call.get("function", {}).get("arguments"))
            for call in calls
        }
        _record("tool-call-single", len(calls) == 1 and len(signatures) == 1,
                f"calls={len(calls)}, unique={len(signatures)}")
    except Exception as error:
        _record("tool-call", False, f"{type(error).__name__}: {error}")
        _record("tool-call-single", False, f"{type(error).__name__}: {error}")

    try:
        forced = message(chat(
            base, key, model,
            [{"role": "user", "content": "Get the weather for Montreal."}],
            max_tokens=48,
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            tool_choice="required",
            chat_template_kwargs={"enable_thinking": False}))
        forced_calls = forced.get("tool_calls") or []
        _record("tool-choice-required", len(forced_calls) == 1,
                f"calls={len(forced_calls)}", required=False)
    except Exception as error:
        _record("tool-choice-required", False, f"{type(error).__name__}: {error}",
                required=False)

    if calls:
        try:
            first = calls[0]
            tool_id = first.get("id") or "qualification-call"
            round_trip = message(chat(
                base, key, model,
                [{"role": "user", "content": "What is the weather in Montreal?"},
                 {"role": "assistant", "content": None, "tool_calls": [first]},
                 {"role": "tool", "tool_call_id": tool_id,
                  "content": "QUALIFICATION-WEATHER-17C"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
                tool_choice="auto",
                chat_template_kwargs={"enable_thinking": False}))
            round_trip_text = visible_text(round_trip)
            _record("tool-result-round-trip",
                    ("17" in round_trip_text
                     and "montreal" in round_trip_text.lower()),
                    round_trip_text)
        except Exception as error:
            _record("tool-result-round-trip", False,
                    f"{type(error).__name__}: {error}")
    else:
        _record("tool-result-round-trip", False,
                "outbound tool call was unavailable")

    if vision:
        try:
            vision_msg = message(chat(
                base, key, model,
                [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": red_png_data_uri()}},
                    {"type": "text",
                     "text": "What is the dominant color? Reply with one word."},
                ]}],
                chat_template_kwargs={"enable_thinking": False}))
            vision_text = visible_text(vision_msg)
            _record("vision-red-image", "red" in vision_text.lower(), vision_text)
        except Exception as error:
            _record("vision-red-image", False, f"{type(error).__name__}: {error}")
    else:
        _record("vision-red-image", True, "skipped: vision disabled",
                skipped=True)
    return checks


def write_result(path, doc):
    blob = json.dumps(doc, indent=1) + "\n"
    if not path:
        sys.stdout.write(blob)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(blob)
    os.replace(tmp, path)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.api_key_file:
        with open(args.api_key_file) as handle:
            key = handle.read().strip()
    else:
        key = os.environ.get("VLLM_API_KEY", "")
    base = args.base_url.rstrip("/")
    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": "",
        "vision_requested": args.vision,
        "checks": [],
        "ok": False,
    }

    def checkpoint(checks):
        # Incremental persistence only makes sense with a destination path.
        # With no --out, write_result streams to stdout, so checkpointing after
        # every check would emit N concatenated JSON documents (unparseable).
        # Mirror benchmark_serving.py: no-op here, single final write below.
        if args.out:
            doc["checks"] = checks
            doc["ok"] = release_ok(checks)
            write_result(args.out, doc)

    try:
        models = json_request(base + "/v1/models", key=key).get("data") or []
    except Exception as error:
        doc["fatal_error"] = (f"model discovery failed: "
                              f"{type(error).__name__}: {error}")
        write_result(args.out, doc)
        return 1
    model = args.model or (models[0]["id"] if models else "")
    doc["model"] = model
    try:
        checks = run(base, key, model, args.vision, auth_enabled=bool(key),
                     checkpoint=checkpoint)
    except BaseException as exc:
        # run() isolates each check, but guard against an unexpected crash so
        # partial results are still persisted with a recorded cause.
        doc["fatal_error"] = f"{type(exc).__name__}: {exc}"
        doc["ok"] = False
        write_result(args.out, doc)
        raise
    # Single final write (unconditional): emits exactly one document to stdout
    # in the no-out case, or the completed result to disk.
    doc["checks"] = checks
    doc["ok"] = release_ok(checks)
    write_result(args.out, doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
