#!/usr/bin/env python3
"""Compact OpenAI-compatible feature qualification for a served model.

The suite keeps generations short so it can accompany every candidate without
distorting rental cost. It covers auth, tokenization, ordinary and thinking
chat, streaming, multi-turn (including preserved reasoning), structured JSON,
tool calling, and optional vision.
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


def http_request(url, payload=None, key="", timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def json_request(url, payload=None, key="", timeout=600):
    with http_request(url, payload, key, timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def message(doc):
    return ((doc.get("choices") or [{}])[0].get("message") or {})


def visible_text(msg):
    return str(msg.get("content") or msg.get("reasoning_content")
               or msg.get("reasoning") or "").strip()


def chat(base, key, model, messages, **extra):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 96,
    }
    payload.update(extra)
    return json_request(base + "/v1/chat/completions", payload, key)


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
            packet = json.loads(data)
            usage = packet.get("usage") or usage
            choices = packet.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                pieces.append(str(
                    delta.get("content") or delta.get("reasoning_content")
                    or delta.get("reasoning") or ""))
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


def record(checks, name, ok, detail="", **data):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400], **data})


def run(base, key, model, vision):
    checks = []
    try:
        json_request(base + "/v1/models", key="definitely-wrong")
        record(checks, "auth-rejects-bad-key", False, "bad key was accepted")
    except urllib.error.HTTPError as error:
        record(checks, "auth-rejects-bad-key", error.code in (401, 403),
               f"HTTP {error.code}")

    tokenized = json_request(
        base + "/tokenize", {"model": model, "prompt": "one two three"}, key)
    record(checks, "tokenize", isinstance(tokenized.get("count"), int),
           f"count={tokenized.get('count')}")

    basic = message(chat(
        base, key, model,
        [{"role": "user", "content":
          "Reply with exactly BANANA-OK and nothing else."}],
        chat_template_kwargs={"enable_thinking": False}))
    basic_text = visible_text(basic)
    record(checks, "chat-no-thinking", "BANANA-OK" in basic_text, basic_text)

    thinking = message(chat(
        base, key, model,
        [{"role": "user", "content":
          "Think briefly, then state the result of 19 * 23."}],
        max_tokens=192, chat_template_kwargs={"enable_thinking": True}))
    thinking_text = visible_text(thinking)
    reasoning = thinking.get("reasoning_content") or thinking.get("reasoning") or ""
    record(checks, "chat-thinking-visible", "437" in thinking_text,
           thinking_text, reasoning_field=bool(reasoning))

    stream_text, usage = stream_chat(base, key, model)
    record(checks, "streaming-with-usage", "STREAM-OK" in stream_text
           and isinstance(usage.get("completion_tokens"), int),
           stream_text, completion_tokens=usage.get("completion_tokens"))

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
    record(checks, "multi-turn-preserve-thinking", "COBALT-731" in turn2_text,
           turn2_text, preserved_reasoning=bool(preserved))

    structured = message(chat(
        base, key, model,
        [{"role": "user", "content": "Return the integer 42 as the answer."}],
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
        chat_template_kwargs={"enable_thinking": False}))
    try:
        structured_doc = json.loads(structured.get("content") or "")
    except (TypeError, ValueError):
        structured_doc = {}
    record(checks, "structured-json", structured_doc.get("answer") == 42,
           structured.get("content") or "")

    tool = message(chat(
        base, key, model,
        [{"role": "user", "content": "What is the weather in Montreal?"}],
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
    calls = tool.get("tool_calls") or []
    record(checks, "tool-call", bool(calls)
           and calls[0].get("function", {}).get("name") == "get_weather",
           json.dumps(calls)[:400])

    if vision:
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
        record(checks, "vision-red-image", "red" in vision_text.lower(), vision_text)
    else:
        record(checks, "vision-red-image", True, "skipped: vision disabled",
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
    models = json_request(base + "/v1/models", key=key).get("data") or []
    model = args.model or (models[0]["id"] if models else "")
    checks = run(base, key, model, args.vision)
    doc = {
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base,
        "model": model,
        "vision_requested": args.vision,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }
    write_result(args.out, doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
