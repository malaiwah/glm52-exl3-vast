#!/usr/bin/env python3
"""Unit tests for the compact live feature suite."""
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import feature_suite as feature  # noqa: E402


class FeatureSuiteTests(unittest.TestCase):
    def test_visible_text_is_content_only(self):
        # The regression this pins down: a miswired reasoning parser routes
        # everything into reasoning_content with an empty content, and the
        # release-gating checks must FAIL on that, not read the reasoning.
        self.assertEqual(feature.visible_text({"content": "answer"}), "answer")
        self.assertEqual(
            feature.visible_text({"content": None,
                                  "reasoning_content": "thought"}), "")
        self.assertEqual(feature.visible_text({"reasoning": "legacy"}), "")
        self.assertEqual(
            feature.reasoning_text({"reasoning_content": "thought"}), "thought")
        self.assertEqual(feature.reasoning_text({"reasoning": "legacy"}), "legacy")

    def test_generated_png_is_valid_data_uri(self):
        value = feature.red_png_data_uri()
        self.assertTrue(value.startswith("data:image/png;base64,"))
        import base64
        raw = base64.b64decode(value.split(",", 1)[1])
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")

    def test_record_never_leaks_unbounded_model_output(self):
        checks = []
        feature.record(checks, "x", False, "a" * 1000, extra=True)
        self.assertEqual(len(checks[0]["detail"]), 400)
        self.assertTrue(checks[0]["extra"])
        self.assertTrue(checks[0]["required"])

    def test_optional_capability_does_not_gate_release(self):
        checks = []
        feature.record(checks, "required", True)
        feature.record(checks, "nice-to-have", False, required=False)
        self.assertTrue(feature.release_ok(checks))
        feature.record(checks, "required-failure", False)
        self.assertFalse(feature.release_ok(checks))

    def test_feature_matrix_exercises_expected_request_shapes(self):
        calls = []

        def fake_json(url, payload=None, key="", timeout=600):
            calls.append((url, payload, key))
            if key == "definitely-wrong":
                raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)
            if url.endswith("/tokenize"):
                return {"count": 3}
            messages = (payload or {}).get("messages") or []
            if messages and messages[-1].get("role") == "tool":
                return {"choices": [{"message": {
                    "content": "Montreal is currently 17 degrees."}}]}
            if payload and payload.get("tools"):
                return {"choices": [{"message": {"tool_calls": [{
                    "id": "call-1",
                    "function": {"name": "get_weather", "arguments":
                                 json.dumps({"city": "Montreal"})}}]}}]}
            if payload and payload.get("response_format"):
                return {"choices": [{"message": {
                    "content": json.dumps({"answer": 42})}}]}
            prompt = str((payload or {}).get("messages", [])[-1:])
            if "19 * 23" in prompt:
                msg = {"content": "437", "reasoning_content": "multiply"}
            elif "What code" in prompt:
                msg = {"content": "COBALT-731"}
            elif "Remember" in prompt:
                msg = {"content": "Acknowledged", "reasoning_content": "remember"}
            else:
                msg = {"content": "BANANA-OK"}
            return {"choices": [{"message": msg}]}

        with mock.patch.object(feature, "json_request", side_effect=fake_json), \
             mock.patch.object(feature, "stream_chat",
                               return_value=("STREAM-OK", {"completion_tokens": 2})):
            checks = feature.run("http://test", "key", "model", vision=False)
        self.assertTrue(all(item["ok"] for item in checks))
        structured = next(
            item for item in checks if item["name"] == "structured-json")
        self.assertTrue(structured["required"])
        structured_thinking = next(
            item for item in checks
            if item["name"] == "structured-json-thinking")
        self.assertTrue(structured_thinking["required"])
        self.assertEqual(
            {item["name"] for item in checks},
            {"auth-rejects-bad-key", "tokenize", "chat-no-thinking",
             "chat-thinking-visible", "streaming-with-usage",
             "multi-turn-preserve-thinking", "structured-json",
             "structured-json-thinking", "tool-call",
             "tool-call-single", "tool-choice-required",
             "tool-result-round-trip", "vision-red-image"})
        self.assertTrue(any(
            payload and payload.get("chat_template_kwargs", {}).get("enable_thinking")
            for _url, payload, _key in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
