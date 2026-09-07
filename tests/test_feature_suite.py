#!/usr/bin/env python3
"""Unit tests for the compact live feature suite."""
import io
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest
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

    def test_optional_capability_does_not_gate_release(self):
        checks = []
        feature.record(checks, "required", True)
        feature.record(checks, "nice-to-have", False, required=False)
        self.assertTrue(feature.release_ok(checks))
        feature.record(checks, "required-failure", False)
        self.assertFalse(feature.release_ok(checks))

    def test_stdout_is_one_completed_document(self):
        output = io.StringIO()

        def run(*_args, checkpoint, **_kwargs):
            checks = [{"name": "required", "required": True, "ok": True}]
            checkpoint(checks)
            checkpoint(checks)
            return checks

        with mock.patch.object(feature, "json_request", return_value={"data": [{"id": "m"}]}), \
             mock.patch.object(feature, "run", side_effect=run), \
             mock.patch("sys.stdout", output):
            self.assertEqual(feature.main(["--model", "m"]), 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["complete"])
        self.assertTrue(report["ok"])

    def test_interrupted_feature_suite_retains_evidence_but_never_passes(self):
        def run(*_args, checkpoint, **_kwargs):
            checkpoint([{"name": "first", "required": True, "ok": True}])
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "features.json")
            with mock.patch.object(feature, "json_request", return_value={"data": [{"id": "m"}]}), \
                 mock.patch.object(feature, "run", side_effect=run):
                with self.assertRaises(KeyboardInterrupt):
                    feature.main(["--model", "m", "--out", path])
            report = json.loads(Path(path).read_text())
        self.assertFalse(report["complete"])
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
