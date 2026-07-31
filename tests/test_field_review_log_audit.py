#!/usr/bin/env python3
"""Tests for the fail-closed field-review server-log audit."""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import field_review_log_audit as log_audit  # noqa: E402


class FieldReviewLogAuditTests(unittest.TestCase):
    def test_clean_ready_log_passes(self):
        report = log_audit.audit(
            "booting\nINFO:     Application startup complete.\nrequest 200 OK\n"
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["ready"])
        self.assertEqual(report["findings"], [])

    def test_response_success_does_not_hide_backend_errors(self):
        report = log_audit.audit(
            "INFO:     Application startup complete.\n"
            "(EngineCore pid=1) ERROR [backend_xgrammar.py] "
            "Failed to advance FSM for request x\n"
            "POST /v1/chat/completions 200 OK\n"
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["counts"]["structured_fsm"], 1)
        self.assertEqual(len(report["findings"]), 1)

    def test_warm_window_can_exclude_deliberate_prime_compilation(self):
        text = (
            "INFO:     Application startup complete.\n"
            "[sparkinfer cute.compile] miss reason=post-engine-start\n"
            "WARM-WINDOW\n"
            "request 200 OK\n"
        )
        full = log_audit.audit(text)
        warm = log_audit.audit(text, start_line=4)
        self.assertFalse(full["ok"])
        self.assertEqual(full["counts"]["post_ready_compile"], 1)
        self.assertTrue(warm["ok"])

    def test_cuda_ipc_warning_is_a_failure(self):
        report = log_audit.audit(
            "INFO:     Application startup complete.\n"
            "Producer process has been terminated before all shared CUDA "
            "tensors released.\n"
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["counts"]["cuda_ipc_lifetime"], 1)

    def test_missing_ready_fails_unless_explicitly_allowed(self):
        self.assertFalse(log_audit.audit("unit-test log\n")["ok"])
        self.assertTrue(
            log_audit.audit("unit-test log\n", require_ready=False)["ok"]
        )

    def test_start_line_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            log_audit.audit("x", start_line=0)

    def test_read_log_preserves_carriage_return_progress_updates(self):
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"progress 10%\rprogress 20%\nready\n")
            stream.flush()
            text = log_audit.read_log(stream.name)
        self.assertEqual(text.count("\n"), 2)
        self.assertEqual(text.count("\r"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
