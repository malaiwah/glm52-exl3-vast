#!/usr/bin/env python3
"""Unit tests for the structured-output/spec-decode compatibility patch."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import patch_structured_output_spec as patch  # noqa: E402


RUNTIME_SOURCE = f"""class StructuredOutputManager:
    def grammar_bitmask(self):
{patch.OLD}
"""


class StructuredOutputPatchTests(unittest.TestCase):
    def test_patch_preflights_only_expected_speculative_rejections(self):
        result, status = patch.patch_text(RUNTIME_SOURCE)
        self.assertEqual(status, "patched")
        self.assertIn(patch.MARKER, result)
        self.assertIn(
            "self._grammar_bitmask[\n"
            "                                        cumulative_index, token >> 5",
            result,
        )
        self.assertIn("& (1 << (token & 31))", result)
        self.assertNotIn("grammar.validate_tokens", result)
        # A committed token bypasses preflight and retains accept_tokens plus
        # the original assertion path.
        self.assertIn("else True", result)
        self.assertIn("grammar.accept_tokens(req_id, [token])", result)
        self.assertIn("elif not post_reasoning_end_in_window:", result)

    def test_patch_is_idempotent(self):
        once, first_status = patch.patch_text(RUNTIME_SOURCE)
        twice, second_status = patch.patch_text(once)
        self.assertEqual(first_status, "patched")
        self.assertEqual(second_status, "already patched")
        self.assertEqual(twice, once)
        self.assertEqual(twice.count(patch.MARKER), 1)

    def test_legacy_validator_patch_is_upgraded(self):
        legacy = RUNTIME_SOURCE.replace(patch.OLD, patch.LEGACY_NEW)
        result, status = patch.patch_text(legacy)
        self.assertEqual(status, "upgraded")
        self.assertIn(patch.MARKER, result)
        self.assertNotIn("grammar.validate_tokens", result)

    def test_unknown_runtime_is_not_modified(self):
        source = "class NewUpstreamImplementation:\n    pass\n"
        result, status = patch.patch_text(source)
        self.assertEqual(status, "anchor not found")
        self.assertEqual(result, source)

    def test_cli_patches_and_compiles_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manager.py"
            target.write_text(RUNTIME_SOURCE)
            self.assertEqual(patch.main([str(target)]), 0)
            compile(target.read_text(), str(target), "exec")
            self.assertEqual(patch.main([str(target)]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
