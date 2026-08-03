#!/usr/bin/env python3
"""Tests for the immutable GG v20-r20 source gate and ledger."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_r20_base.py"
SPEC = importlib.util.spec_from_file_location("verify_r20_base", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class R20BaseGateTests(unittest.TestCase):
    def test_exact_files_and_native_markers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exl3 = root / "exl3.py"
            source = "\n".join(gate.EXL3_MARKERS).encode()
            exl3.write_bytes(source)
            with mock.patch.object(gate, "EXPECTED", {exl3: sha(source)}), \
                    mock.patch.object(gate, "EXL3_PATH", exl3), \
                    mock.patch.object(gate, "ENV_MARKERS", ()):
                    result = gate.verify()
            self.assertEqual(result["status"], "verified")

    def test_unknown_or_partial_native_tree_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exl3 = Path(tmp) / "exl3.py"
            exl3.write_text(gate.EXL3_MARKERS[0], encoding="utf-8")
            with mock.patch.object(gate, "EXPECTED", {exl3: gate.digest(exl3)}), \
                    mock.patch.object(gate, "EXL3_PATH", exl3):
                with self.assertRaisesRegex(ValueError, "incomplete native r20"):
                    gate.verify()

    def test_ledger_pins_the_same_release_as_the_dockerfile(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r20" / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(ledger["release"]["manifest_digest"], dockerfile)
        self.assertEqual(ledger["composed_sources"]["vllm"],
                         "72c35f14b65857110a7434e9243ca18b8cabd032")

    def test_mixed_trellis_cache_key_field_fix_is_built_and_ledgered(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r20" / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        overlays = {item["name"]: item for item in ledger["turnkey_overlays"]}
        self.assertEqual(
            overlays["sparkinfer-mixed-trellis-tier-cache-key"]["status"],
            "field-fix-pending-upstream",
        )
        self.assertIn(
            "python3 /opt/scripts/patch_sparkinfer_mixed_trellis_cache_key.py",
            dockerfile,
        )
        patcher = ROOT / "scripts" / "patch_sparkinfer_mixed_trellis_cache_key.py"
        self.assertIn(hashlib.sha256(patcher.read_bytes()).hexdigest(), dockerfile)


if __name__ == "__main__":
    unittest.main()
