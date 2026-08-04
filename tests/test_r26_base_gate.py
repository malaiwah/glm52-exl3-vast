#!/usr/bin/env python3
"""Tests for the immutable GG v20-r26 source gate and ledger."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_r26_base.py"
SPEC = importlib.util.spec_from_file_location("verify_r26_base", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class R26BaseGateTests(unittest.TestCase):
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
                with self.assertRaisesRegex(ValueError, "incomplete native r26"):
                    gate.verify()

    def test_a_pinned_post_overlay_hash_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exl3 = Path(tmp) / "exl3.py"
            source = "\n".join(gate.EXL3_MARKERS).encode()
            exl3.write_bytes(source)
            with mock.patch.object(
                    gate, "EXPECTED", {exl3: ("not-this", sha(source))}), \
                    mock.patch.object(gate, "EXL3_PATH", exl3), \
                    mock.patch.object(gate, "ENV_MARKERS", ()):
                result = gate.verify()
            self.assertEqual(result["status"], "verified")

    def test_ledger_pins_the_same_release_as_the_dockerfile(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r26" / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(ledger["release"]["manifest_digest"], dockerfile)
        self.assertEqual(ledger["composed_sources"]["vllm"],
                         "f5981f14b4d39979bc0d799c020d42002b707257")

    def test_mixed_trellis_runtime_counts_fix_is_built_and_ledgered(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r26" / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        native = {item["name"]: item for item in ledger["native_fixes"]}
        self.assertEqual(
            native["sparkinfer-mixed-trellis-runtime-counts"]["source"],
            "local-inference-lab/sparkinfer#117@cfeee9b42d21c19a74d85ed5576f8387168df53c",
        )
        self.assertIn(
            "edf3790e04252fb3d65507ff778e8af1d497dcd119ec0cd7dd61c3af4d47bc82",
            SCRIPT.read_text(),
        )
        self.assertNotIn(
            "raw.githubusercontent.com/local-inference-lab/sparkinfer",
            dockerfile,
        )

    def test_tp4_dcp4_policy_and_r26_sparkinfer_deltas_are_pinned(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r26" / "ledger.json").read_text())
        native = {item["name"]: item for item in ledger["native_fixes"]}
        self.assertIn("tp4-dcp4-lossless-auto-policy", native)
        self.assertIn("sparkinfer-fp8-gather-validity", native)
        self.assertIn("sparkinfer-online-k6-e8m0-small-row-scales", native)
        source = SCRIPT.read_text()
        for digest in (
            "3bfeb2cf7e2db8fa80f4a9dedc8a7a816793a480ce6850c7618aa4b5e199626c",
            "f560e2179101dd428d32711cefe32d3084c78ee92583b0656404a2268498a726",
            "c30d77498ca00677dae9da552e803f94e7301b58cf26041f82801e6c245d0945",
        ):
            self.assertIn(digest, source)

    def test_serial_mtp_warning_fix_is_built_and_ledgered(self) -> None:
        ledger = json.loads(
            (ROOT / "patches" / "field-review-r26" / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        overlays = {item["name"]: item for item in ledger["turnkey_overlays"]}
        self.assertEqual(
            overlays["vllm-serial-spec-warning"]["status"],
            "field-fix-pending-upstream",
        )
        patcher = ROOT / "scripts" / "patch_vllm_serial_spec_warning.py"
        self.assertIn(hashlib.sha256(patcher.read_bytes()).hexdigest(), dockerfile)
        self.assertIn("python3 /opt/scripts/patch_vllm_serial_spec_warning.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
