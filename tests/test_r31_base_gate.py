#!/usr/bin/env python3
"""Tests for the immutable GG v20-r31 and vLLM #258 source gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_r31_base.py"
SPEC = importlib.util.spec_from_file_location("verify_r31_base", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class R31BaseGateTests(unittest.TestCase):
    def test_exact_files_and_native_markers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exl3 = Path(tmp) / "exl3.py"
            source = "\n".join(gate.EXL3_MARKERS).encode()
            exl3.write_bytes(source)
            with mock.patch.object(gate, "EXPECTED", {exl3: sha(source)}), \
                    mock.patch.object(gate, "EXL3_PATH", exl3), \
                    mock.patch.object(gate, "B12X_MARKERS", {}), \
                    mock.patch.object(gate, "VLLM_FILES", {}):
                result = gate.verify()
            self.assertEqual(result["status"], "verified")

    def test_unknown_or_partial_native_tree_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exl3 = Path(tmp) / "exl3.py"
            exl3.write_text(gate.EXL3_MARKERS[0], encoding="utf-8")
            with mock.patch.object(gate, "EXPECTED", {exl3: gate.digest(exl3)}), \
                    mock.patch.object(gate, "EXL3_PATH", exl3), \
                    mock.patch.object(gate, "B12X_MARKERS", {}), \
                    mock.patch.object(gate, "VLLM_FILES", {}):
                with self.assertRaisesRegex(ValueError, "incomplete native r31"):
                    gate.verify()

    def test_release_ledger_pins_included_and_overlay_heads(self) -> None:
        bundle = ROOT / "patches" / "v20-r31-vllm258"
        ledger = json.loads((bundle / "ledger.json").read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(ledger["release"]["manifest_digest"], dockerfile)
        self.assertEqual(
            ledger["composed_sources"]["vllm"],
            "fa13d334a2962756f9f7e9b562deb85387359f42",
        )
        self.assertEqual(
            ledger["composed_sources"]["b12x"],
            "acee6e504209068bd0cbb01cb2b98966bddcf042",
        )
        native = {item["name"]: item for item in ledger["native_fixes"]}
        self.assertEqual(
            native["b12x-target-and-native-mtp-route-pack-prewarm"]["source"],
            "local-inference-lab/b12x#126@6022e6e7c7ea1199a06a27cf5a777c2804b13cfb",
        )
        self.assertEqual(
            native["vllm-mixed-trellis-route-pack-prewarm"]["supersedes"],
            "local-inference-lab/vllm#250@2a34b0760f4d9bbd5d2ff6809238593098bd46ff",
        )
        self.assertEqual(
            ledger["turnkey_overlays"][0]["source"],
            "local-inference-lab/vllm#258@02fb59c5367a3650a0ae6f8805e4f4d3f5cf815f",
        )

    def test_parity_overlay_runtime_hash_is_explicitly_pinned(self) -> None:
        expected = gate.EXPECTED[
            gate.VLLM_SITE
            / "vllm/model_executor/layers/quantization/exl3.py"
        ]
        self.assertIn(
            "7ad8637502b00cb8f95155f305acd4bba5512d884c9ec36a2696386f25e553a3",
            expected,
        )

    def test_vllm_258_bundle_validates_and_build_order_is_fail_closed(self) -> None:
        bundle = ROOT / "patches" / "v20-r31-vllm258"
        manifest = bundle / "manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "apply_field_review_patches.py"),
                "--manifest",
                str(manifest),
                "--validate-only",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("validated 1 field-review patch components", result.stdout)

        dockerfile = (ROOT / "Dockerfile").read_text()
        before = dockerfile.index("verify_r31_base.py")
        apply = dockerfile.index("apply_field_review_patches.py")
        after = dockerfile.rindex("verify_r31_base.py")
        self.assertLess(before, apply)
        self.assertLess(apply, after)


if __name__ == "__main__":
    unittest.main()
