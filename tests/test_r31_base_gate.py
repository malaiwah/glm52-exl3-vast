#!/usr/bin/env python3
"""Tests for the immutable GG v20-r31 runtime-memory source gate."""

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
        bundle = ROOT / "patches" / "v20-r31-memory-stack"
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
        self.assertIn(
            "local-inference-lab/b12x#126@6022e6e7c7ea",
            native["target-and-native-mtp-route-pack-prewarm"]["source"],
        )
        overlays = {
            item["name"]: item["source"] for item in ledger["turnkey_overlays"]
        }
        self.assertIn(
            "vllm#258@cc5a286de507",
            overlays["prompt-logprobs-memory-accounting-and-bounded-v1-runtime"],
        )
        self.assertIn(
            "vllm#270@244d85a6fe99",
            overlays["mixed-prefill-buffer-sharing-and-parity-staging-accounting"],
        )
        self.assertIn(
            "b12x#130@9ead9eaa188c",
            overlays["mixed-trellis-persistent-buffer-layout-contract"],
        )

    def test_runtime_memory_output_hashes_are_explicitly_pinned(self) -> None:
        exl3_expected = gate.EXPECTED[
            gate.VLLM_SITE
            / "vllm/model_executor/layers/quantization/exl3.py"
        ]
        self.assertIn(
            "049ef17bad2072f6bf4cf719f2fa0096dcf5addfbf085f42512122eed6fa7370",
            exl3_expected,
        )
        layout_expected = gate.EXPECTED[
            gate.B12X_SITE / "moe/_shared/w4a16_layout.py"
        ]
        self.assertIn(None, layout_expected)
        self.assertIn(
            "72d7aa9380a9b9654782d9f39051af3e00eeb6b2a0722db5b085cefd222acc4e",
            layout_expected,
        )

    def test_runtime_memory_bundle_validates_and_build_order_is_fail_closed(self) -> None:
        bundle = ROOT / "patches" / "v20-r31-memory-stack"
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
        self.assertIn("validated 2 field-review patch components", result.stdout)

        dockerfile = (ROOT / "Dockerfile").read_text()
        before = dockerfile.index("verify_r31_base.py")
        apply = dockerfile.index("apply_field_review_patches.py")
        after = dockerfile.rindex("verify_r31_base.py")
        self.assertLess(before, apply)
        self.assertLess(apply, after)


if __name__ == "__main__":
    unittest.main()
