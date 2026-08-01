#!/usr/bin/env python3
"""Tests for the immutable GG v20-r17 native-source gate and ledger."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_r17_base.py"
SPEC = importlib.util.spec_from_file_location("verify_r17_base", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class R17BaseGateTests(unittest.TestCase):
    def test_exact_files_and_native_markers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exl3 = root / "exl3.py"
            header = root / "ipc_handle_registry.h"
            source = "\n".join(gate.EXL3_MARKERS).encode()
            exl3.write_bytes(source)
            header.write_bytes(b"header\n")
            expected = {exl3: sha(source), header: sha(b"header\n")}
            with mock.patch.object(gate, "EXPECTED", expected):
                result = gate.verify()
            self.assertEqual(result["status"], "verified")

    def test_unknown_or_partial_native_tree_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exl3 = Path(tmp) / "exl3.py"
            exl3.write_text(gate.EXL3_MARKERS[0], encoding="utf-8")
            with mock.patch.object(gate, "EXPECTED", {exl3: gate.digest(exl3)}):
                with self.assertRaisesRegex(ValueError, "incomplete native"):
                    gate.verify()

    def test_r17_ledger_accounts_for_every_retired_r14_overlay(self) -> None:
        r14 = json.loads(
            (ROOT / "patches" / "field-review-r14" / "manifest.json").read_text()
        )
        r17 = json.loads(
            (ROOT / "patches" / "field-review-r17" / "ledger.json").read_text()
        )
        old_names = {item["name"] for item in r14["components"]}
        ledger = {item["name"]: item for item in r17["retired_r14_overlays"]}
        self.assertEqual(set(ledger), old_names)
        self.assertTrue(
            all(item["status"] in {"included", "superseded", "not-applicable"}
                for item in ledger.values())
        )
        self.assertIn("#222", ledger["vllm-210-211"]["by"])
        self.assertIn("#222", ledger["vllm-exl3-shape-aware-prefill"]["by"])


if __name__ == "__main__":
    unittest.main()
