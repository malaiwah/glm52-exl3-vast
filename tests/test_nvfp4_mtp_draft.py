#!/usr/bin/env python3
"""Regression tests for the standalone Luke NVFP4 MTP78 draft builder."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_nvfp4_mtp_draft.py"


class Nvfp4MtpDraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source = root / "snapshot"
        self.output = root / "draft"
        blobs = root / "blobs"
        self.source.mkdir()
        blobs.mkdir()

        payloads = {
            "model-mtp.safetensors": b"mtp weights",
            "model-mtp-inputscales.safetensors": b"input scales",
        }
        for name, contents in payloads.items():
            blob = blobs / name
            blob.write_bytes(contents)
            # Real Hugging Face snapshots are symlinks into the sibling blobs
            # directory. The builder must resolve these before hard-linking.
            (self.source / name).symlink_to(Path("..") / "blobs" / name)

        (self.source / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "modelopt"}})
        )
        (self.source / "tokenizer.json").write_text("{}")
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 999},
                    "weight_map": {
                        "model.layers.0.target_only": "model-00001-of-00081.safetensors",
                        "model.layers.78.experts": "model-mtp.safetensors",
                        "model.layers.78.input_scale":
                            "model-mtp-inputscales.safetensors",
                    },
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_builder(self):
        return subprocess.run(
            [sys.executable, str(BUILDER), str(self.source), str(self.output)],
            text=True,
            capture_output=True,
        )

    def test_builds_atomic_standalone_draft_from_snapshot_symlinks(self):
        result = self.run_builder()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "model-mtp.safetensors").read_bytes(),
                         b"mtp weights")
        self.assertEqual(
            (self.output / "model-mtp-inputscales.safetensors").read_bytes(),
            b"input scales",
        )
        self.assertFalse((self.output / "model-mtp.safetensors").is_symlink())

        index = json.loads(
            (self.output / "model.safetensors.index.json").read_text()
        )
        self.assertEqual(
            set(index["weight_map"]),
            {
                "model.layers.78.experts",
                "model.layers.78.input_scale",
            },
        )
        self.assertEqual(
            index["metadata"]["total_size"],
            len(b"mtp weights") + len(b"input scales"),
        )
        marker = json.loads((self.output / ".draft-complete").read_text())
        self.assertEqual(marker["tensors"], 2)
        self.assertEqual(marker["source_repo"], "lukealonso/GLM-5.2-NVFP4")

    def test_complete_output_is_idempotent(self):
        self.assertEqual(self.run_builder().returncode, 0)
        result = self.run_builder()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already complete", result.stdout)

    def test_refuses_unknown_existing_output(self):
        self.output.mkdir()
        (self.output / "user-file").write_text("preserve me")
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to replace", result.stderr)
        self.assertEqual((self.output / "user-file").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main(verbosity=2)
