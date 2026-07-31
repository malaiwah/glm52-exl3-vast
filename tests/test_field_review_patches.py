#!/usr/bin/env python3
"""Fail-closed tests for the r14 field-review source patch applicator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPLIER = ROOT / "scripts" / "apply_field_review_patches.py"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FieldReviewPatchTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        patch_dir = root / "bundle"
        site = root / "site"
        source = root / "source"
        patch_dir.mkdir()
        for target in (site, source):
            (target / "demo").mkdir(parents=True)
            (target / "demo" / "value.py").write_text(
                "VALUE = 'before'\n", encoding="utf-8"
            )

        before = b"VALUE = 'before'\n"
        after = b"VALUE = 'after'\n"
        patch = (
            "diff --git a/demo/value.py b/demo/value.py\n"
            "index 4c45130..e0b8fc9 100644\n"
            "--- a/demo/value.py\n"
            "+++ b/demo/value.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 'before'\n"
            "+VALUE = 'after'\n"
        ).encode()
        (patch_dir / "demo.patch").write_bytes(patch)
        manifest = {
            "schema": 1,
            "components": [
                {
                    "name": "demo",
                    "base_tree": "base-tree",
                    "tested_head": "tested-head",
                    "patch": "demo.patch",
                    "patch_sha256": digest(patch),
                    "targets": ["site-packages", "vllm-source"],
                    "files": {
                        "demo/value.py": {
                            "before": digest(before),
                            "after": digest(after),
                        }
                    },
                }
            ],
        }
        manifest_path = patch_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, site, source

    def run_applier(
        self, manifest: Path, site: Path, source: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(APPLIER),
                "--manifest",
                str(manifest),
                "--site-packages",
                str(site),
                "--vllm-source",
                str(source),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def test_apply_and_idempotent_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            first = self.run_applier(manifest, site, source)
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertEqual(
                (site / "demo/value.py").read_text(), "VALUE = 'after'\n"
            )
            self.assertEqual(
                (source / "demo/value.py").read_text(), "VALUE = 'after'\n"
            )

            second = self.run_applier(manifest, site, source)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(second.stdout.count("already applied"), 2)

    def test_unknown_state_is_rejected_without_touching_other_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            (site / "demo/value.py").write_text(
                "VALUE = 'unknown'\n", encoding="utf-8"
            )
            result = self.run_applier(manifest, site, source)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("refusing mixed/unknown source state", result.stdout)
            self.assertEqual(
                (source / "demo/value.py").read_text(), "VALUE = 'before'\n"
            )

    def test_tampered_patch_is_rejected_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            patch = manifest.parent / "demo.patch"
            patch.write_bytes(patch.read_bytes() + b"\n")
            result = self.run_applier(manifest, site, source)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("patch SHA-256 mismatch", result.stdout)
            self.assertEqual(
                (site / "demo/value.py").read_text(), "VALUE = 'before'\n"
            )


if __name__ == "__main__":
    unittest.main()
