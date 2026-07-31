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
    def test_packaged_manifest_validates_exact_successor_order(self) -> None:
        manifest = ROOT / "patches" / "field-review-r14" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        names = [component["name"] for component in payload["components"]]
        self.assertLess(
            names.index("vllm-semantic-pcie-channels"),
            names.index("vllm-semantic-pcie-finalizer"),
        )
        self.assertLess(
            names.index("sparkinfer-pcie-ipc-hardening"),
            names.index("sparkinfer-pcie-semantic-prewarm"),
        )
        heads = {
            component["name"]: component["tested_head"]
            for component in payload["components"]
        }
        self.assertEqual(
            heads["vllm-semantic-pcie-finalizer"],
            "f99e1e7b8636ca3811ab6d23084ac6da63420dc3",
        )
        self.assertEqual(
            heads["sparkinfer-pcie-semantic-prewarm"],
            "bc62980543b3ca59a9bee971df1b19ce6181964c",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(APPLIER),
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
        self.assertIn("validated 8 field-review patch components", result.stdout)

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
        self,
        manifest: Path,
        site: Path,
        source: Path,
        launcher: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        launcher = launcher or manifest.parent / "launcher"
        launcher.mkdir(exist_ok=True)
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
                "--launcher-bin",
                str(launcher),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def add_successor(self, manifest_path: Path) -> bytes:
        before = b"VALUE = 'after'\n"
        after = b"VALUE = 'final'\n"
        patch = (
            "diff --git a/demo/value.py b/demo/value.py\n"
            "index e0b8fc9..0a7ad57 100644\n"
            "--- a/demo/value.py\n"
            "+++ b/demo/value.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 'after'\n"
            "+VALUE = 'final'\n"
        ).encode()
        patch_path = manifest_path.parent / "successor.patch"
        patch_path.write_bytes(patch)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["components"].append(
            {
                "name": "demo-successor",
                "base_tree": "demo-tested-head",
                "tested_head": "successor-head",
                "patch": patch_path.name,
                "patch_sha256": digest(patch),
                "targets": ["site-packages", "vllm-source"],
                "files": {
                    "demo/value.py": {
                        "before": digest(before),
                        "after": digest(after),
                    }
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return after

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

    def test_overlapping_successor_chain_applies_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            final = self.add_successor(manifest)

            first = self.run_applier(manifest, site, source)
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertEqual((site / "demo/value.py").read_bytes(), final)
            self.assertEqual((source / "demo/value.py").read_bytes(), final)

            second = self.run_applier(manifest, site, source)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(second.stdout.count("already applied"), 4)

    def test_overlapping_successor_resumes_from_prerequisite_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            final = self.add_successor(manifest)
            prerequisite = b"VALUE = 'after'\n"
            (site / "demo/value.py").write_bytes(prerequisite)
            (source / "demo/value.py").write_bytes(prerequisite)

            result = self.run_applier(manifest, site, source)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.count("already applied"), 2)
            self.assertEqual((site / "demo/value.py").read_bytes(), final)
            self.assertEqual((source / "demo/value.py").read_bytes(), final)

    def test_disjoint_components_resume_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            before = b"OTHER = 'before'\n"
            after = b"OTHER = 'after'\n"
            for root in (site, source):
                (root / "other").mkdir()
                (root / "other" / "value.py").write_bytes(before)
                # Simulate the prior bundle having already completed only the
                # independent demo component.
                (root / "demo" / "value.py").write_text(
                    "VALUE = 'after'\n", encoding="utf-8"
                )
            patch = (
                "diff --git a/other/value.py b/other/value.py\n"
                "index 74b944a..94dc80a 100644\n"
                "--- a/other/value.py\n"
                "+++ b/other/value.py\n"
                "@@ -1 +1 @@\n"
                "-OTHER = 'before'\n"
                "+OTHER = 'after'\n"
            ).encode()
            patch_path = manifest.parent / "independent.patch"
            patch_path.write_bytes(patch)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["components"].append(
                {
                    "name": "independent",
                    "base_tree": "independent-base",
                    "tested_head": "independent-head",
                    "patch": patch_path.name,
                    "patch_sha256": digest(patch),
                    "targets": ["site-packages", "vllm-source"],
                    "files": {
                        "other/value.py": {
                            "before": digest(before),
                            "after": digest(after),
                        }
                    },
                }
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            result = self.run_applier(manifest, site, source)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual((site / "other/value.py").read_bytes(), after)
            self.assertEqual((source / "other/value.py").read_bytes(), after)

    def test_discontinuous_successor_chain_is_rejected_by_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _site, _source = self.make_fixture(Path(tmp))
            self.add_successor(manifest)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["components"][1]["files"]["demo/value.py"]["before"] = (
                digest(b"VALUE = 'not-the-prerequisite'\n")
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(APPLIER),
                    "--manifest",
                    str(manifest),
                    "--validate-only",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("patch chain is discontinuous", result.stdout)

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

    def test_launcher_target_is_hash_checked_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, site, source = self.make_fixture(root)
            launcher = root / "launcher"
            launcher.mkdir()
            before = b"PRELOAD='unsafe'\n"
            after = b"PRELOAD='safe'\n"
            (launcher / "runtime-env.sh").write_bytes(before)
            patch = (
                "diff --git a/runtime-env.sh b/runtime-env.sh\n"
                "index 9afc2d4..28b89a8 100644\n"
                "--- a/runtime-env.sh\n"
                "+++ b/runtime-env.sh\n"
                "@@ -1 +1 @@\n"
                "-PRELOAD='unsafe'\n"
                "+PRELOAD='safe'\n"
            ).encode()
            patch_path = manifest_path.parent / "launcher.patch"
            patch_path.write_bytes(patch)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["components"].append(
                {
                    "name": "launcher",
                    "base_tree": "launcher-base",
                    "tested_head": "launcher-head",
                    "patch": patch_path.name,
                    "patch_sha256": digest(patch),
                    "targets": ["launcher-bin"],
                    "files": {
                        "runtime-env.sh": {
                            "before": digest(before),
                            "after": digest(after),
                        }
                    },
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            first = self.run_applier(manifest_path, site, source, launcher)
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertEqual((launcher / "runtime-env.sh").read_bytes(), after)

            second = self.run_applier(manifest_path, site, source, launcher)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn(
                f"already applied at {launcher.resolve()}", second.stdout
            )

    def test_later_unknown_target_is_rejected_before_first_target_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, site, source = self.make_fixture(Path(tmp))
            (source / "demo/value.py").write_text(
                "VALUE = 'unknown'\n", encoding="utf-8"
            )
            result = self.run_applier(manifest, site, source)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn(
                "preflight rejected the source tree before mutation", result.stdout
            )
            self.assertEqual(
                (site / "demo/value.py").read_text(), "VALUE = 'before'\n"
            )
            self.assertEqual(
                (source / "demo/value.py").read_text(), "VALUE = 'unknown'\n"
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
