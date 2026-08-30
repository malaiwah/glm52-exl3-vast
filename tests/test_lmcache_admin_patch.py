#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_lmcache_admin_api as patcher  # noqa: E402


SOURCE = """from pathlib import Path

class HTTPAPIRegistry:
    def register_all_apis(self):
        apis_path = Path(__file__).parent / "http_apis"
        apis_package = "lmcache.v1.multiprocess.http_apis"
        for r in discover_api_routers(apis_path, apis_package):
            self.router.include_router(r)
"""


class LMCacheAdminPatchTests(unittest.TestCase):
    def test_patch_keeps_only_info_api_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "http_api_registry.py"
            target.write_text(SOURCE, encoding="utf-8")
            os.chmod(target, 0o640)

            self.assertEqual(patcher.patch(target), "patched")
            patched = target.read_text(encoding="utf-8")
            self.assertIn('if path.stem != "info_api"', patched)
            self.assertIn("exclude=excluded", patched)
            self.assertNotIn(
                "discover_api_routers(apis_path, apis_package):", patched)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(patcher.patch(target), "verified")
            self.assertEqual(
                patcher.patch(target, verify_only=True), "verified")
            self.assertEqual(
                patcher.main([str(target), "--verify-only"]), 0)

    def test_image_applies_and_verifies_the_restriction(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        apply_call = "python3 /opt/scripts/patch_lmcache_admin_api.py;"
        verify_call = (
            "python3 /opt/scripts/patch_lmcache_admin_api.py --verify-only;")
        self.assertEqual(dockerfile.count(apply_call), 1)
        self.assertEqual(dockerfile.count(verify_call), 1)
        self.assertLess(dockerfile.index(apply_call),
                        dockerfile.index(verify_call))

    def test_verify_only_rejects_unpatched_source(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "http_api_registry.py"
            target.write_text(SOURCE, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not applied"):
                patcher.patch(target, verify_only=True)

    def test_unknown_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "http_api_registry.py"
            target.write_text("def changed_upstream():\n    pass\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reviewed source"):
                patcher.patch(target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
