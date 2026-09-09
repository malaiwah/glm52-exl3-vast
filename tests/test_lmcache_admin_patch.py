#!/usr/bin/env python3
"""Prove the LMCache MP HTTP restriction against the real Gilded registry bytes.

The fixtures under tests/fixtures/lmcache-admin are the exact installed sources
extracted read-only from the live Gilded Gnosis v20 r34 maintenance container,
so these tests need neither a container nor fastapi.
"""
import ast
import hashlib
import json
import os
import pkgutil
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

SILENT_LOGGER = SimpleNamespace(info=lambda *a, **k: None,
                                warning=lambda *a, **k: None)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/lmcache-admin"
sys.path.insert(0, str(ROOT / "scripts"))

import patch_lmcache_admin_api as patcher  # noqa: E402

# Every *_api module the installed image actually ships under http_apis. Only
# info_api is read-only liveness/version; common_api aggregates
# internal_api_server/common, which includes run_script_api and env_api.
INSTALLED_API_MODULES = (
    "cache_api", "common_api", "config_api", "info_api",
    "quota_api", "reconfigure_api",
)


def gilded_source(fixture):
    """Return the reviewed installed source, preferring live-extracted bytes."""
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    entry = next(e for e in provenance["files"] if e["fixture"] == fixture)
    root = os.environ.get("LIL_GG_INSTALLED_BEFORE")
    path = Path(root) / entry["source_path"] if root else FIXTURES / fixture
    if not path.is_file():
        path = FIXTURES / fixture
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"unreviewed installed LMCache source: {path}")
    return path, data


def exec_without_imports(path, source, namespace):
    """Execute installed module bodies with only their imports substituted."""
    tree = ast.parse(source, str(path))
    tree.body = [
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0)
    ] + [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class Router:
    """Stand-in for fastapi.APIRouter that records inclusion order."""

    def __init__(self, tag=None):
        self.tag = tag
        self.included = []

    def include_router(self, router):
        self.included.append(router)


def load_discovery():
    """The real Gilded discover_api_routers, with only module import stubbed."""
    path, source = gilded_source("router_discovery.py")

    def import_module(full_name):
        return SimpleNamespace(router=Router(full_name.rsplit(".", 1)[-1]))

    namespace = {
        "Path": Path,
        "APIRouter": Router,
        "importlib": SimpleNamespace(import_module=import_module),
        "pkgutil": pkgutil,
        "init_logger": lambda name: SILENT_LOGGER,
    }
    exec_without_imports(path, source, namespace)
    return namespace["discover_api_routers"]


def register(registry_text, api_modules=INSTALLED_API_MODULES):
    """Run register_all_apis from *registry_text* over a stub http_apis tree."""
    with tempfile.TemporaryDirectory() as directory:
        module_path = Path(directory) / "http_api_registry.py"
        module_path.write_text(registry_text, encoding="utf-8")
        apis = Path(directory) / "http_apis"
        apis.mkdir()
        for name in (*api_modules, "dependencies", "schemas"):
            (apis / f"{name}.py").write_text("router = None\n", encoding="utf-8")
        namespace = {
            "Path": Path,
            "APIRouter": Router,
            "FastAPI": Router,
            "discover_api_routers": load_discovery(),
            "init_logger": lambda name: SILENT_LOGGER,
            "__file__": str(module_path),
            "__package__": "lmcache.v1.multiprocess",
        }
        exec_without_imports(module_path, registry_text, namespace)
        app = Router("app")
        registry = namespace["HTTPAPIRegistry"](app)
        registry.register_all_apis()
        assert len(app.included) == 1, app.included
        return [r.tag for r in app.included[0].included]


class LMCacheAdminPatchTests(unittest.TestCase):
    def setUp(self):
        self.path, self.source = gilded_source("http_api_registry.py")
        self.text = self.source.decode("utf-8")

    def test_reviewed_anchors_match_the_installed_registry(self):
        self.assertEqual(self.text.count(patcher.BEFORE), 1)
        self.assertNotIn(patcher.AFTER, self.text)

    def test_installed_registry_registers_every_api_before_patching(self):
        self.assertEqual(sorted(register(self.text)),
                         sorted(INSTALLED_API_MODULES))

    def test_patched_registry_registers_only_read_only_liveness_routes(self):
        patched = self.text.replace(patcher.BEFORE, patcher.AFTER)
        self.assertEqual(register(patched), ["info_api"])

    def test_patch_is_idempotent_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "http_api_registry.py"
            target.write_bytes(self.source)
            os.chmod(target, 0o640)

            self.assertEqual(patcher.patch(target), "patched")
            self.assertEqual(register(target.read_text(encoding="utf-8")),
                             ["info_api"])
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
            target.write_bytes(self.source)
            with self.assertRaisesRegex(RuntimeError, "not applied"):
                patcher.patch(target, verify_only=True)
            self.assertEqual(target.read_bytes(), self.source)

    def test_unknown_source_fails_closed(self):
        for mutation in ("def changed_upstream():\n    pass\n",
                         self.text.replace("apis_package)", "apis_package, )")):
            with self.subTest(mutation=mutation[:32]), \
                    tempfile.TemporaryDirectory() as raw:
                target = Path(raw) / "http_api_registry.py"
                target.write_text(mutation, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "reviewed source"):
                    patcher.patch(target)
                self.assertEqual(target.read_text(encoding="utf-8"), mutation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
