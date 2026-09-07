#!/usr/bin/env python3
"""Exercise canonical cache-key isolation with the shipped connector helper."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import patch_scopedlmcache_retrieve as installer

# Bytes extracted read-only from the live Gilded Gnosis v20 r34 maintenance
# container; these are the exact pre-overlay installed sources the appliance
# image patches at build time.
GILDED = Path(os.environ.get("LIL_GG_INSTALLED_BEFORE", "/tmp/lil-gg-installed-before"))
CONNECTOR = "lmcache/integration/vllm/lmcache_mp_connector.py"
VLLM_CONFIG = "vllm/config/vllm.py"


def gilded_source(relative):
    path = GILDED / relative
    if not path.is_file():
        raise unittest.SkipTest(f"Gilded installed-before bytes unavailable: {path}")
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_namespace():
    tree = ast.parse(installer.default_payload(layout=True).read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "scopedlmcache_model_name")
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {}
    exec(compile(module, "shipped_lmcache_namespace", "exec"), namespace)
    return namespace[helper.name]


def load_vllm_block_size_rule():
    """Execute the installed vLLM validate_block_size itself, not a paraphrase."""
    tree = ast.parse(gilded_source(VLLM_CONFIG).read_text())
    rule = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == "validate_block_size")
    module = ast.Module(body=[rule], type_ignores=[])
    namespace = {"logger": SimpleNamespace(warning_once=lambda *a, **k: None)}
    exec(compile(module, "installed_vllm_config", "exec"), namespace)
    return namespace[rule.name]


def config(dcp=4, block=64, cp=1, legacy=64):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw"),
        cache_config=SimpleNamespace(block_size=block, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=cp,
            dcp_kv_cache_interleave_size=legacy,
        ),
    )


class LayoutNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.name = load_namespace()

    def test_scheduler_and_worker_share_key_before_after_vllm_normalization(self):
        scheduler = config(cp=1, legacy=64)
        worker = config(cp=64, legacy=64)
        before = deepcopy(scheduler)
        self.assertEqual(self.name(scheduler), self.name(worker))
        self.assertEqual(scheduler, before)

    def test_incompatible_page_layouts_cannot_hit_the_same_key(self):
        configs = [config(), config(cp=1, legacy=1), config(dcp=2), config(block=128)]
        names = [self.name(c) for c in configs]
        self.assertEqual(len(set(names)), len(configs))
        # Old objects lack layout identity and must not become warm hits.
        self.assertNotIn(configs[0].model_config.model, names)

    def test_cp_option_remains_effective_when_legacy_option_is_one(self):
        self.assertEqual(self.name(config(cp=4, legacy=1)),
                         self.name(config(cp=1, legacy=4)))
        self.assertNotEqual(self.name(config(cp=4, legacy=1)),
                            self.name(config(cp=1, legacy=1)))

    def test_dcp_disabled_does_not_apply_legacy_override(self):
        self.assertEqual(self.name(config(dcp=1, cp=1, legacy=64)),
                         self.name(config(dcp=1, cp=1, legacy=1)))

    def test_invalid_interleave_is_rejected_not_silently_reduced(self):
        # legacy=128 exceeds the 64-token block; legacy=3 does not divide it.
        for invalid in (config(legacy=128), config(legacy=3), config(cp=0, legacy=1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.name(invalid)

    def test_non_positive_geometry_is_rejected(self):
        for invalid in (config(dcp=0), config(block=0, legacy=1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.name(invalid)


class GildedPrecedenceParityTests(unittest.TestCase):
    """The namespace must encode exactly the installed vLLM interleave rule."""

    def setUp(self):
        self.name = load_namespace()
        self.rule = load_vllm_block_size_rule()

    def vllm_outcome(self, cfg):
        """Return the effective interleave vLLM settles on, or None if rejected."""
        try:
            self.rule(cfg)
        except Exception:
            return None
        return cfg.parallel_config.cp_kv_cache_interleave_size

    def helper_outcome(self, cfg):
        try:
            name = self.name(cfg)
        except ValueError:
            return None
        match = re.search(r"##appliance-kv-layout-v1-d(\d+)-b(\d+)-i(\d+)$", name)
        self.assertIsNotNone(match, name)
        self.assertEqual(int(match.group(1)), cfg.parallel_config.decode_context_parallel_size)
        self.assertEqual(int(match.group(2)), cfg.cache_config.block_size)
        return int(match.group(3))

    def test_namespace_interleave_matches_installed_vllm_rule(self):
        checked = 0
        for dcp in (2, 4, 8):
            for block in (16, 64, 128):
                for cp in (1, 3, 4, 16, 64, 128):
                    for legacy in (1, 3, 4, 64, 128):
                        cfg = config(dcp=dcp, block=block, cp=cp, legacy=legacy)
                        expected = self.vllm_outcome(config(
                            dcp=dcp, block=block, cp=cp, legacy=legacy))
                        self.assertEqual(
                            self.helper_outcome(cfg), expected,
                            f"dcp={dcp} block={block} cp={cp} legacy={legacy}")
                        checked += 1
        # The matrix must actually contain both accepted and rejected layouts.
        self.assertEqual(checked, 3 * 3 * 6 * 5)

    def test_matrix_covers_rejection_and_acceptance(self):
        self.assertIsNone(self.vllm_outcome(config(dcp=4, block=64, legacy=128)))
        self.assertEqual(self.vllm_outcome(config(dcp=4, block=64, cp=1, legacy=64)), 64)

    def test_installed_vllm_rule_identity_is_pinned(self):
        self.assertEqual(digest(gilded_source(VLLM_CONFIG)),
                         installer.VLLM_CONFIG_SHA256)


class GildedLayoutInstallTests(unittest.TestCase):
    def install_kwargs(self):
        return {"layout": True, "vllm_config_path": gilded_source(VLLM_CONFIG)}

    def test_gilded_connector_matches_the_pinned_before_state(self):
        self.assertEqual(digest(gilded_source(CONNECTOR)),
                         installer.LAYOUT_BEFORE_SHA256)
        self.assertEqual(digest(installer.default_payload(layout=True)),
                         installer.LAYOUT_AFTER_SHA256)

    def test_install_on_gilded_bytes_is_idempotent_and_verifiable(self):
        source = gilded_source(CONNECTOR).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "lmcache_mp_connector.py"
            target.write_bytes(source)
            target.chmod(0o640)
            kwargs = self.install_kwargs()
            with self.assertRaisesRegex(RuntimeError, "not applied"):
                installer.patch(target, verify_only=True, **kwargs)
            self.assertEqual(installer.patch(target, **kwargs), "patched")
            self.assertEqual(digest(target), installer.LAYOUT_AFTER_SHA256)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(installer.patch(target, **kwargs), "verified")
            self.assertEqual(
                installer.patch(target, verify_only=True, **kwargs), "verified")
            self.assertEqual(installer.main([str(target), "--layout",
                                             "--verify-only", "--vllm-config",
                                             str(kwargs["vllm_config_path"])]), 0)

    def test_mutated_target_is_rejected_without_a_write(self):
        mutated = gilded_source(CONNECTOR).read_text() + "\n# local edit\n"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "lmcache_mp_connector.py"
            target.write_text(mutated)
            with self.assertRaisesRegex(RuntimeError, "reviewed source"):
                installer.patch(target, **self.install_kwargs())
            self.assertEqual(target.read_text(), mutated)

    def test_unreviewed_vllm_rules_refuse_layout_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connector.py"
            original = gilded_source(CONNECTOR).read_bytes()
            path.write_bytes(original)
            vllm_config = Path(directory) / "vllm.py"
            vllm_config.write_text("# different normalization contract\n")
            with self.assertRaises(RuntimeError):
                installer.patch(path, layout=True, vllm_config_path=vllm_config)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
