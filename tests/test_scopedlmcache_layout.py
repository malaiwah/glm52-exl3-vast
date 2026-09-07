#!/usr/bin/env python3
"""Exercise canonical cache-key isolation with the shipped connector helper."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import patch_scopedlmcache_retrieve as installer


def load_namespace():
    tree = ast.parse(installer.default_payload(layout=True).read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "scopedlmcache_model_name")
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {}
    exec(compile(module, "shipped_lmcache_namespace", "exec"), namespace)
    return namespace[helper.name]


def config(dcp=4, block=64, cp=1, legacy=64):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw"),
        cache_config=SimpleNamespace(block_size=block),
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
        for invalid in (config(legacy=128), config(legacy=3), config(cp=0, legacy=1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.name(invalid)

    def test_unreviewed_vllm_rules_refuse_layout_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connector.py"
            original = installer.default_payload(layout=True).read_bytes()
            path.write_bytes(original)
            vllm_config = Path(directory) / "vllm.py"
            vllm_config.write_text("# different normalization contract\n")
            with self.assertRaises(RuntimeError):
                installer.patch(path, layout=True, vllm_config_path=vllm_config)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
