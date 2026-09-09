"""Provenance must follow the encoder loading contract, not import its frontend."""
import hashlib
import os
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import write_runtime_provenance as provenance


class RuntimeProvenanceTests(unittest.TestCase):
    def test_configured_encoder_is_hashed_without_importing_serving_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('vllm', 'b12x', 'torch', 'triton', 'lmcache', 'torchao', 'tilelang'):
                package = root / name
                package.mkdir()
                (package / '__init__.py').write_text('raise AssertionError("do not import GPU runtime")\n')
            encoder = root / 'configured-encoder'
            quantize = encoder / 'modules/quant/exl3_lib/quantize.py'
            quantize.parent.mkdir(parents=True)
            (encoder / '__init__.py').write_text('raise AssertionError("do not import serving frontend")\n')
            content = b'def quantize_exl3():\n    return 1\n'
            quantize.write_bytes(content)
            with mock.patch.object(sys, 'path', [str(root), *sys.path]), \
                 mock.patch.dict(os.environ, {'VLLM_EXL3_ENCODER_SOURCE': str(encoder)}):
                record = provenance.module_records()['exllamav3']
                self.assertEqual(record['encoder']['sha256'], hashlib.sha256(content).hexdigest())
                self.assertEqual(record['package_roots'], [str(encoder)])
                quantize.unlink()
                with self.assertRaises(FileNotFoundError):
                    provenance.module_records()

class LayeredSourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "opt/venv/lib/python3.12/site-packages"
        self.mirror = self.directory / "opt"
        self.modules = {name: {"package_roots": [str(self.root / name)]}
                        for name in ("vllm", "b12x")}
        self.refresh = provenance.apply_glm53_refresh
        self.selected = provenance.apply_glm53_selected
        for installer, payload_dir, mirror in (
                (self.refresh, ROOT / "patches/glm53-refresh", self.mirror / "vllm"),
                (self.selected, ROOT / "patches/glm53-selected", self.mirror)):
            for payload, target, _, _ in installer.resolve_targets(self.root, mirror):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(payload_dir / payload, target)
            original = installer.install_targets
            replacement = mock.patch.object(
                installer, "install_targets",
                side_effect=lambda original=original, mirror=mirror:
                    original(self.root, mirror))
            replacement.start()
            self.addCleanup(replacement.stop)
            replacement = mock.patch.object(installer, "DEFAULT_MIRROR_ROOT", mirror)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_final_sources_preserve_parser_and_record_selected_scheduler_lineage(self):
        records = provenance.installed_overlay_sources(self.modules)
        by_path = {record["path"]: record for record in records}
        selected_ledger = json.loads((ROOT / "patches/glm53-selected/provenance.json").read_text())
        for entry in selected_ledger["files"]:
            for target in (self.root / entry["path"],
                           self.mirror / Path(entry["path"]).parts[0] / entry["path"]):
                record = by_path[str(target)]
                self.assertEqual(record["sha256"], entry["after_sha256"])
                self.assertEqual(record["expected_sha256"], entry["after_sha256"])
        scheduler = by_path[str(self.root / "vllm/v1/core/sched/scheduler.py")]
        self.assertEqual([layer["installer"] for layer in scheduler["source_layers"]],
                         ["apply_glm53_refresh", "apply_glm53_selected"])
        self.assertEqual(scheduler["source_layers"][0]["after_sha256"],
                         scheduler["source_layers"][1]["before_sha256"])
        parser = by_path[str(self.root / "vllm/parser/glm47_moe.py")]
        self.assertEqual(parser["installer"], "apply_glm53_refresh")
        self.assertEqual(parser["sha256"], hashlib.sha256(
            (ROOT / "patches/glm53-refresh/glm47_moe.py").read_bytes()).hexdigest())
        self.assertEqual(parser["role"], "imported_runtime")
        mirrored = by_path[str(self.mirror / "vllm/vllm/parser/glm47_moe.py")]
        self.assertEqual(mirrored["role"], "base_image_source_tree")

    def test_unowned_runtime_under_opt_cannot_be_reported_as_source_mirror(self):
        self.modules["b12x"]["package_roots"] = [str(self.root / "wrong_b12x")]
        with self.assertRaises(RuntimeError):
            provenance.installed_overlay_sources(self.modules)
        self.modules["b12x"]["package_roots"] = [str(self.root / "b12x")]
        records = provenance.installed_overlay_sources(self.modules)
        by_path = {record["path"]: record for record in records}
        relative = "b12x/comm/pcie/pcie_oneshot.py"
        self.assertEqual(by_path[str(self.root / relative)]["role"], "imported_runtime")
        self.assertEqual(by_path[str(self.mirror / "b12x" / relative)]["role"],
                         "base_image_source_tree")

    def test_old_refresh_hash_cannot_claim_selected_runtime_is_installed(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                target = (self.mirror / "vllm" if mirror else self.root) / "vllm/v1/engine/core.py"
                original = target.read_bytes()
                shutil.copyfile(ROOT / "patches/glm53-refresh/engine_core.py", target)
                with self.assertRaisesRegex(RuntimeError, "installed source mismatch"):
                    provenance.installed_overlay_sources(self.modules)
                target.write_bytes(original)

    def test_provenance_supports_absent_optional_source_mirrors(self):
        shutil.rmtree(self.mirror / "vllm")
        shutil.rmtree(self.mirror / "b12x")
        records = provenance.installed_overlay_sources(self.modules)
        self.assertEqual({record["role"] for record in records}, {"imported_runtime"})
        self.assertEqual({record["module"] for record in records}, {"vllm", "b12x"})

    def test_discontinuous_layer_is_rejected_before_emitting_evidence(self):
        targets = self.selected.install_targets()
        changed = []
        for payload, path, before, after in targets:
            if path == self.root / "vllm/config/scheduler.py":
                before = "0" * 64
            changed.append((payload, path, before, after))
        with mock.patch.object(self.selected, "install_targets", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "discontinuous critical source"):
                provenance.installed_overlay_sources(self.modules)



if __name__ == '__main__':
    unittest.main()
