import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "apply_glm53_runtime_overlays.py"
PAYLOADS = REPO / "patches" / "glm53-runtime"
SPEC = importlib.util.spec_from_file_location("apply_glm53_runtime_overlays", SCRIPT)
OVERLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OVERLAY)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Glm53RuntimeOverlayTests(unittest.TestCase):
    def test_image_pins_and_applies_the_qualified_runtime(self):
        dockerfile = (REPO / "Dockerfile").read_text()
        self.assertIn(
            "FROM docker.io/verdictai/glm53-flash-exl3-k4@sha256:"
            "0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692",
            dockerfile,
        )
        self.assertIn("COPY patches/glm53-runtime/ /opt/glm53-runtime/",
                      dockerfile)
        self.assertIn(
            "python3 /opt/scripts/apply_glm53_runtime_overlays.py "
            "/opt/glm53-runtime --verify-only",
            dockerfile,
        )
        self.assertIn(
            "python3 /opt/scripts/apply_glm53_runtime_overlays.py "
            "/opt/glm53-runtime;",
            dockerfile,
        )

    def test_manifest_pins_every_checked_in_payload(self):
        manifest_names = [item[0] for item in OVERLAY.OVERLAYS]
        payload_names = sorted(path.name for path in PAYLOADS.glob("*.py"))
        self.assertEqual(len(manifest_names), 27)
        self.assertEqual(sorted(manifest_names), payload_names)
        self.assertEqual(len(set(manifest_names)), len(manifest_names))
        self.assertEqual(
            len({item[1] for item in OVERLAY.OVERLAYS}),
            len(OVERLAY.OVERLAYS),
        )
        for source_name, _target, _before, after in OVERLAY.OVERLAYS:
            self.assertEqual(OVERLAY.file_sha256(PAYLOADS / source_name), after)
        host_entry = next(
            item for item in OVERLAY.OVERLAYS
            if item[0] == "b12x_w4a16_host.py"
        )
        self.assertEqual(
            host_entry[2],
            "69bc0b31df3da4063d650ed1bd44922d4933c5e54629fe84958d5368cc31c224",
        )

    def test_nope_import_assertions_ignore_nsa_fp8_rope_mode(self):
        smem = (PAYLOADS / "b12x_mla_smem.py").read_text()
        smem_mg = (PAYLOADS / "b12x_mla_smem_mg.py").read_text()
        self.assertEqual(smem.count("fp8_rope=False"), 2)
        self.assertEqual(smem_mg.count("fp8_rope=False"), 1)

    def test_full_glm_dcp_layers_without_indexers_skip_output_validation(self):
        backend = (PAYLOADS / "b12x_mla_sparse.py").read_text()
        self.assertIn(
            "if indexer is not None:\n"
            "            indexer_outputs_physical = "
            "bool(indexer.output_physical_slots)",
            backend,
        )
        self.assertNotIn(
            "else:\n"
            "            indexer_outputs_physical = "
            "not self.indexer_outputs_logical",
            backend,
        )

    def test_mixed_trellis_compiler_gets_exact_route_namespace(self):
        exl3 = (PAYLOADS / "exl3_patched.py").read_text()
        self.assertIn(
            '"route_num_experts" in compile_parameters',
            exl3,
        )
        self.assertIn(
            "elif route_num_experts != sum(",
            exl3,
        )
        self.assertIn(
            "launch = compile_mixed(**compile_kwargs)",
            exl3,
        )

    def test_dense_trellis_allocates_small_capture_scratch(self):
        exl3 = (PAYLOADS / "exl3_patched.py").read_text()
        self.assertNotIn(
            "if rows <= 128 and small_m_scratch is None",
            exl3,
        )
        self.assertIn(
            "return min(out_features * padded_rows, _B12X_TRELLIS_C_TMP_CAP)",
            exl3,
        )


    def test_projection_tight_mixed_trellis_separates_route_namespace(self):
        mixed = (PAYLOADS / "b12x_mixed_trellis.py").read_text()
        host = (PAYLOADS / "b12x_w4a16_host.py").read_text()
        self.assertIn("route_num_experts: int | None = None", mixed)
        self.assertIn(
            "route_num_experts=int(launch.topk_sum.route_num_experts)",
            mixed,
        )
        for symbol in (
            "build_projection_tiered_maps",
            "bind_mixed_trellis",
            "run_bound_mixed_trellis",
        ):
            self.assertIn(f"def {symbol}(", mixed)
        self.assertIn("def route_pack_warmup_token_counts(", host)
        self.assertIn("from .mixed_kernel import (", mixed)
        self.assertTrue((PAYLOADS / "b12x_mixed_kernel.py").is_file())

    def test_kpool_warmup_rejects_inapplicable_architectures_before_import(self):
        warmup = (PAYLOADS / "glm5_kpool_warmup.py").read_text()
        applicability = warmup.index("if pool_size <= 1")
        lazy_import = warmup.index(
            "from vllm.model_executor.layers.sparse_attn_indexer_kpool import"
        )
        self.assertLess(applicability, lazy_import)


    def test_install_verifies_applies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sources = root / "sources"
            sources.mkdir()
            old = b"old runtime\n"
            first = b"first overlay\n"
            second = b"new runtime file\n"
            target_one = root / "runtime" / "one.py"
            target_two = root / "runtime" / "two.py"
            target_one.parent.mkdir()
            target_one.write_bytes(old)
            (sources / "one.py").write_bytes(first)
            (sources / "two.py").write_bytes(second)
            manifest = (
                (
                    "one.py",
                    str(target_one),
                    digest(old),
                    digest(first),
                ),
                (
                    "two.py",
                    str(target_two),
                    None,
                    digest(second),
                ),
            )
            with mock.patch.object(OVERLAY, "OVERLAYS", manifest):
                OVERLAY.install(sources, verify_only=True)
                self.assertEqual(target_one.read_bytes(), old)
                self.assertFalse(target_two.exists())
                OVERLAY.install(sources)
                self.assertEqual(target_one.read_bytes(), first)
                self.assertEqual(target_two.read_bytes(), second)
                OVERLAY.install(sources)
                self.assertEqual(target_one.read_bytes(), first)
                self.assertEqual(target_two.read_bytes(), second)

    def test_mixed_target_state_fails_closed_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sources = root / "sources"
            sources.mkdir()
            old_one = b"old one\n"
            old_two = b"old two\n"
            new_one = b"new one\n"
            new_two = b"new two\n"
            target_one = root / "one.py"
            target_two = root / "two.py"
            target_one.write_bytes(new_one)
            target_two.write_bytes(old_two)
            (sources / "one.py").write_bytes(new_one)
            (sources / "two.py").write_bytes(new_two)
            manifest = (
                ("one.py", str(target_one), digest(old_one), digest(new_one)),
                ("two.py", str(target_two), digest(old_two), digest(new_two)),
            )
            with mock.patch.object(OVERLAY, "OVERLAYS", manifest):
                with self.assertRaisesRegex(RuntimeError, "mixed GLM-5.3"):
                    OVERLAY.install(sources)
            self.assertEqual(target_one.read_bytes(), new_one)
            self.assertEqual(target_two.read_bytes(), old_two)

    def test_unknown_target_state_fails_closed_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sources = root / "sources"
            sources.mkdir()
            old = b"old\n"
            new = b"new\n"
            unknown = b"unknown\n"
            target = root / "target.py"
            target.write_bytes(unknown)
            (sources / "overlay.py").write_bytes(new)
            manifest = (
                ("overlay.py", str(target), digest(old), digest(new)),
            )
            with mock.patch.object(OVERLAY, "OVERLAYS", manifest):
                with self.assertRaisesRegex(RuntimeError, "unknown GLM-5.3"):
                    OVERLAY.install(sources)
            self.assertEqual(target.read_bytes(), unknown)


if __name__ == "__main__":
    unittest.main()
