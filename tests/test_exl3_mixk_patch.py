import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_exl3_mixk.py"
SPEC = importlib.util.spec_from_file_location("patch_exl3_mixk", SCRIPT)
PATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PATCH)


def fixture_source() -> str:
    return f"""\
class Fixture:
    def config(self, metadata):
{PATCH.E1_OLD}

    def allocate(self, rank_sliced, suffix):
        return allocate(
            preallocate=rank_sliced and suffix in {{"suh", "svh", "trellis"}},
        )

{PATCH.E3_OLD}
        return api

    def apply(self, layer, x, topk_weights, topk_ids):
{PATCH.E5_OLD}
        return runtime, m
"""


class Exl3MixedKPatchTests(unittest.TestCase):
    def test_applies_exact_mixed_k_patch_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text(fixture_source())
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                PATCH.main()
                first = target.read_text()
                self.assertIn(PATCH.MARKER_V5, first)
                self.assertIn("self.mixed_k_values = tuple(k_values)", first)
                self.assertIn('"output_expert_map": mix_tier', first)
                self.assertIn("api.plan_weights(", first)
                self.assertIn('"experts" if "_load_sparkinfer_fused_moe"', first)
                self.assertIn("max_batched_tokens = int(", first)
                self.assertIn("_runtime_owner_token(", first)
                self.assertIn(
                    '"VLLM_EXL3_TRELLIS_MIN_M", _DEFAULT_TRELLIS_MIN_M',
                    first,
                )
                self.assertIn("param.exl3_backing = None", first)
                self.assertIn("torch.cuda.empty_cache()", first)
                PATCH.main()
                self.assertEqual(target.read_text(), first)

    def test_unknown_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text("def unrelated():\n    pass\n")
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                with self.assertRaisesRegex(SystemExit, "ANCHOR MISMATCH"):
                    PATCH.main()

    def test_r14_native_mixed_k_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            source = "\n\n".join(PATCH.NATIVE_R14_MARKERS) + "\n"
            target.write_text(source)
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                PATCH.main()
            self.assertEqual(target.read_text(), source)

    def test_partial_native_mixed_k_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text(PATCH.NATIVE_R14_MARKERS[0] + "\n")
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                with self.assertRaisesRegex(
                    SystemExit, "INCOMPLETE NATIVE MIXED-K API"
                ):
                    PATCH.main()


if __name__ == "__main__":
    unittest.main()
