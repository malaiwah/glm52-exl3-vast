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
                self.assertIn(PATCH.MARKER_V3, first)
                self.assertIn("self.mixed_k_values = tuple(k_values)", first)
                self.assertIn("output_expert_map=mix_tier", first)
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


if __name__ == "__main__":
    unittest.main()
