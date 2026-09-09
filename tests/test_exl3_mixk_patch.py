import __future__
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
    def test_installed_patch_handles_mixed_metadata_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text(fixture_source())
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                PATCH.main()
                first = target.read_text()
                namespace = {"RoutedExperts": object,
                             "Exl3MoEMethod": type("Exl3MoEMethod", (), {})}
                exec(compile(first, str(target), "exec",
                             flags=__future__.annotations.compiler_flag), namespace)
                config = namespace["Fixture"]()
                config.config({"bits": "mixed", "k_values": [4, 3], "codebook": "mcg"})
                self.assertEqual(config.mixed_k_values, (3, 4))
                self.assertIsNone(config.bits)
                with self.assertRaises(ValueError):
                    config.config({"bits": "mixed", "k_values": [2], "codebook": "mcg"})
                config.config({"bits": 4, "codebook": "mcg"})
                self.assertEqual(config.bits, 4.0)
                self.assertIsNone(config.mixed_k_values)
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
            source = "\n\n".join(marker + "):\n    pass" for marker in PATCH.NATIVE_R14_MARKERS) + "\n"
            target.write_text(source)
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                PATCH.main()
            self.assertEqual(target.read_text(), source)

    def test_partial_native_mixed_k_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text(PATCH.NATIVE_R14_MARKERS[0] + "):\n    pass\n")
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                with self.assertRaisesRegex(
                    SystemExit, "INCOMPLETE NATIVE MIXED-K API"
                ):
                    PATCH.main()

    def test_forged_current_marker_and_duplicate_anchor_do_not_mutate_source(self):
        for source in (fixture_source() + "\n" + PATCH.MARKER_V5,
                       fixture_source() + "\n" + fixture_source()):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "exl3.py"
                target.write_text(source)
                with mock.patch.object(PATCH, "find_exl3", return_value=target):
                    with self.assertRaises(SystemExit):
                        PATCH.main()
                self.assertEqual(target.read_text(), source)

    def test_failed_patch_compilation_preserves_current_source(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            source = fixture_source()
            target.write_text(source)
            with mock.patch.object(PATCH, "find_exl3", return_value=target), \
                 mock.patch.object(PATCH, "APPEND", "invalid syntax !!!"):
                with self.assertRaises(SystemExit):
                    PATCH.main()
            self.assertEqual(target.read_text(), source)


if __name__ == "__main__":
    unittest.main()
