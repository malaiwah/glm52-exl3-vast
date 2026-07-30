import importlib.util
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_exl3_parity_abi.py"
SPEC = importlib.util.spec_from_file_location("patch_exl3_parity_abi", SCRIPT)
PATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PATCH)


class Exl3ParityAbiPatchTests(unittest.TestCase):
    def test_adds_final_extension_argument_once(self):
        source = "prefix\n" + PATCH.OLD + "suffix\n"
        patched, status = PATCH.patch_text(source)
        self.assertEqual(status, "patched")
        self.assertEqual(patched.count(PATCH.NEW), 1)
        self.assertNotIn(PATCH.OLD, patched)

    def test_is_idempotent(self):
        patched, status = PATCH.patch_text("prefix\n" + PATCH.NEW)
        self.assertEqual(status, "already patched")
        self.assertEqual(patched, "prefix\n" + PATCH.NEW)

    def test_unknown_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "source anchor not found"):
            PATCH.patch_text("def unrelated(): pass\n")


if __name__ == "__main__":
    unittest.main()
