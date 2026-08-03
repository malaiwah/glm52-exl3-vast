import importlib.util
import unittest
from pathlib import Path


_path = Path(__file__).parents[1] / "scripts" / "patch_vllm_serial_spec_warning.py"
_spec = importlib.util.spec_from_file_location("serial_spec_warning_patch", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _source() -> str:
    return (
        "            scheduled_token_delta = 0\n"
        + _module.WARNING_ANCHOR
        + "                logger.warning_once('low budget')\n"
    )


class SerialSpecWarningPatchTests(unittest.TestCase):
    def test_patch_guards_warning_with_real_drafting_delta(self) -> None:
        patched, status = _module.patch_text(_source())
        self.assertEqual(status, "patched")
        self.assertIn("scheduled_token_delta > 0", patched)
        self.assertIn("max_num_scheduled_tokens < 8192", patched)

    def test_patch_is_idempotent_and_fails_closed(self) -> None:
        patched, _ = _module.patch_text(_source())
        repeated, status = _module.patch_text(patched)
        self.assertEqual(status, "already patched")
        self.assertEqual(repeated, patched)

        with self.assertRaisesRegex(ValueError, "matched 0 times"):
            _module.patch_text("not the expected vLLM source")


if __name__ == "__main__":
    unittest.main()
