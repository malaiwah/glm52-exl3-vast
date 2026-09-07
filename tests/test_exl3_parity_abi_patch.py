import pathlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import patch_exl3_parity_abi as PATCH
import patch_deepseek_mtp as MTP


def parity_source():
    return "class Method:\n    def apply(self, x):\n        out32 = extension(\n" + PATCH.OLD


class Exl3ParityAbiPatchTests(unittest.TestCase):
    def test_repaired_call_executes_against_eight_argument_extension(self):
        source, _ = PATCH.patch_text(parity_source())

        def extension(a, b, c, d, e, f, g, h):
            return SimpleNamespace(to=lambda dtype: (h, dtype))

        namespace = {"extension": extension}
        exec(source, namespace)
        self.assertEqual(namespace["Method"]().apply(SimpleNamespace(dtype="fp16")), (0, "fp16"))
        self.assertEqual(PATCH.patch_text(source)[0], source)

    def test_unknown_duplicate_and_mixed_sources_fail_closed(self):
        for source in ("def unrelated(): pass\n", PATCH.OLD * 2,
                       PATCH.NEW * 2, PATCH.OLD + PATCH.NEW):
            with self.subTest(source=source), self.assertRaises(ValueError):
                PATCH.patch_text(source)

    def test_failed_compilation_retains_current_source_not_stale_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            source = parity_source() + "invalid syntax !!!\n"
            target.write_text(source)
            target.with_suffix(".py.turnkey-original").write_text("stale = True\n")
            with mock.patch.object(PATCH, "TARGET", target):
                self.assertEqual(PATCH.main(), 1)
            self.assertEqual(target.read_text(), source)


class DeepseekMtpPatchTests(unittest.TestCase):
    def test_normalizes_rank_sliced_names_and_skips_unowned_weights(self):
        source = ("class Loader:\n    def load(self, weights):\n" + MTP.OLD
                  + "\n            yield name, loaded_weight\n")
        patched, _ = MTP.patch_text(source)
        namespace = {}
        exec(patched, namespace)
        loader = namespace["Loader"]()
        loader.quant_config = SimpleNamespace(
            normalize_rank_sliced_weight_name=lambda name: {
                "rank0": "owned", "rank1": None, "rotary_emb.inv_freq": "rotary_emb.inv_freq"}[name])
        self.assertEqual(list(loader.load([
            ("rank0", 7), ("rank1", 8), ("rotary_emb.inv_freq", 9)])), [("owned", 7)])
        self.assertEqual(MTP.patch_text(patched)[0], patched)

    def test_markers_and_mixed_anchors_cannot_hide_unpatched_calls(self):
        for source in (MTP.OLD * 2, MTP.NEW * 2, MTP.OLD + MTP.NEW,
                       '# normalize_rank_sliced_weight_name\n' + MTP.OLD):
            with self.subTest(source=source), self.assertRaises(ValueError):
                MTP.patch_text(source)


if __name__ == "__main__":
    unittest.main()
