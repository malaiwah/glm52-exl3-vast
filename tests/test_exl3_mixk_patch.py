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


class FakeTensor:
    pass


class FakeTorch:
    Tensor = FakeTensor
    uint8 = object()
    allocations = []

    @classmethod
    def empty(cls, size, *, dtype, device):
        tensor = FakeTensor()
        tensor.size = size
        tensor.dtype = dtype
        tensor.device = device
        cls.allocations.append(tensor)
        return tensor

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
                self.assertIn(PATCH.MARKER_V6, first)
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

    def test_prefill_arena_is_reused_only_at_exact_device_and_capacity(self):
        FakeTorch.allocations = []
        namespace = {
            "torch": FakeTorch,
            "Exl3MoEMethod": type("Exl3MoEMethod", (), {}),
        }
        exec(PATCH.APPEND, namespace)
        shared_arena = namespace["_mixk_shared_prefill_arena"]

        first, reused = shared_arena(("cuda", 0), 1024)
        self.assertFalse(reused)
        same, reused = shared_arena(("cuda", 0), 1024)
        self.assertIs(same, first)
        self.assertTrue(reused)

        larger, reused = shared_arena(("cuda", 0), 2048)
        self.assertFalse(reused)
        other_device, reused = shared_arena(("cuda", 1), 1024)
        self.assertFalse(reused)
        self.assertIsNot(larger, first)
        self.assertIsNot(other_device, first)
        same, reused = shared_arena(("cuda", 0), 1024)
        self.assertIs(same, first)
        self.assertTrue(reused)
        self.assertEqual(len(FakeTorch.allocations), 3)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            shared_arena(("cuda", 0), 0)

    def test_prefill_pool_does_not_capture_decode_arena(self):
        self.assertIn(
            "decode_arena, _ = arena_for([t[\"decode_plan\"] for t in tiers_rt])",
            PATCH.APPEND,
        )
        self.assertIn(
            "share_across_owners=True",
            PATCH.APPEND,
        )

    def test_unknown_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "exl3.py"
            target.write_text("def unrelated():\n    pass\n")
            with mock.patch.object(PATCH, "find_exl3", return_value=target):
                with self.assertRaisesRegex(SystemExit, "ANCHOR MISMATCH"):
                    PATCH.main()


if __name__ == "__main__":
    unittest.main()
