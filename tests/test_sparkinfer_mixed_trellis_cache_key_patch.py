import importlib.util
from pathlib import Path

_path = (
    Path(__file__).parents[1]
    / "scripts"
    / "patch_sparkinfer_mixed_trellis_cache_key.py"
)
_spec = importlib.util.spec_from_file_location("mixed_trellis_cache_key_patch", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _r20_source() -> str:
    return (
        "class W4A16MixedTrellisKernel:\n"
        + _module.ABI_ANCHOR
        + "    @property\n"
        + "    def __cache_key__(self):\n"
        + "        return (\n"
        + _module.KEY_ANCHOR
        + "        )\n"
    )


def test_patch_adds_tier_partition_and_changes_abi():
    patched, status = _module.patch_text(_r20_source())
    assert status == "patched"
    assert _module.ABI_REPLACEMENT in patched
    assert "self.tier0.num_experts" in patched
    assert "self.tier1.num_experts" in patched


def test_r20_collision_is_removed_for_the_336_layer_splits():
    common = (("driver",), ("dynamic-k3",), ("dynamic-k4",), 1, 1)
    legacy_206_50 = ("w4a16_mixed_trellis", 4, *common)
    legacy_160_96 = ("w4a16_mixed_trellis", 4, *common)
    assert legacy_206_50 == legacy_160_96

    fixed_206_50 = ("w4a16_mixed_trellis", 5, *common[:3], 206, 50, *common[3:])
    fixed_160_96 = ("w4a16_mixed_trellis", 5, *common[:3], 160, 96, *common[3:])
    assert fixed_206_50 != fixed_160_96


def test_patch_is_idempotent_and_fails_closed():
    patched, _ = _module.patch_text(_r20_source())
    repeated, status = _module.patch_text(patched)
    assert status == "already patched"
    assert repeated == patched

    try:
        _module.patch_text("not the r20 SparkInfer source")
    except ValueError as error:
        assert "ABI=0, cache_key=0" in str(error)
    else:
        raise AssertionError("source mismatch must fail closed")
