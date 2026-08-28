#!/usr/bin/env python3
"""Fail closed unless the immutable GG v20-r28 source contract is complete."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


EXL3_PATH = Path("/opt/vllm/vllm/model_executor/layers/quantization/exl3.py")
ENV_PATH = Path("/opt/vllm/vllm/envs.py")
MIXED_TRELLIS_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/"
    "kernels/w4a16/mixed_trellis.py"
)
PCIE_DCP_A2A_CUDA_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/"
    "pcie_dcp_a2a.cu"
)
PCIE_DCP_A2A_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/"
    "pcie_dcp_a2a.py"
)
MICRO_KERNEL_PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/"
    "kernels/micro.py"
)
VLLM_CONFIG_PATH = Path("/opt/vllm/vllm/config/vllm.py")
DCP_POLICY_PATH = Path("/usr/local/bin/glm52-dcp-prefill-policy.sh")

EXPECTED = {
    EXL3_PATH:
        "4165eec4f61aadb62be270600d84ca1db04a4b51ae35777d573db70393ab10fe",
    Path(
        "/opt/venv/lib/python3.12/site-packages/vllm/"
        "model_executor/layers/quantization/exl3.py"
    ): (
        "4165eec4f61aadb62be270600d84ca1db04a4b51ae35777d573db70393ab10fe",
    ),
    ENV_PATH:
        "8d7bfe52c1c8883ea452cd6488106875eb06906459e49dd9aaebe8a03d29e6b5",
    Path("/opt/venv/lib/python3.12/site-packages/vllm/envs.py"):
        "8d7bfe52c1c8883ea452cd6488106875eb06906459e49dd9aaebe8a03d29e6b5",
    Path(
        "/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/"
        "ipc_handle_registry.h"
    ): "74caba86e4b73941d064e040e946caa19870374349e46ad43cdfc5c220343439",
    Path("/usr/local/bin/glm52-pcie-runtime-env.sh"):
        "e3a35eefabd13f94beb8122d983c4b98bc6373575d7d9ea3e81fe83632ab2758",
    Path(
        "/opt/vllm/vllm/model_executor/layers/quantization/"
        "exl3_online_cache.py"
    ): "8d2b2098d715d88337207db7d7a2a2bc7eead6e30a7ea435b41e1a4dfe615e9b",
    Path(
        "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
        "quantization/exl3_online_cache.py"
    ): "8d2b2098d715d88337207db7d7a2a2bc7eead6e30a7ea435b41e1a4dfe615e9b",
    MIXED_TRELLIS_PATH:
        "99e22ada2853d5c091c0a8bb8cca0d02bb8448d69ce8317b5bb747a2914aa855",
    PCIE_DCP_A2A_CUDA_PATH:
        "f560e2179101dd428d32711cefe32d3084c78ee92583b0656404a2268498a726",
    PCIE_DCP_A2A_PATH:
        "fdd3b20c74efb9f2feb3c0088de22f00656c8c8aea98adc66c0a5616b4b2badd",
    MICRO_KERNEL_PATH:
        "c30d77498ca00677dae9da552e803f94e7301b58cf26041f82801e6c245d0945",
    VLLM_CONFIG_PATH: (
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
        "49ee79fd79dde0453009577a2a82549cf63e91427f1aa50f2df4a3cb29c2e477",
    ),
    Path("/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py"): (
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
        "49ee79fd79dde0453009577a2a82549cf63e91427f1aa50f2df4a3cb29c2e477",
    ),
    DCP_POLICY_PATH:
        "3bfeb2cf7e2db8fa80f4a9dedc8a7a816793a480ce6850c7618aa4b5e199626c",
}

EXL3_MARKERS = (
    "def _online_trellis_bits(",
    "def _online_trellis_shape_supported(",
    "def _load_exl3_online_quantizer(",
    "def _resolve_prefill_capacity(",
    "def _load_sparkinfer_mixed_trellis(",
    "def _prepare_mixed_rank_sliced_weights(",
    "run_mixed_trellis(",
    "_configure_online_cache_identity(",
    "_get_bf16_online_linear_method(",
    '_SHARED_H_ROTATION_LAYOUT = "shared_h_v1"',
    "((3, 206), (4, 50))",
    "((3, 148), (4, 108))",
)

ENV_MARKERS = (
    "VLLM_EXL3_PREFILL_CAPACITY",
    "VLLM_EXL3_ONLINE_TRELLIS_BITS",
    "VLLM_EXL3_ONLINE_CACHE_DIR",
    "VLLM_EXL3_ONLINE_CACHE_MODE",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, object]:
    failures: list[str] = []
    observed: dict[str, str] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            failures.append(f"missing required r28 file: {path}")
            continue
        actual = digest(path)
        observed[str(path)] = actual
        allowed = (expected,) if isinstance(expected, str) else expected
        if actual not in allowed:
            failures.append(
                f"unexpected r28 file hash: {path}: expected one of "
                f"{allowed}, got {actual}"
            )

    if EXL3_PATH.is_file():
        source = EXL3_PATH.read_text(encoding="utf-8")
        missing = [marker for marker in EXL3_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native r28 EXL3 API: {missing!r}")

    if ENV_PATH.is_file():
        source = ENV_PATH.read_text(encoding="utf-8")
        missing = [marker for marker in ENV_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native r28 env registry: {missing!r}")

    if MIXED_TRELLIS_PATH.is_file():
        source = MIXED_TRELLIS_PATH.read_text(encoding="utf-8")
        for marker in (
            "tier0_num_experts",
            "tier1_num_experts",
            "Expert counts are runtime artifact data",
        ):
            if marker not in source:
                failures.append(
                    f"incomplete native r28 SparkInfer #117 contract: {marker!r}"
                )

    if VLLM_CONFIG_PATH.is_file():
        source = VLLM_CONFIG_PATH.read_text(encoding="utf-8")
        if "scheduled_token_delta = (" not in source:
            failures.append("missing speculative scheduler delta calculation")

    if DCP_POLICY_PATH.is_file():
        source = DCP_POLICY_PATH.read_text(encoding="utf-8")
        for marker in (
            'elif [[ "${tp}:${dcp}" == "4:4" ]]; then',
            '4:4|8:4) indexer_shards=2 ;;',
        ):
            if marker not in source:
                failures.append(
                    f"incomplete native r28 DCP4 auto policy: {marker!r}"
                )

    result: dict[str, object] = {
        "release": "GG-v20-r28",
        "vllm_tree": "e1e94267f014eeace6d40337611046d567f6cd83",
        "sparkinfer_tree": "200c1db7ef98ff8bbfd4f621555326e20f42282e",
        "lmcache_tree": "9a05c8818bae48d15b79c7e876418bb813c08cd0",
        "files": observed,
        "status": "failed" if failures else "verified",
    }
    if failures:
        result["failures"] = failures
        raise ValueError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    try:
        result = verify()
    except (OSError, UnicodeError, ValueError) as error:
        print(f"FATAL: r28 native-source gate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
