#!/usr/bin/env python3
"""Fail closed unless the immutable GG v20-r25 source contract is complete."""

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
VLLM_CONFIG_PATH = Path("/opt/vllm/vllm/config/vllm.py")

EXPECTED = {
    EXL3_PATH:
        "b86b61bbc6037fe6bfd9e0f24bdc4a1043d5a5e8a5f137bcf562ebe32239151b",
    Path(
        "/opt/venv/lib/python3.12/site-packages/vllm/"
        "model_executor/layers/quantization/exl3.py"
    ): "b86b61bbc6037fe6bfd9e0f24bdc4a1043d5a5e8a5f137bcf562ebe32239151b",
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
        "edf3790e04252fb3d65507ff778e8af1d497dcd119ec0cd7dd61c3af4d47bc82",
    VLLM_CONFIG_PATH:
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
    Path("/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py"):
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
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
            failures.append(f"missing required r25 file: {path}")
            continue
        actual = digest(path)
        observed[str(path)] = actual
        if actual != expected:
            failures.append(
                f"unexpected r25 file hash: {path}: expected {expected}, got {actual}"
            )

    exl3 = EXL3_PATH
    if exl3.is_file():
        source = exl3.read_text(encoding="utf-8")
        missing = [marker for marker in EXL3_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native r25 EXL3 API: {missing!r}")

    env_source = ENV_PATH
    if env_source.is_file():
        source = env_source.read_text(encoding="utf-8")
        missing = [marker for marker in ENV_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native r25 env registry: {missing!r}")

    if MIXED_TRELLIS_PATH.is_file():
        source = MIXED_TRELLIS_PATH.read_text(encoding="utf-8")
        for marker in (
            "tier0_num_experts",
            "tier1_num_experts",
            "Expert counts are runtime artifact data",
        ):
            if marker not in source:
                failures.append(
                    f"incomplete native r25 SparkInfer #117 contract: {marker!r}"
                )

    if VLLM_CONFIG_PATH.is_file():
        source = VLLM_CONFIG_PATH.read_text(encoding="utf-8")
        if "scheduled_token_delta = (" not in source:
            failures.append("missing speculative scheduler delta calculation")

    result: dict[str, object] = {
        "release": "GG-v20-r25",
        "vllm_tree": "f5981f14b4d39979bc0d799c020d42002b707257",
        "sparkinfer_tree": "978cdb3593367469abd16bc8bdbc4ed0ea2787da",
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
        print(f"FATAL: r25 native-source gate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
