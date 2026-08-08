#!/usr/bin/env python3
"""Fail closed on the immutable GG v20-r31 and vLLM #258 contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


VLLM_SOURCE = Path("/opt/vllm")
SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
VLLM_SITE = SITE_PACKAGES
B12X_SITE = SITE_PACKAGES / "b12x"
EXL3_PATH = VLLM_SOURCE / "vllm/model_executor/layers/quantization/exl3.py"
VLLM_CONFIG_PATH = VLLM_SOURCE / "vllm/config/vllm.py"

VLLM_FILES = {
    "vllm/envs.py": (
        "6fb1c8f2f5b9cd6a515caa416a90da632180556816a4812c41ed65a5d3f11eaf",
        "3eee46ffa0eb7c7c021b12415d24c560ab477d36075aafa4dda7e5cb308bff0a",
    ),
    "vllm/v1/core/kv_cache_utils.py": (
        "3860730aaf287e4307d64ab2144356ad129fe7d46e86b668eb1563c52c582291",
        "b71b7819066a912c3b9743219447a8147a3d60656418801f6189a142fe5f41e8",
    ),
    "vllm/v1/worker/gpu/model_runner.py": (
        "feb877f6eaf09249fb95e76a9dc2a2b4d75b201cade30f1915f47d7585c90eea",
        "e8f13cbabadc6c591bc644874abbfcca5a0c935396993e1046ace173f227b0d0",
    ),
    "vllm/v1/worker/gpu/sample/prompt_logprob.py": (
        "5e3768d2e9652eafeea4df56342af0d1657879413ecc92d8537652be43815ca9",
        "d5447f2240b00ccdaf0e78d261d496de4fb2c35188ef5ca5dea2a362780637de",
    ),
}

EXPECTED = {
    EXL3_PATH:
        "70c1abb81af3c93b1015bddbeb42c9e4139a2752fe6ce413b9176e37aa48d0f9",
    VLLM_SITE / "vllm/model_executor/layers/quantization/exl3.py": (
        "70c1abb81af3c93b1015bddbeb42c9e4139a2752fe6ce413b9176e37aa48d0f9",
        # Turnkey's reviewed parity-call compatibility overlay. The source
        # checkout remains immutable; the installed runtime gains only the
        # extension binding's final integer argument.
        "7ad8637502b00cb8f95155f305acd4bba5512d884c9ec36a2696386f25e553a3",
    ),
    VLLM_CONFIG_PATH: (
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
        "49ee79fd79dde0453009577a2a82549cf63e91427f1aa50f2df4a3cb29c2e477",
    ),
    VLLM_SITE / "vllm/config/vllm.py": (
        "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
        "49ee79fd79dde0453009577a2a82549cf63e91427f1aa50f2df4a3cb29c2e477",
    ),
    B12X_SITE / "moe/_shared/kernels/w4a16/host.py":
        "02d37b856702fde1f0b2ddacb2e380c3b5584022a00a4458564dfbe34cbe5e0a",
    B12X_SITE / "moe/_shared/kernels/w4a16/mixed_trellis.py":
        "313022ca6ed5785a081a95e6e58b08c96cf3a706976f6721b99c36f3943c54e1",
    B12X_SITE / "moe/fused_moe/_impl.py":
        "060c3d04a567f3f907236470781ed8c4f6c28c99ce59c724e4793b20c5acf942",
}
for relative, allowed in VLLM_FILES.items():
    EXPECTED[VLLM_SOURCE / relative] = allowed
    EXPECTED[VLLM_SITE / relative] = allowed

EXL3_MARKERS = (
    "warmup_mixed_trellis_route_pack=module.warmup_mixed_trellis_route_pack",
    "def warmup_exl3_mixed_trellis_route_pack(",
    '"warmup_exl3_mixed_trellis_route_pack"',
)

B12X_MARKERS = {
    B12X_SITE / "moe/_shared/kernels/w4a16/host.py": (
        "def route_pack_warmup_token_counts(",
        "live_numel",
    ),
    B12X_SITE / "moe/_shared/kernels/w4a16/mixed_trellis.py": (
        "def warmup_mixed_trellis_route_pack(",
        "route_ids_dtype",
    ),
    B12X_SITE / "moe/fused_moe/_impl.py": (
        "route_pack_warmup_token_counts",
        "for route_ids_dtype in (torch.int32, torch.int64)",
    ),
}

PROMPT_LOGPROBS_MARKERS = {
    "vllm/envs.py": ("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE",),
    "vllm/v1/core/kv_cache_utils.py": (
        "blocks implied by available KV memory",
    ),
    "vllm/v1/worker/gpu/model_runner.py": (
        "self.prompt_logprobs_worker.profile_run(",
    ),
    "vllm/v1/worker/gpu/sample/prompt_logprob.py": (
        "def profile_run(",
        "LogprobsTensors.empty_cpu(",
        "self.chunk_size",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, object]:
    failures: list[str] = []
    observed: dict[str, str] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            failures.append(f"missing required r31 file: {path}")
            continue
        actual = digest(path)
        observed[str(path)] = actual
        allowed = (expected,) if isinstance(expected, str) else expected
        if actual not in allowed:
            failures.append(
                f"unexpected r31 file hash: {path}: expected one of "
                f"{allowed}, got {actual}"
            )

    if EXL3_PATH.is_file():
        source = EXL3_PATH.read_text(encoding="utf-8")
        missing = [marker for marker in EXL3_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native r31 vLLM #228 contract: {missing!r}")

    for path, markers in B12X_MARKERS.items():
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in source]
        if missing:
            failures.append(f"incomplete native r31 b12x #126 contract: {missing!r}")

    for root in (VLLM_SOURCE, VLLM_SITE):
        for relative, markers in PROMPT_LOGPROBS_MARKERS.items():
            path = root / relative
            if not path.is_file() or digest(path) != VLLM_FILES[relative][1]:
                continue
            source = path.read_text(encoding="utf-8")
            missing = [marker for marker in markers if marker not in source]
            if missing:
                failures.append(f"incomplete vLLM #258 overlay: {path}: {missing!r}")

    result: dict[str, object] = {
        "release": "GG-v20-r31+vLLM-258",
        "vllm_tree": "fa13d334a2962756f9f7e9b562deb85387359f42",
        "b12x_tree": "acee6e504209068bd0cbb01cb2b98966bddcf042",
        "vllm_overlay": "02fb59c5367a3650a0ae6f8805e4f4d3f5cf815f",
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
        print(f"FATAL: r31 native-source gate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
