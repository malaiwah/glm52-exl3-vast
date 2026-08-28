#!/usr/bin/env python3
"""Fail closed unless the immutable GG v20-r17 native stack is complete.

The Dockerfile pins the parent manifest digest.  This second gate makes the
runtime contract explicit and catches the packaging failure seen in the first
r17 candidate: the PCIe CUDA sources existed, but their local header did not,
so the runtime silently fell back to PyNCCL.  It also prevents the retired r14
#210/#219 overlay bundle from being mistaken for the native #222 tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


EXPECTED = {
    Path("/opt/vllm/vllm/model_executor/layers/quantization/exl3.py"):
        "de2ee9a27cc129bacc23e15866c2fd75702e4fc7b701c59308458223eda98efd",
    Path(
        "/opt/venv/lib/python3.12/site-packages/vllm/"
        "model_executor/layers/quantization/exl3.py"
    ): "de2ee9a27cc129bacc23e15866c2fd75702e4fc7b701c59308458223eda98efd",
    Path("/opt/vllm/vllm/envs.py"):
        "61f14ca1998a535e3af07e0743b70c232e92d9b9d5c7bceae3dd32f575aae329",
    Path("/opt/venv/lib/python3.12/site-packages/vllm/envs.py"):
        "61f14ca1998a535e3af07e0743b70c232e92d9b9d5c7bceae3dd32f575aae329",
    Path(
        "/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/"
        "ipc_handle_registry.h"
    ): "6ab94c3e575126ae13292209978dbe5a6d071ccc01870214ddf6a62f26b39fdd",
    Path("/usr/local/bin/glm52-pcie-runtime-env.sh"):
        "e3a35eefabd13f94beb8122d983c4b98bc6373575d7d9ea3e81fe83632ab2758",
}

EXL3_MARKERS = (
    "def _resolve_prefill_capacity(",
    '"VLLM_EXL3_PREFILL_CAPACITY", max_batched_tokens',
    "def _load_sparkinfer_mixed_trellis(",
    "def _prepare_mixed_rank_sliced_weights(",
    "def _apply_mixed_rank_sliced(",
    '"serial_tiers": tuple(serial_tiers)',
    '"prefill_capacity": prefill_capacity',
    "run_mixed_trellis(",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, object]:
    failures: list[str] = []
    observed: dict[str, str] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            failures.append(f"missing required r17 file: {path}")
            continue
        actual = digest(path)
        observed[str(path)] = actual
        if actual != expected:
            failures.append(
                f"unexpected r17 file hash: {path}: expected {expected}, got {actual}"
            )

    exl3 = next(iter(EXPECTED))
    if exl3.is_file():
        source = exl3.read_text(encoding="utf-8")
        missing = [marker for marker in EXL3_MARKERS if marker not in source]
        if missing:
            failures.append(f"incomplete native vLLM #222 API: {missing!r}")

    env_source = Path("/opt/vllm/vllm/envs.py")
    if env_source.is_file() and "VLLM_EXL3_PREFILL_CAPACITY" not in env_source.read_text(
        encoding="utf-8"
    ):
        failures.append("native r17 env registry lacks VLLM_EXL3_PREFILL_CAPACITY")

    result: dict[str, object] = {
        "release": "GG-v20-r17",
        "vllm_pr": "#222@6206ac40fbd75be0ed18817bde64a63b6b838d2f",
        "sparkinfer_pr": "#105@45033eebcf5bb184587a35c528a62721a46acdf5",
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
        print(f"FATAL: r17 native-source gate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
