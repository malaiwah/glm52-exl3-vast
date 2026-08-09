#!/usr/bin/env python3
"""Repair the GG v20-r11 small/parity EXL3 extension call contract.

The r11 Python wrapper omitted the final integer argument accepted by the
reviewed EXL3 extension's ``exl3_moe`` binding. Normal compiled serving shapes
usually take the fused path, but eager/small-shape warmup (including the KLD
protocol) reaches this call and fails before inference starts.

Exit 0 means patched or already patched. Exit 1 means the pinned source anchor
was not found, which is an image/source requalification boundary.
"""

from __future__ import annotations

import os
import py_compile
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "EXL3_PY_PATH",
        "/opt/venv/lib/python3.12/site-packages/vllm/"
        "model_executor/layers/quantization/exl3.py",
    )
)

OLD = """\
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
            )
        return out32.to(x.dtype)
"""

NEW = """\
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                0,
            )
        return out32.to(x.dtype)
"""


def patch_text(source: str) -> tuple[str, str]:
    # Fail closed on an ambiguous anchor: a source carrying the broken call at
    # more than one site must not be silently half-patched (replace(..., 1)
    # would fix only the first and report success). Mirror the sibling
    # appliers, which require the anchor to occur exactly once.
    if NEW in source:
        return source, "already patched"
    if OLD not in source:
        raise ValueError("r11 EXL3 parity-call source anchor not found")
    if source.count(OLD) != 1:
        raise ValueError(
            "r11 EXL3 parity-call source anchor is ambiguous: expected "
            f"exactly one eager-parity call site, found {source.count(OLD)}"
        )
    return source.replace(OLD, NEW, 1), "patched"


def main() -> int:
    try:
        source = TARGET.read_text()
        patched, status = patch_text(source)
    except (OSError, ValueError) as error:
        print(f"exl3/parity-abi: {error}")
        return 1

    if status == "already patched":
        print("exl3/parity-abi: already patched")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".turnkey-original")
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    try:
        if not backup.exists():
            backup.write_text(source)
        temporary.write_text(patched)
        os.replace(temporary, TARGET)
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if backup.exists():
            os.replace(backup, TARGET)
        print(f"exl3/parity-abi: patch failed and was rolled back: {error}")
        return 1

    print("exl3/parity-abi: patched OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
