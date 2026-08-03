"""Make mixed-Trellis compilation sensitive to the per-tier expert split.

SparkInfer's r20 mixed EXL3 kernel uses dynamic tier GEMMs, whose individual
cache keys intentionally omit ``num_experts``.  The outer mixed dispatcher is
still specialized for the two tier-local expert extents, but its cache key did
not add those extents back.  Checkpoints whose split varies by layer (for
example 206/50 then 160/96) therefore reused the first compiled launch for the
second layout and could hit an asynchronous illegal-address fault.

This field patch changes the mixed-kernel ABI and adds both tier sizes to the
outer cache key.  It is safe for fixed-split checkpoints: they retain one
compiled launch per shape.  Apply it to the installed SparkInfer source before
starting vLLM; no vLLM source change is required.
"""

from __future__ import annotations

import os
import py_compile
from pathlib import Path

TARGET = Path(
    os.environ.get(
        "SPARKINFER_MIXED_TRELLIS_PATH",
        "/opt/venv/lib/python3.12/site-packages/sparkinfer/"
        "moe/_shared/kernels/w4a16/mixed_trellis.py",
    )
)

ABI_ANCHOR = "    ABI_VERSION = 4\n"
ABI_REPLACEMENT = "    ABI_VERSION = 5\n"

KEY_ANCHOR = """\
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.blocks_per_sm,
"""

KEY_REPLACEMENT = """\
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            # The child kernels dynamically size E and intentionally omit it
            # from their keys.  The mixed dispatcher still specializes the
            # tier-local descriptor bounds, so the split belongs here.
            self.tier0.num_experts,
            self.tier1.num_experts,
            self.blocks_per_sm,
"""

PATCH_MARKER = "            self.tier0.num_experts,\n"


def patch_text(source: str) -> tuple[str, str]:
    if PATCH_MARKER in source:
        if ABI_REPLACEMENT not in source:
            raise ValueError("expert counts are present but ABI_VERSION is not 5")
        return source, "already patched"

    abi_count = source.count(ABI_ANCHOR)
    key_count = source.count(KEY_ANCHOR)
    if abi_count != 1 or key_count != 1:
        raise ValueError(
            "r20 SparkInfer source anchors matched "
            f"ABI={abi_count}, cache_key={key_count} times"
        )

    result = source.replace(ABI_ANCHOR, ABI_REPLACEMENT, 1)
    result = result.replace(KEY_ANCHOR, KEY_REPLACEMENT, 1)
    return result, "patched"


def main() -> int:
    try:
        source = TARGET.read_text(encoding="utf-8")
        result, status = patch_text(source)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"sparkinfer/mixed-trellis-cache-key: {error}")
        return 1

    if status == "already patched":
        print("sparkinfer/mixed-trellis-cache-key: already patched")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".turnkey-cache-key-original")
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    try:
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        temporary.write_text(result, encoding="utf-8")
        os.replace(temporary, TARGET)
        py_compile.compile(str(TARGET), doraise=True)
    except (OSError, py_compile.PyCompileError) as error:
        temporary.unlink(missing_ok=True)
        if backup.exists():
            os.replace(backup, TARGET)
        print(
            "sparkinfer/mixed-trellis-cache-key: patch failed and was rolled back: "
            f"{error}"
        )
        return 1

    print("sparkinfer/mixed-trellis-cache-key: patched OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
