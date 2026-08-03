"""Suppress the low-scheduler-budget warning when serial speculation adds no slots.

vLLM reserves scheduler headroom only for speculative methods that can insert
additional draft slots.  Its low-budget warning was nevertheless emitted for
all speculative configurations, including serial GLM MTP where the calculated
slot delta is zero.  Keep the warning for configurations whose draft path
actually reduces the schedulable budget, and leave validation unchanged.
"""

from __future__ import annotations

import os
import py_compile
from pathlib import Path


DEFAULT_TARGETS = (
    Path("/opt/vllm/vllm/config/vllm.py"),
    Path("/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py"),
)

WARNING_ANCHOR = (
    "            if self.scheduler_config.max_num_scheduled_tokens < 8192:\n"
)
WARNING_REPLACEMENT = (
    "            # Serial speculative methods such as GLM MTP add no scheduler\n"
    "            # slots.  Do not recommend extra batch capacity when the\n"
    "            # calculated drafting delta is zero.\n"
    "            if (\n"
    "                scheduled_token_delta > 0\n"
    "                and self.scheduler_config.max_num_scheduled_tokens < 8192\n"
    "            ):\n"
)
PATCH_MARKER = "                scheduled_token_delta > 0\n"


def patch_text(source: str) -> tuple[str, str]:
    if PATCH_MARKER in source and WARNING_REPLACEMENT in source:
        return source, "already patched"

    count = source.count(WARNING_ANCHOR)
    if count != 1:
        raise ValueError(f"vLLM warning anchor matched {count} times")
    return source.replace(WARNING_ANCHOR, WARNING_REPLACEMENT, 1), "patched"


def targets() -> tuple[Path, ...]:
    override = os.environ.get("VLLM_CONFIG_PATH")
    return (Path(override),) if override else DEFAULT_TARGETS


def main() -> int:
    for target in targets():
        try:
            source = target.read_text(encoding="utf-8")
            result, status = patch_text(source)
            if status == "patched":
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_text(result, encoding="utf-8")
                py_compile.compile(str(temporary), doraise=True)
                os.replace(temporary, target)
            print(f"vllm/serial-spec-warning: {target}: {status}")
        except (OSError, UnicodeError, ValueError, py_compile.PyCompileError) as error:
            print(f"vllm/serial-spec-warning: {target}: {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
