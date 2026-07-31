#!/usr/bin/env python3
"""Fail closed on runtime defects in a field-review server log.

The response-level feature and benchmark tools cannot see failures that vLLM
or a backend catches internally.  This audit makes those log-only failures a
separate release gate.  Use ``--start-line`` after deliberately priming cold
request shapes so a warm measurement proves that it did not compile again.
"""

import argparse
import json
import re
import sys
from pathlib import Path


READY_MARKER = "Application startup complete."
PATTERNS = (
    (
        "post_ready_compile",
        re.compile(r"\[sparkinfer cute\.compile\].*reason=post-engine-start"),
    ),
    ("structured_fsm", re.compile(r"Failed to advance FSM")),
    (
        "cuda_ipc_lifetime",
        re.compile(
            r"Producer process has been terminated before all shared CUDA "
            r"tensors released"
        ),
    ),
    (
        "cuda_runtime",
        re.compile(
            r"CUDA out of memory|illegal memory access|invalid resource handle|"
            r"\bXid\b|CUDA error"
        ),
    ),
    (
        "distributed_runtime",
        re.compile(r"NCCL.*(?:error|failed)|deadlock|barrier hang", re.IGNORECASE),
    ),
    (
        "process_failure",
        re.compile(
            r"Worker failed|EngineCore failed|segmentation fault|"
            r"Traceback \(most recent call last\)"
        ),
    ),
    # Unknown ERROR-level records are failures too.  More specific categories
    # above win because each physical line is reported once.
    ("runtime_error_level", re.compile(r"(?:\)|^)\s*ERROR(?:\s|:)")),
)


def read_log(path: str) -> str:
    """Read without universal-newline rewriting of progress-bar CR bytes."""
    return Path(path).read_bytes().decode("utf-8", errors="replace")


def audit(text: str, start_line: int = 1, require_ready: bool = True) -> dict:
    """Return a bounded, machine-readable audit report."""
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    # Evidence offsets are normally captured with `wc -l` or `sed -n`.
    # Split only on LF so carriage-return progress bars do not silently change
    # the meaning of --start-line.
    lines = text.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    ready_lines = [
        index for index, line in enumerate(lines, start=1) if READY_MARKER in line
    ]
    findings = []
    for line_number, line in enumerate(lines, start=1):
        if line_number < start_line:
            continue
        for kind, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "kind": kind,
                        "line": line_number,
                        "excerpt": line[:500],
                    }
                )
                break
    ready = bool(ready_lines)
    return {
        "schema": 1,
        "start_line": start_line,
        "line_count": len(lines),
        "ready": ready,
        "ready_lines": ready_lines,
        "require_ready": require_ready,
        "findings": findings,
        "counts": {
            kind: sum(item["kind"] == kind for item in findings)
            for kind, _pattern in PATTERNS
        },
        "ok": not findings and (ready or not require_ready),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--allow-no-ready", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    try:
        # Decode bytes directly: text-mode universal-newline handling rewrites
        # standalone CR progress updates into LF and invalidates wc/sed offsets.
        text = read_log(args.log)
        report = audit(
            text,
            start_line=args.start_line,
            require_ready=not args.allow_no_ready,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        output = Path(args.out)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    sys.stdout.write(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
