#!/usr/bin/env python3
"""Emit log-collector-friendly model download progress.

Hugging Face uses tqdm carriage returns for snapshot progress. Container log
collectors commonly buffer those updates until tqdm writes its final newline,
which makes an active multi-minute download look stalled. This helper observes
the local snapshot directory and writes newline-delimited milestones to stdout
and the timestamped appliance boot log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import time
from pathlib import Path
from typing import Callable


GIB = 2**30


def directory_bytes(directory: str) -> int:
    """Return logical bytes currently visible below *directory*."""
    total = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # A parallel downloader can rename a completed temporary file
                # between os.walk and getsize. The next sample will see it.
                pass
    return total


def crossed_milestones(previous: int | None, done: int, expected: int) -> list[int]:
    """Return newly crossed 5-percent milestones, capping in-progress reports at 95%."""
    if expected <= 0:
        return []
    current = min(95, max(0, int(done * 100 / expected) // 5 * 5))
    if previous is None:
        # A resumed download should describe where it resumed, rather than
        # claiming that twenty old thresholds were crossed at this instant.
        return [current]
    return list(range(max(5, previous + 5), current + 1, 5))


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def format_progress(percent: int, done: int, expected_gib: float) -> str:
    return (
        f">>> Download progress: {percent}% "
        f"({done / GIB:.1f} of ~{expected_gib:g} GiB)"
    )


def append_boot_note(path: str, message: str) -> None:
    if not path:
        return
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except OSError:
        # Progress is advisory. A read-only or disappearing runtime directory
        # must never turn a resumable model download into a failed boot.
        pass


def monitor(
    pid: int,
    directory: str,
    expected_gib: float,
    boot_notes: str = "",
    interval: float = 5.0,
    emit: Callable[[str], None] = lambda message: print(message, flush=True),
) -> None:
    expected = int(expected_gib * GIB)
    previous: int | None = None
    unknown_last = 0.0
    while process_alive(pid):
        done = directory_bytes(directory)
        if expected > 0:
            milestones = crossed_milestones(previous, done, expected)
            for percent in milestones:
                message = format_progress(percent, done, expected_gib)
                emit(message)
                append_boot_note(boot_notes, message)
            if milestones:
                previous = milestones[-1]
        else:
            now = time.monotonic()
            if unknown_last == 0.0 or now - unknown_last >= 60:
                message = (
                    f">>> Download progress: {done / GIB:.1f} GiB downloaded "
                    "(total size unknown)"
                )
                emit(message)
                append_boot_note(boot_notes, message)
                unknown_last = now
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--expected-gib", type=float, default=0)
    parser.add_argument("--boot-notes", default="")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()

    # PID 1 can stop the helper immediately after reaping the downloader.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: raise_system_exit())
    monitor(
        args.pid,
        args.directory,
        args.expected_gib,
        args.boot_notes,
        args.interval,
    )
    return 0


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())
