#!/usr/bin/env python3
"""Root-side SOUL supervisor with a post-hardening credential handshake."""
from __future__ import annotations

import argparse
import os
import pwd
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_API_KEY_BYTES = 16 * 1024
READY_TIMEOUT_S = 30
STOP_TIMEOUT_S = 5
_CHILD: subprocess.Popen | None = None
_STOP_SIGNAL = 0


def _uid_pids(uid: int) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status_lines = (entry / "status").read_text(
                encoding="utf-8", errors="replace").splitlines()
            state_line = next(
                line for line in status_lines if line.startswith("State:"))
            if state_line.split()[1] == "Z":
                continue
            uid_line = next(
                line for line in status_lines if line.startswith("Uid:"))
            process_uids = {int(value) for value in uid_line.split()[1:5]}
        except (OSError, StopIteration, ValueError):
            continue
        if uid in process_uids:
            pids.append(int(entry.name))
    return pids


def _kill_uid(uid: int, sig: int) -> None:
    """Remove detached shell sessions belonging to the dedicated SOUL uid."""
    for _attempt in range(3):
        pids = _uid_pids(uid)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.05)
    remaining = _uid_pids(uid)
    if remaining:
        raise RuntimeError(
            f"could not reap {len(remaining)} process(es) for uid {uid}")


def _signal_child(sig: int) -> None:
    if _CHILD is None or _CHILD.poll() is not None:
        return
    try:
        os.killpg(_CHILD.pid, sig)
    except ProcessLookupError:
        pass


def _on_signal(signum: int, _frame) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    _signal_child(signal.SIGTERM)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])


def main(argv=None) -> int:
    global _CHILD, _STOP_SIGNAL
    _CHILD = None
    _STOP_SIGNAL = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="soul")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a controller command is required after --")
    if os.geteuid() != 0:
        parser.error("the SOUL launcher must run as root")

    api_key = os.environ.pop("SOUL_ROOT_API_KEY", "")
    encoded_key = api_key.encode("utf-8")
    if len(encoded_key) > MAX_API_KEY_BYTES:
        parser.error("the model API key exceeds the bounded handoff size")
    uid = pwd.getpwnam(args.user).pw_uid

    # The account is dedicated to SOUL. A previous exec session may have called
    # setsid(2) and escaped the old controller's process group, so remove every
    # process for that uid before creating any credential-bearing descriptor.
    _kill_uid(uid, signal.SIGKILL)

    key_read, key_write = os.pipe()
    ready_read, ready_write = os.pipe()
    child_env = dict(os.environ)
    child_env.update({
        "SOUL_API_KEY_FD": str(key_read),
        "SOUL_KEY_READY_FD": str(ready_write),
    })
    child_command = [
        "nice", "-n", "10", "ionice", "-c", "3",
        "runuser", "-u", args.user, "--preserve-environment", "--", *command,
    ]

    for watched_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(watched_signal, _on_signal)
    try:
        _CHILD = subprocess.Popen(
            child_command,
            env=child_env,
            pass_fds=(key_read, ready_write),
            start_new_session=True,
        )
        if _STOP_SIGNAL:
            _signal_child(signal.SIGTERM)
            return 128 + _STOP_SIGNAL
        os.close(key_read)
        key_read = -1
        os.close(ready_write)
        ready_write = -1

        readable, _writable, _errors = select.select(
            [ready_read], [], [], READY_TIMEOUT_S)
        if _STOP_SIGNAL:
            _signal_child(signal.SIGTERM)
            return 128 + _STOP_SIGNAL
        if not readable or os.read(ready_read, 1) != b"R":
            raise RuntimeError(
                "SOUL controller did not harden itself before key handoff")
        os.close(ready_read)
        ready_read = -1
        if _STOP_SIGNAL:
            _signal_child(signal.SIGTERM)
            return 128 + _STOP_SIGNAL
        _write_all(key_write, len(encoded_key).to_bytes(4, "big"))
        _write_all(key_write, encoded_key)
        os.close(key_write)
        key_write = -1
        api_key = ""
        encoded_key = b""

        while _CHILD.poll() is None and not _STOP_SIGNAL:
            try:
                _CHILD.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if _STOP_SIGNAL and _CHILD.poll() is None:
            try:
                _CHILD.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _signal_child(signal.SIGKILL)
                _CHILD.wait(timeout=STOP_TIMEOUT_S)
        return int(_CHILD.returncode or 0)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FATAL: SOUL launcher: {exc}", file=sys.stderr)
        _signal_child(signal.SIGKILL)
        if _CHILD is not None:
            try:
                _CHILD.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
        return 1
    finally:
        for fd in (key_read, key_write, ready_read, ready_write):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        _signal_child(signal.SIGKILL)
        _kill_uid(uid, signal.SIGKILL)
        _CHILD = None


if __name__ == "__main__":
    raise SystemExit(main())
