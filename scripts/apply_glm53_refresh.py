#!/usr/bin/env python3
"""Install the GLM-5.3 parser/cadence refresh without changing the runtime ABI.

The eight payloads are the Gilded Gnosis r34 maintenance sources plus the
local-vllm #640, #639 and #546 backports; nothing else. These source fixes are
not a GPU/context-capacity qualification. Accept only the exact complete before
state or the exact complete after state.

Gilded installs the runtime under site-packages and also ships a byte-identical
debug source tree under /opt/vllm. When that mirror is present every payload is
installed and pinned in both roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("/opt/venv/lib/python3.12/site-packages")
DEFAULT_MIRROR_ROOT = Path("/opt/vllm")
PROVENANCE_SHA256 = '64aab6eeb6085fe22f735338c9d5371dc462a6ef101faebda31b8b982ab32346'

OVERLAYS = (
    (
        'token_id_scanner.py',
        'vllm/parser/engine/token_id_scanner.py',
        'c9db6d6a29865d65ba1adc523015488aad4d7dbef46b7539041e9efbe81bd034',
        'd478c6304cb12e084df52831977e0702ce2cc262919111a12a7ef37d38ef32f1',
    ),
    (
        'parser_engine_config.py',
        'vllm/parser/engine/parser_engine_config.py',
        'f7350e0ca9124001684f1f874ee72bf6a34932d3e4b84cc84567bbccf2f3e4b9',
        '92de004c7ca85975d940d578773b03f43de9da5a373c9a1c96f9c00dd1840d0b',
    ),
    (
        'streaming_parser_engine.py',
        'vllm/parser/engine/streaming_parser_engine.py',
        '2eace718fc728b46676cd5d01eee2c893aec396a4551f05c529e83a120d07715',
        '722051a19deda2fd54fe9e1a59af61cc987621809f69d78a76a25cde7bbfa18e',
    ),
    (
        'glm47_moe.py',
        'vllm/parser/glm47_moe.py',
        'ce3629319e56e882d25cb75d62e3e7088a4eec1518885fc69fc696eafb4a97b2',
        'b76ea87090a1b823c272aa0d338cadf53a5aa3d845c3b3cdaed6258fa5a00ff8',
    ),
    (
        'scheduler_config.py',
        'vllm/config/scheduler.py',
        'a816cf79a3e74ffc0984f9bebb274275b26f46be8b28cb77a29388a0996263c8',
        'ecda4b0c12a2e40dddafa6e9c1f5328740a0bae29d7cb8dc5bc1a8d3de56021e',
    ),
    (
        'scheduler_interface.py',
        'vllm/v1/core/sched/interface.py',
        '2592ff4d53fa684349dd13b8d236aa918857a9f24b8fd24817fb0487deecffd6',
        '01684d9eb30820c210d4fccbda5fd2dc304fc58e04704b30d574c83edce3f8ff',
    ),
    (
        'scheduler.py',
        'vllm/v1/core/sched/scheduler.py',
        '23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6',
        'f569c1e58ef6d2e1f244d488d7172f39a2017d9c5f7b7653bfdcd13d857580ab',
    ),
    (
        'engine_core.py',
        'vllm/v1/engine/core.py',
        'a4471ea8fc4b1698af448da5be235eb274d8420e3ca4cd6ccd94ae0ffd8ea885',
        '276186db3c5a3df793594ed596974c2729b916bccae7d94a892a39c5301e183d',
    ),
)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_targets(root: Path = DEFAULT_ROOT,
                    mirror_root: Path | None = DEFAULT_MIRROR_ROOT,
                    ) -> list[tuple[str, Path, str, str]]:
    """Every install target: the runtime root, then the debug source mirror.

    Pure path composition; nothing here touches the filesystem, so callers can
    enumerate the manifest outside the image.
    """
    targets = [(payload, root / relative, before, after)
               for payload, relative, before, after in OVERLAYS]
    if mirror_root is None:
        return targets
    return targets + [(payload, mirror_root / relative, before, after)
                      for payload, relative, before, after in OVERLAYS]


def install_targets(root: Path = DEFAULT_ROOT,
                    mirror_root: Path | None = DEFAULT_MIRROR_ROOT,
                    ) -> list[tuple[str, Path, str, str]]:
    """Install targets that exist in this layout.

    Gilded ships the debug source tree, but an image that strips it is still a
    valid layout. The mirror is therefore all-or-nothing: a partially present
    mirror is an unknown layout and is refused rather than half-installed.
    """
    targets = resolve_targets(root, None)
    if mirror_root is None:
        return targets
    mirrored = resolve_targets(root, mirror_root)[len(targets):]
    present = [target for target in mirrored if target[1].exists()]
    if not present:
        return targets
    if len(present) != len(mirrored):
        missing = [str(target[1]) for target in mirrored if not target[1].exists()]
        raise RuntimeError(
            "refusing partial GLM-5.3 refresh mirror under "
            f"{mirror_root}: missing {', '.join(missing)}"
        )
    return targets + mirrored


def verify_payloads(source_dir: Path) -> None:
    ledger_path = source_dir / "provenance.json"
    if file_sha256(ledger_path) != PROVENANCE_SHA256:
        raise RuntimeError("refusing unknown GLM-5.3 refresh provenance")
    ledger = json.loads(ledger_path.read_text())
    if (ledger["install_root"], ledger["mirror_root"]) != (
            str(DEFAULT_ROOT), str(DEFAULT_MIRROR_ROOT)):
        raise RuntimeError("provenance ledger roots do not match this installer")
    for source_name, relative, before, after in OVERLAYS:
        source = source_dir / source_name
        if file_sha256(source) != after:
            raise RuntimeError(f"refusing unknown GLM-5.3 refresh payload: {source}")
        records = [entry for entry in ledger["files"]
                   if entry["payload"] == source_name]
        if len(records) != 1:
            raise RuntimeError(f"expected unique provenance record: {source_name}")
        record = records[0]
        if (record["before_sha256"], record["after_sha256"],
                record["path"]) != (before, after, relative):
            raise RuntimeError(f"provenance ledger mismatch: {source_name}")
        # Reverse the exact hunks to prove that payloads differ from the pinned
        # installed base by nothing except these reviewed backports.
        original = source.read_text()
        for replacement in reversed(record["replacements"]):
            old, new = replacement["before"], replacement["after"]
            if not old or not new or original.count(new) != 1:
                raise RuntimeError(f"non-unique refresh anchor: {source_name}")
            original = original.replace(new, old, 1)
        if hashlib.sha256(original.encode()).hexdigest() != before:
            raise RuntimeError(f"refresh does not reproduce pinned base: {source_name}")


def install(source_dir: Path, *, verify_only: bool = False,
            root: Path = DEFAULT_ROOT,
            mirror_root: Path | None = DEFAULT_MIRROR_ROOT) -> None:
    verify_payloads(source_dir)
    targets = install_targets(root, mirror_root)
    states = []
    for _source, target, before, after in targets:
        actual = file_sha256(target)
        if actual == before:
            states.append("before")
        elif actual == after:
            states.append("after")
        else:
            raise RuntimeError(
                f"refusing unknown GLM-5.3 refresh state: {target}: "
                f"expected base={before} or refresh={after}, got={actual}"
            )
    if all(state == "after" for state in states):
        print(f">>> GLM-5.3 refresh already installed and verified "
              f"({len(targets)} targets)")
        return
    if not all(state == "before" for state in states):
        raise RuntimeError("refusing mixed GLM-5.3 refresh state: " + ", ".join(
            f"{target}={state}"
            for (_source, target, _before, _after), state in zip(targets, states)))
    if verify_only:
        print(">>> GLM-5.3 refresh base, payloads and unique-anchor provenance verified")
        return

    staged: list[tuple[Path, Path]] = []
    try:
        for source_name, target, _before, after in targets:
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.glm53-", dir=target.parent)
            os.close(fd)
            temporary_path = Path(temporary)
            staged.append((temporary_path, target))
            shutil.copyfile(source_dir / source_name, temporary_path)
            os.chmod(temporary_path, target.stat().st_mode & 0o777)
            if file_sha256(temporary_path) != after:
                raise RuntimeError(f"staged refresh verification failed: {target}")
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)

    for _source, target, _before, after in targets:
        if file_sha256(target) != after:
            raise RuntimeError(f"installed refresh verification failed: {target}")
    print(f">>> installed and verified {len(targets)} GLM-5.3 refresh files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="runtime root that contains the vllm package")
    parser.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR_ROOT,
                        help="debug source root mirroring the vllm package")
    args = parser.parse_args()
    install(args.source_dir, verify_only=args.verify_only, root=args.root,
            mirror_root=args.mirror_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
