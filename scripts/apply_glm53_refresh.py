#!/usr/bin/env python3
"""Install the GLM-5.3 parser/cadence refresh without changing the runtime ABI.

The eight old-source payloads contain only local-vllm #640, #639 and #546
backports. These source fixes are not a GPU/context-capacity qualification.
Accept only the exact complete before state or exact complete after state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("/opt/infernal-invocation/vllm")
PROVENANCE_SHA256 = 'af21d8d36de85740621858e74e619dd144fc1687b188eed97a78e99a8bd4673c'

OVERLAYS = (
    (
        'token_id_scanner.py',
        '/opt/infernal-invocation/vllm/vllm/parser/engine/token_id_scanner.py',
        'c9db6d6a29865d65ba1adc523015488aad4d7dbef46b7539041e9efbe81bd034',
        'd478c6304cb12e084df52831977e0702ce2cc262919111a12a7ef37d38ef32f1',
    ),
    (
        'parser_engine_config.py',
        '/opt/infernal-invocation/vllm/vllm/parser/engine/parser_engine_config.py',
        '83460a34f86d3bb639975feea0ddaa468de2814ab18bf9984c0d836e8042120e',
        '895af219fae67671e63d2e639e6a8e716d27ff989ac727744e8e9718f87b3fa4',
    ),
    (
        'streaming_parser_engine.py',
        '/opt/infernal-invocation/vllm/vllm/parser/engine/streaming_parser_engine.py',
        '0f4625066e178ea5b3fd4784812103ce3b1362fbc278700518b8ab6b8ba57074',
        '26997d3653f1668e7f100752dbb3d4d6ee6df7c721e414b069a823187eee3b29',
    ),
    (
        'glm47_moe.py',
        '/opt/infernal-invocation/vllm/vllm/parser/glm47_moe.py',
        'ce3629319e56e882d25cb75d62e3e7088a4eec1518885fc69fc696eafb4a97b2',
        'b76ea87090a1b823c272aa0d338cadf53a5aa3d845c3b3cdaed6258fa5a00ff8',
    ),
    (
        'scheduler_config.py',
        '/opt/infernal-invocation/vllm/vllm/config/scheduler.py',
        '3cf5d41a5d662ab0eada6f3128e2538abc0a250d6aef82acab7df7dd84e498bc',
        'd70698ec13248054248aea0824ffa9661edc9150be124424d0738cc775ecbfd7',
    ),
    (
        'scheduler_interface.py',
        '/opt/infernal-invocation/vllm/vllm/v1/core/sched/interface.py',
        'be6c008664096c5660a8879de72b315bbe479bf1d688be9b25891063ec53a367',
        '049b1f3cb82d69295d0e642af9f8bb4c07f971168d2e8543de43c29538bd9350',
    ),
    (
        'scheduler.py',
        '/opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py',
        '2488d3dd87764fc04e1f4ec770528b2d5fbb0e357315b91bb125479eab9007e5',
        '8cfbb80da8420138ef291af89a96c436fdaf578b5e795a6837e32dde578a6573',
    ),
    (
        'engine_core.py',
        '/opt/infernal-invocation/vllm/vllm/v1/engine/core.py',
        'e1f1892cf625db6ba5896a6d49a055e924ad94a8301cfdc3bb9d13e1d89ca9dd',
        '27b5bfe913f95c1ed9593df6210274b8a58f102ec43ae446a44c8248f49b4390',
    ),
)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_payloads(source_dir: Path) -> None:
    ledger_path = source_dir / "provenance.json"
    if file_sha256(ledger_path) != PROVENANCE_SHA256:
        raise RuntimeError("refusing unknown GLM-5.3 refresh provenance")
    ledger = json.loads(ledger_path.read_text())
    for source_name, target_name, before, after in OVERLAYS:
        source = source_dir / source_name
        if file_sha256(source) != after:
            raise RuntimeError(f"refusing unknown GLM-5.3 refresh payload: {source}")
        records = [entry for entry in ledger["files"]
                   if entry["payload"] == source_name]
        if len(records) != 1:
            raise RuntimeError(f"expected unique provenance record: {source_name}")
        record = records[0]
        if (record["before_sha256"], record["after_sha256"],
                str(DEFAULT_ROOT / record["path"])) != (before, after, target_name):
            raise RuntimeError(f"provenance ledger mismatch: {source_name}")
        # Reverse the exact hunks to prove that payloads differ from the pinned
        # installed base by nothing except these reviewed old-ABI backports.
        original = source.read_text()
        for replacement in reversed(record["replacements"]):
            old, new = replacement["before"], replacement["after"]
            if not old or not new or original.count(new) != 1:
                raise RuntimeError(f"non-unique refresh anchor: {source_name}")
            original = original.replace(new, old, 1)
        if hashlib.sha256(original.encode()).hexdigest() != before:
            raise RuntimeError(f"refresh does not reproduce pinned base: {source_name}")


def install(source_dir: Path, *, verify_only: bool = False,
            root: Path = DEFAULT_ROOT) -> None:
    verify_payloads(source_dir)
    targets = [(source, root / Path(target).relative_to(DEFAULT_ROOT), before, after)
               for source, target, before, after in OVERLAYS]
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
        print(">>> GLM-5.3 refresh already installed and verified")
        return
    if not all(state == "before" for state in states):
        raise RuntimeError("refusing mixed GLM-5.3 refresh state: " + ", ".join(states))
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
    print(f">>> installed and verified {len(OVERLAYS)} GLM-5.3 refresh files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="vLLM checkout root (contains the vllm package)")
    args = parser.parse_args()
    install(args.source_dir, verify_only=args.verify_only, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
