#!/usr/bin/env python3
"""Install selected B12X and V2-compatible fairness sources after the GLM refresh.

Accept only a complete pinned before or after state across the runtime and all
present package mirrors. The new compute_fairness module must be absent before
installation. No CUDA imports, runner selection, launcher replacement, or MTP
prototype changes are part of this layer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from apply_glm53_refresh import DEFAULT_ROOT, file_sha256

DEFAULT_MIRROR_ROOT = Path("/opt")
PROVENANCE_SHA256 = "4a3839167b5c7655b1c4774c10c998eee2a8bb360610bf1921701c8d496c784a"

# payload, runtime-relative destination, post-refresh baseline, selected source
OVERLAYS = (
    ("b12x/moe/_shared/kernels/w4a16/route_pack.py", "b12x/moe/_shared/kernels/w4a16/route_pack.py",
     "ad7ee1b0a86e5af43587051fb82aa182fcb4f3b50a51575476f6eefe271caafe",
     "a31fe50488145da58fb875ddad6fb6fecc724f76ae6880b0d69f09af78f8c298"),
    ("b12x/comm/pcie/pcie_oneshot.py", "b12x/comm/pcie/pcie_oneshot.py",
     "d8be62ce5330b92cf8fb73dc36384f7268494fff1b3795496a1ee4898dfc312e",
     "c4614bc6f3a2fea80e72ce087af2e76e94dc1737025b37a3ce07e79a0a37bcf9"),
    ("vllm/config/scheduler.py", "vllm/config/scheduler.py",
     "ecda4b0c12a2e40dddafa6e9c1f5328740a0bae29d7cb8dc5bc1a8d3de56021e",
     "aae93351af40efbdb28d75e82cb91b6948c703fc18b7a1bd2a6d7714b5ba7a4f"),
    ("vllm/config/vllm.py", "vllm/config/vllm.py",
     "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
     "b40fc3b5d131851d15f2e88d550a02abd11f043d998ec927c60ba299e424b853"),
    ("vllm/engine/arg_utils.py", "vllm/engine/arg_utils.py",
     "8449722b2285fd17271be4123d70b9c9c84b550e6107f00cfe53c6274e2c41be",
     "64d56a589b71f3cabe38de9b786ae7ffb26995ebc9df8e126ba0d41eadaf1233"),
    ("vllm/v1/core/sched/compute_fairness.py", "vllm/v1/core/sched/compute_fairness.py", None,
     "e46de15e31a1d1a52fa63c8d9ebec7867365132d07fe258c7a25aa2a972dc10a"),
    ("vllm/v1/core/sched/interface.py", "vllm/v1/core/sched/interface.py",
     "01684d9eb30820c210d4fccbda5fd2dc304fc58e04704b30d574c83edce3f8ff",
     "1cad843c243dc6447e23365859c2a81d58098c2b69790582374ed062bf98110a"),
    ("vllm/v1/core/sched/output.py", "vllm/v1/core/sched/output.py",
     "91fcedcc9fa478b5e5bf718c8fc50052c52bf418193c02cd471ca20e7f7518c6",
     "eb1d522c1dcd050bcf3668957063db31940a299692c1a349fe404728e79c6df5"),
    ("vllm/v1/core/sched/scheduler.py", "vllm/v1/core/sched/scheduler.py",
     "f569c1e58ef6d2e1f244d488d7172f39a2017d9c5f7b7653bfdcd13d857580ab",
     "072f9bd18b98d2d6189d34d55742cf6c14bdb940d012951618ce16323948d83e"),
    ("vllm/v1/engine/core.py", "vllm/v1/engine/core.py",
     "276186db3c5a3df793594ed596974c2729b916bccae7d94a892a39c5301e183d",
     "142b645bdd125806b6af9af1b501ef03fc7df25d104a6565add86a3f9cb3717b"),
    ("vllm/v1/metrics/stats.py", "vllm/v1/metrics/stats.py",
     "9ec1157ff57ab65e7b71ef79c10523f9d5aaef5ea2de60d8f63ed1e3c489a923",
     "dd2ee1676db2aaa707a2e46361cdedddcfe6e2be0c309c74da85faefb963319e"),
    ("vllm/v1/metrics/loggers.py", "vllm/v1/metrics/loggers.py",
     "114edcb850ea3c21234ae04c3bde5bb1b2f7d3b40a72f707b3288e7d608ee637",
     "1532c87d3b31fda21aa130b9c664718e1338f1a2600fd9dfe919d4755c823b3f"),
)


def resolve_targets(root: Path = DEFAULT_ROOT,
                    mirror_root: Path | None = DEFAULT_MIRROR_ROOT,
                    ) -> list[tuple[str, Path, str | None, str]]:
    """Enumerate site-packages and /opt/{package}/{package} debug sources."""
    targets = [(payload, root / relative, before, after)
               for payload, relative, before, after in OVERLAYS]
    if mirror_root is not None:
        targets += [(payload, mirror_root / Path(relative).parts[0] / relative, before, after)
                    for payload, relative, before, after in OVERLAYS]
    return targets


def install_targets(root: Path = DEFAULT_ROOT,
                    mirror_root: Path | None = DEFAULT_MIRROR_ROOT,
                    ) -> list[tuple[str, Path, str | None, str]]:
    """Each optional package mirror must be complete, not a partial checkout."""
    targets = resolve_targets(root, None)
    if mirror_root is None:
        return targets
    mirrored = resolve_targets(root, mirror_root)[len(targets):]
    for package in sorted({Path(relative).parts[0] for _, relative, _, _ in OVERLAYS}):
        package_root = mirror_root / package / package
        if not package_root.exists() and not package_root.is_symlink():
            continue
        if package_root.is_symlink() or not package_root.is_dir():
            raise RuntimeError(f"refusing unknown selected mirror: {package_root}")
        package_targets = [row for row in mirrored if row[1].is_relative_to(package_root)]
        missing = [str(target) for _, target, before, _ in package_targets
                   if before is not None and not target.exists()]
        if missing:
            raise RuntimeError("refusing partial selected mirror: " + ", ".join(missing))
        targets.extend(package_targets)
    return targets


def verify_payloads(source_dir: Path) -> None:
    ledger_path = source_dir / "provenance.json"
    if file_sha256(ledger_path) != PROVENANCE_SHA256:
        raise RuntimeError("refusing unknown selected runtime provenance")
    ledger = json.loads(ledger_path.read_text())
    if (ledger["install_root"], ledger["mirror_root"]) != (
            str(DEFAULT_ROOT), str(DEFAULT_MIRROR_ROOT)):
        raise RuntimeError("selected provenance roots do not match installer")
    records = tuple((entry["payload"], entry["path"], entry["before_sha256"],
                     entry["after_sha256"]) for entry in ledger["files"])
    if records != OVERLAYS:
        raise RuntimeError("selected provenance files do not match installer")
    for payload, _, _, after in OVERLAYS:
        if file_sha256(source_dir / payload) != after:
            raise RuntimeError(f"refusing unknown selected runtime payload: {payload}")


def install(source_dir: Path, *, verify_only: bool = False,
            root: Path = DEFAULT_ROOT,
            mirror_root: Path | None = DEFAULT_MIRROR_ROOT) -> None:
    verify_payloads(source_dir)
    targets = install_targets(root, mirror_root)
    states = []
    # All source, layout, and destination checks precede any staging or writes.
    for _, target, before, after in targets:
        if target.is_symlink():
            raise RuntimeError(f"refusing symlink selected runtime target: {target}")
        if not target.parent.is_dir() or target.parent.resolve() != target.parent.absolute():
            raise RuntimeError(f"refusing missing or symlink target directory: {target.parent}")
        actual = file_sha256(target)
        if actual == before:
            states.append("before")
        elif actual == after:
            states.append("after")
        else:
            raise RuntimeError(f"refusing unknown selected runtime state: {target}: "
                               f"expected base={before} or selected={after}, got={actual}")
    if all(state == "after" for state in states):
        print(f">>> selected runtime already installed and verified ({len(targets)} targets)")
        return
    if not all(state == "before" for state in states):
        raise RuntimeError("refusing mixed selected runtime state")
    if verify_only:
        print(f">>> selected runtime baseline and payloads verified ({len(targets)} targets)")
        return

    staged: list[tuple[Path, Path]] = []
    try:
        for payload, target, _, after in targets:
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.glm53-", dir=target.parent)
            os.close(fd)
            temporary_path = Path(temporary)
            staged.append((temporary_path, target))
            shutil.copyfile(source_dir / payload, temporary_path)
            os.chmod(temporary_path, target.stat().st_mode & 0o777 if target.exists() else 0o644)
            if file_sha256(temporary_path) != after:
                raise RuntimeError(f"staged selected verification failed: {target}")
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    for _, target, _, after in targets:
        if file_sha256(target) != after:
            raise RuntimeError(f"installed selected verification failed: {target}")
    print(f">>> installed and verified {len(targets)} selected runtime files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR_ROOT,
                        help="parent of optional b12x/b12x and vllm/vllm debug trees")
    args = parser.parse_args()
    install(args.source_dir, verify_only=args.verify_only, root=args.root,
            mirror_root=args.mirror_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
