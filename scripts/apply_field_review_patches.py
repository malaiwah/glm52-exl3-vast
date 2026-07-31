#!/usr/bin/env python3
"""Apply the independently qualified r14 field-review patches fail-closed.

The upstream image does not retain Git metadata for every installed package, so
the patch bundle carries a manifest of exact before/after file hashes.  A
component is accepted only when every affected file is either:

* at the recorded r14 input state, in which case ``patch`` runs with fuzz
  disabled and the output hashes are verified; or
* already at the complete tested output state, in which case the operation is
  idempotently skipped.

A mixed or unknown state is fatal.  This prevents a later parent-image refresh
from partially applying a field repair to source it was not reviewed against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIFF_PATH_RE = re.compile(r"^diff --git a/(.+) b/(.+)$", re.MULTILINE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"expected a regular file: {path}")
    return sha256_bytes(path.read_bytes())


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe manifest path: {value!r}")
    return path


def validate_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be null or a lowercase SHA-256")
    return value


def resolve_target(
    target: str,
    *,
    site_packages: Path,
    vllm_source: Path,
) -> Path:
    if target == "site-packages":
        return site_packages
    if target == "vllm-source":
        return vllm_source
    raise ValueError(f"unknown patch target: {target!r}")


def validate_component(component: dict[str, Any], manifest_dir: Path) -> None:
    required = {
        "name",
        "base_tree",
        "tested_head",
        "patch",
        "patch_sha256",
        "targets",
        "files",
    }
    missing = sorted(required - component.keys())
    if missing:
        raise ValueError(
            f"component is missing required keys: {', '.join(missing)}"
        )

    name = component["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("component name must be a non-empty string")
    if not isinstance(component["base_tree"], str) or not component["base_tree"]:
        raise ValueError(f"{name}: base_tree must be a non-empty string")
    if (
        not isinstance(component["tested_head"], str)
        or not component["tested_head"]
    ):
        raise ValueError(f"{name}: tested_head must be a non-empty string")

    patch_rel = safe_relative_path(component["patch"])
    patch_path = manifest_dir / patch_rel
    if not patch_path.is_file():
        raise ValueError(f"{name}: patch is missing: {patch_path}")
    expected_patch_hash = validate_hash(
        component["patch_sha256"], f"{name}.patch_sha256"
    )
    actual_patch_hash = sha256_bytes(patch_path.read_bytes())
    if actual_patch_hash != expected_patch_hash:
        raise ValueError(
            f"{name}: patch SHA-256 mismatch: "
            f"expected {expected_patch_hash}, got {actual_patch_hash}"
        )

    targets = component["targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(target not in {"site-packages", "vllm-source"} for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError(f"{name}: targets must be a unique non-empty target list")

    files = component["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{name}: files must be a non-empty object")
    for rel, states in files.items():
        safe_relative_path(rel)
        if not isinstance(states, dict) or set(states) != {"before", "after"}:
            raise ValueError(
                f"{name}: {rel!r} must have exactly before/after hashes"
            )
        before = validate_hash(states["before"], f"{name}.{rel}.before")
        after = validate_hash(states["after"], f"{name}.{rel}.after")
        if before == after:
            raise ValueError(f"{name}: {rel!r} does not change")

    patch_paths: set[str] = set()
    for old, new in DIFF_PATH_RE.findall(
        patch_path.read_text(encoding="utf-8", errors="strict")
    ):
        patch_paths.add(old)
        patch_paths.add(new)
    if patch_paths != set(files):
        missing_from_manifest = sorted(patch_paths - set(files))
        missing_from_patch = sorted(set(files) - patch_paths)
        raise ValueError(
            f"{name}: patch/manifest path mismatch; "
            f"unmanifested={missing_from_manifest}, absent={missing_from_patch}"
        )


def state_for(
    root: Path, files: dict[str, dict[str, str | None]]
) -> tuple[bool, bool, list[str]]:
    all_before = True
    all_after = True
    details: list[str] = []
    for rel, states in sorted(files.items()):
        actual = file_sha256(root / safe_relative_path(rel))
        before = states["before"]
        after = states["after"]
        all_before &= actual == before
        all_after &= actual == after
        if actual != before and actual != after:
            details.append(
                f"{rel}: expected before={before} or after={after}, got={actual}"
            )
        elif actual == before and actual != after:
            details.append(f"{rel}: before")
        elif actual == after and actual != before:
            details.append(f"{rel}: after")
    return all_before, all_after, details


def apply_component(
    component: dict[str, Any],
    *,
    manifest_dir: Path,
    site_packages: Path,
    vllm_source: Path,
) -> None:
    name = component["name"]
    patch_path = manifest_dir / safe_relative_path(component["patch"])
    files = component["files"]
    for target in component["targets"]:
        root = resolve_target(
            target,
            site_packages=site_packages,
            vllm_source=vllm_source,
        )
        if not root.is_dir():
            raise RuntimeError(f"{name}: patch root does not exist: {root}")

        all_before, all_after, details = state_for(root, files)
        if all_after:
            print(f">>> field review {name}: already applied at {root}")
            continue
        if not all_before:
            rendered = "\n  ".join(details)
            raise RuntimeError(
                f"{name}: refusing mixed/unknown source state at {root}:\n"
                f"  {rendered}"
            )

        result = subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                "--no-backup-if-mismatch",
                "-p1",
                "-d",
                str(root),
                "-i",
                str(patch_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(
                f"{name}: patch failed at {root} (rc={result.returncode}):\n"
                f"{result.stdout}"
            )

        _, all_after, details = state_for(root, files)
        if not all_after:
            rendered = "\n  ".join(details)
            raise RuntimeError(
                f"{name}: post-patch hash verification failed at {root}:\n"
                f"  {rendered}"
            )
        print(
            f">>> field review {name}: applied at {root} "
            f"(base tree {component['base_tree']}, "
            f"tested head {component['tested_head']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/opt/field-review-patches/manifest.json"),
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path(sysconfig.get_paths()["purelib"]),
    )
    parser.add_argument(
        "--vllm-source",
        type=Path,
        default=Path("/opt/vllm"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate manifest and patch digests without touching targets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("field-review manifest schema must be 1")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("field-review manifest needs at least one component")

    for component in components:
        if not isinstance(component, dict):
            raise ValueError("each field-review component must be an object")
        validate_component(component, manifest_path.parent)

    if args.validate_only:
        print(f">>> validated {len(components)} field-review patch components")
        return 0

    for component in components:
        apply_component(
            component,
            manifest_dir=manifest_path.parent,
            site_packages=args.site_packages.resolve(),
            vllm_source=args.vllm_source.resolve(),
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: field-review patch bundle: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
