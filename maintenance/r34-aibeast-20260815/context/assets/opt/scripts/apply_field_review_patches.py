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
import shutil
import subprocess
import sys
import sysconfig
import tempfile
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
    launcher_bin: Path,
) -> Path:
    if target == "site-packages":
        return site_packages
    if target == "vllm-source":
        return vllm_source
    if target == "launcher-bin":
        return launcher_bin
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
        or any(
            target not in {"site-packages", "vllm-source", "launcher-bin"}
            for target in targets
        )
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


def apply_component_at_root(
    component: dict[str, Any],
    *,
    manifest_dir: Path,
    root: Path,
    announce: bool = True,
) -> None:
    name = component["name"]
    patch_path = manifest_dir / safe_relative_path(component["patch"])
    files = component["files"]
    if not root.is_dir():
        raise RuntimeError(f"{name}: patch root does not exist: {root}")

    all_before, all_after, details = state_for(root, files)
    if all_after:
        if announce:
            print(f">>> field review {name}: already applied at {root}")
        return
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
    if announce:
        print(
            f">>> field review {name}: applied at {root} "
            f"(base tree {component['base_tree']}, "
            f"tested head {component['tested_head']})"
        )


def target_components(
    components: list[dict[str, Any]], target: str
) -> list[dict[str, Any]]:
    return [component for component in components if target in component["targets"]]


def target_component_chains(
    components: list[dict[str, Any]], target: str
) -> list[list[dict[str, Any]]]:
    """Group ordered components that share files on one target.

    Disjoint components are independently resumable.  Components that touch a
    common file form one ordered chain so a successor can consume the exact
    output hash of its prerequisite without weakening idempotence.
    """

    chains: list[list[dict[str, Any]]] = []
    chain_paths: list[set[str]] = []
    for component in target_components(components, target):
        paths = set(component["files"])
        overlapping = [
            index for index, existing in enumerate(chain_paths) if paths & existing
        ]
        if not overlapping:
            chains.append([component])
            chain_paths.append(paths)
            continue

        first = overlapping[0]
        merged_components: list[dict[str, Any]] = []
        merged_paths = set(paths)
        for index in overlapping:
            merged_components.extend(chains[index])
            merged_paths.update(chain_paths[index])
        merged_components.append(component)
        chains[first] = merged_components
        chain_paths[first] = merged_paths
        for index in reversed(overlapping[1:]):
            del chains[index]
            del chain_paths[index]
    return chains


def component_snapshots(
    ordered: list[dict[str, Any]], target: str
) -> list[dict[str, str | None]]:
    """Return exact file-hash snapshots for an ordered target patch chain."""

    initial: dict[str, str | None] = {}
    for component in ordered:
        for rel, states in component["files"].items():
            initial.setdefault(rel, states["before"])

    current = dict(initial)
    snapshots = [dict(current)]
    for component in ordered:
        name = component["name"]
        for rel, states in component["files"].items():
            if current[rel] != states["before"]:
                raise ValueError(
                    f"{name}: ordered {target} patch chain is discontinuous "
                    f"for {rel}: previous={current[rel]}, "
                    f"before={states['before']}"
                )
        for rel, states in component["files"].items():
            current[rel] = states["after"]
        snapshots.append(dict(current))
    return snapshots


def snapshot_index(
    root: Path,
    snapshots: list[dict[str, str | None]],
    *,
    target: str,
) -> int:
    actual = {rel: file_sha256(root / safe_relative_path(rel)) for rel in snapshots[0]}
    matches = [index for index, snapshot in enumerate(snapshots) if actual == snapshot]
    if matches:
        # Repeated snapshots are possible when a later patch returns every tracked
        # file to an earlier byte state.  The latest exact state is the only
        # idempotent interpretation.
        return matches[-1]

    details = []
    for rel, actual_hash in sorted(actual.items()):
        expected = sorted(
            {snapshot[rel] for snapshot in snapshots},
            key=lambda value: "" if value is None else value,
        )
        details.append(f"{rel}: expected one of {expected}, got={actual_hash}")
    raise RuntimeError(
        f"{target}: refusing mixed/unknown source state at {root}:\n  "
        + "\n  ".join(details)
    )


def copy_snapshot_files(
    source: Path, destination: Path, files: dict[str, str | None]
) -> None:
    for rel in files:
        source_path = source / safe_relative_path(rel)
        if not source_path.exists():
            continue
        destination_path = destination / safe_relative_path(rel)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def preflight_components(
    components: list[dict[str, Any]],
    *,
    manifest_dir: Path,
    site_packages: Path,
    vllm_source: Path,
    launcher_bin: Path,
) -> list[tuple[str, Path, list[dict[str, Any]], int]]:
    """Validate every target before allowing the first source mutation.

    The image build applies several independently qualified components.  A
    later component or secondary target must not be able to strand an earlier
    target in the patched state when its own source has drifted.  Exact hashes
    establish the state; a dry run also proves that GNU patch accepts every
    complete before-state with fuzz disabled.
    """

    failures: list[str] = []
    plans: list[tuple[str, Path, list[dict[str, Any]], int]] = []
    for target in ("site-packages", "vllm-source", "launcher-bin"):
        chains = target_component_chains(components, target)
        if not chains:
            continue
        root = resolve_target(
            target,
            site_packages=site_packages,
            vllm_source=vllm_source,
            launcher_bin=launcher_bin,
        )
        if not root.is_dir():
            failures.append(f"{target}: patch root does not exist: {root}")
            continue
        for ordered in chains:
            try:
                snapshots = component_snapshots(ordered, target)
                start = snapshot_index(root, snapshots, target=target)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(str(exc))
                continue
            plans.append((target, root, ordered, start))

    if failures:
        raise RuntimeError(
            "field-review preflight rejected the source tree before mutation:\n  "
            + "\n  ".join(failures)
        )

    for target, root, ordered, start in plans:
        with tempfile.TemporaryDirectory(prefix="field-review-preflight-") as tmp:
            scratch = Path(tmp)
            snapshots = component_snapshots(ordered, target)
            copy_snapshot_files(root, scratch, snapshots[0])
            try:
                for component in ordered[start:]:
                    apply_component_at_root(
                        component,
                        manifest_dir=manifest_dir,
                        root=scratch,
                        announce=False,
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"{target}: patch-chain dry run failed at {root} "
                    f"before mutation: {exc}"
                ) from exc

    return plans


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
        "--launcher-bin",
        type=Path,
        default=Path("/usr/local/bin"),
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
    for target in ("site-packages", "vllm-source", "launcher-bin"):
        for chain in target_component_chains(components, target):
            component_snapshots(chain, target)

    if args.validate_only:
        print(f">>> validated {len(components)} field-review patch components")
        return 0

    site_packages = args.site_packages.resolve()
    vllm_source = args.vllm_source.resolve()
    launcher_bin = args.launcher_bin.resolve()
    plans = preflight_components(
        components,
        manifest_dir=manifest_path.parent,
        site_packages=site_packages,
        vllm_source=vllm_source,
        launcher_bin=launcher_bin,
    )
    for _target, root, ordered, start in plans:
        for component in ordered[:start]:
            print(
                f">>> field review {component['name']}: "
                f"already applied at {root}"
            )
        for component in ordered[start:]:
            apply_component_at_root(
                component,
                manifest_dir=manifest_path.parent,
                root=root,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: field-review patch bundle: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
