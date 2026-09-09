#!/usr/bin/env python3
"""Write build-time evidence of the installed runtime, without loading CUDA.

Run with the serving interpreter after all runtime installers, before publishing.
The parent image reference is a build input; the output image digest is necessarily
unknown until the image is pushed and must be attached to the release externally.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import apply_glm53_refresh
import apply_glm53_selected
import glm_config
import patch_lmcache_admin_api
import patch_scopedlmcache_retrieve


def file_record(path: Path, expected: str | None = None) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"required regular file missing: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    identity = digest.hexdigest()
    if expected is not None and identity != expected:
        raise RuntimeError(f"installed source mismatch: {path}: {identity} != {expected}")
    result = {"path": str(path), "resolved_path": str(resolved),
              "size_bytes": after.st_size, "sha256": identity}
    if expected is not None:
        result["expected_sha256"] = expected
    return result


def module_records() -> dict:
    # Top-level find_spec does not import the GPU runtime. A checkout without
    # distribution metadata is recorded explicitly, not assigned a fake version.
    distribution_names = importlib.metadata.packages_distributions()
    result = {}
    for name in ("vllm", "b12x", "torch", "triton", "lmcache", "tilelang"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not spec.submodule_search_locations:
            raise RuntimeError(f"required runtime package unavailable: {name}")
        distributions = distribution_names.get(name, [])
        result[name] = {
            "source": file_record(Path(spec.origin)),
            "package_roots": [str(Path(p).resolve(strict=True))
                              for p in spec.submodule_search_locations],
            "distributions": [{"name": d, "version": importlib.metadata.version(d)}
                              for d in sorted(distributions)],
            "version_status": "distribution_metadata" if distributions
                              else "no_distribution_metadata; source identity recorded",
        }
    # vLLM intentionally bypasses ExLlama's serving frontend and loads the
    # encoder from VLLM_EXL3_ENCODER_SOURCE under a private module namespace.
    # Top-level importability is therefore not its runtime contract.
    encoder = Path(os.environ.get(
        "VLLM_EXL3_ENCODER_SOURCE", glm_config.EXL3_ENCODER_SOURCE)).resolve(strict=True)
    result["exllamav3"] = {
        "source": file_record(encoder / "__init__.py"),
        "encoder": file_record(encoder / "modules/quant/exl3_lib/quantize.py"),
        "package_roots": [str(encoder)],
        "distributions": [],
        "version_status": "configured source encoder; frontend intentionally not imported",
    }
    return result


def installed_overlay_sources(modules: dict) -> list[dict]:
    """Compose reviewed layers, then hash each final target exactly once."""
    targets = {}
    for installer in (apply_glm53_refresh, apply_glm53_selected):
        if not installer.OVERLAYS:
            raise RuntimeError(f"empty critical installer manifest: {installer.__name__}")
        runtime_paths = {Path(path) for _, path, _, _ in
                         installer.resolve_targets(mirror_root=None)}
        mirror_paths = {Path(path) for _, path, _, _ in installer.resolve_targets(
            mirror_root=installer.DEFAULT_MIRROR_ROOT)} - runtime_paths
        for payload, destination, before, expected in installer.install_targets():
            path = Path(destination)
            previous = targets.get(path)
            if previous is not None and previous["expected"] != before:
                raise RuntimeError(f"discontinuous critical source layers: {path}")
            layers = [] if previous is None else previous["layers"]
            layers.append({"installer": installer.__name__, "payload": payload,
                           "before_sha256": before, "after_sha256": expected})
            targets[path] = {"expected": expected, "layers": layers,
                             "is_mirror": path in mirror_paths}
    result = []
    for path, target in targets.items():
        record = file_record(path, target["expected"])
        installed = Path(record["resolved_path"])
        owners = [name for name, info in modules.items()
                  if any(installed.is_relative_to(Path(root))
                         for root in info["package_roots"])]
        # A graft reaches mirror trees through symlinks from the unpack
        # prefix, so compare prefix-normalized paths rather than raw
        # absolute equality; behavior is unchanged without the prefix.
        prefix = os.environ.get("TURNKEY_ROOTFS_PREFIX", "").rstrip("/")
        installed_s, path_s = str(installed), str(path)
        if prefix and installed_s.startswith(prefix + "/"):
            installed_s = installed_s[len(prefix):]
        if prefix and path_s.startswith(prefix + "/"):
            path_s = path_s[len(prefix):]
        mirror_match = target["is_mirror"] and installed_s == path_s
        if len(owners) == 1:
            record.update(module=owners[0], role="imported_runtime")
        elif not owners and mirror_match:
            record.update(module=None, role="base_image_source_tree")
        else:
            raise RuntimeError(f"critical target not in one resolved runtime package: {path}")
        final_layer = target["layers"][-1]
        record.update(installer=final_layer["installer"], payload=final_layer["payload"],
                      source_layers=target["layers"])
        result.append(record)
    return result


def installed_sources(modules: dict) -> list[dict]:
    result = installed_overlay_sources(modules)
    cache_path = patch_scopedlmcache_retrieve.default_target()
    record = file_record(cache_path, patch_scopedlmcache_retrieve.AFTER_SHA256)
    record.update(installer="patch_scopedlmcache_retrieve",
                  module=patch_scopedlmcache_retrieve.MODULE)
    result.append(record)
    layout_path = patch_scopedlmcache_retrieve.default_target(layout=True)
    patch_scopedlmcache_retrieve.patch(layout_path, verify_only=True, layout=True)
    record = file_record(layout_path, patch_scopedlmcache_retrieve.LAYOUT_AFTER_SHA256)
    record.update(installer="patch_scopedlmcache_retrieve",
                  module=patch_scopedlmcache_retrieve.LAYOUT_MODULE)
    result.append(record)
    admin_path = patch_lmcache_admin_api.default_target()
    patch_lmcache_admin_api.patch(admin_path, verify_only=True)
    record = file_record(admin_path)
    record.update(installer="patch_lmcache_admin_api", verification="installer_verify_only")
    result.append(record)
    return result


def native_libraries(modules: dict) -> dict:
    # The parent explicitly selects NCCL through this variable. Do not invent
    # a fallback library path or pretend a link name establishes its bytes.
    selected = os.environ.get("VLLM_NCCL_SO_PATH")
    if not selected or not Path(selected).is_absolute():
        raise RuntimeError("VLLM_NCCL_SO_PATH must identify the installed NCCL library")
    nccl = {Path(selected)}
    for name in ("NCCL_LOCAL_INFERENCE_PATH", "NCCL_PR2127_PATH"):
        value = os.environ.get(name)
        if value:
            nccl.add(Path(value))
    for value in re.split(r"[:\s]+", os.environ.get("LD_PRELOAD", "")):
        if "nccl" in value:
            if not Path(value).is_absolute():
                raise RuntimeError(f"NCCL preload must be an absolute path: {value}")
            nccl.add(Path(value))
    lib_dirs = {Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p}
    lib_dirs.update(Path(p) / "lib" for p in modules["torch"]["package_roots"])
    lib_dirs.update(Path(p) / "nvidia/nccl/lib" for p in sys.path if p)
    lib_dirs.add(Path("/opt"))
    lib_dirs.add(Path("/usr/lib/x86_64-linux-gnu"))
    for directory in lib_dirs:
        if directory.is_dir():
            nccl.update(directory.glob("libnccl*.so*"))
    # Gilded Gnosis splits ExLlama: the encoder Python source is loaded from
    # VLLM_EXL3_ENCODER_SOURCE, while the native extension lives in the
    # separate VLLM_EXL3_EXT_PATH directory. Record both.
    exllama = set()
    roots = {Path(p) for p in modules["exllamav3"]["package_roots"]}
    ext_path = Path(os.environ.get("VLLM_EXL3_EXT_PATH", "/opt/exllamav3"))
    if ext_path.is_dir():
        roots.add(ext_path)
    for root in roots:
        exllama.update(root.rglob("*.so"))
    for directory in {Path(p) for p in sys.path if p}:
        if directory.is_dir():
            exllama.update(directory.glob("exllama*.so"))
    if not exllama:
        raise RuntimeError("required installed ExLlama native extension was not found")
    return {"nccl_selected": selected,
            "nccl": [file_record(p) for p in sorted(nccl)],
            "exllama": [file_record(p) for p in sorted(exllama)],
            "scope": "installed libraries on configured runtime paths; not a live linker/GPU trace"}


def collect(parent_image: str, source_revision: str | None) -> dict:
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", parent_image):
        raise ValueError("parent image must be an immutable repository@sha256:digest reference")
    modules = module_records()
    scripts = (apply_glm53_refresh, apply_glm53_selected, glm_config,
               patch_lmcache_admin_api, patch_scopedlmcache_retrieve)
    return {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent_image": parent_image,
        "parent_identity_source": "build argument; installed content separately hashed below",
        "image_digest": None,
        "image_digest_status": "unknown_at_build_time; attach pushed manifest digest externally",
        "appliance_source_revision": source_revision,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "critical_sources": installed_sources(modules),
        "modules": modules,
        "packages": sorted(
            ({"name": dist.metadata["Name"], "version": dist.version}
             for dist in importlib.metadata.distributions() if dist.metadata["Name"]),
            key=lambda d: (d["name"].lower(), d["version"])),
        "native_libraries": native_libraries(modules),
        "model_families": glm_config.FAMILIES,
        "model_profiles": {
            name: {key: variant[key] for key in
                   ("family", "repo", "revision", "tested", "defaults", "runtime_env")
                   if key in variant}
            for name, variant in sorted(glm_config.VARIANTS.items())
        },
        "registry_and_installers": [file_record(Path(module.__file__)) for module in scripts],
        "generator": file_record(Path(__file__)),
        "qualification": "filesystem provenance only; no GPU or 500K capacity qualification",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-image", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path, default=Path("/opt/runtime-provenance.json"))
    args = parser.parse_args()
    document = collect(args.parent_image, args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=f".{args.output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Runtime provenance: {args.output} ({len(document['critical_sources'])} critical sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
