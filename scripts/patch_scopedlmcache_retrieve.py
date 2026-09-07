#!/usr/bin/env python3
"""Install reviewed retrieve and layout overlays into the imported LMCache wheel.

Source: local-inference-lab/LMCache tree e045d729bc5c4c63a40e13d032f42923de97812f,
adapter git blob dd5aa0e6033292eaad06dd8f0b7961d58cf6b2f1. The complete after
identity includes deadline consumption, device-event polling, terminal failure
recompute and fatal timeout/query-error/health-loss ownership retention. No runtime shim.
The --layout overlay isolates opaque DCP pages using the exact installed vLLM
configuration precedence, pinned independently from LMCache.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile

MODULE = "lmcache.integration.vllm.vllm_multi_process_adapter"
BEFORE_SHA256 = "ea580badefb9a0fad5fa2ac1bdcff4f2b36fc85138d95a5be4b180f07dd2c874"
AFTER_SHA256 = "c6840310b59375123416f9f91a77db19124b5b1487e22d9b4962eab1940fbd66"
LAYOUT_MODULE = "lmcache.integration.vllm.lmcache_mp_connector"
LAYOUT_BEFORE_SHA256 = "c52ce0807698db5899a4031e4a76a12fdbdea914404c324709bbfb7f60b1cedc"
LAYOUT_AFTER_SHA256 = "9179f51b2dd4854641175d79722e5c0d0e4a30c776d88d39185d983fdf2c99c0"
VLLM_CONFIG_SHA256 = "7cf467579647fceeea885d40fe069b6187baee3fd717ac2f51fe7f0ee1bf0a96"


def package_source(package: str, relative: str) -> Path:
    """Resolve one installed source without executing package initializers."""
    spec = importlib.util.find_spec(package)
    roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(roots) != 1:
        raise RuntimeError(f"Expected one unambiguous installed {package} package root")
    root = Path(roots[0]).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise RuntimeError(f"Installed {package} source is unavailable: {target}")
    return target


def default_target(layout: bool = False) -> Path:
    # The candidate builds a wheel from /opt/infernal-invocation/lmcache, but
    # imports /opt/venv/.../site-packages/lmcache. Never patch the build checkout.
    filename = "lmcache_mp_connector.py" if layout else "vllm_multi_process_adapter.py"
    return package_source("lmcache", f"integration/vllm/{filename}")


def default_payload(layout: bool = False) -> Path:
    root = Path(__file__).resolve().parents[1]
    filename = "lmcache_mp_connector.py" if layout else "vllm_multi_process_adapter.py"
    installed = root / "scopedlmcache" / filename
    if installed.is_file():
        return installed
    return root / "patches" / "scopedlmcache" / filename


def patch(
    path: Path, payload: Path | None = None, verify_only: bool = False,
    *, layout: bool = False, vllm_config_path: Path | None = None,
) -> str:
    if layout:
        if vllm_config_path is None:
            vllm_config_path = package_source("vllm", "config/vllm.py")
        if hashlib.sha256(vllm_config_path.read_bytes()).hexdigest() != VLLM_CONFIG_SHA256:
            raise RuntimeError("Unreviewed vLLM interleave normalization; refusing layout patch")
    before_hash = LAYOUT_BEFORE_SHA256 if layout else BEFORE_SHA256
    after_hash = LAYOUT_AFTER_SHA256 if layout else AFTER_SHA256
    path = path.resolve()
    original = path.read_bytes()
    identity = hashlib.sha256(original).hexdigest()
    if identity == after_hash:
        compile(original, str(path), "exec")
        return "verified"
    if identity != before_hash:
        raise RuntimeError(
            f"LMCache adapter does not match reviewed source: {path}: {identity}; "
            "refusing to replace unknown installed code"
        )
    if verify_only:
        raise RuntimeError("Reviewed LMCache overlay is not applied")
    replacement = (payload or default_payload(layout)).read_bytes()
    if hashlib.sha256(replacement).hexdigest() != after_hash:
        raise RuntimeError("LMCache payload identity mismatch")
    compile(replacement, str(path), "exec")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(replacement)
            stream.flush()
            os.fchmod(stream.fileno(), path.stat().st_mode)
        # Build-time installation is single-owner; still reject source drift.
        if path.read_bytes() != original:
            raise RuntimeError("LMCache adapter changed during installation")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "patched"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--vllm-config", type=Path)
    args = parser.parse_args(argv)
    target = args.path or default_target(args.layout)
    result = patch(
        target, args.payload, args.verify_only, layout=args.layout,
        vllm_config_path=args.vllm_config,
    )
    kind = "canonical layout namespace" if args.layout else "fail-stop retrieve deadline"
    print(f"LMCache {kind}: {result}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
