#!/usr/bin/env python3
"""Install reviewed retrieve and layout overlays into the imported LMCache wheel.

Base: Gilded Gnosis v20 r34 (docker.io/voipmonitor/vllm@sha256:820181fb...) with
the reviewed r34 maintenance layer already applied, so the adapter before state is
the 0944cccb bounded-retrieve baseline. The complete after identity adds fatal
timeout / completion-query-error / health-loss ownership retention on top of it:
a retrieve is only ever reported failed once a device completion query succeeded.
No runtime shim; the installed adapter already accepts kv_connector_extra_config
and retrieve_timeout.
The --layout overlay isolates opaque DCP pages using the exact installed vLLM
configuration precedence, pinned independently from LMCache. Its target is the
external lmcache.integration.vllm.lmcache_mp_connector module, which the installed
vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py resolves
LMCacheMPConnector to unless LMCACHE_USE_UPSTREAM_MP is set.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile

MODULE = "lmcache.integration.vllm.vllm_multi_process_adapter"
BEFORE_SHA256 = "0781f930e304992c75ebf596030b6a3f6d0bf697558de14168168531ee51fe21"
AFTER_SHA256 = "c6840310b59375123416f9f91a77db19124b5b1487e22d9b4962eab1940fbd66"
LAYOUT_MODULE = "lmcache.integration.vllm.lmcache_mp_connector"
LAYOUT_BEFORE_SHA256 = "c6e0bf5c79e5b21a703ade532258514f4c2967626841ec1e9ca6eafa9d7ccae3"
LAYOUT_AFTER_SHA256 = "6a5f422b7425e70605379388e7d0eae4786320e18f73e227d5d06a554c7d2e2d"
# Both exact compositions retain the same validate_block_size implementation.
# The selected fairness layer changes other configuration code, not KV layout.
VLLM_CONFIG_SHA256S = frozenset({
    "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f",
    "b40fc3b5d131851d15f2e88d550a02abd11f043d998ec927c60ba299e424b853",
})


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
    # Resolve the imported package (/opt/venv/.../site-packages/lmcache), never a
    # build checkout: the wheel source tree is not what the runtime imports.
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
        if hashlib.sha256(vllm_config_path.read_bytes()).hexdigest() not in VLLM_CONFIG_SHA256S:
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
