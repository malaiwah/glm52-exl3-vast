#!/usr/bin/env python3
"""Restrict LMCache's unauthenticated HTTP server to read-only liveness routes."""
import argparse
import importlib.util
import os
from pathlib import Path

BEFORE = """        for r in discover_api_routers(apis_path, apis_package):
            self.router.include_router(r)
"""
AFTER = """        # The MP protocol is the appliance's cache control plane. Its HTTP
        # listener is unauthenticated, so expose only read-only liveness/version
        # routes. In particular, never register /run_script, /env, cache mutation,
        # quota, config, or backend-reconfiguration handlers.
        excluded = {
            path.stem for path in apis_path.glob("*_api.py")
            if path.stem != "info_api"
        }
        for r in discover_api_routers(
            apis_path, apis_package, exclude=excluded
        ):
            self.router.include_router(r)
"""


def default_target() -> Path:
    spec = importlib.util.find_spec(
        "lmcache.v1.multiprocess.http_api_registry")
    if spec is None or not spec.origin:
        raise RuntimeError("LMCache HTTP API registry is not installed")
    return Path(spec.origin)


def patch(path: Path, verify_only: bool = False) -> str:
    text = path.read_text(encoding="utf-8")
    if text.count(AFTER) == 1 and BEFORE not in text:
        compile(text, str(path), "exec")
        return "verified"
    if verify_only:
        raise RuntimeError("LMCache HTTP API restriction is not applied")
    if text.count(BEFORE) != 1 or AFTER in text:
        raise RuntimeError(
            "LMCache HTTP API registry does not match the reviewed source; "
            "refusing an unsafe partial patch")
    patched = text.replace(BEFORE, AFTER)
    compile(patched, str(path), "exec")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(patched, encoding="utf-8")
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, path)
    return "patched"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    target = args.path or default_target()
    result = patch(target, verify_only=args.verify_only)
    print(f"LMCache HTTP API restriction: {result}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
