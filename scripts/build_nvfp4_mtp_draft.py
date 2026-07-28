#!/usr/bin/env python3
"""Build a standalone Luke NVFP4 MTP78 draft from a partial snapshot.

Usage: build_nvfp4_mtp_draft.py <source_dir> <out_dir>

Only two checkpoint payloads are needed from lukealonso/GLM-5.2-NVFP4:
model-mtp.safetensors and model-mtp-inputscales.safetensors.  The upstream
index also names the target model's other 78 layers, so it cannot be used
unchanged in a directory that intentionally contains only the speculative
layer.  This script keeps the layer-78 entries, links the two payloads and
metadata into an atomic standalone directory, and leaves the source snapshot
untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PAYLOADS = {
    "model-mtp.safetensors",
    "model-mtp-inputscales.safetensors",
}
METADATA = (
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
MARKER = ".draft-complete"
SOURCE_REPO = "lukealonso/GLM-5.2-NVFP4"


def link_or_copy(source: Path, destination: Path) -> None:
    # Hugging Face snapshots expose payloads as relative symlinks into blobs/.
    # Linking that symlink into another directory preserves its relative target
    # and creates a dangling link.  Resolve the immutable blob before linking.
    source = source.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.splitlines()[2])

    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if (output / MARKER).is_file():
        print(f"NVFP4 MTP78 draft already complete: {output}")
        return 0
    if output.exists():
        raise SystemExit(
            f"refusing to replace incomplete or unknown draft directory: {output}"
        )

    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise SystemExit(f"source snapshot is missing config/index metadata: {source}")

    upstream = json.loads(index_path.read_text())
    weight_map = {
        name: shard
        for name, shard in (upstream.get("weight_map") or {}).items()
        if name.startswith("model.layers.78.")
    }
    if not weight_map:
        raise SystemExit("upstream index contains no model.layers.78 tensors")
    shards = set(weight_map.values())
    if not shards.issubset(PAYLOADS):
        unexpected = ", ".join(sorted(shards - PAYLOADS))
        raise SystemExit(f"layer 78 unexpectedly references: {unexpected}")
    missing = [name for name in sorted(shards) if not (source / name).is_file()]
    if missing:
        raise SystemExit("partial snapshot is missing: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        for name in sorted(shards):
            link_or_copy(source / name, tmp / name)
        for name in METADATA:
            path = source / name
            if path.is_file():
                link_or_copy(path, tmp / name)

        # Preserve Luke's ModelOpt/NVFP4 quantization declaration.  It includes
        # the ignore list for the BF16-sensitive pieces of layer 78 and is what
        # makes vLLM select modelopt_fp4 for this external draft.
        shutil.copy2(config_path, tmp / "config.json")
        total = sum((tmp / shard).stat().st_size for shard in shards)
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": weight_map},
                indent=1,
            )
            + "\n"
        )
        (tmp / ".source-repo").write_text(SOURCE_REPO + "\n")
        (tmp / MARKER).write_text(
            json.dumps(
                {
                    "source_repo": SOURCE_REPO,
                    "tensors": len(weight_map),
                    "payloads": sorted(shards),
                    "bytes": total,
                },
                sort_keys=True,
            )
            + "\n"
        )
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(
        f"NVFP4 MTP78 draft built: {output} "
        f"({len(weight_map)} tensors, {total / 2**30:.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
