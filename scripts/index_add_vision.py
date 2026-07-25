#!/usr/bin/env python3
"""Register the vision tower + projector shards in the checkpoint's index.

Usage: index_add_vision.py <model_dir> [--revert]

vLLM resolves weights through model.safetensors.index.json; the two vision
shards are copied in alongside the text shards, so their tensors must appear in
the weight_map or they are simply never loaded (silent text-only behaviour).
Backs up the original index to model.safetensors.index.json.text-only.
"""
import json
import os
import sys

from safetensors import safe_open

model_dir = sys.argv[1]
IDX = os.path.join(model_dir, "model.safetensors.index.json")
BAK = IDX + ".text-only"

if "--revert" in sys.argv:
    if os.path.exists(BAK):
        os.replace(BAK, IDX)
        print("index reverted (text-only)")
    sys.exit(0)

if not os.path.exists(BAK):
    import shutil
    shutil.copy2(IDX, BAK)

idx = json.load(open(BAK))
wm = dict(idx["weight_map"])
added = 0
for shard in ("vision_tower.safetensors", "mm_projector.safetensors"):
    path = os.path.join(model_dir, shard)
    if not os.path.exists(path):
        raise SystemExit(f"missing {shard}")
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            wm[k] = shard
            added += 1
idx["weight_map"] = wm
idx.setdefault("metadata", {})["total_size"] = sum(
    os.path.getsize(os.path.join(model_dir, s)) for s in set(wm.values())
    if os.path.exists(os.path.join(model_dir, s)))
json.dump(idx, open(IDX, "w"))
print(f"index: +{added} vision tensors registered")
