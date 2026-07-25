#!/usr/bin/env python3
"""Graft the validated MTP78 3bpw-keep0 trellis overlay onto the checkpoint,
in place and revertibly.

Forward:  python3 graft_mtp78.py <model_dir> <overlay_dir>
Revert:   python3 graft_mtp78.py <model_dir> --revert

Forward keeps .orig backups of the three touched files (layer-78 shard,
index, config); revert restores them. The trellis draft was validated at
BF16-parity acceptance (MAL 3.06 vs 3.05) — see
huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78.
"""
import json
import os
import shutil
import sys

LSHARD = "model-layer-078.safetensors"
IDX = "model.safetensors.index.json"
CFG = "config.json"

model_dir = sys.argv[1]

if "--revert" in sys.argv:
    for f in (LSHARD, IDX, CFG):
        b = os.path.join(model_dir, f + ".orig")
        if os.path.exists(b):
            os.replace(b, os.path.join(model_dir, f))
    marker = os.path.join(model_dir, ".mtp78-grafted")
    if os.path.exists(marker):
        os.remove(marker)
    print("MTP78 graft reverted to BF16 draft.")
    sys.exit(0)

overlay_dir = sys.argv[2]
trellis_path = os.path.join(overlay_dir, "tr3-layer-078.safetensors")
assert os.path.exists(trellis_path), f"missing {trellis_path}"

from safetensors import safe_open
from safetensors.torch import save_file

# backups (idempotent: only if not already backed up)
for f in (LSHARD, IDX, CFG):
    src = os.path.join(model_dir, f)
    bak = src + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(src, bak)

# merged replacement shard: 23 non-expert layer-78 tensors + all trellis
tensors = {}
with safe_open(os.path.join(model_dir, LSHARD + ".orig"), framework="pt") as f:
    for k in f.keys():
        if not k.startswith("model.layers.78.mlp.experts."):
            tensors[k] = f.get_tensor(k)
carried = len(tensors)
with safe_open(trellis_path, framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
tmp = os.path.join(model_dir, LSHARD + ".tmp")
save_file(tensors, tmp)
os.replace(tmp, os.path.join(model_dir, LSHARD))
print(f"shard rebuilt: carried={carried} total={len(tensors)}")

# index: drop old per-expert .weight entries, add trellis names
idx = json.load(open(os.path.join(model_dir, IDX + ".orig")))
wm = {k: v for k, v in idx["weight_map"].items()
      if not k.startswith("model.layers.78.mlp.experts.")}
for k in tensors:
    wm[k] = LSHARD
idx["weight_map"] = wm
idx["metadata"]["total_size"] = sum(
    os.path.getsize(os.path.join(model_dir, s)) for s in set(wm.values()))
json.dump(idx, open(os.path.join(model_dir, IDX), "w"))

# config: span the MTP layer + stop ignoring it
cfg = json.load(open(os.path.join(model_dir, CFG + ".orig")))
t = cfg.get("hybrid_tr3_tail")
if t and int(t["moe_layers"][1]) < 78:
    t["moe_layers"] = [int(t["moe_layers"][0]), 78]
ig = cfg.get("quantization_config", {}).get("ignore", [])
cfg["quantization_config"]["ignore"] = [p for p in ig if "layers.78" not in p]
json.dump(cfg, open(os.path.join(model_dir, CFG), "w"), indent=1)

open(os.path.join(model_dir, ".mtp78-grafted"), "w").write("3bpw-keep0\n")
print("MTP78-GRAFT-OK (trellis draft active; ~3.8 GB/GPU freed for KV)")
