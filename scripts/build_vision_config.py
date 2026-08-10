#!/usr/bin/env python3
"""Merge the GLM-5.2 vision wrapper config onto our EXL3-TR3 text checkpoint.

Usage: build_vision_config.py <model_dir> <vision_dir>
       build_vision_config.py <model_dir> --revert

Produces a `Glm5vForConditionalGeneration` wrapper config in <model_dir>:
  - vision_config      : from the vision repo (MoonViT-3d, Kimi-K2.6 tower)
  - text_config        : our existing EXL3 config, verbatim (quantization_config,
                         hybrid_tr3_tail, index_topk_pattern, ... all preserved)
  - quantization_config: kept at TOP level too — vLLM resolves the quant config
                         from the wrapper, and the EXL3 loader is prefix-robust
                         (it collapses `language_model.` segments), so the text
                         weights still load rank-sliced under the wrapper.
The original config.json is backed up to config.json.text-only for --revert.
"""
import json
import os
import shutil
import sys

model_dir = sys.argv[1]
CFG = os.path.join(model_dir, "config.json")
BAK = CFG + ".text-only"

if "--revert" in sys.argv:
    if os.path.exists(BAK):
        os.replace(BAK, CFG)
        print("vision config reverted (text-only)")
    for f in (".vision-enabled",):
        p = os.path.join(model_dir, f)
        if os.path.exists(p):
            os.remove(p)
    sys.exit(0)

vision_dir = sys.argv[2]
if not os.path.exists(BAK):
    _cur = json.load(open(CFG))
    _is_wrapper = ("text_config" in _cur
                   or any("Glm5v" in a for a in _cur.get("architectures") or []))
    if _is_wrapper:
        # config.json is ALREADY a Glm5v wrapper (a prior run wrapped it, or
        # reconcile rewrote it, without leaving a text-only backup). Copying it
        # verbatim as config.json.text-only would double-nest text_config on the
        # next build and make --revert restore a wrapper. Derive the backup from
        # the wrapper's own text_config instead of snapshotting the wrapper.
        _inner = _cur.get("text_config")
        if isinstance(_inner, dict):
            with open(BAK, "w") as f:
                json.dump(_inner, f, indent=1)
        else:
            sys.exit("FATAL: config.json is already a vision wrapper with no "
                     "recoverable text_config; refusing to snapshot a wrapper as "
                     "the text-only backup. Restore a text-only config.json first.")
    else:
        shutil.copy2(CFG, BAK)

text_cfg = json.load(open(BAK))              # pristine text config
vis_cfg = json.load(open(os.path.join(vision_dir, "config.json")))

merged = {k: v for k, v in vis_cfg.items() if k not in ("text_config",)}
merged["text_config"] = text_cfg
# the wrapper must advertise OUR quantization (exl3), not the vision repo's
if "quantization_config" in text_cfg:
    merged["quantization_config"] = text_cfg["quantization_config"]
else:
    merged.pop("quantization_config", None)
# hybrid_tr3_tail drives the rank-sliced runtime; surface it at top level too
if "hybrid_tr3_tail" in text_cfg:
    merged["hybrid_tr3_tail"] = text_cfg["hybrid_tr3_tail"]
merged["architectures"] = ["Glm5vForConditionalGeneration"]

_cfg_tmp = CFG + ".tmp"
with open(_cfg_tmp, "w") as f:
    json.dump(merged, f, indent=1)
os.replace(_cfg_tmp, CFG)
open(os.path.join(model_dir, ".vision-enabled"), "w").write("glm5v\n")
print(f"vision config built: arch={merged['architectures'][0]} "
      f"text_arch={text_cfg.get('architectures')} "
      f"quant={(merged.get('quantization_config') or {}).get('quant_method')}")
