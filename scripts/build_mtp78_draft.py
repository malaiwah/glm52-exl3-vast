#!/usr/bin/env python3
"""Build a standalone MTP78 draft directory — no surgery on the target checkpoint.

Usage: build_mtp78_draft.py <model_dir> <overlay_dir> <out_dir>

The "graft" this replaces was only ever three things:
  1. swap model-layer-078.safetensors (19 GB BF16 -> ~4 GB trellis)
  2. hybrid_tr3_tail.moe_layers  [3, 77] -> [3, 78]
  3. drop "model.layers.78*" from quantization_config.ignore

All three are expressible in a *separate* directory that vLLM loads as the
speculative draft via --speculative-config '{"model": "<out_dir>", ...}'. That
leaves the target checkpoint byte-identical, which matters for three reasons:

  * The base checkpoint is often shared (hardlinked, read-only, or reused by
    another config). In-place rewriting means .orig backups, revert paths and a
    ".mtp78-grafted" marker — state on the volume that can disagree with the
    state of the container, which is exactly how the vision plugin produced a
    158-restart crash-loop.
  * It is trivially reversible: rm -rf the draft dir. No revert script that can
    itself fail halfway.
  * Our own 4-arm QC measured the override path as the *better* one:
    MAL 3.548 / 84.9% accept vs 3.517 / 83.9% for the grafted checkpoint.

The layer-78 shard is hardlinked when possible, so the draft dir costs ~0 extra
bytes on the same filesystem.
"""
import json
import os
import shutil
import sys

model_dir, overlay_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# The overlay ships tr3-layer-078.safetensors (+ a .done.json manifest).
src_shard = None
for name in ("tr3-layer-078.safetensors", "model-layer-078.safetensors"):
    cand = os.path.join(overlay_dir, name)
    if os.path.exists(cand):
        src_shard = cand
        break
if src_shard is None:
    raise SystemExit(f"no layer-78 shard found in {overlay_dir}")

os.makedirs(out_dir, exist_ok=True)
dst_shard = os.path.join(out_dir, "model-layer-078.safetensors")
if os.path.exists(dst_shard):
    os.remove(dst_shard)
try:
    os.link(src_shard, dst_shard)          # same fs: free
except OSError:
    shutil.copy2(src_shard, dst_shard)     # cross-device: pay once

# Tokenizer/aux files: the draft ModelConfig resolves them from its own dir.
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "generation_config.json", "vocab.json", "merges.txt"):
    s = os.path.join(model_dir, f)
    d = os.path.join(out_dir, f)
    if os.path.exists(s) and not os.path.exists(d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

# Config: the target's, plus the two edits that make layer 78 a trellis MoE.
cfg = json.load(open(os.path.join(model_dir, "config.json")))
# UNWRAP a multimodal wrapper. With vision installed, the target's config.json is
# the Glm5v wrapper, whose auto_map points at configuration_glm5v.Glm5vConfig —
# a file that does not exist in the draft dir, so the draft dies with
#   OSError: ... does not appear to have a file named configuration_glm5v.py
# The MTP draft operates on the TEXT model, so the text_config is what it wants
# (the same substitution chronarion's plugin makes for the speculative config).
# The wrapper holds quantization_config, so carry it down if the text side lacks it.
if "text_config" in cfg and isinstance(cfg["text_config"], dict):
    outer_quant = cfg.get("quantization_config")
    cfg = dict(cfg["text_config"])
    if outer_quant and not cfg.get("quantization_config"):
        cfg["quantization_config"] = outer_quant
    cfg.pop("auto_map", None)
_src_moe = cfg.get("hybrid_tr3_tail", {}).get("moe_layers") or [3]
cfg.setdefault("hybrid_tr3_tail", {})["moe_layers"] = [int(_src_moe[0]), 78]
q = cfg.get("quantization_config") or {}
if q:
    q["ignore"] = [x for x in (q.get("ignore") or [])
                   if not x.startswith("model.layers.78")]
    cfg["quantization_config"] = q
json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

# Read tensor names straight out of the safetensors header (8-byte LE length,
# then a JSON blob) rather than importing safetensors — this script also runs on
# hosts without that package, and it only needs the key names.
import struct  # noqa: E402


def tensor_names(path):
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
    header_offset = 8 + hdr_len
    end = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        offs = v.get("data_offsets") if isinstance(v, dict) else None
        if offs and len(offs) == 2:
            end = max(end, int(offs[1]))
    if os.path.getsize(path) < header_offset + end:
        raise ValueError(
            "truncated safetensors: %s declares %d bytes of tensor data but "
            "only %d bytes are on disk" % (path, header_offset + end,
                                            os.path.getsize(path)))
    return [k for k in header if k != "__metadata__"]


keys = tensor_names(dst_shard)

# The overlay carries ONLY the trellis expert tensors. Layer 78 also has ~23
# non-expert tensors (enorm/hnorm, input_layernorm, mlp.gate, eh_proj, the
# attention projections) that stay BF16 and live in the checkpoint's original
# layer-78 shard. Without them the draft cannot load. Hardlink that shard in and
# map just those names to it — the file is shared, so this costs no extra bytes,
# and we avoid re-materialising a merged 4 GiB shard the way the in-place graft does.
base_shard = os.path.join(model_dir, "model-layer-078.safetensors")
weight_map = {k: "model-layer-078.safetensors" for k in keys}
carried = 0
if os.path.exists(base_shard):
    # EXTRACT the non-expert tensors into their own small shard. Hardlinking the
    # whole BF16 layer-78 shard and merely leaving its experts out of weight_map
    # does NOT work: vLLM's loader walks the FILES referenced by the index and
    # yields every tensor inside them, so the 768 BF16 expert weights reach the
    # model anyway and it dies with
    #   KeyError: 'model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight'
    # (the same mixed BF16+trellis failure the keep64 variants hit — the EXL3
    # runtime has no mixed-expert path). Only these ~23 tensors may come from BF16.
    from safetensors import safe_open          # read only what we need
    from safetensors.torch import save_file

    have = set(keys)
    want = [k for k in tensor_names(base_shard)
            if k.startswith("model.layers.78.") and ".experts." not in k
            and k not in have]
    if want:
        base_name = "nonexpert-layer-078.safetensors"
        # safe_open + per-key get_tensor keeps peak RSS to these tensors only,
        # instead of pulling the whole 19 GB BF16 shard into memory.
        with safe_open(base_shard, framework="pt") as fh:
            save_file({k: fh.get_tensor(k) for k in want},
                      os.path.join(out_dir, base_name))
        for k in want:
            weight_map[k] = base_name
        carried = len(want)
print(f"  trellis tensors: {len(keys)}  |  BF16 non-expert carried: {carried}")
total = sum(os.path.getsize(os.path.join(out_dir, f))
            for f in set(weight_map.values())
            if os.path.exists(os.path.join(out_dir, f)))
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))

print(f"MTP78 draft dir built: {out_dir} "
      f"({len(keys)} tensors, {os.path.getsize(dst_shard)/2**30:.1f} GiB, "
      f"target checkpoint untouched)")
