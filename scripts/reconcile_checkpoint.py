#!/usr/bin/env python3
"""Make config.json + the weight index agree with the TENSORS ACTUALLY ON DISK.

Why this exists. Both existing revert paths restore a config SNAPSHOT:

  * graft_mtp78.py --revert       restores config.json.orig
  * build_vision_config.py --revert restores config.json.text-only

A snapshot is only correct if nothing else changed in between, and in practice
things do. Snapshot the config while layer 78 is grafted, revert the graft
later, and the restored snapshot still says
`hybrid_tr3_tail.moe_layers = [3, 78]` while the layer-78 weights are BF16 —
the engine then dies with

    KeyError: model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight

The same shape of bug strips the Glm5v wrapper (vision "installed" but the
config is text-only, so image requests 400 with "GLM-5.2 is not a multimodal
model"). Any config transition therefore ends here, and this script derives the
three DERIVED fields from observation rather than from any saved state:

    hybrid_tr3_tail.moe_layers      <- are layer-78 experts trellis or BF16?
    quantization_config.ignore      <- same question
    architectures / wrapper         <- are the vision shards present AND wanted?
    model.safetensors.index.json    <- rebuilt from the files that exist

Usage:
    reconcile_checkpoint.py <model_dir> --vision 0|1 [--dry-run] [--quiet]

Exit 0 on success (whether or not anything changed), 1 on an unusable
checkpoint. Idempotent: running it twice changes nothing the second time.
"""
import argparse
import copy
import json
import os
import re
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc  # noqa: E402

CFG = "config.json"
IDX = "model.safetensors.index.json"
LSHARD = "model-layer-078.safetensors"
VISION_SHARDS = ("vision_tower.safetensors", "mm_projector.safetensors")
MTP_LAYER = 78
TRELLIS_EXPERT_RE = re.compile(
    r"^model\.layers\.%d\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.rank\d+\."
    r"(?:trellis|suh|svh|mcg|mul1)$" % MTP_LAYER
)
BF16_EXPERT_RE = re.compile(
    r"^model\.layers\.%d\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.(?:weight|weight_scale|weight_scale_inv)$"
    % MTP_LAYER
)
PACKED_TRELLIS_RE = re.compile(
    r"^model\.layers\.%d\..*experts.*"
    r"(?:trellis|suh|svh|mcg|mul1)$" % MTP_LAYER
)


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------

def tensor_names(path):
    """Tensor names straight from the safetensors header (8-byte LE length +
    JSON). No torch, no safetensors package — this must run early in boot.

    Also validates that the file is not truncated: the declared tensor data
    offsets must fit inside the bytes actually on disk. A truncated shard
    raises ValueError so the callers (which already catch it) treat the shard
    as unusable instead of silently building an index that misses tensors."""
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


def observe_layer78(model_dir):
    """-> 'trellis' | 'bf16' | 'absent'

    BF16 layer 78 stores one `.weight` tensor per expert. Current rank-sliced
    EXL3 overlays are also per-expert, but store one
    `.rank{r}.{trellis|suh|svh|mcg|mul1}` payload per TP rank and projection.
    Older overlays used packed expert names without the numeric expert segment.
    Check the payload suffixes, not merely whether the name is per-expert: the
    latter misclassified the current 3bpw-keep0 overlay as BF16 and made the
    reconciler undo a valid graft before every v29 boot."""
    path = os.path.join(model_dir, LSHARD)
    if not os.path.exists(path):
        return "absent"
    try:
        names = tensor_names(path)
    except (OSError, ValueError, struct.error):
        return "absent"
    if any(TRELLIS_EXPERT_RE.match(n) or PACKED_TRELLIS_RE.match(n)
           for n in names):
        return "trellis"
    if any(BF16_EXPERT_RE.match(n) for n in names):
        return "bf16"
    return "absent"


def observe_vision(model_dir):
    return all(os.path.exists(os.path.join(model_dir, s)) for s in VISION_SHARDS)


def unwrap(cfg):
    """-> (text_config, wrapper_or_None). The multimodal wrapper nests the whole
    text config under text_config.

    Deep copies deliberately: the caller mutates nested dicts (hybrid_tr3_tail,
    quantization_config) and then compares the result against the config as
    read. A shallow copy aliases those dicts, the comparison says "no change",
    and the fix is silently not written."""
    if isinstance(cfg.get("text_config"), dict):
        wrapper = copy.deepcopy({k: v for k, v in cfg.items() if k != "text_config"})
        return copy.deepcopy(cfg["text_config"]), wrapper
    return copy.deepcopy(cfg), None


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# baseline (pristine values for the fields the graft rewrites)
# --------------------------------------------------------------------------

def _layer78_ignore(cfg):
    q = cfg.get("quantization_config") or {}
    return [p for p in (q.get("ignore") or []) if "layers.%d" % MTP_LAYER in p]


def _moe_layers(cfg):
    tail = cfg.get("hybrid_tr3_tail") or {}
    ml = tail.get("moe_layers")
    if isinstance(ml, list) and len(ml) == 2:
        try:
            return [int(ml[0]), int(ml[1])]
        except (TypeError, ValueError):
            return None
    return None


def baseline(model_dir, current_text, log):
    """Pristine moe_layers + the layer-78 ignore patterns, cached on the volume.

    Sources, best first: the cached baseline, the graft's own pre-graft backup,
    the vision installer's pre-vision backup, then the current config. A source
    is only trusted for moe_layers if it does NOT already say 78 — that is the
    grafted value, and taking it would re-introduce exactly the stale-snapshot
    bug this file exists to fix."""
    current_checkpoint_id = os.environ.get("MODEL_REPO") or os.path.realpath(model_dir)
    cached = gc.read_json(gc.p_baseline())
    if (cached.get("moe_layers") and cached.get("captured")
            and cached.get("checkpoint_id") == current_checkpoint_id):
        return cached
    if cached.get("captured") and cached.get("checkpoint_id") != current_checkpoint_id:
        log(f">>> reconcile: discarding stale baseline cache (was for "
            f"{cached.get('checkpoint_id')!r}, now {current_checkpoint_id!r})")

    candidates = [("config.json.orig", os.path.join(model_dir, CFG + ".orig")),
                  ("config.json.text-only", os.path.join(model_dir, CFG + ".text-only"))]
    docs = [("current", current_text)]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                docs.insert(0, (name, unwrap(load_cfg(path))[0]))
            except (OSError, ValueError):
                pass

    moe, moe_src = None, "fallback"
    for name, doc in docs:
        ml = _moe_layers(doc)
        if ml and ml[1] != MTP_LAYER:
            moe, moe_src = ml, name
            break
    if moe is None:
        ml = _moe_layers(current_text)
        moe = [ml[0] if ml else 3, MTP_LAYER - 1]
        log(f"!!! reconcile: no pre-graft moe_layers found in any backup; assuming {moe}. "
            "If layer 78 is BF16 and the engine still asks for trellis expert tensors, "
            "delete %s and re-run." % gc.p_baseline())

    ign, ign_src = [], "fallback"
    for name, doc in docs:
        got = _layer78_ignore(doc)
        if got:
            ign, ign_src = got, name
            break
    if not ign:
        log("!!! reconcile: no layer-78 entries found in any quantization_config.ignore "
            "backup; a BF16 layer 78 will not be re-added to the ignore list.")

    doc = {"captured": True, "checkpoint_id": current_checkpoint_id,
           "moe_layers": moe, "moe_layers_source": moe_src,
           "layer78_ignore": ign, "layer78_ignore_source": ign_src,
           "architectures": current_text.get("architectures")}
    gc.write_json_atomic(gc.p_baseline(), doc, mode=0o644)
    log(f">>> reconcile: captured checkpoint baseline (moe_layers={moe} from {moe_src}, "
        f"{len(ign)} layer-78 ignore pattern(s) from {ign_src})")
    return doc


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------

def reconcile_text_config(text, base, layer78, log):
    """Fix ONLY the fields that describe layer 78, from what layer 78 IS."""
    changed = []
    tail = text.get("hybrid_tr3_tail")
    if isinstance(tail, dict):
        want = ([int(base["moe_layers"][0]), MTP_LAYER] if layer78 == "trellis"
                else list(base["moe_layers"]))
        if _moe_layers(text) != want:
            changed.append("hybrid_tr3_tail.moe_layers %s -> %s" % (_moe_layers(text), want))
            tail["moe_layers"] = want
    q = text.get("quantization_config")
    if isinstance(q, dict):
        ign = list(q.get("ignore") or [])
        have = [p for p in ign if "layers.%d" % MTP_LAYER in p]
        if layer78 == "trellis" and have:
            ign = [p for p in ign if "layers.%d" % MTP_LAYER not in p]
            changed.append("quantization_config.ignore -= %d layer-78 pattern(s)" % len(have))
        elif layer78 != "trellis" and not have and base.get("layer78_ignore"):
            ign = ign + list(base["layer78_ignore"])
            changed.append("quantization_config.ignore += %d layer-78 pattern(s)"
                           % len(base["layer78_ignore"]))
        q["ignore"] = ign
    if base.get("architectures") and text.get("architectures") != base["architectures"]:
        changed.append("architectures %s -> %s" % (text.get("architectures"),
                                                   base["architectures"]))
        text["architectures"] = base["architectures"]
    for c in changed:
        log(f">>> reconcile: {c}")
    return text


def build_wrapper(model_dir, text, current_wrapper, log):
    """Re-derive the Glm5v wrapper around the CURRENT text config.

    The vision installer builds this from a snapshot too; rebuilding it here
    means the wrapper always nests the text config we just reconciled, instead
    of whatever the text config looked like when vision was installed."""
    vis_cfg = None
    vpath = os.path.join(model_dir, ".vision", CFG)
    if os.path.exists(vpath):
        try:
            vis_cfg = load_cfg(vpath)
        except (OSError, ValueError):
            vis_cfg = None
    if vis_cfg is None and current_wrapper is not None:
        vis_cfg = dict(current_wrapper)
    if vis_cfg is None:
        log("!!! reconcile: vision requested but no wrapper config available "
            f"({vpath} missing) — leaving the config text-only")
        return None
    merged = {k: v for k, v in vis_cfg.items() if k != "text_config"}
    merged["text_config"] = text
    if "quantization_config" in text:
        merged["quantization_config"] = text["quantization_config"]
    else:
        merged.pop("quantization_config", None)
    if "hybrid_tr3_tail" in text:
        merged["hybrid_tr3_tail"] = text["hybrid_tr3_tail"]
    merged["architectures"] = ["Glm5vForConditionalGeneration"]
    return merged


def reconcile_index(model_dir, vision_on, log, dry_run=False):
    """Rebuild weight_map from the shards that exist. vLLM loads every tensor in
    every referenced file, so the map must reference exactly the right files."""
    idx_path = os.path.join(model_dir, IDX)
    if not os.path.exists(idx_path):
        return False
    shards = sorted(f for f in os.listdir(model_dir)
                    if f.endswith(".safetensors") and not f.startswith("."))
    if not vision_on:
        shards = [s for s in shards if s not in VISION_SHARDS]
    if not shards:
        log("!!! reconcile: no shards found; leaving the index alone")
        return False
    wm, total = {}, 0
    for shard in shards:
        path = os.path.join(model_dir, shard)
        try:
            for name in tensor_names(path):
                wm[name] = shard
        except (OSError, ValueError, struct.error) as e:
            log(f"!!! reconcile: cannot read {shard} ({e}); leaving the index alone")
            return False
        total += os.path.getsize(path)
    try:
        old = load_cfg(idx_path).get("weight_map") or {}
    except (OSError, ValueError):
        old = {}
    if old == wm:
        return False
    added = len(set(wm) - set(old))
    removed = len(set(old) - set(wm))
    moved = sum(1 for k in set(wm) & set(old) if wm[k] != old[k])
    log(f">>> reconcile: index rebuilt from disk (+{added} -{removed} ~{moved} entries, "
        f"{len(shards)} shards)")
    if not dry_run:
        tmp = idx_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f)
        os.replace(tmp, idx_path)
    return True


def reconcile_chat_template(model_dir, vision_on, log, dry_run=False):
    """The vision wrapper needs the vision chat template (the text-only one never
    emits the image placeholders, and the processor then fails with 'Failed to
    apply prompt replacement for mm_items'). Keep the pair consistent."""
    live = os.path.join(model_dir, "chat_template.jinja")
    text_only = live + ".text-only"
    vision_tpl = os.path.join(model_dir, ".vision", "chat_template.jinja")
    def same(a, b):
        try:
            with open(a, "rb") as fa, open(b, "rb") as fb:
                return fa.read() == fb.read()
        except OSError:
            return False

    try:
        if vision_on and os.path.exists(vision_tpl):
            already = same(live, vision_tpl)
            # Only snapshot the live template as the text-only backup when it is
            # NOT already the vision one — otherwise a second run would "back up"
            # the vision template as the text-only one and the restore path would
            # never produce a text-only template again.
            if os.path.exists(live) and not os.path.exists(text_only) and not already:
                if not dry_run:
                    shutil.copy2(live, text_only)
            if not already:
                log(">>> reconcile: installing the vision chat template")
                if not dry_run:
                    shutil.copy2(vision_tpl, live)
        elif not vision_on and os.path.exists(text_only):
            if not os.path.exists(live) or open(live, "rb").read() != open(text_only, "rb").read():
                log(">>> reconcile: restoring the text-only chat template")
                if not dry_run:
                    shutil.copy2(text_only, live)
    except OSError as e:
        log(f"!!! reconcile: chat template check failed ({e})")


def markers(model_dir, layer78, vision_on, log, dry_run=False):
    """Markers describe reality, they do not decide it."""
    pairs = [(".mtp78-grafted", layer78 == "trellis", "3bpw-keep0\n"),
             (".vision-enabled", vision_on, "glm5v\n")]
    for name, want, body in pairs:
        path = os.path.join(model_dir, name)
        have = os.path.exists(path)
        if want and not have:
            log(f">>> reconcile: marker {name} was missing while the checkpoint says "
                "otherwise — creating it")
            if not dry_run:
                with open(path, "w") as f:
                    f.write(body)
        elif have and not want:
            log(f">>> reconcile: marker {name} contradicts the weights on disk — removing it")
            if not dry_run:
                try:
                    os.remove(path)
                except OSError:
                    pass


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--vision", default="0", choices=["0", "1"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    model_dir = args.model_dir
    cfg_path = os.path.join(model_dir, CFG)
    if not os.path.exists(cfg_path):
        print(f"!!! reconcile: {cfg_path} does not exist", file=sys.stderr)
        return 1
    try:
        cfg = load_cfg(cfg_path)
    except (OSError, ValueError) as e:
        print(f"!!! reconcile: cannot read {cfg_path}: {e}", file=sys.stderr)
        return 1

    text, wrapper = unwrap(cfg)
    layer78 = observe_layer78(model_dir)
    vision_assets = observe_vision(model_dir)
    vision_on = args.vision == "1" and vision_assets
    if args.vision == "1" and not vision_assets:
        log("!!! reconcile: vision requested but the tower/projector shards are not on "
            "disk — reconciling to text-only")
    log(f">>> reconcile: layer 78 on disk = {layer78}, vision shards = "
        f"{'present' if vision_assets else 'absent'}, vision wanted = {args.vision}")
    if layer78 == "absent":
        log("!!! reconcile: layer-78 shard missing or unreadable; leaving the MTP fields "
            "untouched")

    base = baseline(model_dir, text, log)
    if layer78 != "absent":
        text = reconcile_text_config(text, base, layer78, log)

    if vision_on:
        new_cfg = build_wrapper(model_dir, text, wrapper, log) or text
    else:
        if wrapper is not None:
            log(">>> reconcile: unwrapping the vision config (vision is off)")
        new_cfg = text

    if new_cfg != cfg:
        log(">>> reconcile: rewriting config.json")
        if not args.dry_run:
            tmp = cfg_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(new_cfg, f, indent=1)
            os.replace(tmp, cfg_path)
    reconcile_index(model_dir, vision_on, log, args.dry_run)
    reconcile_chat_template(model_dir, vision_on, log, args.dry_run)
    markers(model_dir, layer78, vision_on, log, args.dry_run)
    log(">>> reconcile: checkpoint config now matches the weights on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
