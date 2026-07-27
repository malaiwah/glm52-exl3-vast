#!/usr/bin/env python3
"""How many GPUs can this container actually use?

`--tensor-parallel-size 4` was a literal, then a knob defaulting to 4. Both are
wrong for an image people rent on arbitrary hardware: a 1-, 2- or 8-GPU pod all
got 4 and failed, opaquely. The default must be what is actually there.

Counting is not just `nvidia-smi -L | wc -l`:

  * NVIDIA_VISIBLE_DEVICES is what the container runtime was told to expose. It
    may be `all`, `none`, or a list of indices/UUIDs.
  * CUDA_VISIBLE_DEVICES narrows it further, INSIDE the container, and CUDA's
    enumeration rule is unusual: the list is honoured up to the first invalid
    entry and everything after it is dropped. `0,5,1` on a 4-GPU box yields one
    device, not two. Getting that wrong means TP is set to more devices than
    torch will enumerate, and the engine dies during init.
  * They compose: the intersection is what torch will see.

Everything here is a pure function over strings so the resolution logic can be
tested against injected device lists — the development host has 4 GPUs and they
are busy serving production traffic.

    gpu_detect.py            -> the count, or 0
    gpu_detect.py --json     -> the full picture, including why
"""
import argparse
import json
import os
import subprocess
import sys


def parse_smi(output: str):
    """`nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader`
    -> [(index, uuid, name)]. Also accepts the `nvidia-smi -L` line format."""
    devices = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            devices.append((int(parts[0]), parts[1],
                            parts[2] if len(parts) > 2 else ""))
        elif line.startswith("GPU "):
            # `nvidia-smi -L` form: "GPU 0: NVIDIA H200 (UUID: GPU-abc...)"
            try:
                idx = int(line.split(":", 1)[0].split()[1])
            except (IndexError, ValueError):
                continue
            uuid, name = "", ""
            if "UUID:" in line:
                head, uuid = line.rsplit("UUID:", 1)
                uuid = uuid.strip().rstrip(")")
                name = head.split(":", 1)[1].strip().rstrip("(").strip()
            devices.append((idx, uuid, name))
    return devices


def _select(devices, spec):
    """Apply one visibility list. CUDA semantics: stop at the first invalid entry."""
    by_index = {str(d[0]): d for d in devices}
    by_uuid = {d[1]: d for d in devices if d[1]}
    out = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            break
        hit = by_index.get(item)
        if hit is None:
            hit = by_uuid.get(item)
        if hit is None:
            # also accept a UUID prefix, which is what people usually paste
            for u, dev in by_uuid.items():
                if u.startswith(item) and len(item) > 8:
                    hit = dev
                    break
        if hit is None or hit in out:
            break                      # invalid or duplicate: enumeration stops
        out.append(hit)
    return out


def visible(devices, env=None):
    """-> (list of (index, uuid), notes[]). Never raises."""
    env = os.environ if env is None else env
    notes = []
    sel = list(devices)

    nvd = env.get("NVIDIA_VISIBLE_DEVICES")
    # Vast's generated SSH wrapper currently exports `void` even though it
    # launches the child container with all requested GPU devices mounted.
    # nvidia-smi is our primary observation; interpreting that wrapper sentinel
    # as a UUID list turns four observed GPUs into zero and strands the rental.
    if nvd is not None and nvd.strip() == "void":
        notes.append(
            "NVIDIA_VISIBLE_DEVICES=void ignored because nvidia-smi observed "
            f"{len(sel)} mounted GPU(s)")
    elif nvd is not None and nvd.strip() not in ("", "all"):
        if nvd.strip() == "none":
            notes.append("NVIDIA_VISIBLE_DEVICES=none: the runtime exposed no GPUs")
            return [], notes
        sel = _select(sel, nvd)
        notes.append(f"NVIDIA_VISIBLE_DEVICES={nvd} narrowed the set to {len(sel)}")

    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        before = len(sel)
        sel = _select(sel, cvd)
        notes.append(f"CUDA_VISIBLE_DEVICES={cvd} narrowed {before} -> {len(sel)}")
        if len(sel) < len([x for x in cvd.split(",") if x.strip()]):
            notes.append("NB: CUDA stops enumerating at the first invalid or repeated "
                         "entry, so the list above is shorter than what was requested")
    elif cvd is not None:
        notes.append("CUDA_VISIBLE_DEVICES is set but EMPTY: no GPU is visible to torch")
        return [], notes
    return sel, notes


def run_smi():
    for args in (["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                 ["nvidia-smi", "-L"]):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def detect(env=None, smi_output=None):
    """-> dict with the count, the devices and every reason behind them."""
    env = os.environ if env is None else env
    raw = run_smi() if smi_output is None else smi_output
    devices = parse_smi(raw)
    sel, notes = visible(devices, env)
    names = sorted({d[2] for d in sel if len(d) > 2 and d[2]})
    out = {"count": len(sel), "total_present": len(devices),
           "devices": [{"index": d[0], "uuid": d[1],
                        "name": d[2] if len(d) > 2 else ""} for d in sel],
           "name": names[0] if len(names) == 1 else ", ".join(names),
           "notes": notes, "source": "nvidia-smi"}
    if not devices:
        out["notes"].append("nvidia-smi reported no devices (or is not installed)")

    # Corroboration, never the primary source: a provider's claim about the pod
    # and what the pod can actually see are different facts, and a mismatch is
    # worth saying out loud because it means the pod is not what it claims.
    claimed = env.get("RUNPOD_GPU_COUNT") or env.get("GPU_COUNT")
    if claimed:
        try:
            claimed_n = int(claimed)
            out["provider_claims"] = claimed_n
            if claimed_n != len(sel):
                out["notes"].append(
                    f"the provider advertises {claimed_n} GPU(s) but {len(sel)} are "
                    "actually visible here — trusting what is visible")
        except ValueError:
            pass
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--name", action="store_true", help="print the GPU model name")
    args = ap.parse_args(argv)
    doc = detect()
    if args.json:
        print(json.dumps(doc, indent=1))
    elif args.name:
        print(doc.get("name", ""))
    else:
        print(doc["count"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
