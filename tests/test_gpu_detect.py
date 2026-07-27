#!/usr/bin/env python3
"""GPU-count detection, against injected device lists.

    python3 tests/test_gpu_detect.py

Nothing here runs nvidia-smi. The development host has 4 GPUs serving
production traffic, and the whole point of the module is that its logic is a
pure function over the strings nvidia-smi would have produced.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import gpu_detect as g  # noqa: E402

SMI4 = "0, GPU-aaa, NVIDIA RTX PRO 6000 Blackwell\n1, GPU-bbb, NVIDIA RTX PRO 6000 Blackwell\n2, GPU-ccc, NVIDIA RTX PRO 6000 Blackwell\n3, GPU-ddd, NVIDIA RTX PRO 6000 Blackwell\n"
SMI1 = "0, GPU-zzz, NVIDIA H200\n"
FAILS, OKS = [], []


def check(name, cond, detail=""):
    (OKS if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))


CASES = [
    ("bare 4-GPU box", SMI4, {}, 4),
    ("bare 1-GPU pod", SMI1, {}, 1),
    ("CUDA_VISIBLE_DEVICES=0", SMI4, {"CUDA_VISIBLE_DEVICES": "0"}, 1),
    ("CUDA_VISIBLE_DEVICES=0,1", SMI4, {"CUDA_VISIBLE_DEVICES": "0,1"}, 2),
    ("CUDA stops at the first invalid entry (0,5,1 -> 1)", SMI4,
     {"CUDA_VISIBLE_DEVICES": "0,5,1"}, 1),
    ("CUDA stops at a repeat (0,0 -> 1)", SMI4, {"CUDA_VISIBLE_DEVICES": "0,0"}, 1),
    ("CUDA_VISIBLE_DEVICES empty means none", SMI4, {"CUDA_VISIBLE_DEVICES": ""}, 0),
    ("NVIDIA_VISIBLE_DEVICES=none", SMI4, {"NVIDIA_VISIBLE_DEVICES": "none"}, 0),
    ("Vast SSH wrapper void sentinel leaves observed GPUs usable", SMI4,
     {"NVIDIA_VISIBLE_DEVICES": "void"}, 4),
    ("NVIDIA_VISIBLE_DEVICES=all is a no-op", SMI4,
     {"NVIDIA_VISIBLE_DEVICES": "all"}, 4),
    ("the two compose (runtime 0,1 then CUDA 1)", SMI4,
     {"NVIDIA_VISIBLE_DEVICES": "0,1", "CUDA_VISIBLE_DEVICES": "1"}, 1),
    ("selection by UUID", SMI4, {"CUDA_VISIBLE_DEVICES": "GPU-ccc"}, 1),
    ("no nvidia-smi at all", "", {}, 0),
    ("nvidia-smi -L output format", "GPU 0: NVIDIA H200 (UUID: GPU-x)\n"
     "GPU 1: NVIDIA H200 (UUID: GPU-y)", {}, 2),
]


def main():
    print("=== counting what the container can actually use ===")
    for name, smi, env, want in CASES:
        got = g.detect(env, smi)["count"]
        check(f"{name}: {got}", got == want, f"want {want}")

    print("\n=== the provider's claim is corroboration, not truth ===")
    d = g.detect({"RUNPOD_GPU_COUNT": "8"}, SMI1)
    check("a provider claiming 8 while 1 is visible yields 1", d["count"] == 1)
    check("and the discrepancy is stated out loud",
          any("advertises 8" in n for n in d["notes"]), str(d["notes"]))
    d = g.detect({"RUNPOD_GPU_COUNT": "1"}, SMI1)
    check("agreement produces no noise",
          not any("advertises" in n for n in d["notes"]), str(d["notes"]))

    print("\n=== the GPU model name reaches the UI ===")
    check("name is extracted from the csv form",
          g.detect({}, SMI4)["name"] == "NVIDIA RTX PRO 6000 Blackwell")
    check("name is extracted from the -L form",
          g.detect({}, "GPU 0: NVIDIA H200 (UUID: GPU-x)")["name"] == "NVIDIA H200")
    check("mixed models are listed, not collapsed to one",
          "," in g.detect({}, "0, GPU-a, NVIDIA H200\n1, GPU-b, NVIDIA A100\n")["name"])

    print("\n=== narrowing is explained, never silent ===")
    d = g.detect({"CUDA_VISIBLE_DEVICES": "0,9,1"}, SMI4)
    check("a truncated list says so",
          any("stops enumerating" in n for n in d["notes"]), str(d["notes"]))

    print("\n=== provider wrapper compatibility ===")
    with open(os.path.join(os.path.dirname(HERE), "Dockerfile")) as handle:
        dockerfile = handle.read()
    check("the stale public Vast onstart path remains a compatibility alias",
          "ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh" in dockerfile)

    print(f"\n{len(OKS)} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
