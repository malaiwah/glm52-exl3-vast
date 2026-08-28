#!/usr/bin/env python3
"""Idempotently apply the deepseek_mtp rank-sliced name-normalization fix to
the image's vLLM (upstream PR: voipmonitor/vllm#11). Required to LOAD a
rank-sliced EXL3 MTP overlay; without it boot fails with
KeyError '...routed_experts.w2_rank0.mcg'.

Exit 0 = patched or already patched; exit 1 = anchor missing (fall back to
BF16 draft).
"""
import os
import shutil
import sys
import vllm.model_executor.models.deepseek_mtp as m

F = m.__file__
s = open(F).read()
if "normalize_rank_sliced_weight_name" in s:
    print("deepseek_mtp: already patched")
    sys.exit(0)
OLD = """        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer
        _pending_fp8_linear: dict = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue"""
NEW = """        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer
        _pending_fp8_linear: dict = {}
        # mtp78: normalize EXL3 rank-sliced checkpoint names (voipmonitor/vllm#11)
        rank_sliced_name = getattr(
            self.quant_config,
            "normalize_rank_sliced_weight_name",
            None,
        ) if getattr(self, "quant_config", None) is not None else None
        for name, loaded_weight in weights:
            if rank_sliced_name is not None:
                name = rank_sliced_name(name)
                if name is None:
                    continue
            if "rotary_emb.inv_freq" in name:
                continue"""
if OLD not in s:
    print("deepseek_mtp: anchor not found in this image build — cannot patch")
    sys.exit(1)
if s.count(OLD) != 1:
    # Every sibling applier enforces exactly-one anchor; an unbounded replace
    # over a future image that duplicates this block would silently
    # double-patch it. The exit-1 fallback (BF16 draft) is the safe answer.
    print("deepseek_mtp: anchor is not unique in this image build — cannot patch")
    sys.exit(1)
# Back up the original source first (atomic), then write the patch through a
# temp file + os.replace so an interrupted write cannot leave a half-patched
# module that breaks every subsequent vLLM boot.
bak = F + ".orig"
if not os.path.exists(bak):
    _bak_tmp = bak + ".tmp"
    shutil.copy2(F, _bak_tmp)
    os.replace(_bak_tmp, bak)
_patch_tmp = F + ".tmp"
with open(_patch_tmp, "w") as fh:
    fh.write(s.replace(OLD, NEW))
os.replace(_patch_tmp, F)
import compileall
if not compileall.compile_file(F, quiet=2):
    os.replace(bak, F)
    raise SystemExit("deepseek_mtp: patch compiled with errors; restored the "
                     "original module from " + bak)
print("deepseek_mtp: patched OK")
