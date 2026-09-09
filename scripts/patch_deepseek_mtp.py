#!/usr/bin/env python3
"""Apply the rank-sliced EXL3 MTP name-normalization fix (voipmonitor/vllm#11).

Exit 0 means the unique reviewed patch is installed; exit 1 means unknown,
ambiguous or invalid source, so the caller must use its BF16 draft fallback.
"""
import os
from pathlib import Path
import tempfile

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


def patch_text(source):
    old_count, new_count = source.count(OLD), source.count(NEW)
    if new_count == 1 and old_count == 0:
        return source, "already patched"
    if old_count != 1 or new_count:
        raise ValueError("expected one unpatched or one patched MTP anchor")
    if "normalize_rank_sliced_weight_name" in source:
        raise ValueError("unrecognized partial MTP normalization patch")
    return source.replace(OLD, NEW, 1), "patched"


def main():
    import vllm.model_executor.models.deepseek_mtp as module

    target = Path(module.__file__)
    temporary = None
    try:
        source = target.read_text()
        patched, status = patch_text(source)
        compile(patched, str(target), "exec")
        if status == "already patched":
            print("deepseek_mtp: already patched")
            return 0
        backup = target.with_suffix(target.suffix + ".orig")
        if not backup.exists():
            with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(source)
            os.replace(temporary, backup)
            temporary = None
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(patched)
        temporary.chmod(target.stat().st_mode)
        os.replace(temporary, target)
    except Exception as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        print(f"deepseek_mtp: patch failed; original source retained: {error}")
        return 1
    print("deepseek_mtp: patched OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
