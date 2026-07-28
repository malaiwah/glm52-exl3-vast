#!/usr/bin/env python3
"""Silence false XGrammar FSM errors at a reasoning/spec-decode boundary.

When reasoning ends in the middle of an MTP window, vLLM must build grammar
bitmasks for draft tokens that were sampled before those masks existed. Such a
draft is only a speculative probe: rejection is expected and the verifier will
resample it. The current manager nevertheless calls ``accept_tokens`` directly,
and XGrammar logs every rejected probe as an engine ERROR before returning
False.

Inspect the already-filled grammar bitmask for post-boundary probes before
calling ``accept_tokens``. This neither changes nor probes the matcher. Only
allowed drafts are accepted temporarily so later rows in the same speculative
window still see the correct FSM state. Committed-token failures keep their
existing hard-error path.

Exit 0 = patched or already patched; exit 1 = the expected source anchor was
not found. An optional first argument overrides the vLLM source path for tests.
"""

from __future__ import annotations

import compileall
import os
import shutil
import sys
from pathlib import Path

MARKER = "turnkey: preflight post-reasoning speculative grammar probes via bitmask"
LEGACY_MARKER = "turnkey: preflight post-reasoning speculative grammar probes"
OLD = """                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )"""
LEGACY_NEW = f"""                    if advance_grammar and not grammar.is_terminated():
                        # {LEGACY_MARKER}. Drafts after a reasoning-end marker
                        # were sampled before their grammar masks existed, so a
                        # rejection is expected speculation, not an FSM error.
                        # Preflight without changing state; valid probes still
                        # advance temporarily so later bitmask rows are exact.
                        accepted = (
                            grammar.validate_tokens([token]) == [token]
                            if post_reasoning_end_in_window
                            else True
                        )
                        if accepted:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )"""
NEW = f"""                    if advance_grammar and not grammar.is_terminated():
                        # {MARKER}. Drafts after a reasoning-end marker
                        # were sampled before their grammar masks existed, so a
                        # rejection is expected speculation, not an FSM error.
                        # The row was just filled from the current FSM state;
                        # inspecting its packed bit avoids probing or mutating
                        # the matcher. Valid probes still advance temporarily
                        # so later bitmask rows are exact.
                        accepted = (
                            bool(
                                int(
                                    self._grammar_bitmask[
                                        cumulative_index, token >> 5
                                    ].item()
                                )
                                & (1 << (token & 31))
                            )
                            if post_reasoning_end_in_window
                            else True
                        )
                        if accepted:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )"""


def default_target() -> Path:
    import vllm

    return (
        Path(vllm.__file__).resolve().parent
        / "v1"
        / "structured_output"
        / "__init__.py"
    )


def patch_text(source: str) -> tuple[str, str]:
    if MARKER in source:
        return source, "already patched"
    if LEGACY_NEW in source:
        return source.replace(LEGACY_NEW, NEW, 1), "upgraded"
    if OLD not in source:
        return source, "anchor not found"
    return source.replace(OLD, NEW, 1), "patched"

def main(argv: list[str]) -> int:
    target = Path(argv[0]) if argv else default_target()
    source = target.read_text()
    patched, status = patch_text(source)
    if status == "anchor not found":
        print(
            "structured-output/spec-decode: source anchor not found; "
            "future runtime may need review"
        )
        return 1
    if status == "already patched":
        print("structured-output/spec-decode: already patched")
        return 0
    # Back up the original source first (atomic), then write the patch through
    # a temp file + os.replace so an interrupted write cannot leave a
    # half-patched module that breaks every subsequent vLLM boot.
    bak = target.with_suffix(target.suffix + ".orig")
    if not bak.exists():
        _bak_tmp = bak.with_suffix(bak.suffix + ".tmp")
        shutil.copy2(target, _bak_tmp)
        os.replace(_bak_tmp, bak)
    _patch_tmp = target.with_suffix(target.suffix + ".tmp")
    _patch_tmp.write_text(patched)
    os.replace(_patch_tmp, target)
    if not compileall.compile_file(str(target), quiet=2):
        os.replace(bak, target)
        print(
            "structured-output/spec-decode: patched module failed to compile; "
            "restored the original from " + str(bak),
            file=sys.stderr,
        )
        return 1
    print(f"structured-output/spec-decode: {status} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
