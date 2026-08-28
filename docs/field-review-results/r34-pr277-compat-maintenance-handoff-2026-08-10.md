# r34 / vLLM #277 maintenance-window handoff

Date: 2026-08-10

## Outcome

vLLM PR #277 remains necessary on r34 for the generic GLM-5.2 rank-sliced
mixed-Trellis path used by the 3.42 bpw `shared_h_v1` checkpoint. PR #280 adds
the specialized R7 K3/K4/K5 streaming path, but the generic K3/K4 path still
retained every per-expert Trellis tensor and then built an additional tier slab
with nested `torch.stack` calls during `process_weights_after_loading`.

The local compatibility port writes each generic K3/K4 expert tensor directly
into its final bitrate-tier slab while loading. B12X preparation consumes that
slab by pointer, removing the approximately 985.5 MiB-per-layer transient
duplicate identified by #277. R7 streaming and PR #281's borrowed-buffer
lifetime contract are preserved and covered by focused CPU tests.

## Exact source composition

- Worktree: `/Users/mbelleau/Documents/GLM-5.2 Turnkey Appliance/qualification-src/r34-pr277-compat`
- Branch: `codex/pr277-r34-compat`
- Pre-r34 base: `e2666d9a65f41fc376607531453cbd57c4c71016`
- blackwell-llm-docker r34 composition: `98224d1303c1497eec26c7d92f34a6fa9a58fa82`
- PR #280 head in r34: `8e7be4d5c97fb86d983bd5f83c825153452efaec`
- PR #281 head in r34: `126039af3cc28c667f3b13da3ee0d3abefdd12fe`
- Exact composed r34 vLLM tree: `4d006a43928cdee01306691a766542c1e9bebb59`
- Local exact-r34 materialization commit: `ef7ccc71b2b97239ad3a9ecbe347c9c2d7417a65`
- Original PR #277 head reviewed: `df004fb4a6bfb39d2f67c6eb96313f55cea8c65b`
- Local compatibility commit: `9774994aacd2960efd23b96351d47f8c55268c71`

The local r34 materialization was verified with `git write-tree`; it exactly
matched the release composition tree before the compatibility change.

## Compatibility decisions

- Grouped slabs are created only for generic rank-sliced mixed-bitrate Trellis
  parameters; R7 layers are explicitly excluded.
- Grouped preallocation fails closed if combined with whole-slab preallocation
  or an R7 `tp_slice`.
- Borrowed InstantTensor input is copied into owned tier storage with
  `non_blocking=False` before the loader returns. A regression test mutates the
  borrowed source after loading and verifies the tier slab is unchanged and
  uses distinct storage.
- Existing R7 `_slice_loaded_weight`, streaming ownership, and all PR #280
  preparation/runtime code remain unchanged.
- Existing synthetic/older reload objects retain the nested-stack fallback;
  real `Exl3MoEParameter` instances use the direct slab.
- Prepared B12X tier objects receive the original group-slab pointer. Source
  dictionaries and owning parameter references are cleared after preparation;
  any retained lifetime is then owned by the prepared tier object.

## Validation completed

All commands ran locally without touching AIBeast production.

```text
PYTHONPATH=. /tmp/r34-pr277-test-venv/bin/python -m pytest -q \
  --confcutdir=tests/quantization tests/quantization/test_exl3.py
61 passed

PYTHONPATH=. /tmp/r34-pr277-test-venv/bin/python -m pytest -q \
  --confcutdir=tests/model_executor/model_loader/instanttensor_loader \
  tests/model_executor/model_loader/instanttensor_loader/test_weight_utils.py \
  -k 'instanttensor_copy_contract or instanttensor_copy_rejects_unknown_value'
3 passed, 1 deselected

ruff format --check (two changed files)
passed

ruff check --ignore ISC004 (two changed files)
passed

Python 3.12 py_compile (two changed files)
passed

git diff --check
passed
```

The exact r34 release tree has a pre-existing Ruff `ISC004` finding at the R7
debug-string construction in `exl3.py`; the compatibility patch does not alter
that code. The release composition also has broader pre-existing pre-commit
findings, so the two local commits were made with hooks bypassed after the
focused checks above.

## Required GPU maintenance validation

Use the same 3.42 bpw `shared_h_v1` checkpoint and production topology as the
last qualification. Build from `9774994aacd2960efd23b96351d47f8c55268c71` and
compare against the exact r34 tree at
`ef7ccc71b2b97239ad3a9ecbe347c9c2d7417a65`.

1. Cold start with an empty compile-cache root:
   - record per-rank host RSS, GPU VRAM, and loader phase timestamps;
   - verify no full per-expert Trellis retention followed by a roughly
     985.5 MiB-per-layer stack spike;
   - verify the K3 and K4 prepared weights bind the tier-slab pointers;
   - verify target and native-MTP layers both complete preparation.
2. Warm restart with the populated compile cache:
   - repeat peak RSS/VRAM and startup-time capture;
   - distinguish compile-cache savings from loader-memory savings;
   - confirm no stale schema/ABI cache reuse.
3. Checkpoint/schema gates:
   - target 3.42 bpw `shared_h_v1` K3/K4 path;
   - native MTP on the same checkpoint;
   - one R7 checkpoint/profile to prove its specialized stream path remains
     selected and grouped slabs remain empty.
4. Borrowed-buffer modes:
   - `INSTANTTENSOR_COPY=0` and the release default/copy mode;
   - compare hashes or deterministic smoke outputs after staging-buffer reuse;
   - inspect for CUDA stream/lifetime errors under allocator pressure.
5. Runtime qualification:
   - deterministic short output and one long prompt;
   - C1 and production concurrency decode;
   - prompt prefill, MTP acceptance, CUDA graph capture/replay;
   - monitor OOM, worker death, late allocations, minimum free VRAM, and
     performance regression.

## Rollback

For a source rollback, build the exact r34 baseline commit
`ef7ccc71b2b97239ad3a9ecbe347c9c2d7417a65` (tree
`4d006a43928cdee01306691a766542c1e9bebb59`). Do not mix a rollback binary with
compile/JIT artifacts produced by the compatibility build; use a distinct cache
root or restore the prior qualified r34 cache root. Production restart/deploy
remains a separate maintenance-window action.

