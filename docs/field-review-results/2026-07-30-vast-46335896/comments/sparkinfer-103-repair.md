## Rental-machine validation — PASS (repaired test oracle)

This supersedes my earlier FAIL attribution at `5c4f8b01962c`: all runtime
CUDA outputs were correct; the failure was a cross-stream observation race and
an invalid changed-slab oracle in the new torture test.

- Tested PR head: `8fd90a4f687895e705bdf52fcff84ef653a5abf0`
- Feature parent: `5c4f8b01962c42d3aab8ac36b1e663974ad2537b`
- Tested base: `6a2babc531b57c2661c508a87f1c1b1d6742dc1c` (`master`)
- Composed stack: #103 includes the reconciled #101 stack
- Host: Vast.ai instance 46335896, Ubuntu 24.04 / Linux 7.0, 2x EPYC 9455, 1.5 TiB RAM
- GPU/topology: 4x RTX PRO 6000 Blackwell 96 GB, all NODE/NUMA0 with P2P
- Driver/CUDA/PyTorch: 610.43.02 / CUDA UMD 13.3 / torch 2.12.0+cu132
- JIT cache: distinct cold/warm and independent-review roots

### Root cause and correction

A 17-collective replay necessarily overwrites both staging slabs, so the old
`exactly one changed slab` assertion could not prove execution parity. The
test also launched replay on the current default stream but synchronized and
sampled its explicit capture stream. The preserved diagnostic observed
intermediate markers `(25, 171)` instead of completed final markers `(40, 41)`.

The test now launches fill/replay on the explicit stream, synchronizes that
stream, requires the final two layer markers in the slabs, and requires the
final marker to alternate slots. A capture-baked fixed selector fails this
oracle. No runtime source changed.

### Commands and results

- Corrected #103 torture, cold — **1 passed in 48.59 s**.
- Same cache warm — **1 passed in 5.07 s**; final warm confirmation **1 passed in 5.06 s**.
- Independent peer review — APPROVE, no blocking findings.
- Independent fresh-cache run, GPUs 0/1, **1,025 graph replays** — **1 passed in 48.23 s**.
- Existing exact-head CPU/nested-capture, DCP2, DCP4, fused RMSNorm, and two-shot cold/warm gates remain PASS.

No Xid, hang, stale-channel adoption, null pointer, parity disagreement, or
capture-time warm-cache compilation was observed.

Artifacts:
[complete report](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/README.md),
[#103 cold](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/artifacts/sparkinfer-replay-fix-agent-pr103-corrected.log),
[#103 warm](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/artifacts/sparkinfer-replay-fix-agent-pr103-corrected-warm.log),
and
[independent 1,025-replay run](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/artifacts/sparkinfer-replay-peer-review-pr103-1025-cold.log)
(SHA-256 `9e04448b22a54ff992707a8339e68508d0eb1417b8610f78a8ac3e29fde0567d`).

Working tree was clean; the only new commit is the test correction above.
