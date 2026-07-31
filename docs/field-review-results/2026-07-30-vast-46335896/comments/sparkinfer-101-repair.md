## Rental-machine validation — PASS (repaired test oracle)

This supersedes my earlier FAIL attribution at `b51825989dfc`: the CUDA output
was correct. The failure was in the newly added torture test.

- Tested PR head: `0959fe807d366628380f41933244e3ccce0a8ae0`
- Feature parent: `b51825989dfcc4258ade1f4544e808c22be82be3`
- Tested base: `b38a60ecd5cb026f05ec27fc96433c9eb5ed326e` (`master`)
- Composed stack: none
- Host: Vast.ai instance 46335896, Ubuntu 24.04 / Linux 7.0, 2x EPYC 9455, 1.5 TiB RAM
- GPU/topology: 4x RTX PRO 6000 Blackwell 96 GB; tests used physical GPUs 0/1, both NODE/NUMA0 with P2P
- Driver/CUDA/PyTorch: 610.43.02 / CUDA UMD 13.3 / torch 2.12.0+cu132
- JIT cache: PR/reviewer-specific cold roots; no appliance or sibling cache reuse

### Root cause and correction

The captured graph contains 17 collectives. A complete replay therefore writes
both staging slabs; because each layer's marker changes, the prior
`exactly one changed slab` assertion is impossible even for correct parity.
The test also captured on an explicit stream, launched `graph.replay()` on the
default stream, then synchronized/probed the capture stream. A diagnostic
sample observed `(25, 171)` while the completed replay's last markers were
`(40, 41)`: a partial replay, not corrupt final output.

The corrected oracle:

1. enqueues fill and replay on the same explicit stream and synchronizes it;
2. requires the slabs to contain the penultimate/final layer markers;
3. requires the slot holding the final marker to alternate across odd-length
   replays.

No runtime CUDA source changed.

### Commands and results

- Ruff lint/format, `py_compile`, and `git diff --check` — PASS.
- Exact #101 corrected warm torture — **1 passed in 5.19 s**.
- Independent review of the identical #101/#103 patch — APPROVE, no blocking findings.
- Independent cold Vast run on GPUs 0/1 with **1,025 graph replays** — **1 passed in 48.23 s**.
- Existing exact-head selector, extension compile/import, and two-shot gates remain PASS.

No Xid, hang, stale-channel adoption, cross-rank parity disagreement, or
capture-time warm-cache compilation was observed.

Artifacts:
[complete report](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/README.md),
[#101 corrected run](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/artifacts/sparkinfer-replay-fix-agent-pr101-corrected-v2.log),
and
[independent 1,025-replay run](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-pr-validation/docs/field-review-results/2026-07-30-vast-46335896/artifacts/sparkinfer-replay-peer-review-pr103-1025-cold.log)
(SHA-256 `9e04448b22a54ff992707a8339e68508d0eb1417b8610f78a8ac3e29fde0567d`).

Working tree was clean; the only new commit is the test correction above.
