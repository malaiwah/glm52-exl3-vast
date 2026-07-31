# Field-review PR validation on Vast

Date: 2026-07-30/31 UTC

Provider instance: Vast.ai `46335896`

Test plan: `FIELD-REVIEW-PR-TEST-PLAN.md`, generated 2026-07-30

## Final closeout

This document preserves the first-pass attribution and its negative controls;
the authoritative final result is the corrected r14 successor described in
[the counter-validation guide](COUNTER-VALIDATION.md) and
[repair-campaign section M](UPSTREAM-REPAIR-CAMPAIGN.md#M-exact-repaired-successors-pass-full-model-workload-requalification).
That exact successor exposed **251,392 logical KV tokens**
(**1.87 GiB/rank**), passed the 13/13 API suite twice, retrieved 5/5 needles at
32K and 20/20 across cold 65K/126K probes, completed two bounded
C1/C2/C4/C8 matrices without a request failure or preemption, and had a clean
post-prime log window. Final scorecards were **96/100 GSM8K** and **44/50
GPQA**; all 44 GPQA normal stops were correct and its only six misses exhausted
the 32K reasoning ceiling. TERM released the wrapper within 2 seconds and all
workers/GPU allocations within 19 seconds. Release appliance
`ghcr.io/malaiwah/glm52-exl3-vast:fa52eda06ab516cd7e1a6628d915d2fd2478478f@sha256:1152b6e23604cd158017964c5ef14d6290779f7d1e67ced2ba4a39c8ec83a5c7`;
immutable evidence
[`5c76a253`](https://github.com/malaiwah/glm52-exl3-vast/tree/5c76a2536e7fc9a5f1cb6bf182531889f5385e65/docs/field-review-results/2026-07-30-vast-46335896/artifacts).

Every `FAIL`, `BLOCKED`, “not releasable”, and “not run” statement below is a
point-in-time result for the immutable first-pass heads. Those statements are
retained as failure attribution and must not be read as the final successor's
current release status.

## Repair-campaign update

The outcome below is the immutable first-pass attribution. Work continued on
the retained host after that checkpoint.

SparkInfer #101/#103's reported one-shot failure was a false negative in the
new torture test, not incorrect CUDA output:

- the 17-collective graph overwrites both staging slabs, so the old
  `exactly one changed slab` assertion cannot hold when every layer marker
  changes;
- replay was launched on the default stream while the test synchronized and
  sampled its separate capture stream, producing a real partial observation
  `(25, 171)` instead of final markers `(40, 41)`.

Repaired heads #101 `0959fe807d366628380f41933244e3ccce0a8ae0` and #103
`8fd90a4f687895e705bdf52fcff84ef653a5abf0` enqueue fill/replay/probe on the
same stream, require the final two layer markers in the two slabs, and require
the final marker to alternate slots. #103 passed cold/warm/final at
48.59/5.07/5.06 seconds; an independent reviewer then passed 1,025 graph
replays cold in 48.23 seconds with a fresh cache. No runtime CUDA source was
changed for this correction.

Evidence:
[#101 corrected](artifacts/sparkinfer-replay-fix-agent-pr101-corrected-v2.log),
[#103 cold](artifacts/sparkinfer-replay-fix-agent-pr103-corrected.log),
[#103 warm](artifacts/sparkinfer-replay-fix-agent-pr103-corrected-warm.log),
[#103 final warm](artifacts/sparkinfer-replay-fix-agent-pr103-final-warm.log),
and
[independent 1,025-replay review](artifacts/sparkinfer-replay-peer-review-pr103-1025-cold.log).
The continuing repair ledger is
[`UPSTREAM-REPAIR-CAMPAIGN.md`](UPSTREAM-REPAIR-CAMPAIGN.md).

The retained host has since completed the full-model vLLM #210 capacity
sweep. At matched TP4/DCP4, MTP0, 3,072 scheduler tokens and GMU 0.90,
reducing the EXL3 prefill arena from 3,072 to 1,024 rows returned
480.1 MiB/GPU and increased KV capacity from 264,960 to 328,960 tokens
(+24.15%). Median 3,072-token prefill moved from 1,845.90 to
1,807.58 tok/s (-2.08%), while C1 TPOT stayed effectively flat
(29.334 versus 29.429 ms). Capacity 512 also passed and reached 344,832 KV
tokens, but cost about 5.1% prefill, making 1,024 the balanced candidate.

The W4A16 #100/#102 composition blocker is also repaired. The exact pre-fix
lineage fails all four M=1/M=3 × ReLU2/SiLU poisoned same-binding replay
cases with NaNs; the narrow inactive-lane mask passes them cold and warm.
The repaired #100 union passed 116/116, and the exact #100 + #102 integration
passed 112 with 8 intentional skips cold and warm. Full evidence and exact
integration SHAs are in the repair ledger.

## Historical first-pass outcome

The focused vLLM changes pass. The individual W4A16 sizing/capture changes
pass, and the claimed planner-size reduction is reproduced exactly. The
individual LMCache changes pass, including a real source-built CUDA
STORE/RETRIEVE round trip.

At this first-pass checkpoint, the complete candidate was **not releasable**:

1. Current SparkInfer PRs
   [#101](https://github.com/local-inference-lab/sparkinfer/pull/101) and
   [#103](https://github.com/local-inference-lab/sparkinfer/pull/103) fail
   their one-shot graph scratch-reuse torture test on exact current heads.
2. Combining SparkInfer #100 and the fixed #102 exposes non-finite W4A16
   output after scratch reuse. The failure is order-dependent and is hidden
   when the target test runs alone.
3. LMCache #18 + #19 pass together, but adding #20 produces a source conflict
   in `lmcache/v1/multiprocess/futures.py`.

Per the plan's correctness-before-performance and no-conflict-resolution
rules, no combined image was built and no full GLM-5.2 PP/TG/KV or capacity
sweep was run at this checkpoint. The first-pass report does not claim
full-model memory or throughput gains that were not measured there; the later
measured successor result is recorded in the final closeout above.

## Host and toolchain

| Item | Observed |
|---|---|
| OS / kernel | Ubuntu 24.04.4 LTS / Linux 7.0.0-28-generic |
| CPU / RAM | 2x AMD EPYC 9455, 192 logical CPUs / 1.5 TiB RAM |
| GPU | 4x NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB each |
| GPU topology | All four GPUs are `NODE`, NUMA node 0; P2P read/write advertised and all 12 directed 4 MiB peer copies passed |
| Driver / CUDA UMD | 610.43.02 / 13.3 |
| Python / PyTorch CUDA | 3.12.3 / 2.12.0+cu132, CUDA 13.2 |
| pytest / compiler | pytest 8.4.1 / GCC and G++ 13.3.0 |
| Container reference | Retained turnkey appliance environment; no Docker socket or nested container engine was available, so the r12 image could not be pulled independently |

Evidence:
[host baseline](artifacts/host-baseline.txt) and
[real P2P probe](artifacts/cuda-peer-probe.log).
The production supervisor was stopped before GPU tests. No other GPU process
was present, and no Xid was observed during the test campaign.

## Historical first-pass exact-head result matrix

The rows below freeze the initial exact-head checkpoint. Later repaired heads,
compositions, and full-model measurements are recorded in the repair ledger.

| Project / PR | Exact tested head | Exact tested base | Individual result | Composed/full result |
|---|---|---|---|---|
| vLLM #209 | — | — | **NOT TESTED**, closed and superseded | — |
| vLLM #211 | merged `30038602b71395f481ef4a6edfe4fcf8551d9c15` (original PR head `7bf5ddd98ffc1d503ad6017e9a0e8edd4bc9bf0d`) | `dev/gilded-gnosis` | **PASS** | PASS with #210; live full-model gate blocked by downstream source failures |
| vLLM #210 | `47dc47d87f428e195f66cd8e7beffd24946a415b` | `dev/gilded-gnosis` | **PASS focused** | PASS with #211 at integration `b135a9b79377386ee805c22dee950e985f07ed28`; full-model capacity sweep not run |
| SparkInfer #100 | `fa7a6ad4e8bbcf241661009697a879630cddb554` | `06032ce74afba49a683d01339ed2c971568cec3f` | **PASS focused** | **FAIL** with #102 |
| SparkInfer #101 | `b51825989dfcc4258ade1f4544e808c22be82be3` | `b38a60ecd5cb026f05ec27fc96433c9eb5ed326e` | **FAIL** | Not composed after exact-head failure |
| SparkInfer #102 | `2bed880a7e9edbd9f2d976ba1a8ee88c9ba6e338` | `06032ce74afba49a683d01339ed2c971568cec3f` | **PASS** after fixes `2c2e220` and `2bed880` | **FAIL** with #100 |
| SparkInfer #103 | `5c4f8b01962c42d3aab8ac36b1e663974ad2537b` | `6a2babc531b57c2661c508a87f1c1b1d6742dc1c` | **FAIL** | Not recomposed after exact-head failure |
| LMCache #18 | `85abae7d2dab3585be9ad920dc634ea37c905333` | `7b7583aef55e98ba05a6c199e71601d78552e794` | **PASS** | PASS with #19; conflict when adding #20 |
| LMCache #19 | `1d4396c70352764d1fa5c85ef2f27dbe948d6481` | `31c4175d2134518e5b43fe8f4a7d072df6043a13` | **PASS** | PASS with #18; conflict when adding #20 |
| LMCache #20 | `9374b2970987a8e6f7027658c802b7835e007b7a` | `7b7583aef55e98ba05a6c199e71601d78552e794` | **PASS** | **FAIL composition** due source conflict |

The original #101/#103 handoff heads moved while testing. Results above are
for the final heads fetched immediately before the final exact-head runs.

## vLLM

### #211 — probabilistic draft sampling

Commands:

```text
python -m pytest -q tests/v1/worker/test_gpu_autoregressive_speculator.py
ruff check <four changed speculator/test files>
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/v1/worker/test_gpu_gumbel_sample.py \
  tests/v1/worker/test_gpu_rejection_sampler_dist.py
```

Results:

- CPU/static: 4 passed, Ruff clean.
- GPU cold: 11 passed in 19.75 s; warm: 11 passed in 16.84 s.
- The requested uncommitted DSpark spy observed exactly
  `sample_pos[:, i] - 2` at the shared sampler boundary.

Logs:
[CPU/static](artifacts/vllm-211-cpu.log),
[cold GPU](artifacts/vllm-211-gpu-cold.log),
[warm GPU](artifacts/vllm-211-gpu-warm.log), and
[DSpark spy](artifacts/vllm-211-dspark-spy.log).

### #210 — EXL3 prefill capacity

Command:

```text
python -m pytest -q tests/quantization/test_exl3_prefill_plan.py
ruff check vllm/envs.py \
  vllm/model_executor/layers/quantization/exl3.py \
  tests/quantization/test_exl3_prefill_plan.py
```

Results:

- Isolated: 12 passed, Ruff clean.
- Current `dev/gilded-gnosis` + #210 integration:
  16 combined CPU tests passed, Ruff clean; #210 repeated at 12 passed;
  #211 GPU tests passed 11 cold and 11 warm.
- Full-model `VLLM_EXL3_PREFILL_CAPACITY` sweep: **not run** because the
  downstream SparkInfer/LMCache composition gates failed first.

Logs:
[isolated](artifacts/vllm-210-cpu.log),
[combined CPU](artifacts/vllm-compose-211-210-cpu.log),
[#210 repeat](artifacts/vllm-compose-210-cpu-repeat.log),
[combined cold GPU](artifacts/vllm-compose-211-gpu-cold.log), and
[combined warm GPU](artifacts/vllm-compose-211-gpu-warm.log).

## SparkInfer

Every CUDA/C++ target used a distinct cold extension cache followed by its own
warm-cache run.

### #100 — W4A16 reachable scratch planning

Results:

- CPU/static: 32 passed, 80 skipped, Ruff clean.
- GPU: 112 passed cold in 68.87 s; 112 passed warm in 7.61 s.
- Exact sizing reproduction:

| Planner shape | Measured |
|---|---:|
| Auto block | 31,491,908 bytes / 30.03302 MiB |
| Explicit block 8 | 31,491,908 bytes / 30.03302 MiB |
| Explicit block 64 | 99,553,972 bytes / 94.94207 MiB |
| Unreachable reserve removed | 68,062,064 bytes / **64.90904 MiB per rank** |
| Trellis 3072 | 1,106,622,184 bytes / 1055.35715 MiB |

The source-level planner claim is verified. It is not a measurement of
additional full-model KV capacity.

Logs:
[CPU/static](artifacts/sparkinfer-100-cpu.log),
[cold GPU](artifacts/sparkinfer-100-gpu-cold.log),
[warm GPU](artifacts/sparkinfer-100-gpu-warm.log), and
[sizing](artifacts/sparkinfer-100-sizing.log).

### #102 — capture resolution

The handoff head `5bea8f3c1c04…` exposed an inactive-helper `relu2` error on
GPU. The test campaign fixed it in the PR branch:

- `2c2e22010073` — avoid the inactive gated helper for `relu2`;
- `2bed880a7e9e` — combine identical W4A8 branches and satisfy Ruff.

The exact final head then produced:

- CPU/static: 29 passed, 87 skipped, Ruff clean.
- GPU cold: 108 passed, 8 skipped in 104.35 s.
- GPU warm: 108 passed, 8 skipped in 105.40 s.

Logs:
[original failure](artifacts/sparkinfer-102-relutest-warm.log),
[final CPU/static](artifacts/sparkinfer-102-v3-cpu.log),
[final cold GPU](artifacts/sparkinfer-102-v3-gpu-cold.log), and
[final warm GPU](artifacts/sparkinfer-102-v3-gpu-warm.log).

### #101 and #103 — PCIe selector/channel replay

Current #101:

- CPU selector: 1 passed, 2 skipped.
- All three source extensions imported: cold 107 s, warm 2 s.
- Two-shot correctness passed cold/warm. Warm microbenchmark:
  SparkInfer RS/AG 342.9/337.6 us versus NCCL BF16 853.4/805.2 us.
- **One-shot torture failed** after the cold build at
  `_run_graph_scratch_reuse`: not every replay changed exactly one slot.

Current #103:

- CPU: 12 passed, 2 skipped; Python Ruff clean.
- DCP2: 1 passed cold and warm.
- DCP4: 1 passed cold and warm.
- Fused one-shot RMSNorm: 1 passed.
- Two-shot correctness passed cold/warm. Warm microbenchmark:
  SparkInfer RS/AG 344.6/338.6 us versus NCCL BF16 854.4/806.9 us.
- **One-shot torture failed cold and warm at the same scratch-reuse
  assertion**. The warm reproduction took 6 s, ruling out JIT timing as the
  sole cause.

Primary logs:

- #101:
  [CPU](artifacts/sparkinfer-101-current-cpu.log),
  [extension cold](artifacts/sparkinfer-101-current-ext-cold.log),
  [extension warm](artifacts/sparkinfer-101-current-ext-warm.log),
  [one-shot failure](artifacts/sparkinfer-101-current-oneshot-torture-cold.log),
  [two-shot cold](artifacts/sparkinfer-101-current-twoshot-cold.log), and
  [two-shot warm](artifacts/sparkinfer-101-current-twoshot-warm.log).
- #103:
  [CPU](artifacts/sparkinfer-103-current-cpu.log),
  [DCP2 cold](artifacts/sparkinfer-103-current-dcp2-cold.log),
  [DCP2 warm](artifacts/sparkinfer-103-current-dcp2-warm.log),
  [DCP4 cold](artifacts/sparkinfer-103-current-dcp4-cold.log),
  [DCP4 warm](artifacts/sparkinfer-103-current-dcp4-warm.log),
  [fused RMSNorm](artifacts/sparkinfer-103-current-oneshot-rms-cold.log),
  [one-shot cold failure](artifacts/sparkinfer-103-current-oneshot-torture-cold.log),
  [one-shot warm failure](artifacts/sparkinfer-103-current-oneshot-torture-warm2.log),
  [two-shot cold](artifacts/sparkinfer-103-current-twoshot-cold.log), and
  [two-shot warm](artifacts/sparkinfer-103-current-twoshot-warm.log).

### #100 + #102 composition

The exact #100 + fixed #102 source merge was clean. The full #102 GPU suite
then failed 2 of 108 executing cases (`relu2` and `silu`) with NaN oracle
metrics; both targets pass when run alone.

The minimum reproducer runs a compact-tail test before the two odd-shape
tests. Diagnostics established:

- the failure is deterministic with a cold or warm cache;
- zeroing `intermediate_cache2` before reuse makes the sequence pass;
- poisoning it makes replay fail;
- reserving reachable blocks through 32 masks the failure;
- Compute Sanitizer reports no out-of-bounds access because the bad read
  stays inside the shared arena.

This is a scratch initialization/lifetime dependency exposed by the smaller
reachable arena, not a Git conflict. It must be corrected before the memory
reduction can be promoted.

Logs:
[#100 + #102 suite](artifacts/sparkinfer-compose-100-102-e2e-ordered.log),
[ordered reproducer](artifacts/sparkinfer-compose-sequence-bisect-a.log),
[zero scratch](artifacts/sparkinfer-pr100-diag-zero-all.log),
[poison replay](artifacts/sparkinfer-pr100-diag-poison-replay.log), and
[Compute Sanitizer](artifacts/sparkinfer-pr100-diag-memcheck.log).

An earlier full Spark integration (`113a519a…`) used the then-current #103
head `22c11df…`; its CPU union passed 74 with 167 skips, diff-only Python Ruff
was clean, and #100's suite passed 112. The same #102 suite failed 2. It was
not rebuilt after #101/#103 moved because both current heads independently
fail their own replay gate.

## LMCache

### Individual heads

- #18 CPU: 59 passed, 12 skipped; Ruff clean.
- #18 GPU: 71 passed cold and 71 passed warm.
- #18 real source-built daemon/worker CUDA test:
  3/3 checksum matches, 512 tokens / 2 chunks each; STORE mean 2.27 ms,
  RETRIEVE mean 0.81 ms; unregister and daemon stop succeeded.
  The harness emitted PyTorch's warning that the producer terminated before
  all shared CUDA tensors were released, so lifetime cleanup deserves a
  follow-up even though the protocol checks passed.
- #19: 29 passed; Ruff clean.
- #20: 66 passed, 13 skipped; Ruff clean.

Logs:
[#18 CPU](artifacts/lmcache-18-cpu.log),
[#18 cold GPU](artifacts/lmcache-18-gpu-cold.log),
[#18 warm GPU](artifacts/lmcache-18-gpu-warm.log),
[#18 real round trip](artifacts/lmcache-18-real-gpu-roundtrip.log),
[#18 server](artifacts/lmcache-18-real-server.log),
[#19](artifacts/lmcache-19-cpu.log), and
[#20](artifacts/lmcache-20-cpu.log).

### Composition

#18 + #19 passed 88 CPU tests with 12 skips and 100 GPU tests.

Applying #20's second commit (`5aba0ddd…`) onto the composed release base
stopped at a conflict in `lmcache/v1/multiprocess/futures.py`. #18 adds
exporter-owned `_retained_resources`; #20 independently adds `_on_timeout`,
idempotent completion, and `_expire`. The conflict was preserved and was not
hand-resolved, as required by the plan.

Logs:
[#18 + #19 CPU](artifacts/lmcache-compose-18-19-cpu.log),
[#18 + #19 GPU](artifacts/lmcache-compose-18-19-gpu.log), and
[#20 conflict](artifacts/lmcache-compose-conflict.log).

## Historical first-pass full-candidate decision

| Gate | Status | Reason |
|---|---|---|
| vLLM source composition | PASS | #210 and #211 focused tests pass together |
| SparkInfer source composition | FAIL | W4A16 NaNs after scratch reuse; current #101/#103 replay assertions fail |
| LMCache source composition | FAIL | unresolved conflict adding #20 |
| Candidate image build | BLOCKED | prior source gates failed |
| GLM-5.2 smoke/PP/TG/MAL/KV | BLOCKED | no valid combined image |
| #210 capacity sweep | BLOCKED | no valid combined image |
| Combined correctness/needle tests | BLOCKED | no valid combined image |

At this checkpoint, no turnkey runtime defaults were changed and only this
evidence package was added. Promotion was correctly withheld pending:

1. a one-shot replay fix on current SparkInfer #101/#103;
2. a W4A16 scratch-lifetime fix that passes the ordered #100 + #102 suite;
3. an upstream reconciliation of LMCache #18 and #20's
   `MessagingFuture` lifecycle.

Those three first-pass blockers were subsequently repaired or reconciled and
the corrected exact successor completed full-model qualification. See the
final closeout above; this list remains only as the historical promotion gate.

The complete raw evidence set is in [`artifacts/`](artifacts/).

## Rental hand-back

The retained turnkey appliance was restored after testing. At
2026-07-31 03:10:57 UTC:

- `/health` returned HTTP 200;
- authenticated `/v1/models` returned `GLM-5.2`;
- an authenticated non-thinking chat request returned exactly
  `FIELD REVIEW RESTORE OK` with `finish_reason=stop`;
- all four vLLM workers were present;
- each GPU had about 514 MiB free after model initialization.

That was the first-pass hand-back; the retained instance was then reused for
the successor repair campaign. At final closeout on 2026-07-31, TERM reached
the isolated process group, the wrapper exited within 2 seconds, and all four
workers/GPU allocations released naturally within 19 seconds. The first
five-second snapshot and the clean final state are both retained in
[shutdown evidence](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-shutdown.txt).
After every copied artifact matched its remote SHA-256, Vast instance
`46335896` was destroyed and a subsequent inventory query returned no matching
instance. No rental remains from this campaign.
