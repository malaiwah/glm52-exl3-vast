# Upstream repair campaign

Started: 2026-07-30/31 UTC

This is the continuation ledger for turning the field-review findings into a
fully composed, production-tested appliance. Keep exact heads, test evidence,
public comments, and remaining work here so compaction cannot erase the
campaign state.

## Success contract

The campaign is complete only when:

1. all nine PRs either pass their exact-head gates or are superseded by tested
   fixes;
2. SparkInfer #100/#102 no longer produces ordered W4A16 scratch-reuse NaNs;
3. SparkInfer #101/#103 advances one-shot replay staging correctly cold/warm;
4. LMCache #18/#19/#20 composes without a future-lifecycle conflict and passes
   real CUDA STORE/RETRIEVE plus outage/recovery;
5. the current vLLM #210/#211 stack passes focused and live model gates;
6. a fresh combined turnkey candidate boots the full GLM-5.2 3.25 bpw profile,
   passes health/correctness/needle/degeneration tests, and has measured
   memory, PP, TG, MAL, concurrency, and KV deltas against the same-host
   baseline;
7. the best verified stack is left serving on the retained Vast instance;
8. upstream PRs/issues contain exact-SHA evidence and any new defects have
   their own tracked issue/PR;
9. after source work is exhausted, GSM8K, GPQA Diamond, and other appropriate
   scorecard workloads run against the final endpoint.

## Frozen upstream state

Snapshot: 2026-07-31 03:16 UTC. All nine PRs had zero unresolved review
threads at this snapshot.

| PR | State | Head |
|---|---|---|
| vLLM #210 | open | `47dc47d87f428e195f66cd8e7beffd24946a415b` |
| vLLM #211 | merged | `7bf5ddd98ffc1d503ad6017e9a0e8edd4bc9bf0d`, merge `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| SparkInfer #100 | open | `fa7a6ad4e8bbcf241661009697a879630cddb554` |
| SparkInfer #101 | open | `b51825989dfcc4258ade1f4544e808c22be82be3` |
| SparkInfer #102 | open | `2bed880a7e9edbd9f2d976ba1a8ee88c9ba6e338` |
| SparkInfer #103 | open | `5c4f8b01962c42d3aab8ac36b1e663974ad2537b` |
| LMCache #18 | open | `85abae7d2dab3585be9ad920dc634ea37c905333` |
| LMCache #19 | open | `1d4396c70352764d1fa5c85ef2f27dbe948d6481` |
| LMCache #20 | open | `9374b2970987a8e6f7027658c802b7835e007b7a` |

## Known-good evidence before repair

- vLLM #210/#211 focused and source-composed tests pass.
- SparkInfer #100 verifies a planner reduction of 64.90904 MiB/rank.
- SparkInfer #102 passes independently after `2c2e220` and `2bed880`.
- SparkInfer #101/#103 DCP2, DCP4, fused RMSNorm, and two-shot paths pass;
  current heads fail one-shot graph scratch reuse.
- LMCache #18/#19/#20 pass independently; #18 + #19 passes together and a
  real #18 CUDA round trip gives 3/3 checksum matches.
- Full raw evidence and the initial report are in this directory.

## Parallel repair tracks

### A. SparkInfer replay

Owner: `spark_replay_fix` subagent.

Target: fix current #101/#103 one-shot graph scratch-reuse assertion without
regressing DCP2/DCP4/two-shot behavior. Main agent will peer-review before any
push and run cold/warm Vast tests.

Checkpoint (2026-07-31 03:35 UTC): the CUDA output was correct; the failing
test launched replay on the default stream but synchronized/probed the
non-default capture stream, and its `exactly one changed slab` oracle was
invalid for a 17-collective graph that overwrites both slabs. A test-only
correction now launches/fills/probes on the same stream and asserts that the
penultimate/final markers occupy the two slabs while the final marker
alternates. Local candidate commits:

- #101 `0959fe807d366628380f41933244e3ccce0a8ae0`;
- #103 `8fd90a4f687895e705bdf52fcff84ef653a5abf0`.

The corrected #103 test passed cold in 48.59 s and warm three times at about
5.1 s. Independent review approved the correction and repeated 1,025 graph
replays from a fresh cache in 48.23 s. Both repaired branches were pushed and
the existing PR/issue comments were updated with exact repaired heads:

- #101 `0959fe807d366628380f41933244e3ccce0a8ae0`;
- #103 `8fd90a4f687895e705bdf52fcff84ef653a5abf0`.

A subsequent launch-safety audit found that the selector's grid-wide atomic
rendezvous is safe only while every participating CTA is simultaneously
resident. The existing one-shot maximum of eight and two-shot maximum of 64
fit this full 188-SM Blackwell host, but were not generic/MIG-safe. Dedicated
repair heads now query
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the exact kernel
specialization, combine it with the visible/MIG SM count using overflow-safe
arithmetic, cache the result per kernel/threads/device, and clamp grid-stride
one-shot, two-shot, and fused launches without dropping row work. They also
package the new header required by JIT-installed wheels:

- #101 repair `6e947cc2528005cb744a6371959344dd9ff80d5b`;
- #103 repair `9db41aa0e2140a5a8563852b742a54a274d262a4`.

The #101 exact head has passed a fresh-cache Vast gate on all four GPUs:

- resident-grid arithmetic/source guards: 4/4;
- one-shot eager, 256 graph replays, and multistream torture: 1/1 in 49.10 s;
- fused add+RMSNorm eager/graph: 1/1 in 5.83 s;
- DCP A2A eager/graph: 1/1 in 39.95 s;
- two-shot reduce-scatter/all-gather eager, alternating replay, and benchmark:
  correct at four ranks;
- zero GPU processes before and after.

[Exact #101 residency GPU gate](artifacts/sparkinfer-replay-residency-101-6e947cc-gpu-cold.log).

The exact #103 head then passed the same four-rank GPU gates from its own fresh
extension cache: one-shot torture in 49.06 s, fused add+RMSNorm in 5.01 s,
DCP A2A in 39.16 s, and correct two-shot eager/alternating graph results. Its
static rerun passed 16/16 plus Ruff and clean-tree checks. The first static
command incorrectly named `test_pcie_two_slot_selector.py`, which does not
exist on this stacked lineage; because that shell intentionally continued to
the independent GPU commands, the error is retained in the complete log and
not counted as a source failure. The corrected static command has its own
artifact.

Both exact lineages additionally passed a warm 1,025-graph-replay stress
cycle on four GPUs (6.86 s for #101, 6.73 s for #103), with 64 eager and 64
multistream iterations and zero residual GPU processes.

The host-side occupancy lookup was then exercised from a completely cold
extension cache with the first collective launch occurring inside CUDA graph
capture. Both repaired lineages passed at two ranks, including eight correct
replays. The #101 log also records an invalid inherited `LD_PRELOAD` path;
the loader ignored it and the test used the bundled NCCL. #103 repeated the
gate with the image's real `/opt/libnccl.so.2.30.4`, so the capture result
does not depend on that harness mistake.

This does **not** yet close launch-safety review. Per-launch residency clamping
does not by itself prove progress if independent stream-owned channels launch
simultaneously on a constrained/MIG device: each launch can be individually
valid while their aggregate resident grid is not. The large-host multistream
gate has substantial occupancy slack and cannot refute that case. This
candidate remains withheld pending an aggregate-overlap safety design or a
proof that such channel launches cannot overlap.

An attempted follow-up converted every rendezvous path to an occupancy-bounded
cooperative launch:

- #101 `f85aff393377f952cb99929f10a51c6e9da68d31`;
- #103 `10d79f3616078d1839c4fff0d0e728aaa4cc9c0b`.

Both exact heads compiled from fresh JIT caches. Their first use inside CUDA
graph capture passed at two ranks. Their independent four-GPU gates also
passed: 15/16 static and lifecycle tests respectively, 1,025 one-shot graph
replays plus multistream stress, fused add+RMSNorm, DCP4, and two-shot
reduce-scatter/all-gather. No GPU process remained after either gate.

Those positive full-GPU smokes are retained but the cooperative candidates
are **rejected** after independent review. They leave two distributed
deadlocks:

1. each rank derives its block count from its local SM/occupancy capacity, so
   heterogeneous partitions can launch unequal grids while peer barriers are
   indexed by `blockIdx.x`;
2. cooperative admission is local to one device. Different GPUs can admit
   separate full channel grids in opposite order, leaving each grid waiting
   for its peer behind the other grid.

The new aggregate-overcommit test only modeled the arithmetic and checked
source strings; it did not exercise either distributed schedule. The reviewer
confirmed the cooperative argument ABI, exact-kernel occupancy specialization,
fused specialization, graph support, and header packaging, but returned
**REQUEST CHANGES** on these two P1 findings. The repair direction is now to
remove the all-CTA rendezvous: publish the channel's dynamic slab selection
from a one-CTA control node immediately before the worker in the same
stream/graph. This chronology must remain visible; neither cooperative head is
safe to publish.

The initial four-rank two-shot timing was correctly withheld from the
two-rank historical control. A matched world-size-2 A/B then excluded one
base compile/warmup and ran three repeats per exact head. Median
reduce-scatter was 334.5 us at base `0959fe8` versus 333.8 us at candidate
`6e947cc` (-0.21%); all-gather was 341.6 versus 337.2 us (-1.29%). These are
noise-sized improvements, so the repair is throughput-neutral on the
non-clamped full-host shape rather than a performance claim.

[Exact #103 residency GPU gate](artifacts/sparkinfer-replay-residency-103-9db41aa-gpu-cold.log)
and
[corrected static gate](artifacts/sparkinfer-replay-residency-103-9db41aa-static-corrected.log).
[Warm #101 stress](artifacts/sparkinfer-replay-residency-101-6e947cc-gpu-warm1025.log)
and
[warm #103 stress](artifacts/sparkinfer-replay-residency-103-9db41aa-gpu-warm1025.log).
[Matched world-size-2 A/B](artifacts/sparkinfer-replay-residency-101-twoshot-world2-ab.log).
[Cold-capture #101](artifacts/sparkinfer-replay-residency-101-cold-capture.log)
and
[cold-capture #103](artifacts/sparkinfer-replay-residency-103-cold-capture.log).
[Rejected cooperative #101 cold capture](artifacts/sparkinfer-coop-101-f85aff3-cold-capture.log),
[#101 four-GPU smoke](artifacts/sparkinfer-coop-101-f85aff3-gpu-cold.log),
[#103 cold capture](artifacts/sparkinfer-coop-103-10d79f3-cold-capture.log),
and
[#103 four-GPU smoke](artifacts/sparkinfer-coop-103-10d79f3-gpu-cold.log).

A subsequent control-node implementation removed the rejected cooperative
launch and put slab selection in a same-stream `<<<1,1>>>` device node. The
first frozen heads (`376298c` / `4505fc3`) exposed an honest cold-build failure:
the new worker used an undefined `ld_flag_acquire_gpu` helper. The corrected
runtime heads (`cd9c2625dbb65adf5a775379e5022a60dfec0b6f` /
`45029f6f101ce8b733d088d2694d3320836b924b`) add the matching GPU-scope
acquire load. Both then compiled all three PCIe extensions from isolated
caches and passed the 168-test communication union cold and warm.

CUDA 13.3 exposed two evidence-harness incompatibilities rather than kernel
failures: the modern binding returns edge-data as a fifth
`cudaGraphGetEdges` field, and legacy `cudaGraphKernelNodeGetParams` can
reject a driver-launched function that the generic `cudaGraphNodeGetParams`
can inspect. The version-compatible test/benchmark fixes are retained in
local heads `5c3e854a8d6f94af39ff0db79f29ff7c7d8889a2` (#101) and
`c32da63ea589f32a33e5cc06dabdc9fb87f7f820` (#103). Their focused GPU
evidence is:

- fused add+RMSNorm eager and captured graph: 2-rank and 4-rank pass;
- opposite-order two-channel stress: 128 eager iterations plus 1,025 graph
  replays at four ranks pass;
- 17-collective scratch reuse: 1,025 graph replays plus multistream stress at
  four ranks pass;
- first collective captured before any eager use: fresh-cache 2-rank and
  4-rank pass;
- two-shot correctness: 2-rank and 4-rank pass; at 2 ranks SparkInfer
  reduce-scatter/all-gather measured 343.5/342.7 us versus NCCL
  854.9/805.6 us, while at 4 ranks it measured 3691.3/3712.7 us versus NCCL
  1025.9/990.0 us. The four-rank result is a real topology/algorithm
  performance loss and is not generalized into a speed claim.

The plain 64-KiB BF16 one-shot control-node cost was measured with four
AB/BA pairs, four ranks, 200 warmups and 2,000 aligned samples per run. Every
run rebuilt into a distinct cache and recorded frozen source/binary hashes,
the direct control-worker graph edge, rank-aligned critical-path samples and
hardware telemetry. Eager was unchanged within noise
(`-0.144 us`, approximate 95% interval `[-1.392, +1.104]`). Captured graph
mean cost increased by `1.273 us`, approximate interval
`[+0.194, +2.351]`, from an aggregate 99.081 to 100.353 us. This is a narrow
net-algorithm comparison, not isolated launch overhead; the schema now
explicitly excludes registered/push one-shot, fused RMSNorm, both two-shot
operations and both DCP operations, and calls the post-preflight observation
`first_sample_us` rather than “cold”.

[Initial CUDA-edge binding failure](artifacts/sparkinfer-final-replay-103-45029f6-fused-rms-dcp2.log),
[generic-node fallback pass at two ranks](artifacts/sparkinfer-final-replay-103-18c1a8c-fused-rms-dcp2.log),
[four-rank fused pass](artifacts/sparkinfer-final-replay-103-18c1a8c-fused-rms-dcp4.log),
[1,025 opposite-order replays](artifacts/sparkinfer-final-replay-103-18c1a8c-opposite-order-1025-dcp4.log),
[1,025 scratch-reuse replays](artifacts/sparkinfer-final-replay-103-18c1a8c-torture-1025-dcp4.log),
[two-rank two-shot](artifacts/sparkinfer-final-replay-103-18c1a8c-twoshot-dcp2.log),
[four-rank two-shot](artifacts/sparkinfer-final-replay-103-18c1a8c-twoshot-dcp4.log),
[fresh first-use capture at two ranks](artifacts/sparkinfer-final-replay-103-18c1a8c-first-use-capture-dcp2.log),
[fresh first-use capture at four ranks](artifacts/sparkinfer-final-replay-103-18c1a8c-first-use-capture-dcp4.log),
and
[complete AB/BA JSON](artifacts/sparkinfer-final-replay-101-cd9c262-control-ab-r2.json).

These positive tests do **not** accept the current replay branches.
Independent strict review returned **REQUEST CHANGES** on defects not made
deterministic by the original plan:

1. #101 still capture-bakes DCP's slab. #103 changes that to per-block parity,
   but a large-small-large grid sequence leaves returning blocks one operation
   behind and can overwrite the smaller operation under rank skew.
2. DCP retains legacy best-effort teardown that discards ownership after
   suppressed unmap/free failures; rollback removes retryable ownership too
   early.
3. one-shot/two-shot setup can free an export after only a subset of peers
   opened it, and two-shot can leak imports opened before a later open fails.
4. GC can unmap imports while asynchronous kernels still dereference them.
5. process-group ranks are incorrectly sorted even though PyTorch returns
   group-rank order; nonmonotonic groups can misroute IPC handles/status.
6. fixed 36/64-block worker barriers still lack a proved residency contract
   on smaller or MIG partitions.

The replacement must use one channel-wide DCP control node, collective
two-phase setup and retryable coordinated teardown, safe GC retention/defer,
group-order preservation, deterministic skew/variable-grid tests, and either
prove/enforce residency or remove that assumption. That implementation and a
fresh independent review are in progress; no replay repair has been pushed.

### B. SparkInfer W4A16 scratch lifetime

Owner: `w4a16_scratch_fix` subagent.

Target: precise initialization/lifetime correction for the #100/#102 ordered
NaN, preserving the 64.91 MiB/rank planner saving and avoiding broad arena
zeroing unless measurement proves it harmless.

Checkpoint (2026-07-31 04:05 UTC): a deterministic same-binding replay now
reproduces stale-tail NaNs on exact #100 for both ReLU2 and SiLU small-M direct
FC2 paths; a narrow lane-mask candidate makes the same cases finite without
whole-arena zeroing. Exact #100 + #102 composition `3c758e6682e1430cf6fa6028fd8fb0e2d7d5528b`
passed 5/5 on Vast; pre-fix failed 2/2 with NaNs and post-fix cold/warm passed
2/2. The 64.90904 MiB/rank planner reduction is unchanged, and GLM's
intermediate size takes the byte-identical wide path.

Independent review requested changes before upstreaming because the committed
regression only exercised the `m>=2` consumer. Commit
`783707cd` parameterizes the same-binding replay over `m=[1,3]`, forces and
asserts the native direct path. Final stacked candidate `5178cb5c` adds only
lint-safe equivalent cleanup required when `micro.py` enters the PR diff. The
exact pre-fix lineage failed all four
M=1/M=3 × ReLU2/SiLU poisoned-replay cases with non-finite output. The narrow
lane mask then passed all four cold and warm. The repaired #100 lineage passed
116/116 focused tests at exact final head `5178cb5c`; exact #100 + #102
integration `25f6a8e3` passed 112 with 8 intentional skips cold in
106.12 seconds and warm in 5.12 seconds. Ruff and `git diff --check` are
clean. Compute Sanitizer then ran all four repaired replay cases with zero
memcheck errors. The 64.90904 MiB/rank planner reduction remains unchanged.

Evidence:
[pre-fix four-case failure](artifacts/sparkinfer-w4a16-m13-prefix-cold.log),
[post-fix cold](artifacts/sparkinfer-w4a16-m13-post-cold.log),
[post-fix warm](artifacts/sparkinfer-w4a16-m13-post-warm.log),
[final-head memcheck](artifacts/sparkinfer-w4a16-pr100-final-v2-memcheck.log),
[#100 final-head union](artifacts/sparkinfer-w4a16-pr100-final-v2-cold.log),
[#100 + #102 final cold](artifacts/sparkinfer-w4a16-pr100-102-final-v2-cold.log),
and
[#100 + #102 final warm](artifacts/sparkinfer-w4a16-pr100-102-final-v2-warm.log).

Checkpoint (2026-07-31 05:13 UTC): independent review approved the
consumer-side stale-tail mask with no blocking correctness, race, out-of-bounds
or production-performance finding. Review suggested extending the regression
to the narrow-range boundaries and proving that the actual custom operator
launched. Test-only commit `d4d101e8` therefore covers
`I=[32,128,224]`, `M=[1,3]` and ReLU2/SiLU, spies on the real
`w4a16_small_m_direct_launch` implementation, requires two launches, and
compares first use with poisoned replay bit-for-bit. The corresponding exact
#100 + #102 head is `3382e15b`.

That stronger cold gate found a **separate numerical defect** on both exact
lineages: all SiLU cases and all `I=128` cases pass, but ReLU2 misses the FP32
oracle at `I=32` (`cos=0.9592/0.9625` for M=1/3) and `I=224`
(`cos=0.9975/0.9927`). Both lineages reproduce the same four failures, so this
is neither a #102 interaction nor a JIT-cache artifact. The failed results are
preserved in
[pure #100 boundary evidence](artifacts/sparkinfer-w4a16-boundaries-pure-d4d101e-cold.log)
and
[#100 + #102 boundary evidence](artifacts/sparkinfer-w4a16-boundaries-composed-3382e15-cold.log).

Root cause: ModelOpt's E4M3 scale grid permutes output rows in 64-row tiles.
For ungated ReLU2, preparation padded a 32-row tail, permuted it, and truncated
the packed result back to the logical row count. That discarded 16 valid
tail-row scales; the micro reader then addressed those missing permuted
positions in following storage. SiLU happened to avoid the defect because its
gated W13 row count is `2I`, a multiple of 64 at both filed boundaries.

Final pure #100 repair head `691995de` retains the final 64-row-padded scale
tile only in the direct-micro representation and makes its reader/expert stride
match. The main packed ABI is unchanged. Exact #100 + #102 composition
`5ab78a09` contains the identical repair.

- Boundary cold: **12/12 passed** on both exact lineages.
- Boundary warm: **12/12 passed**.
- Compute Sanitizer: **12/12 passed, 0 errors**.
- Pure #100 affected union: **124/124 passed** cold and warm.
- #100 + #102 union: **120 passed, 8 intentional skips** cold and warm.
- Ruff and `git diff --check`: clean.

Evidence:
[pure boundary cold](artifacts/sparkinfer-w4a16-boundaries-pure-691995d-cold.log),
[composed boundary cold](artifacts/sparkinfer-w4a16-boundaries-composed-5ab78a0-cold.log),
[composed boundary warm](artifacts/sparkinfer-w4a16-boundaries-composed-5ab78a0-warm.log),
[Compute Sanitizer](artifacts/sparkinfer-w4a16-boundaries-pure-691995d-memcheck.log),
[pure union cold](artifacts/sparkinfer-w4a16-pr100-final-691995d-union-cold.log),
[pure union warm](artifacts/sparkinfer-w4a16-pr100-final-691995d-union-warm.log),
[composed union cold](artifacts/sparkinfer-w4a16-pr100-102-final-5ab78a0-union-cold.log),
and
[composed union warm](artifacts/sparkinfer-w4a16-pr100-102-final-5ab78a0-union-warm.log).

At GLM's `I=1856`, both row counts are already divisible by 64, so the scale
allocation and reader stride are unchanged. The final independent review is
**REQUEST CHANGES**, so neither repair head has been published upstream.
The review confirmed that the scale-padding repair and operator spy are sound,
but found two remaining generic non-64 gaps:

1. wide FC2 still derives validity from `ceil(n/64)`; at supported `I=352`,
   its final chunk has 12 logical lanes/plane while the guard admits 16,
   allowing stale scratch and cross-row W2/scale consumption;
2. narrow `I=32/224` masks scratch inputs exactly but still guards W2 and
   block-scale loads with coarse `w_valid`, leaving logically out-of-bounds
   reads hidden by zero activation and allocator padding.

The existing narrow numerical and Compute Sanitizer evidence therefore does
not discharge generic tensor-bound safety. A new repair must guard scratch,
W2, and scale reads with exact logical validity in both narrow and wide paths,
then reproduce/pass `I=144/352`, M=1/3, ReLU2/SiLU before re-review.

Review also found a separate generic-shape bug for ModelOpt intermediates
divisible by 16 but not 64 (for example `I=144`): padded-grid validity can
permit raw W/scale loads beyond logical rows. A separate repair/test track is
active; the filed GLM shape `I=1856` is not exposed.

Checkpoint (2026-07-31 07:54 UTC): the generic repair is now pinned at exact
head `9535f545a7538d4e5a95f800e7d6c91e808fccaa`. It extends the prior
`5ab78a0955e23ae395c33a1832c1893e206dbeee` repair in four ways:

1. E4M3 narrow and wide FC2 readers predicate packed W2 words and scale
   elements using exact logical indices rather than a coarse 64-row tile;
2. the M>1 planner only chooses FC1 chunks whose widths are integral
   16-value blocks, while the M=1 retile path backs down to a valid divisor;
3. the N=16 path cannot select a zero FC1-chunk count;
4. native E8M0 W13 uses the independently padded 64-row scale stride instead
   of the logical 224-row stride.

The investigation retained two useful negative controls. A first graph test
incorrectly passed the baseline because pytest's working directory shadowed
the requested `PYTHONPATH`; rerunning from outside the checkout with
`--import-mode=importlib` made the base fail 6/12 configuration cases and
abort under Compute Sanitizer with 278 invalid global reads. A first E8M0
repair padded the tensor but left the reader at stride 224; its ReLU2 cosine
fell to 0.40–0.58 until the reader stride was corrected. Neither false pass
nor intermediate failure is counted as candidate evidence.

The exact final head passed on physical GPU 2:

- Ruff, format and `git diff --check`;
- E4M3 non-64 planner, graph replay and bounds matrix: **40/40**;
- native E8M0 `I=224` matrix: **24 passed, 8 intentional ReLU2-limit skips**;
- E4M3 Compute Sanitizer: **8/8, 0 errors**;
- E8M0 Compute Sanitizer: **24 passed, 8 skips, 0 errors**.

A matched post-warmup A/B used one external harness, fixed 600 W power limit,
three alternating runs per source, 50 warmups and 500 CUDA-graph replays per
run. Outputs and all allocation/peak-memory fields were byte-identical for
GLM-aligned `I=1856`, `K=2688/6144`, M=1/3. Candidate latency deltas were
respectively -0.07%, -0.12%, -0.15%, and +0.09%: throughput-neutral within
noise, with no memory change on the already aligned production shape. The
repair claim is therefore generic correctness/memory safety, not a GLM speed
or memory gain.

Evidence:
[E4M3 exact-head matrix](artifacts/w4a16-9535f54-e4m3-gpu2.log),
[E4M3 memcheck](artifacts/w4a16-9535f54-e4m3-memcheck-gpu2.log),
[E8M0 exact-head matrix](artifacts/w4a16-9535f54-e8m0-i224-gpu2.log),
[E8M0 memcheck](artifacts/w4a16-9535f54-e8m0-memcheck-gpu2.log), and the
matched A/B JSON records
[base 1](artifacts/w4a16-ab-base-1.json) /
[candidate 1](artifacts/w4a16-ab-candidate-1.json),
[base 2](artifacts/w4a16-ab-base-2.json) /
[candidate 2](artifacts/w4a16-ab-candidate-2.json), and
[base 3](artifacts/w4a16-ab-base-3.json) /
[candidate 3](artifacts/w4a16-ab-candidate-3.json).

Independent review **rejected** `9535f545` after finding one last wide-tail
residue. The activation mask used padded scale-grid fullness rather than
logical row completeness. At `I=496`, for example, the scale geometry filled
all eight control blocks while FC1 wrote only 248 of 256 packed activation
slots. Exact W/scale predicates could therefore still evaluate
`0 * stale-NaN`. The same gap covered residue representatives
`I=464/480/496`; existing `I=352` happened to take the masked branch and did
not expose it.

Focused successor `aa2c4c6f38175f5bf0a4f7bb0866a32746dc8b15` makes both M=1
and M>=2 wide W4A16 paths key activation masking to logical
`n % 256 != 0`. Aligned W4A16 and non-W4A16 paths retain their old compiled
arm. The exact head passed Ruff/format, **52/52** committed contract,
poisoned-graph-replay and bounds cases, and **4/4** `I=496` Compute
Sanitizer cases with zero errors.

Because the first successor commit only parameterized `I=496`, an external
exact-source harness also exercised `I=464/480` and aligned `I=512` across
M=1/3, ReLU2/SiLU, contract, poisoned graph replay, and bounds: **36/36**.
The corresponding 12 bounds cases passed Compute Sanitizer with zero errors.
Those external controls prove the code path but do not replace committed
regressions; the complete residue/control matrix still needs to be added
before final re-review.

Evidence:
[successor matrix](artifacts/w4a16-aa2c4c6-wide-gpu2.log),
[`I=496` memcheck](artifacts/w4a16-aa2c4c6-i496-memcheck-gpu2.log),
[external residue/control matrix](artifacts/w4a16-aa2c4c6-external-residues-gpu2.log),
and
[external residue/control memcheck](artifacts/w4a16-aa2c4c6-external-residues-memcheck-gpu2.log).

Final test-only successor `cae3aad137bd739881833c064e1912177698156f`
commits all three missed residue representatives plus aligned `I=512`.
Independent diff review **approved** it. The exact composed head passed:

- contract + poisoned eager/16-replay graph + bounds: **88/88**;
- all 24 one-expert bounds cases under Compute Sanitizer: **24/24,
  0 errors**;
- complete #102 compile-cache + W4A16 GPU union: **220 passed, 16
  intentional skips** cold in 171.98 s and warm in 5.91 s.

The byte-identical source repair was replayed onto pure PR #100 lineage and
published as current head `c2f135c66032ac5e9d0778067dd19ae1910cff47`.
Its focused 88-case matrix and 24-case sanitizer matrix pass. Its broader
file reports 196 passed, 16 skips, and one inactive-`relu2` helper failure;
that is the already known defect fixed by PR #102, and the exact repaired
#100 + #102 composition above passes the complete union.

The final matched three-run A/B remains throughput- and memory-neutral on the
production `I=1856` shapes. Candidate latency deltas for K=2688/6144,
M=1/3 were +0.068%, -0.084%, +0.059%, and +0.039%; outputs, allocated,
reserved, and peak memory were identical. Thus the original
**64.90904 MiB/rank planner saving remains the only memory-performance claim**;
the added repairs make that saving safe across generic ModelOpt shapes.

Evidence:
[final exact matrix](artifacts/sparkinfer-w4a16-cae3aad-full-gpu2.log),
[final warm union](artifacts/sparkinfer-w4a16-cae3aad-full-warm-gpu2.log),
[pure PR #100 exact gate](artifacts/sparkinfer-pr100-c2f135c-w4a16-gpu2.log),
[pure bounds memcheck](artifacts/sparkinfer-pr100-c2f135c-bounds-memcheck-gpu2.log),
and
[final matched A/B summary](artifacts/w4a16-final-ab-summary.log).

### C. LMCache future lifecycle

Owner: `lmcache_lifecycle_fix` subagent.

Target: compose retained CUDA event resources with timeout, idempotent
completion, `_expire`, late-response safety, and exact-once resource release.

Checkpoint (2026-07-31 04:05 UTC): reconstructed release + #7/#8 prerequisites
+ #18/#19/#20 at local integration `caf7417be83f225c93e426f0885b935cefcc388c`.
The unified completion transition found an additional traceback-retention bug:
storing and raising the same timeout exception retained its future/CUDA event.
`2014fbb2271eae516b9f2c61dd0098601e99bd19` raises a fresh timeout while
keeping the terminal sentinel traceback-free. The union passed 124 CPU tests
(13 skipped), 137 CUDA-visible tests cold and warm, a 10-test outage/recovery
gate, and a real native c_ops REGISTER + 3 STORE/RETRIEVE + UNREGISTER round
trip.

Independent review nevertheless **rejected** this composition on lifecycle
safety. A caller timeout can release a CUDA IPC exporter while the daemon
still uses it because the transport has no cancellation acknowledgement;
recovery can register after shutdown, remote UNREGISTER errors skip later
cleanup, replaced transfer contexts leak, concurrent CUDA consumers can race
materialization, and timeout callbacks/observability are inconsistent. These
are correctness/reliability blockers, not test nitpicks. A new repair branch
is implementing separate caller/transport lifetime, stop-wins recovery,
finally-safe cleanup, old-context retirement, consumer serialization, and the
required deterministic regressions. Nothing has been pushed.

Checkpoint (2026-07-31 07:58 UTC): the replacement repair is now pinned at
`3d22b3ceb215b1173f262f6ac3a60ad913960dc6`. It separates caller completion
from the transport lease, quarantines an in-flight CUDA exporter after caller
timeout until a late reply or transport reset/close makes release safe,
serializes one-time CUDA-event materialization, contains timeout callbacks,
publishes timeout observability once, makes recovery stop-wins, atomically
replaces and closes old transfer contexts, and runs shutdown cleanup even
when remote unregister fails.

The first exact rerun at predecessor `29208935` is intentionally retained as
a harness/static failure: Ruff requested one formatting change, and adding
the source checkout to `PYTHONPATH` hid the image's compiled
`native_storage_ops` module. The formatting-only successor above was rerun
with the unchanged image `c_ops` and `native_storage_ops` binaries linked
temporarily into the source package and removed afterward. Its exact CPU
union passed **135 tests with 13 CUDA skips** in 57.19 s; Ruff, format and
`git diff --check` passed, and the post-cleanup tree was clean.

Evidence:
[retained predecessor static/harness failure](artifacts/lmcache-29208935-static.log),
[retained predecessor collection failure](artifacts/lmcache-29208935-cpu.log),
[final static gate](artifacts/lmcache-3d22b3ce-static.log), and
[final CPU union](artifacts/lmcache-3d22b3ce-cpu.log).

The exact final head then passed both cold and warm CUDA-visible unions:
**148/148** in 76.63/77.41 s. A focused **20/20** gate exercised callback
failure containment, one-shot observability, concurrent event
materialization, pre-send versus in-flight timeout, late transport reply,
reset/close release, saturated dead-client progress, stop-during-recovery,
context replacement/failure, and unregister-error cleanup. A real native
server round trip completed REGISTER, three 512-token STORE/RETRIEVE pairs
with three exact checksum matches, and UNREGISTER. Cold STORE averaged
2.85 ms; warm RETRIEVE averaged 2.51 ms. No test failed, and physical GPU 3
returned to idle after server teardown.

Evidence:
[cold CUDA union](artifacts/lmcache-3d22b3ce-gpu3-cold.log),
[warm CUDA union](artifacts/lmcache-3d22b3ce-gpu3-warm.log),
[focused lifecycle gate](artifacts/lmcache-3d22b3ce-lifecycle-gpu3.log),
[real server](artifacts/lmcache-3d22b3ce-real-server.log), and
[real round trip](artifacts/lmcache-3d22b3ce-real-roundtrip-gpu3.log).

Independent lifecycle review remains mandatory before this repair can be
accepted.

### D. vLLM/full appliance

Owner: main agent.

Target: refresh current `dev/gilded-gnosis` + #210, re-run focused gates, then
use the model already present on the rental to measure the full #210 capacity
sweep while the independent source fixes are prepared.

Checkpoint (2026-07-31 04:05 UTC): current
`dev/gilded-gnosis@30038602b71395f481ef4a6edfe4fcf8551d9c15` plus exact #210
produced throwaway integration `4fa1dd849dccae50ed7fa9104b873ef9a44cfedb`.
Focused source union: 16 passed and Ruff clean. GPU probabilistic union:
11/11 cold and 11/11 warm after the appliance was fully quiesced. The first
GPU attempt is invalid infrastructure evidence because PID 1 had respawned a
second supervisor; both child supervisors are now stopped and no vLLM process
uses the GPUs.

The reproducible full-model one-process harness is
`scripts/field_review_full_gate.sh`. It isolates state, credentials and JIT
caches, keeps the checkpoint read-only, disables all appliance auxiliaries,
and refuses to run when any GPU process is present.

The matched r14 control at GMU 0.90, MTP0, 3,072 scheduler tokens and 131,072
model length booted with:

- mixed Trellis arena: 759.8 MiB/rank, `prefill_capacity=3072`;
- GPU KV capacity: 264,960 tokens;
- resident GPU process memory: about 89.5 GiB/rank;
- model weights: 81.93 GiB/rank;
- initialization: 116.13 s after engine creation, including 40.12 s compile.

Unset current-base + #210 reproduced exactly 759.8 MiB and 264,960 KV tokens.
That first source-import attempt exposed two harness contaminants and is
diagnostic only: missing local-package FlashAttention `.so` links produced
fallback ERROR logs, and the turnkey structured-output compatibility patch
mutated the reviewed worktree. The binary links now point to the unchanged r14
image extensions, the candidate tree is clean, and candidate gates explicitly
skip that default-on runtime mutation to preserve exact-SHA attribution.
The installed PCIe calibration helper also exposed a separate reliability bug
when `LD_PRELOAD` is unset; its upstream source/fix is being tracked
independently.

The full current-base + exact #210 sweep is now complete. Every point booted
the 3.25 bpw model, served the exact `391` correctness response, completed all
requests, and exited without an engine/CUDA error. Results are matched at
TP4/DCP4, MTP0, dynamic NVFP4 KV, GMU 0.90, scheduler capacity 3,072,
131,072 model length, and one sequence:

| EXL3 prefill capacity | Trellis arena/rank | Arena vs 3,072 | GPU KV tokens | KV vs 3,072 | 3,072-token prefill median | Prefill delta | C1 TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,072 | 759.8 MiB | control | 264,960 | control | 1,845.90 tok/s | control | 29.334 ms |
| 1,536 | 399.7 MiB | -360.1 MiB | 312,832 | +18.07% | 1,827.01 tok/s | -1.02% | 29.434 ms |
| 1,024 | 279.7 MiB | -480.1 MiB | 328,960 | +24.15% | 1,807.58 tok/s | -2.08% | 29.429 ms |
| 512 | 159.7 MiB | -600.1 MiB | 344,832 | +30.15% | 1,751.77 tok/s | -5.10% | 29.374 ms |

The 1,024-token capacity is the current balanced candidate: it returns
480.1 MiB/rank and 64,000 KV tokens without a meaningful decode change for a
roughly 2% short-prefill cost. Capacity 512 is valid and maximizes headroom,
but its repeatable prefill tradeoff is large enough that it should remain an
explicit capacity-first option.

All cold runs exposed an appliance warmup gap rather than a #210 regression:
the server advertises ready before first-request Triton/CuTeDSL shapes have
compiled. A dedicated prime was excluded from measured medians; after it, no
new prefill JIT appeared in the steady repeats. This deserves separate
turnkey warmup work.

Evidence:
[3,072 server](artifacts/vllm-current210-cap3072-mtp0-v2-server.log),
[3,072 benchmark](artifacts/vllm-current210-cap3072-mtp0-v2-bench-measure-a.json),
[1,536 server](artifacts/vllm-current210-cap1536-mtp0-v1-server.log),
[1,536 benchmark](artifacts/vllm-current210-cap1536-mtp0-v1-bench-measure-a.json),
[1,024 server](artifacts/vllm-current210-cap1024-mtp0-v1-server.log),
[1,024 benchmark](artifacts/vllm-current210-cap1024-mtp0-v1-bench-measure-a.json),
[512 server](artifacts/vllm-current210-cap512-mtp0-v1-server.log),
and
[512 steady repeat](artifacts/vllm-current210-cap512-mtp0-v1-bench-measure-b.json).

The first MTP3/1,024 boot was correctly attributed as a **harness failure**,
not a candidate failure. SSH-launched processes inside the retained container
did not inherit PID 1's OCI `VLLM_EXL3_EXT_PATH`; MTP0 never imports the native
draft extension, while MTP3 failed at its first draft call with
`Unable to import exllamav3_ext`. The same SSH environment also omitted the
image's NCCL `LD_PRELOAD`, triggering the independently filed calibration
launcher bug. No Xid or residual GPU allocation followed the clean engine
shutdown. The harness now restores the immutable image paths
`/opt/exllamav3` and `/opt/libnccl-local-inference.so.2.30.4`, prints both,
and will rerun under a fresh evidence label/cache.
[Failed first MTP3 boot](artifacts/vllm-current210-cap1024-mtp3-v1-server.log).

The corrected MTP3 rerun reached the intended candidate code, completed PCIe
calibration, loaded the target and rank-sliced draft, and captured all graphs.
The measured 1,024-token Trellis arenas were 279.7 MiB/rank for the target and
414.1 MiB/rank for the draft. It then failed the standard GMU 0.90 profile's
KV-capacity check: only 0.01 GiB remained while the 131,072-token gate needed
0.98 GiB (estimated maximum length 1,792). This is a valid negative result for
the 3.25-bpw + MTP3 + GMU 0.90 shape, not an OOM, Xid, or #210 functional
failure. The engine shut down cleanly and released all four GPUs. A follow-up
uses GMU 0.92 while keeping the candidate, model, MTP, graph, and capacity
controls fixed.
[Standard-GMU MTP3 capacity failure](artifacts/vllm-current210-cap1024-mtp3-v2-server.log).

That controlled GMU 0.92 follow-up **passed** at exact composed vLLM head
`4fa1dd849dccae50ed7fa9104b873ef9a44cfedb`. It exposed 1.91 GiB/rank for
257,024 KV tokens (1.96 full 131,072-token requests), completed graph capture
in seven seconds using another 0.14 GiB/rank, and served without request
errors, preemptions, OOMs, or Xids. The arithmetic smoke returned exactly
`391`. A tokenizer-measured 126,327-token prompt retrieved the single needle
at 50% depth (`Kyoto: AJY-4896`) with no degeneration.

The matched steady 3,072-token C1 sample reported:

- median prefill: 1,783.0 tok/s (1.36% below the MTP0/capacity-1,024 control);
- median TPOT: 15.967 ms, or 62.63 tok/s, versus 29.430 ms / 33.98 tok/s
  without MTP;
- mean acceptance length: 3.111 of four total tokens/step;
- draft-token acceptance: 70.37%;
- GPU memory during the measured interval: 91,259–92,911 MiB/rank;
- observed peak power: 438.43 W on GPU 0.

The deliberately fresh JIT cache exposed the already tracked appliance warmup
gap: four first-request and four first-3,072-prefill kernels compiled after
readiness. A separate prime absorbed them; the measured sample and 126K
retrieval caused no further JIT.

The direct unset-capacity MTP3 control at the same GMU 0.92 then made the
memory result conclusive. Its target and draft arenas were respectively
759.8 MiB and 1,054.2 MiB/rank, versus 279.7 MiB and 414.1 MiB with capacity
1,024: **1,120.2 MiB/rank returned**. The control exposed only 0.82 GiB for
KV, failed the 131,072-token validation, and estimated a 110,080-token
maximum. The capacity-1,024 candidate exposed 1.91 GiB and 257,024 tokens.
Thus #210 changed this exact MTP3 shape from a deterministic boot failure into
a correct 126K-capable service. The negative control exited without an OOM,
Xid, or residual allocation.

A capacity-2,048 follow-up was also healthy, with 519.7 MiB target and
734.1 MiB draft arenas, 1.37 GiB available KV, and 183,552 KV tokens. It
improved median 3,072-token prefill by only 0.29% over capacity 1,024
(1,788.2 versus 1,783.0 tok/s), while returning 73,472 fewer KV tokens.
Median TPOT was unchanged (15.971 versus 15.967 ms); MAL/draft acceptance
were 3.161/72.04% versus 3.111/70.37%, well within this three-request sample's
content variance. Capacity 1,024 therefore remains the measured balanced
choice; 2,048 is not promoted on a performance claim that small.

A matched MTP3-versus-MTP5 follow-up kept the exact source, capacity 1,024,
GMU 0.92, 3,072-token prompts, six prompt bodies, and reusable compile cache
fixed. Both modes exposed the same 257,024-token KV capacity. MTP5's three
steady prefill measurements had a 1,775.5 tok/s median versus 1,767.3 tok/s
for MTP3 (+0.46%), which is noise-sized. Its decode result was not:

| Draft depth | Draft acceptance | MAL | Mean TPOT | Median TPOT | p95 TPOT |
|---:|---:|---:|---:|---:|---:|
| MTP3 | 84.10% | 3.523 | 16.962 ms | 15.059 ms | 23.647 ms |
| MTP5 | 39.84% | 2.992 | 25.104 ms | 13.663 ms | 48.709 ms |

MTP5 produced four superficially fast 13–14 ms requests and two 48.4–48.8 ms
requests. Its attractive median therefore concealed a 48% worse mean and a
106% worse p95 than MTP3. The MTP3 prime absorbed all post-ready JIT warnings;
the six measured requests caused no additional compilation. MTP3 remains the
balanced candidate for this workload.

This rerun also validated the field harness's process-group teardown: sending
TERM to the recorded wrapper stopped the complete entrypoint/vLLM tree and
released all four GPUs without a residual worker.

Evidence:
[server](artifacts/vllm-current210-cap1024-mtp3-v3-server.log),
[arithmetic](artifacts/vllm-current210-cap1024-mtp3-v3-correctness.json),
[steady benchmark](artifacts/vllm-current210-cap1024-mtp3-v3-bench-measure-a.json),
[GPU telemetry](artifacts/vllm-current210-cap1024-mtp3-v3-gpu.csv), and
[126K retrieval](artifacts/vllm-current210-cap1024-mtp3-v3-needle-126k.json).
[Unset-capacity MTP3 control](artifacts/vllm-current210-unset-mtp3-v1-server.log).
[Capacity-2,048 benchmark](artifacts/vllm-current210-cap2048-mtp3-v1-bench-measure-a.json)
and
[server](artifacts/vllm-current210-cap2048-mtp3-v1-server.log).
[Matched MTP3 benchmark](artifacts/vllm-current210-cap1024-mtp3-v4-matched-bench-measure-b.json),
[MTP3 server/teardown log](artifacts/vllm-current210-cap1024-mtp3-v4-matched-server.log),
[matched MTP5 benchmark](artifacts/vllm-current210-cap1024-mtp5-v1-bench-measure-b.json),
and
[MTP5 server](artifacts/vllm-current210-cap1024-mtp5-v1-server.log).

### E. PCIe calibration launcher reliability

The full-model gates independently exposed a release-launcher defect outside
the nine-PR inventory. `blackwell-llm-docker@2464cc03` evaluates bare
`$LD_PRELOAD` under `set -u`; an unset variable aborts
`glm52-pcie-runtime-env.sh` before calibration and silently forces the turnkey
fallback posture.

Stacked candidate `b4a2f25` normalizes `${LD_PRELOAD:-}`, idempotently prepends
the local NCCL shim, and unconditionally exports the final value—including
when a shell-local, unexported value already contains the shim. Results:

- exact parent `2464cc03` with unset `LD_PRELOAD`: exit 1,
  `unbound variable`;
- candidate shell release gate: PASS;
- real retained Vast environment: unset value becomes exported
  `/opt/libnccl-local-inference.so.2.30.4`;
- a second invocation leaves exactly one shim entry;
- local Bash 5 release gate, ShellCheck, 14 Python calibration tests, syntax,
  and `git diff --check`: clean.

Evidence:
[parent reproduction](artifacts/blackwell-ldpreload-parent-repro.log) and
[Vast candidate gate](artifacts/blackwell-ldpreload-vast-gate.log).
Independent re-review returned **APPROVE** after reproducing the parent
failure and checking Bash 3.2/5.3, unset/empty/exported/unexported values,
entry preservation, metacharacters, idempotence, and child inheritance.
Published upstream as
[blackwell-llm-docker issue #12](https://github.com/local-inference-lab/blackwell-llm-docker/issues/12)
and
[repair PR #13](https://github.com/local-inference-lab/blackwell-llm-docker/pull/13)
at current exact head `5eee240d6c32e08516287380131c96ca7493f0f6`. The
implementation was independently approved at `b4a2f25`; the current commit
only adds the peer-suggested unexported-unrelated and space-separated preload
tests. The exact current head passed the complete release helper on Vast.
[Current-head Vast gate](artifacts/blackwell-ldpreload-5eee240-vast-gate.log).
The immutable r14 image remains affected.

## Public status

The first field results were posted to all nine PRs and all nine mapped issues
at 2026-07-31 03:07 UTC. Subsequent comments must edit or clearly supersede
those results with exact repaired SHAs; do not erase the original failure
record.

## Rental state

Vast instance `46335896` was restored healthy after the first campaign:
4 workers, `/health` 200, authenticated `GLM-5.2` model listing, and exact
restore-smoke response. The instance is explicitly retained for this repair
and battle-test campaign.

During repair, PID 1 remains alive while both known child supervisors (PIDs 176
and 70970 at this checkpoint) are stopped. Always re-check the full process
tree and `nvidia-smi` before attributing a GPU result. Restore exactly one
supervisor only after the combined stack has been selected.

### F. Exact r14-base reconstruction

The first focused gates intentionally followed each PR's filed/current base.
The deployable candidate has an additional boundary: the immutable r14 image
is a three-repository composition, not any one PR base. The release lock files
from `blackwell-llm-docker@2464cc03` were therefore replayed before adding the
accepted field-review deltas.

All three reconstructed trees matched their published lock exactly before a
single field-review commit was applied:

| component | release base | published r14 tree |
|---|---|---|
| vLLM | `f978d009fab996617f9a3cadef36ce727bcd83cd` | `749050edab1b6664937c52fa1b0be360be632c1e` |
| SparkInfer | `9b852b281250123fe323f63ccb1df3cac0f3bbca` | `8110e3ea417794bfb08aff1fba20135102e5536b` |
| LMCache | `9cebd405d0caf4bebe01d694b5a8bf4e3e354314` | `a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa` |

The exact r14 vLLM tree then accepted merged #211 and both #210 commits
without a conflict. Candidate `5a7ac1481e1fc24b0bbc35efe160b2ff34797bee`
(tree `46f6f5bf387ee2c07712808cfe64f722fa2b99f3`) passed:

- Ruff on every changed runtime/test file;
- the #210/#211 CPU union: **16/16**;
- the probabilistic GPU union on physical GPU 0: **11/11 cold** and
  **11/11 warm**.

The unchanged r14 extension binaries were temporarily linked into the source
worktree because `PYTHONPATH` otherwise shadows the installed native modules.
Their SHA-256 values are in the log; all links were removed and the source
worktree was clean after each gate.

The final appliance patch order was also replayed in a detached copy of that
candidate. With `EXL3_PY_PATH` explicitly bound to the detached source, the
turnkey parity-ABI compatibility patch added its one required trailing integer,
passed `py_compile` and `diff --check`, then reported `already patched` on the
second invocation. The r14 mixed-K compatibility probe correctly recognized
the native implementation and made no edit. The post-compatibility #210/#211
CPU union remained **16/16**, and the probabilistic GPU union remained
**11/11 cold** and **11/11 warm** on physical GPU 0. An initial diagnostic
invocation had accidentally left `EXL3_PY_PATH` unset and therefore inspected
the detached tree while patching the installed package; that target error is
retained in the raw log and is not counted as evidence.

The exact r14 SparkInfer tree accepted final #100 source and the reviewed
#100/#102 merge. One source-composition conflict was recorded in
`tests/moe/test_w4a16_e2e.py`: both PRs edited the same import block. The
explicit resolution retains #102's `route_pack_capacity` import. Runtime
changes merged automatically. Candidate
`e2205cba8d78db03427a3945a75a1f647b51ef5a` (tree
`1d378d599538daecfc1e8c11a44010b27d3ddfe5`) passed Ruff, `diff --check`,
and its no-GPU union with **89 passed / 259 expected CUDA skips**. The complete
seven-file GPU union then passed **332 passed / 16 intentional skips** cold
in 237.03 seconds and identically warm in 10.03 seconds on physical GPU 2,
using a fresh candidate-specific extension cache. Compared with the
independently approved exact #100/#102 tree `cae3aad1`, its only runtime
differences are r14's paged planner and native mixed-K Trellis module; the
common repaired source is otherwise identical. Final #101/#103 replay
composition remains pending.

The exact r14 LMCache tree accepted #18/#19/#20 plus the lifecycle repair
without a conflict. Candidate
`0056361275815f7b283366d993a9fa8b069ecd8f` (tree
`00f710eb833ba720027d9f0694b19e82ff0c0f6b`) passed:

- Ruff and `diff --check`;
- **135 passed / 13 CUDA skips** on CPU;
- **148/148** on physical GPU 3 cold;
- all six component suites separately warm, again all passing.

The installed r14 `native_storage_ops` binary was linked only for collection,
at SHA-256
`6e99b95161126bfd7fa1638613a726edd9e0f4ad7edb23a9f0b2a70167598eb8`,
then removed. The recurrent PyTorch
`Producer process has been terminated before all shared CUDA tensors released`
warning was isolated to
`tests/v1/multiprocess/test_mq.py::test_mq_register_kv_cache`: its client
process exits before the still-running test server releases the shared fixture
tensors. No other warm suite emits it and no GPU process survives.

Independent review nevertheless **rejected promotion**: the fixture is only
one manifestation of a production CUDA IPC ownership defect. A
`CudaIPCWrapper` creates one PyTorch exporter counter, but initial registration
materializes it once for device detection and again for the retained CUDA cache
context. Recovery has the inverse problem: a fresh wrapper/export is sent, then
the existing-instance fast path returns without importing or explicitly
releasing it. The real checksum-verified three-cycle GPU round trip also emits
the warning after an acknowledged unregister, proving this is not fixture-only
noise. Because LMCache disables GitHub Issues, the defect and acceptance
contract are tracked in [rtx6kpro #49](https://github.com/local-inference-lab/rtx6kpro/issues/49).
Candidate `00563612` remains a useful exact composition base but is not
deployable until one wire export is matched by exactly one import or explicit
release on every success, recovery, race, exception and teardown path.

Evidence:
[vLLM CPU/static](artifacts/vllm-r14-210-211-static-cpu-v2.log),
[vLLM GPU cold/warm](artifacts/vllm-r14-210-211-gpu0-cold-warm.log),
[vLLM appliance patch order](artifacts/vllm-r14-field-runtime-overlay.log),
[vLLM post-compat GPU cold/warm](artifacts/vllm-r14-field-runtime-overlay-gpu0.log),
[SparkInfer CPU/static](artifacts/sparkinfer-r14-w4-100-102-static-cpu.log),
[SparkInfer GPU cold](artifacts/sparkinfer-r14-w4-e2205cba-gpu2-cold.log),
[SparkInfer GPU warm](artifacts/sparkinfer-r14-w4-e2205cba-gpu2-warm.log),
[LMCache CPU/static](artifacts/lmcache-r14-00563612-static-cpu-v2.log),
[LMCache CUDA cold](artifacts/lmcache-r14-00563612-gpu3-cold.log), and
[LMCache warm split](artifacts/lmcache-r14-00563612-gpu3-warm-split.log).
The earlier candidate's source-identical ownership path is captured by the
[real checksum round trip](artifacts/lmcache-3d22b3ce-real-roundtrip-gpu3.log).

### G. Second-stage lifecycle safety review

The first post-replay SparkInfer hardening candidate,
`f0eb0f7763f9c1c521bcc714353c8321acd3d8f2`, is **rejected** despite positive
focused results. It passed 180 CPU/static tests (19 expected GPU skips), a
two-rank reduced-SM collective rejection before IPC allocation, and a DCP2
1,025-replay/skew/teardown gate. Its corresponding DCP4 warm gate then stalled
for more than 6.5 minutes immediately after reporting the four 188-SM devices;
the parent and all four workers were terminated and the GPUs were verified
idle. That run is retained as HUNG/INVALID evidence, never a pass.

Independent source review found five remaining lifecycle defects:

1. the no-op one-shot destructor did not retain the Torch `rank_data` tensor
   whose raw pointer is held by native code, so real GC could recycle storage
   still referenced by queued/captured kernels;
2. one-shot capture exit did not restore/remove stream-key aliases in strict
   LIFO order, allowing a recycled capture key to adopt graph-owned staging;
3. process-local stream handles and encounter order did not give channels a
   cross-rank logical identity, so asymmetric A/B discovery could pair A with
   B and deadlock;
4. CUDA runtime/device/extension/layout failures before shared-buffer
   allocation were not collectively reported, so one rank could fail while
   peers entered allocation; and
5. public direct constructors bypassed the new SM-residency gate.

No SparkInfer replay patch will enter the r14 bundle until a new commit fixes
all five, passes an independent re-review, and repeats the asymmetric-order,
actual-GC, nested-capture, one-rank setup-failure, reduced-SM, DCP2 and DCP4
GPU gates from fresh caches.

The first LMCache ownership repair was likewise rejected before GPU promotion.
The replacement design is being reviewed against a stricter contract:
serialization must be one-shot and race-safe; an accepted ZeroMQ send must
make the entire wrapper-batch ownership commit non-throwing and atomic; any
ambiguous post-send state must retain the full payload for process lifetime;
unknown-handler and frame-count failures must decode/release every known or
generic extension frame; and async forwarding must pin the export before the
handler's `finally` cleanup can run.

SparkInfer replacement `cfb257b` is now frozen for independent review. Its
112-test CPU/static gate passed with five expected CUDA skips, Ruff and
`diff --check` clean. It adds full Torch-owner quarantine, LIFO alias
restoration, collective pre-allocation and constructor verdicts, canonical
logical channels, and a collectively checked monotonic compatibility path for
r14's existing no-argument `capture(stream=...)` callers. The capture ordinal
is part of checkpoint/rollback state so disposable profiling does not consume
or collide with later target/draft graph identities. No GPU result is
attributed to this SHA until the independent source review approves it.

LMCache ownership candidate `0da021f6` is **rejected before GPU promotion**.
Independent review produced deterministic counterexamples that its passing
203-test CPU suite did not cover:

1. the sender encoded one payload graph but committed ownership by re-walking
   the caller's mutable object after send, so a concurrent mutation could drop
   the producer reservation after the receiver owned the wire copy;
2. a typed-decode failure in a known frame position skipped generic fallback
   for that same frame, leaking a transferred CUDA extension under schema
   skew;
3. an unknown/malformed request type escaped before rejection cleanup, killing
   the server loop and leaking all trailing IPC frames;
4. an early transfer-participant discovery failure quarantined only the
   wrappers already visited, then an async lease released another accepted
   wrapper;
5. normal-response decode failure removed a future from tracking without
   completing either the future or its retained transport resources; and
6. direct CUDA wire tests did not mark the encoded sender as transferred,
   permitting a false-positive two-decrement ownership sequence; and
7. the new opt-in participation probe defaulted to false, silently excluding
   third-party wrappers that implemented the older public mark/release
   contract from strict ownership transfer.

The successor must bind an immutable ownership snapshot to the encoded frames,
quarantine that entire snapshot on any post-send ambiguity, generically
release every rejected frame even under schema/type skew, keep the daemon
alive, complete malformed-response futures, and make the direct-wire GPU tests
model sender and receiver ownership explicitly.

SparkInfer successor `025c7e1549aee0b37ad24ca8545251b9a7344774`
(parent `cfb257b`) is likewise **rejected before GPU promotion** after its
independent frozen-SHA review. Its focused CPU/static gate passed 98 tests
with one skip and Ruff/`diff --check` clean, but positive unit evidence does
not discharge these integration/lifecycle blockers:

1. distributed one-shot and DCP capture now require a semantic channel ID, so
   this SparkInfer commit is usable only atomically with a reviewed vLLM
   caller patch covering target/draft and profile/production owners;
2. handle/verdict exchange and peer-unmap failures can deliberately leave a
   CUDA export/import alive while losing the Python object that could retry or
   diagnose its release; only `cudaFree` failure currently enters the retry
   registry;
3. logical-ID normalization/rejection can happen locally before a collective,
   so one invalid rank can strand peers that entered first-use allocation;
4. the legacy `default` eager channel still binds to the first stream and
   rejects a second stream; every production multi-stream caller therefore
   needs an explicit stable ID or another collectively safe contract;
5. real `del` plus `gc.collect()` ownership is not proven for DCP/two-shot;
   manually invoking `__del__` on a still-referenced object is insufficient;
6. the single-grid residency proof does not yet cover concurrently resident
   opposite-order channels at their maximum worker-grid sizes.

The next successor must retain every unresolved allocation with exactly-once
retry ownership, collectively preflight malformed/unknown IDs, close the
eager-callsite contract with its matching vLLM patch, and add bounded
asymmetric-failure, real-GC and maximum-overlap tests before any fresh-cache
GPU run.

The atomic vLLM half is now independently **approved** at exact clean SHA
`a89816a3a1062a2ae82e9ac87abd67e2bdcdfe98` (base `5a7ac148`). It assigns
stable target/draft/encoder and profile/production identities across all V1
and V2 GLM speculative paths, rolls disposable profiling channels back only
after graph destruction and synchronized KV cleanup, and fails closed when a
distributed caller omits the new semantic ID. Static gates passed; Linux
target/draft capture and replay remain required with the corrected SparkInfer
half. [Independent review](artifacts/vllm-a89816a3-independent-review.md).

LMCache ownership successor `4ab7112e2a442e05928ba4c78e4cf09076b419b9`
then cleared every available GPU behavior gate: **172/172 cold**, **172/172
warm**, focused registration cleanup, and a native daemon registration plus
three checksum-exact STORE/RETRIEVE cycles and UNREGISTER. Every warning gate
was clean. It is nevertheless **rejected before appliance promotion** after a
second independent review found four deterministic lifecycle gaps: caught
exception traceback retention, permanent quarantine of empty ownership
records, worker-pool/notifier shutdown ordering, and malformed response-header
futures left pending. It also needs upstream LMCache shutdown commit
`b20e6151` plus a live-registration serving teardown gate. This distinction is
important: the original CUDA exporter warning is repaired, but daemon outage
and shutdown safety are not yet complete.

Evidence:
[independent review](artifacts/lmcache-4ab7112e-independent-review.md),
[focused registration](artifacts/lmcache-r14-field-4ab7112e-register-gpu3.log),
[cold union](artifacts/lmcache-r14-field-4ab7112e-gpu3-cold.log),
[warm union](artifacts/lmcache-r14-field-4ab7112e-gpu3-warm.log),
[native client round trip](artifacts/lmcache-r14-field-4ab7112e-real-roundtrip-gpu3.log),
and
[native server log](artifacts/lmcache-r14-field-4ab7112e-real-server-gpu3.log).

### H. Fail-closed appliance packaging

The first deployable appliance boundary was exercised before adding either
still-pending IPC repair. Branch `codex/field-review-combined` at
`384c013fdc2942da74faabeab206364291084c68` contains only the independently
accepted exact-r14 vLLM #210/#211 delta. The Docker build:

- copies a content-addressed field-repair manifest and patch bundle;
- preflights every component, source/runtime target and patch dry run before
  mutating any tree;
- applies the vLLM repair to both `/opt/vllm` and the active virtual
  environment's `site-packages`;
- fails the image build on a missing target, base-tree mismatch, patch
  mismatch, partial component, or post-apply verification failure;
- invalidates the GitHub image cache when any file below `patches/` changes;
  and
- moves the runtime compile-cache namespace to
  `turnkey-exl3native-field1`, preventing an older compiled extension from
  hiding source changes.

The local applicator contract passed 4/4 tests, the appliance family suite
passed 275 tests, and ShellCheck was clean. Manual GitHub Actions run
[30623908189](https://github.com/malaiwah/glm52-exl3-vast/actions/runs/30623908189)
completed every lint/test/build job and pushed:

```text
ghcr.io/malaiwah/glm52-exl3-vast:384c013fdc2942da74faabeab206364291084c68
sha256:de0e912e15c3345f65988a677442d0ea271f6103f911f24a5e7cb3b29c66bf79
```

The build log explicitly records successful application at both vLLM target
trees. This image is packaging evidence, **not** the final candidate:
SparkInfer #101/#103 and LMCache #18/#19/#20 remain excluded until their
second-stage lifecycle reviews and fresh-GPU gates pass.

The next bundle revision also packages independently approved
blackwell-llm-docker PR #13 as an exact `launcher-bin` component instead of
depending on a future parent-image refresh. Its fail-closed boundary is the
r14 `/usr/local/bin/glm52-pcie-runtime-env.sh` hash
`4838499b...b36d9`; the accepted output is `e3a35eef...2758`. A pure-r14
fixture on the retained Vast host applied all three currently accepted
components atomically, reported every component already applied on replay,
and propagated `/opt/libnccl-local-inference.so.2.30.4` to a child shell from
an initially empty `LD_PRELOAD`. The applicator contract is now 5/5, including
an isolated launcher-target hash/idempotence test. Evidence:
[three-component apply/replay](artifacts/field-review-bundle-v3-launcher-apply.log).

### I. Full-model pre-composition control

Before either pending IPC repair was admitted, the retained Vast host ran a
full GLM-5.2 control with the accepted exact-r14 vLLM #210/#211 overlay only.
The installed r14 SparkInfer and LMCache remained unchanged. The exact profile
was TP4/DCP4, the willfalco 3.25-bpw mixed K3/K4 checkpoint, MTP3
probabilistic sampling, EXL3 prefill capacity 1,024, scheduler capacity 2,048,
graph/sequence window 32/8, GMU 0.92, and 131,072 maximum model length.

Startup established the memory and graph baseline:

- target mixed-Trellis arena: **294.6 MiB/rank** at decode 32 and prefill
  capacity 1,024;
- rank-sliced MTP draft arena: **414.1 MiB/rank**;
- weights: **83.28 GiB/rank**;
- profiled peak activation: **1.32 GiB/rank**;
- actual CUDA-graph pool: **0.14 GiB/rank**;
- available KV memory: **1.88 GiB/rank**, exposing **252,416 tokens**;
- engine initialization: **200.23 seconds**.

The authenticated feature gate passed health/model discovery, tokenize,
thinking and non-thinking chat, streamed usage, multi-turn thinking
preservation, strict JSON with and without thinking, automatic and required
tool calls, and tool-result round trip. Vision was intentionally skipped
because this checkpoint/profile has no vision tower. The short correctness
gate and five-depth 32K retrieval passed.

The independent long-context gate then ran two seeds at each tested length.
All **20/20** planted values were recovered: five depths
(1/25/50/75/99%) at approximately 65.9K and 126.3K tokens, twice. Every
response was non-degenerate. Durations were 37.4 seconds at 65.9K and 73.1
seconds at 126.3K.

After all prior traffic had warmed the engine, fresh non-prefix-cached
prefills measured **1,898.7 tok/s** at 3,088 prompt tokens and
**1,849.0 tok/s** at 32,780 prompt tokens. The original matrix's first 3K
point measured only 558.3 tok/s while a new shape compiled; it is retained as
cold-shape evidence and is not used as the steady baseline. Its 32K point
measured 1,862.8 tok/s.

The matched eight-request concurrency sweep used 3,075-token prompts,
128-token outputs, temperature 1.0, and a fixed seed:

| concurrency | aggregate output tok/s | prompt tok/s | p50 TPOT | p95 TPOT | MAL | draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.182 | 653.014 | 19.779 ms | 27.149 ms | 2.772 | 59.06% |
| 2 | 33.935 | 815.230 | 32.689 ms | 73.957 ms | 2.937 | 64.57% |
| 4 | 35.579 | 854.719 | 72.481 ms | 136.258 ms | 2.393 | 46.45% |
| 8 | 63.026 | 1,514.102 | 59.379 ms | 101.233 ms | 2.675 | 55.82% |

All 32 decode requests completed with zero request error or preemption.
These are rental-host workload measurements, not the r14 release host's CC1
microbenchmark.

This same matrix closes the plan's separate #211 live-probabilistic gate.
Unlike the original greedy release recipe, it explicitly used temperature
1.0 and a fixed seed. In the V2 runner, `method=mtp` selects
`MTPSpeculator`, which inherits `AutoRegressiveSpeculator`; non-null
probabilistic draft logits therefore delegate to #211's shared salted sampler.
The 32 live requests generated 4,096 tokens across 1,526 speculative steps:
4,578 draft tokens, 2,563 accepted tokens, pooled MAL **2.680**, and pooled
draft acceptance **55.99%**. The prior exact-r14 GPU distribution suite passed
11/11 cold and warm, including the greedy/temperature-zero controls. No
separate plain-checkpoint download is needed to exercise this code path.

This exact-source control is **not a clean final-stack PASS** despite correct
responses. Its cold-window audit records 28 post-ready SparkInfer cache
misses/JIT events while the feature and benchmark harness deliberately primed
previously unseen shapes. That cold compilation is permitted by the plan; the
warm window beginning after the final miss contains no compile, ERROR,
CUDA/IPC, or process-failure finding. XGrammar did log four
`Failed to advance FSM` errors during otherwise successful strict-JSON
responses. The exact-source gate intentionally disabled the appliance's
structured-output compatibility patch to preserve SHA attribution, so the
final built-image gate must prove that those four errors disappear.
Response-level success does not erase that remaining log-level requirement.

Evidence:
[server log](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-server.log),
[feature gate](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-features.json),
[32K verification](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-verify32k.json),
[65K/126K needle matrix](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-needles.json),
[concurrency matrix](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-benchmark.json),
[cold-window log audit](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-log-audit.json),
[clean warm-window log audit](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-warm-log-audit.json),
and
[steady prefill control](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-steady-prefill.json).

### J. Reproducible scorecard control

A dependency-free generation scorecard now records the exact dataset bytes,
harness bytes, source indices, choice permutations, raw visible/reasoning
fields, API usage, latency and before/after speculative counters. GSM8K uses
the EleutherAI v3 fixed eight-shot CoT prompt adapted to a chat endpoint; GPQA
Diamond uses OpenAI simple-evals' deterministic zero-shot permutation and
answer format. The accepted harness scores only user-visible `content`—a
tentative value in hidden thinking is diagnostic, not a completed answer.

The first harness iteration is retained but not accepted as the matched
control. Completion-only GSM stop strings could occur inside GLM's thinking
trace, and its fallback could score hidden reasoning when visible content was
empty. It reported 93/100 GSM8K. Its 4,096-token GPQA run reported 32/50, but
19/50 responses reached the length cap and only 1 of those 19 was correct.
Those records explain the harness correction; they are not silently folded
into the final score.

The corrected exact harness
`c1028f59c73f8f86c27cac6fe95dfb8066244d5a738cfa54501b3edc8b59e46c`
then ran the same fixed-seed 100-question GSM8K subset at concurrency 8 with a
2,048-token ceiling. Dataset bytes were
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Results:

| metric | control |
|---|---:|
| strict exact score | **96/100** |
| API errors / preemptions | **0 / 0** |
| wall time | 121.244 s |
| prompt throughput | 595.906 tok/s |
| aggregate output throughput | 264.978 tok/s |
| mean acceptance length | 3.5444 |
| draft-token acceptance | 84.81% |

All four misses completed normally rather than truncating or failing the
backend. Source 454 is a known data inconsistency: its question says both
people eat four apples per day, while the reference solution silently changes
one person to one apple; the model answered the literal question. The strict
score remains 96/100 without hand-adjustment. Sources 255, 306 and 649 are
retained as ordinary model misses. The complete serving log had zero ERROR,
traceback, OOM, Xid, CUDA-IPC warning or preemption through this control.

Evidence:
[exact harness](artifacts/evaluate_scorecard-c1028f59.py),
[accepted GSM8K JSON](artifacts/vllm210211-control-gsm8k100-v2.json),
[accepted progress log](artifacts/vllm210211-control-gsm8k100-v2.log),
[superseded GSM8K JSON](artifacts/vllm210211-control-gsm8k100.json), and
[4K-cap GPQA diagnostic](artifacts/vllm210211-control-gpqa50.json).

The same exact harness then repeated the fixed-seed 50-question GPQA Diamond
subset with an 8,192-token ceiling. It completed all 50 API requests without
an error or preemption and scored **28/50 (56%)** in 806.031 seconds. Aggregate
output throughput was 299.691 tok/s, pooled mean acceptance length was 3.1788,
and pooled draft-token acceptance was 72.63%. This is reproducible matched
control evidence, but it is not a clean unconstrained quality estimate:
21/50 requests exhausted even the 8,192-token ceiling while still reasoning,
returned no user-visible `content`, and were correctly scored as misses. The
raw reasoning traces remain in the artifact; the harness neither promotes a
hidden tentative choice to an answer nor hides the model-behavior limitation.
The 56% result is therefore a cap-constrained lower bound, not an unconstrained
GLM-5.2 quality claim. A future matched run must give both control and
candidate at least 16K, preferably 32K, completion tokens and report the
remaining truncation rate separately. No additional scorecard run will
displace GPU qualification work.

The harness now makes that contract durable: its task-aware default is 4,096
tokens for GSM8K and 32,768 for GPQA Diamond.  An explicit `--max-tokens`
continues to override either value, and any capped response is still retained
and reported rather than scored from hidden reasoning.

[8K-cap matched GPQA JSON](artifacts/vllm210211-control-gpqa50-v2.json) and
[progress log](artifacts/vllm210211-control-gpqa50-v2.log). The
[complete compressed control serving log](artifacts/scorecard-control-v1-server.log.gz)
contains zero ERROR, traceback, OOM, Xid, preemption, or failed-FSM record
through graceful shutdown; all four GPU workers exited.

### K. Frozen PCIe and LMCache successors

The public SparkInfer #101/#103 heads remain rejected evidence; their safe
replacement is the independently reviewed frozen commit
`31fa6a48116471ce423f0338047453ec1032c202`.  It keeps semantic channel
allocation collective and fail-closed while allowing ranks to capture already
prepared channels in different local orders.  A fresh extension-cache run on
all four rental GPUs passed the exact adversarial case that rejected the prior
successor: two channels prepared in opposite rank-local order, 128 eager
iterations and 1,025 CUDA-graph replays (**1 passed in 6.36 seconds**).

The same frozen source also passed:

- the four-rank one-shot torture gate with 64 eager iterations, 1,025 graph
  replays and 64 overlapping-stream iterations (**7.11 seconds**);
- four-rank fused RMSNorm correctness (**5.71 seconds**);
- collective argument rejection, one-rank pre-allocation failure, and the
  corrected synthetic 32-visible-SM rejection before CUDA IPC allocation;
- four-rank DCP eager/capture correctness across 64 graph replays plus an
  injected unmap failure and coherent teardown retry (**115.01 seconds**);
- four-rank two-shot eager/capture, alternating staging slots and clean
  teardown; and
- the complete communication CPU/static suite (**239 passed / 21 expected
  skips**).

The first reduced-SM command omitted its required synthetic
`SPARKINFER_PCIE_TEST_VISIBLE_SM_COUNT=32`.  The guard correctly detected that
allocation had started and failed that invalid harness invocation.  The
corrected exact command passed; both logs are retained.  The two-shot
microbenchmark was also deliberately not promoted as a speed claim on this
shape/topology: SparkInfer reduce-scatter/all-gather measured 3,714.8/3,767.3
microseconds versus NCCL's 1,024.3/990.5 microseconds.  Its value here is the
correctness, capture and ownership gate.

The atomic caller boundary uses vLLM
`be1e289a8ca6cc043b582b26c788efc4b1f5d0a8`.  With the frozen SparkInfer head
on `PYTHONPATH`, all **64/64** fused-allreduce and DCP tests passed on Linux/GPU
in 102.17 seconds.  This proves the semantic eager/target/draft/encoder channel
contract as one unit rather than qualifying either half against an old
installed dependency.

LMCache converged through three retained GPU observations.  Commit `0c71e15f`
removed the invalid same-process reopening of an exporter-owned CUDA event,
then exposed a legitimate asynchronous visibility window: STORE had completed
its GPU copy but the 5 ms completion dispatcher had not yet released the L1
write lock.  Commit `79fedc4a` made the test retry a complete LOOKUP under the
existing 20-second deadline; it cannot hide a dropped completion because the
600-second write TTL cannot expire during that window, and RETRIEVE still
compares every output byte.  Finally,
`f24812ee33dc9196e788d1004b1d68f473741e2b` releases the capability probe's one
`_share_cuda_()` reservation and lets the spawned daemon follow production's
SIGTERM cleanup path.  It also logs forced cleanup of wholly unlocked cache
objects at INFO while retaining WARN, with an exact count, whenever any read
or write lock exists.

On physical GPU 3, the final same-ID reset/register/STORE/LOOKUP/RETRIEVE/
UNREGISTER test passed in 5.14 seconds.  Its contract compared all 32 source
and destination BF16 layer tensors with `torch.allclose(..., atol=1e-4)`; all
comparisons passed.  Teardown reported zero locked objects, normal
`MPCacheServer closed`, and no PyTorch shared-CUDA producer warning.  The exact
final composed CUDA union passed **184/184** in 70.14 seconds.  Independent
review approved the runtime ownership and test contracts; the final image
still needs the full-model startup/traffic/shutdown gate before this bundle is
called deployable.

Evidence:
[opposite-order gate](artifacts/spark-31fa6a4-opposite-order-dcp4-1025.log),
[one-shot torture](artifacts/spark-31fa6a4-oneshot-torture-dcp4-1025.log),
[fused RMSNorm](artifacts/spark-31fa6a4-fused-rms-dcp4.log),
[retained invalid reduced-SM invocation](artifacts/spark-31fa6a4-preallocation-rejection-dcp4.log),
[corrected reduced-SM gate](artifacts/spark-31fa6a4-reduced-sm-rejection-dcp4-corrected.log),
[DCP teardown retry](artifacts/spark-31fa6a4-dcp4-64-teardown-retry.log),
[two-shot gate](artifacts/spark-31fa6a4-twoshot-dcp4.log),
[atomic vLLM/SparkInfer union](artifacts/atomic-vllm-be1e289-spark-31fa6a4-linux-gpu-full.log),
[LMCache final CUDA union](artifacts/lmcache-f24812e-composed-gpu3-union.log),
and
[LMCache clean recovery round trip](artifacts/lmcache-f24812e-same-id-cuda-roundtrip.log).
