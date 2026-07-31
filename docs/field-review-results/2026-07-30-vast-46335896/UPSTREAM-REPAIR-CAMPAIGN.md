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
Independent review is still required before publication.

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
