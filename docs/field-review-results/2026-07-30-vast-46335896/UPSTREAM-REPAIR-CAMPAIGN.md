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
5.1 s. Independent peer review is in progress; nothing has been pushed.

### B. SparkInfer W4A16 scratch lifetime

Owner: `w4a16_scratch_fix` subagent.

Target: precise initialization/lifetime correction for the #100/#102 ordered
NaN, preserving the 64.91 MiB/rank planner saving and avoiding broad arena
zeroing unless measurement proves it harmless.

Checkpoint (2026-07-31 03:37 UTC): a deterministic same-binding replay now
reproduces stale-tail NaNs on exact #100 for both ReLU2 and SiLU small-M direct
FC2 paths; a narrow lane-mask candidate makes the same cases finite without
whole-arena zeroing. Warm regression and performance/code-generation review are
still pending. Do not treat the earlier poison-before-first-run pass as a
disproof: FC1 overwrote the tested scratch before FC2 in that ordering; the
predecessor/replay sequence is the reachable stale-lifetime case.

### C. LMCache future lifecycle

Owner: `lmcache_lifecycle_fix` subagent.

Target: compose retained CUDA event resources with timeout, idempotent
completion, `_expire`, late-response safety, and exact-once resource release.

Checkpoint (2026-07-31 03:37 UTC): reconstructed release + #7/#8 prerequisites
+ #18/#19/#20 at local integration `caf7417be83f225c93e426f0885b935cefcc388c`.
The unified completion transition found an additional traceback-retention bug:
storing and raising the same timeout exception retained its future/CUDA event.
`2014fbb2271eae516b9f2c61dd0098601e99bd19` raises a fresh timeout while
keeping the terminal sentinel traceback-free. The union passed 124 CPU tests
(13 skipped) and 137 CUDA-visible tests warm. Real round-trip and
outage/recovery gates remain before peer review; nothing has been pushed.

### D. vLLM/full appliance

Owner: main agent.

Target: refresh current `dev/gilded-gnosis` + #210, re-run focused gates, then
use the model already present on the rental to measure the full #210 capacity
sweep while the independent source fixes are prepared.

Checkpoint (2026-07-31 03:30 UTC): current
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
