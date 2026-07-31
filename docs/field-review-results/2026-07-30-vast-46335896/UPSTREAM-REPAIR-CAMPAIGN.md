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
`3eb363e` parameterizes the same-binding replay over `m=[1,3]`, forces and
asserts the native direct path, and is awaiting its cold/warm Vast gate.
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
Explicit capacity 3,072 is now running from a fresh JIT cache; 1,536/1,024/512
follow. The installed PCIe calibration helper also exposed a separate
reliability bug when `LD_PRELOAD` is unset; its upstream source/fix is being
tracked independently.

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
