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

### B. SparkInfer W4A16 scratch lifetime

Owner: `w4a16_scratch_fix` subagent.

Target: precise initialization/lifetime correction for the #100/#102 ordered
NaN, preserving the 64.91 MiB/rank planner saving and avoiding broad arena
zeroing unless measurement proves it harmless.

### C. LMCache future lifecycle

Owner: `lmcache_lifecycle_fix` subagent.

Target: compose retained CUDA event resources with timeout, idempotent
completion, `_expire`, late-response safety, and exact-once resource release.

### D. vLLM/full appliance

Owner: main agent.

Target: refresh current `dev/gilded-gnosis` + #210, re-run focused gates, then
use the model already present on the rental to measure the full #210 capacity
sweep while the independent source fixes are prepared.

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
