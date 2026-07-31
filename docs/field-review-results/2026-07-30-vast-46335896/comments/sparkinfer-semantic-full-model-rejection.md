Superseding field result: **the frozen semantic-channel successor is not yet
deployable**.

The focused four-rank eager/capture/torture and atomic vLLM/SparkInfer suites
all passed, but the first full GLM-5.2 startup gate found a contract the focused
tests missed. During `determine_available_memory()`'s uncaptured warm-up, a
compiled TP all-reduce requested `vllm:eager:allreduce` on a physical CUDA
stream that was already owned by the active `vllm:target:profile` semantic
scope. Its descriptor warm-ups run before CUDA itself reports the stream as
capturing, so SparkInfer did not apply its active-scope override, selected the
static eager channel, and correctly refused to alias the two channels:

```text
RuntimeError: CUDA stream key ... is already bound to another logical PCIe oneshot channel
```

The engine failed before KV allocation/health, so there is no performance or
deployability claim for SparkInfer `31fa6a4` + vLLM `be1e289`. The SHA-only
appliance image containing them will not be promoted.

The complete call trace makes the repair direction deliberately narrow:

1. while a semantic capture scope is active, route a pre-capture warm-up on
   that scope's owner stream to its top logical channel;
2. do not extend that routing to unrelated side streams, and retain hard
   failures for genuine out-of-scope stream/channel mismatches;
3. apply the rule symmetrically to one-shot and DCP pools; and
4. make failure-path teardown idempotent when Torch has already removed the
   process group from the world-group map.

Re-qualification will include the same full-model boot/traffic/shutdown gate,
not just focused tests.

Complete immutable server log and analysis:

- [full rejected log](https://github.com/malaiwah/glm52-exl3-vast/blob/8992ce37b77bf7b3cb04b17084e89954ceea2e5a/docs/field-review-results/2026-07-30-vast-46335896/artifacts/field-review-final-4c880eb-mtp3-c8-v1-server.log)
- [campaign section L](https://github.com/malaiwah/glm52-exl3-vast/blob/codex/field-review-combined/docs/field-review-results/2026-07-30-vast-46335896/UPSTREAM-REPAIR-CAMPAIGN.md#l-full-model-gate-rejects-the-frozen-pcie-successor)
