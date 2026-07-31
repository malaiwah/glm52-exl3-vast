# vLLM semantic PCIe channel companion review

Verdict: **APPROVE for atomic integration with a corrected SparkInfer
successor only**

- Candidate: `a89816a3a1062a2ae82e9ac87abd67e2bdcdfe98`
- Base: `5a7ac1481e1fc24b0bbc35efe160b2ff34797bee`
- Exact candidate was clean; `diff --check`, `py_compile` on all 17 changed
  files and Ruff 0.15.12 lint/format passed.

The review covered V2 target profile/production IDs; autoregressive
MTP/Eagle/Gemma draft prefill/decode profile/production IDs; DFlash/DSpark
profile/production IDs; V1/GLM target/draft/encoder profile/production IDs;
target-only V1 FULL/non-spec capture; and matching semantic IDs across TP/PP
custom all-reduce and DCP pools.

V1 PIECEWISE ordering is sound: target graphs capture without the drafter,
then replay their embedded target-channel kernels on the same outer stream
immediately before draft capture. This preserves target-output dependency and
stream order while draft communication binds to its distinct channel. V1
profiling checkpoints before capture; graph/encoder wrappers are destroyed and
profiling KV cleanup synchronizes before rollback in an inner `finally`,
including capture failure. V2 has equivalent rollback.

The old SparkInfer candidate does not implement the required `channel_id=`
contract and remains rejected. Linux focused tests plus real target/draft graph
capture and replay are promotion gates for the eventual atomic pair.
