# Release notes and pins

The README carries only a current-state summary; this file records the release
lineage and exact pins that used to open the README.

## GG v20-r14 (current)

The appliance now pins **GG v20-r14**: the reviewed r13 vLLM/LMCache/XGrammar
stack plus SparkInfer's native mixed-K K3/K4 EXL3 path for the 3.25-bpw
checkpoint. It retains r9's paired dynamic-token NVFP4 MLA cache ABI and exact
adaptive sparse-indexer folding, while consolidating EXL3 on SparkInfer's
fused-MoE API, fixing target/draft small-row plans and using a repeatable
post-warmup Trellis arena peak for KV sizing. The flagship EXL3 profile uses the complete
dynamic record after a reproduced mean KLD of `0.1167701185`, repeated
near-maximum retrieval gates, and an exact 522,360-token five-depth pass.
The reviewed static GLM-5.2 scale artifact remains available for variants
that have not qualified the dynamic record. XGrammar 0.2.5 fixes GLM
`tool_choice=required` termination.
The provider TLS helper is refreshed to Lego 4.35.2, the latest v4 maintenance
release. Lego 5 is intentionally deferred because it changes CLI and account
storage semantics and needs its own certificate-renewal migration test.

Base runtime image:
`voipmonitor/vllm@sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2`
(pinned GG v20-r14). The release tag records composed vLLM tree `749050e`
(including the calibrated NVFP4 path, query-split prefill,
repeatable activation profiling, and
[PR #190](https://github.com/local-inference-lab/vllm/pull/190)),
composed SparkInfer tree `8110e3e` (including the exact fixed-capacity
Trellis arena from
[PR #92](https://github.com/local-inference-lab/sparkinfer/pull/92)), and
the native mixed-K integration from
[SparkInfer PR #104](https://github.com/local-inference-lab/sparkinfer/pull/104).
It retains FlashInfer `801d57a`, DCP-aware LMCache and XGrammar 0.2.5. The
image-owned EXL3 parity compatibility patch remains an explicit
cache/requalification boundary. It also includes native vLLM support for
`Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen parsers, and MTP
speculative decoding.

The r13/r14 lineage pins exact reviewed heads rather than following their
moving branches. The r13 post-release documentation correction changed only the stock-r11
comparison: matched MTP0 decode is `44.66` tok/s on stock r11 versus `48.61`
on r13 (`+8.85%`); the registry image and source locks did not change.
For filesystem cache users,
[LMCache PR #4211](https://github.com/LMCache/LMCache/pull/4211) documents a
silent mixed-object-size L2 store failure in its native multiprocess adapter.
Our exact GLM filesystem restore passed, but NVMe remains opt-in and bounded;
watch LMCache L2 store/error metrics and verify a cold restart restore before
depending on it. The appliance also retains the hard capacity posture tracked
by [vLLM PR #165](https://github.com/local-inference-lab/vllm/pull/165), so a
cache cannot silently consume the whole local RAID0 device.

### Profile checkpoints

- GLM: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` at `9297b9f1…`
- higher-fidelity GLM option: `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` (mixed
  3/4-bit experts) at `d7d79c2d…`, the revision pinned by the shipped config
  and qualified by the r14 field repair
  ([docs/glm52-r14-maintenance-plan.md](docs/glm52-r14-maintenance-plan.md));
  the earlier `61d2b6b7…` qualification is recorded in
  [docs/glm52-3.25-offload-qualification.md](docs/glm52-3.25-offload-qualification.md)
- MadeBy561 control: `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` release bundle
  `66f3623…`; its 184 weight shards remain the immutable `68babde2…` payload
- Qwen: `nvidia/Qwen3.6-27B-NVFP4` at `0893e160…`
