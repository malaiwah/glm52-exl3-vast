# Release notes and pins

The README carries only a current-state summary; this file records the release
lineage and exact pins that used to open the README.

## GG v20-r25 (current)

The appliance now pins the immutable GG v20-r25 manifest
`sha256:042936fd8d9e4c2aa579ab9b736dd0a2faf2678c6ba36bf4dfce7db566c6fd11`.
It contains SparkInfer
[#117](https://github.com/local-inference-lab/sparkinfer/pull/117) at
`cfeee9b42d21c19a74d85ed5576f8387168df53c`: mixed-Trellis expert counts are
runtime artifact data, so one compiled contract safely serves both 3.36-bpw
partitions (206/50 at layer 3 and 160/96 at layers 4–77). The appliance no
longer carries its superseded mixed-tier cache-key overlay and instead fails
closed on the exact r25 source hashes.

AIBeast independently passed TP4/DCP4/MTP3 startup, graphs through 32,
2,363/2,285/2,144 tok/s unique-prefix PP, 100.7/162.2/240.1/297.0 tok/s
aggregate C1/C2/C4/C8 decode, strict structured output, and 5/5 needles in an
actual 521,275-token prompt. Production retains dynamic-token NVFP4 KV because
the matched FP8 alternative, despite improving KLD from 0.08251 to 0.06867,
needed a 512-row arena and lost 18–21% PP and 22.9% C8 throughput at its safe
512,000-token shape. Full evidence is in
[docs/glm52-r25-3.36-qualification.md](docs/glm52-r25-3.36-qualification.md).

The remaining turnkey overlay suppresses vLLM's misleading sub-8192 scheduler
warning only when the calculated speculative slot delta is zero, as it is for
serial GLM MTP. Validation remains unchanged and genuinely slot-consuming
speculative methods still warn.

## GG v20-r17 (previous)

The appliance now pins **GG v20-r17**. vLLM #222 natively supersedes the
appliance's former #210/#219 overlays: the one-grid mixed K3/K4 path remains
active for decode and small M, while large-M prefill uses bounded serial
homogeneous K3/K4 plans with FP32 tier accumulation. SparkInfer #105 packages
and verifies the runtime-compiled PCIe extension's local headers, preventing
the silent PyNCCL fallback seen in the pre-release candidate. The runtime also
retains r9's paired dynamic-token NVFP4 MLA cache ABI, adaptive exact sparse-
indexer folding, DCP-aware LMCache, and XGrammar 0.2.5. The flagship EXL3
profile uses the complete
dynamic record after a reproduced mean KLD of `0.1167701185`, repeated
near-maximum retrieval gates, and an r17 exact 521,276-token five-depth pass.
The reviewed static GLM-5.2 scale artifact remains available for variants
that have not qualified the dynamic record. XGrammar 0.2.5 fixes GLM
`tool_choice=required` termination.
The provider TLS helper is refreshed to Lego 4.35.2, the latest v4 maintenance
release. Lego 5 is intentionally deferred because it changes CLI and account
storage semantics and needs its own certificate-renewal migration test.

Base runtime image:
`voipmonitor/vllm@sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727`
(pinned GG v20-r17). The release tag records composed vLLM tree `db29328`,
composed SparkInfer tree `b2bff71`, and reproducible build `f5ba50b0`. Its
reviewed changes include native mixed-K
[vLLM PR #222](https://github.com/local-inference-lab/vllm/pull/222), the
complete PCIe wheel and lifecycle repair in
[SparkInfer PR #105](https://github.com/local-inference-lab/sparkinfer/pull/105),
and the fixed-capacity Trellis foundation from
[SparkInfer PR #92](https://github.com/local-inference-lab/sparkinfer/pull/92).
It retains FlashInfer `801d57a`, DCP-aware LMCache and XGrammar 0.2.5. The
image-owned EXL3 parity compatibility patch remains an explicit
cache/requalification boundary. It also includes native vLLM support for
`Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen parsers, and MTP
speculative decoding.

The r13-r17 lineage pins exact reviewed heads rather than following their
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
  and qualified on the native r17 path
  ([docs/glm52-r17-maintenance-results.md](docs/glm52-r17-maintenance-results.md));
  the earlier `61d2b6b7…` qualification is recorded in
  [docs/glm52-3.25-offload-qualification.md](docs/glm52-3.25-offload-qualification.md)
- MadeBy561 control: `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` release bundle
  `66f3623…`; its 184 weight shards remain the immutable `68babde2…` payload
- Qwen: `nvidia/Qwen3.6-27B-NVFP4` at `0893e160…`
