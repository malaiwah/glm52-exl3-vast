# Release notes and pins

The README carries only a current-state summary; this file records the release
lineage and exact pins that used to open the README.

## GLM-5.3 full-model 3.42bpw

The appliance now live-qualifies `MODEL_PROFILE=glm53-3.42bpw` for
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@8bef807a0fcdd180e984a26b50e731cdba9a8ff2`
on four RTX PRO 6000 Blackwell 96 GiB GPUs. The complete `glm_moe_dsa`
structure matches GLM-5.2, so the profile retains its TP4/DCP4 sparse-MLA
topology, online shared-expert K6, per-layer mixed K3/K4 routed experts,
native probabilistic MTP3, dynamic-NVFP4 KV, 3,072-token scheduler, C8,
GMU 0.93, and bounded LMCache tier. GLM-5.3-Flash settings remain isolated
from this full-model family.

Release merge `3ff9c7207c3c1754cb9665f0099316ec09056b8c` published
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:aec4075e02241b71321b5601763eba35e86ea02b13c3c3c43ac34fae30161160`;
`latest` resolves to that digest. An exact-image boot with the reconciled
checkpoint mounted read-only and no source-code bind mounts reached the
verified serving phase with zero restarts. Default API authentication rejected
an unauthenticated request with 401, accepted authenticated models and chat
requests with 200, and the dashboard enforced its persisted token gate.

The 81 weight shards matched the checkpoint's published hashes. The upstream
`MANIFEST.sha256` has three stale non-weight entries (`.gitattributes`,
`README.md`, and `config.json`); their actual Git object ids match the pinned
Hugging Face revision, so this is upstream manifest drift rather than download
corruption. Unlike willfalco's shared-H GLM-5.2 payload, this release stores
per-expert rank-sliced transforms: every routed layer has 148 K3 and 108 K4
experts, including native MTP layer 78.

The immutable r28 base needed six additional overlays, bringing the
fail-closed runtime manifest to 27 entries. Qualification exposed and fixed
FP8-RoPE leakage into B12X import probes, GLM-5.3-Flash environment leakage,
full-attention layers being forced through sparse-output validation, mixed
compiler API drift, a non-coherent mixed-kernel/route-pack pair, overly narrow
fused-kernel admission, early GLM-5-Next imports on the old architecture, and
undersized graph-owned Trellis scratch. Mixed-K uses an isolated coherent
kernel and route pack rather than changing the uniform-K6 path. The final
`exl3_patched.py` payload SHA-256 is
`ffef5aea103117a1bfb0023a43a59fba15b704566ad5acb0ddc47a18b9acede4`;
persistent JIT paths use namespace `turnkey-glm53-runtime-o27-v3`.

The inherited GLM-5.2 520,192-token/4.21-GiB KV envelope passed startup,
short/structured checks, and 32K retrieval, but the first temperature-1
1,024-input/512-output sampler request OOMed in top-p selection with only
3 MiB physically free on one rank. The qualified profile instead pins
3,415,867,392 KV bytes per GPU: exactly 393,216 logical KV tokens and one
maximum-length request. Model loading used 82.42 GiB/rank, graph capture used
0.61 GiB/rank, and two-second `nvidia-smi` sampling retained at least 665 MiB
physical free memory after first-use compilation and C8 sampling.

Unique-prefix prefill measured 2,465 / 2,444 / 2,380 / 2,282 tok/s at
8K / 32K / 64K / 128K. Aggregate temperature-1 decode for
1,024-input/512-output requests measured 60.40 / 153.93 / 227.55 tok/s at
C1 / C4 / C8, with mean acceptance lengths 3.41 / 3.45 / 3.51 and draft-token
acceptance of 80.45% / 81.57% / 83.63%. All 72 requests completed without
failure, preemption, or GPU/LMCache prefix reuse.

The startup arithmetic, factual, instruction, strict structured-output, and
32K three-depth retrieval gate passed. The independent OpenAI feature suite
passed tokenize, thinking on/off, streaming usage, preserved-thinking
multi-turn, strict JSON with thinking, tool choice, tool calls, and tool-result
continuation. A five-depth matrix retrieved all 15 facts from tokenizer-exact
131,407-, 261,192-, and 389,959-token documents. The post-stress short and
structured gate passed; a 3,469-line ready-state log audit found no post-ready
compile, structured-FSM, CUDA, distributed, process-failure, or error-level
findings.

## GLM-5.3-Flash K8 (quality-max)

The appliance now live-qualifies `MODEL_PROFILE=glm53-k8` for
`malaiwah/GLM-5.3-Flash-TR3-8bpw@b5ef443adce36ba5a10f2d5aa682fc9f2f0d0fae`
on four RTX PRO 6000 Blackwell 96 GiB GPUs. It retains the K6 topology and
correctness envelope—TP4/DCP4 A2A, B12X sparse MLA, Triton MoE, calibrated
NVFP4-DS MLA KV, MTP off, C8, GMU 0.93, and a 458,752-token request limit—but
bounds the scheduler and EXL3 parity arena at 512 tokens and enforces eager
execution.

The first K8 launch correctly failed closed because B12X's fused Trellis path
admits only integral K3/K4/K5/K6. Widening that admission would be wrong: eight
overlapping 16-bit MCG windows span 72 bits, beyond its two-word decoder. The
fallback initially produced repetitive corrupt output because its extension
calls still passed the old hard-coded K3 bitrate. The qualified overlay passes
the checkpoint's actual K8 bitrate to ExLlamaV3's compiled K8 MoE kernel and
does not load the fused-MoE preparation APIs for K8. The fail-closed overlay
payload SHA-256 is
`5e94629db2111aced6e3407addda85af294a4343a5b7258e0ca568e34626182c`.

The packaged candidate passed arithmetic, factual, instruction, strict
structured-output, and 32K tokenizer-exact retrieval startup gates. It loaded
76.31 GiB of model tensors per rank. Per-GPU profiling reported
78.94–78.97 GiB for weights plus non-torch allocations, 2.25 GiB peak
activations, zero graph memory, and 7.10–7.14 GiB KV. The engine exposed
6,610,733 logical KV tokens, or 14.41 maximum-length requests.

Unique-prefix K8 prefill measured 2,684 / 2,825 / 2,938 / 2,986 tok/s at
8K / 32K / 64K / 128K. Aggregate target-only decode at C1 / C4 / C8 measured
10.29 / 38.79 / 75.66 tok/s with a 256-token input,
8.42 / 23.43 / 28.46 tok/s at 32K, and
5.42 / 12.05 / 13.25 tok/s at 128K. All 72 measured decode requests completed
without failure, preemption, or prefix reuse.

Two independent 448K trials each built a tokenizer-exact 449,461-token
document and retrieved all three facts in 170.775 and 175.086 seconds. A
post-stress short and structured-output gate also passed. K8's panel KLD is
0.012384 versus K6's 0.013723, but it adds about 77 GB / 30% checkpoint bytes,
uses about 15.2 GiB more non-KV memory per GPU, gives up about 14.4 GiB KV per
GPU. K6 is 5.3–7.3× faster at short context and 8.3–24.4× faster when the
measured 32K/128K prefill cost is included. K8 is a qualified quality-max
alternative; K6 remains the production default.

## GLM-5.3-Flash K6 (production default)

The appliance now provides `MODEL_PROFILE=glm53-k6` for
`malaiwah/GLM-5.3-Flash-TR3-6bpw@be51877455a8786ebdd5f96053aff6dc74a0996f`.
It pins the immutable parent
`verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
and fail-closes on the before/after SHA-256 state of 21 runtime overlays copied
from the live-qualified service. The profile is one contract: TP4/DCP4 A2A,
EXL3 K6, B12X sparse MLA, Triton MoE, calibrated NVFP4-DS MLA KV, native
prefix caching, no speculation, a 3,072-token scheduler, eight sequences,
GMU 0.93, and a conservative 458,752-token request limit.

On four RTX PRO 6000 Blackwell GPUs at JarvisLabs, the final packaged-image
boot exposed 20,043,933 logical KV tokens (43.69x the advertised request
limit), with 21.52--21.56 GiB of KV memory per GPU. Per-rank memory profiling
reported 63.74--63.78 GiB for weights plus non-torch allocations, 3.02 GiB
peak activations, and 0.45--0.46 GiB of CUDA graphs. The appliance's short,
structured-output, and 32K retrieval startup gates passed.
Measured unique-prefix prefill was 2,983 tok/s at 8K and 4,322 / 4,637 / 4,907
client-observed tok/s at 32K / 64K / 128K; server accounting at those three
lengths was 5,238 / 5,326 / 5,326 tok/s. Aggregate target-only decode at
zero context measured 75.15 / 241.31 / 397.19 tok/s at C1 / C4 / C8; at
128K it measured 64.72 / 223.01 / 323.64 tok/s.

Context qualification rejected the provisional 520,192-token limit. Retrieval
passed at 384,612 tokens and in two independent 448K trials measuring 449,461
and 449,462 document tokens, with all three facts found each time. A 480K trial
exhausted both 2,048- and 4,096-token answer budgets; a concurrent 505K stress
trial caused persistent degenerate output until restart.
The shipped 458,752-token request envelope leaves about 9K tokens beyond the
longest passing document for template, query, and generated tokens.

The parent does not ship the legacy GLM-5.2 static scale path. The appliance
therefore downloads the immutable metadata-corrected public sidecar at build
time and verifies SHA-256
`ac68fe6af3056ec35299361293c9ae568769d21696756548493f67ff17881ece`;
its numeric calibration payload is unchanged. Persistent JIT paths have a new
`turnkey-glm53-k6-o21-v1` namespace so older GG-v20 artifacts cannot leak into
the new runtime.

## GG v20-r28 (previous)

The appliance pins immutable GG v20-r28 manifest
`sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0`.
r28 retains r26's lossless TP4/DCP4 automatic policy and adds the complete
`shared_h_v1` plus runtime-dynamic mixed-Trellis contract needed by
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw@a350292c...`. The build fails closed on
the exact vLLM `e1e9426`, SparkInfer `200c1db`, and LMCache `9a05c88` source
trees before applying the serial-MTP warning overlay.

AIBeast qualified the online-K6/dynamic-NVFP4 profile with TP4/DCP4/MTP3,
eight sequences, exactly 2,032 blocks / 520,192 logical KV tokens, a
3,072-token scheduler/EXL3 arena, and 125 GiB DRAM plus bounded 512 GiB NVMe
LMCache. The natural 524,800-token pool was rejected after a first-request
128K OOM; the pinned pool passed C8, all API/tool/structured-output gates, and
45/45 salted five-depth needles through a 516,096-token prompt with a 4,096
token reserve. Matched PP was 2,367 / 2,263 / 2,137 tok/s at 3K/32K/128K.

The profile pins probabilistic MTP3 proposals. A matched greedy arm improved
C1 decode from 60.5 to 76.7 tok/s but reduced aggregate C4/C8 throughput from
149.4/173.5 to 125.5/153.5 tok/s and lowered acceptance at concurrency. MTP5
was rejected because its 48-wide graph/trellis contract consumes the execution
margin required by the workload-safe 520,192-token pool.

Matched one-window KLD was 0.082039 with native weights and dynamic NVFP4,
0.089888 with online K6 and dynamic NVFP4, and 0.077949 with online K6 plus
FP8 KV/BF16 RoPE. FP8 remains experimental because its 656-byte record reduced
the practical context envelope to about 295K; production keeps the 368-byte
dynamic-NVFP4 record for the 520K Hermes Agents requirement.

## GG v20-r26 (previous)

The appliance pins the immutable GG v20-r26 manifest
`sha256:c7a202cf3ccd155973a151235acb9677aa98f61765372f839bb0c193ff594ec4`.
r26 repairs TP4/DCP4 automatic prefill policy: that topology has one query
partition, so exact query split and full CKV gather use two indexer shards with
owner merge disabled. The entrypoint calibration contract now passes owner and
indexer selection as `auto`, rather than silently retaining its earlier
zero-shard value, and the 3.36-bpw profile uses the measured PCIe DMA crossover.

On AIBeast, the NFS-backed 3.36-bpw/K6/MTP3 production shape retained exactly
524,288 GPU-KV tokens and measured 2,453--2,458 / 2,350--2,370 /
2,197--2,238 tok/s unique-prefix PP at 3K/32K/128K. It passed every API and
structured-output gate plus 5/5 needles in an actual 522,359-token prompt with
no degeneration or OOM. The public 38% headline is valid against r25's old
automatic policy; the already owner-merge-off appliance gained about 2.5--4.4%
in production shape and 10.5--13.8% in a matched official-image old-auto A/B.
Full evidence and the interpretation boundary are in
[docs/glm52-r26-3.36-qualification.md](docs/glm52-r26-3.36-qualification.md).

## GG v20-r25 (previous)

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
