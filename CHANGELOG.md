# Release notes and pins

The README carries only a current-state summary; this file records the release
lineage and exact pins that used to open the README.

## Unreleased — full GLM-5.3 3.42bpw 520K, measured AIBeast optimization

### General-build integration

The normal Dockerfile installs the selected B12X #226/#256/#257 and
V2-compatible fairness Python sources after the existing GLM refresh, with
fail-closed source hashes and final-stack provenance. Historical experiment
payloads remain immutable; the archived host entrypoint and experimental MTP
constraint changes are not installed.

Registered `PREFILL_FAIRNESS_ENGINE` (`off` / `compute_share`) and
`PREFILL_COMPUTE_SHARE` controls use the existing configuration precedence and
serving-argument path. Fairness remains off by default; explicitly enabled
fairness requires cadence 1 and a valid strictly-between-zero-and-one share.
Public rental defaults and AIBeast's running deployment are not changed by
this source integration. The earlier GPU receipts qualify the selected
experimental image, not automatically every newly built appliance image.

### Selected optimization (2026-09-08)

The authorized controlled campaign selected a local image built from immutable
`8006d209…` with B12X #226/#256/#257 and V2-compatible measured prefill-service
fairness. The host uses a **0.6 prefill-service share**, a **2048-row EXL3 arena**,
the unchanged **3072-token scheduler**, TP4/DCP4, native probabilistic MTP3,
fixed KV bytes and **520,192-token limit**. It advertises **`GLM-5.3` and
`local-primary`**, not the obsolete GLM-5.2 alias. Public rental defaults,
weights, precision, power cap and reboot/restart policy are unchanged.

Matched 500K-plus-overlap arms measured **613–669 MiB free/GPU** at 2048 versus
101 MiB at 3072, for **6.7%** median 64K prefill-delay cost. The 1024 arm gained
more margin but cost **11.0%**, exceeding the chosen 10% target. The active V2
MTP proposal-constraint prototype passed native CUDA graph/rejection checks but
was not adopted because a reliable rate improvement was not established.
The first V1 proposer edit was inactive on this V2 runtime; a forced-V1 fairness
startup failed and rolled back. These corrections and all measured arms remain
in the [optimization evidence](TEST_RESULTS.md#controlled-aibeast-optimization-2026-09-08)
and [source-locked recipe](maintenance/glm53-optimization-20260908/manifest.json).

### Initial cutover and preservation (historical)

**Repository preservation (2026-09-08):** reconciled the August 30 memory
postmortem and August 4 r26 research plan as explicitly historical documents
on the published AIBeast release line. Archived four workspace notes and eleven
K8 receipts with hashes and credential redaction; see the
[preservation and runtime audit](TEST_RESULTS.md#workspace-preservation-and-runtime-capacity-audit-2026-09-08).
The local runtime-capacity candidate remains preserved as open B12X PR #256;
read-only inspection found it absent from the running r34 image. This
documentation integration changes no runtime source or production settings.

**2026-09-08: the authorized AIBeast cutover completed.** Full non-Flash
`MODEL_PROFILE=glm53-3.42bpw-500k`, architecture family `glm52`, variant
`exl3-tr3-glm53-3.42bpw-500k`, now serves **520,192 total tokens** at the
old GLM-5.2 resource policy on port **8000**. `GLM-5.2` and `local-primary`
aliases are preserved, with `GLM-5.3` added; `AUTH=none` and `LANDING=0`
remain the old local-client contract. This is not the authenticated rental
default. Two >=500K retrieval trials and a real-client soak passed; the
**full maintenance, performance and quality matrix remains open**.
No global `latest`/`main` promotion or boot-policy change occurred, and
`restart=no` remains unchanged.

The [active receipt](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/active.json)
pins image
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:8006d209b8f1d1bbf815983514e430fb77bbf01bd66075578483473d9310416a`,
source `499d34e`, container `8273cb4d…`
(`glm53-turnkey-r34-parity-20260908t015846z`).
The host manual-NVIDIA-device driver-injection hook was corrected in
`scripts/run-local-podman.sh`, not by runtime-source hot patches. The first
successful resource-parity boot started **02:13:47 UTC**, ready **02:29:15**.
It retrieved **3/3 facts from 501,098 exact haystack tokens in 336.364 s**,
then passed post-stress correctness and **512-token temperature-1 sampling
in 19.344 s** ([initial receipt](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/needle-initial-parity.json)).

At **02:58:58 UTC**, a deliberate restart restored old explicit B12X settings
through host `TUNE_` overrides: legacy `SPARK_*` names did not control the
installed B12X fold budget. **Auto / 64 MiB** now applies instead of the
absent-setting default 256 MiB. Other restored controls match defaults or are
likely inert for GLM; the mHC 3072 threshold is not a demonstrated GLM speedup
([restoration](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/b12x-parity-restoration.json),
[installed effective policy](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/b12x-effective-policy.json)).
The durable source fixes in this checkout are **not inside the running
`8006d209…` image**; host overrides make its effective B12X policy correct now.
The final boot retrieved **3/3 facts from 501,099 exact haystack tokens in
275.874 s**, followed by correctness checks and **512-token temperature-1
sampling in 7.701 s** ([final receipt](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/needle-final-parity.json)).
Counts exclude the chat template and question; the probes reserved 4096
tokens. These timings are not a causal A/B: warmth, compilation and load differ.

The **03:36:34 UTC accepted soak** recorded **64 healthy / 0 failed samples**,
**31.55 minutes** since the first external client, **87 external POST 200
headers**, **96 completed engine requests including probes**, and **0 engine
error requests** across real Chat Completions and Responses traffic. No OOM,
engine crash or unexpected restart was observed. Minimum sampled raw free
VRAM was **215 / 245 / 217 / 215 MiB** on GPUs 0–3. **215 MiB is thin
headroom**, sampled about every 30 seconds, not a continuous worst-case margin
or safety guarantee ([soak receipt](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/soak-accepted.json)).
This short soak does not supersede the user's approximately 24-day stable
GLM-5.2 service history.

At **03:37:31 UTC**, accepted cleanup removed only the old
`/mnt/fast/build/r34-aibeast-maintenance-20260815/runtime/lmcache` and
`/compile-cache` directories, reclaiming **370,534,412,288 bytes (345.09 GiB)**;
health remained HTTP 200. Old image/container, weights, state and logs,
new GLM-5.3 caches/state and cachefilesd were preserved. **Fast warm-cache
rollback was intentionally relinquished**; restoring the old service requires
regenerating removed caches ([cleanup receipt](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/old-cache-cleanup.json)).

Checkpoint pin:
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@99c6f951333d2b38f1efefa533c7afadf0d376e3`.
All 81 LFS weight SHA-256/size pairs match independently evaluated
`8bef807a0fcdd180e984a26b50e731cdba9a8ff2`; metadata and the chat template
changed. The AIBeast [integrity read](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/weight-integrity.json)
verified **81/81 files / 355,150,499,456 bytes** and primed cachefilesd.
A read-only local metadata-only derivative reconciled the native-MTP
`model.layers.78.eh_proj*` ignore entry; canonical weights remain unmodified
([source/derived hashes and change](maintenance/glm53-aibeast-500k/evidence-aibeast-20260908/metadata-reconciliation.json)).
The earlier spot feature suite passed, but the complete parser/continuation
matrix remains open. The unchanged public-profile rental floor uses TP4/DCP4,
dynamic NVFP4 KV/FP8 RoPE, 4,518,907,904 KV bytes/GPU, scheduler 2048,
prefill arena 1024, C8, native probabilistic MTP3, online K6 and GMU 0.93;
successful retrieval is not C8 capacity proof. No 750K claim is made.
The previous 393,216-token full-model qualification is historical and does
not transfer to the refreshed Gilded image. GLM-5.2 is an explicit alternative.

Exact safetensors-header accounting corrects the initial lower-bit preference:
5.3 3.42bpw adds 0.848 GiB/rank versus the preserved 5.2 3.42bpw tensors, largely
rotation-vector storage. Their BF16 carrier is identical. Preserve 3.42bpw
and 520,192 tokens; this accounting does not prove that GLM-5.3 requires
smaller workspaces. The optional 3.25bpw experiment is not a fallback here.
The full accounting is retained in `maintenance/glm53-aibeast-500k/memory-comparison.json`.

AIBeast maintenance followed the user's explicit **parity-first** request,
not a conservative-floor first attempt. `MAINTENANCE_TRIAL=parity` remains the
maintenance default, overriding the unchanged public profile defaults using
the [preserved GLM-5.2 environment and serving argv](maintenance/glm53-aibeast-500k/production-baseline.json):
12 sequences, scheduler/prefill 3072/3072, GMU 0.95, maximum CUDA graph capture
48 (sizes 4,8,12,16,20,24,28,32,36,40,44,48), Trellis maximum M 48 and
125 GiB initial LMCache RAM. Only after an operator observes and records
failure or insufficient margin may `MAINTENANCE_TRIAL=rental-floor` select
8 sequences, 2048/1024, GMU 0.93, graphs 32 (sizes 4,8,12,16,20,24,28,32),
Trellis 32 and 20 GiB initial RAM. There is no automatic shrink.
Both preserve the same checkpoint/3.42bpw, 520,192 context, TP4/DCP4,
interleave 64, native probabilistic MTP3, KV 4,518,907,904 bytes/GPU,
125 GiB RAM ceiling and 384 GiB disk tier. No token or precision reduction.
The fallback default name adds `-rental-floor`; each trial has isolated
caches, state and stage manifest, with the selector in stage identity.

Parity required a **new image build**, since the `200b1841…` validator rejects
3072 scheduler/prefill settings. That digest's GPU evidence remains floor-only;
the later `8006d209…` image has its own AIBeast boot/retrieval/soak evidence
above. The rental-floor arm was not substituted during this cutover.

Read-only research found no demonstrated dominating stack under the same
full-model, four-GPU, 520K/C12 and quality constraints; “near the practical
constrained frontier” is an inference, not Pareto-optimality proof.
Independent evaluations support gains on some tasks, but exact 3.42bpw
gain retention is unproven and the published GPQA result supplies contrary
evidence. No new KLD, paired task-quality or broad performance experiment was
run as part of that review. The [dated deployment and constrained-frontier
appendix](docs/glm52-prefill-optimization-research-2026-08-09.md#2026-09-08-glm-53-deployment-and-constrained-frontier-review)
links primary evidence and distinguishes proposed investigations from
completed deployment gates.

The runtime parent is now local-inference-lab's **Gilded Gnosis v20 r34**,
`docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`
(vLLM `e2666d9a65f41fc376607531453cbd57c4c71016`, integration tree
`4d006a43928cdee01306691a766542c1e9bebb59`; B12X/SparkInfer
`7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b`, tree
`cd3ce190f0f1917402cdfd5773724267cc9a63f8`; LMCache `0.5.2+glm52dcp.4`;
FlashInfer `1ac6942776b383c6b03c7a5805a22e72a3e3349f`; NCCL 2.30.4;
ExLlamaV3 encoder `704aefd743b390af4bd0fb429d1906f9b964c7d8`; Torch
2.12.0+cu132; CUDA 13.2.1). This replaces the third-party
`verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
parent used by the preceding candidate, and returns the appliance to the exact
runtime family that has been serving GLM-5.2 in production. Its CUDA/Torch/NCCL
stack is 13.2.1 / 2.12 / 2.30.4, not the Verdict candidate's 13.3 / 2.13 / 2.31.2.

The 27 VerdictAI-layout overlays are deleted from this build because their
targets and runtime layout differ. **The Gilded base does not already contain
all 27 fixes.** Required maintenance work is re-derived for its installed
sources, rather than treating overlay removal as equivalent qualification.
The selected SHA-256-pinned payloads in the importable runtime and `/opt/vllm`
source tree include the reviewed r34 maintenance sources (vLLM PR277
compatibility, hybrid external-cache invalid-block recovery, bounded LMCache
multiprocess retrieve) and the expired-L1-read-lease recovery patch
(`67561538a08ce3db621f51f0615b67537c0c8361`, upstream draft
LMCache/LMCache#4691) that makes the candidate's `LMCACHE_L1_READ_TTL=900` and
`LMCACHE_SESSION_TTL_SECONDS=5400` real rather than inert; the surgical parser
#639/#640 and scheduler #546 backports, re-derived against Gilded bytes
(the Gilded parser has no `TOOL_DIRECT_*` states or message-header buffering,
so those hunks were re-anchored, never back-ported from Verdict); and the
fail-closed LMCache retrieve-deadline/cache-layout payloads, re-pinned to the
Gilded connector `c6e0bf5c…` and adapter `0781f930…`.

**GLM-5.3-Flash is no longer served by this image.** `vllm.models.glm5next`,
`b12x.attention.glm_pooled_indexer`, `b12x.attention.gdn_decode` and
`vllm.model_executor.warmup.glm5_kpool_warmup` are absent from the Gilded base,
so `glm53-k6` / `glm53-k8` fail closed and name the VerdictAI-derived lineage
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:9d7ab60a3ad666edb8d38812ec6709909e6752a78fad464c842c7b17659f5d5b`
instead of silently substituting another profile. Missing runtime support in
this build does **not** prove that Flash cannot be ported. Qualification of
the separate Verdict reference does not qualify the refreshed Gilded image.

These source changes are not inherited upstream-head guarantees. Only the
specific images and observed spot/AIBeast gates have live evidence.
The parser repairs literal argument delimiters and stripped-stop recovery while
preserving the installed parser ABI. The non-DP scheduler fix does not by itself
enable prefill throttling: `prefill_schedule_interval` remains 1 (inert).
Any interval above 1 needs explicit overlap/progress qualification; DP's
synchronized counter behavior is unchanged.
The build-time `/opt/runtime-provenance.json` hashes actual installed sources
and native libraries and records package/module versions and registry pins;
the stale copied r26 ledger is no longer current-runtime evidence. The image's
own pushed digest must be recorded externally after publication.

Operator fixes include full variant precedence, refusing invalid startup
configs, warning for unqualified variants, and attempt-snapshot protection
against stale verification/rollback state. Podman 4.9 staging uses explicit
devices, separate state/cache/name/port, no automatic production stop and no
restart policy. Rollback preserves the old image, environment and state;
the accepted AIBeast cleanup intentionally removed its warm caches, as above.
LMCache gains an optional aggregate `LMCACHE_L1_MAX_GB` ceiling (0 by default;
candidate 125 GiB; initial arena 125 GiB parity / 20 GiB rental floor);
it does not enlarge active context.

Spot evidence is tied to
[`ghcr.io/malaiwah/glm52-exl3-vast@sha256:200b1841453b6a46c91f0b7a2866589cda7e52625b29590bb7a69fe90948e6c9`](https://github.com/malaiwah/glm52-exl3-vast/pkgs/container/glm52-exl3-vast),
source [`e96231359fc5dca2df9d4a397d7a814ba9deae30`](https://github.com/malaiwah/glm52-exl3-vast/commit/e96231359fc5dca2df9d4a397d7a814ba9deae30),
[CI 34168581945](https://github.com/malaiwah/glm52-exl3-vast/actions/runs/34168581945)
and [PR #58](https://github.com/malaiwah/glm52-exl3-vast/pull/58).
`skopeo` + verified `umoci` unpacked the rootfs; the provider blocks
mount/user namespaces, so this was an OS graft, **not an exact OCI launch**.
Final verification matched 19 critical sources, seven module initializers and
seven image-recorded native libraries. Additional provider NCCL 2.23.4 files remained
and were disclosed; live process maps showed image NCCL 2.30.4 loaded. These
are scoped observations, not comprehensive filesystem/security equivalence.
Later helper/source changes are not retroactively GPU-qualified.

Predecessor **483634** resumed as **500157**, with four spot GPUs (approximately
**$3.96/hour GPU quote**), 320 GiB shared memory and 1200 GB persistent home.
**500157 was paused after evidence capture**, confirmed by a fresh provider
listing, preserving old user storage that remained billable while paused.
Subsequently the user explicitly requested destruction: the destroy API
succeeded, `jl list` returned `[]`, and all six resource counts were zero.
The [new destroy receipt](maintenance/glm53-aibeast-500k/evidence-spot-20260908/destroy.json)
is separate from the unchanged historical pause receipt.
The 131,409-token prefix took 62.4 s then 1.3 s with native
GPU cache only. A second 501,098-token pressure probe retrieved 3/3 facts in
360.891 s; the original prefix then returned correctly in **3.115 s**, adding
**115,200 external-prefix hit tokens** and 15,872 native-hit tokens. This
measures **LMCache DRAM retrieval after GPU pressure**, not complete GPU
eviction or a DRAM-only request. Disk L2, restart, concurrency, fault recovery
and a new KLD comparison remain unproved by these receipts. See
[spot results and remaining gates](TEST_RESULTS.md#jarvislabs-spot-container-proof-2026-09-08).

Root/non-root (#20), host/bridge networking (#21), parent dependency index
selection (#24), and complete core-runtime hash locks (#25) remain open
hardening boundaries; scoped SOUL locks and loopback cache restrictions do
not close them. The parent declares `LicenseRef-ShapleyMCG-1.0`; the full
model uses the custom GLM-5.3 license. The appliance MIT license is not an
umbrella license. See [attribution and provenance](README.md#attribution-licenses-and-runtime-provenance).

The entries below are historical release evidence. Their “default” and
“latest” statements describe those releases, not this operationally accepted
AIBeast deployment or an automatic global release promotion.

## GLM-5.3 full-model 3.42bpw

### Post-qualification review hardening

An adversarial workflow review found that LMCache's ZMQ `--host` did not also
bind its separate HTTP administration server. The appliance now requires every
LMCache listener to remain on IPv4 loopback, passes `--http-host` explicitly,
and removes token, API-key, password, private-key, and cloud-access credential
variables from the LMCache subprocess while leaving the authenticated model
server's environment intact. A fail-closed image patch registers only
LMCache's read-only information router: `/env`, `/run_script`, and every cache,
quota, configuration, and backend mutation router are absent. Loopback access
from the unprivileged SOUL identity therefore cannot become root code execution.

The same review closed a bearer-key boundary in optional SOUL autonomy levels
2/3. A root launcher now removes stale processes for the dedicated `soul` uid,
passes the controller empty key/readiness pipes, and writes the key only after
the controller confirms `PR_SET_DUMPABLE=0`; the key never enters the SOUL
process environment. The launcher and PID 1 reap the whole uid on stop,
including detached shell sessions. The pinned Nanobot SDK parses provider
configuration through an anonymous non-inheritable descriptor, so no literal
provider key remains in the SOUL-owned runtime tree, process environment, or
argv. A pre-forked non-dumpable verifier worker inherits the key only in memory;
the controller terminates it at the deep probe's 600-second outer bound, then
exits so PID 1 can replace the controller/worker pair without a retry loop.

The authoritative startup verifier now includes a tokenizer-calibrated,
cache-unique 1,024-token temperature-1 request, requires all 512 output tokens,
and rechecks engine health afterward. A profile-form switch now discards
untouched values displayed from the old variant instead of persisting them
above the new variant's defaults, and rejects a stale rendered revision rather
than treating it as user intent. Known-good rollback re-minimizes the full
stored effective configuration against the current container's startup
environment, verifies exact reproduction of the family, variant, and every
applicable knob without comparing target-family-inapplicable defaults, and
refuses a topology that is invalid for the GPUs on the replacement host.

The GLM-5.3 full profile rejects the unsafe inherited KV/context envelope,
GLM-5.2 layer-78 grafts, and Flash-only runtime variables, including attempts
to reintroduce them through `TUNE_*`. The rootless Podman launcher no longer
injects a GLM-5.2 variant or served name into other public profiles. CI now
compiles all 27 runtime-overlay payloads and executes the launcher regression;
the runtime-log audit recognizes both SparkInfer and B12X compile records plus
explicit inference-JIT warnings.

The appliance now live-qualifies `MODEL_PROFILE=glm53-3.42bpw` for
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@8bef807a0fcdd180e984a26b50e731cdba9a8ff2`
on four RTX PRO 6000 Blackwell 96 GiB GPUs. The complete `glm_moe_dsa`
structure matches GLM-5.2, so the profile retains its TP4/DCP4 sparse-MLA
topology, online shared-expert K6, per-layer mixed K3/K4 routed experts,
native probabilistic MTP3, dynamic-NVFP4 KV, 3,072-token scheduler, C8,
GMU 0.93, and bounded LMCache tier. GLM-5.3-Flash settings remain isolated
from this full-model family.

Initial qualification merge `3ff9c7207c3c1754cb9665f0099316ec09056b8c`
published
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:aec4075e02241b71321b5601763eba35e86ea02b13c3c3c43ac34fae30161160`.
Post-review release merge `55194ff329271ada376b691b2733ab35012cbd68`
published
`ghcr.io/malaiwah/glm52-exl3-vast@sha256:6e2475d0568fd110eeaa1193157c7662747e096b476b05ed71ab247e081e9b82`;
`latest` resolves to the latter digest. Its exact-image boot mounted the
reconciled checkpoint read-only, used no source-code bind mounts, and reached
verified serving with zero restarts. Startup passed short and strict-structured
checks, a complete 512-token temperature-1 sample plus post-sample health, and
3/3 retrieval facts at 32,853 tokenizer-exact tokens. The independent OpenAI
feature suite passed. Public API authentication returned 401 without a key and
200 for authenticated models and chat, and the deployed dashboard rendered its
read-only surface while rejecting a stale configuration form without changing
state.

The exact hardening image bound LMCache HTTP to `127.0.0.1:8089`; its OpenAPI
surface contained only `/`, `/healthcheck`, `/status`, `/version`,
`/lmc_version`, and `/commit_id`. `/env` and `/run_script` returned 404 from
the unprivileged `soul` identity, the process had no credential-like
environment names, and the port timed out externally. A synthetic SOUL launch
completed the length-framed post-hardening key handshake and the root launcher
left zero active `soul` processes after reaping a detached child.

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
short/structured checks, and 32K retrieval. Its first sampler request used
a 1,024-token raw-prompt target (1,027 API prompt tokens after chat templating)
and 512 temperature-1 output tokens, then OOMed in top-p selection with only
3 MiB physically free on one rank. The qualified profile instead pins
3,415,867,392 KV bytes per GPU: exactly 393,216 logical KV tokens and one
maximum-length request. Model loading used 82.42 GiB/rank, graph capture used
0.61 GiB/rank, and two-second `nvidia-smi` sampling retained at least 665 MiB
physical free memory after first-use compilation and C8 sampling.
The rejected-arm minima were transcribed during live diagnosis; its raw server
and `nvidia-smi` traces were not retained. Selected-arm benchmark, feature,
retrieval, post-stress, and server-log artifacts remain on the qualification host.

Unique-prefix prefill measured 2,465 / 2,444 / 2,380 / 2,282 tok/s at
8K / 32K / 64K / 128K. Aggregate temperature-1 decode for
a 1,024-token raw-prompt target (1,027 API prompt tokens after chat templating)
and 512 output tokens measured 60.40 / 153.93 / 227.55 tok/s at
C1 / C4 / C8, with mean acceptance lengths 3.41 / 3.45 / 3.51 and draft-token
acceptance of 80.45% / 81.57% / 83.63%. All 72 requests completed without
failure, preemption, or GPU/LMCache prefix reuse.

The startup arithmetic, factual, instruction, strict structured-output, and
32K three-depth retrieval gate passed. The independent OpenAI feature suite
passed tokenize, thinking on/off, streaming usage, preserved-thinking
multi-turn, strict JSON with thinking, tool choice, tool calls, and tool-result
continuation. A five-depth matrix retrieved all 15 facts from tokenizer-exact
131,407-, 261,192-, and 389,959-token documents. The post-stress short and
structured gate passed. Review of the 3,469-line ready-state log found
first-use CuTeDSL/Triton JIT after readiness, invalidating the earlier
zero-post-ready-compile audit result; the structured-FSM, CUDA, distributed,
process-failure, and error-level categories remained clean.
A corrected audit of the warmed line 3,030–3,469 window found zero findings in
all categories.

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
