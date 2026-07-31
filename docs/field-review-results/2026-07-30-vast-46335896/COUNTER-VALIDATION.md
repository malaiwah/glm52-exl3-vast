# GLM-5.2 r14 field-repair counter-validation guide

Snapshot: 2026-07-31 UTC

This is the canonical handoff for independent review of the r14 field-repair
stack. It identifies exactly what to apply, what has been measured, which
historical predecessors failed, and the final release records. Do not
substitute moving PR heads or an unpinned image tag and still attribute the
result to this stack.

> **Qualification status:** production qualification of the exact repaired
> SparkInfer/vLLM successor is complete. It passed full-model startup, the
> complete text feature suite twice, 32K verification, all 20 planted needles
> across four cold 65K/126K probes, and bounded C1/C2/C4/C8 serving matrices
> with zero request failures or preemptions; its post-prime warm log window is
> clean. Final scorecards were **96/100 GSM8K** and **44/50 GPQA** with zero
> request errors; all 44 GPQA normal stops were correct and its only six misses
> exhausted the 32K reasoning ceiling. TERM released the wrapper within 2
> seconds and all workers/GPU allocations within 19 seconds. Release appliance
> `<FINAL_IMAGE_TAG>@<FINAL_IMAGE_DIGEST>`;
> immutable evidence `<EVIDENCE_COMMIT>` / `<EVIDENCE_URL>`. Independent
> counter-validation remains encouraged, but no listed release gate is open.

## Immutable starting point

| item | exact value |
|---|---|
| parent image tag | `voipmonitor/vllm:gilded-gnosis-v20-vllm749050e-si8110e3e-fi801d57a-cu132-20260730-r14` |
| parent registry digest | `sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2` |
| pulled image ID seen in qualification | `sha256:bdf8fe02d0d44f1d4704149363e86aa2265d42ac739b3200c4f58788586614c0` |
| r14 build source | `local-inference-lab/blackwell-llm-docker@2464cc03d0298493bf345bbc797f77c4455efda8` |
| vLLM r14 tree | `749050edab1b6664937c52fa1b0be360be632c1e` |
| SparkInfer r14 tree | `8110e3ea417794bfb08aff1fba20135102e5536b` |
| LMCache r14 tree | `a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa` |
| checkpoint | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| EXL3 extension | `brandonmmusic-max/exllamav3`, `a1-retile-sm120@704aefd` |
| runtime | PyTorch `2.12.0+cu132`; CUDA runtime `13.2.1`; FlashInfer `801d57a`; XGrammar `0.2.5` |
| turnkey manifest | [`patches/field-review-r14/manifest.json`](../../../patches/field-review-r14/manifest.json), SHA-256 `6ba55c5f39711b875dbd044e310ee5a2473ae597cbc4b57a85911e9a4ba868eb` |
| immutable evidence commit | `<EVIDENCE_COMMIT>` |
| runtime release commit | `<RELEASE_COMMIT>` |
| runtime image | `<FINAL_IMAGE_TAG>@<FINAL_IMAGE_DIGEST>` |

The base image must be pulled by digest:

```bash
docker pull docker.io/voipmonitor/vllm@sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2
docker image inspect \
  docker.io/voipmonitor/vllm@sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2 \
  --format '{{json .RepoDigests}} {{.Id}}'
```

## Exact eight-component patch order

Apply the following components in manifest order. The applicator verifies the
patch digest and every affected file's before/after SHA-256, runs all dry-runs
before the first mutation, uses zero fuzz, rejects mixed states, and is
idempotent only for a complete known state.

| # | manifest component | exact base -> tested head | patch SHA-256 | public review surface |
|---:|---|---|---|---|
| 1 | `vllm-210-211` | `5a6979f176ef7535c2f44ab653c78f8292887076` -> [`5a7ac1481e1fc24b0bbc35efe160b2ff34797bee`](https://github.com/malaiwah/vllm-voipmonitor/commit/5a7ac1481e1fc24b0bbc35efe160b2ff34797bee) | `0f2c10110fc9bc0701003a46c8379c335dd9018cecbf64722e0db64eea748fb1` | [vLLM #210](https://github.com/local-inference-lab/vllm/pull/210) at `47dc47d87f428e195f66cd8e7beffd24946a415b`; merged [#211](https://github.com/local-inference-lab/vllm/pull/211), merge `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| 2 | `vllm-semantic-pcie-channels` | `5a7ac1481e1fc24b0bbc35efe160b2ff34797bee` -> [`be1e289a8ca6cc043b582b26c788efc4b1f5d0a8`](https://github.com/malaiwah/vllm-voipmonitor/commit/be1e289a8ca6cc043b582b26c788efc4b1f5d0a8) | `25ef7b7a56ac82ddcdddcf7c4ae3a57cc6af33995af2464fb97cf7f7315e1ee7` | caller half of the semantic eager/target/draft/encoder PCIe-channel contract; patch-identical clean current-base companion series ending at [`d344da36`](https://github.com/malaiwah/vllm-voipmonitor/commit/d344da368da4496eeb308a75faf314cb6376dc62) |
| 3 | `vllm-semantic-pcie-finalizer` | `be1e289a8ca6cc043b582b26c788efc4b1f5d0a8` -> [`f99e1e7b8636ca3811ab6d23084ac6da63420dc3`](https://github.com/malaiwah/vllm-voipmonitor/commit/f99e1e7b8636ca3811ab6d23084ac6da63420dc3) | `d69ff9c4a212e59838b031d3d2ec8bd40fc8c8e0c5ce6e2b23220fec6bdbf685` | process-group-aware abnormal-GC containment; same clean companion series, with abnormal-finalizer lifecycle tracked by [vLLM #215](https://github.com/local-inference-lab/vllm/issues/215) |
| 4 | `sparkinfer-w4a16-100-102` | `9b852b281250123fe323f63ccb1df3cac0f3bbca` -> `e2205cba8d78db03427a3945a75a1f647b51ef5a` | `247fa59a633144933e5993299dab35ee079b1dd3e4f7c12f113f1103c2ad2c69` | [SparkInfer #100](https://github.com/local-inference-lab/sparkinfer/pull/100) at `c2f135c66032ac5e9d0778067dd19ae1910cff47` + [#102](https://github.com/local-inference-lab/sparkinfer/pull/102) at `2bed880a7e9edbd9f2d976ba1a8ee88c9ba6e338`; reviewed composition `cae3aad137bd739881833c064e1912177698156f`; exact r14 replay head `e2205cba...` has **no standalone public commit**, so use the manifest patch |
| 5 | `sparkinfer-pcie-ipc-hardening` | `9b852b281250123fe323f63ccb1df3cac0f3bbca` -> [`31fa6a48116471ce423f0338047453ec1032c202`](https://github.com/malaiwah/sparkinfer/commit/31fa6a48116471ce423f0338047453ec1032c202) | `114c782ce99da46394b8110de68cb72b368cc7257e88693e405375218384c720` | first portion of draft [SparkInfer #105](https://github.com/local-inference-lab/sparkinfer/pull/105), which supersedes the unsafe public [#101](https://github.com/local-inference-lab/sparkinfer/pull/101)/[#103](https://github.com/local-inference-lab/sparkinfer/pull/103) runtime lineage |
| 6 | `sparkinfer-pcie-semantic-prewarm` | `31fa6a48116471ce423f0338047453ec1032c202` -> [`bc62980543b3ca59a9bee971df1b19ce6181964c`](https://github.com/malaiwah/sparkinfer/commit/bc62980543b3ca59a9bee971df1b19ce6181964c) | `be1ce8b40b8e17e30722cce5b335b463c22cff0845c6cb7878f6f5ba736e6e70` | current exact head of draft [SparkInfer #105](https://github.com/local-inference-lab/sparkinfer/pull/105) |
| 7 | `lmcache-lifecycle-hardening` | `8e2857e6306bb05882c7e4035337733ff1a01cfb` -> [`f24812ee33dc9196e788d1004b1d68f473741e2b`](https://github.com/malaiwah/LMCache/commit/f24812ee33dc9196e788d1004b1d68f473741e2b) | `faea3c4ffd73789b54f251fef50073a0765d7af826c1edac92fd70b5fc1c4540` | composes [LMCache #18](https://github.com/local-inference-lab/LMCache/pull/18), [#19](https://github.com/local-inference-lab/LMCache/pull/19), and [#20](https://github.com/local-inference-lab/LMCache/pull/20), then adds lifecycle/qualification repairs; **no upstream PR for the final successor yet** |
| 8 | `blackwell-launcher-13` | `2464cc03d0298493bf345bbc797f77c4455efda8` -> [`5eee240d6c32e08516287380131c96ca7493f0f6`](https://github.com/malaiwah/blackwell-llm-docker/commit/5eee240d6c32e08516287380131c96ca7493f0f6) | `9cfde60e9b37a01ed3b82609e53674d9d89c6e3716ce824a646beada4ae1b5ae` | [blackwell-llm-docker #13](https://github.com/local-inference-lab/blackwell-llm-docker/pull/13), filed from [issue #12](https://github.com/local-inference-lab/blackwell-llm-docker/issues/12) |

All eight components are present in the appliance image so one immutable
artifact can be reviewed. Component 7 is intentionally dormant in the
full-model attribution gate because prefix offload is disabled. Its 184-test
CUDA union and native round trip qualify the focused LMCache successor; they
do not constitute a live GLM+LMCache serving qualification.

Do **not** cherry-pick public SparkInfer #101/#103 and call it equivalent.
Their immediate selector tests were repaired, but the field review found
additional collective setup, teardown, rank-order, capture ownership and
grid-residency defects. Draft PR #105 at `bc629805...` is the reviewed
replacement. Likewise, the two semantic vLLM commits are the required caller
half of #105; SparkInfer alone is not the tested combination.

The public issue map for reviewers is:

- vLLM [#208](https://github.com/local-inference-lab/vllm/issues/208) for
  V1 semantic PCIe-channel rollback and
  [#215](https://github.com/local-inference-lab/vllm/issues/215) for safe
  finalization after process-group teardown;
- vLLM [#205](https://github.com/local-inference-lab/vllm/issues/205) for
  probabilistic draft sampling; SparkInfer
  [#93](https://github.com/local-inference-lab/sparkinfer/issues/93) for the
  EXL3/W4A16 memory path;
- SparkInfer [#95](https://github.com/local-inference-lab/sparkinfer/issues/95)
  and [#97](https://github.com/local-inference-lab/sparkinfer/issues/97) for
  replay/PCIe ownership, and
  [#98](https://github.com/local-inference-lab/sparkinfer/issues/98) for
  capture-safe W4A16 resolution;
- rtx6kpro [#45](https://github.com/local-inference-lab/rtx6kpro/issues/45),
  [#47](https://github.com/local-inference-lab/rtx6kpro/issues/47),
  [#48](https://github.com/local-inference-lab/rtx6kpro/issues/48), and
  [#42](https://github.com/local-inference-lab/rtx6kpro/issues/42) for the
  LMCache failure and lifecycle findings.

## Expected gain, with claim boundaries

| result | status | exact scope |
|---|---|---|
| **1,120.2 MiB/rank returned** | measured full-model A/B | vLLM #210, MTP3, EXL3 prefill capacity `1,024` versus unset capacity: target/draft arenas `759.8/1,054.2` -> `279.7/414.1 MiB/rank` |
| **64.90904 MiB/rank planner saving** | measured sizing claim, separate from the GLM result | SparkInfer #100 removes an unreachable W4A16 reserve for unpinned generic planning; this GLM EXL3 profile pins its block sizes, so the saving is not active KV here |
| `110,080` estimated maximum -> `257,024` KV tokens | measured full-model consequence | exact MTP3/GMU `0.92` negative control failed 131K admission with prefill capacity unset; capacity `1,024` served 126K retrieval |
| about 2% short-prefill cost | measured tradeoff | prefill capacity `1,024` measured `1,807.6` versus `1,845.9 tok/s` at the matched MTP0 3K point; decode was unchanged |
| MTP3 `62.63 tok/s` versus MTP0 `33.98 tok/s` | measured profile benefit, **not a patch speedup** | same rental family, 3K C1 sample; MTP3 MAL `3.111`, draft acceptance `70.37%` |
| W4A16 production-shape PP/TG change | no claim | the final matched `I=1856` kernel A/B was throughput-neutral within noise and showed no allocated/reserved peak difference |
| semantic PCIe, LMCache and launcher changes | reliability/correctness only | no direct PP/TG or KV claim |

The defensible memory headline is therefore:

> **1,120.2 MiB/rank recovered in a full GLM-5.2 MTP3 run. SparkInfer also
> proves a separate 64.909 MiB/rank generic-planner saving, but the shipped GLM
> path pins that planner and the two numbers must not be added for this profile.**

Do not convert the memory sum directly into a fixed KV-token promise. Arena
alignment, graph pools and the exact capture catalog can move the observed KV
capacity by small amounts between otherwise similar shapes.

## Counter-validation host and serving profile

The reference host was Vast.ai instance `46335896`:

- 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each;
- driver `610.43.02`, CUDA UMD `13.3`, CUDA runtime `13.2.1`;
- PyTorch `2.12.0+cu132`;
- same-NUMA `NODE` topology, but CUDA peer access false for every pair;
- 600 W/card ceiling, no competing GPU process during attributed runs.

Attach your own `nvidia-smi`, `nvidia-smi topo -m`, peer-access probe, CPU/RAM,
disk, image inspect and package-version output. Different topology or power
limits are valid counter-evidence, but must be labeled rather than pooled into
the matched result.

Use this exact isolation profile first:

| knob | value |
|---|---:|
| model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` revision `d7d79c2...` |
| model mode | text-only; checkpoint read-only; vision disabled |
| TP / DCP | `4 / 4` |
| speculative decoding | native MTP3, probabilistic draft, standard rejection |
| model length | `131072` |
| maximum sequences | `8` |
| maximum batched tokens | `2048` |
| EXL3 prefill capacity | `1024` |
| graph / Trellis maximum | `32 / 32` |
| capture sizes | `4,8,12,16,20,24,28,32` |
| GPU memory utilization | `0.92` |
| KV cache | dynamic-token `nvfp4_ds_mla` |
| CKV gather / workspace / prefetch | `140000 / 1024 MiB / 0` |
| prefix offload | disabled (`OFFLOAD_FRACTION=0`, disk `0`) to isolate active GPU memory |
| API | local port `18000`, unauthenticated only on the isolated trusted host |
| compile cache | a fresh candidate-specific root; never reuse the rejected stack's cache |

This 131K profile is deliberately an attribution gate. It does not by itself
qualify the final 512K production profile, vision, or live LMCache DRAM/NVMe
offload.

## Expected startup fingerprint

A correct run should include all of the following before traffic:

- every routed layer reports `tiers=((3, 192), (4, 64))`;
- target mixed-Trellis arena approximately **294.6 MiB/rank** at this graph
  shape and prefill capacity;
- rank-sliced native MTP draft arena approximately **414.1 MiB/rank**;
- model weights approximately **83.28 GiB/rank**;
- profiled peak activation approximately **1.32 GiB/rank**;
- actual CUDA-graph pool approximately **0.14 GiB/rank**;
- available KV approximately **1.87–1.91 GiB/rank**, normally about
  **251K–257K logical tokens** on the reference host;
- both PIECEWISE and FULL target/MTP graph profiling and capture complete;
- `Application startup complete.` and `/health` returns 200.

The corrected successor reached **251,392 logical KV tokens**, completed graph
capture, became API-ready, and then passed the feature, long-context and
bounded-load gates below. The recorded per-rank profile was approximately
294.6 MiB target Trellis, 414.1 MiB native MTP draft, 83.28 GiB weights,
1.32 GiB peak activation and 0.14 GiB CUDA graphs, leaving 1.87 GiB for KV.
Its complete server log and companion result artifacts are fixed by
`<EVIDENCE_COMMIT>` / `<EVIDENCE_URL>`.

Hard failures include any stream-key collision, OOM, Xid, CUDA/IPC warning,
worker death, repeated JIT on a previously exercised warm shape, failed
structured-output FSM advance, request error, preemption during the bounded
matrix, degeneration, missing needle, or residual GPU allocation after the
graceful deadline. A first-use compile for a new scorecard shape must be
reported and followed by a clean repeated-workload window; it is not silently
folded into the clean established-shape audit.

## Corrected-stack qualification results

These results are from exact SparkInfer `bc629805...` plus vLLM `f99e1e7...`
on the reference host and profile above. By themselves, they close the
startup, feature, retrieval, bounded concurrency/performance and warm-log
portions of the gate; the later final records below close GPQA, shutdown, and
release packaging.

### Features, correctness and retrieval

- The initial and warm feature suites each passed **13/13 checks**: all 12
  required checks and the optional required-tool-choice check. This includes
  authentication-disabled behavior, tokenization, thinking and non-thinking
  chat, streaming usage, multi-turn preservation of thinking, strict JSON with
  thinking off and on, exactly one tool call, and tool-result round trip.
  Vision was the one intentional passed skip because this profile is
  explicitly text-only.
- The 32K verification passed arithmetic, factual and instruction checks,
  strict JSON while thinking, and **5/5 needles** at depths
  1/25/50/75/99 percent in 32,858 actual tokens, with no degeneration.
- The independent two-seed matrix recovered **20/20 needles**: 5/5 at every
  depth for each of 65,536 and 126,000 target tokens with seeds 20260730 and
  20260731. Actual prompt lengths were 65,893, 65,892, 126,315 and 126,333
  tokens; durations were 41.072, 37.298, 72.781 and 72.760 seconds. No probe
  reported degeneration.

### Matched bounded matrix

The first matrix used the same `field-baseline-v2` prompt corpus, seed,
request counts and limits as the pre-composition control. All 32 decode
requests completed, with zero failures and zero preemptions.

| load | aggregate TG tok/s | prompt tok/s | TPOT mean / p50 / p95 ms | MAL | draft acceptance |
|---:|---:|---:|---:|---:|---:|
| C1 | 30.257 | 726.877 | 17.358 / 17.341 / 18.121 | 3.3214 | 77.38% |
| C2 | 33.388 | 802.092 | 38.619 / 35.572 / 67.995 | 2.9169 | 63.90% |
| C4 | 37.668 | 904.905 | 74.751 / 68.702 / 135.268 | 2.7896 | 59.65% |
| C8 | 61.837 | 1,485.536 | 60.925 / 59.846 / 89.499 | 2.7056 | 56.85% |

Its 3K and 32K single-request prefill points were 557.634 and 1,867.277
tok/s. The 3K point was a new-shape compile point; it is not the warmed 3K
rate. Against the exact pre-composition control, candidate TG deltas were
`+11.31% / -1.61% / +5.87% / -1.89%` at C1/C2/C4/C8, while 3K/32K prefill
changed `-0.12% / +0.24%`. The alternating signs and small bounded sample make
this reliability-neutral within workload/run variance, not an honest speedup
claim for the lifecycle patches.

### Repeat warm matrix and log audit

A second, already-primed matrix used a distinct `field-baseline-v2-warm`
corpus. It is repeat-stability evidence, not an exact A/B against the control.
Again, all 32 requests completed with zero failures and zero preemptions.

| load | aggregate TG tok/s | prompt tok/s | TPOT mean / p50 / p95 ms | MAL | draft acceptance |
|---:|---:|---:|---:|---:|---:|
| C1 | 29.119 | 699.997 | 22.190 / 22.126 / 26.990 | 2.5975 | 53.25% |
| C2 | 36.191 | 869.987 | 41.089 / 37.349 / 61.885 | 2.3318 | 44.39% |
| C4 | 40.063 | 963.088 | 73.237 / 68.326 / 106.681 | 2.1414 | 38.05% |
| C8 | 62.338 | 1,498.538 | 63.326 / 58.697 / 94.788 | 2.5222 | 50.74% |

The fully warmed 3K and 32K prefill points were **2,011.072** and
**1,857.989 tok/s**. The whole-log audit intentionally failed only its strict
cold criterion: it recorded 28 `post_ready_compile` misses while the fresh
cache learned workload shapes and zero structured-FSM, CUDA/IPC, CUDA runtime,
distributed runtime, process-failure or runtime-error findings. After that
prime, the audited warm window (lines 1,397–1,556) was completely clean:
zero findings in every category, including zero post-ready compile misses.

## Required negative controls

### Memory control

Keep every setting above fixed but unset `VLLM_EXL3_PREFILL_CAPACITY`. The
recorded control used target/draft arenas of `759.8/1,054.2 MiB/rank`, exposed
only `0.82 GiB` for KV, estimated `110,080` maximum tokens, and failed the
131,072-token admission check. The capacity-1,024 successor used
`279.7/414.1 MiB/rank`, exposed `1.91 GiB`, admitted `257,024` tokens and
retrieved the 126K needle. Evidence:
[unset-capacity control](artifacts/vllm-current210-unset-mtp3-v1-server.log),
[capacity-1,024 server](artifacts/vllm-current210-cap1024-mtp3-v3-server.log),
and [126K retrieval](artifacts/vllm-current210-cap1024-mtp3-v3-needle-126k.json).

### PCIe lifecycle control

The rejected combination was SparkInfer `31fa6a48116471ce423f0338047453ec1032c202`
plus vLLM `be1e289a8ca6cc043b582b26c788efc4b1f5d0a8`. During uncaptured descriptor
warm-up inside semantic `vllm:target:profile`, it selected the static eager
channel and failed closed with:

```text
RuntimeError: CUDA stream key ... is already bound to another logical PCIe oneshot channel
```

The complete failure is preserved in
[the rejected full-model log](artifacts/field-review-final-4c880eb-mtp3-c8-v1-server.log).
Do not promote the previously built `4c880eb` appliance image. The successor
must be SparkInfer `bc629805...` together with vLLM `f99e1e7b...` and a fresh
compile-cache namespace.

## Reproduction commands

### 1. Validate and build the content-addressed bundle

From a clean checkout containing this exact manifest:

```bash
python3 scripts/apply_field_review_patches.py \
  --manifest patches/field-review-r14/manifest.json \
  --validate-only
python3 -m pytest -q tests/test_field_review_patches.py

docker build --pull --no-cache \
  -t glm52-field-review-r14-countervalidate .
```

The validation command must print:

```text
>>> validated 8 field-review patch components
```

The Dockerfile applies the same manifest to `/opt/vllm`, active
`site-packages`, and `/usr/local/bin`, then verifies every output hash. A
missing or drifted file must fail the build; never resolve that by enabling
patch fuzz.

### 2. Launch one isolated server

Inside the candidate container, with the checkpoint available at
`/workspace/GLM-5.2-EXL3-TR3-3.25bpw` and all four GPUs otherwise idle:

```bash
export EVIDENCE_ROOT=/workspace/field-review-tests/artifacts
export RUN_LABEL=countervalidate-r14-bc62980-f99e1e7-mtp3-c8
export SERVER_LOG="$EVIDENCE_ROOT/$RUN_LABEL-server.log"
export SERVER_PID="$EVIDENCE_ROOT/$RUN_LABEL.pid"
mkdir -p "$EVIDENCE_ROOT"

nohup env \
  RUN_LABEL="$RUN_LABEL" MODE=baseline PORT=18000 \
  MODEL_ROOT=/workspace \
  JIT_ROOT="/workspace/field-review-tests/jit/$RUN_LABEL" \
  PREFILL_CAPACITY=1024 MTP_TOKENS_VALUE=3 \
  GPU_MEMORY_UTILIZATION_VALUE=0.92 \
  MAX_MODEL_LEN_VALUE=131072 MAX_NUM_SEQS_VALUE=8 \
  MAX_NUM_BATCHED_TOKENS_VALUE=2048 \
  MAX_CUDAGRAPH_CAPTURE_SIZE_VALUE=32 \
  CUDAGRAPH_CAPTURE_SIZES_VALUE=4,8,12,16,20,24,28,32 \
  TRELLIS_MAX_M_VALUE=32 \
  GPU_BLOCKS_OVERRIDE_VALUE=0 OFFLOAD_FRACTION_VALUE=0 \
  PREFIX_CACHE_BACKEND_VALUE=lmcache PREFIX_CACHE_DISK_GB_VALUE=0 \
  VISION_VALUE=0 \
  bash /opt/scripts/field_review_full_gate.sh \
  >"$SERVER_LOG" 2>&1 &
echo $! >"$SERVER_PID"
```

`MODE=baseline` here means “use the packages installed in the candidate
image,” not “use unpatched r14.” Source-overlay reviewers may instead use
`MODE=candidate`, an exact `CANDIDATE_PYTHONPATH`, and
`CANDIDATE_SOURCE_REVIEW=0`; report that path and every imported module.

### 3. Run correctness and feature gates before benchmarks

```bash
curl --fail --silent --show-error http://127.0.0.1:18000/health
curl --fail --silent --show-error http://127.0.0.1:18000/v1/models

python3 /opt/scripts/feature_suite.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-features.json"

python3 /opt/scripts/verify_serving.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --max-model-len 131072 --needle-tokens 32768 \
  --depths 0.01,0.25,0.5,0.75,0.99 --needle-timeout 1800 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-verify32k.json"

python3 /opt/scripts/needle_matrix.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --max-model-len 131072 --reserve-tokens 4096 \
  --sizes 65536,126000 --depths 0.01,0.25,0.5,0.75,0.99 \
  --seeds 20260730,20260731 --timeout 1800 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-needles.json"
```

The feature suite must pass health/model discovery, tokenization, thinking and
non-thinking chat, streaming usage, multi-turn preservation, strict JSON with
thinking off/on, automatic tool call and tool-result round trip. Vision is an
intentional skip for this text-only checkpoint. The needle matrix must recover
all 20 planted values and report no degeneration.

### 4. Prime once, then measure the bounded serving matrix

```bash
python3 /opt/scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --input-tokens 3072 --output-tokens 128 \
  --concurrency 1,2,4,8 --requests-per-level 8 \
  --prefill-tokens 3072,32768 --warmup 2 \
  --temperature 1.0 --seed 20260731 \
  --prompt-seed "$RUN_LABEL" --timeout 1800 \
  --metadata stack=bc629805-f99e1e7 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-benchmark.json"
```

Record PP, aggregate TG, TPOT, MAL and draft acceptance at every concurrency.
Do not compare a cold compile point with a warm control. Capture a log line
offset after the deliberate prime and require the subsequent warm window to
be clean:

```bash
WARM_START=$(( $(wc -l <"$SERVER_LOG") + 1 ))
python3 /opt/scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:18000 --model GLM-5.2 \
  --input-tokens 3072 --output-tokens 128 \
  --concurrency 1,2,4,8 --requests-per-level 8 \
  --prefill-tokens 3072,32768 --warmup 0 \
  --temperature 1.0 --seed 20260731 \
  --prompt-seed "$RUN_LABEL-warm2" --timeout 1800 \
  --metadata stack=bc629805-f99e1e7 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-benchmark-warm2.json"
python3 /opt/scripts/field_review_log_audit.py \
  --log "$SERVER_LOG" --start-line "$WARM_START" \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-warm-log-audit.json"
```

### 5. Run the scorecards last

GPU qualification preempts benchmarks. Only after the source, startup,
correctness, retrieval, concurrency and log gates pass, repeat the inexpensive
GSM8K control on the final successor, then run GPQA with enough room for GLM
to finish reasoning:

```bash
python3 /opt/scripts/evaluate_scorecard.py \
  --task gsm8k_cot --base-url http://127.0.0.1:18000 \
  --model GLM-5.2 --limit 100 --seed 0 --concurrency 8 \
  --max-tokens 2048 --timeout 1800 \
  --label "$RUN_LABEL-gsm8k100-c8-2k" \
  --metadata stack=bc629805-f99e1e7 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-gsm8k100-c8-2k.json"

python3 /opt/scripts/evaluate_scorecard.py \
  --task gpqa_diamond --base-url http://127.0.0.1:18000 \
  --model GLM-5.2 --limit 50 --seed 0 --concurrency 4 \
  --max-tokens 32768 --timeout 3600 \
  --label "$RUN_LABEL-gpqa50-32k" \
  --metadata stack=bc629805-f99e1e7 \
  --out "$EVIDENCE_ROOT/$RUN_LABEL-gpqa50-32k.json"
```

The server profile supplies `reasoning_effort=high`; the scorecard explicitly
enables thinking. Report truncations separately and score only visible final
answers. The earlier 8K result is a cap-constrained lower bound and must not be
used as the quality control for this run.

### 6. Prove graceful shutdown

```bash
kill -TERM "$(cat "$SERVER_PID")"
for _ in $(seq 1 90); do
  WRAPPER_ALIVE=
  GPU_APPS=
  VLLM_PROCS=
  kill -0 "$(cat "$SERVER_PID")" 2>/dev/null && WRAPPER_ALIVE=1
  GPU_APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)
  VLLM_PROCS=$(pgrep -f '[V]LLM::(EngineCore|Worker_)|/[v]llm serve ' || true)
  if [ -z "$WRAPPER_ALIVE$GPU_APPS$VLLM_PROCS" ]; then break; fi
  sleep 2
done
if kill -0 "$(cat "$SERVER_PID")" 2>/dev/null ||
   [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)" ] ||
   [ -n "$(pgrep -f '[V]LLM::(EngineCore|Worker_)|/[v]llm serve ' || true)" ]; then
  echo "FATAL: serving tree still owns processes or GPUs after deadline" >&2
  exit 1
fi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader
```

All four workers must exit, the recorded wrapper must be gone, and every GPU
must return to its idle footprint. A destructor traceback, process-group error,
CUDA exporter warning, or retained worker is a failure even if requests passed.

## Evidence ledger

Accepted controls and focused evidence are linked from the detailed
[`UPSTREAM-REPAIR-CAMPAIGN.md`](UPSTREAM-REPAIR-CAMPAIGN.md). The most useful
entry points are:

- [vLLM #210 capacity-1,024 full-model server](artifacts/vllm-current210-cap1024-mtp3-v3-server.log),
  [benchmark](artifacts/vllm-current210-cap1024-mtp3-v3-bench-measure-a.json),
  and [126K retrieval](artifacts/vllm-current210-cap1024-mtp3-v3-needle-126k.json);
- [pre-composition control server](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-server.log),
  [features](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-features.json),
  [needles](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-needles.json),
  [concurrency](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-benchmark.json),
  and [warm log audit](artifacts/vllm-r14-field-baseline-mtp3-c8-v2-warm-log-audit.json);
- [rejected predecessor full-model log](artifacts/field-review-final-4c880eb-mtp3-c8-v1-server.log);
- [SparkInfer atomic vLLM union](artifacts/atomic-vllm-be1e289-spark-31fa6a4-linux-gpu-full.log),
  [LMCache CUDA union](artifacts/lmcache-f24812e-composed-gpu3-union.log), and
  [LMCache native round trip](artifacts/lmcache-f24812e-same-id-cuda-roundtrip.log).

The corrected successor's completed workload artifacts are:

- initial [feature suite](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-features.json)
  and repeated [warm feature suite](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-warm-features.json),
  both 13/13;
- [32K verification](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-verify32k.json),
  including 5/5 retrieval;
- [65K/126K two-seed needle matrix](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-needles.json),
  20/20 with no degeneration;
- matched [bounded concurrency/performance matrix](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-benchmark.json)
  and the [repeat warm matrix](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-warm-benchmark.json),
  each 32/32 requests with no error or preemption; and
- [whole-log cold audit](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-log-audit.json),
  which isolates 28 expected fresh-shape compile misses, plus the clean
  [post-prime warm audit](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-warm-log-audit.json).
- final-successor [GSM8K JSON](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gsm8k100-c8-2k.json)
  and [clean workload audit](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gsm8k100-c8-2k-log-audit.json);
- [GPQA JSON](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gpqa50-c4-32k.json)
  and [workload audit](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gpqa50-c4-32k-log-audit.json).
  GPQA introduced exactly four one-per-rank W4A16 disk-cache misses for its
  previously unseen long-reasoning shape; there were no FSM, CUDA/IPC,
  distributed, process, or runtime-error findings, and the immediately
  subsequent GSM8K window had zero further compilation;
- [GPU telemetry summary](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-telemetry-summary.md),
  [graceful shutdown](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-shutdown.txt),
  [rental release](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-rental-release.txt),
  [serving-runtime identity](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-serving-runtime-identity.txt),
  and [artifact SHA-256 manifest](artifacts/field-review-final-bc62980-f99e1e7-mtp3-c8-v2-sha256.txt).

Final release records:

- GSM8K: **96/100**, 100/100 normal stops, zero request errors, MAL 3.5314,
  draft-token acceptance 84.38%;
- GPQA: **44/50 (88%)**, 44/44 normal stops correct, six 32,768-token
  reasoning-ceiling truncations, zero request errors, MAL 3.2614,
  draft-token acceptance 75.38%;
- graceful shutdown/GPU release: **PASS** — wrapper gone within 2 seconds,
  all four workers and GPU allocations gone within 19 seconds, 14 MiB/GPU
  idle footprint;
- immutable appliance: `<FINAL_IMAGE_TAG>@<FINAL_IMAGE_DIGEST>`
- complete successor log and JSON evidence: `<EVIDENCE_COMMIT>` / `<EVIDENCE_URL>`

## What to report back

For every counter-validation, post:

1. exact parent digest, manifest SHA-256, all eight tested heads and whether
   the bundle applied or was already present;
2. complete hardware/software/topology/power inventory and whether P2P works;
3. fresh JIT/cache path and confirmation that no other GPU process ran;
4. target/draft arena, weights, activation, graph-pool, free-KV and logical-KV
   startup lines;
5. health, feature, 32K and 65K/126K results;
6. C1/C2/C4/C8 PP/TG/TPOT/MAL/acceptance and request/preemption counts;
7. cold and post-prime warm log audits;
8. GSM8K and GPQA results with completion ceilings and truncation counts; and
9. graceful-shutdown result and durable artifact URLs.

Label deviations. Do not collapse source correctness, startup reliability,
model quality and performance into a single “passed” line.

## Discord-ready handoff

```text
GLM-5.2 r14 field-repair stack ready for independent counter-validation
(production-qualified exact release; independent reproduction requested).

Base:
voipmonitor/vllm@sha256:cb03f2079d8a74915f01cda15f6bdf505762d13cc3fff192f7ebdaaf6e318bf2

Review/apply this combination, in the turnkey manifest's exact order:
1) vLLM #210 @ 47dc47d + merged #211 @ 3003860; r14 integration 5a7ac148
2) vLLM semantic PCIe caller + safe finalizer through f99e1e7b
3) SparkInfer #100 @ c2f135c + #102 @ 2bed880; r14 replay e2205cba
4) SparkInfer draft #105 @ bc629805 (supersedes #101/#103 runtime lineage)
5) LMCache #18/#19/#20 + lifecycle successor f24812ee
6) blackwell-llm-docker #13 @ 5eee240d

Important: the deployable representation is eight hash-locked manifest
components because several rows above are ordered successors. Do not test only
the public PR heads and call it the same stack.

Measured memory result: vLLM #210 returned 1,120.2 MiB/rank in a full
GLM-5.2 MTP3 A/B and moved the exact shape from a 110,080-token admission
failure to 257,024 KV tokens with clean 126K retrieval. SparkInfer #100 has a
separate 64.909 MiB/rank generic-planner saving, but this GLM path pins the
planner; do not add the two values for this profile. No extra PP/TG gain is
claimed for the lifecycle repairs.

Test profile: 4x RTX PRO 6000 Blackwell, TP4/DCP4, willfalco 3.25 bpw @
d7d79c2, MTP3 probabilistic, maxlen 131072, seqs 8, batch 2048, EXL3 prefill
capacity 1024, graph/trellis 32, GMU .92, dynamic-token NVFP4 KV, offload and
vision disabled, fresh compile cache.

Negative control: 31fa6a4 + be1e289 fails full startup with a semantic
stream-key collision. Successor: bc629805 + f99e1e7 passed full startup with
251,392 logical KV tokens, 13/13 features twice, 5/5 at 32K, 20/20 at
65K/126K, and two C1/C2/C4/C8 matrices (64/64 total decode requests) with no
error or preemption. The post-prime log window is clean. Fully warm prefill was
2,011 tok/s at 3K and 1,858 tok/s at 32K; warm aggregate TG was 29.1/36.2/
40.1/62.3 tok/s at C1/C2/C4/C8. These are qualification numbers, not a
lifecycle-patch speedup claim.

Final closeout: GSM8K 96/100; GPQA 44/50 (44/44 normal stops correct, six 32K
truncations); graceful shutdown released every worker/GPU allocation within
19 seconds; appliance <FINAL_IMAGE_TAG>@<FINAL_IMAGE_DIGEST>; immutable
evidence <EVIDENCE_COMMIT> / <EVIDENCE_URL>.

Canonical commands, patch digests, expected startup lines, controls and
evidence ledger:
https://github.com/malaiwah/glm52-exl3-vast/blob/<EVIDENCE_COMMIT>/docs/field-review-results/2026-07-30-vast-46335896/COUNTER-VALIDATION.md
```
