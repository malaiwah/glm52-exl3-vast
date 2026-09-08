# K6 SPILL-OVER — GLM-5.3 Flash K6 Appliance (Phase 2)

Created: 2026-08-28, after GLM-5.2 3.42bpw baseline passed verify_serving.py.

Source: `HANDOFF.md` (GLM-5.3 Flash K6 mission brief, verified 2026-08-28).


---

## Live TP4/DCP4 qualification (2026-08-29)

This supersedes the initial failed attempt below. GLM-5.3 Flash K6 is live and
verified on JarvisLabs VM 485913 at `151.185.34.24:8000`.

### Exact production shape

| Field | Value |
|-------|-------|
| Image | `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2` |
| Checkpoint | `/home/turnkey/GLM-5.3-Flash-TR3-6bpw` |
| Served model | `GLM-5.3-Flash-K6` |
| Hardware | 4× RTX PRO 6000 Blackwell |
| Parallelism | TP4, DCP4/A2A, `B12X_PCIE_ONESHOT_DMA` with PyNCCL fallback |
| Quantization | EXL3 K6 routed experts; NVFP4 DS-MLA KV |
| Request limit | 520,192 tokens |
| Scheduler | 3,072 batched tokens, 8 sequences, chunked prefill |
| Memory/cache | GMU 0.93, prefix caching enabled |
| Triton cache | `/home/turnkey/.triton-warningfix-20260829-v8` |
| JIT diagnostics | Default warning mode, `jit_monitor_verbose=False`, no callsite tracer |

### Qualification evidence

- `verify_serving.py` passed health, arithmetic, factual, instruction,
  thinking plus strict structured output, and all three needles in an exact
  32,933-token prompt after the clean production restart.
- The router-gate cutover improved every cell in the 3-context × 3-concurrency
  matrix: +1.2% to +3.3% aggregate decode throughput at 0/32K/128K and
  concurrency 1/4/8.
- Prefix-cache qualification queried 5,210,174 tokens, recorded 4,692,480 hits
  (about 90%), returned 102 HTTP 2xx responses, and had zero preemptions.
- Fresh-cache verbose validation warmed, on every rank, two EXL3 route-pack
  variants, four sparse-MLA metadata variants, and four K-pool variants.
  The 32K/128K prefill exercise then emitted no inference-time Triton JIT
  warning for route packing, prefill metadata, or K-pool cache writes.
- One cold validation rollout hit an asynchronous CUDA illegal-address failure
  during graph capture. The service was rolled back, the metadata probe was
  validated independently, and two subsequent cold startups succeeded without
  disabling custom all-reduce. The production restart also succeeded.
- Clean production startup and post-verification logs contain no warning,
  error, traceback, deprecation warning, or inference-time JIT warning.

### Warning fixes and upstream disposition

| Area | Resolution | Upstream |
|------|------------|----------|
| Route-pack scalar/capacity variants | Runtime scalar bounds plus explicit EXL3 alignment warmup | [B12X #256](https://github.com/local-inference-lab/b12x/pull/256) |
| PCIe teardown barrier device warning | Bind all close barriers to the pool CUDA device | [B12X #257](https://github.com/local-inference-lab/b12x/pull/257) |
| Sparse-MLA prefill metadata JIT | Stable scalar signature plus valid in-process runtime warmup | [vLLM #54345](https://github.com/vllm-project/vllm/pull/54345), with field findings also recorded on #50175 |
| TorchAO Enum pytree warning | Preserve native Enum handling; register only non-Enum classes | [TorchAO #4849](https://github.com/pytorch/ao/pull/4849), fixing #4848 |
| GLM router gate | Apply the gate once | Already upstream as `015dcd423` |
| K-pool cache-write/tail JIT | Runtime token count plus aligned/unaligned initialized-cache warmup | Qualified for this image; current vLLM has replaced these snapshot-specific kernels |

The exact production overlay manifest is maintained in `K6-FALLBACKS.md`.

---

## K6 Bring-up Attempt (2026-08-29)

### What was attempted
- Downloaded K6 checkpoint (237 GB, 120 shards) to JarvisLabs VM
- Used Brandon v84 image (`verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2`)
- Patched EXL3 quantization to accept K6/K8 bits (was K4-only)
- Patched TP validation to allow TP4 with unsliced_tp_stream checkpoint
- Attempted TP4/DCP4 serve with both B12X and triton MoE backends

### What worked
- ✅ Architecture resolved: `Glm5NextForConditionalGeneration`
- ✅ EXL3 K6 quantization config accepted (after patch)
- ✅ TP4 with unsliced checkpoint (after patch)
- ✅ All 120 safetensors shards loaded successfully at TP4
- ✅ 4 workers initialized (world_size=4, NCCL connected)

### What failed (hard blocker)
- ❌ Expert inference: `ValueError: no valid W4A16 tile config for M/N/K=3072/1024/4096, moe_block_size=128`
- The B12X W4A16 kernel only has tile configs for K4 trellis, not K6
- Switching `--moe-backend triton` does NOT help — the EXL3 quant layer calls B12X directly via `_apply_rank_sliced()` → `plan_tp_moe_scratch` → `compile_w4a16_fused_moe` → `_select_tile_config`
- This is the exact gap B12X PR #245 addresses (generalize fused Trellis decode to K2-K6/MCG)

### Patches applied (need upstream PRs)
| Patch | File | What | Needs PR to |
|-------|------|------|-------------|
| K4/K6/K8 bits | `exl3.py:1130` | Accept `(4, 6, 8)` instead of `4` | `local-inference-lab/vllm` |
| Dynamic bits metadata | `exl3.py:1269` | `hf_config.quantization_config.get("bits", 4)` | `local-inference-lab/vllm` |
| TP4 unsliced | `exl3.py:2758` | Skip TP check for `unsliced_tp_stream` | `local-inference-lab/vllm` |
| K6 tile configs | `b12x/.../w4a16/kernel.py` | Add K6 tile configs | `local-inference-lab/b12x` (#245) |

### Full fallback log
See `K6-FALLBACKS.md` for every workaround and its status.

### Conclusion
The K6 checkpoint loads correctly at TP4. The architecture and quantization are properly
handled by Brandon's vLLM + our patches. The **only** blocker is B12X #245 — the W4A16
kernel needs K6 tile configs. This is the #1 priority PR and was already identified as
such in HANDOFF.md. Once #245 is merged, K6 should load and serve on the first try.

---

## What changed since the handoff

The GLM-5.2 baseline appliance is now **serving and verified** on JarvisLabs VM 485913
(4× RTX PRO 6000, `151.185.34.24`). All infrastructure lessons from that deploy are
in `SESSION-NOTES.md` §11 (speed-run guide). The K6 work below starts from that
proven foundation.

### Baseline accomplishments (Phase 1, complete)
- [x] Workspace cleaned, 4 fresh clones, GLM-5.2 evidence archived (commit `fbe0d27`)
- [x] Docker image built + pushed: `malaiwah/glm52-exl3-vast:latest` (38.6 GB)
- [x] JarvisLabs VM hardened: P2P override, UVM HMM disable, GRUB PCIe params, dist-upgrade, DCGM masked
- [x] Container deployed with `exl3-tr3-3.42bpw` variant, online K6, 520K context
- [x] `verify_serving.py` full gate passed (arithmetic, factual, instruction, structured output, long-context needle)
- [x] 3 code commits (not yet pushed to GitHub): `6981d2e`, `53e6dce`, `03d3448`
- [ ] Michel's manual test (blocked — waiting on him)
- [ ] Push 3 commits to GitHub
- [ ] Destroy VM 485913 after Michel approves

---

## K6 checkpoint: exact state

| Field | Value |
|-------|-------|
| HF repo | [`malaiwah/GLM-5.3-Flash-TR3-6bpw`](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw) |
| HF SHA | `0072eeb95fb328a5a09ca4965cac565f4e8fcd80` |
| Internal revision | `b1967181a3917ae70a437f4884748f6b8e3a1f4d` |
| Architecture | `Glm5NextForConditionalGeneration` / `glm5_next` (image-text) |
| Max position | 1,048,576 |
| Layers | 45 (34 KDA linear-attention + 11 sparse/MLA) |
| Routed experts | 288, top-8 routing |
| Size | ~253.5 GB across 120 safetensors shards (~77% of FP8) |
| Quantization | K6 TR3/MCG on routed experts + MTP; BF16 on non-routed weights |
| Trellis | multiplier `0xCBAC1FED`, 96-word trellis |
| Offline KLD | mean 0.013723, 5 runs, 51,175 positions/run, passes `<0.06` gate |
| Target topology | TP4 on 4×96 GB SM120 |
| Serving receipt | Live diagnostic TP4/DCP4 verification passes; published `serving_reader_qualified=false` remains unchanged pending the full bound receipt |

### Publication inconsistency to fix
- `receipts/k6-five-run-kld.json` says quality gate passed ✓
- `receipts/checkpoint.json` says `qualified=false`, `measured_mean_kld=null` ✗
- `exl3-mcg-storage-abi.json` still says `serving_reader_qualified=false`; this remains correct until the full receipt below is bound and published
- **Action**: reconcile quality evidence separately from serving-reader qualification. The live load/inference evidence is necessary but does not replace reload/replay, source-tree provenance, and the remaining qualification gates.

---

## Two-base strategy

### Base 1: K6 bring-up (diagnostic milestone, NOT the ultimate appliance)
- Digest-pinned Brandon v84 image/source snapshot
- Modified only enough to admit K6 TP4
- Goal: first audited K6 reader/inference receipt (load, generate, reload)
- Brandon v84 image: `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2`
- Brandon v84 digest: `sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
- Brandon repo HEAD: `57db68f1db0d1eefe1dcd2b9350d3f6968c786f0`
- **Caveat**: Brandon is uniform K4 on TP2; ours is K6 on TP4 — different residency/topology

### Base 2: Final appliance (source-locked, auditable)
- [`local-inference-lab/blackwell-llm-docker` PR #28](https://github.com/local-inference-lab/blackwell-llm-docker/pull/28)
- Head: `release/glm53-dflash2-mxfp8-20260828`, SHA `d3920aaa7c0546594d1052ec3dd530732e739719`
- Targets NVFP4 + DFlash2 MXFP8 (not EXL3 K6 — K6 overlay must be added)
- CUDA 13.3, PyTorch 2.13, TP4 on 4× RTX PRO 6000
- Docker Hub manifest: `sha256:1a210a6dcd4eeef4b9515aa03b8117dbd1bad70ed101cdf72ec4692015e2ac4b`
- **Reuse from #28**: Glm5Next wiring, TP4 graph/runtime, vision/multimodal, DFlash/MTP, DeepGEMM/B12X, source-locking style
- **Must add for K6**: EXL3/TR3/MCG loader, K6→TP4 admission, K6 B12X fused decode, dense/online EXL3 prefill dispatch, K6 graph warmup, BF16 non-routed handling

---

## Release-critical PR chain

Dependency chain (not independent cherry-picks):

```
B12X #243 → #245          (cooperative fused Trellis decode K2-K6 MCG)
vLLM #314 → #316 → #318 → #397  (graph priming → reconstruct+hgemm → K6 prefill routing → shared scratch arena)
```

| PR | Head SHA | What | Status |
|----|----------|------|--------|
| [B12X #245](https://github.com/local-inference-lab/b12x/pull/245) | `1a3ea83f` | Generalizes fused Trellis decode to K2-K6 MCG. **Highest priority.** Stacked on #243. | Open, RTX 5090 results forthcoming |
| [vLLM #314](https://github.com/local-inference-lab/vllm/pull/314) | `7917c928` | Graph-decode priming for dense EXL3. Proven on dense checkpoint. | Open, GLM-5.3 MoE applicability must be checked |
| [vLLM #316](https://github.com/local-inference-lab/vllm/pull/316) | `8451183e` | Reconstruct+hgemm prefill, +113%/+118% prefill. Changes FP16 summation order. | Open, has opt-out |
| [vLLM #318](https://github.com/local-inference-lab/vllm/pull/318) | `2b96dad4` | Routes K6/MCG away from B12X decode kernel at prefill row counts. Stacked on #316. | Open, measured on Qwen3.8 not GLM-5.3 |
| [vLLM #397](https://github.com/local-inference-lab/vllm/pull/397) | `96972e10` | Shared reconstruct-scratch arena. Stacked on #318. Recovers ~0.60 GiB. | Open |

### Additional PRs to inspect (not blockers)
- vLLM #455: opt-in int8 embeddings (capacity option, not default)
- vLLM #454 + exllamav3 #299: per-layer split/fused-uniform QKV PoC
- vLLM #439: skip spec decode for single-token requests
- vLLM #312: preserve BF16 unsupported shards (relevant — K6 keeps non-routed BF16)
- vLLM #277: mixed Trellis direct tier slabs (load-memory relevance, already in local r34)
- LMCache #4691, #4600, #4517: expired L1 lease, failed-retrieve recompute, bounded MP completion
- B12X #130, #126: persistent buffer layout, mixed Trellis prewarm


### Memory-control PRs (qualify for OOM-free 500K+ context)

These PRs from Michel's local-inference-lab work directly address GPU memory
headroom and OOM prevention. The JarvisLabs baseline OOM (PCIe DMA allreduce
buffer, 36 MB allocation failure at 99.3% GMU) was worked around by lowering
GMU to 0.93, but the root-cause fixes are in these PRs:



| PR | What | Memory impact | Priority |
|----|------|----------------|----------|
| [vLLM #271](https://github.com/local-inference-lab/vllm/pull/271) | Prewarm full-CKV prefill kernels **before** KV sizing | **Root cause of our OOM**: late JIT compilation after KV sizing eats headroom. Prewarming moves the cost into profiling so KV is sized correctly. | **Highest** — fixes the OOM at the source |
| [vLLM #397](https://github.com/local-inference-lab/vllm/pull/397) | Shared reconstruct-scratch arena across prefill geometries | Recovers ~0.60 GiB by sharing one scratch buffer instead of one per geometry. Stacked on #314→#316→#318. | High — in the release-critical chain |
| [vLLM #270](https://github.com/local-inference-lab/vllm/pull/270) | Share compatible native prefill buffers across geometries | Reduces persistent buffer count. Decode buffers stay owner-local (CUDA graph addresses). | Medium |
| [vLLM #266](https://github.com/local-inference-lab/vllm/pull/266) | Reject `prompt_logprobs=-1` (OOM DoS) | Prevents unbounded allocation when all-vocab logprobs requested. | Medium — hardening |
| [vLLM #258](https://github.com/local-inference-lab/vllm/pull/258) | Account for prompt-logprobs memory in profiling | **MERGED** — the `VLLM_PROMPT_LOGPROBS_CHUNK_SIZE` work. Already in our image, and we fixed the knob wiring (commit `6568ff6`). | Done |
| [vLLM #455](https://github.com/local-inference-lab/vllm/pull/455) | Opt-in int8 embeddings | Embedding table memory savings. Capacity option, not a default. | Low — optional |
| [vLLM #307](https://github.com/local-inference-lab/vllm/pull/307) | Fungible Quant: live fixed-budget MoE precision re-tiering | Runtime precision re-tiering within a memory budget. Experimental. | Research |
| [LMCache #18](https://github.com/local-inference-lab/lmcache/pull/18) | Preserve CUDA IPC event ownership | Fixes MP cache event lifecycle that can leak GPU memory. | Medium |
| [LMCache #19](https://github.com/local-inference-lab/lmcache/pull/19) | Contain MP future polling errors | Prevents error propagation that can leave cache in bad state. | Medium |
| [LMCache #20](https://github.com/local-inference-lab/lmcache/pull/20) | Keep shared client loop responsive | Prevents MP client loop stalls that can hang retrieves. | Medium |



**Qualification order**: #271 first (root-cause fix for the OOM we just hit),
then #397 (in the release chain), then #270/#266 (standalone hardening), then
LMCache #18/#19/#20 (cache lifecycle). With #271 merged, we may be able to
restore GMU to 0.95 while keeping 500K+ context.

---

## Qualification matrix (GPU, 4×96 GB SM120)

Progressive gates. First success criterion is **faithful load/inference/reload receipt**, not max throughput.

1. Load only, eager, no speculation, no LMCache, conservative short context
2. One-token + short deterministic generations; compare target logits/KLD
3. Reload and replay (catch reader/autotune/process-lifetime failures)
4. Streaming TTFT and ordinary chat
5. Built-in MTP3 acceptance and output parity
6. Language-only, then vision/multimodal
7. Tool calls, structured output, prompt logprobs, adversarial request validation
8. CUDA graph decode after all required K6 shapes safely primed
9. Prefill dispatch and scratch-arena A/B (speed + KV capacity)
10. Concurrency and mixed prefill/decode
11. FP8 KV qualification, then NVFP4 KV as separate arm
12. Context ramp: 32K → 64K → 128K → 256K → 512K → higher only if accounting justifies
13. LMCache L1, then L2/offload, with failure/recompute/deadline tests
14. Long soak: GPU memory, temp, clocks, throttling, power, MTP acceptance, throughput, cache errors, restarts

**Keep a no-LMCache qualification arm** so a cache-connector issue cannot be mistaken for a model-loader issue.

First serving-reader receipt must bind: checkpoint SHA, image digest, vLLM/B12X/exllamav3 trees, TP topology, toolchain, launch args, logs, outputs, test artifacts. Only then update `qualified_tp_sizes` and `serving_reader_qualified`.

---

## Open decisions (need Michel's input)

1. **Bring-up base**: Brandon v84 digest-pinned, or rebuilt source-equivalent?
2. **Final base**: PR #28, its successor/merge, or another exact GLM-5.3 snapshot?
3. **Repository**: generalize `glm52-exl3-vast` or create `glm-flash-turnkey` / `glm53-flash-turnkey`?
4. **First release scope**: language-only MTP3? (Recommended: yes. DFlash2/multimodal as later profiles.)
5. **LMCache**: in first public image or second overlay? (Recommended: qualify bare reader first, then add LMCache.)
6. **Codename**: **Auric Apotheosis** (recommended) vs Trellis Transcendence vs Auric Aegis vs Trellis Ascendant
7. **KV fidelity**: FP8 KV first, NVFP4 only after separate gate; int8 embeddings and FP8 prefill default off?

---

## Infrastructure notes from Phase 1 (apply to K6 deploy)

- **JarvisLabs VM hardening**: mask `nvidia-dcgm.service` BEFORE first boot (port 5555 conflict with vLLM ZMQ). See `SESSION-NOTES.md` §11.
- **P2P driver override**: required for PHB topology. `/etc/modprobe.d/nvidia-p2p-override.conf` with `ForceP2P=0x11`.
- **Disk budget**: K6 checkpoint is ~253.5 GB (smaller than GLM-5.2's 328 GB). 435 GB VM disk has more headroom.
- **VM cost**: $1.89/hr on-demand (4× RTX-PRO6000). Destroy after qualification.
- **Docker auth**: `sudo cp ~/.docker/config.json /root/.docker/config.json` for root docker push.
- **AIBeast**: NEVER touch production (10.15.0.166). Use JarvisLabs for K6 GPU work.

---

## Secondary backlog (not first-K6-load blockers)

- Cooperative cache-aware preemption/QoS modeling (time-varying LMCache residency)
- Peer-review LMCache PR #4612 against production MP connector
- Bounded, aged, cache-aware waiting order (vLLM #51384/#51375 prior art)
- vLLM #52054: chunk prompt-logprobs logits (memory/DoS)
- vLLM #53964: request-static YaRN profiles for mRoPE
- exllamav3 #284: fused additive EXL3 kernels (performance follow-up)

---

## Definition of done (ultimate K6 appliance)

- [ ] Every workspace change classified and preserved or discarded with reason
- [ ] Active appliance tree clean, based on intentional current source line
- [ ] Every patch in a `local-inference-lab` repo with exact commit/result-tree provenance
- [ ] No stale/superseded patch silently reapplied
- [ ] K6 checkpoint and serving receipts internally consistent
- [ ] TP4 reader qualification passes: load, generation, reload/replay, streaming, graph, MTP, vision (if enabled), tool/structured-output, context-ramp, soak
- [ ] LMCache behavior separately qualified (cannot hide base-reader failure)
- [ ] Image digest, SBOM/licenses, source manifest, checkpoint SHA, launch profiles, test receipts published
- [ ] Production untouched until Michel explicitly authorizes deployment
- [ ] Mechanical version (`glm53-k6-tp4-r1`) and human codename chosen and documented
