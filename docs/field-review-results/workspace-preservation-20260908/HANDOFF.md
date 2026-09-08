# GLM-5.3 Flash K6 Turnkey Appliance — Session Handoff

Verified and written: **2026-08-28**

Workspace: `/Users/mbelleau/Documents/GLM-5.2 Turnkey Appliance`

## Mission

Michel owns the new K6 checkpoint and intends this project to be its first real serving qualification and daily-use appliance:

- Checkpoint: [`malaiwah/GLM-5.3-Flash-TR3-6bpw`](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw)
- Build a clean, source-locked, reproducible GLM-5.3 Flash appliance around it.
- Recover and classify everything useful in the current workspace before changing or discarding anything.
- Sweep Michel's GitHub PRs for relevant work created since the last appliance session.
- Compare the K6 needs against Brandon's current GLM-5.3 Flash K4 runtime—not merely the older v44 image originally discussed.
- Track every non-upstream patch in `local-inference-lab` GitHub repositories and reference exact commits from the appliance image. The desired result is an auditable **Gilded Gnosis + Brandon-functionality overlay**, not an opaque one-off container.
- End with a working tree whose contents and provenance are understood, then build and test the new “ultimate appliance” checkpoint.
- Give this release and future releases a durable name/version scheme.

The immediate job in the new session is **triage and preservation first**, then implementation. Do not begin by resetting, cleaning, rebasing, rebuilding, or launching production.

## Executive summary

1. The workspace root is not a trustworthy canonical Git repository. Its `main` branch has no commits and most content is untracked. The root `README.md` is unrelated Venti LoRA quantization material. Treat the root as a container of nested worktrees until it is deliberately reorganized.
2. The principal old appliance repository is `glm52-exl3-vast-r31/`. It contains valuable uncommitted r31/r33-era overlay work, but it is dirty and its branch diverges from current GitHub `main`.
3. Public Gilded Gnosis did **not** publish a normal merged r34. The latest merged public Gilded Gnosis line is **v20 r33** (blackwell-llm-docker PR #20). The “v20 r34” Michel remembers is real as an **exact local/AIBeast maintenance composition**, not as the latest public merged release.
4. Brandon's GLM-5.3 EXL3 runtime advanced rapidly. The original reference was v44; the current public repository now documents a **v84** image with TP2/EP2/DCP2, multimodal DFlash2, language-only DFlash2, and language-only MTP3 profiles.
5. `local-inference-lab/blackwell-llm-docker` PR #28 is an open, source-locked GLM-5.3 runtime build. It targets NVFP4 plus DFlash2 rather than EXL3 K6, but its GLM5Next, CUDA/PyTorch, graph, vision, toolchain, and backend work is a much cleaner forward foundation than blindly extending the old r31 branch.
6. The K6 checkpoint is sealed and high quality, but it intentionally has **no audited serving-reader/TP receipt yet**. That missing receipt is the primary qualification goal.
7. The highest-priority K6 patches are already visible upstream: B12X #245 and vLLM #314/#316/#318/#397. Their stack and applicability to GLM-5.3 must be validated rather than copied wholesale from a different dense checkpoint.
8. Recommended two-step strategy:
   - Use a digest-pinned Brandon v84-derived bring-up image to minimize time to the first K6 TP4 load/inference receipt.
   - In parallel, construct the final source-locked appliance from the current `local-inference-lab` GLM-5.3 lineage, importing only proven K6/EXL3 and maintenance overlays by exact commit.

## Non-negotiable safety and scope

- The live GLM-5.2 production service on AIBeast is out of bounds for appliance experiments without separate authorization.
- AIBeast production host: `10.15.0.166`.
- Do not stop or restart its serving container, change its power cap, start a GPU maintenance window, or replace its image without explicit authorization.
- A recurring monitor named `aibeast-glm-memory-guard` watches that service. At the latest checked interval it was healthy and materially unchanged.
- Last known production container: `glm52-turnkey-r34-maint-20260815-v1`.
- Last known production image: `localhost/glm52-turnkey:r34-aibeast-maintenance-v1`.
- Production port: `8000`; restart count `0`; `OOMKilled=false`; configured host cap `375 W`; maximum model length `520192`.
- Latest known monitor evidence: `/mnt/fast/build/r31-memory-stack-20260808/evidence/monitor/20260828T195044Z-heartbeat.txt` plus its supplementary GPU-summary correction.
- Use a separate authorized host/window for K6 GPU work. A normal local inspection, CPU/static test, branch creation, or image-manifest analysis is safe.

## New K6 checkpoint: exact state

Hugging Face repository:

- Repository: [`malaiwah/GLM-5.3-Flash-TR3-6bpw`](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw)
- Public, ungated, MIT.
- Current Hugging Face repository SHA: `0072eeb95fb328a5a09ca4965cac565f4e8fcd80`.
- Last modified when checked: `2026-08-28T15:23:59Z`.
- Internal `revision.txt`: `b1967181a3917ae70a437f4884748f6b8e3a1f4d`. Do not confuse this encoder/source revision with the Hugging Face repository SHA.
- Architecture: `Glm5NextForConditionalGeneration` / `glm5_next`, image-text model.
- Maximum position setting: `1,048,576`.
- 45 layers: 34 KDA linear-attention layers and 11 sparse/MLA layers.
- 288 routed experts, top-8 routing.
- Approximately 253.5 GB across 120 safetensors shards, roughly 77% of official FP8 size.
- Routed experts and MTP use K6 TR3/MCG with multiplier `0xCBAC1FED` and a 96-word trellis. Non-routed weights intentionally remain native BF16.
- Sealed five-run KLD: mean `0.013723`, zero reported standard deviation, 51,175 positions per run, passing the `<0.06` gate.
- Intended first serving topology: TP4 on 4×96 GB SM120. The reference runtime's admission map routes K6 to TP4.

Known publication inconsistency to fix before calling the model fully qualified:

- The model card and `receipts/k6-five-run-kld.json` say the quality gate passed.
- `receipts/checkpoint.json` still reports `qualified=false` and `measured_mean_kld=null`.
- `exl3-mcg-storage-abi.json` still has `serving_reader_qualified=false` and `qualified_tp_sizes=[]`, correctly reflecting that no audited GLM-5.3 TP load/inference receipt exists yet.
- Reconcile the quality receipt separately from the serving-reader qualification. Do not mark the reader qualified until a real load, inference, and reload/replay test passes.

Do not download the full 253.5 GB checkpoint merely to orient the session. Pin the SHA first, plan storage, then perform the download once on the designated qualification host.

## Baseline and lineage map

### 1. Public Gilded Gnosis

- Repository: [`local-inference-lab/blackwell-llm-docker`](https://github.com/local-inference-lab/blackwell-llm-docker)
- Current `main` when checked: `cc2ac998e8f7b5f04d4271a79e6647b4debad3db`.
- `main` has moved on to the Infernal Invocation r21 line.
- Latest merged public GLM/Gilded Gnosis release: PR #20, **v20 r33**.
- There is no normal public GitHub release/tag or merged r34 branch. Release identity is carried by PR/image lineage.

### 2. Local/AIBeast r34 maintenance composition

This is the r34 Michel remembers. It is a deliberately materialized local composition, not a public merged GG release.

Key lineage:

- Pre-r34 base: `e2666d9...`
- `blackwell-llm-docker` r34 composition: `98224d1303c1497eec26c7d92f34a6fa9a58fa82`
- vLLM PR #280 head: `8e7be4d...`
- vLLM PR #281 head: `126039a...`
- Exact resulting r34 vLLM tree: `4d006a43928cdee01306691a766542c1e9bebb59`
- Local exact-r34 materialization commit: `ef7ccc71...`
- Original vLLM PR #277 head: `df004fb...`
- r34 compatibility port: `9774994...`

The maintenance appliance adds five important behaviors to immutable GG v20 r34:

1. vLLM PR #277 compatibility port.
2. Hybrid external-cache invalid-block recovery.
3. Bounded LMCache MP retrieve completion.
4. Per-adapter filesystem L2 LRU plus 180-second deadline.
5. Expired L1 read-lease recovery, associated with LMCache PR #4691.

Primary local references:

- `evidence/r34-pr277-compat-maintenance-handoff-2026-08-10.md`
- `maintenance/r34-aibeast-20260815/README.md`
- `qualification-src/r34-pr277-compat/`
- `qualification-src/lmcache-r34-maintenance/`

### 3. Brandon's EXL3 K4 runtime

Repository: [`brandonmmusic-max/glm-5.3-flash-exl3-4bpw`](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw)

Historical reference originally discussed:

- v44 image: `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-v44`
- Digest: `sha256:15192e3930b4ae5558271ebe7d1a5a02da6dcc5a6c292c44e79a3fb8c883b5e1`
- Current model repository: [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)

Current reference when checked:

- Repository HEAD: `57db68f1db0d1fefe1dcd2b9350d3f6968c786f0`.
- Current runtime generation: **v84**.
- Image: `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2`.
- Image digest: `sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`.
- Same digest is published under the language-only alias.
- Three explicit TP2/EP2/DCP2 profiles:
  - multimodal DFlash2;
  - language-only DFlash2;
  - language-only built-in MTP3, the capacity/default profile.
- CUDA graphs and calibrated NVFP4 MLA KV are used.
- v84 includes a native PyTorch vision-RoPE fallback, corrected DFlash2 attention semantics, provenance verification, and a complete hash-verified R10 encoder closure.
- Current documented results include 32K cold prefill around 6.2K tok/s, C1 decode around 145–151 tok/s, and DFlash2 mean acceptance around 5.4–5.7 on the cited samples.

Important caveats:

- Brandon's path is uniform K4 on TP2; ours is K6 on TP4 with a different residency/topology envelope.
- The v84 target-quality evidence is inherited from v75 because the v84 changes were in speculation, vision packaging, and launch profiles, not target weights/logits.
- DFlash2 has a separate non-commercial checkpoint/license. The first K6 appliance should default to built-in MTP3 unless DFlash2 is explicitly desired and its license is acceptable.
- Do not assume the v44 digest represents current functionality. Inventory the exact v84 sources and image labels before deriving an overlay.

### 4. Current local-inference-lab GLM-5.3 source-locked runtime

- PR: [`blackwell-llm-docker#28`](https://github.com/local-inference-lab/blackwell-llm-docker/pull/28)
- Title: `Build GLM-5.3 DFlash2 MXFP8 TP4 runtime`
- State: open.
- Head branch: `release/glm53-dflash2-mxfp8-20260828`.
- Head SHA when checked: `d3920aaa7c0546594d1052ec3dd530732e739719`.
- Qualified topology: 4× RTX PRO 6000 Blackwell, TP4.
- Target is NVFP4 and draft is DFlash2 MXFP8, not EXL3 K6.
- CUDA 13.3, PyTorch 2.13.
- vLLM base `015dcd...`, immutable snapshot PR #491 `b77333...`, result tree `82bbab85...`.
- B12X base `2fcf23...`, PR #250 `dd8cf605...`, result tree `fdbb504...`.
- Current Docker Hub image manifest: `sha256:1a210a6dcd4eeef4b9515aa03b8117dbd1bad70ed101cdf72ec4692015e2ac4b`.

What to reuse from #28:

- `Glm5Next` model wiring and config handling.
- TP4 GLM-5.3 graph/runtime/toolchain integration.
- GLM GDN, vision, multimodal, DFlash/MTP, DeepGEMM/B12X, CUDA 13.3, and PyTorch 2.13 compatibility work.
- Source-locking, immutable snapshots, image provenance, and qualification style.

What must change or be added for K6:

- EXL3/TR3/MCG loader and storage-ABI path.
- K6→TP4 admission and shard planning.
- K6/MCG B12X fused decode and safe fallback.
- Dense/online EXL3 prefill dispatch and scratch budgeting.
- K6 graph warmup/capture contracts.
- The K6 checkpoint's BF16 non-routed weights and routed-expert/MTP split.
- LMCache and appliance control-plane features if they are kept in the first release.

## Current workspace inventory

### Root

- Root Git branch `main` has no commits.
- Most files/directories are untracked at the root.
- Root `README.md` is unrelated Venti LoRA material.
- Do not make a sweeping root commit. First decide whether the root becomes a manifest/orchestration repository or remains a workspace container.

### Primary old appliance worktree: `glm52-exl3-vast-r31/`

- Remote: `https://github.com/malaiwah/glm52-exl3-vast.git`
- Branch: `codex/r31-vllm258`
- HEAD/upstream: `6256adc648887eb1f99e2d9d808378b6eabb7838`
- Dirty: 14 tracked modified files, approximately 535 insertions and 50 deletions, plus numerous untracked evidence/upstream-draft documents.
- Branch relationship to GitHub `main`:
  - branch has 3 commits not in `main`;
  - remote `main` has 1 commit not in the branch: PR #51 at `b0012efaf13f7fc2100bcf0bed83404663ad8e9f`;
  - merge base: `eba43a533969be45547e4b4fb8ebecc547300352`.

Tracked modifications:

- `CHANGELOG.md`
- `Dockerfile`
- `README.md`
- `entrypoint.sh`
- `patches/v20-r31-memory-stack/ledger.json`
- `patches/v20-r31-memory-stack/manifest.json`
- `patches/v20-r31-memory-stack/vllm-258-270-runtime.patch`
- `scripts/glm52_lmcache_wrapper.sh`
- `scripts/glm_config.py`
- `scripts/run-local-podman.sh`
- `scripts/verify_r31_base.py`
- `tests/test_families.py`
- `tests/test_lmcache_wrapper.sh`
- `tests/test_r31_base_gate.py`

Untracked material worth triaging, not deleting:

- AIBeast r33 live and maintenance evidence.
- GLM-5.2 prefill/serving research.
- r31 3.42 qualification notes.
- Multiple upstream issue/PR comment drafts for B12X, vLLM, LMCache, MRV2, projection cones, fused norm/RoPE, and profile scratch.
- `.DS_Store` is disposable only after everything else is classified.

The dirty overlay expands the old r31 stack to:

- vLLM #258: `63b77c8031499bdb360540871fe63c487cc04908`
- vLLM #270: `244d85a6fe99eca9b9b4180638334a4486bde16a`
- vLLM #271: `310c3ac9718f8d28a6d2c6e009ca4376e54f4457`
- B12X #130: `9ead9eaa188c2d36f091c8e5225e196896545721`
- Resulting vLLM tree: `d1ef769b8509bf80bb6ba63ea2652b19bb556a09`
- Proposed old release string: `GG-v20-r31+vLLM-258-270-271+B12X-130`

`git diff --check` currently reports nine trailing-whitespace errors in the large vLLM patch. Do not call this tree clean or release-ready.

PR #51 on current `glm52-exl3-vast` main is a substantial safety/config/TLS/vision/SOUL/termination/patcher/docs bundle. It must be reviewed and normally become the clean starting point before replaying older work.

### Focused nested source worktrees

Preserve all of these; several are the canonical patch implementations:

| Worktree | Branch / HEAD | State and significance |
|---|---|---|
| `qualification-src/r34-pr277-compat` | `codex/r34-aibeast-maintenance` / `a25483af...` | Clean. Exact r34 materialization, PR #277 port, hybrid-cache recovery. |
| `qualification-src/lmcache-r34-maintenance` | `codex/r34-retrieve-deadline` / `67561538...` | Clean. Bounded MP retrieve plus expired L1 lease recovery. |
| `qualification-src/lmcache-retrieve-deadline` | `codex/lmcache-retrieve-deadline` / `a99c1ad...` | Clean but ahead 1/behind 51 upstream `dev`; reconcile, do not assume current. |
| `qualification-src/lmcache-read-lease-recovery` | `codex/mp-expired-read-lease-recovery` / `8e54b34...` | Clean focused lease-recovery branch. |
| `qualification-src/vllm-lmcache-deferred-spin-fix` | `codex/fix-lmcache-deferred-spin` / `df004fb...` | Clean focused deferred-spin work. |
| `qualification-src/vllm` | `codex/memory-stack-20260808` / `5b42f8e...` | Clean memory-stack source. |
| `qualification-src/b12x` | `codex/memory-stack-20260808` / `f44fc8f...` | Clean B12X memory-stack source. |
| `qualification-src/r31-vllm` | old r31 branch | Massively staged/unstaged; salvage separately, never use as canonical without classification. |
| `qualification-apply-smoke/vllm-source` | detached | Six modified files; likely qualification residue. |
| `upstream-prewarm/vllm-work` | `codex/exl3-runtime-memory` / `244d85...` | Clean vLLM #270 implementation. |
| `upstream-prewarm/vllm-b12x-ckv-prefill-warmup` | / `310c3ac...` | Clean vLLM #271 implementation. |
| `upstream-prewarm/vllm-prompt-logprobs-fix` | / `63b77c...` | Clean vLLM #258 implementation. |
| `aiboss-r28-warmup/b12x-pr126` | `codex/mixed-trellis-buffer-layout` | Clean B12X #126 work. |
| `sparkinfer-pr123-review` | / `1bab07...` | Clean review tree. |
| `spark-occupancy-fix` | local-only | Preserve; local remote path. |
| `spark-w4a16-boundary-fix` | local-only | Preserve; local remote path. |
| `glm52-342-profile` | `codex/glm52-342-qualified-profile` / `90e906...` | Clean; upstream gone, so preserve explicitly. |

`upstream-prewarm/vllm` and `upstream-prewarm/sparkinfer` appear to have whole worktrees staged as deleted. These may be sparse/worktree artifacts. Do not “clean” or restore them blindly; inspect their worktree metadata first.

## GitHub PR triage ledger

This is the initial relevance sweep, not permission to stack every open PR. For each candidate, compare it against the exact chosen base and classify it as `include`, `already upstream`, `superseded`, `experiment`, or `not applicable`, with an exact reason and commit.

### Release-critical K6/EXL3 candidates

1. [`local-inference-lab/b12x#245`](https://github.com/local-inference-lab/b12x/pull/245) — **highest priority**.
   - Open; head `1a3ea83fbb16d7b800a7d50e6b19a5710eaf30c0`.
   - Generalizes cooperative fused Trellis decode to K2–K6 MCG.
   - Stacked on B12X #243 and says to merge after it.
   - Its body still says full RTX 5090 results are forthcoming. Do not treat it as GLM-5.3 K6 qualified yet.
2. [`local-inference-lab/vllm#314`](https://github.com/local-inference-lab/vllm/pull/314) — graph-decode priming.
   - Open; head `7917c9283bf6ffea75c3a53c37071e54c035d168`.
   - Enables dense EXL3 decode graphs by pre-priming finite GEMM autotune buckets.
   - Proven on a dense checkpoint; GLM-5.3 MoE/K6 applicability must be checked, especially expert concurrency and graph mode.
3. [`local-inference-lab/vllm#316`](https://github.com/local-inference-lab/vllm/pull/316) — reconstruct+hgemm for prefill.
   - Open; head `8451183e139451a2e0a779919547aa175c559cea`.
   - Reports +113%/+118% prefill with decode unchanged on the cited dense checkpoint.
   - Changes FP16 summation order; carries an opt-out and disclosed small numerical difference.
4. [`local-inference-lab/vllm#318`](https://github.com/local-inference-lab/vllm/pull/318) — route K6/MCG away from the B12X decode kernel at prefill row counts.
   - Open; head `2b96dad45b2c9a1dd59b4bf3f9f33b06cb70f42a`.
   - Stacked on #316.
   - Directly relevant to K6, but measured on Qwen3.8, not this GLM-5.3 checkpoint.
   - Contains an unrelated opt-in int8 embedding overlay; separate that concern before inclusion.
5. [`local-inference-lab/vllm#397`](https://github.com/local-inference-lab/vllm/pull/397) — shared reconstruct-scratch arena.
   - Open; head `96972e10f5bf11a47a09b3ca52628b2f402c0b7e`.
   - Stacked on #318; review only its final arena commit when evaluating the delta.
   - Strong memory/KV-capacity relevance; reports recovering about 0.60 GiB on its qualification model.

These five form a dependency/validation chain, not five independent cherry-picks:

`B12X #243 → #245` and `vLLM #314 → #316 → #318 → #397`.

Rebase the chain onto the chosen GLM-5.3 source tree, split unrelated features, and run focused CPU/static tests before any GPU build.

### Additional vLLM candidates to inspect

- #455: opt-in int8 embeddings. Capacity option, not a default for the first K6 fidelity release.
- #454 plus exllamav3 #299: per-layer split/fused-uniform QKV PoC. Include only if the K6 checkpoint topology actually needs it.
- #439: skip speculative decode for requests needing at most one output token. Useful but not a K6 blocker.
- #403: hybrid+connector local-hit gate. The PR itself calls the condition necessary but not sufficient/refuted; research only.
- #398: persistent decode wrappers keyed by shape. Potential multi-shape/graph safety relevance.
- #393: cherry-pick upstream #51113/#51812. Check whether the Jovian/GLM-5.3 base already contains them.
- #312: preserve BF16 unsupported shards. Assess carefully because this K6 checkpoint deliberately keeps non-routed weights BF16.
- #306: old PR #228 release checks. Likely incorporated/superseded; verify.
- #299: hot-swappable MSRT cartridges. Not required for first K6 release.
- #277: mixed Trellis direct tier slabs. Strong load-memory relevance; already ported into local r34.
- #270/#271: native prefill buffer sharing and CKV prewarm from the older GLM-5.2 stack. Inspect for supersession and GLM5Next applicability.
- #268/#266/#265: security changes. Confirm whether the selected modern base already includes equivalent fixes.
- #258: prompt-logprobs accounting, merged on the old line. Confirm equivalent upstream behavior.
- #249: heterogeneous expert widths. Potential architecture relevance; inspect against GLM-5.3 shapes.
- #307/#269: research/draft quantization ideas; not first-release dependencies.

### B12X candidates

- #245: direct K6 requirement, above.
- #130: persistent buffer layout from the old GG overlay; check for supersession.
- #126: mixed Trellis prewarm; old but potentially useful prior art.
- #123: native GLM H8 NVFP4 decode; comparison/prior art, not the K6 target path.
- Merged security PRs #183–#190/#203/#207/#208 are probably already in a modern base. Audit commit ancestry instead of reapplying closed work.

### LMCache candidates

- #4691: expired L1 read-lease recovery; draft and used in the production maintenance overlay.
- #4600: mark failed retrieves as load errors and recompute; open, relevant resilience.
- #4517: bounded MP retrieve completion; open and used in the production maintenance overlay.

LMCache should be added after the bare K6 reader/runtime works. Keep a no-LMCache qualification arm so a cache-connector issue cannot be mistaken for a model-loader issue.

### Upstream vLLM/exllamav3 candidates

- vLLM #52054: chunk prompt-logprobs logits; relevant memory/DoS protection if prompt logprobs are enabled.
- vLLM #52530: closed impossible-KV-request handling; determine the current superseding behavior.
- vLLM #38261: hybrid KV offload; secondary LMCache/hybrid backlog.
- vLLM #37443: MTP `hf_overrides`; determine whether Jovian already includes equivalent support.
- vLLM #53964: request-static YaRN profiles for mRoPE; probably not an initial K6 blocker, but verify against GLM5Next position handling.
- exllamav3 #299: companion to vLLM #454.
- exllamav3 #284: fused additive EXL3 kernels; possible performance follow-up.

Michel's qwen38 PRs #2/#3/#4 are useful prior art for fused FP4/GDN but are not initial GLM-5.3 K6 blockers. Do not import them unless profiling identifies the same bottleneck.

## Recommended clean integration plan

### Phase 0 — freeze and inventory the current state

Before changing any nested worktree:

1. Record `git status --short --branch`, remotes, HEAD, upstream, worktree list, submodules, and `git diff --stat` for every nested Git repository.
2. Save full tracked diffs and a manifest/hash list of untracked files under a timestamped local evidence directory.
3. Classify each dirty tree as canonical source, experiment, qualification residue, evidence/docs, or disposable cache.
4. Do not use `git reset --hard`, `git clean`, broad checkout restoration, or recursive deletion.
5. Do not merge the root workspace into one commit.

### Phase 1 — build the source and PR ledger

Create a machine-readable ledger for every appliance delta with:

- repository;
- PR/issue URL;
- exact head/base/result commits;
- patch SHA-256 if materialized;
- current upstream state;
- dependency chain;
- classification (`include`, `already upstream`, `superseded`, `experiment`, `not applicable`);
- CPU/static tests;
- GPU qualification receipt;
- license/security implications;
- appliance/image versions using it.

All patches necessary to reproduce Brandon functionality or K6 support must live in appropriate `local-inference-lab` repositories before final release. A patch that exists only inside a Docker build layer or local worktree is not release-ready.

### Phase 2 — choose two explicit bases

Use separate identities for bring-up and release:

1. **K6 bring-up base:** digest-pinned Brandon v84 image/source snapshot, modified only enough to admit K6 TP4 and produce the first audited reader/inference receipt.
2. **Final appliance base:** current local-inference-lab GLM-5.3 source-locked lineage, likely PR #28 or its successor, with the K6/EXL3 chain rebased and fully tracked.

Do not call the bring-up image the ultimate appliance. It is a diagnostic milestone.

The old local r34 stack remains the source of proven LMCache/control-plane and memory-safety behavior, but should be harvested selectively rather than treated as the GLM-5.3 runtime base.

### Phase 3 — establish a clean appliance repository/branch

1. Preserve the dirty `glm52-exl3-vast-r31` tree first.
2. Start a clean `codex/` integration branch from current `malaiwah/glm52-exl3-vast` `main`, which includes PR #51, or create a clearly named GLM-5.3 appliance repository if the existing GLM-5.2 name is no longer appropriate.
3. Prefer a generalized product repository name such as `glm-flash-turnkey` or an explicit `glm53-flash-turnkey`; decide before public release.
4. Replay only ledger-approved commits/patches.
5. Keep model profiles declarative and versioned; do not bury K6 decisions in shell conditionals.

### Phase 4 — implement the GLM-5.3 K6 profile

Minimum profile information:

- model repository and immutable revision;
- `Glm5NextForConditionalGeneration` architecture;
- TP4 requirement for K6;
- K6 TR3/MCG routed-expert and MTP handling;
- BF16 non-routed shard handling;
- vision-on and language-only variants;
- built-in MTP3 default;
- optional DFlash2 profile kept separate with its license and resident-memory cost visible;
- FP8 and NVFP4 KV profiles qualified separately;
- maximum context and scheduler limits derived from measured capacity, not copied from K4;
- graph mode and warmup requirements;
- B12X fused path eligibility and safe fallback;
- no-LMCache and LMCache launch variants;
- provenance labels for every source tree, patch, checkpoint, toolchain, and image digest.

### Phase 5 — CPU/static and build gates

Before GPU use:

- validate the checkpoint manifests, shard count, hashes, K6 metadata, BF16 exclusions, MTP weights, and architecture fields;
- reconcile stale checkpoint-quality receipts without falsely qualifying the serving reader;
- add K6→TP4 admission tests and reject invalid TP sizes;
- test patch application from immutable bases in a clean throwaway worktree;
- test graph-shape planning, prefill dispatch, scratch-arena ownership, and fallback behavior without CUDA where possible;
- run shell/static appliance tests, image provenance checks, configuration round trips, and security/config tests inherited from PR #51;
- enforce `git diff --check` and patch SHA checks;
- build with an immutable source/result-tree manifest.

### Phase 6 — authorized GPU qualification

Run on 4×96 GB SM120, initially isolated from production.

Progressive matrix:

1. Load only, eager, no speculation, no LMCache, conservative short context.
2. One-token and short deterministic generations; compare target logits/KLD where practical.
3. Reload and replay to catch reader/autotune/process-lifetime failures.
4. Streaming TTFT and ordinary chat.
5. Built-in MTP3 acceptance and output parity.
6. Language-only, then vision/multimodal.
7. Tool calls, structured output, prompt logprobs, and adversarial request validation.
8. CUDA graph decode after all required K6 shapes are safely primed.
9. Prefill dispatch and scratch-arena A/B, measuring both speed and KV capacity.
10. Concurrency and mixed prefill/decode behavior.
11. FP8 KV qualification, then NVFP4 KV as a separate quality/performance arm.
12. Context ramp: 32K, 64K, 128K, 256K, 512K, then higher only if accounting and headroom justify it.
13. LMCache L1, then L2/offload, with failure/recompute/deadline tests.
14. Long soak with GPU memory, temperature, clocks, throttling, power, MTP acceptance, throughput, cache errors, and restarts recorded.

The first serving-reader receipt must bind checkpoint SHA, image digest, vLLM/B12X/exllamav3 trees, TP topology, toolchain, launch arguments, logs, outputs, and all test artifacts. Only then update `qualified_tp_sizes` and `serving_reader_qualified`.

### Phase 7 — release and naming

Use two identifiers:

- a mechanical immutable version, e.g. `glm53-k6-tp4-r1`;
- a human codename following the existing mythic/alliterative tradition.

Recommended first codename: **Auric Apotheosis**. It suggests the higher-fidelity K6 checkpoint and fits Gilded Gnosis / Infernal Invocation without impersonating either upstream line.

Alternatives to decide with Michel before publication:

- **Trellis Transcendence**
- **Auric Aegis**
- **Trellis Ascendant**

Future releases should keep immutable mechanical versions while receiving distinct two-word alliterative codenames. Do not encode an unverified upstream release number into the product name.

## Open decisions for the new session

1. Confirm the first bring-up base: Brandon v84 digest or a rebuilt source-equivalent image.
2. Confirm the final base: blackwell-llm-docker PR #28, its successor/merge, or another exact GLM-5.3 snapshot.
3. Decide whether to generalize `glm52-exl3-vast` or create a clean GLM-5.3 appliance repository.
4. Decide whether first release is language-only MTP3, with multimodal/DFlash2 as later profiles. Recommended: yes.
5. Decide whether LMCache is in the first public image or a second overlay. Recommended: qualify the bare reader first, then include LMCache before declaring the ultimate appliance.
6. Decide the release codename. Recommended: **Auric Apotheosis**.
7. Decide how aggressively to preserve K6 fidelity: FP8 KV first, NVFP4 only after a separate gate; int8 embeddings and FP8 prefill experiments default off.

## Secondary backlog carried from the previous session

This is important but not a blocker for the first K6 load.

- Cooperative cache-aware preemption/QoS: a large LMCache hit still requires temporary GPU KV residency, but that residency is bounded by the request's output lifetime—approximately `max_output_tokens / effective generation rate`, with load-dependent interleaved-prefill effects. Model the time-varying residency and completion benefit rather than treating the hit as permanently resident. :codex-annotation{index="1"}
- Peer-review LMCache PR #4612 against the production MP connector and post concrete findings/concerns as a GitHub review comment. This was requested but not completed here. :codex-annotation{index="2"}
- Plan a bounded, aged, cache-aware waiting order, using vLLM PRs #51384 and #51375 as prior art, and open an upstream tracking issue before implementation. Include starvation protection, hit-confidence/transfer-cost estimates, TTFT goals, preemption budget, and rollback metrics. :codex-annotation{index="3"}

## Definition of done for the ultimate K6 appliance

- Every workspace change is classified and preserved or intentionally discarded with a recorded reason.
- The active appliance tree is clean and based on an intentional current source line.
- Every patch is available in a `local-inference-lab` repository with exact commit/result-tree provenance.
- No stale/superseded patch is silently reapplied.
- K6 checkpoint and serving receipts are internally consistent.
- TP4 reader qualification passes load, generation, reload/replay, streaming, graph, MTP, vision (if enabled), tool/structured-output, context-ramp, and soak gates.
- LMCache behavior is separately qualified and cannot hide a base-reader failure.
- Image digest, SBOM/licenses, source manifest, checkpoint SHA, launch profiles, and test receipts are published.
- Production remains untouched until Michel explicitly authorizes deployment.
- Mechanical version and human codename are chosen and documented.

## First actions in the fresh session

1. Read this file completely.
2. Re-check remote SHAs and PR states; all external state above is a 2026-08-28 snapshot.
3. Inventory every nested worktree and write timestamped status/diff/untracked manifests without modifying them.
4. Create the patch/PR ledger and classify the release-critical chain first.
5. Inspect Brandon v84's provenance, launchers, and exact vLLM/B12X/exllamav3 source commits; compare them to blackwell-llm-docker PR #28.
6. Produce a concrete base-selection recommendation and dependency graph before creating the new integration branch.
7. Preserve the dirty old appliance tree, then create the clean `codex/` branch/repository.
8. Add CPU/static K6 admission and manifest tests.
9. Build a bring-up image, but do not start GPU qualification until the host/window is explicitly authorized.

## Do not forget

- “Latest GG” means public v20 r33; “r34” means our exact local maintenance composition.
- Brandon's current comparison is v84, not only the historical v44 digest.
- K6 is TP4 and currently unqualified for serving despite passing offline KLD.
- The first success criterion is not maximum throughput. It is a reproducible, faithful K6 load/inference/reload receipt with measured memory headroom.
- Preserve evidence first. Never make the worktree look clean by erasing history.
