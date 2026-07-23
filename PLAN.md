# GLM-5.2 EXL3 vast.ai turnkey template — build & release plan (2026-07-23)

Michel's spec: "anyone can reference it to start a vast.ai instance (container)
with 4x RTX6000 Pro and get a 512K usable context GLM-5.2 inference endpoint,
EXL3 weights + FP8 KV cache, maximum-performance MTP, DRAM offload auto-sized
to 70% of host memory at startup, single click. SSH per vast standards. vLLM
logs on container console (vast UI grabs stdout). No needle self-test baked,
no tee/tail. Tune default MAX_NUM_SEQS by benchmarking where perf drops."

## Status
- [x] Research: vast "docker ENTRYPOINT" launch mode = image runs as designed,
      stdout = vast console, SSH works with arbitrary entrypoint images
      (docs.vast.ai templates + vast-ai/base-image README). Best practices:
      stdout logging, no secrets in public template env, weights on first boot.
- [x] entrypoint.sh + Dockerfile written and built locally
      (/mnt/fast/build/glm52-vast/, local tag glm52-exl3-vast:test).
      Base: verdictai/glm52-exl3-sparkinfer@sha256:bfd6d667 (pinned digest).
      Adds: hf_hub+hf_transfer, nvtop, htop, entrypoint.
- [ ] Registry: docker.io login MISSING on AIBeast (Michel to run
      `podman login docker.io` himself). Interim: can push test tag to ghcr.io
      via gh auth token. Final name TBD: docker.io/malaiwah/glm52-exl3-vast.
- [ ] Rental validation (doctrine, direct-image mode: --image OURS --ssh,
      ENTRYPOINT launch): boots turnkey (auto-download weights), 512K KV
      confirmed (blocks-override 2048 => 524,288 tokens fp8 — proven in Arm B),
      offload engages at 70% host RAM, MTP3 gate (accept rate + stability;
      fallback MTP2 then 1), needle 150/190/250 + deep ~440-500K, PP sweep,
      **concurrency sweep C=1..32 to pick default MAX_NUM_SEQS** (find knee
      where aggregate throughput stops scaling / per-stream collapses).
- [ ] Bake benchmark-chosen MAX_NUM_SEQS default, rebuild, final push, run-7
      evidence gist (template + numbers + instructions), update doctrine ledger.
- [ ] Template instructions for README/gist: vast template fields = image ref,
      launch mode ENTRYPOINT, docker options `-p 8000:8000 --ipc=host`,
      env (optional: HF_TOKEN, OFFLOAD_FRACTION, MTP_TOKENS, MAX_NUM_SEQS),
      disk >= 400 GB, recommend >=1Gbps-net hosts (first-boot download ~332GB).

## Key design decisions (with why)
- fp8 KV NOT nvfp4: nvfp4_ds_mla silently corrupts >~150K on stock-driver
  hosts (root-caused, gist cae272443a); fp8 clean to 440K everywhere tested.
- 512K via --num-gpu-blocks-override 2048 (=524,288 tokens KV exactly) at
  util 0.93 — Arm B validated on 95.01-GiB rental cards; no util bump needed
  (0.96 runtime-OOMs mid-prefill on rentals).
- --no-async-scheduling: publisher v2 correctness guard; MTP probabilistic
  k=3 default pending rental gate (their MTP1-greedy guard looks conservative;
  shared-YAML ran MTP2+async on v1 cleanly; we keep async off + push k).
- Offload fraction 0.70 of MemTotal computed at boot; OffloadingConnector
  requires PYTORCH_CUDA_ALLOC_CONF cleared (expandable_segments conflict).
- interleave 64 + ag_rs-hybrid a2a: publisher contract, fp8-validated clean.
- VLLM_ENGINE_READY_TIMEOUT_S=2400: first-boot JIT exceeds 5-min default.

## Costs so far: EXL3 campaign ~$85 rentals; this validation ~$8-15 expected.
