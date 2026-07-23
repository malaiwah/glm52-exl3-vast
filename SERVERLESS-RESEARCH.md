# Serverless / community-endpoint research (2026-07-23)

Question: can vast.ai Serverless (or anything else) give the community a stable,
scale-to-zero GLM-5.2 URL on top of our turnkey template? Research below; raw
source links at the end of each section.

## Vast.ai Serverless — how it works

- Hierarchy: **Endpoint** (scaling params) → **Workergroup** (template hash +
  offer search filters) → **Workers** (instances from the template).
- Stable OpenAI-compatible URL out of the box: `https://openai.vast.ai/<ENDPOINT>`,
  `Authorization: Bearer <owner's vast API key>`, SSE streaming supported.
  Alternative: `run.vast.ai/route` returns a signed per-request worker URL.
- **PyWorker effectively required**: their Python shim between router and model
  server (payload validation, token-cost accounting, startup benchmark,
  readiness from logs, load reporting). Ready-made vLLM worker exists
  (github.com/vast-ai/pyworker). Raw-HTTP workers undocumented/unsupported.
- Scale-to-zero: `min_load=0` + positive `inactivity_timeout` (+`cold_workers`).
- Cold-start answer = **cold workers**: STOPPED instances kept in reserve, disk
  (weights) retained at storage+bandwidth rates; reactivation pays container
  start + model load + JIT only. No VRAM-snapshot tech (no FlashBoot analog).
- Autoscaler runs a **learning phase**: recruits + benchmarks a VARIETY of
  candidate machines from search_params → for us, several 332 GB downloads
  up front. `target_queue_time`/`target_util`/`cold_mult` are the main knobs.
- Billing: normal instance rates, endpoint owner pays everything; callers use
  the owner's API key (no per-request billing, no scoped keys documented).
- GPU targeting: search_params supports gpu_name + num_gpus eq 4 — expressible.
- **Nobody publicly runs multi-GPU TP workers on it.** All examples are 1-GPU
  8B-class. TP4 is not forbidden, just unproven.
- Risk: stopped instances can fail to restart if another renter took the GPUs
  ("scheduling" limbo). Cold worker = cached weights, NOT cached capacity.

Docs: docs.vast.ai/serverless (architecture, serverless-parameters,
managing-scale, worker-states, pricing, openai-compatible-api, vllm,
creating-new-pyworkers), github.com/vast-ai/pyworker.

## Prior art — nobody does 300B+ serverless; they avoid it

- **RunPod Serverless**: up to 8-GPU workers incl. RTX PRO 6000 class;
  TENSOR_PARALLEL_SIZE first-class. FlashBoot = probabilistic warm-cache
  (600ms best case ONLY on hit); misses pay full load. Network volumes SLOW
  (made an 8B cold start worse, ~2 min; issue worker-vllm#111). "Cached
  Models" (curated HF list, placement on hosts with weights local) is the
  right primitive but likely excludes a custom 332 GB EXL3 repo. No public
  300B+ serverless deployment found. Idle = $0; network volume $0.07/GB/mo.
- **Modal**: GPU memory snapshots (CUDA+VRAM checkpoint) 9-10x on small
  models; best independent number 27B FP8 460s→70s (vLLM sleep mode +
  snapshot + compile cache). At our scale snapshot ≈ VRAM size (300GB+) —
  moves the problem, doesn't solve it. TP>1 snapshot support UNKNOWN.
- **Baseten**: serves DeepSeek-R1/V3 671B multi-node, but always-on. Their
  Delivery Network = 3-tier weight cache (node NVMe → peer cache ring →
  blob origin, >2 GB/s/node) — the production blueprint for weight movement.
- **Replicate**: 3-5 min cold boots for big models; wins are adapter-swap
  fine-tune boots and torch.compile artifact caching (idea relevant to us).
- **Petals**: dead (405B/8x22B "not enough servers") — no incentives for GPU
  donors. **AI Horde**: alive via kudos economy, caps ~70B (single-volunteer-
  GPU granularity). **OpenRouter**: requires 95%+ uptime — incompatible with
  scale-to-zero unless a warm floor exists.
- Industry answer to "serverless 671B" = DeepInfra/Fireworks/Together model:
  **aggregate demand so it never scales to zero**.
- Cold-start tech consensus: local NVMe + parallel/streamed loading
  (NVIDIA Run:ai Model Streamer in vLLM: ~10 GB/s from NVMe → 332 GB ≈ 35 s
  theoretical floor; s5cmd/aria2 parallel range reads from R2/B2 = 5-20x
  faster than HF Hub's ~40-100 MB/s/connection). Image-baked weights at
  300 GB = anti-pattern (registry limits, build cache breakage).
- Our exact niche (300B+ on marketplace GPUs, turnkey) appears UNOCCUPIED.

## Recommendations (ordered)

1. **R2/B2 weight mirror + parallel range-read download in the entrypoint**
   (s5cmd or aria2c fallback to HF). Turns first boot 30-60 min → 2-6 min on
   1-3 Gbps hosts. Helps the plain template AND any serverless future.
   Cost: R2 ~$5/mo for 332 GB, zero egress fees.
2. **Document "scale-to-warm" for individuals**: personal vast endpoint or
   just a stopped instance; weights parked at storage rates (~$20-60/mo,
   host-dependent); honest about GPU-reclaim risk; ntfy push on ready.
3. **Community stable URL = "Horde router, marketplace muscle"**: tiny
   always-on VPS owns the URL (our deSEC/TLS design ports directly), queues
   requests, spins template instances on demand, publishes honest tiers:
   warm (s), NVMe-cold (~3-8 min), full-cold (2-6 min post-mirror). Funding:
   metered cost-share or patron pool — Petals proves volunteering won't hold
   a 4x96GB floor.
4. **Scoped vast-serverless pilot** (optional): PyWorker into our image, one
   endpoint, workergroup num_gpus=4 + RTX PRO 6000 filters, min_load=0,
   cold_workers=1. ~$30-50 rental budget. Would be the first public TP4
   serverless data point. Blockers to watch: benchmark-on-boot cost,
   openai.vast.ai proxy behavior on multi-minute streams (UNKNOWN), learning-
   phase multi-download.

## Key unknowns

- vast: TP4 workers in practice; proxy timeout on long streams; volume
  support in workergroups; cold-worker replacement policy on GPU loss.
- RunPod Cached Models eligibility for custom 332 GB repos.
- Modal snapshots at TP4/300GB. EXL3 + streamer-style loaders.
