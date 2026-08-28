# AIBeast r33 live-port E2E benchmark — 2026-08-10

Status: three complete trials against the production endpoint. These are
real-service observations, not uncontaminated laboratory maxima. Production
traffic remained enabled, every trial is retained, and the tables report the
per-cell median with the best observed result in parentheses.

## Exact serving stack

- appliance image: `localhost/glm52-turnkey:r33-memory-stack-vllm277-qual-v1`
- immutable base: Gilded Gnosis v20 r33,
  `voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`
- vLLM: `0.11.2.dev280+gilded.gnosis.v20.vllmfa13d33.b12x06db0f4.fi1ac6942.cu132.20260809.r33`
- appliance overlays: vLLM #258/#270/#271 plus the direct mixed-Trellis
  loader from vLLM #277 at `df004fb4a6bfb39d2f67c6eb96313f55cea8c65b`
- B12X integration: r33 tree `06db0f4b27dbd19eb934da0da27eff7a7c49d8c4`
  with the appliance integration commit
  `9bbae67841e4818e7472e1edcdca8ebcbda68611`
- checkpoint: `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` at
  `a350292cb2038f2c31732569a711a89e5d72fd46`
- model: online EXL3 Trellis K6; eligible dense/shared-expert projections in
  MXFP8; routed experts remain EXL3; protected attention projections and
  `lm_head` remain unconverted
- topology: TP4 / DCP4 / native MTP3, probabilistic proposals, standard
  rejection sampling, B12X sparse MLA attention and MoE target path, Triton
  draft MoE
- scheduler: async scheduling, C12, 3,072 max batched tokens, graph/Trellis
  ceiling 48, chunked prefill and prefix caching
- KV: dynamic-token NVFP4 DS-MLA with FP8 RoPE, exactly
  `4,518,907,904` bytes/rank and 520,192 logical tokens
  (2,032 global blocks x 64 tokens x DCP4)
- LMCache: 125 GiB DRAM L1 plus bounded 512 GiB NVMe L2
- PCIe: custom C++ all-reduce enabled, 24 MiB lossless DMA threshold, TP4
  remote-push disabled, GPU order `2,1,0,3`
- default API reasoning effort: `high`; benchmark requests used the synthetic
  streaming protocol below with EOS ignored and 1,024 output tokens

## AIBeast hardware and power profile

- 4x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported
  memory/card; NVIDIA driver 595.71.05; CUDA 13.2
- AMD Ryzen Threadripper 9970X, 32 cores / 64 threads; 256 GiB host RAM; one
  NUMA node
- all four GPUs connect through PCIe within the same NUMA node (`NODE` in
  `nvidia-smi topo -m`); there is no NVLink
- LACT: 375 W/card, configured +1,000 MHz core offset with a 2,700 MHz ceiling,
  configured +6,000 MHz memory offset (16,365 MHz observed memory clock), and
  per-slot fan curves

## Protocol

The benchmark was
`local-inference-lab/llm-inference-bench` 0.4.29 at
`86cf05c2f42f4d21b909b6e684424ca1aab89fd5`. All three runs used the same
command against the live host-network endpoint:

```bash
python llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model local-primary \
  --concurrency 1,4,8,12 \
  --contexts 0,16k,64k \
  --duration 30 \
  --max-tokens 1024 \
  --run-burst \
  --burst-requests-per-concurrency 3 \
  --display-mode plain \
  --hw-monitor-interval 0.5
```

The default phase is 30-second sustained decode. The finite-burst/E2E phase
then submits three waves per concurrency and measures
`sum(completion_tokens) / profiling_wall_time`, including admission,
scheduling, prefill/cache behavior and completion. The benchmark discovered
the engine's 520,192-token aggregate budget and therefore omitted 64K C8/C12;
those cells do not fit concurrently and would measure scheduler deferral rather
than the requested concurrency.

The service, prefix cache and LMCache were not cleared between repetitions.
This is the naturally warm production posture achievable after ordinary use.

## Finite-burst E2E results

Values are aggregate output tok/s, median across all three trials with best
observed in parentheses. Every represented burst request completed and the
sum of `num_errors` was zero in every cell.

| context | C1 | C4 | C8 | C12 |
|---:|---:|---:|---:|---:|
| 0 | 99.7 (99.7) | 203.5 (248.2) | 298.3 (312.7) | 358.4 (360.9) |
| 16K | 89.5 (92.2) | 200.2 (226.2) | 296.6 (315.9) | 365.6 (366.3) |
| 64K | 46.6 (90.7) | 189.7 (223.0) | capacity-limited | capacity-limited |

Median TTFT at 0 context was 0.224 / 0.357 / 0.466 / 0.660 seconds at
C1/C4/C8/C12. At 16K it was 0.302 / 0.380 / 0.568 / 1.299 seconds. The 64K
C1 aggregate is the most contamination-sensitive cell: its three observations
spanned 24.8--90.7 tok/s, so the best value should not be treated as a
guaranteed single-user result.

## Sustained decode corroboration

The sustained phase reached median aggregate throughput of 372.1 tok/s at
0-context C12 (399.9 best) and 375.1 tok/s at 16K C12 (386.9 best). The cleanest
coarse vLLM ten-second windows observed 433--437 aggregate tok/s at C12. Those
coarse windows corroborate momentary capability but are not substituted for
the client-measured E2E rows above.

## Live-traffic disclosure and stability

| trial | UTC window | benchmark completions | Hermes/live-agent completions | other live completions | min free/rank | max GPU temp | max card power |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 01:01:54--01:30:32 | 371 | 53 | 3 | 749 MiB | 85 C | 384 W |
| 2 | 01:31:33--01:58:54 | 373 | 29 | 18 | 675 MiB | 84 C | 388 W |
| 3 | 01:59:54--02:24:53 | 353 | 22 | 28 | 675 MiB | 82 C | 381 W |

All three trials exited zero. The container remained running with restart count
zero, `OOMKilled=false`, HTTP health 200, no request errors, no scheduler
preemptions, no CUDA/worker/connector/cache failures and no NVIDIA thermal
throttle reason.

Trial 2 exposed two useful warmup effects. A previously unseen
`SparseNSAFusedIndexerKernel` shape with `max_seq_capacity=260096` compiled on
all ranks after engine start, producing a logged latency warning and a uniform
2 MiB/rank memory step. Later, free memory moved by another 72 MiB/rank. The
separate process allocation identifies the latter strongly—but not
conclusively—as LMCache's lazy GPU staging pool reaching its historical
662 MiB/card steady footprint. It then remained exactly flat throughout the
rest of Trials 2 and 3. The final post-campaign state was 675 MiB free/rank at
idle, 0 running, 0 waiting, restart zero and OOM false.

Complete raw JSON, sampler, container/image manifests and time-bounded server
logs remain in the AIBeast working evidence root at
`/mnt/fast/build/r31-memory-stack-20260808/evidence/e2e-live/20260810T010154Z/`.
A checksum-verified archival copy is on NFS at
`/mnt/vault/llm/benchmark-evidence/aibeast/glm52-r33-live-e2e-20260810T010154Z/`.
