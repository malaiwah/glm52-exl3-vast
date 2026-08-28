# AIBeast r33 live-port GPQA Diamond — 2026-08-10

Status: one complete 198-item GPQA Diamond run plus one exact 32-item
capped-cohort replay against the production port. The results measure the
deployed appliance while unrelated clients remained online. They are capped,
live-service observations rather than uncontaminated or uncapped ceilings for
the model.

## Result

| metric | observed |
|---|---:|
| accuracy | **83.84% (166/198)** |
| Wilson 95% interval | **78.08–88.31%** |
| Biology | 63.16% (12/19) |
| Chemistry | 79.57% (74/93) |
| Physics | 93.02% (80/86) |
| malformed / unparseable / request errors | 0 / 0 / 0 |
| `max_tokens=49,152` hits | 32 |
| answered correctly at the cap | 12 |
| answered incorrectly at the cap | 4 |
| cap reached without a final answer | 16 |

The 32 length-limited trajectories matter when interpreting the score. Twelve
still exposed the correct final option and four exposed an incorrect option;
the other 16 were scored incorrect because no final answer was present. No
truncated item was retried and the token limit was not reduced. The published
83.84% is therefore the exact score of this C8/49,152-token deployment test,
not a claim about what an uncapped or lower-concurrency run would score.

## Targeted capped-cohort replay

Every one of the 32 capped items was replayed exactly once at C2,
temperature zero and `max_tokens=131072`. The successful run used the same
benchmark commit, protected dataset cache, model endpoint and serving stack.
The temporary driver derived cohort membership from protected per-item result
metadata, validated 32 unique IDs, injected those exact cached items in their
original run-index order, and verified the returned membership and order. The
ordered cohort-ID hash is
`f618594dc29bd9b3b8ef85bc39386764cfcf70cd4c071fed92b060e8c91f7b14`;
IDs, questions, references and trajectories remain restricted.

The healed result unconditionally replaces all 32 original capped outcomes
with their replay outcomes, including regressions, while retaining the other
166 original outcomes unchanged. It is therefore a **two-pass targeted
capped-cohort composite**, not a single official GPQA run and not an uncapped
model ceiling.

| result | correct | total | accuracy | Wilson 95% interval |
|---|---:|---:|---:|---:|
| original C8 live-port run | 166 | 198 | 83.84% | 78.08–88.31% |
| replayed capped cohort only | 16 | 32 | 50.00% | 33.63–66.37% |
| targeted two-pass composite | **170** | **198** | **85.86%** | **80.32–90.03%** |

The original capped cohort changed from 12 correct / four wrong / 16 without
a final answer to 16 correct / five wrong / 11 without a final answer. The
net gain is four correct items, or 2.02 percentage points on the 198-item
composite. Fourteen replay trajectories still reached 131,072 tokens: one was
correct, two were wrong and 11 had no final answer. The other 18 stopped
normally: 15 correct and three wrong.

| original outcome → replay outcome | items |
|---|---:|
| correct → correct | 8 |
| correct → wrong | 1 |
| correct → no final answer | 3 |
| wrong → correct | 2 |
| wrong → wrong | 1 |
| wrong → no final answer | 1 |
| no final answer → correct | 6 |
| no final answer → wrong | 3 |
| no final answer → no final answer | 7 |

This matrix is the audit that the composite was not healed by cherry-picking:
all five regressions and all seven unchanged no-answer cases remain included.

The protected replay command was:

```bash
python gpqa_targeted_replay.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model local-primary \
  --test-profile gpqa-diamond \
  --profile-runs 32 \
  --profile-concurrency 2 \
  --max-tokens 131072 \
  --completion-stats-temperature 0 \
  --completion-stats-no-prefill-scout \
  --completion-stats-save-text \
  --display-mode plain \
  --hw-monitor-interval 0.5 \
  --output results.json
```

The driver is restricted evidence rather than public benchmark code because
it handles protected item IDs. Source commit
`86cf05c2f42f4d21b909b6e684424ca1aab89fd5` and dataset SHA-256
`a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c`
match the original run. The original and replay result hashes are
`ac2bad8f86e8bc69c4af4ab86c0495b0a6ce63b329a8e181bb25a427066c5836`
and `4ec5fb9c690e2532c6f14d413c78c6c79abe248f5762e771436e4a4b1374b29a`.

One launch attempt was terminated after about 75 seconds because the safety
monitor incorrectly matched NVIDIA's `Not Active` text as an active thermal
event. It produced no result file or completed benchmark outcome. The monitor
was corrected to compare the exact field value, the aborted evidence was
preserved separately, and the successful 32-item run then executed without a
request retry, semantic retry or omitted item.

### Replay performance and live-load disclosure

The replay generated 2,350,413 completion tokens over 15,508.248 seconds:
**151.56 aggregate output tok/s across the complete C2 wall clock**. The
benchmark's 78.09 aggregate generation tok/s divides by summed individual
request time and is not the concurrent campaign rate.

| replay metric | p25 | p50 | p75 | p90 | maximum |
|---|---:|---:|---:|---:|---:|
| completion tokens | 15,575 | 69,869 | 131,072 | 131,072 | 131,072 |
| TTFT | — | 0.342 s | — | 0.423 s | 1.066 s |
| request wall time | — | 1,054.02 s | — | 1,770.69 s | 2,225.23 s |

Completion-token minimum / mean were 3,022 / 73,450.4. All 32 requests
completed at the transport layer; request/transport errors, preemptions,
restarts and OOMs were zero.

This was an unusually contaminated production soak. Frozen before/after vLLM
counters separate the 32 replay completions from **513 unrelated live
completions**. Live clients contributed 58,918,544 prompt tokens and 369,279
generated tokens; sampled scheduler maxima were four running and two waiting
requests. Those clients materially inflate wall time and contaminate TTFT and
throughput, but they do not change the replay's per-item scoring.

High-frequency benchmark telemetry measured 99.47% average GPU utilization,
1,103.9 / 1,497.6 W average / peak aggregate GPU power and an 88 C peak GPU
temperature. Independent production monitoring found a 673 MiB/rank physical
free-memory floor. There was no NVIDIA hardware thermal slowdown, CUDA OOM,
worker or EngineCore death, traceback, connector/cache failure, preemption,
container restart or `OOMKilled` event; post-run `/health` remained HTTP 200.

Restricted replay evidence is under
`/mnt/fast/build/r31-memory-stack-20260808/evidence/gpqa-diamond-heal/20260810T123514Z/`.
`aggregate_privacy_safe.json` and `contamination_privacy_safe.json` contain no
question, reference, item ID or model response text. The raw `results.json`
and temporary targeted driver remain protected and must not be redistributed.

## Exact serving stack

- appliance image: `localhost/glm52-turnkey:r33-memory-stack-vllm277-qual-v1`
- immutable base: Gilded Gnosis v20 r33,
  `voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82`
- checkpoint: `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` at
  `a350292cb2038f2c31732569a711a89e5d72fd46`
- model path: online EXL3 Trellis K6; eligible dense/shared-expert
  projections in MXFP8; routed experts remain EXL3
- topology: TP4 / DCP4 / native MTP3 with probabilistic proposals, standard
  rejection sampling, B12X target path and Triton draft MoE
- scheduler: async scheduling, engine C12, 3,072 max batched tokens,
  graph/Trellis ceiling 48, chunked prefill and prefix caching
- benchmark load: fixed C8 for the 198-item run and fixed C2 for the replay;
  both shared the C12 production scheduler with live work
- reasoning: the endpoint's default `high`; the benchmark supplied no
  request-level reasoning override
- KV: dynamic-token NVFP4 DS-MLA with FP8 RoPE, exactly
  `4,518,907,904` bytes/rank and 520,192 logical tokens
- LMCache: 125 GiB DRAM L1 plus bounded 512 GiB NVMe L2
- GPU order: `2,1,0,3`; custom PCIe collective enabled; 24 MiB lossless DMA
  threshold; TP4 remote-push disabled
- hardware: 4x RTX PRO 6000 Blackwell, NVIDIA driver 595.71.05, CUDA 13.2
- LACT: 375 W/card, 16,365 MHz observed memory clock and the qualified
  per-slot fan curves

The conservative C8 admission estimate was 393,216 output tokens plus short
item prompts, below the live 520,192-token pool. During the campaign the
engine reached ten simultaneous requests (eight benchmark plus two live),
with zero scheduler waiting or preemption.

## Reproduction contract

The benchmark was `local-inference-lab/llm-inference-bench` 0.4.29 at commit
`86cf05c2f42f4d21b909b6e684424ca1aab89fd5`. The encrypted/cache-protected
GPQA Diamond split contained all 198 items and had SHA-256
`a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c`.
No question, reference answer or model trajectory is committed here.

```bash
python llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model local-primary \
  --test-profile gpqa-diamond \
  --profile-runs 198 \
  --profile-concurrency 8 \
  --max-tokens 49152 \
  --completion-stats-temperature 0 \
  --completion-stats-no-prefill-scout \
  --completion-stats-save-text \
  --display-mode plain \
  --hw-monitor-interval 0.5 \
  --output results.json
```

The run lasted 7,358.694 seconds and generated 2,530,367 completion tokens.
That is **343.86 aggregate output tok/s over the complete live C8 campaign**,
including admission, TTFT, generation, long-tail trajectories and batch
turnover. The benchmark's `aggregate_gen_tok_s=44.26` and
`aggregate_e2e_tok_s=44.18` divide by the sum of individual request times;
they are token-weighted per-request rates, not C8 aggregate throughput.

| latency / token metric | p50 | p90 | maximum |
|---|---:|---:|---:|
| TTFT | 0.465 s | 0.482 s | 1.230 s |
| completion tokens | 2,954 | 49,152 | 49,152 |
| request wall time | 69.13 s | 982.36 s | 1,526.96 s |

Per-request generation throughput was 44.24 tok/s at p50 and 50.93 tok/s at
p90. The unusually high completion-token and wall-time p90 values reflect the
32 deliberate 49,152-token caps, not request failures.

## Live-traffic disclosure and stability

The frozen vLLM metric delta contains 227 completed requests: 198 benchmark
items and 29 unrelated production completions. Live clients contributed a net
3,015,031 prompt tokens and 53,928 generated tokens while the benchmark was
active. Access logs recorded 42 non-local request admissions; some crossed
the evidence-window boundaries. Consequently, accuracy is exact for the 198
items, while TTFT and throughput are deliberately production-contaminated.

- benchmark request errors, unparseable responses and malformed answers: zero
- vLLM preemptions and maximum waiting requests: zero
- maximum running requests: ten
- minimum physical free VRAM: 673 MiB/rank
- average / peak aggregate GPU power: 1,263.5 / 1,451.2 W
- peak sampled card power: 379.1 W at a 375 W configured cap
- peak GPU temperature: 86 C; no NVIDIA thermal-slowdown reason
- post-run `/health`: HTTP 200
- container: running, restart count zero, `OOMKilled=false`
- CUDA OOM, worker/EngineCore death, traceback, connector/cache failure: zero

The restricted raw evidence remains on AIBeast at
`/mnt/fast/build/r31-memory-stack-20260808/evidence/gpqa-diamond-live/20260810T084850Z/`.
It includes protected dataset-derived response text and must not be committed
or redistributed with the public appliance. `aggregate-summary.json`,
`contamination-summary.json`, the exact command/protocol, metrics, GPU sampler
and time-bounded server alerts provide the privacy-safe audit surface.
