# GG v20-r17 AIBeast maintenance qualification

Qualified 2026-08-01 on AIBeast during an explicit maintenance window. Raw
credential-free evidence is retained on the host under
`/mnt/fast/build/r17-maintenance-20260801T163000Z/`.

## Immutable boundary

| item | qualified value |
|---|---|
| hardware | 4x RTX PRO 6000 Blackwell 96 GB, all `NODE`, 280 W/card |
| driver / CUDA | NVIDIA driver 595.71.05 / CUDA 13.2 |
| base image | `voipmonitor/vllm:gilded-gnosis-v20-vllmdb29328-sib2bff71-fi801d57a-cu132-20260801-r17@sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727` |
| turnkey image | `localhost/glm52-turnkey:r17-native1`, image ID `1cb477cdc3bfda99b51dd87f6b1a6b6c5bf8ef06d2bd98f1c40af539a7d75aed` |
| target | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw`, appliance revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`, local immutable snapshot `61d2b6b757f6a4ac7098a78d861f2033497532dc` |
| topology | TP4 / DCP4, native mixed-K MTP5 probabilistic draft |
| serving geometry | max length 524,288; sequences 8; scheduler 2,048; EXL3 prefill capacity 1,024; graph/Trellis 48 |
| KV / cache | 2,048 blocks = 524,288 logical tokens; dynamic-token NVFP4 MLA; LMCache 125 GiB DRAM + bounded 512 GiB NVMe |

The turnkey rebase uses r17's native vLLM #222 mixed-K implementation and the
SparkInfer #105 complete PCIe wheel. The old #210/#219 overlays are not stacked
on r17. Exact base file hashes are build-gated, and runtime selected
`B12X_PCIE_ONESHOT_DMA` rather than silently falling back to PyNCCL.

## Release and matched-control gates

The upstream MTP0/262K release shape passed on the 280 W host at 2,443 / 2,314 /
2,175 tok/s for 3K/32K/128K cold PP. The upstream full-power 3,761 / 3,670 /
3,306 tok/s reference is contextual, not a matched regression claim.

The exact production-shape r14 and r17 A/B retained byte-for-byte memory
accounting: 82.81 GiB weights, 1.49 GiB activation peak, 0.54 GiB non-Torch,
0.20 GiB CUDA graphs, and 5.4 GiB fixed active KV per rank.

| stack | PP 3K | PP 32K | PP 128K | PP geometric mean |
|---|---:|---:|---:|---:|
| r14 control | 2,209.8 | 1,989.9 | 1,875.5 | 2,020.3 |
| r17 native #222 | 2,174.0 | 1,958.1 | 1,854.2 | 1,991.1 |
| r17 delta | -1.6% | -1.6% | -1.1% | -1.4% |

MTP5 decode varied with acceptance. Raw TG is retained in the JSON evidence;
route decisions use matched longer samples, MAL, TPOT, and TG/MAL together.

## Candidate decisions

| candidate | measured outcome | decision |
|---|---|---|
| NCCL automatic channel count vs cap 4 | NCCL already selected four collective channels | skip no-op; leave CTA/channel cap unset |
| `NCCL_BUFFSIZE=1048576` | approximately 208 MiB/GPU idle headroom; repeated PP 2,189 / 1,977 / 1,861, parity to +1% versus r17 control; no normalized decode loss | retain on the AIBeast production overlay |
| speculative verifier `auto` | safe; same KV; no repeatable PP/TG win after acceptance normalization | retain profile value `0` |
| calibrated DMA crossover 24 MiB | microbenchmark first favored DMA at 24 MiB, but sustained GLM PP lost 3-6% with no memory gain | reject; retain lossless DMA minimum 6 MiB |
| DCP A2A cap 32 | C4 normalized work rate improved materially with negligible memory cost | advance |
| DCP A2A cap 48 | versus cap 32, C1 was flat, C8 TG/MAL improved 3.3%, C8 TPOT fell 19.57 to 17.67 ms, raw C8 rose 319 to 376 tok/s; about 2 MiB/GPU cost | retain |
| owner merge 0 | PP 2,230 / 2,052 / 1,950 (+1.9/+3.8/+4.7% versus owner1), C1 slightly better, about 96 MiB/GPU recovered; normalized C4/C8 decode about 12% lower but C8 still 292 tok/s | retain for the explicit PP-first long-context priority |
| shared-expert stream | upstream source still labels broad TP/DP correctness coverage incomplete | do not test or promote |

The selected AIBeast overlay is:

```text
NCCL_BUFFSIZE=1048576
VLLM_PCIE_DMA_MIN_BYTES=6291456
VLLM_DCP_A2A_MAX_TOKENS=48
VLLM_DCP_TOPK_OWNER_MERGE=0
```

These are topology/workload-qualified production overlays, not yet portable
provider defaults. The appliance exposes them with `TUNE_` variables while the
cross-provider model profile retains conservative settings until rentals
counter-validate them.

## Correctness and capacity

The winner passed ordinary and thinking chat, streamed usage, preserved-
thinking multi-turn, structured JSON with and without thinking, automatic and
required tool calls, duplicate suppression, tool-result continuation, and both
`GLM-5.2` and `local-primary` aliases.

| prompt | actual tokens | depths | result | duration |
|---|---:|---|---:|---:|
| 256K | 261,195 | 1%, 25%, 50%, 75%, 99% | 5/5, no degeneration | 146.0 s |
| near maximum | 521,276 | 1%, 25%, 50%, 75%, 99% | 5/5, no degeneration | 333.4 s |

The near-maximum request reached 99.3% GPU-KV usage without preemption or OOM.
The first 521K request legitimately compiled twelve new W4A16 `m=120` K3/K4
artifacts after readiness; it still completed correctly. A clean production
restart then reused the r17-only compile namespace and repeated the exact
521,276-token five-depth matrix with seed `20260802`: 5/5, no degeneration,
312.9 seconds. All 27 post-ready kernel lookups were disk-cache hits, including
all twelve `m=120` artifacts; there were no post-ready compiles. The final log
audit found zero structured-FSM, CUDA, distributed-runtime, process-failure,
or engine-error findings.

## Selected production service

AIBeast now serves `GLM-5.2` and the fungible `local-primary` alias on port
8000 from `localhost/glm52-turnkey:r17-native1`, with `unless-stopped` restart
policy. It retains TP4/DCP4, native MTP5, 524,288 active tokens, the selected
four-variable overlay above, and 125 GiB DRAM plus bounded 512 GiB NVMe
LMCache. The clean production start reached `/health` in 13 minutes, passed
the complete short API suite through `local-primary`, and remained healthy
after the second-seed maximum-context gate.

## Deferred work

- Re-test the four selected overlay values on at least one rental topology
  before changing provider-wide model-profile defaults.
- Leave CKV prefetch at 0 and query split in the qualified state. The standalone
  calibrator suggested prefetch 1/query split off, but previous real GLM tests
  refuted those microbench choices; neither was promoted without a new matched
  workload A/B.
- Shared-expert overlap remains disabled until upstream's documented
  correctness uncertainty is resolved and a separate quality gate is run.
