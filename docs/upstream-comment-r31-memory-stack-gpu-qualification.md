## AIBeast composed-stack GPU qualification

The previously GPU-pending exact heads have now run in the production-shape
turnkey stack on 4x RTX PRO 6000 Blackwell:

- GG v20 r31 base: vLLM `fa13d334a296`, B12X `acee6e504209`;
- vLLM #258 `63b77c803149`, #270 `244d85a6fe99`, #271
  `310c3ac9718f`;
- B12X #130 `9ead9eaa188c` on the #126/#228 route-prewarm foundation;
- checkpoint `willfalco/GLM-5.2-EXL3-TR3-3.42bpw`
  at `a350292cb2038f2c31732569a711a89e5d72fd46`;
- TP4/DCP4, native MTP3 probabilistic, online K6, dynamic NVFP4 MLA + FP8
  RoPE, C12, 3,072 batched rows, graph/Trellis 48;
- exact byte-pinned KV: `4,518,907,904` bytes/rank = 520,192 logical tokens;
  no `num_gpu_blocks_override`.

### Memory/reliability evidence

- CUDA graph capture: 0.74 GiB/rank; about 3.24 GiB/rank free immediately
  after startup.
- After first-use/JIT paths: about 1.00-1.10 GiB/rank free.
- Minimum after cold 64K/128K prefill: about 0.84 GiB/rank free.
- Four concurrent 16K `prompt_logprobs=20` requests returned exact row counts.
- No worker restart, preemption, OOM, Xid, or post-profile arena growth in
  this arm.

### Performance/concurrency evidence

- uncached PP: 2,323.9 tok/s at 65,528 tokens; 2,199.5 tok/s at 131,073;
- C1/C4/C12 aggregate decode: 108.7 / 226.5 / 266.8 tok/s;
- MTP MAL: 3.609 / 3.575 / 3.408;
- zero request failures/preemptions in the retained decode matrix.

The composed stack also served `GLM-5.2` and `local-primary`, passed structured
output/tool-call and long-context retrieval gates, and remained healthy under
agent traffic.

This closes the stated production-shape GPU-pending integration gate for these
exact heads. It is **not** isolated attribution of the memory/performance delta
to either PR individually, and it does not prove arbitrary manual KV byte/block
overrides safe. The immutable patch ledger records the qualified heads and
file hashes; raw evidence remains under
`/mnt/fast/build/r31-memory-stack-20260808/evidence/` on AIBeast.

AI assistance disclosure: evidence reconciliation and this summary were
prepared with Codex assistance; the submitter remains responsible for the
claims.
