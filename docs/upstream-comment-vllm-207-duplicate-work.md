## r31 follow-up: two repeated-work targets are larger than the original host-cleanup estimate

I re-audited the exact GG r31 vLLM/B12X path specifically for work repeated
across layers and steps.

### 1. Reuse shared-topK physical mappings across the 57 shared-index layers

GLM creates a new sparse index on only 21 of 78 target layers. The other 57
reuse the previous logical top-K, but every layer still repeats the
global-to-gathered/local-physical mapping, minima, causal-length construction,
and page-table masking in `b12x_mla_sparse.py:2564-2639`.

A safe target-only optimization can preserve graph-stable mapped buffers until
the next layer with `layer.indexer is not None`. MTP layers must be excluded:
they retain an indexer object and compact live rows between draft steps.

- frequency: 57 redundant mapping sequences per target PP/TG pass;
- source-derived estimate: 1-5% PP, 0.5-3% TG;
- required gate: exact index/count equality, mixed request compaction, C1/C8/C12,
  MTP3, and long-context retrieval.

### 2. Precompute gathered-slot geometry once per batch

`_append_current_chunk_to_gathered` (`b12x_mla_sparse.py:2281-2341`) rebuilds
request IDs, query starts, global positions, DCP owners, local positions, and
destination slots on every one of 78 layers. The KV values differ by layer;
the destination geometry does not.

Precompute one graph-stable `int64[max_num_batched_tokens]` slot buffer in the
metadata builder and reuse it. At 3,072 tokens the persistent cost is about
24 KiB.

- frequency: approximately 8-12 small tensor operations times 78 per eligible
  prefill chunk;
- source-derived estimate: 1-4% PP;
- required gate: exact slot identity for 1/2/8/12 requests, uneven chunks,
  DCP interleave 64, graph replay, then normal PP/needle qualification.

The estimates overlap and must not be added. Nsight launch/timeline evidence
should decide the implementation order. These findings fit this issue's
existing batch-constant CKV/index metadata scope, so I did not open duplicates.

Two separate findings were filed independently:

- B12X #134: caller-owned/profiled two-level paged-fold scratch;
- local vLLM #275: compact no-indexer fused norm/RoPE grid and conditional
  B12X top-K preclear.

AI assistance disclosure: the source audit and benchmark contract were
prepared with Codex assistance; the submitter remains responsible for the
claims and resulting implementation.
