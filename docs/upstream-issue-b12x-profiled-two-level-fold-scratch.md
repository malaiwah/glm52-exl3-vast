## Summary

`index_topk_fp8()` selects the exact two-level paged-indexer fold at runtime,
then allocates `fold_values`, `fold_indices`, and `fold_lengths` with
`torch.empty` / `torch.full` inside every invocation. These buffers are not
part of `B12XIndexerPagedScratchPlan`, so startup activation/scratch profiling
does not reserve their peak. A tight-memory server can consequently pass
startup and later OOM only when an intermediate long-context shape selects the
two-level route.

Audited source: B12X
[`f6b5579febd356498393ba54f71d12b3dd8ce796`](https://github.com/local-inference-lab/b12x/commit/f6b5579febd356498393ba54f71d12b3dd8ce796).

- policy selection:
  [`paged.py`](https://github.com/local-inference-lab/b12x/blob/f6b5579febd356498393ba54f71d12b3dd8ce796/b12x/attention/nsa_indexer/paged.py#L96-L207)
- late allocations:
  [`paged.py`](https://github.com/local-inference-lab/b12x/blob/f6b5579febd356498393ba54f71d12b3dd8ce796/b12x/attention/nsa_indexer/paged.py#L942-L964)
- caller-owned plan that currently excludes these buffers:
  [`scratch.py`](https://github.com/local-inference-lab/b12x/blob/f6b5579febd356498393ba54f71d12b3dd8ce796/b12x/attention/nsa_indexer/scratch.py#L2094-L2168)

## Exact production relevance

For GLM-5.2 at `q_rows=3072`, `topk=2048`, each value+index candidate slice is
approximately 48 MiB:

| Slices | Late transient slab |
|---:|---:|
| 2 | 96.01 MiB |
| 3 | 144.01 MiB |
| 4 | 192.01 MiB |
| 5 | 240.01 MiB |
| 6 | 288.01 MiB |

The slabs are sequential across indexer layers, not multiplied by the number
of layers, but the peak is absent from startup accounting and allocations are
repeated.

The live GG r31 launcher logs
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB=64`, while the installed B12X source
consumes `B12X_INDEXER_TWO_LEVEL_FOLD*`. The worker has no effective
`B12X_INDEXER_TWO_LEVEL_FOLD*` override, so `auto` uses the 256 MiB default.
At a full 3,072-row DCP4 prefill chunk, the route is eligible at roughly
128K-320K global context depending on page rounding. Current logs do not expose
the selected route, so this is a deterministic source/configuration finding,
not a claim that a particular observed request used it.

## Proposed contract

1. Extend `B12XIndexerPagedScratchPlan` with bounded fold-value, fold-index and
   fold-length views.
2. Include those bytes in startup scratch/activation profiling.
3. Select two-level folding only when sufficient caller-owned fixed-address
   storage exists; otherwise use the existing exact streaming-carry route.
4. Emit one-time route telemetry containing rows, slices, candidate bytes,
   budget, and chosen route.
5. Keep the output exact and CUDA-graph-safe.

Primary value is deterministic memory accounting and removal of surprise OOM;
allocator/init removal may yield 0-2% PP but is not claimed without measurement.

## Qualification

- Model-free shapes: q rows `1,12,2048,3072`; context
  `64K,128K,192K,256K,320K,384K,500K`.
- Compare current 256 MiB auto, fixed 64 MiB/streaming carry, and patched
  caller-owned two-level routes.
- Record chosen route, slab bytes, allocation count, peak allocated/reserved
  memory, and CUDA-event time after warmup.
- Require bitwise-identical indices and scores.
- Full-server PP at 8K/64K/128K/192K/256K/320K; C1/C8/C12; concurrent long
  prefill; five-depth long-context retrieval; no new OOM, graph, or allocator
  instability.

## AI assistance disclosure

The source audit, size calculation, and test plan were prepared with Codex
assistance. The submitter remains responsible for the claims and resulting
implementation.
