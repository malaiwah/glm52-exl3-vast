## r31 revalidation: original large-M defect is fixed

I revalidated this issue against the exact GG r31 vLLM/B12X overlays and the
live AIBeast r31 production stack.

The native mixed large-M implementation requested here shipped in B12X #112
(merge `77154c105f441777355df1817ab660a8151fb294`). It preserves one-grid
mixed K3/K4 execution, uses exact paired-M8 FC2 subtiles, and qualifies route
block 32 for the GLM-5.2 large-M geometry.

Measured #112 qualification:

- mixed-kernel time: -6.6%
- persistent scratch: -47.1 MiB/GPU
- full-server prefill: +1.9% to +3.5%
- decode: parity

B12X #117 (`cc8c27054e13d6f8c407225482d1043fc0a1150b`) subsequently made
the tier counts and shared-H layout runtime-dynamic.

The exact r31 vLLM dispatcher now uses one-grid block-8 for decode and
one-grid block-32 for qualified GLM prefill. Live AIBeast r31 startup confirms:

```text
tiers=((3, 206), (4, 50))
one-grid decode=48 one-grid prefill=3072/3072 block_m=32
buffers=34.8+767.1 MiB prefill_reused=False

tiers=((3, 148), (4, 108))
one-grid decode=48 one-grid prefill=3072/3072 block_m=32
buffers=34.8+767.1 MiB prefill_reused=True
```

The original r14 block-8 large-M regression therefore no longer exists, and
the serial homogeneous block-64 workaround is not the current r31 execution
path.

I recommend closing #107 as fixed. Further cooperative-grid sizing or FC2
reuse experiments would be incremental optimization work and should get a
separate issue only after a repeatable kernel and full-server gain is measured.
