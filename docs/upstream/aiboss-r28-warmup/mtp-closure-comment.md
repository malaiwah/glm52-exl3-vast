## Native MTP-draft allocation gap closed on AIBoss

The b12x head is now `6022e6e7c7ea1199a06a27cf5a777c2804b13cfb`. The rank-sliced full-rotation plans used by native MTP drafts prewarm every reachable route-capacity/scalar class, both `int32` and `int64` route-ID specializations, and mapped/unmapped routes. Route-ID dtype is part of the prewarm cache key.

Exact GGv20r28 A/B on one RTX 5090, native MTP1, FP8 MLA KV, TP1/DCP1, eager, max model length/batch 2,048, max sequences 4, GMU 0.75:

- post-start draft route-pack JIT: `small_prefix + sort` -> **none**
- logical KV: **1,865,920 -> 1,865,728** (-192 tokens, -0.010%)
- newly accounted persistent residency: exactly **2.0 MiB**
- accepted drafts: **451/571 (79.0%) -> identical**
- mean accepted length: **1.790 -> identical**
- generated output: **byte-identical**, `FRUIT-MTP-OK`
- cold initialization: 30.27 -> 43.04 s because an empty cache now compiles the bounded draft set before KV sizing
- exact final head, populated cache: **6.80 s** initialization, still no post-start route-pack JIT

This is a reliability fix, not a throughput claim. It converts a small, previously invisible MTP draft allocation into startup-accounted residency before vLLM commits the remaining memory to KV. TP4/DCP4 and CUDA-graph capture remain the outstanding qualification gates; AIBeast production was not interrupted.

A subsequent read-only AIBeast audit confirmed this is an active production risk: the MTP3 TP4/DCP4 r28 stack failed inside Triton's `load_binary()` for `_pack_topk_routes_post_prefix_kernel` with `CUDA: out of memory` while active KV usage was only 1.18%. The engine restarted, then lazily loaded `post_prefix` and `sort` again. That trace establishes the route-pack residency failure family, while the final patch closes both the target and distinct native-draft owners.

The full evidence comment and PR descriptions have been updated with provenance and raw-log SHA-256 values.
