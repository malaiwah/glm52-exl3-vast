Rental revalidation now **passes** on repaired PR #103 head
`8fd90a4f687895e705bdf52fcff84ef653a5abf0`.

The previous one-shot FAIL was a test false negative: a 17-op graph overwrites
both slabs and replay was launched on the default stream while the explicit
capture stream was synchronized/probed. The corrected same-stream oracle
passed cold and repeatedly warm; an independent fresh-cache run completed
1,025 graph replays. Existing CPU/nested-capture, DCP2, DCP4, fused RMSNorm,
and two-shot gates remain green.

Full evidence and diagnosis:
https://github.com/local-inference-lab/sparkinfer/pull/103#issuecomment-5138805824
