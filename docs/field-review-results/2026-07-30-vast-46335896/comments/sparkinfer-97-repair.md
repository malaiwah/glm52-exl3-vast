Rental revalidation now **passes** on repaired PR #101 head
`0959fe807d366628380f41933244e3ccce0a8ae0`.

The prior failure did not refute the bounded two-slot selector: the 17-op graph
correctly overwrote both slabs, while the test launched replay on one stream
and sampled another. The corrected same-stream oracle verifies the final two
layer markers and alternating final-marker slot. Exact #101 warm passed; an
independent fresh-cache 1,025-replay run passed in 48.23 s with no Xid, hang,
or parity disagreement.

Full evidence and diagnosis:
https://github.com/local-inference-lab/sparkinfer/pull/101#issuecomment-5138805610
