# AIBeast r34 maintenance candidate

This context composes the immutable GG v20 r34 image with five reviewed runtime
changes while preserving the exact turnkey control-plane assets deployed by the
r33 rollback container.

- vLLM PR277 compatibility on r34 (`9774994aac`)
- hybrid external-cache invalid-block recovery (`a25483af35`)
- bounded LMCache multiprocess retrieve completion (`0944cccb`)
- per-adapter filesystem L2 LRU and a 180-second retrieve deadline
- expired L1 read-lease recovery with configurable request-session retention
  (`67561538`, upstream draft PR LMCache/LMCache#4691)

The next maintenance configuration raises the L1 read lease from 300 to 900
seconds and retains MP request sessions for 5,400 seconds. These source and
configuration changes do not affect the running container; GPU/live validation
is deliberately deferred to the next authorized maintenance window.

The launch recipe uses a new v2 image/container name and an isolated
`/mnt/fast/build/r34-aibeast-read-lease-20260821` state root, so qualification
cannot accidentally share state, compilation artifacts, or L2 data with the
currently running r34 container.

The Dockerfile validates all r34 baseline and candidate file hashes. Production
uses isolated state, compilation cache, and L2 directories; the old r33 image,
container, 512 GiB L2, state, and JIT cache remain untouched for rollback.
