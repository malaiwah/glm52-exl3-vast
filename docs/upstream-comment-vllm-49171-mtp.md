I audited this optimization against an MRV2 native-MTP path
(`local-inference-lab/vllm@e2666d9a65`, GLM-5.2, MTP3).

One useful clarification: this PR's no-scheduled-drafts branch should already
skip target LM-head/sampling work for a pure unfinished-prefill batch even when
an MTP speculator is configured. The remaining speculative gaps are narrower:

1. A mixed batch with decode rows and unfinished-prefill rows enters the global
   draft-token branch, which still gives every request a target bonus-logit row.
2. The MTP drafter must run its prefill forward to hydrate draft KV, but it then
   computes/samples the first proposal and runs the remaining autoregressive
   MTP steps for unfinished rows. The scheduler explicitly discards those
   proposals.

I opened a source-backed follow-up with the exact call sites, a hydrate-only
design, risks and GPU A/B contract:
https://github.com/local-inference-lab/vllm/issues/272

For the 753B GLM shape I would expect low-single-digit benefit under concurrent
long prefill, not assume this PR's Qwen3.5-4B headline. No GLM performance claim
is being made before the matched GPU test.
