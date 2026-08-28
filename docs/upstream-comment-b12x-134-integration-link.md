Integration cross-reference: the vLLM-side repeated-index/metadata audit is
tracked in https://github.com/local-inference-lab/vllm/issues/207 and the
turnkey next-window plan retains this issue as the memory-accounting contract.

One configuration landmine is now explicit in that plan: GG r31 logs
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB=64`, while the installed B12X source
consumes `B12X_INDEXER_TWO_LEVEL_FOLD*`. The next baseline must verify the
effective worker environment, selected fold route, exact candidate bytes and
profiled scratch before trusting the nominal cap.

No current production request is claimed to have traversed the two-level
route; source/configuration make it eligible at the documented shapes, and the
proposed route telemetry is what will turn that into request-level evidence.
