Immutable source links for the audited r31 tree
[`e2666d9a65f`](https://github.com/local-inference-lab/vllm/commit/e2666d9a65f41fc376607531453cbd57c4c71016):

- decoded/prefill topology and counts:
  [`b12x_mla_sparse.py#L973-L994`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/mla/b12x_mla_sparse.py#L973-L994)
- batch-wide decode-route selection:
  [`b12x_mla_sparse.py#L2766-L2775`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/mla/b12x_mla_sparse.py#L2766-L2775)
- unified mixed-batch extend call:
  [`b12x_mla_sparse.py#L2834-L2886`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/mla/b12x_mla_sparse.py#L2834-L2886)
- pure-prefill CKV-gather eligibility:
  [`b12x_mla_sparse.py#L1026-L1028`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/mla/b12x_mla_sparse.py#L1026-L1028)
  and
  [`#L2141-L2153`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/mla/b12x_mla_sparse.py#L2141-L2153)
- generic contiguous batch reorder/split contract:
  [`utils.py#L566-L720`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/attention/backends/utils.py#L566-L720)
