Immutable source links for the audited r31 tree
[`e2666d9a65f`](https://github.com/local-inference-lab/vllm/commit/e2666d9a65f41fc376607531453cbd57c4c71016):

- target bonus-logit row construction:
  [`model_runner.py#L1225-L1271`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/model_runner.py#L1225-L1271)
- target logits/sampling:
  [`model_runner.py#L1460-L1497`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/model_runner.py#L1460-L1497)
- unfinished-row suppression after sampling:
  [`input_batch.py#L447-L493`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/input_batch.py#L447-L493)
- MTP proposal invocation:
  [`model_runner.py#L2007-L2041`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/model_runner.py#L2007-L2041)
- required draft prefill plus first sampled proposal:
  [`speculator.py#L445-L483`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L445-L483)
- subsequent MTP draft steps:
  [`speculator.py#L485-L510`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L485-L510)
- scheduler discarding unfinished-prefill drafts:
  [`scheduler.py#L2111-L2131`](https://github.com/local-inference-lab/vllm/blob/e2666d9a65f41fc376607531453cbd57c4c71016/vllm/v1/core/sched/scheduler.py#L2111-L2131)
