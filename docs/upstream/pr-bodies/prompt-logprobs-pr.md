## Summary

- include the Model Runner V2 prompt-logprobs logits/all-gather/top-k path in startup memory profiling on the last pipeline stage
- make the existing 1,024-row logits chunk an explicit validated `VLLM_PROMPT_LOGPROBS_CHUNK_SIZE` setting
- accumulate chunked-prefill prompt-logprob results in a preallocated CPU buffer instead of retaining every chunk on GPU
- warn when `num_gpu_blocks_override` exceeds the capacity implied by available KV memory

Closes #257.

## Root cause and impact

A production TP4/DCP4 GLM-5.2 service had 218.81 MiB physically free per rank when four valid requests used `prompt_logprobs=20`. The V1 prompt-logprobs path attempted a 304 MiB allocation on every rank and killed EngineCore:

```text
1024 prompt rows * 154880 vocabulary * 2 BF16 bytes = 302.5 MiB
```

The allocation is the full-vocabulary tensor-parallel logits all-gather performed before top-k selection. Startup memory profiling did not exercise this path, so KV sizing could consume the required headroom. Model Runner V2 is implemented beneath the V1 engine namespace at `vllm/v1/worker/gpu/`; the similarly named legacy runner is `vllm/v1/worker/gpu_model_runner.py`.

MRv2 also retained each prompt-logprobs result chunk on GPU until prompt completion and concatenated the full result on GPU. For long or concurrent prompts that memory grows across scheduler steps. This change adopts the bounded CPU-accumulation design already used by the legacy runner.

The observed 92.2% logical KV occupancy was workload context, not the cause of the missing physical memory: the GPU KV tensor is normally allocated at startup.

## Duplicate-work check

- vllm-project/vllm#45327 changes the **legacy** runner from a full `log_softmax` tensor to `compute_topk_logprobs`; this fork's MRv2 path already uses top-k and the PR does not profile its TP logits workspace or fix cross-step GPU retention.
- vllm-project/vllm#37518 profiles sampler logits workspaces; its discussion explicitly notes that prompt logprobs bypass the sampler and remain a separate path.
- no open `local-inference-lab/vllm` PR references #257 or implements this repair.

## Validation

Passed locally on macOS/CPU:

```text
.venv/bin/python -m pytest tests/v1/worker/test_prompt_logprobs.py tests/v1/core/test_kv_cache_utils.py::test_warns_when_num_gpu_blocks_override_exceeds_profiled_capacity -q
8 passed
```

Focused coverage verifies:

- invalid chunk settings fail closed
- every logits call stays within the configured row bound
- startup profiles the full batch and forwards `max_logprobs=-1`
- two scheduler steps preserve CPU-accumulated values and order
- only the last PP rank profiles prompt logprobs
- unsafe block overrides emit a warning

`ruff check`, `ruff format`, `typos`, SPDX, configuration validation, and the other applicable pre-commit hooks passed. The full pre-commit invocation also surfaced one pre-existing custom-branch mypy error at `vllm/v1/worker/gpu/model_runner.py:1987` (`bool | None` passed to `defer_copy_event: bool`); this patch does not change that code.

GPU qualification is deliberately left pending in this draft. The planned gate is TP2/TP4 startup-capacity comparison plus concurrent `prompt_logprobs=20`, long chunked prefill, and MTP smoke tests on an authorized non-production host.

## AI assistance disclosure

AI assistance was used for source auditing, implementation, and test drafting. The human submitter reviewed the incident evidence and is expected to review every changed line and the GPU qualification results before this PR is marked ready.
