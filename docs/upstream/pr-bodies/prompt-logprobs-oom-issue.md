## Summary

Model Runner V2, which still lives under the V1 engine namespace, can kill the
entire EngineCore when `prompt_logprobs` is requested on a memory-dense serving
profile. Its prompt-logprobs worker does chunk logits processing, but the chunk
is hard-coded to 1,024 tokens. For GLM-5.2's 154,880-token vocabulary, the
tensor-parallel full-vocabulary BF16 logits all-gather for one such chunk is
approximately 303 MiB per rank before top-k selection.

On a production TP4/DCP4 GLM-5.2 service, fixed startup allocation left each
rank with 218.81 MiB physically free. Four concurrent requests carrying
`logprobs=20, prompt_logprobs=20` caused every rank to attempt a 304 MiB
allocation, OOM, and terminate EngineCore. The appliance supervisor recovered
the service, but reloading this checkpoint took about 13 minutes.

This is a special workload, but it is a valid OpenAI-compatible request and a
sharp reliability edge for tightly sized production profiles. It is especially
relevant to KLD/evaluation clients and other likelihood-based workloads.

## Exact environment

- Image: `voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28`
- vLLM: `0.11.2.dev280+gilded.gnosis.v20.vllme1e9426.si200c1db.fi801d57a.cu132.20260804.r28`
- PyTorch: `2.12.0+cu132`
- GPUs: 4 x RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB each
- Driver: `595.71.05`; CUDA 13.2 runtime; 280 W/card power limit
- Model: GLM-5.2 EXL3-TR3 3.42 bpw, online K6
- Parallelism: TP4 / DCP4 / MTP3
- KV: dynamic NVFP4 MLA, 1,954 blocks, 500,000 maximum model length
- Scheduler: `max_num_seqs=8`, `max_num_batched_tokens=3072`
- LMCache MP connector enabled (125 GB DRAM + bounded 512 GB NVMe)

## Trigger

At 2026-08-07 23:49:42 UTC the scheduler admitted four new requests in one
3,072-token step. Prompt lengths were 1,110, 525, 1,112, and 868 tokens. Each
request used:

```text
logprobs=20
prompt_logprobs=20
max_tokens=384
```

The scheduler dump reported 92.2% logical KV occupancy as workload context:

```text
num_running_reqs=4
num_waiting_reqs=0
num_skipped_waiting_reqs=3
kv_cache_usage=0.9221710189452125
total_num_scheduled_tokens=3072
```

## Failure

All four ranks failed in the same path:

```text
PromptLogprobsWorker.compute_prompt_logprobs
  -> compute_prompt_logprobs_with_chunking
  -> logits_fn(prompt_hidden_states[start_idx:end_idx])
  -> DeepseekV2ForCausalLM.compute_logits
  -> LogitsProcessor._gather_logits
  -> tensor_model_parallel_all_gather
  -> CudaCommunicator.all_gather
```

Representative error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 304.00 MiB.
GPU 0 has a total capacity of 94.97 GiB of which 218.81 MiB is free.
Including non-PyTorch memory, this process has 93.84 GiB memory in use.
Of the allocated memory 91.19 GiB is allocated by PyTorch, with 118.00 MiB
allocated in private pools (e.g., CUDA Graphs), and 744.36 MiB is reserved by
PyTorch but unallocated.
```

EngineCore then died with `EngineDeadError`. The wrapper logged:

```text
!!! vLLM exited (attempt 0)
!!! restarting vLLM — attempt 1/5 in 15s
```

## Root cause

`vllm/v1/worker/gpu/sample/prompt_logprob.py` currently uses:

```python
# Since materializing the full prompt logits can take too much memory,
# we compute it in chunks.
CHUNK_SIZE = 1024
```

The requested top-20 does not bound the transient allocation: vLLM first
materializes and TP-all-gathers full-vocabulary logits for up to 1,024 prompt
positions, and only then computes the requested top-k. For this vocabulary:

```text
1024 * 154880 * 2 bytes = 302.5 MiB
```

The observed aligned allocation was 304 MiB. This transient is not represented
in KV admission control or startup memory sizing. The request therefore remains
admissible even when the allocation cannot fit.

There is a second device-memory gap for chunked prefills. MRv2 retains every
chunk's `LogprobsTensors` on GPU until the prompt finishes and then concatenates
the complete result on GPU. That retention grows with prompt length and
concurrency. The legacy GPU model runner already avoids this by preallocating a
CPU result and copying each completed slice into it.

The reported 92.2% logical KV occupancy did not itself consume the missing
304 MiB: the GPU KV tensor is normally allocated at startup. It describes the
live workload, while the proven cause is the unprofiled prompt-logprobs
allocation exceeding physical headroom.

This is related to the broad historical report vllm-project/vllm#5907, but
Model Runner V2 now has chunked logits processing; the remaining problem is
that its fixed 1,024-token chunk can still exceed runtime headroom and kill the
engine.

## Expected behavior

A valid prompt-logprobs request should not kill EngineCore on an otherwise
admissible production profile. At minimum, operators need a supported way to
bound this prompt-only transient. Startup memory sizing should account for the
same path used at runtime.

## Proposed repair

Exercise the internally chunked prompt-logprobs path over the full profile
batch during the existing startup memory profile, including `compute_logits`,
the TP full-vocabulary all-gather, top-k/log-softmax work, retained outputs, and
final concatenation. Its repeatable peak will then be deducted before KV cache
sizing, just like other supported serving-path transients.

Move cross-step prompt-logprobs accumulation to a preallocated CPU buffer, as
the legacy runner already does, so long and concurrent prompts cannot retain an
unbounded number of result chunks on GPU. Release each full-vocabulary logits
chunk before final concatenation.

Also make the Model Runner V2 prompt-logprobs logits chunk size an explicit,
validated runtime setting while preserving the current 1,024 default. This
bounds the profiled and live path with the same value. A tightly packed service
can select 256 (approximately 75.6 MiB of BF16 gathered logits per chunk for
this model), trading additional all-gather launches only for this special
workload.

`num_gpu_blocks_override` intentionally bypasses the capacity implied by the
available KV memory budget. The exact reproducer uses it, so that setting must
be reduced or removed after the fixed profiler reports the safe block count. A
warning when an override exceeds that implied capacity would make this sharp
edge clearer without changing override semantics. This wording also covers the
case where `kv_cache_memory_bytes` replaces profiling with a manual budget.

Catch-and-retry after a distributed CUDA OOM is not proposed as the primary
safety mechanism.
