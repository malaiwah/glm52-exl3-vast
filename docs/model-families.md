# Model families

The serve line used to answer every architectural question with "GLM-5.2".
Some of those answers were defaults; others were literals. `--tensor-parallel-size 4`
was a literal, which meant the image aborted at the GPU-count gate on any host
with fewer than four cards — before vLLM was ever invoked, for a model that
might have fitted on one.

A **family** owns everything true about an architecture rather than about this
deployment: which checkpoints exist, which engine flags are needed, which knobs
mean anything, and which of the measured failure rules apply.

| | |
|---|---|
| `scripts/glm_config.py` | `FAMILIES` registry, family-aware resolver, family-scoped validation |
| `entrypoint.sh` | family-guarded env block, generic serve line + `FAMILY_SERVE_ARGS` |
| `scripts/gpu_detect.py` | what the container can actually use, from nvidia-smi ∩ the visibility variables |
| `tests/test_families.py` | release defaults, Qwen/custom coherence, and rule scoping |
| `tests/test_gpu_detect.py` | injected visibility/device-list behavior |
| `tests/test_knob_wiring.py` | every knob has a consumer; the UI hardcodes nothing |

---

## 1. What is generic and what is not

Audited against the serve invocation as it stood. **GLM-5.2/MLA/EXL3-specific**,
now supplied by the family and absent for anything else:

| flag | why it is family-specific |
|---|---|
| `--decode-context-parallel-size`, `--dcp-comm-backend a2a`, `--dcp-kv-cache-interleave-size` | DCP shards the **MLA** KV cache. There is no MLA and no DCP in a dense model. |
| `--attention-backend B12X_MLA_SPARSE` | the MLA sparse-indexer kernel |
| `--moe-backend b12x` | GLM-5.2 is MoE; Qwen3.6-27B is dense |
| `--quantization exl3` | checkpoint-specific — and naming a method the checkpoint does not carry is a boot failure |
| `--tool-call-parser glm47`, `--reasoning-parser glm45` | GLM chat-template grammars |
| `--hf-overrides {"use_index_cache":true,"index_topk_pattern":"FFFSSS…"}` | the sparse indexer's per-layer pattern; meaningless elsewhere |
| `--default-chat-template-kwargs {"reasoning_effort":"high"}` | a GLM chat-template kwarg |
| `custom_ops:["all"]`, `pass_config.fuse_allreduce_rms` in the compilation config | b12x fusion passes |
| the whole MTP78 apparatus (graft, overlay download, draft dir, `DRAFT_*`) | surgery on GLM-5.2's layer 78 |
| the Glm5v vision wrapper | a GLM-5.2 config.json wrapper + plugin |
| `VLLM_EXL3_*`, `VLLM_USE_B12X_*`, `B12X_MLA_*`, `VLLM_DCP_*`, PCIe all-reduce | b12x/MLA/EXL3 kernels |

**Generic**, and still on the shared serve line: `--served-model-name`, host/port,
`--trust-remote-code`, `--tensor-parallel-size` (now a knob), `--kv-cache-dtype`,
`--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`,
`--max-num-batched-tokens`, `--max-cudagraph-capture-size`,
`--enable-chunked-prefill`, `--enable-prefix-caching`,
`--enable-prompt-tokens-details`, `--enable-force-include-usage`,
`--no-async-scheduling`, the KV-offload connector, TLS, auth, and the KV pool pin.

### One thing deliberately did NOT move

Speculative decoding. It is easy to assume MTP is a GLM feature, but Qwen3.6's
checkpoint includes MTP weights and the bundled vLLM exposes the generic
`mtp` method. So `MTP_TOKENS` is a **generic** knob and each family supplies
its default; what is GLM-only is the MTP78 draft machinery —
`MTP_DRAFT`, `DRAFT_MODEL`, `DRAFT_QUANTIZATION`.

### Where each half lives, and why the split looks untidy

* **CLI args are in `glm_config.py`**, because the landing page must be able to
  show what a family implies and to refuse knobs that cannot apply to it.
* **The env block stays in `entrypoint.sh`**, guarded by
  `if [ "$FAMILY_ENV_BLOCK" = "glm52" ]`. That block is the "each one is a
  measured decision" list; it must run *after* the image's own `ENV` in order to
  override it, and the whole `TUNE_` override mechanism is defined in terms of
  that ordering. Moving it into Python would have created a second source of
  truth for the settings with the most expensive history in this repo.

---

## 2. The inheritance model gains a layer

```
built-in defaults  <  FAMILY defaults  <  startup environment  <  JSON state file
```

The family layer is what "GLM-5.2 wants TP=4 and Qwen3.6 wants TP=1" means. It
sits below the environment so an operator can still override it from the
template, and below the state file so the landing page can. `/config` shows the
source of every value, and `family` is now one of them.

`minimize()` — the function that decides what actually gets written — compares
each knob against **the selected family's** baseline, so a value left at that
family's own default is not pinned into the file and the family stays free to
change it later. The family key itself is compared against the *no-file*
resolution instead; comparing it against a baseline built from itself is
circular, and an earlier revision did exactly that: the chosen family always
equalled its own baseline, was never written, and every apply silently reverted
to the previous family, taking its validation rules with it. There is a
regression test.

---

## 3. Inapplicable knobs are refused, not ignored

A knob scoped to a family it is not in resolves with source `n/a`:

* `/config` renders it **disabled**, with "not applicable to this model family"
  in place of the input, and an `n/a` badge next to the key.
* `minimize()` **drops** it instead of writing it, and the apply banner says
  which keys were dropped and why — switching family therefore cannot leave a
  stale `DCP` or `MTP_DRAFT` behind.
* A hand-edited state file that carries one is an **error**, naming the key —
  the same treatment the termination switches get.

GLM-only: `MTP_DRAFT`, `MTP_DRAFT_SAMPLE_METHOD`, `DRAFT_MODEL`,
`DRAFT_QUANTIZATION`, `DCP`, `VLLM_EXL3_TRELLIS_MAX_M`, `VISION`, and
`VISION_CHUNKS`.

This is the point of the exercise: nobody should be able to assemble a
configuration that hits the `m=3` capture class of bug on a model where the
trellis does not exist in the first place.

### The measured rules are scoped too

`concurrency-window`, `tr3-draft-needs-exl3`, `draft-quant-inherit`,
`kv-nvfp4-uncalibrated`, `vision-long-context`, `vision-kv-pressure`,
`dcp-divides-tp` and `dcp-reduces-pool` now fire **only** for `glm52`. They
describe EXL3 kernels, an MLA KV layout and a GLM vision wrapper; letting them
apply everywhere would reject configurations that are fine and, worse, teach
users to distrust the validator.

New, family-generic rules: `tp-exceeds-gpus` (error — the engine cannot start),
`variant-family-mismatch` (error), `kv-dtype-family` (error — `nvfp4_ds_mla` is
an MLA KV layout), `knob-inapplicable` (error), `family-untested` (warn),
`tp-off-measured` (warn).

### The long-context gate stays family-independent

`verify_serving.py` has no family branch and does not import the config
registry. The needle probe is the gate for **any** model: short prompts remain
insufficient regardless of architecture, and the probe simply sizes itself
against whatever `MAX_MODEL_LEN` the family resolved to (512K for GLM, 256K for
the upstream Qwen model, with a conservative 32K appliance default for the
NVFP4 profile). A family cannot opt out of being verified.

---

## 4. The Qwen3.6-27B NVFP4 preset

Facts below are from the [model card](https://huggingface.co/Qwen/Qwen3.6-27B);
quantization details come from the
[NVIDIA NVFP4 checkpoint card](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4).
The full 27B profile has completed cold download and checkpoint inspection on a
live Blackwell host. The same image/profile path was boot-qualified with the
small Qwen3.5 representative model, including reasoning, tools, vision and MTP.
The remaining qualification item is a full 27B serving boot and benchmark.

| | from the model card |
|---|---|
| architecture | **dense** 27B, hidden 5120, 64 layers, hybrid: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| attention | Gated DeltaNet (48 V heads / 16 QK heads, dim 128) + Gated Attention (24 Q / 4 KV heads, dim 256, RoPE). **Not MLA, not MoE.** |
| context | 262,144 native, ~1,010,000 with YaRN rope scaling |
| size | ~55.6 GB at BF16; the selected NVFP4 checkpoint downloads about 21 GiB |
| reasoning | thinking on by default; `chat_template_kwargs: {"enable_thinking": false}`, `{"preserve_thinking": true}` |
| card's vLLM line | `vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3` |
| appliance support | bundled vLLM recognizes `Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen parsers, and the `mtp` speculative method |

Preset choices and why:

| knob | value | reasoning |
|---|---|---|
| repo | `nvidia/Qwen3.6-27B-NVFP4` | the vendor-published low-weight development checkpoint |
| `--quantization` | `modelopt` | matches the checkpoint metadata and the quantizer bundled in the pinned image |
| `TENSOR_PARALLEL_SIZE` | **1** | the point of this profile is an inexpensive single-Blackwell development rental |
| `MAX_MODEL_LEN` | **32768** | conservative profile default; the upstream architecture supports 262,144 natively, but the full NVFP4 memory envelope still needs measurement |
| `MTP_TOKENS` | **0** | opt in only after a full-checkpoint boot; the method, when enabled, is `mtp` |
| `KV_CACHE_DTYPE` | `auto` | let vLLM choose; `fp8` is offered, `nvfp4_ds_mla` is refused as an MLA layout |
| `GPU_BLOCKS_OVERRIDE` | 0 | the 2048-block pin is a GLM-specific 512K trick |
| `GPU_MEMORY_UTILIZATION` | 0.90 | conservative; 0.93 was tuned for the GLM memory profile |
| parsers | `qwen3` reasoning, `qwen3_coder` tools | both are present in the pinned vLLM and were exercised with the representative Qwen model |
| multimodal | off by default | `--language-model-only`; opt in after budgeting the vision encoder's VRAM |

The conservative defaults intentionally separate feature-development testing
from final model qualification. See `TEST_RESULTS.md` for the live evidence and
the explicit remaining residual.

---

## 5. Tensor parallelism follows the hardware

`--tensor-parallel-size 4` was a literal; then it was a knob defaulting to 4.
Both are wrong for an image people rent on arbitrary hardware — a 1-, 2- or
8-GPU host all got 4 and failed opaquely. **It now defaults to the number of
GPUs the container can actually use**, and the resolution shows up as source
`detected` in the boot log and on `/config` beside `default`/`family`/`env`/
`file`.

Counting is `scripts/gpu_detect.py`, and it is not `nvidia-smi -L | wc -l`:

* `NVIDIA_VISIBLE_DEVICES` is what the container runtime was told to expose
  (`all`, `none`, or a list of indices/UUIDs).
* `CUDA_VISIBLE_DEVICES` narrows it further inside the container, and CUDA's
  enumeration rule is unusual: **the list is honoured up to the first invalid or
  repeated entry and everything after it is dropped**. `0,5,1` on a 4-GPU box
  yields one device, not two. Getting this wrong sets TP higher than the number
  of devices torch will enumerate, and the engine dies during init.
* The two compose; the intersection is what torch sees.
* A provider's advertised count (`RUNPOD_GPU_COUNT`, `GPU_COUNT`) is
  **corroboration, never the source**. When it disagrees with what is visible,
  the boot log says so — a mismatch means the pod is not what it claims.
* If detection yields zero, the entrypoint **refuses to start** and names the
  knob, rather than silently falling back to 4. A wrong TP is a boot failure
  either way; the only thing that changes is whether the user can tell why.

`DCP` follows the detected count too, for the GLM family: one KV shard per rank
is what the 4-GPU configuration always did, and DCP must divide TP.

### What moves with the rank count, and what does not

Worked through rather than assumed, because the measured GLM numbers all come
from one 4× RTX PRO 6000 box:

| setting | moves with TP? |
|---|---|
| `DCP` | **yes** — it is a KV shard count, one per rank; defaults to TP |
| `GPU_BLOCKS_OVERRIDE` | **yes** when explicitly pinned. The release default is 0, so vLLM profiles the pool for the actual per-rank memory envelope. |
| the memory profile / KV headroom | **yes** — per-rank weights change, so the pool that fits changes |
| `MAX_NUM_SEQS × (1 + MTP_TOKENS) ≤ 32` | **NO.** This is a per-kernel batch width — the CUDA-graph capture window and the EXL3 trellis window — and has nothing to do with how many ranks exist. The rule applies unchanged at any TP, and the warning says so explicitly so nobody "fixes" it by scaling with GPUs |
| `dcp-kv-cache-interleave-size 64` | independent of rank count (it is an interleave granularity); left alone |

GLM-5.2 below TP=4 is now a hard **error**, not a warning: 753B is ~308 GiB of
weights, which is 154 GiB per rank at TP=2 — it cannot fit on any card that
exists. The message says so and points at `MODEL_FAMILY=qwen36`. TP above 4 is
allowed with a warning that the published numbers (835,584-token pool,
121.3 tok/s, needle 5/5 at 490K) were all measured at TP=4 and should not be
assumed to transfer.

### Running on one GPU

```
-e MODEL_FAMILY=qwen36
```

is enough. TP comes from detection, so nothing else needs setting.

---

## 6. Checking a configuration without renting anything

```
docker run --rm -e CONFIG_SMOKE=1 -e MODEL_FAMILY=qwen36 <image>
```

`CONFIG_SMOKE=1` resolves the full configuration, prints every knob with its
source, prints the repo and directory that WOULD be downloaded, and prints the
**exact argv** the engine would be given — then exits. It downloads nothing,
touches no GPU, starts no landing page, mints no API key, and issues no ACME or
DNS calls.

This exists because three separate "the knob was ignored" reports were
investigated on rented hardware, at real cost, when a local run of this would
have answered all three in seconds. It is the first thing to run after changing
anything in the config layer, and the first thing to ask for when someone
reports that a setting had no effect.

## 7. Remaining qualification

The generic one-GPU path, profile switching, parsers, tools, multimodal input,
MTP, provider proxying, and GPU visibility handling were exercised live with a
small Qwen3.5 checkpoint. The exact Qwen3.6-27B NVFP4 checkpoint has completed
cold download and metadata inspection but still needs a full serving
performance run. GLM-5.2 is the flagship hardware-qualified profile; custom
checkpoints remain compatibility-by-vLLM rather than a blanket support claim.
